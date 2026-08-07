#!/usr/bin/env python3
"""One read inside Ray at a chosen ``target_max_block_size``, reporting per-task USS.

Why this exists
---------------
The same read costs 23 MiB of private memory in a bare process and ~1 GiB inside
a Ray read task. The gap is transport-independent (it is the same on local disk
and on S3), it is not the decode budget (2 -> 128 MiB moves it 7%), and it is not
the crate's row-group retention (``retained_bytes`` says 20 MiB per row group).

What it *is* the size of is the input file: lineitem is ~6.0M rows x 172 B/row =
1.03 GiB decoded, and per-task USS measures 958-1035 MiB. So a read task appears
to hold its whole file's decoded output rather than streaming it out.

If that is block accumulation, per-task USS is a function of
``target_max_block_size`` and this sweep will show it rising with the knob. If
per-task USS is flat across a 32x sweep, blocks are not what is retained and the
next suspect is the fragment loop itself -- a reader that materializes a whole
file per yielded table would look exactly like this and would be immune to every
knob we have tried.

Each arm must be its own process: Ray reuses worker processes, and
``MemoryProfiler`` samples whole-process private memory, so a second arm in the
same cluster inherits the first arm's high-water mark.

Usage
-----
    python block_retention_probe.py --source ~/arrow_rs_local/lineitem \
        --reader arrow_rs --block-mib 128 --out out/exp6/arrow_rs_128.json
"""

import argparse
import json
import os
import shutil
import sys
import time
from typing import Any, Dict

MiB = 1024 * 1024


def read_stats(ds) -> Dict[str, Any]:
    """Per-task memory for the read operator, straight off Ray's own stats.

    Ray already accumulates this (``MemoryProfiler`` in ``map_operator``); the
    release harness surfaces it as ``read_avg_max_uss_per_task_bytes``. Pulled
    out here rather than importing ``benchmark.py`` so this probe stays runnable
    on its own.

    Use the PUBLIC ``get_stats_summary``: ``write_parquet`` executes an internal
    dataset stored at ``ds._write_ds`` and only the public accessor knows to
    look there (dataset.py:7361). Reading the private summary reports on the
    outer plan, which never executed, and yields zero tasks and zero memory --
    silently, because "no samples" and "no memory" render identically.
    """
    from ray.data._internal.stats import DatasetStatsSummary

    summary = ds.get_stats_summary()
    seen = []
    for node in DatasetStatsSummary._collect_dataset_stats_summaries(summary):
        extra = getattr(node, "extra_metrics", {}) or {}
        names = [op.operator_name for op in (node.operators_stats or [])]
        seen.extend(names)
        # The read is the only operator producing decoded bytes; a fused
        # Read->Write reports under the fused name, so match on substring.
        if not any("Read" in n for n in names):
            continue
        dist = extra.get("max_uss_bytes")
        dist = dist if isinstance(dist, dict) else {}
        # Take task_rows from the READ operator specifically. Picking the first
        # operator that has it lands on the Write in a fused Read->Write, which
        # emits one metadata row per task -- that is where "rows/task = 1" came
        # from, and it made the task count unverifiable.
        rows = next(
            (
                op.task_rows
                for op in (node.operators_stats or [])
                if "Read" in op.operator_name and getattr(op, "task_rows", None)
            ),
            None,
        )
        return {
            "operators": names,
            "avg_max_uss_per_task": extra.get("average_max_uss_per_task"),
            "max_uss_per_task": extra.get("max_uss_per_task"),
            # The independent task count: one USS sample per task that ran. If
            # this disagrees with task_count, trust this one.
            "uss_num_samples": dist.get("num_samples", 0),
            "uss_p90": dist.get("p90"),
            "task_count": getattr(rows, "count", None),
            "rows_per_task_mean": getattr(rows, "mean", None),
        }
    # An empty dict would print as 0 MiB / 0 tasks, which reads as a measurement
    # rather than a failure. Say which operators were actually there.
    return {"stats_error": f"no Read operator among {seen or '[]'}"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", required=True)
    parser.add_argument("--reader", choices=("pyarrow", "arrow_rs"), required=True)
    parser.add_argument(
        "--block-mib",
        type=int,
        required=True,
        help="DataContext.target_max_block_size, in MiB. Ray's default is 128.",
    )
    parser.add_argument(
        "--decode-budget-mib",
        type=int,
        default=0,
        help="crate decode budget, in MiB. 0 leaves the reader's default (2). "
        "Ray's streaming unit is target_max_block_size, so what likely matters "
        "is this as a FRACTION of --block-mib: batches much smaller than a "
        "block make the block builder accumulate and then concatenate, which "
        "needs inputs and output alive together. Ignored by the pyarrow arm.",
    )
    parser.add_argument(
        "--chunk-mib",
        type=int,
        default=0,
        help="DataContext.parquet_chunker_target_chunk_size, in MiB. This -- not "
        "override_num_blocks, which the V2 path ignores entirely -- is what sizes "
        "a read task: ParquetFileChunker splits a file only when its ON-DISK size "
        "exceeds this, and the built-in default is 1 GiB, so ordinary files are "
        "never split and each task decodes a whole file. 0 leaves the default.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="RAY_DATA_READ_FILES_NUM_THREADS -- how many fragments a read task "
        "decodes concurrently (file_reader.py:44, default 4). Shared by both "
        "readers. The crate retains ~20 MiB per row group, so 4 in flight is "
        "~82 MiB live no matter how small the task is; that is the leading "
        "candidate for the ~104 MiB fixed floor arrow-rs carries over PyArrow. "
        "0 leaves the default.",
    )
    parser.add_argument(
        "--write-to",
        default=None,
        help="fuse a write onto the read (the release write_parquet shape). "
        "Omitted = iterate bundles, which keeps the read unfused.",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    # Both flags must be set before ray.data builds the plan.
    os.environ["RAY_DATA_USE_DATASOURCE_V2"] = "1"
    os.environ["RAY_DATA_USE_ARROW_RS_PARQUET_READER"] = (
        "1" if args.reader == "arrow_rs" else "0"
    )
    if args.decode_budget_mib:
        os.environ["RAY_DATA_ARROW_RS_DECODE_BUDGET_BYTES"] = str(
            args.decode_budget_mib * MiB
        )
    # Read before ray.data imports: _DEFAULT_NUM_THREADS is a module-level
    # env_integer, so setting it afterwards has no effect at all.
    if args.threads:
        os.environ["RAY_DATA_READ_FILES_NUM_THREADS"] = str(args.threads)

    import ray
    from ray.data import DataContext

    ray.init(ignore_reinit_error=True)
    ctx = DataContext.get_current()
    ctx.target_max_block_size = args.block_mib * MiB
    if args.chunk_mib:
        ctx.parquet_chunker_target_chunk_size = args.chunk_mib * MiB

    if args.write_to:
        shutil.rmtree(args.write_to, ignore_errors=True)
        os.makedirs(args.write_to, exist_ok=True)

    started = time.perf_counter()
    ds = ray.data.read_parquet(args.source)
    if args.write_to:
        ds.write_parquet(args.write_to)
    else:
        # Consume without materializing: retention we are hunting is inside the
        # read task, and .materialize() would add the object store on top of it.
        for _ in ds.iter_internal_ref_bundles():
            pass
    wall = time.perf_counter() - started

    result: Dict[str, Any] = {
        "reader": args.reader,
        "block_mib": args.block_mib,
        "decode_budget_mib": args.decode_budget_mib or None,
        "chunk_mib": args.chunk_mib or None,
        "threads": args.threads or None,
        "mode": "write" if args.write_to else "iter_bundles",
        "source": args.source,
        "wall_s": round(wall, 2),
    }
    result.update(read_stats(ds))
    if result.get("stats_error"):
        print(f"WARNING: {result['stats_error']}", file=sys.stderr)
    print(json.dumps(result, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump(result, handle, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
