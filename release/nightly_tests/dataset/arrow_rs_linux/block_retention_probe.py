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
        out = {
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
        # Finding the operator is not the same as finding its numbers. A summary
        # taken off an un-executed plan has the right operator names and no
        # samples at all, which renders as "0 MiB / 0 tasks" -- indistinguishable
        # from a real measurement of a read that used no memory. Fail loudly.
        if not out["uss_num_samples"] or not out["avg_max_uss_per_task"]:
            out["stats_error"] = (
                f"Read operator {names} carries no USS samples "
                f"(num_samples={out['uss_num_samples']}, "
                f"avg={out['avg_max_uss_per_task']}) -- the summary is almost "
                "certainly off an un-executed plan"
            )
        return out
    # An empty dict would print as 0 MiB / 0 tasks, which reads as a measurement
    # rather than a failure. Say which operators were actually there.
    return {"stats_error": f"no Read operator among {seen or '[]'}"}


def read_profile(profile_dir: str) -> Dict[str, Any]:
    """Fold the crate's JSONL records into the arm's result.

    Answers two questions a USS number alone cannot: **which decode branch ran**
    (``s3_plan.col_group_rgs`` -- every measurement to date has reported 0, so
    the column-group arm has to prove it fired before its numbers mean anything),
    and **how much decoded data was held at once** (``s3_rg.retained_bytes``
    against ``rg_uncompressed_bytes``). The Hstack arm accumulates every batch of
    every column group before emitting, so a ratio near 1.0 there is the
    mechanism itself, measured rather than argued.

    Records land one file per worker pid. A missing directory is reported, not
    raised: the local reader path emits nothing at all, and an arm that
    legitimately has no S3 records should not fail.
    """
    recs = []
    try:
        names = sorted(os.listdir(profile_dir))
    except OSError:
        return {"profile_error": f"no profile dir at {profile_dir}"}
    for name in names:
        if not name.endswith(".jsonl"):
            continue
        with open(os.path.join(profile_dir, name)) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    recs.append(json.loads(line))
                except ValueError:
                    continue  # a worker killed mid-write leaves a partial line
    if not recs:
        return {"profile_error": f"no records under {profile_dir}"}

    # Both halves of the reader key their records on "kind" and share the
    # directory (see _prof_record in the reader), so the Rust planner records and
    # the Python fragment records join on pid in one pass.
    plans = [r for r in recs if r.get("kind") == "s3_plan"]
    rgs = [r for r in recs if r.get("kind") == "s3_rg"]
    out: Dict[str, Any] = {
        "prof_files": len(names),
        "prof_plans": len(plans),
        "prof_row_groups": len(rgs),
        "prof_col_group_rgs": sum(p.get("col_group_rgs", 0) for p in plans),
        "prof_row_window_rgs": sum(p.get("row_window_rgs", 0) for p in plans),
        "prof_oversized_units": sum(p.get("oversized_units", 0) for p in plans),
    }
    if rgs:
        retained = [r.get("retained_bytes", 0) for r in rgs]
        out["prof_max_retained_mib"] = round(max(retained) / MiB, 1)
        out["prof_avg_retained_mib"] = round(sum(retained) / len(retained) / MiB, 1)
        # Only the column-group records carry the row group's uncompressed size,
        # which is the denominator that makes "retained the whole group" a claim
        # about a ratio rather than about an absolute number of MiB.
        sized = [r for r in rgs if r.get("rg_uncompressed_bytes")]
        if sized:
            out["prof_retained_over_rg"] = round(
                max(
                    r.get("retained_bytes", 0) / r["rg_uncompressed_bytes"]
                    for r in sized
                ),
                3,
            )
    modes = sorted({r.get("mode", "?") for r in rgs})
    if modes:
        out["prof_modes"] = ",".join(modes)
    # A fallback silently turns an arrow-rs arm into a PyArrow arm, which would
    # make an A/B read as "no difference" for the wrong reason. Surface both the
    # count and the distinct reasons.
    falls = [r for r in recs if r.get("kind") == "fallback"]
    if falls:
        out["prof_fallbacks"] = len(falls)
        out["prof_fallback_reasons"] = ",".join(
            sorted({str(r.get("reason", "?")) for r in falls})
        )[:200]
    return out


def _clamp_files(source: str, limit: int):
    """The first ``limit`` .parquet files under ``source``, as an explicit list.

    Sorted by name, so the local and S3 arms pick the SAME files rather than
    whatever order the listing happens to return -- lineitem files differ in row
    count, and "4 files" that are not the same 4 files is not a transport
    comparison.

    ``FileSystem.from_uri`` handles both ``s3://`` and bare local paths, and
    returns the path with the scheme stripped, which is what ``get_file_info``
    wants. The scheme is put back for S3 because ``read_parquet`` resolves a
    bare ``bucket/key`` as a local relative path.
    """
    from pyarrow.fs import FileSelector, FileSystem, FileType

    fs, root = FileSystem.from_uri(source)
    infos = fs.get_file_info(FileSelector(root, recursive=True))
    files = sorted(
        i.path for i in infos if i.type == FileType.File and i.path.endswith(".parquet")
    )
    if not files:
        raise SystemExit(f"no .parquet files under {source}")
    scheme = source.split("://")[0] + "://" if "://" in source else ""
    return [scheme + p for p in files[:limit]]


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
        "--fetch-window-mb",
        type=int,
        default=0,
        help="RAY_DATA_ARROW_RS_FETCH_WINDOW_MB -- how many compressed bytes of "
        "one row group the S3 path holds in flight before decoding them "
        "(default 16). S3-ONLY: the local path memory-maps and has no fetch "
        "window at all, which is why standalone arrow-rs costs 23 MiB on local "
        "disk and 174 MiB on S3 for the same data. 0 leaves the default.",
    )
    parser.add_argument(
        "--prefetch-budget-mb",
        type=int,
        default=0,
        help="RAY_DATA_ARROW_RS_PREFETCH_BUDGET_MB -- total compressed bytes "
        "admitted ahead of the decoder across ALL in-flight units. Defaults to "
        "-1, meaning 4 x max(fetch_window_mb, column_fetch_mb) = 64 MiB at the "
        "shipping defaults. This is the other half of the S3 working set: peak "
        "~= prefetch_budget + decode_budget. 0 leaves the default.",
    )
    parser.add_argument(
        "--column-fetch-mb",
        type=int,
        default=0,
        help="RAY_DATA_ARROW_RS_COLUMN_FETCH_MB (default 16) -- compressed MiB "
        "per COLUMN group. When one row group's projected compressed bytes "
        "exceed this, plan_s3_units sends it down RgDecode::Hstack instead of "
        "RgDecode::Windows. Pass -1 for the crate's `0`, which disables column "
        "grouping outright: that is the same-fixture A/B for the branch, since "
        "the Hstack arm accumulates every batch of every group before emitting "
        "and so cannot saturate. 0 leaves the default.",
    )
    parser.add_argument(
        "--profile-to",
        default=None,
        help="RAY_DATA_ARROW_RS_PROFILE_DIR -- have the crate and the Python "
        "reader emit JSONL per read task (s3_plan carries col_group_rgs; s3_rg "
        "carries retained_bytes and rg_uncompressed_bytes). Cheap, but it writes "
        "one file per worker pid, so keep it off the timed arms you intend to "
        "compare on wall time.",
    )
    parser.add_argument(
        "--write-to",
        default=None,
        help="fuse a write onto the read (the release write_parquet shape). "
        "Omitted = iterate bundles, which keeps the read unfused.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="read only the first N .parquet files under --source, sorted by "
        "name. Needed to compare transports: exp5 copies 4 files to local disk "
        "but stage_data.py stages the WHOLE sf10 prefix to S3, so an unclamped "
        "S3 arm reads ~11 files against local's 4 -- different data, and the "
        "MiB-decoded-per-task axis every fit is against would be wrong. "
        "0 reads everything.",
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
    if args.fetch_window_mb:
        os.environ["RAY_DATA_ARROW_RS_FETCH_WINDOW_MB"] = str(args.fetch_window_mb)
    if args.prefetch_budget_mb:
        os.environ["RAY_DATA_ARROW_RS_PREFETCH_BUDGET_MB"] = str(
            args.prefetch_budget_mb
        )
    if args.column_fetch_mb:
        # -1 on the command line means the crate's 0 ("disable column grouping").
        # 0 on the command line means "leave the default alone", so the two
        # cannot share an encoding.
        os.environ["RAY_DATA_ARROW_RS_COLUMN_FETCH_MB"] = str(
            0 if args.column_fetch_mb < 0 else args.column_fetch_mb
        )
    if args.profile_to:
        os.makedirs(args.profile_to, exist_ok=True)
        os.environ["RAY_DATA_ARROW_RS_PROFILE"] = "1"
        os.environ["RAY_DATA_ARROW_RS_PROFILE_DIR"] = args.profile_to
    # exp7 turns profiling on through the environment for every arm rather than
    # through this flag, so honour either. Without this the JSONL would be written
    # and then never read.
    profile_dir = args.profile_to or (
        os.environ.get("RAY_DATA_ARROW_RS_PROFILE_DIR")
        if os.environ.get("RAY_DATA_ARROW_RS_PROFILE", "").lower()
        not in ("", "0", "false")
        else None
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

    # A URI is not a local directory: rmtree/makedirs on "s3://..." would create
    # a literal ./s3:/ tree next to the script and leave the real prefix
    # untouched, so an S3 write arm would silently append to the previous arm's
    # output. Only clean paths we can actually clean.
    if args.write_to and "://" in args.write_to:
        return f"FATAL: --write-to must be a local path, got {args.write_to}"
    if args.write_to:
        shutil.rmtree(args.write_to, ignore_errors=True)
        os.makedirs(args.write_to, exist_ok=True)

    source = (
        _clamp_files(args.source, args.max_files) if args.max_files else args.source
    )

    started = time.perf_counter()
    ds = ray.data.read_parquet(source)
    if args.write_to:
        ds.write_parquet(args.write_to)
    else:
        # Consume without materializing: the retention we are hunting is inside
        # the read task, and .materialize() would add the object store on top.
        #
        # NOT ds.iter_internal_ref_bundles(): it calls
        # _execute_to_iterator(capture_executor=False) (dataset.py:7398) so the
        # executor is not retained -- which also leaves ds._current_executor as
        # None, so get_stats_summary() falls through to _raw_stats(), the
        # UN-EXECUTED plan, and reports zero tasks and zero memory. Calling
        # _execute_to_iterator directly runs the identical execution with the
        # executor captured, which is what the stats read needs.
        bundle_iter, _, _ = ds._execute_to_iterator()
        for _ in bundle_iter:
            pass
    wall = time.perf_counter() - started

    result: Dict[str, Any] = {
        "reader": args.reader,
        "block_mib": args.block_mib,
        "decode_budget_mib": args.decode_budget_mib or None,
        "chunk_mib": args.chunk_mib or None,
        "threads": args.threads or None,
        "fetch_window_mb": args.fetch_window_mb or None,
        "prefetch_budget_mb": args.prefetch_budget_mb or None,
        "column_fetch_mb": args.column_fetch_mb or None,
        "mode": "write" if args.write_to else "iter_bundles",
        "source": args.source,
        "num_files": len(source) if isinstance(source, list) else None,
        "wall_s": round(wall, 2),
    }
    result.update(read_stats(ds))
    if profile_dir:
        result.update(read_profile(profile_dir))
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
