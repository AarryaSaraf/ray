#!/usr/bin/env python3
"""Read (and optionally write) Parquet with no Ray in the picture at all.

Why this exists
---------------
``write_parquet`` is the one release regression that survived checking against
the cluster's own per-node memory scrape: 3.36x more memory on every one of ten
machines. The operator is the fused ``ReadFilesParquetV2->Write``, and **both
arms write with PyArrow** -- there is no Rust writer -- so the extra memory has
to originate on the read side. That leaves exactly two candidates:

1. the crate hands back more memory than PyArrow's scanner does, or
2. the crate is fine and Ray's integration accumulates what it hands back
   (BlockOutputBuffer, the fused-op path, retained output blocks).

Nothing we have measured can tell those apart, because every measurement so far
ran inside Ray. This runs the same read in a bare process. If arrow-rs is much
heavier here, it is the crate. If the two are at parity here, no amount of crate
tuning will fix the regression and the bug is in the integration.

The transport is a second axis for the same reason: the crate's S3 planner
(``plan_s3_units``, ``partition_columns_by_budget``, the prefetch admission
loop) has no counterpart on the local path. Running both transports over the
same bytes says whether S3 is required to reproduce the problem.

Fairness notes
--------------
* The PyArrow arm uses ``pyarrow.dataset``'s **scanner**, not
  ``pq.ParquetFile.iter_batches``. That is what Ray's V2 reader uses, and the
  scanner holds the whole decoded row group where ``iter_batches`` streams.
  Benchmarking against ``iter_batches`` would manufacture an arrow-rs win that
  Ray would never see.
* Both arms get the same ``batch_size`` and write through the same
  ``pq.ParquetWriter``.
* The crate is called with the reader's own defaults (decode budget 2 MiB, k=1,
  128 MiB split threshold, 16 MiB fetch window / column fetch, prefetch budget
  4x the larger of those) so this measures the shipping configuration.
* Memory baseline is taken *after* imports and after the first footer read.
  An earlier harness measured worker import ramp and produced 133->211 MB swings
  between identical runs; see Agents.md on why RSS max-minus-min was abandoned.

Usage
-----
    python standalone_parquet_bench.py --source /data/lineitem --reader arrow_rs
    python standalone_parquet_bench.py --source s3://bucket/pfx --reader pyarrow \
        --write-to /data/out --out result.json
"""

import argparse
import json
import os
import resource
import shutil
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import pyarrow as pa
import pyarrow.dataset as pds
import pyarrow.parquet as pq
from pyarrow.fs import FileSelector, FileSystem, FileType

MiB = 1024 * 1024

# The shipping defaults from arrow_rs_parquet_file_reader.py. Kept as literals
# rather than imported so this script runs without Ray on the path -- but they
# must be updated together if the reader's defaults change.
DECODE_BUDGET_BYTES = 2 * MiB
K = 1
SPLIT_THRESHOLD_BYTES = 128 * MiB
FETCH_WINDOW_MB = 16
COLUMN_FETCH_MB = 16
PREFETCH_BUDGET_MB = 4 * max(FETCH_WINDOW_MB, COLUMN_FETCH_MB)


class MemorySampler:
    """Sample this process's own USS/RSS from a daemon thread.

    Self-sampling rather than external polling: an external poller cannot read
    another process's USS without privileges, and misses sub-100ms transients.
    """

    def __init__(self, interval_s: float = 0.01):
        import psutil

        self._proc = psutil.Process()
        self._interval = interval_s
        self._samples: List[Tuple[float, int, int]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t0 = 0.0
        # USS needs privileges on macOS and is absent on some platforms. Decide
        # once, up front, so every sample in a run measures the same thing --
        # silently mixing uss and rss samples would produce a meaningless peak,
        # and a failed read returning 0 would make "rise" negative.
        self.metric = "uss"
        try:
            info = self._proc.memory_full_info()
            if getattr(info, "uss", None) is None:
                self.metric = "rss"
        except Exception:  # noqa: BLE001 - permission denied on macOS
            self.metric = "rss"

    def read(self) -> Tuple[int, int]:
        """(private, rss). ``private`` is USS where available, else RSS."""
        try:
            if self.metric == "uss":
                info = self._proc.memory_full_info()
                return info.uss, info.rss
            info = self._proc.memory_info()
            return info.rss, info.rss
        except Exception:  # noqa: BLE001 - never let a sampling failure end the run
            return 0, 0

    def _sample(self):
        private, rss = self.read()
        if private:  # a failed read is dropped, not recorded as zero
            self._samples.append((time.perf_counter() - self._t0, private, rss))

    def _loop(self):
        while not self._stop.wait(self._interval):
            self._sample()

    def start(self):
        self._t0 = time.perf_counter()
        # Sample immediately: a run shorter than the poll interval would
        # otherwise finish with an empty series.
        self._sample()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, Any]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._sample()
        if not self._samples:
            return {"metric": self.metric, "samples": 0}
        private = [s[1] for s in self._samples]
        rss = [s[2] for s in self._samples]
        return {
            "metric": self.metric,
            "samples": len(self._samples),
            "uss_peak": max(private),
            "uss_final": private[-1],
            "rss_peak": max(rss),
        }


def list_parquet(source: str) -> Tuple[FileSystem, List[str]]:
    fs, path = FileSystem.from_uri(source)
    info = fs.get_file_info(path)
    if info.type == FileType.Directory:
        files = [
            f.path
            for f in fs.get_file_info(FileSelector(path, recursive=True))
            if f.path.endswith(".parquet")
        ]
    else:
        files = [path]
    return fs, sorted(files)


def read_pyarrow(fs: FileSystem, path: str, batch_size: int, readahead: int = -1):
    """Ray's V2 PyArrow path: a dataset scanner, one fragment at a time.

    ``use_threads=False`` disables only Arrow's CPU thread pool. The scanner
    still keeps ``fragment_readahead=4`` / ``batch_readahead=16`` and still runs
    I/O on the separate I/O pool, so it pipelines reads against decode. The
    crate's LOCAL path does not pipeline at all (the windowed-async path is
    S3-only), so the default configuration compares a pipelined reader with an
    unpipelined one. ``readahead=0`` serializes the scanner to make the decode
    cost itself comparable; ``-1`` keeps PyArrow's defaults, which is what Ray
    actually runs.
    """
    dataset = pds.dataset(path, filesystem=fs, format="parquet")
    kwargs = {"batch_size": batch_size, "use_threads": False}
    if readahead >= 0:
        kwargs["fragment_readahead"] = readahead
        kwargs["batch_readahead"] = readahead
    yield from dataset.scanner(**kwargs).to_batches()


def read_arrow_rs(fs: FileSystem, path: str, batch_size: int, readahead: int = -1):
    """The crate, called exactly as ArrowRsParquetFileReader calls it."""
    import ray_data_arrow_rs
    from pyarrow.fs import S3FileSystem

    if isinstance(fs, S3FileSystem):
        bucket, _, key = path.partition("/")
        reader = ray_data_arrow_rs.read_row_groups_s3(
            bucket,
            key,
            os.environ.get("AWS_DEFAULT_REGION", "us-west-2"),
            False,  # anonymous
            endpoint=None,
            access_key_id=None,
            secret_access_key=None,
            session_token=None,
            allow_http=False,
            virtual_hosted_style=False,
            row_groups=None,
            columns=None,
            batch_size=batch_size,
            decode_budget_bytes=DECODE_BUDGET_BYTES,
            fetch_window_mb=FETCH_WINDOW_MB,
            k=K,
            split_threshold_bytes=SPLIT_THRESHOLD_BYTES,
            predicate_json=None,
            column_fetch_mb=COLUMN_FETCH_MB,
            prefetch_budget_mb=PREFETCH_BUDGET_MB,
        )
    else:
        reader = ray_data_arrow_rs.read_row_groups(
            path,
            None,
            None,
            batch_size,
            DECODE_BUDGET_BYTES,
            K,
            SPLIT_THRESHOLD_BYTES,
            None,
        )
    yield from pa.RecordBatchReader.from_stream(reader)


def run_file(
    fs: FileSystem,
    path: str,
    reader: str,
    batch_size: int,
    writer_dir: Optional[str],
    index: int,
    readahead: int = -1,
) -> Dict[str, int]:
    source = read_arrow_rs if reader == "arrow_rs" else read_pyarrow
    writer = None
    rows = nbytes = batches = 0
    try:
        for batch in source(fs, path, batch_size, readahead):
            rows += batch.num_rows
            nbytes += batch.nbytes
            batches += 1
            if writer_dir is not None:
                if writer is None:
                    writer = pq.ParquetWriter(
                        os.path.join(writer_dir, f"part-{index:05d}.parquet"),
                        batch.schema,
                        compression="snappy",
                    )
                writer.write_batch(batch)
    finally:
        if writer is not None:
            writer.close()
    return {"rows": rows, "nbytes": nbytes, "batches": batches}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", required=True, help="local dir/file or s3://...")
    parser.add_argument("--reader", choices=("pyarrow", "arrow_rs"), required=True)
    parser.add_argument(
        "--write-to",
        default=None,
        help="write the decoded batches here (local dir). Omitted = read only.",
    )
    parser.add_argument("--batch-size", type=int, default=131072)
    parser.add_argument("--max-files", type=int, default=0, help="0 = all")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="threads over files. 1 is the clean crate-vs-scanner comparison; "
        "higher approximates Ray's concurrent read tasks, but in ONE process, "
        "so it is not equivalent to N Ray workers.",
    )
    parser.add_argument(
        "--readahead",
        type=int,
        default=-1,
        help="PyArrow scanner fragment/batch readahead. -1 keeps PyArrow's "
        "defaults (4/16), which is what Ray runs and which pipelines I/O "
        "against decode. 0 serializes the scanner, so the arms compare decode "
        "cost rather than pipelining. Ignored by the arrow_rs arm.",
    )
    parser.add_argument("--out", default=None, help="write a result JSON here")
    args = parser.parse_args()

    fs, files = list_parquet(args.source)
    if args.max_files:
        files = files[: args.max_files]
    if not files:
        print(f"no parquet under {args.source}", file=sys.stderr)
        return 1

    if args.write_to:
        # Start from empty so a re-run measures a write, not an overwrite of a
        # warm page cache full of last run's output.
        shutil.rmtree(args.write_to, ignore_errors=True)
        os.makedirs(args.write_to, exist_ok=True)

    if args.reader == "arrow_rs":
        import ray_data_arrow_rs  # noqa: F401  -- fail now, not mid-run

    # Touch one footer before baselining so the import ramp and the first S3
    # connection are outside the measurement. Agents.md documents an earlier
    # harness that skipped this and swung 133->211 MB between identical runs.
    with fs.open_input_file(files[0]) as handle:
        pq.read_metadata(handle)

    sampler = MemorySampler()
    baseline = sampler.read()[0]
    sampler.start()
    started = time.perf_counter()
    # rusage sums CPU across ALL threads of the process, so cpu/wall > 1 means
    # the arm used more than one core -- the only way to tell a slower decoder
    # apart from an unpipelined one when both are reported as wall time.
    cpu0 = resource.getrusage(resource.RUSAGE_SELF)

    totals = {"rows": 0, "nbytes": 0, "batches": 0}
    if args.concurrency > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [
                pool.submit(
                    run_file,
                    fs,
                    p,
                    args.reader,
                    args.batch_size,
                    args.write_to,
                    i,
                    args.readahead,
                )
                for i, p in enumerate(files)
            ]
            for fut in futures:
                for key, val in fut.result().items():
                    totals[key] += val
    else:
        for i, path in enumerate(files):
            for key, val in run_file(
                fs, path, args.reader, args.batch_size, args.write_to, i, args.readahead
            ).items():
                totals[key] += val

    wall = time.perf_counter() - started
    cpu1 = resource.getrusage(resource.RUSAGE_SELF)
    cpu = (cpu1.ru_utime - cpu0.ru_utime) + (cpu1.ru_stime - cpu0.ru_stime)
    mem = sampler.stop()

    result = {
        "source": args.source,
        "transport": "s3" if args.source.startswith("s3://") else "local",
        "reader": args.reader,
        "mode": "write" if args.write_to else "read",
        "concurrency": args.concurrency,
        "readahead": args.readahead,
        "files": len(files),
        "wall_s": round(wall, 2),
        "cpu_s": round(cpu, 2),
        # >1 means the arm used more than one core. If the pyarrow arm is above
        # 1 and the arrow_rs arm is at 1, a wall-time gap is pipelining, not a
        # slower decoder -- and cpu_s is then the fair comparison.
        "cpu_per_wall": round(cpu / wall, 2) if wall > 0 else None,
        "baseline_uss": baseline,
        # The number that matters: private memory the read actually added, above
        # a post-import baseline. Absolute peak buries it under the interpreter.
        # max(0, ...) because a peak below baseline means the sampler failed, not
        # that the read freed memory it never allocated.
        "uss_rise": max(0, mem.get("uss_peak", 0) - baseline),
        **totals,
        **mem,
    }
    print(json.dumps(result, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump(result, handle, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
