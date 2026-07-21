"""Fair PyArrow-vs-arrow-rs benchmark for the Ray Data V2 Parquet reader.

Measures **wall time** and **RSS-over-time** (peak + trace) for reading a
Parquet workload two ways through the *same* Ray Data code path — the only
thing that changes is ``DataContext.use_arrow_rs_parquet_reader``.

Memory note: Ray records per-task peak USS in block metadata, but only on
Linux (macOS lacks the ``shared`` field). So we sample cluster-wide RSS of the
Ray worker processes over wall-clock via psutil, which works on both platforms;
on Linux we *also* surface Ray's per-task ``max_uss_bytes`` from ``ds.stats()``.

This is a prototype harness. The workload matrix is CLI-configurable; point it
at the file sizes / row-group sizes / schema widths you want to sweep.

Example
-------
    RAY_DATA_USE_DATASOURCE_V2=1 python arrow_rs_read_benchmark.py \
        --rows 4_000_000 --row-group-size 1_000_000 --num-files 1 \
        --consume count --out-dir /tmp/arrow_rs_bench
"""
import argparse
import csv
import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
import psutil
import pyarrow as pa
import pyarrow.parquet as pq


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def make_fixture(
    out_dir: str,
    rows: int,
    num_files: int,
    row_group_size: int,
    num_int_cols: int,
    num_float_cols: int,
    num_str_cols: int,
    str_width: int,
    compression: str,
) -> List[str]:
    """Write ``num_files`` Parquet files (page index on) and return their paths."""
    os.makedirs(out_dir, exist_ok=True)
    rows_per_file = rows // num_files
    paths = []
    rng = np.random.default_rng(0)
    for f in range(num_files):
        cols: Dict[str, pa.Array] = {}
        base = f * rows_per_file
        cols["id"] = pa.array(np.arange(base, base + rows_per_file, dtype=np.int64))
        for i in range(num_int_cols):
            cols[f"i{i}"] = pa.array(
                rng.integers(0, 1 << 30, rows_per_file, dtype=np.int64)
            )
        for i in range(num_float_cols):
            cols[f"f{i}"] = pa.array(rng.random(rows_per_file))
        for i in range(num_str_cols):
            cols[f"s{i}"] = pa.array(
                [_rand_str(rng, str_width) for _ in range(rows_per_file)]
            )
        table = pa.table(cols)
        path = os.path.join(out_dir, f"part-{f:04d}.parquet")
        pq.write_table(
            table,
            path,
            row_group_size=row_group_size,
            write_page_index=True,
            compression=compression,
        )
        paths.append(path)
    total_bytes = sum(os.path.getsize(p) for p in paths)
    rg = pq.ParquetFile(paths[0]).num_row_groups
    print(
        f"fixture: {num_files} file(s), {rows} rows, {rg} row-group(s)/file, "
        f"{total_bytes/1e6:.1f} MB on disk, schema="
        f"{num_int_cols}i+{num_float_cols}f+{num_str_cols}s(w{str_width})"
    )
    return paths


def _rand_str(rng, width: int) -> str:
    # Deterministic-ish variable content at fixed width.
    return "".join(chr(97 + int(x)) for x in rng.integers(0, 26, width))


# --------------------------------------------------------------------------- #
# RSS-over-time sampler (cluster-wide, Ray worker processes)
# --------------------------------------------------------------------------- #
class RSSSampler:
    """Polls per-process RSS of Ray worker processes on a background thread.

    Matches processes whose cmdline looks like a Ray task/actor worker
    (``ray::`` or ``default_worker.py``). Tracks, *per PID*, the min and max
    RSS seen over the window, plus the peak summed total. Works on macOS and
    Linux (plain RSS, includes shared pages — treat absolute macOS numbers as
    directional; the pyarrow-vs-arrow-rs *delta* is the signal).

    Why per-PID: the OOM-risk metric is how much the *busiest single worker's
    own footprint* grows during the read. Subtracting a global "max single
    worker at baseline" from "max single worker at peak" (an earlier version)
    silently compares two *different* processes when the pool's busiest worker
    changes — which zeroed out the single-fragment case. Per-PID (max - min)
    measures each process's own growth and takes the largest.
    """

    def __init__(self, interval_s: float = 0.02):
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # trace rows: (t_s, total_worker_rss_mb, max_single_worker_rss_mb)
        self.trace: List[tuple] = []
        self.peak_total_mb: float = 0.0
        # Per-PID min/max RSS (bytes) over the sampling window.
        self._pid_min: Dict[int, int] = {}
        self._pid_max: Dict[int, int] = {}
        self._t0 = 0.0

    @staticmethod
    def _sample_pids() -> Dict[int, int]:
        """Return {pid: rss_bytes} for the current Ray worker processes."""
        out: Dict[int, int] = {}
        for p in psutil.process_iter(["pid", "cmdline", "memory_info"]):
            try:
                cmd = " ".join(p.info["cmdline"] or [])
                if "ray::" in cmd or "default_worker.py" in cmd:
                    out[p.info["pid"]] = p.info["memory_info"].rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return out

    @staticmethod
    def _worker_rss_mb() -> tuple:
        """Return (total_rss_mb, max_single_worker_rss_mb) over Ray workers."""
        pids = RSSSampler._sample_pids()
        total = sum(pids.values())
        biggest = max(pids.values()) if pids else 0
        return total / (1024 * 1024), biggest / (1024 * 1024)

    @property
    def incr_peak_single_mb(self) -> float:
        """Largest single-process RSS growth (max-min) over the window, in MB.

        Self-baselining per PID: the first samples catch each worker at its
        warm/idle footprint, so ``max - min`` is that worker's read working
        set. This is the number that predicts per-worker OOM.
        """
        best = 0
        for pid, mx in self._pid_max.items():
            best = max(best, mx - self._pid_min.get(pid, mx))
        return best / (1024 * 1024)

    def _run(self):
        while not self._stop.is_set():
            pids = self._sample_pids()
            total = sum(pids.values())
            single = max(pids.values()) if pids else 0
            t = time.perf_counter() - self._t0
            self.trace.append((t, total / (1024 * 1024), single / (1024 * 1024)))
            self.peak_total_mb = max(self.peak_total_mb, total / (1024 * 1024))
            for pid, rss in pids.items():
                if rss < self._pid_min.get(pid, rss + 1):
                    self._pid_min[pid] = rss
                if rss > self._pid_max.get(pid, -1):
                    self._pid_max[pid] = rss
            self._stop.wait(self._interval)

    def __enter__(self):
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
@dataclass
class Result:
    reader: str
    consume: str
    wall_s: float
    # Incremental peak RSS above the post-warmup baseline — the honest number.
    incr_peak_total_rss_mb: float
    incr_peak_single_rss_mb: float
    baseline_total_rss_mb: float
    rows: int
    rows_per_s: float
    read_op_wall_s: Optional[float] = None
    max_task_uss_mb: Optional[float] = None  # Linux only
    rss_trace: List[tuple] = field(default_factory=list)


def _consume(ds, mode: str) -> int:
    if mode == "decode_drop":
        # Isolates the *decode* working set: a fused map_batches decodes every
        # column of each block, keeps only the row count, and drops the block.
        # No output blocks are retained (unlike iter_batches/materialize) and
        # every column is touched (unlike `sum("id")`, which projection-pushes
        # down to a single column). This is the right probe for what the decode
        # knobs (budget / K) actually move.
        import pyarrow.compute as _pc  # noqa: F401

        def _touch(batch):
            # Reference every column so the reader can't prune the decode.
            n = 0
            for col in batch.columns:
                n += len(col)
            return {"n": [batch.num_rows]}

        total = 0
        for out in ds.map_batches(_touch, batch_format="pyarrow").iter_batches(
            batch_format="pyarrow"
        ):
            total += int(sum(out["n"].to_pylist()))
        return total
    if mode == "sum":
        # Forces a FULL decode of every block, then drops it — the decoder's
        # working set is exercised without retaining output blocks. This is the
        # standalone benchmark's "decode + sum a column + drop" analog and is
        # the right metric for the decode transient. `.count()` short-circuits
        # to Parquet footer metadata and never decodes, so don't use it here.
        ds.sum("id")
        return ds.count()
    if mode == "count":
        return ds.count()
    if mode == "materialize":
        return ds.materialize().count()
    if mode == "iter_batches":
        n = 0
        for b in ds.iter_batches(batch_format="pyarrow"):
            n += b.num_rows
        return n
    raise ValueError(mode)


def run_one(paths: List[str], reader: str, consume: str, num_cpus: int) -> Result:
    """Run one reader in a FRESH Ray cluster to avoid cross-run contamination.

    Reported memory is *incremental* above a post-warmup baseline: we warm the
    workers (paying import/allocator startup once), sample the settled baseline
    RSS, then sample again during the read and subtract. This isolates the read
    working set from the fixed worker footprint (Agents.md §6.2).
    """
    import ray
    from ray.data.context import DataContext

    ray.shutdown()
    ray.init(ignore_reinit_error=True, num_cpus=num_cpus, logging_level="ERROR")

    ctx = DataContext.get_current()
    ctx.use_datasource_v2 = True
    ctx.use_arrow_rs_parquet_reader = reader == "arrow_rs"

    # Warm the workers so we measure incremental read memory, not cold import.
    # Touch all files so every worker that will read is spun up + settled.
    ray.data.read_parquet(paths).limit(len(paths)).count()
    time.sleep(0.3)
    baseline_total, _ = RSSSampler._worker_rss_mb()

    sampler = RSSSampler()
    with sampler:
        t0 = time.perf_counter()
        ds = ray.data.read_parquet(paths)
        rows = _consume(ds, consume)
        wall = time.perf_counter() - t0

    read_wall, max_uss = _extract_stats(ds)

    return Result(
        reader=reader,
        consume=consume,
        wall_s=round(wall, 3),
        # Total is peak summed RSS above the pre-read baseline. Single is the
        # largest *per-process* growth (self-baselining), so it stays valid
        # even when the pool's busiest worker changes identity mid-read.
        incr_peak_total_rss_mb=round(sampler.peak_total_mb - baseline_total, 1),
        incr_peak_single_rss_mb=round(sampler.incr_peak_single_mb, 1),
        baseline_total_rss_mb=round(baseline_total, 1),
        rows=rows,
        rows_per_s=round(rows / wall, 0) if wall else 0,
        read_op_wall_s=read_wall,
        max_task_uss_mb=max_uss,
        rss_trace=sampler.trace,
    )


def _extract_stats(ds):
    try:
        summary = ds._get_stats_summary()
    except Exception:
        return None, None
    read_wall = None
    max_uss = None
    for op in getattr(summary, "operators_stats", []):
        if "Read" in op.operator_name:
            read_wall = round(op.time_total_s, 3)
    # max_uss_bytes lives on per-block TaskExecWorkerStats (Linux only).
    return read_wall, max_uss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=2_000_000)
    ap.add_argument("--num-files", type=int, default=1)
    ap.add_argument("--row-group-size", type=int, default=500_000)
    ap.add_argument("--int-cols", type=int, default=4)
    ap.add_argument("--float-cols", type=int, default=4)
    ap.add_argument("--str-cols", type=int, default=2)
    ap.add_argument("--str-width", type=int, default=16)
    ap.add_argument("--compression", default="snappy")
    ap.add_argument(
        "--consume",
        default="sum",
        choices=["sum", "count", "materialize", "iter_batches", "decode_drop"],
        help="decode_drop = decode every column, drop blocks (isolates the "
        "decode working set — the knob-tuning metric); sum = decode id + "
        "drop (projection-pushed, narrow); count = footer-only (no decode); "
        "materialize/iter_batches = include retained blocks",
    )
    ap.add_argument("--readers", default="pyarrow,arrow_rs")
    ap.add_argument("--num-cpus", type=int, default=4)
    ap.add_argument("--out-dir", default="/tmp/arrow_rs_bench")
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    data_dir = args.data_dir or os.path.join(args.out_dir, "data")
    paths = make_fixture(
        data_dir,
        args.rows,
        args.num_files,
        args.row_group_size,
        args.int_cols,
        args.float_cols,
        args.str_cols,
        args.str_width,
        args.compression,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    results: List[Result] = []
    for reader in args.readers.split(","):
        res = run_one(paths, reader, args.consume, args.num_cpus)
        results.append(res)
        # Write the RSS trace CSV per reader.
        trace_path = os.path.join(args.out_dir, f"rss_trace_{reader}.csv")
        with open(trace_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["t_s", "total_worker_rss_mb", "max_single_worker_rss_mb"])
            w.writerows(res.rss_trace)
        print(
            f"  {reader:9s}  wall={res.wall_s:6.2f}s  "
            f"incr_peak_total={res.incr_peak_total_rss_mb:7.1f}MB  "
            f"peak_single_worker={res.incr_peak_single_rss_mb:7.1f}MB  "
            f"rows/s={res.rows_per_s:,.0f}  trace→{trace_path}"
        )

    # Summary table + JSON.
    summary_path = os.path.join(args.out_dir, "results.json")
    with open(summary_path, "w") as fh:
        json.dump(
            [{k: v for k, v in asdict(r).items() if k != "rss_trace"} for r in results],
            fh,
            indent=2,
        )

    if len(results) == 2:
        pa_r = next(r for r in results if r.reader == "pyarrow")
        rs_r = next(r for r in results if r.reader == "arrow_rs")
        print("\n=== comparison (incremental RSS above warm baseline) ===")
        print(
            f"  total worker mem: pyarrow {pa_r.incr_peak_total_rss_mb:.0f}MB vs "
            f"arrow-rs {rs_r.incr_peak_total_rss_mb:.0f}MB  "
            f"({pa_r.incr_peak_total_rss_mb / max(rs_r.incr_peak_total_rss_mb,1):.1f}x)"
        )
        print(
            f"  peak single worker: pyarrow {pa_r.incr_peak_single_rss_mb:.0f}MB vs "
            f"arrow-rs {rs_r.incr_peak_single_rss_mb:.0f}MB"
        )
        print(
            f"  time:   pyarrow {pa_r.wall_s:.2f}s vs "
            f"arrow-rs {rs_r.wall_s:.2f}s  "
            f"({pa_r.wall_s / max(rs_r.wall_s,0.001):.1f}x)"
        )
    print(f"\nresults → {summary_path}")


if __name__ == "__main__":
    main()
