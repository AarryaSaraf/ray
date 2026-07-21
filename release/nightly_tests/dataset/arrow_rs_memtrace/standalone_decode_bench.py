"""Ray-FREE raw-decode benchmark — isolates the arrow-rs vs PyArrow decode CPU.

This is "test #2" (the ~25-40% K=1 decode-CPU gap, and the O(n) scaling proof) with
ALL Ray overhead removed: no ray.init, no workers, no object store, no scheduler.
It drives the native crate's decode loop directly (the SAME code path a Ray worker's
`_iter_fragment_tables` runs) and PyArrow's `iter_batches`, decoding every batch and
dropping it, timing the full pass. Because nothing is materialized, wall time IS raw
decode throughput.

Use this to iterate on the crate: rebuild `ray_data_arrow_rs` (`maturin develop
--release`) with a change, rerun this, and read the µs/row column. Sweep the decode
budget and K here without touching Ray. When a change looks good here, confirm it in
the Ray-integrated suite (`bench_suite.py scaling,tuning`).

What each reader does (mirrors the integrated path exactly):
- arrow-rs: `ray_data_arrow_rs.read_row_groups(path, row_groups=None, columns=None,
  batch_size, decode_budget_bytes, k, split_threshold_bytes)` -> Arrow C-stream ->
  `pa.RecordBatchReader.from_stream(...)`, iterate + drop. `k`/`split_threshold_bytes`
  expose the intra-row-group split; `decode_budget_bytes` sets the byte budget.
- PyArrow: `pq.ParquetFile(path).iter_batches(batch_size=...)`, iterate + drop.

Usage:
  python standalone_decode_bench.py                      # default sweep
  python standalone_decode_bench.py --rows 8000000 --budgets 1,8,32 --ks 1,2,4
"""
import argparse
import time

import pyarrow as pa
import pyarrow.parquet as pq

import fixtures as fx

MB = 1024 * 1024
ray_data_arrow_rs = __import__("ray_data_arrow_rs")


def _time_arrow_rs(path, budget_bytes, k, split_threshold_bytes, batch_size=131072):
    # process_time() = CPU seconds across ALL threads (so K-split accumulates K
    # workers' CPU). perf_counter() = wall. cpu/wall ~= 1 on a single-thread path
    # with a warm cache means the run is CPU-bound (no I/O stall to pipeline away);
    # cpu/wall > k means threads are stalling on I/O (pipelining would help).
    w0, c0 = time.perf_counter(), time.process_time()
    stream = ray_data_arrow_rs.read_row_groups(
        path, None, None, batch_size, budget_bytes, k, split_threshold_bytes
    )
    reader = pa.RecordBatchReader.from_stream(stream)
    n = 0
    for batch in reader:  # decode one byte-budget batch, drop it
        n += batch.num_rows
    return time.perf_counter() - w0, time.process_time() - c0, n


def _time_pyarrow(path, batch_size, threads=None):
    if threads is not None:
        pa.set_cpu_count(threads)
    w0, c0 = time.perf_counter(), time.process_time()
    pf = pq.ParquetFile(path)
    n = 0
    for batch in pf.iter_batches(batch_size=batch_size):
        n += batch.num_rows
    return time.perf_counter() - w0, time.process_time() - c0, n


def _budget_rows(path, budget_bytes):
    """Mirror the crate's byte_budget_rows so the PyArrow baseline uses a comparable
    batch size (footer uncompressed bytes / rows)."""
    md = pq.ParquetFile(path).metadata
    rg = md.row_group(0)
    bpr = max(rg.total_byte_size / max(rg.num_rows, 1), 1.0)
    return max(2048, int(budget_bytes / bpr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=str, default="1000000,2000000,4000000,8000000")
    ap.add_argument("--budgets", type=str, default="8", help="MB, comma-sep")
    ap.add_argument("--ks", type=str, default="1", help="comma-sep")
    ap.add_argument("--schema", type=str, default="wide_str")
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()

    row_list = [int(x) for x in args.rows.split(",")]
    budgets = [int(x) for x in args.budgets.split(",")]
    ks = [int(x) for x in args.ks.split(",")]
    full_threads = pa.cpu_count()  # hardware thread count, captured before we pin it

    def best_of(fn):
        # best-of wall, carrying that run's cpu time and row count.
        runs = [fn() for _ in range(args.repeats)]
        return min(runs, key=lambda r: r[0])

    print(f"schema={args.schema} repeats={args.repeats} (best-of wall)")
    print("Baseline = PyArrow-1thread (Ray-representative: OMP_NUM_THREADS=1, but "
          "pre_buffer I/O still on).\narrow-rs is swept over budget/k; compare BEST "
          "arrow-rs to that baseline. cpu/wall~1 => CPU-bound (no I/O to pipeline).")
    print(f"\n{'rows':>10s} {'reader/cfg':>22s} {'wall_s':>8s} {'us/row':>8s} "
          f"{'cpu/wall':>8s} {'vs pa':>7s}")
    for rows in row_list:
        spec = {"rows": rows, "num_files": 1, "row_group_size": rows, "schema": args.schema}
        d = fx.make_fixture(f"sa_{args.schema}_{rows}", spec)
        import glob
        import os
        path = sorted(glob.glob(os.path.join(d, "*.parquet")))[0]

        # PyArrow baseline at a comparable batch size (8 MB budget -> rows).
        # Report BOTH: 1-thread (what a Ray worker actually gets — OMP_NUM_THREADS=1)
        # and full-thread (what you get standalone, NOT what Ray gives).
        pa_bs = _budget_rows(path, 8 * MB)
        pa_wall, pa_cpu, _ = best_of(lambda: _time_pyarrow(path, pa_bs, threads=1))
        pa_full, _, _ = best_of(lambda: _time_pyarrow(path, pa_bs, threads=full_threads))
        pa.set_cpu_count(1)  # keep arrow-rs comparison honest (crate owns its threads)
        print(f"{rows:>10d} {'pyarrow-1thread':>22s} {pa_wall:8.3f} "
              f"{pa_wall/rows*1e6:8.4f} {pa_cpu/pa_wall:8.2f} {'1.00':>7s}")
        print(f"{rows:>10d} {'pyarrow-%dthread' % full_threads:>22s} {pa_full:8.3f} "
              f"{pa_full/rows*1e6:8.4f} {'':>8s} {pa_full/pa_wall:7.2f}")

        best = None
        for b in budgets:
            for k in ks:
                cfg = f"arrow_rs b={b}MB k={k}"
                wall, cpu, _ = best_of(
                    lambda b=b, k=k: _time_arrow_rs(path, b * MB, k, 0 if k > 1 else 128 * MB)
                )
                print(f"{rows:>10d} {cfg:>22s} {wall:8.3f} "
                      f"{wall/rows*1e6:8.4f} {cpu/wall:8.2f} {wall/pa_wall:7.2f}")
                if best is None or wall < best[1]:
                    best = (cfg, wall)
        print(f"{rows:>10d} {'-> BEST arrow-rs':>22s} {best[1]:8.3f} "
              f"{best[1]/rows*1e6:8.4f} {'':>8s} {best[1]/pa_wall:7.2f}   ({best[0]})")
    print("\nInterpretation:")
    print("  us/row flat-or-decreasing across rows  => O(n) (no per-batch reader rebuild).")
    print("  cpu/wall ~= 1 on a k=1 warm-cache run   => CPU-bound; missing I/O pipelining")
    print("     is NOT the local gap (there is no I/O stall to hide). The gap is the")
    print("     string-decode kernel. Pipelining only matters on cold cache / S3.")
    print("  'vs pa' for BEST arrow-rs is the honest tuned speed ratio (<1 = faster).")


if __name__ == "__main__":
    main()
