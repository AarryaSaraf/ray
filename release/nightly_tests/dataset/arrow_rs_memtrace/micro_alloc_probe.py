"""Bypass-Ray microbenchmark: is the multi-row-group memory loss the crate, or
the integration (Ray's per-row-group fan-out + the system allocator)?

Ray hands the arrow-rs reader ONE row group per `_iter_fragment_tables` call, so a
10-group file becomes 10 sequential `read_row_groups` calls in a worker. The
standalone bench instead read all groups through ONE reader AND linked mimalloc.
This probe pulls Ray out of the picture and decodes the SAME file three ways in
three fresh child processes, sampling this process's USS the whole time:

  pyarrow      — pq.ParquetFile.iter_batches over all row groups (the baseline)
  rs_onecall   — ray_data_arrow_rs.read_row_groups(path, ALL row groups)  [standalone-like]
  rs_perloop   — one read_row_groups(path, [rg]) call PER row group        [Ray-like fan-out]

If rs_onecall is flat and rs_perloop balloons ~linearly in group count, the loss is
the per-group-call + allocator-retention pattern, not a byte-budget bug — and
LD_PRELOAD-ing jemalloc/mimalloc (run this whole script under it) should flatten
rs_perloop too. Peak USS is the private decode heap (excludes shared libs), the
same quantity the suite's worker sampler records.

Usage:
  python micro_alloc_probe.py <parquet_dir_or_file> [budget_mb]
  # A/B the allocator:
  LD_PRELOAD=$(dpkg -L libjemalloc2 | grep 'so.2$') python micro_alloc_probe.py <dir>
"""
import os
import subprocess
import sys
import threading
import time

MB = 1024 * 1024


def _peak_uss_while(fn):
    """Run fn() in this process; sample USS at 2 ms in a thread; return (peak_mb, secs)."""
    import psutil

    proc = psutil.Process()
    peak = [0]
    stop = [False]

    def _sample():
        while not stop[0]:
            try:
                peak[0] = max(peak[0], proc.memory_full_info().uss)
            except Exception:
                break
            time.sleep(0.002)

    t = threading.Thread(target=_sample, daemon=True)
    t.start()
    t0 = time.time()
    fn()
    secs = time.time() - t0
    stop[0] = True
    t.join(timeout=1)
    return peak[0] / MB, secs


def _one_file(path):
    """First .parquet under path (or path itself)."""
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for f in sorted(files):
                if f.endswith(".parquet"):
                    return os.path.join(root, f)
        raise SystemExit(f"no .parquet under {path}")
    return path


def _run_child(mode, path, budget_bytes):
    """The actual decode, run in a FRESH process so each allocator starts clean."""
    import pyarrow.parquet as pq

    n_groups = pq.ParquetFile(path).metadata.num_row_groups
    rss = list(range(n_groups))

    def pyarrow_read():
        pf = pq.ParquetFile(path)
        total = 0
        for b in pf.iter_batches():
            total += b.num_rows
        return total

    def rs_onecall():
        import pyarrow as pa
        import ray_data_arrow_rs as rs

        stream = rs.read_row_groups(path, rss, None, 131072, budget_bytes, 1, 128 * MB)
        rb = pa.RecordBatchReader.from_stream(stream)
        total = 0
        for batch in rb:
            total += batch.num_rows
        return total

    def rs_perloop():
        import pyarrow as pa
        import ray_data_arrow_rs as rs

        total = 0
        for g in rss:  # mimic Ray: one call per row group, sequential
            stream = rs.read_row_groups(path, [g], None, 131072, budget_bytes, 1, 128 * MB)
            rb = pa.RecordBatchReader.from_stream(stream)
            for batch in rb:
                total += batch.num_rows
        return total

    def baseline():
        # interpreter + pyarrow floor only (no arrow-rs, no decode) — the constant
        # that Ray workers already carry in their warm baseline.
        import pyarrow  # noqa: F401

        return 0

    def rs_import():
        # + import the native extension and touch it once (build a reader, read
        # ZERO batches) so first-use init is paid but no row group is decoded. The
        # delta rs_import - baseline is the fixed per-worker cost that, in Ray, is
        # NOT in the warm baseline (pyarrow IS) and gets summed across workers.
        import pyarrow as pa
        import ray_data_arrow_rs as rs

        stream = rs.read_row_groups(path, [], None, 131072, budget_bytes, 1, 128 * MB)
        _ = pa.RecordBatchReader.from_stream(stream).schema
        return 0

    fn = {
        "baseline": baseline,
        "rs_import": rs_import,
        "pyarrow": pyarrow_read,
        "rs_onecall": rs_onecall,
        "rs_perloop": rs_perloop,
    }[mode]
    peak, secs = _peak_uss_while(fn)
    print(f"{mode}\t{n_groups}\t{peak:.1f}\t{secs:.3f}")


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    if len(sys.argv) >= 4 and sys.argv[3] == "__child__":
        # child entry: _run_child(mode, path, budget_bytes)
        _run_child(sys.argv[1], sys.argv[2], int(sys.argv[4]))
        return

    path = _one_file(sys.argv[1])
    budget_bytes = int(float(sys.argv[2]) * MB) if len(sys.argv) > 2 else 2 * MB
    print(f"file: {path}")
    print(f"budget: {budget_bytes // MB} MiB   LD_PRELOAD={os.environ.get('LD_PRELOAD', '(none)')}")
    print("mode\trow_groups\tpeak_uss_MB\tsecs")
    for mode in ["baseline", "rs_import", "pyarrow", "rs_onecall", "rs_perloop"]:
        # fresh child per mode so no allocator carryover between modes
        subprocess.run(
            [sys.executable, __file__, mode, path, "__child__", str(budget_bytes)],
            check=True,
            env=os.environ,
        )


if __name__ == "__main__":
    main()
