"""Per-task memory of ONE parquet read in Ray v1 / v2 / v3, measured with the
same 5ms per-worker USS sampler bench_suite uses (ds.stats() does not surface
peak heap on this Ray 3.0 nightly, so we read it off the worker USS traces).

    v1 -> use_datasource_v2=False              (V1 datasource, fragment.to_batches)
    v2 -> use_datasource_v2=True               (V2 reader, ParquetFile.iter_batches)
    v3 -> use_datasource_v2=True + arrow_rs    (our arrow-rs reader)

Usage:
    python mem3.py <v1|v2|v3> <parquet_path> [batch_size]   # bs forwarded to all modes

Reports, for the read window:
  node-sum peak  = SUM of every worker's absolute private heap (USS) -- what the
                   OOM killer / Ray memory monitor actually act on.
  node-sum incr  = same, minus each worker's warm baseline -- the EXTRA heap the
                   read itself caused. This is the number to compare across v1/v2/v3.
"""
import os
import shutil
import sys
import time

import ray

import bench_suite as B  # reuse HOOKDIR, OUT, MB, and the USS aggregators

FLAGS = {"v1": ("0", "0"), "v2": ("1", "0"), "v3": ("1", "1")}


def main():
    mode, path = sys.argv[1], sys.argv[2]
    bs = int(sys.argv[3]) if len(sys.argv) > 3 else None
    use_v2, use_ars = FLAGS[mode]

    trace_dir = os.path.join(B.OUT, f"mem3__{mode}")
    shutil.rmtree(trace_dir, ignore_errors=True)
    os.makedirs(trace_dir, exist_ok=True)
    # Per-fragment native/fallback trace: the arrow-rs reader appends "native" or
    # "fallback" to path_<host>_<pid>.log here (only v3 writes it). Lets us PROVE
    # the crate actually decoded rather than silently falling back to PyArrow.
    path_dir = os.path.join(B.OUT, f"mem3__{mode}__path")
    shutil.rmtree(path_dir, ignore_errors=True)
    os.makedirs(path_dir, exist_ok=True)

    ray.shutdown()
    env_vars = {
        "RAY_MEM_TRACE_DIR": trace_dir,
        "RAY_MEM_TRACE_INTERVAL_S": "0.005",
        "RAY_DATA_ARROW_RS_PATH_TRACE": path_dir,
        "RAY_DATA_USE_DATASOURCE_V2": use_v2,
        "RAY_DATA_USE_ARROW_RS_PARQUET_READER": use_ars,
    }
    # Forward the arrow-rs tuning knobs to the workers so they can be swept from
    # the shell (the read task runs on a worker, which only sees runtime_env
    # env_vars -- not the driver's ambient env). These are the v3 analog of
    # batch_size: DECODE_BUDGET_BYTES = bytes per decode batch, K = intra-row-
    # group split, FETCH_WINDOW_MB = S3 in-flight cap.
    knobs = []
    for k in ("RAY_DATA_ARROW_RS_DECODE_BUDGET_BYTES",
              "RAY_DATA_ARROW_RS_K",
              "RAY_DATA_ARROW_RS_FETCH_WINDOW_MB"):
        if k in os.environ:
            env_vars[k] = os.environ[k]
            knobs.append(f"{k[len('RAY_DATA_ARROW_RS_'):]}={os.environ[k]}")
    runtime_env = {
        "working_dir": B.HOOKDIR,
        "worker_process_setup_hook": "worker_mem_sampler.setup",
        "env_vars": env_vars,
    }
    cluster = bool(os.environ.get("RAY_ADDRESS")) or os.path.exists(
        "/tmp/ray/ray_current_cluster"
    )
    ray.init(address="auto" if cluster else None, ignore_reinit_error=True,
             log_to_driver=False, runtime_env=runtime_env)

    from ray.data.context import DataContext
    c = DataContext.get_current()
    c.use_datasource_v2 = use_v2 == "1"
    c.use_arrow_rs_parquet_reader = use_ars == "1"

    # Warm the workers so their USS baseline at window-start already includes
    # imports/allocator floor (the incr number then isolates decode growth).
    ray.data.range(1000).map_batches(lambda b: b).materialize()
    time.sleep(0.5)

    # Forward batch_size to ALL modes so the comparison is apples-to-apples: the
    # SAME user code read_parquet(path, batch_size=bs) run against each reader.
    # v1's scanner accumulates one decoded chunk per batch (the ray#49158 leak);
    # v2/v3 stream a row group at a time and coalesce downstream, so they're
    # immune to the setting -- that immunity is exactly what we're demonstrating.
    kw = {"batch_size": bs} if bs else {}
    # slice each block to 1 row so the object store / driver retain ~nothing;
    # the READ task still fully decodes, so its per-worker USS is unaffected.
    ds = ray.data.read_parquet(path, **kw).map_batches(
        lambda b: b.slice(0, 1), batch_format="pyarrow")
    t0 = time.time()
    ds.materialize()
    t1 = time.time()
    time.sleep(0.3)  # let the 5ms samplers flush their tails

    absol = B._node_sum_peak_mb(trace_dir, t0, t1)
    incr = B._node_sum_incr_peak_mb(trace_dir, t0, t1)

    # Tally which decode path the arrow-rs reader chose per fragment.
    import glob
    verdicts = []
    for fn in glob.glob(os.path.join(path_dir, "path_*.log")):
        with open(fn) as fh:
            verdicts += fh.read().split()
    n_native = verdicts.count("native")
    n_fallback = verdicts.count("fallback")

    knob_str = ("  " + " ".join(knobs)) if knobs else ""
    print(f"\n===== {mode}  (use_v2={use_v2} arrow_rs={use_ars})  bs={bs}{knob_str} =====")
    print(f"wall           : {t1 - t0:.2f}s")
    print(f"node-sum peak  : {absol:7.0f} MB   (absolute -- what the OOM killer sees)")
    print(f"node-sum incr  : {incr:7.0f} MB   (read-caused extra -- compare THIS)")
    if mode == "v3":
        if n_native or n_fallback:
            print(f"decode path    : native={n_native}  fallback={n_fallback}  "
                  f"({'ARROW-RS' if n_native and not n_fallback else 'MIXED/FALLBACK'})")
        else:
            print("decode path    : (no trace written -- reader never hit "
                  "_iter_fragment_tables?)")
    ray.shutdown()


if __name__ == "__main__":
    main()
