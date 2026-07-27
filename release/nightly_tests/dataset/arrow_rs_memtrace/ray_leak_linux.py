"""Per-task memory of ONE parquet read in Ray v1 / v2 / v3, measured with the
same 5ms per-worker USS sampler bench_suite uses (ds.stats() does not surface
peak heap on this Ray 3.0 nightly, so we read it off the worker USS traces).

    v1 -> use_datasource_v2=False              (V1 datasource, fragment.to_batches)
    v2 -> use_datasource_v2=True               (V2 reader, ParquetFile.iter_batches)
    v3 -> use_datasource_v2=True + arrow_rs    (our arrow-rs reader)

Usage:
    python mem3.py <v1|v2|v3> <parquet_path> [batch_size]   # batch_size: v1 only
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

    ray.shutdown()
    runtime_env = {
        "working_dir": B.HOOKDIR,
        "worker_process_setup_hook": "worker_mem_sampler.setup",
        "env_vars": {
            "RAY_MEM_TRACE_DIR": trace_dir,
            "RAY_MEM_TRACE_INTERVAL_S": "0.005",
            "RAY_DATA_USE_DATASOURCE_V2": use_v2,
            "RAY_DATA_USE_ARROW_RS_PARQUET_READER": use_ars,
        },
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

    ray.data.range(1000).map_batches(lambda b: b).materialize()  # warm workers
    time.sleep(0.5)

    kw = {"batch_size": bs} if (bs and mode == "v1") else {}
    ds = ray.data.read_parquet(path, **kw).map_batches(
        lambda b: b.slice(0, 1), batch_format="pyarrow")
    t0 = time.time()
    ds.materialize()
    t1 = time.time()
    time.sleep(0.3)

    absol = B._node_sum_peak_mb(trace_dir, t0, t1)
    incr = B._node_sum_incr_peak_mb(trace_dir, t0, t1)
    print(f"\n===== {mode}  (use_v2={use_v2} arrow_rs={use_ars})  bs={bs} =====")
    print(f"wall           : {t1 - t0:.2f}s")
    print(f"node-sum peak  : {absol:7.0f} MB   (absolute -- what the OOM killer sees)")
    print(f"node-sum incr  : {incr:7.0f} MB   (read-caused extra -- compare THIS)")
    ray.shutdown()


if __name__ == "__main__":
    main()