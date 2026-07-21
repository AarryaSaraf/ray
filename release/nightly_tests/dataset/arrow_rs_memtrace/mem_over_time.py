"""Memory-over-time instrument for the arrow-rs vs PyArrow reader.

Measures the ONE quantity Ray's scheduler does NOT model: each worker's private
heap (USS = memory_full_info().uss, the same field Ray's MemoryProfiler samples
but never gates on). Per-worker USS is captured *inside* each worker by a
setup-hook sampler (hookdir/worker_mem_sampler.py) at 5 ms; the driver samples
Ray's object-store usage (what the scheduler DOES see) in parallel.

For each (reader x consume x fixture): fresh Ray session, warm the workers, then
run the measured read and record its wall-clock window. Emits per-run:
  uss_<pid>.csv           per-worker private-heap trace (epoch, uss, rss)
  objstore.csv            Ray's object-store-used trace (epoch, bytes)
  window.json             {t_start, t_end, reader, consume, fixture, wall_s}

Plotting is a separate step (plot_mem.py) so we never re-run to re-draw.
"""
import glob
import json
import os
import shutil
import threading
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import ray
from ray._private.internal_api import get_memory_info_reply, get_state_from_address

HERE = os.path.dirname(os.path.abspath(__file__))
HOOKDIR = os.path.join(HERE, "hookdir")
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "runs")

FIXTURES = {
    # name: (rows, num_files, row_group_size, str_cols, str_width)
    "big_1rg_200mb": (2_000_000, 1, 2_000_000, 2, 16),
    "huge_1rg_800mb": (8_000_000, 1, 8_000_000, 3, 48),
    "multi_rg_1file": (2_000_000, 1, 100_000, 2, 16),
    "eight_files": (2_000_000, 8, 250_000, 2, 16),
    # Tiny import-warmup fixture (10k rows, 1 rg) — never materializes a big group.
    "_warm": (10_000, 1, 10_000, 2, 16),
}
READERS = ["pyarrow", "arrow_rs"]
CONSUMES = ["decode_drop", "iter_batches"]


def _rand_str(rng, width):
    return "".join(chr(97 + int(x)) for x in rng.integers(0, 26, width))


def make_fixture(name):
    rows, num_files, rg, scols, swidth = FIXTURES[name]
    d = os.path.join(DATA, name)
    if os.path.isdir(d) and glob.glob(os.path.join(d, "*.parquet")):
        return d
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(0)
    per = rows // num_files
    for f in range(num_files):
        base = f * per
        cols = {"id": pa.array(np.arange(base, base + per, dtype=np.int64))}
        for i in range(4):
            cols[f"i{i}"] = pa.array(rng.integers(0, 1 << 30, per, dtype=np.int64))
        for i in range(4):
            cols[f"f{i}"] = pa.array(rng.random(per))
        for i in range(scols):
            cols[f"s{i}"] = pa.array([_rand_str(rng, swidth) for _ in range(per)])
        pq.write_table(pa.table(cols), os.path.join(d, f"part-{f:04d}.parquet"),
                       row_group_size=rg, write_page_index=True, compression="snappy")
    return d


class ObjStoreSampler:
    """Driver-side: sample Ray object-store bytes used (the scheduler's view)."""

    def __init__(self, interval=0.02):
        self._interval = interval
        self._stop = threading.Event()
        self._rows = []
        self._state = get_state_from_address(ray.get_runtime_context().gcs_address)
        self._t = None

    def __enter__(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join(timeout=2)

    def _run(self):
        while not self._stop.is_set():
            try:
                used = get_memory_info_reply(self._state).store_stats.object_store_bytes_used
                self._rows.append((time.time(), used))
            except Exception:
                pass
            self._stop.wait(self._interval)

    def write(self, path):
        with open(path, "w") as fh:
            fh.write("epoch_s,objstore_bytes\n")
            for t, b in self._rows:
                fh.write(f"{t:.6f},{b}\n")


def consume(ds, mode):
    if mode == "decode_drop":
        def _touch(batch):
            n = 0
            for col in batch.columns:
                n += len(col)
            return {"n": [batch.num_rows]}
        total = 0
        for out in ds.map_batches(_touch, batch_format="pyarrow").iter_batches(
            batch_format="pyarrow"):
            total += int(sum(out["n"].to_pylist()))
        return total
    if mode == "iter_batches":
        n = 0
        for b in ds.iter_batches(batch_format="pyarrow"):
            n += b.num_rows
        return n
    raise ValueError(mode)


def run_one(reader, mode, fixture, num_cpus=4):
    run_dir = os.path.join(OUT, f"{fixture}__{mode}__{reader}")
    if os.path.isdir(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir)
    path = make_fixture(fixture)

    ray.shutdown()
    ray.init(
        num_cpus=num_cpus, include_dashboard=False, ignore_reinit_error=True,
        log_to_driver=False,
        runtime_env={
            "working_dir": HOOKDIR,
            "worker_process_setup_hook": "worker_mem_sampler.setup",
            "env_vars": {
                "RAY_MEM_TRACE_DIR": run_dir,
                "RAY_MEM_TRACE_INTERVAL_S": "0.005",
                "RAY_DATA_USE_DATASOURCE_V2": "1",
                "RAY_DATA_USE_ARROW_RS_PARQUET_READER": "1" if reader == "arrow_rs" else "0",
                "RAY_DATA_ARROW_RS_K": "1",
                "RAY_DATA_ARROW_RS_DECODE_BUDGET_BYTES": str(8 * 1024 * 1024),
            },
        },
    )
    from ray.data.context import DataContext
    ctx = DataContext.get_current()
    ctx.use_datasource_v2 = True
    ctx.use_arrow_rs_parquet_reader = (reader == "arrow_rs")

    # Warm IMPORTS ONLY on a tiny separate file (routes through the same reader
    # so ray_data_arrow_rs / pyarrow modules import), WITHOUT materializing a big
    # row group — so the measured read below is a clean first big decode on an
    # import-only heap, not one contaminated by a prior decode's retained memory.
    warm_path = make_fixture("_warm")
    consume(ray.data.read_parquet(warm_path), mode)
    time.sleep(1.0)  # let workers settle to a warm-import idle floor

    t_start = time.time()
    with ObjStoreSampler() as oss:
        rows = consume(ray.data.read_parquet(path), mode)
    t_end = time.time()
    oss.write(os.path.join(run_dir, "objstore.csv"))

    with open(os.path.join(run_dir, "window.json"), "w") as fh:
        json.dump({"t_start": t_start, "t_end": t_end, "reader": reader,
                   "consume": mode, "fixture": fixture, "wall_s": t_end - t_start,
                   "rows": rows}, fh)
    ray.shutdown()
    time.sleep(0.5)
    print(f"  done {fixture}/{mode}/{reader}: wall={t_end-t_start:.2f}s rows={rows}")


def main():
    import sys
    fixtures = sys.argv[1].split(",") if len(sys.argv) > 1 else list(FIXTURES)
    os.makedirs(OUT, exist_ok=True)
    for fixture in fixtures:
        for mode in CONSUMES:
            for reader in READERS:
                print(f"RUN {fixture} / {mode} / {reader}")
                run_one(reader, mode, fixture)
    print("ALL DONE")


if __name__ == "__main__":
    main()
