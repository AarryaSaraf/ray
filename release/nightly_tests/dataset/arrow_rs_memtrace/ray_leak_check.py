"""End-to-end reproduction of ray#49158 THROUGH Ray Data (not standalone pyarrow).

Reproduces the issue's own repro shape:
    ds = ray.data.read_parquet(path)
    ds = ds.map_batches(lambda b: b)
    print(ds.materialize().stats())

...but toggles the read path so we can compare the leaky scanner vs the
streaming reader inside a real Ray task:

    v1          -> use_datasource_v2=False  (V1 datasource, fragment.to_batches -> LEAKY)
    v2_pyarrow  -> use_datasource_v2=True   (V2 reader, ParquetFile.iter_batches)
    v2_arrow_rs -> use_datasource_v2=True + use_arrow_rs_parquet_reader=True

Run ONE config per process (fresh Ray) for a clean baseline. A background thread
samples node RSS (driver + all Ray worker children) so we get a real memory peak
even on macOS where per-task USS is unavailable. We also print ds.stats(), which
carries the same "Peak heap memory usage (MiB)" line the issue quoted.

Usage:
    python ray_leak_check.py <v1|v2_pyarrow|v2_arrow_rs> <parquet_path> [batch_size]
"""
import os
import sys
import threading
import time

import psutil

MB = 1024 * 1024


class NodeRSSSampler(threading.Thread):
    """Sample summed RSS across this process + all descendants (Ray workers)."""

    def __init__(self, interval=0.02):
        super().__init__(daemon=True)
        self.interval = interval
        self._stop_event = threading.Event()
        self.peak_mb = 0.0
        self.baseline_mb = 0.0

    def _node_rss_mb(self):
        me = psutil.Process()
        total = me.memory_info().rss
        for c in me.children(recursive=True):
            try:
                total += c.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return total / MB

    def snapshot_baseline(self):
        self.baseline_mb = self._node_rss_mb()

    def run(self):
        while not self._stop_event.is_set():
            self.peak_mb = max(self.peak_mb, self._node_rss_mb())
            time.sleep(self.interval)

    def stop(self):
        self._stop_event.set()
        self.join(timeout=2)


def _extract_peak_heap(stats_str):
    """Pull the 'Peak heap memory usage (MiB): a min, b max, c mean' line."""
    for line in stats_str.splitlines():
        if "Peak heap memory usage" in line:
            return line.strip()
    return "(no peak-heap line in stats)"


def main():
    mode = sys.argv[1]
    path = sys.argv[2]
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else None

    import ray
    from ray.data.context import DataContext

    ray.init(num_cpus=1, object_store_memory=512 * MB, log_to_driver=False)

    ctx = DataContext.get_current()
    if mode == "v1":
        ctx.use_datasource_v2 = False
        ctx.use_arrow_rs_parquet_reader = False
    elif mode == "v2_pyarrow":
        ctx.use_datasource_v2 = True
        ctx.use_arrow_rs_parquet_reader = False
    elif mode == "v2_arrow_rs":
        ctx.use_datasource_v2 = True
        ctx.use_arrow_rs_parquet_reader = True
    else:
        raise SystemExit(f"unknown mode {mode!r}")

    read_kwargs = {}
    if batch_size is not None:
        # In Ray 3.0 read_parquet forwards unknown kwargs straight to
        # pyarrow's to_batches, so batch_size is passed directly (V1 path).
        read_kwargs["batch_size"] = batch_size

    sampler = NodeRSSSampler()
    sampler.start()

    # Warm up Ray workers with a trivial op so the baseline excludes spawn cost.
    ray.data.range(1).materialize()
    time.sleep(0.3)
    sampler.snapshot_baseline()

    t0 = time.time()
    ds = ray.data.read_parquet(path, **read_kwargs)
    ds = ds.map_batches(lambda b: b)
    mat = ds.materialize()
    wall = time.time() - t0
    stats = mat.stats()

    time.sleep(0.2)
    sampler.stop()

    print("=" * 70)
    print(f"mode={mode}  batch_size={batch_size}  wall={wall:.2f}s")
    print(f"node RSS baseline (post-init, warm): {sampler.baseline_mb:.0f} MB")
    print(f"node RSS PEAK (driver+workers):      {sampler.peak_mb:.0f} MB")
    print(f"node RSS delta over baseline:        "
          f"{sampler.peak_mb - sampler.baseline_mb:.0f} MB")
    print(f"ray-reported: {_extract_peak_heap(stats)}")
    print(f"rows={mat.count():,}")
    print("=" * 70)

    ray.shutdown()


if __name__ == "__main__":
    main()
