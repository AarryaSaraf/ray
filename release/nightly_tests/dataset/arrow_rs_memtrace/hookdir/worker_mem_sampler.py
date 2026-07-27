"""worker_process_setup_hook: sample THIS worker's private heap (USS) over time,
and log the epoch window of every read task it executes.

Runs once per Ray worker process at startup. Two artifacts per worker, both in
``RAY_MEM_TRACE_DIR``:

  uss_<host>_<pid>.csv    (epoch_s, uss_bytes, rss_bytes) at a fine interval
  tasks_<host>_<pid>.csv  (t_start, t_end, tables) — one row per read task

WHAT MEMORY THIS IS, AND WHY IT IS WHAT THE TASK "ACTUALLY HAS":
USS (unique set size) = pages resident in THIS process and no other = the
worker's private malloc/Rust/PyArrow heap. It is measured ABSOLUTE — no
baseline subtraction — because the kernel OOM killer and Ray's memory monitor
act on absolute process memory: interpreter floor, imports, allocator
retention and the decode working set all count, for both readers equally.
USS deliberately EXCLUDES shared pages, i.e. the plasma object store — the
part of a task's footprint Ray's admission control DOES account for
(resource_manager.py:776 gates on object-store bytes only). So absolute USS
is precisely the per-task memory that is invisible to Ray's scheduler and
that Ray's own design assumes stays near 2 x target_max_block_size per task
(context.py:44). It is the same quantity Ray's MemoryProfiler records into
TaskExecWorkerStats.max_uss_bytes (map_operator.py:854) — ours adds the
over-time shape.

Task windows come from patching ``FileReader.read`` — the one per-task read
entrypoint BOTH readers inherit (file_reader.py:192; ArrowRsParquetFileReader
subclasses ParquetFileReader subclasses FileReader and neither overrides
``read``). Ray workers run one task at a time, so clipping this worker's USS
series to [t_start, t_end] is that task's memory over time. Importing
ray.data here (at worker startup, before any task) also equalizes the import
floor across readers, so no reader pays imports "inside" a task window.

Line-buffered writes so a killed/reused worker still leaves its samples.
"""
import atexit
import os
import socket
import threading
import time

import psutil

_started = False


def _patch_task_windows(trace_dir, host, pid):
    """Wrap ``read`` so every read task appends (t_start, t_end) to
    tasks_<host>_<pid>.csv.

    Both readers must be covered, but they no longer share ONE read entrypoint:
    the pyarrow reader inherits ``FileReader.read``, while
    ``ArrowRsParquetFileReader`` OVERRIDES ``read`` (the PyArrow-free native path,
    2026-07-24 migration) and only calls ``super().read()`` on its fallback
    branches. So we patch EACH class that defines ``read``. To avoid double-
    counting a native-then-fallback arrow-rs read (its override calls the base
    override), a per-instance reentrancy flag records only the OUTERMOST call —
    the one true task window. If the arrow-rs reader isn't importable (crate
    absent), only the base is patched and the sampler still runs."""
    tpath = os.path.join(trace_dir, f"tasks_{host}_{pid}.csv")

    def _record(t0, n):
        try:
            new = not os.path.exists(tpath)
            with open(tpath, "a", buffering=1) as tf:
                if new:
                    tf.write("t_start,t_end,tables\n")
                tf.write(f"{t0:.6f},{time.time():.6f},{n}\n")
        except Exception:
            pass

    def _wrap(cls):
        orig_read = cls.read

        def read(self, input_split):
            outer = not getattr(self, "_mem_trace_in_read", False)
            if outer:
                self._mem_trace_in_read = True
            t0 = time.time()
            n = 0
            try:
                for tbl in orig_read(self, input_split):
                    n += 1
                    yield tbl
            finally:
                if outer:
                    self._mem_trace_in_read = False
                    _record(t0, n)

        cls.read = read

    try:
        from ray.data._internal.datasource_v2.readers.file_reader import FileReader
    except Exception:
        return  # datasource_v2 unavailable — sampler still runs
    _wrap(FileReader)
    try:
        from ray.data._internal.datasource_v2.readers.arrow_rs_parquet_file_reader import (  # noqa: E501
            ArrowRsParquetFileReader,
        )

        _wrap(ArrowRsParquetFileReader)
    except Exception as e:  # crate/module absent — base patch covers pyarrow
        # Leave a breadcrumb so a missing arrow-rs task trace is diagnosable
        # (was it "the wrap never applied" vs "the worker died mid-read"?).
        try:
            with open(os.path.join(trace_dir, f"hookdbg_{host}_{pid}.log"), "a") as fh:
                fh.write(f"arrow_rs wrap SKIPPED: {type(e).__name__}: {e}\n")
        except Exception:
            pass


def setup():
    global _started
    if _started:
        return
    _started = True

    trace_dir = os.environ.get("RAY_MEM_TRACE_DIR")
    if not trace_dir:
        return
    interval = float(os.environ.get("RAY_MEM_TRACE_INTERVAL_S", "0.005"))

    proc = psutil.Process()
    pid = proc.pid
    # Namespace by hostname so multiple nodes writing to a SHARED trace dir (the
    # multi-node verification: RAY_MEM_TRACE_DIR on Anyscale /mnt/cluster_storage)
    # never collide on pid. Single-node is unaffected (one hostname). The reader's
    # `uss_*.csv` glob still matches, and the host prefix lets us group by node.
    host = socket.gethostname()
    path = os.path.join(trace_dir, f"uss_{host}_{pid}.csv")
    fh = open(path, "w", buffering=1)  # line-buffered
    fh.write("epoch_s,uss_bytes,rss_bytes\n")

    def _run():
        while True:
            try:
                mi = proc.memory_full_info()
                fh.write(f"{time.time():.6f},{mi.uss},{mi.rss}\n")
            except Exception:
                break
            time.sleep(interval)

    threading.Thread(target=_run, name="uss-sampler", daemon=True).start()

    _patch_task_windows(trace_dir, host, pid)

    @atexit.register
    def _flush():
        try:
            fh.flush()
            fh.close()
        except Exception:
            pass
