"""worker_process_setup_hook: sample THIS worker's private heap (USS) over time.

Runs once per Ray worker process at startup. Samples
``psutil.Process().memory_full_info().uss`` — the SAME quantity Ray's own
MemoryProfiler records (util.py:_estimate_uss) but never uses for scheduling —
at a fine interval, writing ``uss_<pid>.csv`` into ``RAY_MEM_TRACE_DIR``.

USS = unique/private set = pages resident ONLY in this process = the decode
working set in the worker's malloc/Rust heap. It EXCLUDES shared plasma
object-store segments and shared libraries, so it is exactly the per-worker
memory that is invisible to Ray's object-store-based admission control.

Line-buffered writes so a killed/reused worker still leaves its samples.
"""
import atexit
import os
import socket
import threading
import time

import psutil

_started = False


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

    @atexit.register
    def _flush():
        try:
            fh.flush()
            fh.close()
        except Exception:
            pass
