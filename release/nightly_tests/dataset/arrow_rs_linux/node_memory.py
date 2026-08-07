#!/usr/bin/env python3
"""Record whole-machine memory the way the release cluster records it.

Why this exists
---------------
Every number we had argued from was ``max_uss_per_task``: Ray samples it inside
each read task via ``MemoryProfiler``, which reads the *whole worker process's*
private memory. Ray reuses worker processes, so memory left behind by an earlier
task is charged to a later one, and the metric drifts toward a worker's lifetime
high-water. It is not wrong, but it answers "how big did this worker process ever
get", not "how much memory did the machine need".

The release cluster independently records the second quantity -- each node's
``memory_usage``, scraped by Prometheus -- and it is uploaded with every release
job as ``metrics.json.gz``. Comparing the two across builds 103100/103101 found
that they agree on ``write_parquet`` (1.70x node peak vs 1.68x per-task USS: a
real regression) and disagree on ``read_from_uris`` (1.00x node peak vs 1.20x
per-task USS: the per-task metric alone).

Nothing on a single box recorded the machine-level number, so no local run could
be compared to release on equal terms. This does that.

Two things worth knowing about the release scrape, both reproduced here:

* Its samples are forward-filled in runs of four, so the true resolution is 60s,
  not the 15s the timestamps suggest. ``report`` therefore prints a "coarse"
  view -- this trace resampled to 60s -- next to the real one. The gap between
  them is how much of a spike the release measurement could not have seen.
* ``memory_usage`` is machine memory in use, i.e. ``MemTotal - MemAvailable``.
  That includes Ray's object store (shared pages), which per-task USS excludes
  by construction. When the two metrics disagree, that difference is the first
  thing to suspect.

Usage
-----
    # started/stopped automatically by run_arm in common.sh
    python node_memory.py record --out out/mem/exp3_arrow_rs.jsonl

    # compare arms afterwards
    python node_memory.py report out/mem/*.jsonl --baseline out/mem/exp3_pyarrow.jsonl
"""

import argparse
import json
import os
import signal
import sys
import time
from typing import Dict, List, Optional

MiB = 1024 * 1024
GiB = 1024**3
# The release scrape is forward-filled to this period; matching it is what makes
# the "coarse" column an honest simulation of what release could have seen.
RELEASE_PERIOD_S = 60.0


def meminfo() -> Dict[str, int]:
    """MemTotal/MemAvailable in bytes. Linux only -- this is a /proc reader."""
    out = {}
    with open("/proc/meminfo") as handle:
        for line in handle:
            key, _, rest = line.partition(":")
            if key in ("MemTotal", "MemAvailable"):
                out[key] = int(rest.split()[0]) * 1024
                if len(out) == 2:
                    break
    return out


def ray_worker_rss() -> Dict[str, float]:
    """Total RSS of the ray:: worker processes, and how many there are.

    This is the bridge between the two metrics: machine memory minus worker RSS
    is roughly the object store plus the raylet, so a regression that shows up in
    machine memory but NOT here is not the decoder's private allocation.
    """
    try:
        import psutil
    except ImportError:
        return {}
    total, count = 0, 0
    for proc in psutil.process_iter(["cmdline", "memory_info"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            if cmdline and cmdline[0].startswith("ray::"):
                total += proc.info["memory_info"].rss
                count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return {"worker_rss": total, "workers": count}


def cmd_record(args: argparse.Namespace) -> int:
    if not os.path.exists("/proc/meminfo"):
        print("node_memory: /proc/meminfo missing -- Linux only", file=sys.stderr)
        return 1
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    stop = {"now": False}
    # Terminate cleanly so the trailing sample and a flushed file always exist;
    # run_arm stops us with SIGTERM the moment the benchmark returns.
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.__setitem__("now", True))

    started = time.time()
    with open(args.out, "w", buffering=1) as handle:
        info = meminfo()
        handle.write(
            json.dumps({"type": "header", "mem_total": info["MemTotal"], "t0": started})
            + "\n"
        )
        while not stop["now"]:
            info = meminfo()
            row = {
                "t": round(time.time() - started, 3),
                "used": info["MemTotal"] - info["MemAvailable"],
            }
            row.update(ray_worker_rss())
            handle.write(json.dumps(row) + "\n")
            # Sleep in slices so a SIGTERM is honoured promptly at any interval.
            deadline = time.time() + args.interval
            while time.time() < deadline and not stop["now"]:
                time.sleep(min(0.1, max(0.0, deadline - time.time())))
    return 0


def load(path: str) -> Dict[str, object]:
    header, rows = {}, []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            (header.update(obj) if obj.get("type") == "header" else rows.append(obj))
    return {"header": header, "rows": rows}


def coarsen(rows: List[dict], period: float) -> List[dict]:
    """Keep one sample per period -- what a 60s scrape would have captured."""
    kept, next_at = [], 0.0
    for row in rows:
        if row["t"] >= next_at:
            kept.append(row)
            next_at = row["t"] + period
    return kept


def summarize(path: str) -> Optional[Dict[str, object]]:
    trace = load(path)
    rows = trace["rows"]
    if not rows:
        return None
    used = [r["used"] for r in rows]
    coarse = [r["used"] for r in coarsen(rows, RELEASE_PERIOD_S)]
    workers = [r.get("worker_rss", 0) for r in rows]
    return {
        "path": path,
        "samples": len(rows),
        "duration_s": rows[-1]["t"],
        "mem_total": trace["header"].get("mem_total"),
        "first": used[0],
        "peak": max(used),
        "final": used[-1],
        # The quantity that matters: memory the run actually added. Both arms
        # start from the same idle baseline, so the absolute peak just dilutes
        # the difference with a large constant.
        "rise": max(used) - used[0],
        "coarse_peak": max(coarse) if coarse else None,
        "coarse_rise": (max(coarse) - coarse[0]) if coarse else None,
        "coarse_samples": len(coarse),
        "worker_rss_peak": max(workers) if workers else 0,
        "still_climbing": used[-1] >= max(used) - MiB,
    }


def cmd_report(args: argparse.Namespace) -> int:
    rows = [s for s in (summarize(p) for p in args.traces) if s]
    if not rows:
        print("no usable traces", file=sys.stderr)
        return 1

    base = None
    if args.baseline:
        base = next(
            (
                r
                for r in rows
                if os.path.abspath(r["path"]) == os.path.abspath(args.baseline)
            ),
            None,
        )
        if base is None:
            print(f"baseline {args.baseline} not among the traces", file=sys.stderr)
            return 1

    head = (
        f"{'trace':<34}{'n':>5}{'secs':>7}{'peak':>9}{'rise':>9}{'vs base':>9}"
        f"{'wkrRSS':>9}{'60s rise':>10}{'blind':>8}{'climb':>7}"
    )
    print(head)
    print("-" * len(head))
    for row in rows:
        rise_ratio = (
            f"{row['rise'] / base['rise']:.2f}x"
            if base and base["rise"] > 0 and row is not base
            else ("baseline" if row is base else "-")
        )
        # How much of the real rise a 60s scrape would have missed.
        blind = (
            f"{100 * (1 - row['coarse_rise'] / row['rise']):.0f}%"
            if row["rise"] > 0 and row["coarse_rise"] is not None
            else "-"
        )
        print(
            f"{os.path.basename(row['path']):<34}{row['samples']:>5}{row['duration_s']:>7.0f}"
            f"{row['peak'] / GiB:>8.2f}G{row['rise'] / GiB:>8.2f}G{rise_ratio:>9}"
            f"{row['worker_rss_peak'] / GiB:>8.2f}G"
            f"{(row['coarse_rise'] or 0) / GiB:>9.2f}G{blind:>8}"
            f"{('yes' if row['still_climbing'] else 'no'):>7}"
        )

    print(
        "\nrise    = peak machine memory minus this run's own starting point"
        "\nwkrRSS  = peak total RSS of ray:: workers (machine rise minus this is"
        " object store + raylet)"
        "\n60s rise= the same rise seen through a 60s scrape, as release sees it"
        "\nblind   = fraction of the real rise a 60s scrape would have missed"
        "\nclimb   = memory was still at its peak when the run ended"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    rec = sub.add_parser("record", help="sample until SIGTERM")
    rec.add_argument("--out", required=True)
    rec.add_argument("--interval", type=float, default=1.0)
    rec.set_defaults(func=cmd_record)

    rep = sub.add_parser("report", help="compare recorded traces")
    rep.add_argument("traces", nargs="+")
    rep.add_argument("--baseline", help="trace to express the others against")
    rep.set_defaults(func=cmd_report)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
