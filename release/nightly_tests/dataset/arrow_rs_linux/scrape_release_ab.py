#!/usr/bin/env python3
"""Compare two release builds (arrow-rs vs PyArrow) from their Buildkite artifacts.

Why this exists
---------------
The A/B comparison we have been arguing from is ``max_uss_per_task``, which is
sampled by ``MemoryProfiler`` inside each read task and reports *whole-process*
USS -- so it drifts toward a worker's lifetime high-water whenever workers are
reused. That makes it a poor cross-environment metric, and it is the only
memory number the release ``result.json`` carries.

Every release job also uploads ``metrics.json.gz``: a per-node Prometheus scrape
(15s resolution) of ``memory_usage``, ``cpu_utilization``, ``spilled_bytes`` and
``network_receive_speed``. That is the operating system's view of each machine,
owes nothing to Ray's instrumentation, and is therefore the independent check.
When the two agree, the regression is real. When they disagree, the per-task
metric is the one to distrust.

The progress lines in the job log are NOT a task count -- ``logging_progress``
reports ``num_output_rows_total()`` (rows). It coincides with the task count
only for write-terminated operators, which emit one metadata row per task.

Usage
-----
    python scrape_release_ab.py                       # both arms, all paired tests
    python scrape_release_ab.py --only write_parquet,iter_batches_numpy
    python scrape_release_ab.py --json out/release_ab.json

Artifacts are cached in the bk-api log dir, so re-runs are free.
"""

import argparse
import gzip
import json
import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional

BK = os.path.expanduser("~/.local/bin/bk-api")
CACHE = os.path.expanduser("~/.cache/ray-buildkite-logs")
PIPELINE = "release"
GiB = 1024**3

# The two builds behind the multi-node A/B. Both ran the same release suite;
# 103100 carried the arrow-rs reader, 103101 the untouched PyArrow baseline.
RS_BUILD = "103100"
PA_BUILD = "103101"

# Buildkite decorates job names with retry/parallelism suffixes.
NAME_SUFFIX = re.compile(r"\s*\(None\)\s*\(\d+\)\s*$")


def bk(*args: str) -> str:
    """Run bk-api and return stdout. The token never leaves the Keychain."""
    proc = subprocess.run([BK, *args], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"bk-api {' '.join(args)}: {proc.stderr.strip()}")
    return proc.stdout


def jobs_for(build: str) -> Dict[str, str]:
    """Map clean job name -> job id, skipping image-build and group steps."""
    payload = json.loads(
        bk("get", f"organizations/ray-project/pipelines/{PIPELINE}/builds/{build}")
    )
    out = {}
    for job in payload.get("jobs", []):
        name = job.get("name") or ""
        if not name or "build custom" in name or job.get("type") != "script":
            continue
        clean = NAME_SUFFIX.sub("", name).strip()
        # A retried job appears twice; the later entry wins.
        if clean:
            out[clean] = job.get("id")
    return out


def artifact(build: str, job: str, wanted: str) -> Optional[str]:
    """Download the named artifact once; return its cached path."""
    listing = bk("arts", PIPELINE, build, job)
    art_id = None
    for line in listing.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2].endswith(wanted):
            art_id = parts[0]
            break
    if art_id is None:
        return None
    path = os.path.join(CACHE, f"{PIPELINE}-{build}-{job}-{art_id}.bin")
    if not os.path.exists(path):
        bk("art", PIPELINE, build, job, art_id)
    return path if os.path.exists(path) else None


def node_memory(path: str) -> Dict[str, Any]:
    """Per-worker peak memory from the cluster scrape.

    The head node is identified by being the lone idle machine: it is the only
    series that spans the full window while sitting near-idle. Rather than guess
    from labels (``node_type`` is not always populated) we drop the single
    lowest-peak series when it is a clear outlier, which is what a head node is
    on a read benchmark.
    """
    with gzip.open(path) as handle:
        data = json.load(handle)

    peaks: List[float] = []
    finals: List[float] = []
    firsts: List[float] = []
    for series in data.get("memory_usage", []):
        values = [float(v) / GiB for _, v in series["values"]]
        if values:
            peaks.append(max(values))
            finals.append(values[-1])
            firsts.append(values[0])
    if not peaks:
        return {}

    order = sorted(range(len(peaks)), key=lambda i: peaks[i])
    # Drop the head node: lowest peak, and less than half the next-lowest.
    if len(peaks) > 2 and peaks[order[0]] < 0.5 * peaks[order[1]]:
        order = order[1:]
    workers = [(peaks[i], finals[i], firsts[i]) for i in order]

    def top(key: str) -> Optional[float]:
        vals = [
            max(float(v) for _, v in s["values"])
            for s in data.get(key, [])
            if s["values"]
        ]
        return max(vals) if vals else None

    return {
        "workers": len(workers),
        "mem_peak_mean_gib": sum(p for p, _, _ in workers) / len(workers),
        "mem_peak_max_gib": max(p for p, _, _ in workers),
        "mem_peak_min_gib": min(p for p, _, _ in workers),
        # Peak minus the node's own first sample. Both arms start from the same
        # idle baseline (worker pool + preallocated object store), so the RISE is
        # what the read actually cost; the absolute peak dilutes it with a large
        # constant and understates the difference between the two readers.
        # NOTE: the scrape is forward-filled -- values repeat in runs of four, so
        # the true resolution is 60s and a short test yields only a few samples.
        "mem_rise_mean_gib": sum(p - s for p, _, s in workers) / len(workers),
        # Workers whose last sample IS their peak never gave the memory back.
        # A high count is the retention signature; PyArrow typically releases.
        "still_climbing": sum(1 for p, f, _ in workers if f >= p - 1e-9),
        "spilled_peak_gib": (top("spilled_bytes") or 0) / GiB,
        "net_recv_peak_mbs": (top("network_receive_speed") or 0) / 1e6,
    }


def result_metrics(path: str) -> Dict[str, Any]:
    with open(path) as handle:
        payload = json.load(handle)
    main = (payload.get("results") or {}).get("main") or {}
    keep = (
        "time",
        "object_store_memory_used_peak_gb",
        "object_store_spilled_total_gb",
        "read_wall_time_s",
        "read_avg_max_uss_per_task_bytes",
        "read_max_uss_per_task_bytes",
        "read_output_size_bytes",
        # Present only on runs carrying the decomposition instrumentation.
        "read_task_rows_count",
        "read_node_count_count",
        "read_uss_num_samples",
    )
    out = {k: main.get(k) for k in keep}
    out["runtime_s"] = payload.get("runtime")
    return out


def collect(build: str, job: str) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    res = artifact(build, job, "result.json")
    if res:
        row.update(result_metrics(res))
    met = artifact(build, job, "metrics.json.gz")
    if met:
        row.update(node_memory(met))
    return row


def ratio(rs: Optional[float], pa: Optional[float]) -> str:
    if not rs or not pa:
        return "-"
    return f"{rs / pa:.2f}x"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rs-build", default=RS_BUILD)
    parser.add_argument("--pa-build", default=PA_BUILD)
    parser.add_argument(
        "--only", help="comma-separated substrings to filter test names"
    )
    parser.add_argument("--json", help="also write the full per-test rows here")
    args = parser.parse_args()

    rs_jobs, pa_jobs = jobs_for(args.rs_build), jobs_for(args.pa_build)
    names = sorted(set(rs_jobs) & set(pa_jobs))
    if args.only:
        wanted = [w.strip() for w in args.only.split(",") if w.strip()]
        names = [n for n in names if any(w in n for w in wanted)]
    if not names:
        print("no paired jobs matched", file=sys.stderr)
        return 1

    print(
        f"{len(names)} paired tests ({args.rs_build} arrow-rs vs {args.pa_build} pyarrow)\n"
    )
    head = (
        f"{'test':<44}{'node mem GiB R/P':>20}{'ratio':>8}"
        f"{'uss avg R/P':>18}{'ratio':>8}{'climb':>8}{'wall':>8}"
    )
    print(head)
    print("-" * len(head))

    everything = {}
    for name in names:
        try:
            rs, pa = collect(args.rs_build, rs_jobs[name]), collect(
                args.pa_build, pa_jobs[name]
            )
        except RuntimeError as exc:
            print(f"{name:<44}  !! {exc}")
            continue
        everything[name] = {"arrow_rs": rs, "pyarrow": pa}

        rmem, pmem = rs.get("mem_peak_mean_gib"), pa.get("mem_peak_mean_gib")
        russ = (rs.get("read_avg_max_uss_per_task_bytes") or 0) / GiB or None
        puss = (pa.get("read_avg_max_uss_per_task_bytes") or 0) / GiB or None
        mem_cell = f"{rmem:.1f}/{pmem:.1f}" if rmem and pmem else "-"
        uss_cell = f"{russ:.2f}/{puss:.2f}" if russ and puss else "-"
        climb = f"{rs.get('still_climbing','-')}/{pa.get('still_climbing','-')}"
        print(
            f"{name:<44}{mem_cell:>20}{ratio(rmem, pmem):>8}"
            f"{uss_cell:>18}{ratio(russ, puss):>8}{climb:>8}"
            f"{ratio(rs.get('time'), pa.get('time')):>8}"
        )

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as handle:
            json.dump(everything, handle, indent=2)
        print(f"\nwrote {args.json}")

    print(
        "\nnode mem = mean over worker nodes of each node's peak memory_usage "
        "(head node excluded).\nclimb = workers whose final sample is their peak "
        "(memory never returned to the OS)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
