#!/usr/bin/env python3
"""Paired A/B of the arrow-rs reader against PyArrow, on the release cases.

The problem with every measurement we have taken so far: one run of each arm,
one ratio, no idea what ratio a *null* change would have produced. The release
suite's own controls (the `map_*` cases, which force `use_datasource_v2=False`
and therefore run identical code in both arms) disagreed by 2.5x -- so on that
harness a 1.10x "regression" is indistinguishable from nothing at all.

This runner fixes that with three things:

1. **Repeats.** Each arm runs N times; we compare medians, not single draws.
2. **Interleaving (ABBA).** Repeat i runs the arms in order [P, R], repeat i+1
   in order [R, P]. Any monotone drift over the session -- page cache, S3
   throttling, a noisy neighbour -- lands on both arms equally instead of on
   whichever ran second.
3. **An A/A control.** A third arm that is PyArrow *again*, run and analysed
   exactly like the treatment. Its ratio against the baseline is the harness's
   own noise, measured on the same machine on the same day. A treatment ratio
   inside that band is not a finding, however far from 1.00 it looks.

Usage::

    python ab_test.py --cases iter_batches_numpy --repeats 3
    python ab_test.py --cases all --repeats 5 --out out/ab
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

MiB = 1024 * 1024
HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.dirname(HERE)

S3_ROOT = os.environ.get(
    "S3_ROOT", "s3://arrowrs-bench-21f6c795/arrow_rs_probe"
).rstrip("/")

# The release cases, transcribed from release/release_data_tests.yaml, with the
# single-node scale-down noted. `expected` is what the multi-node A/B (builds
# 103100 vs 103101) measured, so the output can say "release said X, we see Y".
CASES: Dict[str, Dict[str, Any]] = {
    "iter_batches_numpy": {
        "args": [
            f"{S3_ROOT}/tpch/sf10/lineitem",
            "--format",
            "parquet",
            "--iter-batches",
            "numpy",
        ],
        "release": "sf10 lineitem, --iter-batches numpy (verbatim, no scale-down)",
        "expected": {"read_max_uss_per_task_bytes": 1.01},
    },
    "write_parquet": {
        "args": [f"{S3_ROOT}/tpch/sf10/lineitem", "--format", "parquet", "--write"],
        "release": "sf1000 lineitem --write; we run sf10 (100x scale-down)",
        "expected": {
            "read_max_uss_per_task_bytes": 2.28,
            "read_avg_max_uss_per_task_bytes": 1.68,
            "time": 1.00,
        },
    },
    "iter_bundles": {
        "args": [
            f"{S3_ROOT}/tpch/sf10/lineitem",
            "--format",
            "parquet",
            "--iter-bundles",
        ],
        "release": (
            "stands in for read_parquet, whose imagenet input is in the "
            "unreadable -internal- bucket; same consume mode, different data"
        ),
        "expected": {},
    },
}

# (label, json key, formatter). Memory first: the migration is memory-first and
# wall time only has to stay at parity.
METRICS = [
    ("read USS avg/task", "read_avg_max_uss_per_task_bytes", "mib"),
    ("read USS max/task", "read_max_uss_per_task_bytes", "mib"),
    ("read USS p50", "read_uss_p50_bytes", "mib"),
    ("read USS p90", "read_uss_p90_bytes", "mib"),
    ("read USS p99", "read_uss_p99_bytes", "mib"),
    ("read USS samples", "read_uss_num_samples", "int"),
    ("read RSS max/task", "read_max_rss_per_task_bytes", "mib"),
    ("read wall", "read_wall_time_s", "sec"),
    ("total wall", "time", "sec"),
    ("object peak GB", "object_store_memory_used_peak_gb", "num"),
]

# arm name -> whether the arrow-rs reader is on. "pyarrow_control" is the A/A:
# byte-identical to the baseline, so any ratio it shows is pure noise.
ARMS = {"pyarrow": False, "arrow_rs": True, "pyarrow_control": False}
BASELINE = "pyarrow"
CONTROL = "pyarrow_control"


def prepare_env() -> None:
    """Make this process's environment safe for a private local Ray.

    common.sh does the same thing for the exp*.sh scripts, but this runner is
    invoked directly, so it cannot rely on having been sourced. Every item here
    is a workspace trap that produces a confusing failure rather than an
    obvious one.
    """
    # A platform-injected runtime-env hook (`cgroup_runtime_plugin` on Anyscale)
    # is imported by ray.init() and is not in this venv: ModuleNotFoundError
    # before a single row is read.
    for var in ("RAY_RUNTIME_ENV_HOOK", "RAY_RUNTIME_ENV_PLUGINS"):
        os.environ.pop(var, None)
    # Never attach to the workspace's managed cluster: different Ray, different
    # reader, meaningless comparison.
    os.environ["RAY_ADDRESS"] = "local"
    # Some 2026-07 nightlies SIGSEGV in the task-event aggregator flush.
    os.environ.setdefault("RAY_task_events_report_interval_ms", "0")
    os.environ.setdefault("RAY_DATA_USE_DATASOURCE_V2", "1")


def run_one(case: str, arm: str, rep: int, out_dir: str) -> Optional[Dict[str, Any]]:
    """Run one arm once and return its parsed result dict (None if it failed)."""
    tag = f"{case}_{arm}_r{rep}"
    result_path = os.path.join(out_dir, f"{tag}.json")
    prof_dir = os.path.join(out_dir, "prof", tag)
    os.makedirs(prof_dir, exist_ok=True)

    env = dict(os.environ)
    env["RAY_DATA_USE_ARROW_RS_PARQUET_READER"] = "1" if ARMS[arm] else "0"
    env["RAY_DATA_ARROW_RS_PROFILE"] = "1"
    env["RAY_DATA_ARROW_RS_PROFILE_DIR"] = prof_dir
    env["TEST_OUTPUT_JSON"] = result_path

    argv = [sys.executable, "read_and_consume_benchmark.py"] + CASES[case]["args"]
    started = time.time()
    print(f"  [{tag}] running...", flush=True)
    with open(os.path.join(out_dir, f"{tag}.log"), "w") as log:
        proc = subprocess.run(argv, cwd=DATASET_DIR, env=env, stdout=log, stderr=log)
    elapsed = time.time() - started

    if proc.returncode != 0:
        print(f"  [{tag}] FAILED rc={proc.returncode} after {elapsed:.0f}s", flush=True)
        # Show why, here: a silent failure repeated 9 times is how you lose an
        # hour to one missing environment variable.
        try:
            with open(os.path.join(out_dir, f"{tag}.log")) as log:
                tail = log.read().strip().splitlines()[-6:]
            for line in tail:
                print(f"      | {line}", flush=True)
        except OSError:
            pass
        return None
    try:
        with open(result_path) as handle:
            raw = json.load(handle)
    except (OSError, ValueError) as exc:
        print(f"  [{tag}] no parsable result ({exc})", flush=True)
        return None
    flat: Dict[str, Any] = {}
    for value in raw.values():
        if isinstance(value, dict):
            flat.update(value)
    print(f"  [{tag}] ok in {elapsed:.0f}s", flush=True)
    return flat or raw


def med(values: List[Any]) -> Optional[float]:
    nums = [v for v in values if isinstance(v, (int, float))]
    return statistics.median(nums) if nums else None


def spread(values: List[Any]) -> Optional[float]:
    """max/min across repeats -- the within-arm variation, in ratio units."""
    nums = [v for v in values if isinstance(v, (int, float)) and v]
    return (max(nums) / min(nums)) if len(nums) > 1 else None


def fmt(value: Optional[float], kind: str) -> str:
    if value is None:
        return "-"
    if kind == "mib":
        return f"{value / MiB:,.0f}Mi"
    if kind == "sec":
        return f"{value:,.1f}s"
    if kind == "int":
        return f"{int(value)}"
    return f"{value:,.2f}"


def verdict(treat: Optional[float], control: Optional[float]) -> str:
    """Is the treatment ratio outside the band the A/A control drew?

    The control ratio should be 1.00 and is not; whatever it is, that is how
    much this harness moves for no reason. A treatment effect must clear it in
    the same direction before it means anything.
    """
    if treat is None:
        return ""
    if control is None:
        return "no control"
    noise = abs(control - 1.0)
    delta = abs(treat - 1.0)
    if delta <= noise:
        return "NOISE"
    if delta <= 2 * noise:
        return "weak"
    return "SIGNAL"


def report(case: str, results: Dict[str, List[Dict[str, Any]]]) -> None:
    base_runs = results[BASELINE]
    print(f"\n{'=' * 100}")
    print(f"case: {case}   ({CASES[case]['release']})")
    print(f"repeats: {', '.join(f'{a}={len(r)}' for a, r in results.items())}")
    print(f"{'=' * 100}")
    header = (
        f"{'metric':<20}{'pyarrow':>11}{'arrow_rs':>11}{'R/P':>8}"
        f"{'A/A':>8}{'verdict':>9}{'P spread':>10}{'R spread':>10}"
    )
    print(header)
    print("-" * len(header))

    treat_runs = results["arrow_rs"]
    ctrl_runs = results.get(CONTROL, [])

    def as_ratio(value: Optional[float]) -> str:
        return f"{value:.2f}x" if value else "-"

    for label, key, kind in METRICS:
        base_vals = [r.get(key) for r in base_runs]
        treat_vals = [r.get(key) for r in treat_runs]
        base, treat, ctrl = (
            med(base_vals),
            med(treat_vals),
            med([r.get(key) for r in ctrl_runs]),
        )
        r_over_p = (treat / base) if (base and treat) else None
        a_over_a = (ctrl / base) if (base and ctrl) else None
        v = verdict(r_over_p, a_over_a) if kind != "int" else ""
        print(
            f"{label:<20}{fmt(base, kind):>11}{fmt(treat, kind):>11}"
            f"{as_ratio(r_over_p):>8}{as_ratio(a_over_a):>8}{v:>9}"
            f"{as_ratio(spread(base_vals)):>10}{as_ratio(spread(treat_vals)):>10}"
        )

    expected = CASES[case].get("expected") or {}
    if expected:
        print("\n  vs the multi-node release A/B (103100 vs 103101):")
        for key, release_ratio in expected.items():
            label = next((lab for lab, k, _ in METRICS if k == key), key)
            base = med([r.get(key) for r in base_runs])
            treat = med([r.get(key) for r in results["arrow_rs"]])
            here = (treat / base) if (base and treat) else None
            here_s = f"{here:.2f}x" if here else "-"
            agree = (
                "reproduced"
                if here and abs(here - release_ratio) < 0.15
                else "NOT reproduced"
            )
            print(
                f"    {label:<22} release {release_ratio:.2f}x   here {here_s:>7}   {agree}"
            )


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cases", default="iter_batches_numpy")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", default=os.path.join(HERE, "out", "ab"))
    parser.add_argument(
        "--no-control",
        action="store_true",
        help="Skip the A/A arm (faster, but then no noise floor -- not advised)",
    )
    args = parser.parse_args(argv)

    names = list(CASES) if args.cases == "all" else args.cases.split(",")
    unknown = [n for n in names if n not in CASES]
    if unknown:
        raise SystemExit(f"unknown case(s) {unknown}; known: {list(CASES)}")
    arms = [a for a in ARMS if not (args.no_control and a == CONTROL)]
    os.makedirs(args.out, exist_ok=True)
    prepare_env()
    everything: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    for case in names:
        print(f"\n### {case}: {args.repeats} repeats x {len(arms)} arms")
        results: Dict[str, List[Dict[str, Any]]] = {a: [] for a in arms}
        # One warm-up run, discarded: the first read of the session pays for
        # cold S3 connections and an unpopulated Ray worker pool, and it would
        # otherwise land entirely on whichever arm happens to go first.
        if run_one(case, BASELINE, -1, args.out) is None:
            print(f"!!! {case}: warm-up failed, skipping the case (see log above)")
            continue
        for rep in range(args.repeats):
            # ABBA: reverse the arm order on odd repeats so a monotone drift
            # over the session cancels instead of favouring one arm.
            order = arms if rep % 2 == 0 else list(reversed(arms))
            for arm in order:
                res = run_one(case, arm, rep, args.out)
                if res:
                    results[arm].append(res)
        everything[case] = results
        if not results[BASELINE] or not results["arrow_rs"]:
            print(f"!!! {case}: an arm produced no results, skipping report")
            continue
        report(case, results)

    summary = os.path.join(args.out, "ab_raw.json")
    with open(summary, "w") as handle:
        json.dump(everything, handle, indent=2, default=str)
    print(f"\nper-run JSON + logs in {args.out} (index: {summary})")


if __name__ == "__main__":
    main()
