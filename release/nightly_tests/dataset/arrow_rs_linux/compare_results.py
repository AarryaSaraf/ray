#!/usr/bin/env python
"""Print an A/B table from `Benchmark.write_result()` JSON files.

Each input is one arm. The last file, or ``--baseline``, is the denominator; a
ratio above 1.00 means that arm is *worse* (slower / heavier), matching the
convention used throughout the arrow-rs comparison docs.

Usage::

    compare_results.py out/exp1_..._arrow_rs.json out/exp1_..._pyarrow.json
    compare_results.py out/exp3_*_arrow_rs.json --baseline out/exp3_write_pyarrow.json
"""

import argparse
import glob
import json
import os
from typing import Any, Dict, Optional

MiB = 1024 * 1024

# (label, key, formatter). Ordered so the memory question comes first: the
# migration is memory-first, and wall time only has to stay at parity.
ROWS = [
    ("read USS avg/task", "read_avg_max_uss_per_task_bytes", "mib"),
    ("read USS max/task", "read_max_uss_per_task_bytes", "mib"),
    ("read USS p90", "read_uss_p90_bytes", "mib"),
    ("read USS samples", "read_uss_num_samples", "int"),
    ("read RSS avg/task", "read_avg_max_rss_per_task_bytes", "mib"),
    ("read RSS max/task", "read_max_rss_per_task_bytes", "mib"),
    ("read wall", "read_wall_time_s", "sec"),
    ("read out bytes", "read_output_size_bytes", "mib"),
    ("read B/row", "read_decoded_bytes_per_row", "num"),
    ("total wall", "time", "sec"),
    ("object peak GB", "object_store_memory_used_peak_gb", "num"),
    ("spilled GB", "object_store_spilled_total_gb", "num"),
]


def load(path: str) -> Dict[str, Any]:
    """Flatten ``{case_name: {metrics}}`` into one dict (drivers write one case)."""
    with open(path) as handle:
        raw = json.load(handle)
    flat: Dict[str, Any] = {}
    for value in raw.values():
        if isinstance(value, dict):
            flat.update(value)
    return flat or raw


def fmt(value: Any, kind: str) -> str:
    if value is None:
        return "-"
    if kind == "mib":
        return f"{value / MiB:,.0f}Mi"
    if kind == "sec":
        return f"{value:,.1f}s"
    if kind == "int":
        return f"{int(value)}"
    return f"{value:,.2f}"


def ratio(arm: Any, base: Any) -> str:
    if not arm or not base:
        return "-"
    return f"{arm / base:.2f}x"


def profile_digest(prof_dir: str) -> Dict[str, Any]:
    """Fold the reader's JSONL profiling into the few scalars that decide H1-H3."""
    plans, rgs, local, fallbacks = [], [], [], 0
    for path in glob.glob(os.path.join(prof_dir, "*.jsonl")):
        with open(path) as handle:
            for line in handle:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                kind = rec.get("kind")
                if kind == "s3_plan":
                    plans.append(rec)
                elif kind == "s3_rg":
                    rgs.append(rec)
                elif kind == "local_rg":
                    local.append(rec)
                elif kind == "fallback":
                    fallbacks += 1

    def total(recs, key):
        return sum(r.get(key) or 0 for r in recs)

    def biggest(recs, key):
        vals = [r.get(key) or 0 for r in recs]
        return max(vals) if vals else 0

    return {
        "fallbacks": fallbacks,
        "s3 plans": len(plans),
        "column-group row groups (H1)": total(plans, "col_group_rgs"),
        "oversized units (H2)": total(plans, "oversized_units"),
        "max unit KiB": biggest(plans, "unit_kib_max"),
        "max retained bytes (H1)": biggest(rgs, "retained_bytes"),
        "fetch wait s": round(total(rgs, "fetch_wait_s"), 2),
        "decode s": round(total(rgs, "decode_s"), 2),
        "floored row groups (H3)": sum(1 for r in local if r.get("floored")),
        "max eff batch bytes": biggest(local, "eff_batch_bytes"),
    }


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+")
    parser.add_argument("--baseline", default=None, help="Denominator (default: last)")
    parser.add_argument("--prof", default=None, help="Profiling dir to digest")
    args = parser.parse_args(argv)

    base_path = args.baseline or args.files[-1]
    arms = [f for f in args.files if f != base_path]
    base = load(base_path)
    loaded = [(os.path.basename(f).replace(".json", ""), load(f)) for f in arms]

    name_w = max([22] + [len(n) for n, _ in loaded])
    header = f"{'metric':<22}{'baseline':>12}"
    for name, _ in loaded:
        header += f"{name[-name_w:]:>{name_w + 2}}{'ratio':>8}"
    print(f"\nbaseline = {os.path.basename(base_path)}")
    print(header)
    print("-" * len(header))
    for label, key, kind in ROWS:
        line = f"{label:<22}{fmt(base.get(key), kind):>12}"
        for _, arm in loaded:
            line += f"{fmt(arm.get(key), kind):>{name_w + 2}}"
            line += f"{ratio(arm.get(key), base.get(key)):>8}"
        print(line)

    if base.get("read_uss_num_samples") == 0:
        print(
            "\nWARNING: zero USS samples -- per-task memory was not collected "
            "(macOS, or the operator never ran). Memory rows above are blank, "
            "not flat."
        )
    if args.prof and os.path.isdir(args.prof):
        print(f"\nprofiling ({args.prof}):")
        for key, value in profile_digest(args.prof).items():
            print(f"  {key:<32}{value}")


if __name__ == "__main__":
    main()
