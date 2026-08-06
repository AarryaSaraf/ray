#!/usr/bin/env python3
"""Print every individual run behind an ab_test.py report.

The report collapses each arm to a median, which is the right thing to compare
but the wrong thing to debug: it cannot show you *which* repeat was the outlier,
and a within-arm spread is the difference between "arrow-rs is heavier" and
"arrow-rs is sometimes heavier", which are different bugs.

Usage:
    python show_runs.py                      # out/ab/ab_raw.json
    python show_runs.py --raw path/to.json
"""

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MiB = 1024 * 1024

# (header, json key, unit). Kept deliberately short -- this is meant to be
# pasted into a message, so it has to fit in a terminal width.
COLUMNS = [
    ("uss avg", "read_avg_max_uss_per_task_bytes", "mib"),
    ("uss max", "read_max_uss_per_task_bytes", "mib"),
    ("uss min", "read_uss_min_bytes", "mib"),
    ("rss max", "read_max_rss_per_task_bytes", "mib"),
    ("read s", "read_wall_time_s", "sec"),
    ("total s", "time", "sec"),
    ("out MB", "read_output_size_bytes", "mb"),
]


def fmt(value, unit):
    if not isinstance(value, (int, float)):
        return "-"
    if unit == "mib":
        return f"{value / MiB:,.0f}"
    if unit == "mb":
        return f"{value / 1e6:,.0f}"
    return f"{value:,.1f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--raw", default=os.path.join(HERE, "out", "ab", "ab_raw.json"))
    args = parser.parse_args()

    with open(args.raw) as handle:
        everything = json.load(handle)

    for case, arms in everything.items():
        print(f"\n=== {case}")
        head = f"{'arm':<16}{'rep':>4}" + "".join(f"{h:>9}" for h, _, _ in COLUMNS)
        print(head)
        print("-" * len(head))
        for arm, runs in arms.items():
            for rep, run in enumerate(runs):
                cells = "".join(fmt(run.get(k), u).rjust(9) for _, k, u in COLUMNS)
                print(f"{arm:<16}{rep:>4}{cells}")
    print("\n(uss/rss in MiB per read task; out MB = decoded output bytes)")


if __name__ == "__main__":
    main()
