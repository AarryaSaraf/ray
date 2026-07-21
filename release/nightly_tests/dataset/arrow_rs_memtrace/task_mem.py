"""THE memory graph of record: per-task absolute USS over time vs Ray's own
per-task expectation, one figure per benchmark config, both readers overlaid.

What each figure shows (and the whole metric — no scalars, no baselines):

  * one line per READ TASK: the executing worker's ABSOLUTE USS, sampled at
    5 ms, clipped to that task's [t_start, t_end] window and aligned so every
    task starts at x=0. Red = pyarrow tasks, blue = arrow_rs tasks.
  * a dashed black line at Ray's expectation E = 2 x target_max_block_size —
    Ray's OWN provisioning assumption (context.py:44: "memory footprint will
    be about 2 * num_cpus * target_max_block_size"), recorded per run in
    meta.json by bench_suite, not chosen by us.

Why absolute USS is what the task "actually has": the kernel OOM killer and
Ray's memory monitor act on absolute process memory, so imports, warmup
retention and the decode working set all count — for both readers equally.
USS excludes shared pages (the plasma object store), which is the one part of
task memory Ray's admission gate DOES account. Task windows come from the
worker hook's FileReader.read patch (one row per read task, both readers);
Ray workers run one task at a time, so a window's samples belong to exactly
that task. Warmup-read tasks are excluded via meta.json's warm_end stamp.

Reading the figure: lines that stay at/below the dashed E are tasks behaving
as Ray schedules for; lines that tower over it are the hidden decode heap
that causes the OOMs we're trying to remove.

Usage:
  python task_mem.py            # every runs/<dir> pair that has tasks_*.csv
  python task_mem.py <config>   # only configs whose name contains <config>
Figures -> figs/task_mem/<config>.png
"""
import csv
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "runs")
FIG = os.path.join(HERE, "figs", "task_mem")
MB = 1024 * 1024

COLORS = {"pyarrow": "#c0392b", "arrow_rs": "#2471a3"}
DEFAULT_EXPECTED_MB = 256.0  # 2 x 128 MiB default target_max_block_size


def _meta(run_dir):
    p = os.path.join(run_dir, "meta.json")
    if os.path.exists(p):
        try:
            return json.load(open(p))
        except Exception:
            pass
    return {}


def _reader_of(run_dir):
    m = _meta(run_dir)
    if m.get("reader") in COLORS:
        return m["reader"]
    name = os.path.basename(run_dir)
    if "__arrow_rs" in name:
        return "arrow_rs"
    if "__pyarrow" in name:
        return "pyarrow"
    # s3 configs are tagged (s3__win16_bud8, ...) — everything but the
    # explicit pyarrow baseline is an arrow-rs config.
    return "arrow_rs"


def _task_series(run_dir):
    """[(t_start, xs, ys_mb)] — one entry per read task, warmup excluded.
    xs = seconds since the task started; ys = ABSOLUTE worker USS in MB.

    Each window is bracketed with the nearest sample on either side (clamped
    to the window edges), so a task shorter than the sampling interval still
    gets a line at the USS level the worker held through it."""
    import bisect

    warm_end = _meta(run_dir).get("warm_end", 0.0)
    out = []
    for tf in sorted(glob.glob(os.path.join(run_dir, "tasks_*.csv"))):
        uf = os.path.join(run_dir, os.path.basename(tf).replace("tasks_", "uss_"))
        if not os.path.exists(uf):
            continue
        samples = [(float(r[0]), float(r[1])) for r in list(csv.reader(open(uf)))[1:] if r]
        epochs = [e for e, _ in samples]
        for r in list(csv.reader(open(tf)))[1:]:
            if not r:
                continue
            t0, t1 = float(r[0]), float(r[1])
            if t0 < warm_end:
                continue  # warmup-read task
            lo = bisect.bisect_left(epochs, t0)
            hi = bisect.bisect_right(epochs, t1)
            xs = [e - t0 for e, _ in samples[lo:hi]]
            ys = [u / MB for _, u in samples[lo:hi]]
            if lo > 0:  # level held entering the task
                xs.insert(0, 0.0)
                ys.insert(0, samples[lo - 1][1] / MB)
            if hi < len(samples):  # level right after the task ended
                xs.append(t1 - t0)
                ys.append(samples[hi][1] / MB)
            if not xs:
                continue
            out.append((t0, xs, ys))
    return out


def _plot_pair(config, dirs_by_reader):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.2))
    counts = {}
    expected = {}
    for reader, run_dir in dirs_by_reader.items():
        expected[reader] = _meta(run_dir).get("expected_task_mb", DEFAULT_EXPECTED_MB)
        tasks = _task_series(run_dir)
        counts[reader] = len(tasks)
        alpha = max(0.15, min(0.85, 6.0 / max(1, len(tasks))))
        for i, (_t0, xs, ys) in enumerate(sorted(tasks)):
            ax.plot(xs, ys, color=COLORS[reader], lw=1.3, alpha=alpha,
                    label=f"{reader} tasks (n={len(tasks)})" if i == 0 else None)
    evals = set(expected.values())
    if len(evals) <= 1:
        e = next(iter(evals), DEFAULT_EXPECTED_MB)
        ax.axhline(e, color="black", ls="--", lw=1.8,
                   label=f"Ray expectation 2x target_max_block_size = {e:.0f} MB")
    else:  # readers ran with different block-size configs — one line each
        for reader, e in expected.items():
            ax.axhline(e, color=COLORS[reader], ls="--", lw=1.8,
                       label=f"Ray expectation ({reader}) = {e:.0f} MB")
    ax.set_xlabel("seconds since task start")
    ax.set_ylabel("worker USS during task (MB, absolute)")
    ax.set_title(f"per-task memory over time vs Ray's expectation — {config}")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, f"{config}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    tag = "  ".join(f"{r}:{n} tasks" for r, n in counts.items())
    print(f"wrote {out}   ({tag})")


def _discover():
    """Pair run dirs: exact __pyarrow/__arrow_rs siblings first, then family
    baselines (tuning__pyarrow, s3__pyarrow) for tagged arrow-rs configs."""
    dirs = [d for d in sorted(glob.glob(os.path.join(OUT, "*")))
            if os.path.isdir(d) and glob.glob(os.path.join(d, "tasks_*.csv"))]
    byname = {os.path.basename(d): d for d in dirs}
    used = set()
    pairs = {}
    for name, d in byname.items():
        if _reader_of(d) != "pyarrow" or "__pyarrow" not in name:
            continue
        partner = name.replace("__pyarrow", "__arrow_rs")
        if partner in byname:
            config = name.replace("__pyarrow", "")
            pairs[config] = {"pyarrow": d, "arrow_rs": byname[partner]}
            used.update([name, partner])
    for name, d in byname.items():
        if name in used or _reader_of(d) != "arrow_rs":
            continue
        family = name.split("__")[0]
        baseline = f"{family}__pyarrow"
        entry = {"arrow_rs": d}
        if baseline in byname:
            entry["pyarrow"] = byname[baseline]
        pairs[name.replace("__arrow_rs", "")] = entry
        used.add(name)
    skipped = [n for n in byname if n not in used
               and not any(n == os.path.basename(v)
                           for p in pairs.values() for v in p.values())]
    return pairs, skipped


def main():
    want = sys.argv[1] if len(sys.argv) > 1 else None
    pairs, skipped = _discover()
    if not pairs:
        print("no runs/<dir> with tasks_*.csv found — re-run bench_suite.py first\n"
              "(task windows are recorded by the updated worker hook)")
        return
    for config in sorted(pairs):
        if want and want not in config:
            continue
        _plot_pair(config, pairs[config])
    for n in skipped:
        print(f"(unpaired, skipped: {n})")


if __name__ == "__main__":
    main()
