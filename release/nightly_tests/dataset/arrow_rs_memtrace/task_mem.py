"""THE memory graph of record: per-task absolute USS over time, one figure per
benchmark config, both readers overlaid, against each reader's measured
*expected-without-decode* line.

What each figure shows (graphs only — no scalars, no baseline subtraction):

  * one line per READ TASK: the executing worker's ABSOLUTE USS, sampled at
    5 ms, clipped to that task's [t_start, t_end] window and aligned so every
    task starts at x=0. Red = pyarrow tasks, blue = arrow_rs tasks.
  * one dashed line per reader: everything a well-behaved task is expected to
    hold EXCEPT the decode working set —

        expected = floor + compressed-in-flight + output block

    floor      = the worker's USS entering the task (measured: median of the
                 task lines' starting levels — imports + warm retained heap).
    in-flight  = the compressed bytes the reader must buffer: the fixture's
                 largest row group's compressed size (recorded in meta.json by
                 bench_suite._note_fixture) for whole-group readers (PyArrow,
                 arrow-rs local K=1); capped at fetch_window_mb for the
                 arrow-rs windowed S3 path.
    block      = target_max_block_size (the output block being assembled in
                 heap before it is yielded to plasma).

    How far a task's line towers ABOVE its dashed line is that reader's decode
    working set — the unaccounted, layout-dependent heap that causes the OOMs.

Why absolute USS: the kernel OOM killer and Ray's memory monitor act on
absolute private memory; USS excludes shared pages (plasma), the part Ray's
object-store accounting already covers. Task windows come from the worker
hook's FileReader.read patch; Ray workers run one task at a time, so a
window's samples belong to exactly that task. Warmup-read tasks are excluded
via meta.json's warm_end stamp.

Usage:
  python task_mem.py            # every runs/<dir> pair that has tasks_*.csv
  python task_mem.py <config>   # only configs whose name contains <config>
Figures -> figs/task_mem/<config>.png
"""
import csv
import glob
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "runs")
FIG = os.path.join(HERE, "figs", "task_mem")
MB = 1024 * 1024

COLORS = {"pyarrow": "#c0392b", "arrow_rs": "#2471a3"}
DEFAULT_BLOCK_MB = 128.0  # target_max_block_size default, used if meta lacks it


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


def _expected_mb(meta, floor_mb):
    """floor + compressed-in-flight + output block, or None if the run's meta
    predates _note_fixture (old runs: no line rather than a made-up one)."""
    comp = meta.get("max_rg_comp_mb")
    if comp is None or floor_mb is None:
        return None
    block = meta.get("target_block_mb", DEFAULT_BLOCK_MB)
    in_flight = comp
    # The windowed S3 path caps compressed bytes in flight at the fetch window;
    # local (and PyArrow anywhere) buffers the whole row group's compressed bytes.
    if (
        meta.get("reader") == "arrow_rs"
        and str(meta.get("fixture", "")).startswith("s3://")
        and meta.get("fetch_window_mb")
    ):
        in_flight = min(comp, float(meta["fetch_window_mb"]))
    return floor_mb + in_flight + block, floor_mb, in_flight, block


def _plot_pair(config, dirs_by_reader):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.2))
    counts = {}
    over = {}
    for reader, run_dir in dirs_by_reader.items():
        meta = _meta(run_dir)
        tasks = _task_series(run_dir)
        counts[reader] = len(tasks)
        alpha = max(0.15, min(0.85, 6.0 / max(1, len(tasks))))
        for i, (_t0, xs, ys) in enumerate(sorted(tasks)):
            ax.plot(xs, ys, color=COLORS[reader], lw=1.3, alpha=alpha,
                    label=f"{reader} tasks (n={len(tasks)})" if i == 0 else None)
        floor = (statistics.median(ys[0] for _, _, ys in tasks) if tasks else None)
        exp = _expected_mb(meta, floor)
        if exp is not None:
            e, fl, infl, blk = exp
            ax.axhline(e, color=COLORS[reader], ls="--", lw=1.6,
                       label=(f"{reader} expected w/o decode = {e:.0f} MB "
                              f"(floor {fl:.0f} + fetch {infl:.0f} + block {blk:.0f})"))
            peaks = [max(ys) for _, _, ys in tasks]
            over[reader] = max(p - e for p in peaks) if peaks else 0.0
    ax.set_xlabel("seconds since task start")
    ax.set_ylabel("worker USS during task (MB, absolute)")
    ax.set_title(f"per-task memory over time vs expected-without-decode — {config}")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, f"{config}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    tag = "  ".join(f"{r}:{n} tasks" for r, n in counts.items())
    if over:
        tag += "   worst overshoot: " + "  ".join(
            f"{r}:{o:+.0f}MB" for r, o in over.items())
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
