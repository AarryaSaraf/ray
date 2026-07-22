"""THE memory graph of record: per-task absolute USS over time, one figure per
benchmark config, both readers overlaid, against the ideal-streaming-reader line.

What each figure shows:

  * one line per READ TASK: the executing worker's ABSOLUTE USS, sampled at
    5 ms, clipped to that task's [t_start, t_end] window and aligned so every
    task starts at x=0. Red = pyarrow tasks, blue = arrow_rs tasks. These are
    measured.
  * one dashed reference line: the ideal streaming reader —

        ideal = floor + one output block (target_max_block_size)

    floor = the MEASURED USS a worker already holds ENTERING the task (imports +
            warm retained heap), median of both readers' task-start levels.
    block = target_max_block_size, the one output block Ray coalesces in the
            worker before sealing it to plasma.

    Why this and not something else. The reference must be DECODER-INDEPENDENT —
    what a *perfect* streaming reader would peak at, not what a given reader
    happens to do. A perfect reader reads compressed bytes off disk (OS page
    cache, file-backed → not USS), decodes them in small bounded batches (a few
    MB → second-order), and streams output out; its one irreducible PRIVATE cost
    is the output block being assembled before handoff to plasma. That block
    size is a Ray property, identical for both readers, and — the whole point —
    FLAT in row-group size. It deliberately excludes the whole compressed row
    group (that is PyArrow's limitation, not a requirement) and the decoded
    group. So how far each reader towers ABOVE this line is the private memory it
    holds BEYOND an ideal streaming reader: for PyArrow ~the whole row group; for
    arrow-rs whatever it over-buffers (block coalescing + allocator retention).

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
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "runs")
DEFAULT_FIG = os.path.join(HERE, "figs", "task_mem")
# FIG is where figures are written. Standalone runs stamp a fresh figs/<ts>/task_mem/
# (see main); summarize.py sets this to its own per-run dir before calling main().
FIG = DEFAULT_FIG
MB = 1024 * 1024

COLORS = {"pyarrow": "#c0392b", "arrow_rs": "#2471a3"}
EXPECTED_COLOR = "#333333"  # the single reference line (ideal streaming reader)
DEFAULT_BLOCK_MB = 128.0  # target_max_block_size fallback if meta lacks it


def _point_latest(run_dir):
    """Best-effort figs/latest symlink → the newest run dir, so it's easy to open
    the most recent figures without hunting for the timestamp."""
    link = os.path.join(HERE, "figs", "latest")
    try:
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(os.path.basename(run_dir), link)
    except OSError:
        pass


def _describe(meta):
    """A one-line 'what was read' summary from a run's meta.json — rows, files,
    row groups, the largest row group's size, compression, and the columns —
    so a figure titled 'one big row group' also says how big and of what."""
    if not meta.get("rows_total"):
        return ""

    def _mb(x):
        return f"{x:.0f}MB" if isinstance(x, (int, float)) else "?"

    rows = meta["rows_total"]
    rows_s = f"{rows / 1e6:.1f}M rows" if rows >= 1e6 else f"{rows / 1e3:.0f}k rows"
    rg = (
        f"rg {_mb(meta.get('max_rg_uncomp_mb'))} decoded / "
        f"{_mb(meta.get('max_rg_comp_mb'))} on disk"
    )
    if meta.get("compression"):
        rg += f" {meta['compression']}"
    parts = [
        rows_s,
        f"{meta.get('num_files', '?')} file(s)",
        f"{meta.get('num_row_groups', '?')} row group(s)",
        rg,
    ]
    if meta.get("schema_desc"):
        parts.append(meta["schema_desc"])
    return " · ".join(parts)


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
        samples = [
            (float(r[0]), float(r[1])) for r in list(csv.reader(open(uf)))[1:] if r
        ]
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

    fig, ax = plt.subplots(figsize=(10, 5.4))
    counts = {}
    over = {}
    metas = []
    tasks_by_reader = {}
    entry_levels = []  # first in-window USS of every task, both readers pooled
    for reader, run_dir in dirs_by_reader.items():
        meta = _meta(run_dir)
        metas.append(meta)
        tasks = _task_series(run_dir)
        tasks_by_reader[reader] = tasks
        counts[reader] = len(tasks)
        alpha = max(0.15, min(0.85, 6.0 / max(1, len(tasks))))
        for i, (_t0, xs, ys) in enumerate(sorted(tasks)):
            ax.plot(
                xs,
                ys,
                color=COLORS[reader],
                lw=1.3,
                alpha=alpha,
                label=f"{reader} tasks (n={len(tasks)})" if i == 0 else None,
            )
            entry_levels.append(ys[0])

    # ONE reference line: the ideal streaming reader — measured warm floor plus
    # ONE output block. Decoder-independent (target_max_block_size is a Ray
    # property, identical for both readers) and FLAT in row-group size. See the
    # module docstring for why this, not the floor and not floor+compressed.
    floor = statistics.median(entry_levels) if entry_levels else None
    if floor is not None:
        block = next(
            (m["target_block_mb"] for m in metas if m.get("target_block_mb")),
            DEFAULT_BLOCK_MB,
        )
        ideal = floor + block
        ax.axhline(
            ideal,
            color=EXPECTED_COLOR,
            ls="--",
            lw=1.8,
            label=(
                f"ideal streaming reader = {ideal:.0f} MB "
                f"(floor {floor:.0f} + 1 block {block:.0f})"
            ),
        )
        for reader, tasks in tasks_by_reader.items():
            peaks = [max(ys) for _, _, ys in tasks]
            if peaks:
                over[reader] = max(p - ideal for p in peaks)

    ax.set_xlabel("seconds since task start")
    ax.set_ylabel("worker USS during task (MB, absolute)")
    subtitle = next((_describe(m) for m in metas if _describe(m)), "")
    ax.set_title(
        f"per-task memory over time vs ideal streaming reader — {config}"
        + (f"\n{subtitle}" if subtitle else ""),
        fontsize=10,
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, f"{config}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    tag = "  ".join(f"{r}:{n} tasks" for r, n in counts.items())
    if over:
        tag += "   peak USS above ideal: " + "  ".join(
            f"{r}:{o:+.0f}MB" for r, o in over.items()
        )
    print(f"wrote {out}   ({tag})")


def _discover():
    """Pair run dirs: exact __pyarrow/__arrow_rs siblings first, then family
    baselines (tuning__pyarrow, s3__pyarrow) for tagged arrow-rs configs."""
    dirs = [
        d
        for d in sorted(glob.glob(os.path.join(OUT, "*")))
        if os.path.isdir(d) and glob.glob(os.path.join(d, "tasks_*.csv"))
    ]
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
    skipped = [
        n
        for n in byname
        if n not in used
        and not any(
            n == os.path.basename(v) for p in pairs.values() for v in p.values()
        )
    ]
    return pairs, skipped


def main():
    global FIG
    want = sys.argv[1] if len(sys.argv) > 1 else None
    # Standalone run: write into a fresh figs/<timestamp>/task_mem/ so each run's
    # figures are self-contained and never overwrite a previous run's. When called
    # from summarize.py, FIG is already set to that run's dir and left untouched.
    if FIG == DEFAULT_FIG:
        run_id = time.strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(HERE, "figs", run_id)
        FIG = os.path.join(run_dir, "task_mem")
        _point_latest(run_dir)
        print(f"figures -> {FIG} (also figs/latest)")
    pairs, skipped = _discover()
    if not pairs:
        print(
            "no runs/<dir> with tasks_*.csv found — re-run bench_suite.py first\n"
            "(task windows are recorded by the updated worker hook)"
        )
        return
    for config in sorted(pairs):
        if want and want not in config:
            continue
        _plot_pair(config, pairs[config])
    for n in skipped:
        print(f"(unpaired, skipped: {n})")


if __name__ == "__main__":
    main()
