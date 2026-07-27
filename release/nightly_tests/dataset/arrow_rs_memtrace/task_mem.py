"""THE memory graph of record: per-task absolute USS over time, one figure per
benchmark config, all readers overlaid, with a table comparing each reader's
wall time and peak USS against the pyarrow V2-scanner baseline.

What each figure shows:

  * one line per READ TASK: the executing worker's ABSOLUTE USS, sampled at
    5 ms, clipped to that task's [t_start, t_end] window and aligned so every
    task starts at x=0. Red = pyarrow (V2 scanner), orange = pyarrow
    (iter_batches), blue = arrow_rs. These are measured.
  * a table on top: absolute wall (s) and peak USS (MB) for each reader, plus
    each one's %Δ (wall and peak USS) against the pyarrow V2-scanner reader —
    i.e. against the read Ray actually performs by default. There is no
    ideal/streaming reference line: with K parallel byte-budgeted range reads
    per row group there's no single decoder-independent "ideal peak" a reader
    could tower over, so the honest comparison is just reader-vs-what-Ray-does.

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
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "runs")
DEFAULT_FIG = os.path.join(HERE, "figs", "task_mem")
# FIG is where figures are written. Standalone runs stamp a fresh figs/<ts>/task_mem/
# (see main); summarize.py sets this to its own per-run dir before calling main().
FIG = DEFAULT_FIG
MB = 1024 * 1024

COLORS = {"pyarrow": "#c0392b", "pyarrow_iter": "#e67e22", "arrow_rs": "#2471a3"}
# Display order + human labels. pyarrow (the V2 scanner path) is the baseline that
# %Δ in the timing table is measured against.
READER_ORDER = ["pyarrow", "pyarrow_iter", "arrow_rs"]
READER_LABEL = {
    "pyarrow": "pyarrow (V2 scanner)",
    "pyarrow_iter": "pyarrow (iter_batches)",
    "arrow_rs": "arrow-rs",
}
BASELINE_READER = "pyarrow"


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
    # Order matters: "__pyarrow_iter" also contains "__pyarrow", so test it first.
    if "__pyarrow_iter" in name:
        return "pyarrow_iter"
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

    # One image = a comparison table on top, the USS-over-time graph below.
    fig, (ax_tbl, ax) = plt.subplots(
        2, 1, figsize=(11, 6.8), gridspec_kw={"height_ratios": [1, 4]}
    )
    # Readers present, in canonical display order (baseline first).
    readers = [r for r in READER_ORDER if r in dirs_by_reader]
    readers += [r for r in dirs_by_reader if r not in readers]  # any unknown, appended

    counts = {}
    peak_by_reader = {}
    wall_by_reader = {}
    metas = []
    tasks_by_reader = {}
    for reader in readers:
        run_dir = dirs_by_reader[reader]
        meta = _meta(run_dir)
        metas.append(meta)
        tasks = _task_series(run_dir)
        tasks_by_reader[reader] = tasks
        # Wall time for the table: the driver-measured end-to-end wall stamped by
        # the axis (_R -> meta.json) is authoritative. Fall back to the read-task
        # window span (max end - min start) for older runs that predate the stamp,
        # so a table still renders (it undercounts driver/consume overhead).
        wall = meta.get("wall_s")
        if wall is None and tasks:
            starts = [t0 for t0, _, _ in tasks]
            ends = [t0 + (xs[-1] if xs else 0.0) for t0, xs, _ in tasks]
            wall = max(ends) - min(starts)
        wall_by_reader[reader] = wall
        counts[reader] = len(tasks)
        peak_by_reader[reader] = max((max(ys) for _, _, ys in tasks), default=None)
        alpha = max(0.15, min(0.85, 6.0 / max(1, len(tasks))))
        for i, (_t0, xs, ys) in enumerate(sorted(tasks)):
            ax.plot(
                xs,
                ys,
                color=COLORS.get(reader, "#555555"),
                lw=1.3,
                alpha=alpha,
                label=f"{READER_LABEL.get(reader, reader)} (n={len(tasks)})"
                if i == 0
                else None,
            )

    ax.set_xlabel("seconds since task start")
    ax.set_ylabel("worker USS during task (MB, absolute)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    # --- comparison table (top axes): each reader's absolute wall + peak USS,
    # and BOTH measured against the pyarrow V2-scanner baseline (the read Ray does
    # by default). No ideal/streaming reference — with K parallel range reads per
    # row group there's no single decoder-independent "ideal peak" to tower over;
    # the honest question is just "how does each reader compare to what Ray does
    # today", on speed and on private memory. ---
    base_wall = wall_by_reader.get(BASELINE_READER)
    base_peak = peak_by_reader.get(BASELINE_READER)

    def _delta(val, base):
        if val is None or not base:
            return "?"
        pct = (val - base) / base * 100.0
        # memory: lower is better; time: lower is better — same phrasing works.
        return f"{pct:+.1f}%  ({'less' if pct < 0 else 'more'})"

    col_labels = [
        "reader",
        "wall (s)",
        "Δ wall vs V2 scanner",
        "peak USS (MB)",
        "Δ USS vs V2 scanner",
    ]
    rows_text = []
    cell_colors = []
    for reader in readers:
        w = wall_by_reader.get(reader)
        pk = peak_by_reader.get(reader)
        if reader == BASELINE_READER:
            dwall = dpeak = "— (baseline)"
        else:
            dwall = _delta(w, base_wall)
            dpeak = _delta(pk, base_peak)
        rows_text.append(
            [
                READER_LABEL.get(reader, reader),
                f"{w:.3f}" if w is not None else "?",
                dwall,
                f"{pk:.0f}" if pk is not None else "?",
                dpeak,
            ]
        )
        # tint the reader-name cell with its line color for at-a-glance matching
        c = COLORS.get(reader, "#555555")
        cell_colors.append([c + "33"] + ["#00000000"] * (len(col_labels) - 1))

    ax_tbl.axis("off")
    if rows_text:
        tbl = ax_tbl.table(
            cellText=rows_text,
            colLabels=col_labels,
            cellColours=cell_colors,
            colLoc="center",
            cellLoc="center",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.0, 1.35)
    subtitle = next((_describe(m) for m in metas if _describe(m)), "")
    # Wrap a long "what was read" subtitle so it can't run past the image edge.
    if subtitle and len(subtitle) > 84:
        mid = subtitle.rfind(" · ", 0, len(subtitle) // 2 + 12)
        if mid > 0:
            subtitle = subtitle[:mid] + " ·\n" + subtitle[mid + 3 :]
    ax_tbl.set_title(
        f"reader comparison ({len(readers)}-way) — {config}"
        + (f"\n{subtitle}" if subtitle else ""),
        fontsize=11,
        fontweight="bold",
    )

    os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, f"{config}.png")
    fig.tight_layout()
    # bbox_inches="tight" so the table/legend/title are captured even if they'd
    # otherwise spill past the fixed figure canvas.
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    tag = "  ".join(f"{r}:{n} tasks" for r, n in counts.items())
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
        if _reader_of(d) != "pyarrow" or not name.endswith("__pyarrow"):
            continue
        stem = name[: -len("__pyarrow")]
        # Gather every reader that read this exact config as a __<reader> sibling
        # (pyarrow_iter and/or arrow_rs). Register only when there's something to
        # compare against — a lone pyarrow dir is left for the Phase-2 family
        # baseline logic, exactly as before.
        entry = {"pyarrow": d}
        for reader in ("pyarrow_iter", "arrow_rs"):
            sib = f"{stem}__{reader}"
            if sib in byname:
                entry[reader] = byname[sib]
        if len(entry) > 1:
            pairs[stem] = entry
            used.update(os.path.basename(v) for v in entry.values())
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
