"""Plot per-worker + node-sum private-heap (USS) over time as STEP functions,
with Ray's object-store view overlaid. One figure per (fixture, consume) with
PyArrow vs arrow-rs side by side, shared y-axis.

USS is a piecewise-constant gauge sampled discretely, so we render it with
`where='post'` steps (hold-last) rather than interpolated lines — no invented
values between samples.
"""
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "runs")
FIG = os.path.join(HERE, "figs")
MB = 1024 * 1024


def load_worker_traces(run_dir):
    traces = {}
    for f in glob.glob(os.path.join(run_dir, "uss_*.csv")):
        pid = os.path.basename(f)[4:-4]
        rows = list(csv.reader(open(f)))[1:]
        if not rows:
            continue
        ep = np.array([float(r[0]) for r in rows])
        uss = np.array([float(r[1]) for r in rows])
        traces[pid] = (ep, uss)
    return traces


def load_objstore(run_dir):
    f = os.path.join(run_dir, "objstore.csv")
    if not os.path.exists(f):
        return None
    rows = list(csv.reader(open(f)))[1:]
    if not rows:
        return None
    return (np.array([float(r[0]) for r in rows]),
            np.array([float(r[1]) for r in rows]))


def hold_last(grid, ep, val, default=0.0):
    """Step-interpolate val(ep) onto grid, hold-last (value before each grid t)."""
    idx = np.searchsorted(ep, grid, side="right") - 1
    out = np.where(idx >= 0, val[np.clip(idx, 0, len(val) - 1)], default)
    return out


def plot_cell(ax, run_dir, win, color):
    t0, t1 = win["t_start"], win["t_end"]
    pre = 0.3
    traces = load_worker_traces(run_dir)
    # Common grid across the (pre-roll + window).
    grid = np.linspace(t0 - pre, t1, 800)
    total = np.zeros_like(grid)
    import_floor = 0.0
    for pid, (ep, uss) in traces.items():
        # Import floor = this worker's USS just before the measured window (warm,
        # imports done, no big decode yet). Absolute USS is plotted; the floor is
        # only a reference line so the decode "explosion" above it is legible.
        pre_mask = ep < t0
        base = uss[pre_mask][-1] if pre_mask.any() else uss.min()
        import_floor += base
        series = hold_last(grid, ep, uss, default=base)
        ax.step((grid - t0), series / MB, where="post", color=color,
                alpha=0.25, linewidth=0.8)
        total += series
    # Node-sum ABSOLUTE private heap (bold) — the real physical memory the node's
    # read workers occupy; entirely invisible to Ray's admission control.
    ax.step((grid - t0), total / MB, where="post", color=color, linewidth=2.4,
            label="node-sum USS (private heap, absolute)")
    warm_floor = import_floor
    # Import-floor reference (sum of warm-import baselines) — memory below this is
    # just interpreter+libs, not the decode explosion.
    ax.axhline(import_floor / MB, color="gray", ls=":", lw=1,
               label=f"import floor (Σ warm baselines) {import_floor/MB:.0f}MB")
    # Ray's view: object-store bytes.
    obj = load_objstore(run_dir)
    if obj is not None:
        oep, ob = obj
        og = hold_last(grid, oep, ob, default=0.0)
        ax.step((grid - t0), og / MB, where="post", color="black", ls="--",
                linewidth=1.6, label="object store (what Ray schedules on)")
    peak = total.max() / MB
    ax.axvline(0, color="k", lw=0.5, alpha=0.4)
    return peak, warm_floor / MB


def node_sum(run_dir, win, grid):
    """Node-sum absolute USS on `grid`, plus object-store series and import floor."""
    t0 = win["t_start"]
    traces = load_worker_traces(run_dir)
    total = np.zeros_like(grid)
    floor = 0.0
    for pid, (ep, uss) in traces.items():
        pre = ep < t0
        base = uss[pre][-1] if pre.any() else uss.min()
        floor += base
        total += hold_last(grid, ep, uss, default=base)
    obj = load_objstore(run_dir)
    og = hold_last(grid, *obj, default=0.0) if obj is not None else None
    return total, og, floor


def plot_overlay(fixture, consume, runs, ax):
    """Both readers' node-sum USS (bold) + object-store (dashed) on ONE axis."""
    colors = {"pyarrow": "#c0392b", "arrow_rs": "#2471a3"}
    t1 = max(runs[(fixture, consume, r)][1]["t_end"]
             - runs[(fixture, consume, r)][1]["t_start"]
             for r in ["pyarrow", "arrow_rs"] if (fixture, consume, r) in runs)
    for reader in ["pyarrow", "arrow_rs"]:
        if (fixture, consume, reader) not in runs:
            continue
        run_dir, win = runs[(fixture, consume, reader)]
        t0 = win["t_start"]
        grid = np.linspace(-0.3, t1, 800)
        total, og, floor = node_sum(run_dir, win, grid + t0)
        peak = total.max() / MB
        ax.step(grid, total / MB, where="post", color=colors[reader], linewidth=2.6,
                label=f"{reader}: USS (private heap)  peak={peak:.0f}MB, wall={win['wall_s']:.2f}s")
        if og is not None:
            ax.step(grid, og / MB, where="post", color=colors[reader], ls="--",
                    linewidth=1.4, alpha=0.8,
                    label=f"{reader}: object store (Ray's view)")
    ax.axvline(0, color="k", lw=0.5, alpha=0.4)
    ax.set_xlabel("seconds since read start")
    ax.set_ylabel("MB")
    ax.set_title(f"{fixture} — {consume}")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.2)


def main():
    os.makedirs(FIG, exist_ok=True)
    runs = {}
    for run_dir in glob.glob(os.path.join(OUT, "*")):
        wj = os.path.join(run_dir, "window.json")
        if not os.path.exists(wj):
            continue
        win = json.load(open(wj))
        runs[(win["fixture"], win["consume"], win["reader"])] = (run_dir, win)

    keys = sorted({(f, c) for (f, c, r) in runs})
    for fixture, consume in keys:
        readers = [r for r in ["pyarrow", "arrow_rs"] if (fixture, consume, r) in runs]
        if not readers:
            continue
        fig, axes = plt.subplots(1, len(readers), figsize=(7 * len(readers), 5),
                                 sharey=True, squeeze=False)
        axes = axes[0]
        for ax, reader in zip(axes, readers):
            run_dir, win = runs[(fixture, consume, reader)]
            color = "#c0392b" if reader == "pyarrow" else "#2471a3"
            peak, floor = plot_cell(ax, run_dir, win, color)
            ax.set_title(f"{reader}   peak node-sum USS={peak:.0f}MB   "
                         f"wall={win['wall_s']:.2f}s")
            ax.set_xlabel("seconds since read start")
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(alpha=0.2)
        axes[0].set_ylabel("MB")
        fig.suptitle(f"{fixture} — {consume}: private-heap USS over time "
                     f"(step); black dashed = Ray's scheduling view",
                     fontsize=11)
        fig.tight_layout()
        out = os.path.join(FIG, f"{fixture}__{consume}.png")
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print("wrote", out)

        # Overlay: both readers on one axis.
        figo, axo = plt.subplots(figsize=(9, 5.5))
        plot_overlay(fixture, consume, runs, axo)
        figo.suptitle("private-heap USS over time (step) — solid = actual worker heap, "
                      "dashed = what Ray schedules on", fontsize=10)
        figo.tight_layout()
        outo = os.path.join(FIG, f"{fixture}__{consume}__overlay.png")
        figo.savefig(outo, dpi=120)
        plt.close(figo)
        print("wrote", outo)


if __name__ == "__main__":
    main()
