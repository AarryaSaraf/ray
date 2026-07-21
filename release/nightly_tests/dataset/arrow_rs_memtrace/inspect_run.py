"""Per-worker USS breakdown for one benchmark run dir — is a node-sum peak a few
big decoders or many small workers each paying a fixed cost?

`node_sum_incr` sums (peak − baseline) across every worker that wrote a trace. A
single-process probe can't show that decomposition; this can. For each
`uss_<host>_<pid>.csv` in the run dir it reports the worker's baseline (first
in-window sample), its in-window peak, and the delta, sorted biggest-delta first,
then a summary:

  n_workers        — how many wrote traces (≈ Ray workers that ran / idled)
  n_grown          — how many grew > 5 MB in the window (the actual decoders)
  sum_delta_MB     — Σ(peak−baseline) ≈ the node_sum_incr the suite reports
  max_delta_MB     — the largest single worker (≈ true per-decode working set)

If sum_delta is large but every per-worker delta is small and n_grown is high,
the "loss" is per-worker fixed cost multiplied across workers (e.g. the arrow-rs
extension imported after t0, not in the warm baseline) — NOT decode memory.

Usage:
  python inspect_run.py runs/layout__small_many_grp__iter_batches__arrow_rs
  python inspect_run.py <run_dir> <t0> <t1>     # restrict to a measured window
  python inspect_run.py <run_dir> --plot        # + per-worker USS-over-time PNG
                                                # (one line per worker = per-task
                                                # memory over time, since a suite
                                                # read is sequential per worker)
"""
import csv
import glob
import os
import sys

MB = 1024 * 1024


def _series(path):
    rows = list(csv.reader(open(path)))[1:]
    return [(float(r[0]), float(r[1])) for r in rows if r]  # (epoch, uss_bytes)


def _plot(run_dir, rows_series):
    """One step-line per worker, USS over time (seconds from first sample), plus
    the sum. Saved next to the run dir's name under figs/worker_<label>.png."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    label = os.path.basename(os.path.normpath(run_dir))
    fig, ax = plt.subplots(figsize=(10, 5))
    all_t0 = min(s[0][0] for _, s in rows_series if s)
    for name, s in rows_series:
        if not s:
            continue
        xs = [e - all_t0 for e, _ in s]
        ys = [u / MB for _, u in s]
        ax.step(xs, ys, where="post", lw=1.2,
                label=name.replace("uss_", "").replace(".csv", ""))
    ax.set_xlabel("seconds since first sample")
    ax.set_ylabel("worker USS (MB)")
    ax.set_title(f"per-worker USS over time — {label}")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"worker_{label}.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"wrote {out}")


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    args = [a for a in sys.argv[1:] if a != "--plot"]
    do_plot = "--plot" in sys.argv
    run_dir = args[0]
    t0 = float(args[1]) if len(args) > 1 else None
    t1 = float(args[2]) if len(args) > 2 else None

    files = sorted(glob.glob(os.path.join(run_dir, "uss_*.csv")))
    if not files:
        raise SystemExit(f"no uss_*.csv in {run_dir}")

    rows = []
    plot_series = []
    for f in files:
        s = _series(f)
        if t0 is not None:
            s = [(e, u) for (e, u) in s if t0 <= e <= t1]
        if not s:
            continue
        plot_series.append((os.path.basename(f), s))
        base = s[0][1]
        peak = max(u for _, u in s)
        t_peak = max(s, key=lambda eu: eu[1])[0] - s[0][0]
        rows.append((os.path.basename(f), base / MB, peak / MB, (peak - base) / MB, t_peak))

    rows.sort(key=lambda r: r[3], reverse=True)
    print(f"{'worker':40s} {'base_MB':>9s} {'peak_MB':>9s} {'delta_MB':>9s} {'t_peak_s':>9s}")
    for name, base, peak, delta, tp in rows:
        print(f"{name:40s} {base:9.1f} {peak:9.1f} {delta:9.1f} {tp:9.2f}")

    n = len(rows)
    grown = [r for r in rows if r[3] > 5.0]
    sum_delta = sum(r[3] for r in rows)
    max_delta = max((r[3] for r in rows), default=0.0)
    print("-" * 80)
    print(f"n_workers={n}  n_grown(>5MB)={len(grown)}  "
          f"sum_delta_MB={sum_delta:.1f}  max_delta_MB={max_delta:.1f}")
    if grown:
        print(f"grown-worker deltas (MB): {', '.join(f'{r[3]:.0f}' for r in grown)}")
    if do_plot:
        _plot(run_dir, plot_series)


if __name__ == "__main__":
    main()
