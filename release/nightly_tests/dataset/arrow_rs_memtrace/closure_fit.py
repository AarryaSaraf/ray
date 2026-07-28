#!/usr/bin/env python3
"""Analyze the `closure` axis: confirm  USS ≈ S_in + W_decode + S_out + M_uncontrolled.

Reads runs/results_closure.json (produced by `python bench_suite.py closure`) and:

  1. Builds a PREDICTED per-worker read-caused USS from independently-known bytes only
     (decoded RG size from the schema, compressed RG size from the footer, decode
     budget, output block cap) — M is NOT in the prediction.
  2. Regresses MEASURED USS on PREDICTED across both readers and all sizes.
        slope ≈ 1, small constant intercept, high R²  ⇒  M is a bounded constant and
        the additive model closes.  slope > 1  ⇒  M grows with size (equation incomplete).
     The intercept is the measured M_uncontrolled; the per-point residual is checked
     to be FLAT in N (M does not scale) — the "increases at most slope-1" condition.
  3. Reads S_out off as (retained − decode_drop) and checks it is size-independent.
  4. Per-reader fit of USS vs decoded_MB: slope≈1 for PyArrow, slope≈0 for arrow-rs.

Usage:  python closure_fit.py            # reads runs/results_closure.json
        python closure_fit.py <path.json>

macOS caveat: USS is RSS-based (directional). The slopes/ratios are the point; run on
Linux for authoritative absolute numbers.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
Y = "max_worker_incr_mb"  # per-worker read-caused private-heap growth (floor subtracted)


def _fit(x, y):
    """Least-squares line y = m·x + b; return (slope, intercept, R²). NaN if <2 points."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    m, b = np.polyfit(x, y, 1)
    resid = y - (m * x + b)
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(m), float(b), float(r2)


def _predict(reader, r):
    """Predicted per-worker read-caused USS (MB) from independently-known bytes only.

    S_out is one output block in flight (target_max_block_size), a size-independent
    constant present in BOTH consume modes — the read task builds a block before it is
    handed downstream regardless of how the driver consumes it. M is deliberately absent
    (it is the residual we test for constancy)."""
    decoded = float(r.get("decoded_mb") or 0.0)
    compressed = float(r.get("max_rg_comp_mb") or 0.0)  # from the footer (geometry)
    budget = float(r.get("budget_mb") or 8.0)
    block = float(r.get("target_block_mb") or 0.0)  # S_out: one block in flight
    if reader == "arrow_rs":
        # W_decode = byte budget; S_in ≈ 0 locally (streamed, no pre_buffer of whole RG)
        return budget + block
    # pyarrow scanner: W_decode = whole decoded row group; S_in = pre_buffered compressed RG
    return decoded + compressed + block


def main():
    path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.path.join(HERE, "runs", "results_closure.json")
    )
    rows = json.load(open(path))
    # index by (rows, reader, mode)
    by = {(r["rows"], r["reader"], r["mode"]): r for r in rows}
    readers = sorted({r["reader"] for r in rows})
    sizes = sorted({r["rows"] for r in rows})

    print(
        f"\n===== CLOSURE: USS ≈ S_in + W_decode + S_out + M  ({os.path.basename(path)}) ====="
    )
    print(
        f"readers={readers}  sizes(M)={[s // 1_000_000 for s in sizes]}  y = {Y} (per-worker, floor-subtracted)\n"
    )

    # ---- 1. per-point table: measured vs predicted, residual = measured − predicted ≈ M ----
    print(
        f"{'reader':9} {'rows':>5} {'mode':>12} {'decoded':>8} {'pred':>7} {'meas':>7} {'resid(M)':>9}"
    )
    pred_all, meas_all, resid_by_reader = [], [], {rd: [] for rd in readers}
    sout = {rd: [] for rd in readers}
    for rd in readers:
        for s in sizes:
            for mode in ("decode_drop", "iter_batches"):
                r = by.get((s, rd, mode))
                if not r or r.get(Y) is None:
                    continue
                pred = _predict(rd, r)
                meas = float(r[Y])
                pred_all.append(pred)
                meas_all.append(meas)
                resid_by_reader[rd].append((s, meas - pred))
                print(
                    f"{rd:9} {s//1_000_000:>4}M {mode:>12} "
                    f"{float(r.get('decoded_mb') or 0):>7.0f} {pred:>7.1f} {meas:>7.1f} {meas - pred:>9.1f}"
                )
            # S_out = retained − decode_drop
            rr = by.get((s, rd, "iter_batches"))
            dd = by.get((s, rd, "decode_drop"))
            if rr and dd and rr.get(Y) is not None and dd.get(Y) is not None:
                sout[rd].append((s, float(rr[Y]) - float(dd[Y])))

    # ---- 2. the master test: measured vs predicted (all readers, all sizes, both modes) ----
    m, b, r2 = _fit(pred_all, meas_all)
    print("\n----- master closure fit: measured USS vs predicted USS -----")
    print(
        f"  slope = {m:.3f}   intercept (≈ M_uncontrolled) = {b:.1f} MB   R² = {r2:.4f}   (n={len(pred_all)})"
    )
    verdict = (
        "CLOSES: slope≈1, bounded constant M"
        if 0.8 <= m <= 1.2
        else "slope>1: M grows with size — equation INCOMPLETE"
        if m > 1.2
        else "slope<1: predicted over-counts (check S_in / block-cap assumptions)"
    )
    print(f"  verdict: {verdict}")

    # ---- 3. M is flat in N (does not scale) ----
    print(
        "\n----- residual (measured − predicted ≈ M_uncontrolled) vs N: must be FLAT -----"
    )
    for rd in readers:
        pts = sorted(resid_by_reader[rd])
        if len(pts) >= 2:
            xs = [s for s, _ in pts]
            ys = [v for _, v in pts]
            mm, bb, rr2 = _fit(xs, ys)
            slope_per_M = mm * 1_000_000  # MB of M per million rows
            print(
                f"  {rd:9}: M slope = {slope_per_M:+.2f} MB per 1M rows, mean = {np.mean(ys):.1f} MB "
                f"({'flat ✓' if abs(slope_per_M) < 5 else 'GROWS ✗'})"
            )

    # ---- 4. per-reader headline: USS(decode_drop) vs decoded_MB ----
    print("\n----- per-reader fit: decode-transient USS vs decoded_MB (headline) -----")
    for rd in readers:
        pts = [
            (
                float(by[(s, rd, "decode_drop")].get("decoded_mb") or 0),
                float(by[(s, rd, "decode_drop")][Y]),
            )
            for s in sizes
            if (s, rd, "decode_drop") in by
            and by[(s, rd, "decode_drop")].get(Y) is not None
        ]
        if len(pts) >= 2:
            mm, bb, rr2 = _fit([p[0] for p in pts], [p[1] for p in pts])
            shape = "∝N (whole group)" if mm > 0.5 else "flat (bounded budget)"
            print(
                f"  {rd:9}: slope = {mm:.3f} MB USS / MB decoded, intercept = {bb:.1f} MB, R²={rr2:.3f}  → {shape}"
            )

    # ---- 5. mode agreement: with small blocks decode_drop ≈ iter_batches ----
    print(
        "\n----- consistency: iter_batches − decode_drop (small blocks ⇒ ≈0, both stream) -----"
    )
    for rd in readers:
        if sout[rd]:
            vals = [v for _, v in sout[rd]]
            print(
                f"  {rd:9}: Δ = {np.mean(vals):+.1f} ± {np.std(vals):.1f} MB across sizes "
                f"{[s // 1_000_000 for s, _ in sout[rd]]}M "
                f"({'consistent ✓' if abs(np.mean(vals)) < 15 else 'DIVERGES ✗'})"
            )

    _plot(by, readers, sizes, pred_all, meas_all, m, b)


def _plot(by, readers, sizes, pred_all, meas_all, m, b):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # matplotlib optional
        print(f"\n(plot skipped: {e})")
        return
    outdir = os.path.join(HERE, "figs", "closure")
    os.makedirs(outdir, exist_ok=True)
    colors = {
        "pyarrow": "tab:red",
        "arrow_rs": "tab:blue",
        "pyarrow_iter": "tab:orange",
    }
    markers = {"decode_drop": "o", "iter_batches": "s"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # left: measured vs predicted (the closure test)
    for rd in readers:
        for mode in ("decode_drop", "iter_batches"):
            xs = [_predict(rd, by[(s, rd, mode)]) for s in sizes if (s, rd, mode) in by]
            ys = [
                float(by[(s, rd, mode)][Y])
                for s in sizes
                if (s, rd, mode) in by and by[(s, rd, mode)].get(Y) is not None
            ]
            n = min(len(xs), len(ys))
            if n:
                ax1.scatter(
                    xs[:n],
                    ys[:n],
                    c=colors.get(rd, "gray"),
                    marker=markers[mode],
                    s=60,
                    label=f"{rd} / {mode}",
                )
    lim = max(pred_all + meas_all + [1]) * 1.1
    ax1.plot([0, lim], [0, lim], "k--", lw=1, label="y = x (perfect closure)")
    xx = np.linspace(0, lim, 50)
    ax1.plot(xx, m * xx + b, "g-", lw=1.5, label=f"fit: y={m:.2f}x+{b:.0f}")
    ax1.set_xlabel("predicted USS (MB)  = S_in + W_decode + S_out")
    ax1.set_ylabel(f"measured USS (MB)  [{Y}]")
    ax1.set_title("Closure: measured vs predicted (slope→1, intercept→M)")
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3)

    # right: decode-transient USS vs decoded_MB (headline: flat vs ∝N)
    for rd in readers:
        pts = [
            (
                float(by[(s, rd, "decode_drop")].get("decoded_mb") or 0),
                float(by[(s, rd, "decode_drop")][Y]),
            )
            for s in sizes
            if (s, rd, "decode_drop") in by
            and by[(s, rd, "decode_drop")].get(Y) is not None
        ]
        if pts:
            pts.sort()
            ax2.plot(
                [p[0] for p in pts],
                [p[1] for p in pts],
                "o-",
                c=colors.get(rd, "gray"),
                label=rd,
            )
    ax2.set_xlabel("decoded row-group size (MB) = N × 72 B")
    ax2.set_ylabel(f"decode-transient USS (MB)  [{Y}]")
    ax2.set_title("W_decode: PyArrow ∝ N vs arrow-rs flat")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    p = os.path.join(outdir, "closure.png")
    fig.savefig(p, dpi=120)
    print(f"\nplot -> {p}")


if __name__ == "__main__":
    main()
