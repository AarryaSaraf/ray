"""Summarize bench_suite.py results into readable tables + a leak-USS plot.

Reads runs/results_<axis>.json and prints one table per axis. For the leak axis
it also renders a step plot of node-sum USS across the 8 repeats per reader, so a
memory ratchet (leak) shows as a rising floor.
"""
import glob
import json
import os
import sys
import time

import task_mem  # reused for _meta / _describe (the fixture-detail subtitle)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "runs")


class _Tee:
    """Duplicate everything printed to stdout into a file too, so the whole
    human-readable digest (every table + the task_mem overshoot lines) is saved
    per run as figs/<ts>/summary.txt — no more copy-pasting the terminal."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for st in self._streams:
            st.write(s)

    def flush(self):
        for st in self._streams:
            st.flush()


# Set per-run in main() to figs/<timestamp>/; the module default is only a
# fallback for importing individual plot fns in a REPL.
FIG = os.path.join(HERE, "figs")
MB = 1024 * 1024


def _desc_for(*run_dirs):
    """Fixture-detail subtitle from the first of `run_dirs` whose meta.json has
    geometry (both readers share a fixture, so either dir's meta works)."""
    for d in run_dirs:
        s = task_mem._describe(task_mem._meta(d))
        if s:
            return s
    return ""


def _load(axis):
    p = os.path.join(OUT, f"results_{axis}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def table_correctness(rows):
    """Scenario × reader digest of the decoder-replacement correctness axis:
    parity verdict (pyarrow vs arrow-rs, value-level), golden-row check,
    chunked-read stability, decode path taken, wall, and peak USS. Mismatch
    details print below the table so a FAIL is immediately actionable."""
    print("\n## CORRECTNESS (differential + golden, corpus.py scenarios)")
    by = {}
    for r in rows:
        by.setdefault(r["scenario"], {})[r["reader"]] = r
    print(
        f"{'scenario':20s} {'parity':7s} {'golden':7s} {'stable':7s} "
        f"{'path(rs)':14s} {'pa_wall':>8s} {'rs_wall':>8s} {'rs_peakMB':>9s}"
    )
    n_ok = 0
    details = []
    for name, d in by.items():
        rs = d.get("arrow_rs", {})
        pa_ = d.get("pyarrow", {})
        parity = rs.get("parity", pa_.get("parity", "?"))
        golden = rs.get("golden_ok")
        golden_s = "-" if golden is None else ("OK" if golden else "FAIL")
        stable = next((v for k, v in rs.items() if k.startswith("stable_vs_")), None)
        stable_s = "-" if stable is None else stable
        nat, fb = rs.get("native", 0) or 0, rs.get("fallback", 0) or 0
        path_s = (
            "native"
            if nat and not fb
            else "fallback"
            if fb and not nat
            else f"mix({nat}/{fb})"
            if (nat or fb)
            else "-"
        )
        exp_hit = rs.get("expected_error_hit")
        if exp_hit is not None:
            path_s = f"err_gate:{'OK' if exp_hit else 'MISS'}"
        ok = parity == "OK" and golden is not False and stable in (None, "OK")
        n_ok += ok
        flag = "" if ok else "  <-- CHECK"
        print(
            f"{name:20s} {parity:7s} {golden_s:7s} {stable_s:7s} {path_s:14s} "
            f"{pa_.get('wall_s', 0):8.3f} {rs.get('wall_s', 0):8.3f} "
            f"{rs.get('node_sum_peak_mb', 0):9.0f}{flag}"
        )
        for r in (rs, pa_):
            for k in ("parity_detail", "golden_bad", "stability_detail", "error"):
                if r.get(k):
                    details.append(f"  {name} [{r['reader']}] {k}: {r[k]}")
    print(f"scenarios clean: {n_ok}/{len(by)}")
    if details:
        print("\n  -- details --")
        seen = set()
        for line in details:
            if line not in seen:
                seen.add(line)
                print(line)


def table_layout(rows):
    print("\n## LAYOUT (wall time, iter_batches)")
    by = {}
    for r in rows:
        by.setdefault(r["layout"], {})[r["reader"]] = r
    print(f"{'layout':16s} {'pyarrow':>9s} {'arrow_rs':>9s} {'rs/pa':>7s}  path(rs)")
    for lay, d in by.items():
        pa = d.get("pyarrow", {}).get("wall_s")
        rs = d.get("arrow_rs", {})
        rw = rs.get("wall_s")
        ratio = rw / pa if pa and rw else float("nan")
        print(
            f"{lay:16s} {pa:9.3f} {rw:9.3f} {ratio:7.2f}  "
            f"native={rs.get('native')} fallback={rs.get('fallback')}"
        )


def table_schema(rows):
    print("\n## SCHEMA (coverage: path taken + parity)")
    by = {}
    for r in rows:
        by.setdefault(r["schema"], {})[r["reader"]] = r
    print(
        f"{'schema':16s} {'expected':9s} {'path_taken':11s} {'parity':7s} "
        f"{'pa_wall':>8s} {'rs_wall':>8s}"
    )
    native_ok = 0
    total = 0
    for sc, d in by.items():
        rs = d.get("arrow_rs", {})
        nat, fb = rs.get("native", 0), rs.get("fallback", 0)
        taken = (
            "native"
            if nat and not fb
            else ("fallback" if fb and not nat else f"mix({nat}/{fb})")
        )
        exp = rs.get("expected")
        parity = rs.get("parity")
        gate_ok = taken == exp
        total += 1
        if gate_ok:
            native_ok += 1
        flag = "" if gate_ok else "  <-- GATE MISMATCH"
        print(
            f"{sc:16s} {exp:9s} {taken:11s} {str(parity):7s} "
            f"{d.get('pyarrow',{}).get('wall_s',0):8.3f} {rs.get('wall_s',0):8.3f}{flag}"
        )
    print(f"gate correct: {native_ok}/{total}")


def table_tuning(rows):
    print("\n## TUNING (decode_budget_bytes sweep, one_large_grp wide_str)")
    base = next((r["wall_s"] for r in rows if r["reader"] == "pyarrow"), None)
    print(f"{'budget':>10s} {'wall_s':>8s} {'vs pyarrow':>11s}")
    if base:
        print(f"{'pyarrow':>10s} {base:8.3f} {'1.00x':>11s}")
    for r in rows:
        if r["reader"] != "arrow_rs":
            continue
        ratio = r["wall_s"] / base if base else float("nan")
        print(f"{str(r['budget_mb'])+'MB':>10s} {r['wall_s']:8.3f} {ratio:10.2f}x")


def table_mixed(rows):
    print(
        "\n## MIXED (7 files: int/float/wide_str/large_str/huge_str + struct native, "
        "+ ray_tensor fallback, one dataset)"
    )
    for r in rows:
        peak = r.get("node_sum_peak_mb", 0)
        print(
            f"{r['reader']:9s} wall={r['wall_s']:.3f}s peak={peak:.0f}MB "
            f"rows={r['rows']} native={r['native']} fallback={r['fallback']}"
        )


def plot_mixed_time(rows):
    """One panel: node-sum USS over time for the 6-file heterogeneous dataset,
    both readers overlaid. arrow-rs routes the 5 flat files native + the struct
    file to PyArrow fallback, in one read — this shows the mixed dataset stays
    correct AND that the byte budget adapts across the different-bytes/row files."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG, exist_ok=True)
    colors = {"pyarrow": "#c0392b", "arrow_rs": "#2471a3"}
    fig, ax = plt.subplots(figsize=(7, 4.6))
    peaks = {}
    for r in rows:
        reader = r["reader"]
        if "t0" not in r:
            continue
        peaks[reader] = r.get("node_sum_peak_mb", 0)
        t, y = _node_sum_series_windowed(
            os.path.join(OUT, f"mixed__{reader}"), r["t0"], r["t1"]
        )
        if t is None:
            continue
        lbl = f"{reader} ({r['native']} native / {r['fallback']} fallback)"
        ax.step(t, y, where="post", color=colors[reader], lw=2.2, label=lbl)
    pr, rr = peaks.get("pyarrow", 0), peaks.get("arrow_rs", 0) or 1
    ratio = f"{pr / rr:.2f}x" if pr and rr else ""
    ax.set_title(
        "6 files, heterogeneous schema (int / float / wide_str / large_str /\n"
        f"huge_str + struct), one dataset — pa {pr:.0f} / rs {rr:.0f} MB ({ratio})",
        fontsize=10,
    )
    ax.set_xlabel("seconds")
    ax.set_ylabel("node-sum USS (MB)")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    out = os.path.join(FIG, "mixed_time.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  wrote {out}")


def table_leak(rows):
    print("\n## LEAK (8 repeats, same file, one session)")
    for r in rows:
        walls = [w["wall_s"] for w in r["windows"]]
        print(f"{r['reader']:9s} per-iter wall: " + " ".join(f"{w:.2f}" for w in walls))
    plot_leak(rows)


def _load_uss(run_dir):
    import numpy as np

    series = []
    for f in glob.glob(os.path.join(run_dir, "uss_*.csv")):
        import csv

        rr = list(csv.reader(open(f)))[1:]
        if not rr:
            continue
        ep = np.array([float(x[0]) for x in rr])
        uss = np.array([float(x[1]) for x in rr])
        series.append((ep, uss))
    return series


def _node_sum_series_windowed(run_dir, t0, t1, n=600, baseline=False):
    """node-sum USS (MB) vs relative time, restricted to the measured read window
    [t0, t1] (epoch seconds) so worker import / warm-up baseline is excluded.

    With baseline=True, each worker's USS at the start of its in-window life is
    subtracted first, so the curve is the EXTRA heap the read caused. That cancels
    the idle pre-started workers Ray spins up on a many-core node (each just holds a
    constant imported-lib heap) — without it the sum sits on a big flat plateau and
    the decode signal is invisible. baseline=False is the raw absolute node-sum."""
    import numpy as np

    series = _load_uss(run_dir)
    if not series:
        return None, None
    grid = np.linspace(t0, t1, n)
    total = np.zeros_like(grid)
    for ep, uss in series:
        idx = np.searchsorted(ep, grid, side="right") - 1
        held = np.where(idx >= 0, uss[np.clip(idx, 0, len(uss) - 1)], np.nan)
        alive = (grid >= ep.min()) & (grid <= ep.max())
        held = np.where(alive, held, np.nan)
        valid = ~np.isnan(held)
        if not valid.any():
            continue
        b = held[int(np.argmax(valid))] if baseline else 0.0
        total += np.where(valid, held - b, 0.0)
    return grid - t0, total / MB


_SHOWCASE = [
    ("small_many_rg", "many small row groups\n(Ray parallelizes → parity)"),
    ("medium_1rg", "one medium row group\n(~200 MB)"),
    ("large_1rg", "one large row group\n(~800 MB, the target case)"),
    ("mixed_rg", "mixed large+small groups\n(no O(n²) trap)"),
    ("many_files_1rg", "4 files x 1 big group\n(concurrent overcommit)"),
]


def _showcase_windows():
    rows = _load("showcase")
    win = {}
    for r in rows or []:
        win[(r["config"], r["reader"])] = (r["t0"], r["t1"], r["node_sum_peak_mb"])
    return win


def mem_time_showcase():
    """5-panel image: node-sum USS vs time, one panel per config, both readers,
    trimmed to the measured read window."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG, exist_ok=True)
    win = _showcase_windows()
    if not win:
        print("  (no showcase results — run: bench_suite.py showcase)")
        return
    colors = {"pyarrow": "#c0392b", "arrow_rs": "#2471a3"}
    fig, axes = plt.subplots(1, 5, figsize=(19, 4.3))
    for ax, (cfg, title) in zip(axes, _SHOWCASE):
        peaks = {}
        for reader in ("pyarrow", "arrow_rs"):
            if (cfg, reader) not in win:
                continue
            t0, t1, peak = win[(cfg, reader)]
            peaks[reader] = peak
            t, y = _node_sum_series_windowed(
                os.path.join(OUT, f"show__{cfg}__{reader}"), t0, t1
            )
            if t is None:
                continue
            ax.step(t, y, where="post", color=colors[reader], lw=2.2, label=reader)
        pr = peaks.get("pyarrow", 0)
        rr = peaks.get("arrow_rs", 1) or 1
        ratio = f"{pr / rr:.1f}x less" if pr and rr else ""
        ax.set_title(
            f"{title}\npyarrow {pr:.0f} / arrow_rs {rr:.0f} MB  ({ratio})", fontsize=9
        )
        ax.set_xlabel("seconds")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("node-sum USS (MB)")
    fig.suptitle(
        "Peak memory over time — where arrow-rs wins (and where it's just parity)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(FIG, "showcase_mem_time.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  wrote {out}")


def speed_time_showcase():
    """One image: cumulative rows delivered to the driver vs time, all 5 configs,
    both readers (solid=arrow_rs, dashed=pyarrow). Slope = end-to-end throughput."""
    import csv

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG, exist_ok=True)
    win = _showcase_windows()
    if not win:
        print("  (no showcase results — run: bench_suite.py showcase)")
        return
    cmap = plt.get_cmap("tab10")
    ls = {"arrow_rs": "-", "pyarrow": "--"}
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (cfg, _title) in enumerate(_SHOWCASE):
        color = cmap(i)
        for reader in ("arrow_rs", "pyarrow"):
            if (cfg, reader) not in win:
                continue
            t0 = win[(cfg, reader)][0]
            p = os.path.join(OUT, f"show__{cfg}__{reader}", "progress.csv")
            if not os.path.exists(p):
                continue
            rr = list(csv.reader(open(p)))[1:]
            if not rr:
                continue
            xs = [float(r[0]) - t0 for r in rr]
            ys = [float(r[1]) / 1e6 for r in rr]
            xs = [0.0] + xs
            ys = [0.0] + ys
            ax.plot(xs, ys, ls[reader], color=color, lw=2, label=f"{cfg} — {reader}")
    ax.set_xlabel("seconds")
    ax.set_ylabel("cumulative rows delivered (millions)")
    ax.set_title(
        "Speed over time: rows delivered to the consumer vs wall clock\n"
        "(steeper = faster; solid = arrow_rs, dashed = pyarrow)"
    )
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    fig.tight_layout()
    out = os.path.join(FIG, "showcase_speed_time.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  wrote {out}")


def plot_sweep(name, suptitle, xlabel_note=""):
    """5-panel memory-vs-time image for a one-variable sweep. One panel per level,
    both readers, trimmed to the measured window. Panels ordered as run."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG, exist_ok=True)
    rows = _load(f"sweep_{name}")
    if not rows:
        print(f"  (no sweep_{name} results — run: bench_suite.py sweep_{name})")
        return
    # Preserve run order of levels.
    levels = []
    for r in rows:
        if r["level"] not in levels:
            levels.append(r["level"])
    win = {
        (r["level"], r["reader"]): (r["t0"], r["t1"], r["node_sum_peak_mb"])
        for r in rows
    }
    colors = {"pyarrow": "#c0392b", "arrow_rs": "#2471a3"}
    n = len(levels)
    fig, axes = plt.subplots(1, n, figsize=(3.8 * n, 4.3))
    if n == 1:
        axes = [axes]
    for ax, level in zip(axes, levels):
        peaks = {}
        for reader in ("pyarrow", "arrow_rs"):
            if (level, reader) not in win:
                continue
            t0, t1, peak = win[(level, reader)]
            peaks[reader] = peak
            t, y = _node_sum_series_windowed(
                os.path.join(OUT, f"sweep_{name}__{level}__{reader}"), t0, t1
            )
            if t is None:
                continue
            ax.step(t, y, where="post", color=colors[reader], lw=2.2, label=reader)
        pr, rr = peaks.get("pyarrow", 0), peaks.get("arrow_rs", 0) or 1
        ratio = f"{pr / rr:.1f}x" if pr and rr else ""
        m = task_mem._meta(
            os.path.join(OUT, f"sweep_{name}__{level}__pyarrow")
        ) or task_mem._meta(os.path.join(OUT, f"sweep_{name}__{level}__arrow_rs"))
        note = ""
        if m.get("rows_total"):
            note = (
                f"\n{m['rows_total'] / 1e6:.1f}M rows · {m.get('num_row_groups', '?')} rg "
                f"· rg {m.get('max_rg_uncomp_mb', 0):.0f}MB decoded"
            )
        ax.set_title(
            f"{level}\npa {pr:.0f} / rs {rr:.0f} MB ({ratio}){note}", fontsize=8
        )
        ax.set_xlabel("seconds")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("node-sum USS (MB)")
    fig.suptitle(suptitle + (f"\n{xlabel_note}" if xlabel_note else ""), fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(FIG, f"sweep_{name}_mem_time.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  wrote {out}")


def table_workloads(rows):
    print("\n## WORKLOADS (decode-heavy, output-light: aggregation + selective filter)")
    print(
        f"{'workload':16s} {'reader':9s} {'wall_s':>8s} {'node-sum peak':>14s} {'path':>10s}"
    )
    for r in rows:
        path = (
            "native"
            if r.get("native") and not r.get("fallback")
            else (
                "fallback"
                if r.get("fallback") and not r.get("native")
                else f"{r.get('native',0)}/{r.get('fallback',0)}"
            )
        )
        print(
            f"{r['workload']:16s} {r['reader']:9s} {r['wall_s']:8.3f} "
            f"{r['node_sum_peak_mb']:11.0f}MB {path:>10s}"
        )


# NOTE: there are deliberately no per-axis memory BAR graphs (and none for
# scaling/schema/tuning, which never record USS traces). The memory metric of
# record is the per-task absolute-USS-over-time graph vs Ray's expectation
# line — one figure per config, both readers overlaid — produced by
# task_mem.py from the tasks_*.csv + uss_*.csv traces. Peak/incremental
# scalars were retired: windowed baseline subtraction systematically
# flattered whichever reader retains more warm heap (Agents.md §3.5), and
# single-number peaks invite selective reading.


def plot_leak(rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(FIG, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"pyarrow": "#c0392b", "arrow_rs": "#2471a3"}
    for r in rows:
        reader = r["reader"]
        rd = os.path.join(OUT, f"leak__{reader}")
        series = _load_uss(rd)
        wins = r["windows"]
        t0 = wins[0]["t_start"]
        grid = np.linspace(0, wins[-1]["t_end"] - t0, 1500)
        total = np.zeros_like(grid)
        for ep, uss in series:
            idx = np.searchsorted(ep - t0, grid, side="right") - 1
            total += np.where(idx >= 0, uss[np.clip(idx, 0, len(uss) - 1)], uss[0])
        ax.step(
            grid,
            total / MB,
            where="post",
            color=colors[reader],
            lw=2,
            label=f"{reader} node-sum USS",
        )
        for w in wins:
            ax.axvline(
                w["t_start"] - t0, color=colors[reader], ls=":", lw=0.5, alpha=0.4
            )
    ax.set_xlabel("seconds")
    ax.set_ylabel("node-sum USS (MB)")
    desc = _desc_for(
        os.path.join(OUT, "leak__pyarrow"), os.path.join(OUT, "leak__arrow_rs")
    )
    ax.set_title(
        "Leak check: 8 repeated reads of the same file (dotted = read starts).\n"
        + (desc + "\n" if desc else "")
        + "A flat floor between reads = no leak; a rising floor = ratchet."
    )
    ax.legend()
    ax.grid(alpha=0.2)
    out = os.path.join(FIG, "leak__uss.png")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"  wrote {out}")


# node-sum USS-over-time colors per (reader, pre_buffer) for the multi-group leak.
_LEAKMG_COLORS = {
    ("pyarrow", "on"): "#922b21",
    ("pyarrow", "off"): "#e74c3c",
    ("pyarrow_iter", "on"): "#6c3483",
    ("pyarrow_iter", "off"): "#bb8fce",
    ("arrow_rs", "na"): "#2471a3",
}
_LEAKMG_MODES = ["one_chunk", "per_group"]


def table_leak_multigrp(rows):
    ng = next((r.get("num_row_groups") for r in rows if r.get("num_row_groups")), "?")
    print(
        f"\n## LEAK (multi-row-group — arrow#39808 geometry: {ng} groups, 1 file)\n"
        "   one_chunk = all groups in 1 read task (scanner spans whole file); "
        "per_group = 1 task per group.\n"
        "   max_task_MB = the decoding worker's USS growth (the reader's working "
        "set); peak/incr include the ~340 MB warm base."
    )
    print(
        f"  {'chunk_mode':10s} {'reader':13s} {'pre_buf':7s} "
        f"{'max_task_MB':>11s} {'incr_USS_MB':>11s} {'peak_USS_MB':>11s} "
        f"{'grown_wk':>8s} {'wall_s':>7s}"
    )
    order = {m: i for i, m in enumerate(_LEAKMG_MODES)}
    reader_order = {"pyarrow": 0, "pyarrow_iter": 1, "arrow_rs": 2}

    def _key(r):
        return (
            order.get(r.get("chunk_mode"), 9),
            reader_order.get(r.get("reader"), 9),
            str(r.get("pre_buffer")),
        )

    for r in sorted(rows, key=_key):
        print(
            f"  {r.get('chunk_mode',''):10s} {r.get('reader',''):13s} "
            f"{str(r.get('pre_buffer','')):7s} "
            f"{(r.get('max_task_mb') or 0):11.1f} "
            f"{(r.get('node_sum_incr_mb') or 0):11.1f} "
            f"{(r.get('node_sum_peak_mb') or 0):11.1f} "
            f"{str(r.get('grown_workers', r.get('n_read_tasks',''))):>8s} "
            f"{(r.get('wall_s') or 0):7.2f}"
        )
    try:
        plot_leak_multigrp(rows)
    except Exception as e:
        print(f"  (skip leak_multigrp plot: {type(e).__name__}: {e})")


def plot_leak_multigrp(rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG, exist_ok=True)
    ng = next((r.get("num_row_groups") for r in rows if r.get("num_row_groups")), "?")
    mode_desc = {
        "one_chunk": f"all {ng} groups in 1 read task (scanner spans whole file)",
        "per_group": f"1 group per read task ({ng} tasks)",
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), squeeze=False, sharey=True)
    for ax, mode in zip(axes[0], _LEAKMG_MODES):
        mrows = [r for r in rows if r.get("chunk_mode") == mode]
        for r in mrows:
            pb = r.get("pre_buffer", "na")
            rd = os.path.join(OUT, f"leakmg__{mode}__{r['reader']}__pb_{pb}")
            grid, total = _node_sum_series_windowed(
                rd, r["t0"], r["t1"], baseline=False
            )
            if grid is None:
                continue
            lbl = r["reader"] + (f"  pre_buffer={pb}" if pb != "na" else "")
            peak = r.get("node_sum_peak_mb")
            if isinstance(peak, (int, float)):
                lbl += f"  (peak {peak:.0f} MB)"
            ax.plot(
                grid,
                total,
                lw=2,
                color=_LEAKMG_COLORS.get((r["reader"], pb), "#555555"),
                label=lbl,
            )
        ax.set_title(f"{mode}  —  {mode_desc.get(mode, '')}")
        ax.set_xlabel("seconds")
        ax.set_ylabel("node-sum USS (MB)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    desc = _desc_for(os.path.join(OUT, "leakmg__one_chunk__pyarrow__pb_on"))
    fig.suptitle(
        f"arrow#39808 leak geometry — {ng} small row groups.  LEFT: all groups in "
        "ONE read task (scanner spans the whole file).  RIGHT: one group per task "
        "(bounded).\n" + (desc or ""),
        fontsize=10,
    )
    out = os.path.join(FIG, "leak_multigrp__uss.png")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=120)
    print(f"  wrote {out}")


_LEAKRG_COLORS = {
    "pyarrow_v1": "#c0392b",  # legacy whole-file scanner (the #49158 surge)
    "pyarrow": "#e67e22",  # V2 scanner (per-row-group)
    "pyarrow_iter": "#8e44ad",  # V2 iter_batches
    "arrow_rs": "#2471a3",  # arrow-rs byte-budget streamer
}
_LEAKRG_GEOMS = ["many_tiny", "few_large"]
_LEAKRG_READER_ORDER = {"pyarrow_v1": 0, "pyarrow": 1, "pyarrow_iter": 2, "arrow_rs": 3}
_LEAKRG_GEOM_DESC = {
    "many_tiny": "~200 tiny row groups (huge cells) — churn / allocator-retention case",
    "few_large": "4 big (~200 MB) row groups — the decode-floor case (arrow-rs's win)",
}


def table_leak_rgsize(rows):
    print(
        "\n## LEAK by ROW-GROUP SIZE (ray#49158 shape: few rows, huge binary cells)\n"
        "   pyarrow_v1 = legacy whole-file scanner (the surge); pyarrow/iter/arrow_rs "
        "= V2 (per-group).\n"
        "   many_tiny = ~200 tiny groups (churn); few_large = 4 big groups (decode "
        "floor).\n"
        "   max_task_MB = busiest worker's USS growth = the reader's working set."
    )
    print(
        f"  {'geom':10s} {'reader':13s} {'#rg':>4s} {'max_task_MB':>11s} "
        f"{'incr_USS_MB':>11s} {'peak_USS_MB':>11s} {'nat/fb':>7s} {'wall_s':>7s}"
    )

    def _key(r):
        return (
            _LEAKRG_GEOMS.index(r["geom"]) if r.get("geom") in _LEAKRG_GEOMS else 9,
            _LEAKRG_READER_ORDER.get(r.get("reader"), 9),
        )

    for r in sorted(rows, key=_key):
        print(
            f"  {r.get('geom',''):10s} {r.get('reader',''):13s} "
            f"{str(r.get('num_row_groups','?')):>4s} "
            f"{(r.get('max_task_mb') or 0):11.1f} "
            f"{(r.get('node_sum_incr_mb') or 0):11.1f} "
            f"{(r.get('node_sum_peak_mb') or 0):11.1f} "
            f"{str(r.get('native',0)) + '/' + str(r.get('fallback',0)):>7s} "
            f"{(r.get('wall_s') or 0):7.2f}"
        )
    try:
        plot_leak_rgsize(rows)
    except Exception as e:
        print(f"  (skip leak_rgsize plot: {type(e).__name__}: {e})")


def plot_leak_rgsize(rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG, exist_ok=True)
    geoms = [g for g in _LEAKRG_GEOMS if any(r.get("geom") == g for r in rows)]
    fig, axes = plt.subplots(
        1, len(geoms), figsize=(7 * len(geoms), 5.4), squeeze=False, sharey=False
    )
    for ax, geom in zip(axes[0], geoms):
        grows = [r for r in rows if r.get("geom") == geom]
        ng = next(
            (r.get("num_row_groups") for r in grows if r.get("num_row_groups")), "?"
        )
        for r in sorted(grows, key=lambda x: _LEAKRG_READER_ORDER.get(x["reader"], 9)):
            rd = os.path.join(OUT, f"leakrg__{geom}__{r['reader']}")
            grid, total = _node_sum_series_windowed(
                rd, r["t0"], r["t1"], baseline=False
            )
            if grid is None:
                continue
            mt = r.get("max_task_mb")
            lbl = r["reader"]
            if isinstance(mt, (int, float)):
                lbl += f"  (max task {mt:.0f} MB)"
            ax.plot(
                grid,
                total,
                lw=2,
                color=_LEAKRG_COLORS.get(r["reader"], "#555555"),
                label=lbl,
            )
        ax.set_title(
            f"{geom} — {_LEAKRG_GEOM_DESC.get(geom, '')} ({ng} groups)", fontsize=9
        )
        ax.set_xlabel("seconds")
        ax.set_ylabel("node-sum USS (MB)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(
        "ray#49158 by row-group size — V1 whole-file scanner SURGES; V2 per-group "
        "stays bounded.\nLEFT: many tiny groups (watch arrow-rs allocator retention; "
        "re-run with MALLOC_ARENA_MAX=2 on Linux).  RIGHT: few big groups "
        "(arrow-rs's byte budget beats the whole-group decode).",
        fontsize=9,
    )
    out = os.path.join(FIG, "leak_rgsize__uss.png")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out, dpi=120)
    print(f"  wrote {out}")


_RKNOB_READER_COLOR = {
    "pyarrow": "#c0392b",
    "pyarrow_iter": "#8e44ad",
    "arrow_rs": "#2471a3",
}
_RKNOB_READER_ORDER = {"pyarrow": 0, "pyarrow_iter": 1, "arrow_rs": 2}


def table_reader_settings(rows):
    print(
        "\n## READER SETTINGS (one_chunk leak geometry: 30 groups in 1 read task)\n"
        "   how each reader knob moves the decoding worker's USS growth (max_task_MB)."
    )
    print(
        f"  {'reader':13s} {'setting':22s} {'max_task_MB':>11s} "
        f"{'peak_USS_MB':>11s} {'wall_s':>7s}"
    )

    def _key(r):
        return (_RKNOB_READER_ORDER.get(r.get("reader"), 9), r.get("setting", ""))

    for r in sorted(rows, key=_key):
        print(
            f"  {r.get('reader',''):13s} {r.get('setting',''):22s} "
            f"{(r.get('max_task_mb') or 0):11.1f} "
            f"{(r.get('node_sum_peak_mb') or 0):11.1f} {(r.get('wall_s') or 0):7.2f}"
        )
    try:
        plot_reader_settings(rows)
    except Exception as e:
        print(f"  (skip reader_settings plot: {type(e).__name__}: {e})")


def plot_reader_settings(rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG, exist_ok=True)
    ordered = sorted(
        rows,
        key=lambda r: (
            _RKNOB_READER_ORDER.get(r.get("reader"), 9),
            r.get("setting", ""),
        ),
    )
    labels = [f"{r['reader']}: {r['setting']}" for r in ordered]
    vals = [(r.get("max_task_mb") or 0) for r in ordered]
    colors = [_RKNOB_READER_COLOR.get(r.get("reader"), "#555555") for r in ordered]
    y = list(range(len(ordered)))[::-1]  # first config at top
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(ordered) + 2))
    ax.barh(y, vals, color=colors)
    for yi, v in zip(y, vals):
        ax.text(v + max(vals) * 0.01, yi, f"{v:.0f}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("decoding worker's USS growth — max_task (MB)")
    ng = next((r.get("num_row_groups") for r in rows if r.get("num_row_groups")), "?")
    ax.set_title(
        f"Reader-setting sweep — {ng} row groups in ONE read task.\n"
        "One knob varied per bar off its default; lower = smaller decode working set."
    )
    ax.grid(alpha=0.2, axis="x")
    out = os.path.join(FIG, "reader_settings__max_task.png")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"  wrote {out}")


def plot_scaling(rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG, exist_ok=True)
    by = {}
    for r in rows:
        by.setdefault(r["reader"], []).append(r)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = {"pyarrow": "#c0392b", "arrow_rs": "#2471a3"}
    for reader, rs in by.items():
        rs = sorted(rs, key=lambda x: x["rows"])
        xs = [x["rows"] / 1e6 for x in rs]
        ax1.plot(
            xs, [x["wall_s"] for x in rs], "o-", color=colors[reader], label=reader
        )
        ax2.plot(
            xs, [x["us_per_row"] for x in rs], "o-", color=colors[reader], label=reader
        )
    ax1.set_xlabel("rows (millions)")
    ax1.set_ylabel("wall (s)")
    ax1.set_title("Wall time vs size (one big row group)")
    ax1.legend()
    ax1.grid(alpha=0.2)
    ax2.set_xlabel("rows (millions)")
    ax2.set_ylabel("µs / row")
    ax2.set_title("µs/row — flat/down = O(n); up = O(n²)")
    ax2.legend()
    ax2.grid(alpha=0.2)
    out = os.path.join(FIG, "scaling.png")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  wrote {out}")


def plot_tuning(rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG, exist_ok=True)
    base = next((r["wall_s"] for r in rows if r["reader"] == "pyarrow"), None)
    pts = sorted(
        [r for r in rows if r["reader"] == "arrow_rs"], key=lambda x: x["budget_mb"]
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        [p["budget_mb"] for p in pts],
        [p["wall_s"] for p in pts],
        "o-",
        color="#2471a3",
        label="arrow-rs",
    )
    if base:
        ax.axhline(base, color="#c0392b", ls="--", label=f"pyarrow ({base:.2f}s)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("decode_budget_bytes (MB, log2)")
    ax.set_ylabel("wall (s)")
    ax.set_title("Budget tuning (one big row group, iter_batches)")
    ax.legend()
    ax.grid(alpha=0.2)
    out = os.path.join(FIG, "tuning.png")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  wrote {out}")


def table_scaling(rows):
    print("\n## SCALING (one big row group, wall/row — flat = O(n), rising = O(n^2))")
    by = {}
    for r in rows:
        by.setdefault(r["rows"], {})[r["reader"]] = r
    print(f"{'rows':>10s} {'pa us/row':>10s} {'rs us/row':>10s} {'rs/pa wall':>11s}")
    for n, d in sorted(by.items()):
        pa = d.get("pyarrow", {})
        rs = d.get("arrow_rs", {})
        ratio = rs.get("wall_s", 0) / pa["wall_s"] if pa.get("wall_s") else float("nan")
        print(
            f"{n:>10d} {pa.get('us_per_row', 0):10.4f} {rs.get('us_per_row', 0):10.4f} "
            f"{ratio:11.2f}"
        )


def plot_concurrency(fixture="big_4x4M", ncpu=4):
    """Overlay node-sum USS over time for both readers on the big concurrent
    fixture — the single-node overcommit, made visual."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(FIG, exist_ok=True)
    colors = {"pyarrow": "#c0392b", "arrow_rs": "#2471a3"}
    fig, ax = plt.subplots(figsize=(10, 5))
    for reader in ["pyarrow", "arrow_rs"]:
        rd = os.path.join(OUT, f"conc__{fixture}__{reader}__cpu{ncpu}")
        series = _load_uss(rd)
        if not series:
            continue
        t0 = min(ep.min() for ep, _ in series)
        t1 = max(ep.max() for ep, _ in series)
        grid = np.linspace(0, t1 - t0, 1500)
        total = np.zeros_like(grid)
        for ep, uss in series:
            idx = np.searchsorted(ep - t0, grid, side="right") - 1
            total += np.where(idx >= 0, uss[np.clip(idx, 0, len(uss) - 1)], 0.0)
        ax.step(
            grid,
            total / MB,
            where="post",
            color=colors[reader],
            lw=2.2,
            label=f"{reader} node-sum USS (peak {total.max()/MB:.0f}MB)",
        )
    ax.set_xlabel("seconds")
    ax.set_ylabel("node-sum USS across all workers (MB)")
    desc = _desc_for(
        os.path.join(OUT, f"conc__{fixture}__pyarrow__cpu{ncpu}"),
        os.path.join(OUT, f"conc__{fixture}__arrow_rs__cpu{ncpu}"),
    )
    ax.set_title(
        f"Single-node overcommit: {ncpu} workers x big row groups ({fixture}).\n"
        + (desc + "\n" if desc else "")
        + "Node-sum private heap = physical RAM the concurrent decodes occupy.",
        fontsize=10,
    )
    ax.legend()
    ax.grid(alpha=0.2)
    out = os.path.join(FIG, f"concurrency__{fixture}__cpu{ncpu}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"  wrote {out}")


def table_concurrency(rows):
    print("\n## CONCURRENCY (N files x 1 big row group, K workers on one node)")
    print(
        f"{'fixture':12s} {'reader':9s} {'cpus':>4s} {'wall_s':>8s} "
        f"{'node-sum peak USS':>18s}"
    )
    # Group by fixture; within each, show rs/pa memory ratio at matched cpus.
    from collections import defaultdict

    byf = defaultdict(list)
    for r in rows:
        byf[r.get("fixture", "?")].append(r)
    for fx_name, rs in byf.items():
        for r in sorted(rs, key=lambda x: (x["reader"], x["num_cpus"])):
            print(
                f"{fx_name:12s} {r['reader']:9s} {r['num_cpus']:>4d} {r['wall_s']:8.3f} "
                f"{r['node_sum_peak_mb']:15.0f}MB"
            )
        # memory ratio at cpu=4
        pa4 = next(
            (
                x["node_sum_peak_mb"]
                for x in rs
                if x["reader"] == "pyarrow" and x["num_cpus"] == 4
            ),
            None,
        )
        rs4 = next(
            (
                x["node_sum_peak_mb"]
                for x in rs
                if x["reader"] == "arrow_rs" and x["num_cpus"] == 4
            ),
            None,
        )
        if pa4 and rs4:
            print(
                f"  -> {fx_name} @cpu4 node-sum peak: pyarrow {pa4:.0f}MB / arrow_rs "
                f"{rs4:.0f}MB = {pa4/rs4:.2f}x"
            )
    plot_concurrency()


def table_s3(rows):
    """Wall summary for the S3 sweep. Memory verdicts live in the per-task
    graphs (task_mem.py); this table keeps wall time + the raw absolute
    node-sum peak as node-level sanity only."""
    print(
        "\n## S3 (real bucket): PyArrow baseline vs arrow-rs (window + budget + allocator sweep)"
    )
    if not rows:
        print("  (no s3 results)")
        return
    base = next((r for r in rows if r["reader"] == "pyarrow"), None)
    print(
        f"{'config':24s} {'wall_s':>8s} {'abs peak':>10s} "
        f"{'wall vs pa':>11s} {'path':>9s}"
    )
    for r in rows:
        path = f"{r.get('native', 0)}/{r.get('fallback', 0)}"
        wallr = ""
        if base and r["reader"] == "arrow_rs":
            wallr = f"{r['wall_s'] / (base['wall_s'] or 1):.2f}x"
        print(
            f"{r.get('tag', r['reader']):24s} {r['wall_s']:8.2f} "
            f"{r['node_sum_peak_mb']:7.0f}MB {wallr:>11s} {path:>9s}"
        )
    if base:
        print(
            "  (wall vs pa = arrow_rs/pyarrow; ~1.0 ⇒ speed parity, the bar. "
            "Memory: see figs/task_mem/ — per-task absolute USS vs Ray's "
            "expectation line, no baseline subtraction.)"
        )


def table_s3geom(rows):
    """Per-geometry PyArrow vs arrow-rs (windowed) on REAL S3 — the option-B
    sweep. Wall + absolute node-sum peak per geometry; the memory verdict of
    record is the per-task graph (figs/task_mem/s3geom__*.png)."""
    print(
        "\n## S3 GEOMETRY SWEEP (real bucket): per-geometry PyArrow vs arrow-rs (windowed)"
    )
    if not rows:
        print("  (no s3_geom results)")
        return
    by = {}
    for r in rows:
        by.setdefault(r["config"], {})[r["reader"]] = r
    print(
        f"{'geometry':16s} {'reader':9s} {'wall_s':>8s} {'abs peak':>10s} "
        f"{'mem vs pa':>10s} {'path':>8s}"
    )
    for geom, d in by.items():
        pa_r = d.get("pyarrow")
        for reader in ("pyarrow", "arrow_rs"):
            r = d.get(reader)
            if not r:
                continue
            memr = ""
            if pa_r and reader == "arrow_rs" and r.get("node_sum_peak_mb"):
                memr = f"{pa_r['node_sum_peak_mb'] / (r['node_sum_peak_mb'] or 1):.2f}x"
            path = f"{r.get('native', 0)}/{r.get('fallback', 0)}"
            print(
                f"{geom:16s} {reader:9s} {r['wall_s']:8.2f} "
                f"{r['node_sum_peak_mb']:7.0f}MB {memr:>10s} {path:>8s}"
            )
    print(
        "  (mem vs pa = pyarrow/arrow_rs node-sum peak; >1 ⇒ arrow-rs uses less. "
        "Per-task memory graphs: figs/task_mem/s3geom__*.png)"
    )


def plot_s3():
    """Two overlaid S3 figures from runs/results_s3.json + the per-run traces:
    * s3_mem_time.png   — one panel per arrow-rs config, PyArrow baseline overlaid
                          (node-sum USS vs time, trimmed to the measured window).
    * s3_speed_time.png — cumulative rows delivered vs time, every config on one
                          axis (dashed = PyArrow baseline).
    """
    import csv as _csv

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG, exist_ok=True)
    rows = _load("s3")
    if not rows:
        print("  (no s3 results — run: python run_s3_benchmark.py)")
        return
    base = next((r for r in rows if r["reader"] == "pyarrow"), None)
    configs = [r for r in rows if r["reader"] == "arrow_rs"]
    if base is None or not configs:
        print("  (s3 results need a pyarrow baseline + arrow_rs configs)")
        return
    colors = {"pyarrow": "#c0392b", "arrow_rs": "#2471a3"}

    # --- memory over time: one panel per arrow_rs config, PyArrow overlaid.
    # ABSOLUTE node-sum USS (baseline=False): the OOM killer sees absolute
    # memory, so no per-worker warm-baseline subtraction (retired — it
    # systematically flattered whichever reader retains more warm heap).
    # Per-task graphs (figs/task_mem/) are the metric of record; this panel is
    # the node-level view of the same traces.
    bt, by = _node_sum_series_windowed(
        os.path.join(OUT, "s3__pyarrow"), base["t0"], base["t1"]
    )
    n = len(configs)
    fig, axes = plt.subplots(1, n, figsize=(3.8 * n, 4.4), squeeze=False)
    for ax, cfg in zip(axes[0], configs):
        if bt is not None:
            ax.step(
                bt,
                by,
                where="post",
                color=colors["pyarrow"],
                lw=2.2,
                label=f"pyarrow {base['node_sum_peak_mb']:.0f}MB",
            )
        ct, cy = _node_sum_series_windowed(
            os.path.join(OUT, f"s3__{cfg['tag']}"), cfg["t0"], cfg["t1"]
        )
        if ct is not None:
            ax.step(
                ct,
                cy,
                where="post",
                color=colors["arrow_rs"],
                lw=2.2,
                label=f"arrow_rs {cfg['node_sum_peak_mb']:.0f}MB",
            )
        w, b = cfg.get("fetch_window_mb"), cfg.get("budget_mb")
        wl = "no-cap" if not w else f"{w}MB"
        bl = f" bud{b}" if b else ""
        al = (
            f" {cfg.get('alloc')}" if cfg.get("alloc") and cfg["alloc"] != "sys" else ""
        )
        ax.set_title(
            f"win={wl}{bl}{al}\n" f"wall {cfg['wall_s'] / (base['wall_s'] or 1):.2f}x",
            fontsize=8,
        )
        ax.set_xlabel("seconds")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, loc="upper left")
    axes[0][0].set_ylabel("node-sum USS (MB, absolute)")
    fig.suptitle(
        "S3 memory-over-time — PyArrow vs arrow-rs (fetch-window + allocator sweep)\n"
        "absolute node-sum private heap; per-task view in figs/task_mem/; wall ~1.0x = speed parity (the bar)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    out = os.path.join(FIG, "s3_mem_time.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  wrote {out}")

    # --- speed over time: cumulative rows delivered, all configs on one axis ---
    t0map = {r.get("tag", r["reader"]): r["t0"] for r in rows}
    ordered = [("pyarrow", base)] + [(c["tag"], c) for c in configs]
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (tag, r) in enumerate(ordered):
        p = os.path.join(OUT, f"s3__{tag}", "progress.csv")
        if not os.path.exists(p):
            continue
        rr = list(_csv.reader(open(p)))[1:]
        if not rr:
            continue
        t0 = t0map.get(tag, float(rr[0][0]))
        xs = [float(x[0]) - t0 for x in rr]
        ys = [float(x[1]) for x in rr]
        if r["reader"] == "pyarrow":
            style, col = "--", colors["pyarrow"]
        else:
            style = "-"
            col = plt.cm.viridis(0.15 + 0.7 * i / max(1, len(ordered)))
        ax.plot(xs, ys, style, color=col, lw=2, label=tag)
    ax.set_xlabel("seconds")
    ax.set_ylabel("cumulative rows delivered to driver")
    ax.set_title(
        "S3 throughput over time — cumulative rows vs wall clock\n"
        "(dashed = PyArrow baseline; solid = arrow-rs configs)"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out = os.path.join(FIG, "s3_speed_time.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  wrote {out}")


def _run():
    for axis, fn in [
        ("correctness", table_correctness),
        ("layout", table_layout),
        ("schema", table_schema),
        ("tuning", table_tuning),
        ("mixed", table_mixed),
        ("scaling", table_scaling),
        ("concurrency", table_concurrency),
        ("leak", table_leak),
        ("leak_multigrp", table_leak_multigrp),
        ("leak_rgsize", table_leak_rgsize),
        ("reader_settings", table_reader_settings),
        ("workloads", table_workloads),
        ("s3", table_s3),
        ("s3geom", table_s3geom),
    ]:
        rows = _load(axis)
        if rows is not None:
            fn(rows)
    mixed_rows = _load("mixed")
    if mixed_rows is not None:
        try:
            plot_mixed_time(mixed_rows)
        except Exception as e:
            print(f"  (skip mixed_time: {type(e).__name__}: {e})")

    # THE memory graphs: per-task absolute USS over time vs Ray's expectation
    # line, one figure per config, both readers overlaid (task_mem.py). Built
    # from the tasks_*.csv + uss_*.csv already on disk (no re-run needed).
    print(
        f"\n## MEMORY GRAPHS (per-task USS vs ideal streaming reader -> {FIG}/task_mem/)"
    )
    try:
        task_mem.FIG = os.path.join(FIG, "task_mem")  # same per-run dir
        task_mem.main()
    except Exception as e:
        print(f"  (skip task_mem: {type(e).__name__}: {e})")

    print("\n## SUPPORTING GRAPHS")
    for name, fn in [
        ("time_showcase", mem_time_showcase),
        ("speed_showcase", speed_time_showcase),
        ("s3", plot_s3),
    ]:
        try:
            fn()
        except Exception as e:  # a missing axis run shouldn't kill the rest
            print(f"  (skip {name}: {type(e).__name__}: {e})")

    # One-variable sweeps: 5-panel memory-vs-time, everything else fixed.
    sweeps = [
        (
            "size",
            "Memory over time — flat int64 table, one big row group, "
            "5 sizes (~14 MB → ~1.4 GB)",
        ),
        ("schema", "Memory over time — 2 M rows, one big row group, 5 column dtypes"),
        (
            "rowgroup",
            "Memory over time — same 400 MB, chopped into 5 row-group layouts "
            "(many tiny → one whole-file group)",
        ),
        (
            "files",
            "Node-sum memory over time — 1 big row group per file, "
            "5 file counts read across 4 workers (the overcommit)",
        ),
        (
            "batch",
            "Memory over time — same 400 MB group, 5 arrow-rs budgets "
            "(iter_batches: retained blocks set the floor, budget barely moves it)",
        ),
        (
            "batch_dd",
            "Memory over time — same 400 MB group, 5 budgets in decode_drop "
            "(K=1 reads the whole group first, so budget barely moves the floor)",
        ),
    ]
    for name, title in sweeps:
        try:
            plot_sweep(name, title)
        except Exception as e:
            print(f"  (skip sweep_{name}: {type(e).__name__}: {e})")


def main():
    global FIG
    # A fresh figs/<timestamp>/ per run so figures never overwrite a prior run's;
    # figs/latest points at it for quick opening.
    run_id = time.strftime("%Y%m%d_%H%M%S")
    FIG = os.path.join(HERE, "figs", run_id)
    os.makedirs(FIG, exist_ok=True)
    task_mem._point_latest(FIG)

    # Save the entire digest to figs/<ts>/summary.txt (all tables + the per-task
    # overshoot lines), teed while still echoing to the terminal.
    summary_path = os.path.join(FIG, "summary.txt")
    real_stdout = sys.stdout
    fh = open(summary_path, "w")
    sys.stdout = _Tee(real_stdout, fh)
    try:
        print(f"===== figures -> {FIG}  (also figs/latest) =====")
        _run()
    except Exception as e:
        # A plotting failure mid-digest (e.g. matplotlib absent locally) must not
        # discard the tables already written — record it and still finalize.
        print(f"\n(digest aborted mid-run: {type(e).__name__}: {e})")
    finally:
        sys.stdout = real_stdout
        fh.close()

    # Refresh runs/summary.csv (machine-readable, one row per read) from the same
    # results_*.json, so it's regenerated on a plain `summarize.py` run too — not
    # only at the end of a bench_suite run.
    try:
        import bench_suite

        bench_suite.write_summary_csv()
    except Exception as e:
        print(f"(skip summary.csv refresh: {type(e).__name__}: {e})")
    print(f"\n===== text digest -> {summary_path} =====")
    print(f"===== machine csv -> {os.path.join(OUT, 'summary.csv')} =====")


if __name__ == "__main__":
    main()
