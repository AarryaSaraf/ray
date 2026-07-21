"""arrow-rs vs PyArrow benchmark suite — the macOS-decisive axes.

Runs the axes whose verdict does NOT depend on absolute memory magnitude (which
is only directional on macOS because USS excludes shared pages). Those are:

  layout   — #1  wall time across the 5 file/row-group shapes (both readers)
  schema   — #2  path taken (native vs PyArrow fallback) + per-column parity,
                  across int/float/str/struct/list/tensor (coverage %)
  tuning   — #3  sweep decode_budget_bytes on one big single-row-group file
  leak     — #8  read the same file N times in ONE session; USS must return to
                  its floor each iteration (no ratchet)
  mixed    — #9  5 files, each a different native schema (narrow ints ... fat
                  strings) in one dataset — does the per-group byte budget adapt

Concurrency (#7) and authoritative memory are deferred to the Linux/S3 box.

Per-run artifacts land in runs/<label>/ (uss_<pid>.csv from the setup hook,
path_<pid>.log from the reader trace). Aggregate results -> results_<axis>.json.
Usage:  python bench_suite.py <axis>[,<axis>...]      (default: all)
"""
import glob
import hashlib
import json
import os
import shutil
import sys
import time

import pyarrow as pa

import ray
import fixtures as fx

HERE = os.path.dirname(os.path.abspath(__file__))
HOOKDIR = os.path.join(HERE, "hookdir")
OUT = os.path.join(HERE, "runs")
MB = 1024 * 1024

WARM = {"rows": 10_000, "num_files": 1, "row_group_size": 10_000, "schema": "int"}


def _fresh_session(reader, trace_dir, budget_bytes=8 * MB, k=1, num_cpus=4,
                   fetch_window_mb=16, malloc_arena_max=None, ld_preload=None):
    ray.shutdown()
    # Let the allocator levers be flipped for the WHOLE suite from the environment,
    # so an axis that doesn't thread them through (e.g. layout) can still be A/B'd
    # against the uncapped system allocator without a code edit. Explicit args win.
    if malloc_arena_max is None:
        malloc_arena_max = os.environ.get("RAY_DATA_ARROW_RS_MALLOC_ARENA_MAX")
    if ld_preload is None:
        ld_preload = os.environ.get("RAY_DATA_ARROW_RS_LD_PRELOAD")
    env_vars = {
        "RAY_MEM_TRACE_DIR": trace_dir,
        "RAY_MEM_TRACE_INTERVAL_S": "0.005",
        "RAY_DATA_ARROW_RS_PATH_TRACE": trace_dir,
        "RAY_DATA_USE_DATASOURCE_V2": "1",
        "RAY_DATA_USE_ARROW_RS_PARQUET_READER": "1" if reader == "arrow_rs" else "0",
        "RAY_DATA_ARROW_RS_K": str(k),
        "RAY_DATA_ARROW_RS_DECODE_BUDGET_BYTES": str(budget_bytes),
        "RAY_DATA_ARROW_RS_FETCH_WINDOW_MB": str(fetch_window_mb),
    }
    # Cap glibc's per-thread arenas in the WORKER processes (must be set before the
    # worker starts, hence via runtime_env, not os.environ). This is the no-code,
    # zero-segfault-risk memory-parity lever for the default (system-allocator)
    # build on Linux: without it ptmalloc retains freed chunks across many arenas
    # and node-sum peak reads high vs PyArrow's jemalloc (Agents.md §7.8). Ignored
    # on macOS. `None` leaves the env untouched (the uncapped baseline).
    if malloc_arena_max is not None:
        env_vars["MALLOC_ARENA_MAX"] = str(malloc_arena_max)
    # Swap the allocator for the WHOLE worker process (Python + Rust + PyArrow) with
    # no recompile (#9): point LD_PRELOAD at a libmimalloc/libjemalloc .so. Unlike the
    # crate's compile-time `mimalloc` feature (which segfaults under Ray workers — a
    # global_allocator in a cdylib fights the host process), LD_PRELOAD interposes
    # malloc process-wide and sidesteps that. Linux-only; `None` = the ambient
    # (system glibc) allocator. See Agents.md §7.8 for how to find the .so path.
    if ld_preload is not None:
        env_vars["LD_PRELOAD"] = str(ld_preload)
    runtime_env = {
        "working_dir": HOOKDIR,
        "worker_process_setup_hook": "worker_mem_sampler.setup",
        "env_vars": env_vars,
    }
    # If a cluster is already running (Anyscale workspace: RAY_ADDRESS set), ATTACH
    # to it. Do NOT start a private local cluster on the workspace — address="local"
    # spins up a fresh dashboard agent that imports Ray's bundled dashboard modules
    # (including a test file) and dies on a missing pytest, spraying tracebacks.
    # Attaching sidesteps that. You may not pass num_cpus when attaching, so
    # concurrency becomes the node's core count; that only weakens the num_cpus-
    # varying axes (concurrency/files) — incremental USS (_node_sum_incr_peak_mb)
    # still isolates the workers that actually decoded from the shared cluster's
    # idle ones. On a PLAIN box (no cluster running) start an isolated local cluster
    # with num_cpus pinned, so those axes' knob does take effect there.
    cluster_running = bool(os.environ.get("RAY_ADDRESS")) or os.path.exists(
        "/tmp/ray/ray_current_cluster"
    )
    if cluster_running:
        ray.init(address="auto", ignore_reinit_error=True, log_to_driver=False,
                 runtime_env=runtime_env)
    else:
        ray.init(num_cpus=num_cpus, include_dashboard=False,
                 ignore_reinit_error=True, log_to_driver=False,
                 runtime_env=runtime_env)
    from ray.data.context import DataContext
    ctx = DataContext.get_current()
    ctx.use_datasource_v2 = True
    ctx.use_arrow_rs_parquet_reader = (reader == "arrow_rs")
    ctx.execution_options.preserve_order = True  # so parity hashes are comparable
    return ctx


def consume(ds, mode, progress_path=None):
    """Consume the dataset. If `progress_path` is given, log (epoch, cum_rows)
    as each batch arrives at the driver — an end-to-end throughput trace
    ("speed vs time") comparable across readers."""
    fh = open(progress_path, "w") if progress_path else None
    if fh:
        fh.write("epoch,cum_rows\n")
    try:
        if mode == "decode_drop":
            def _touch(b):
                return {"n": [b.num_rows]}
            total = 0
            for out in ds.map_batches(_touch, batch_format="pyarrow").iter_batches(
                    batch_format="pyarrow"):
                total += int(sum(out["n"].to_pylist()))
                if fh:
                    fh.write(f"{time.time()},{total}\n")
            return total
        n = 0
        for b in ds.iter_batches(batch_format="pyarrow"):
            n += b.num_rows
            if fh:
                fh.write(f"{time.time()},{n}\n")
        return n
    finally:
        if fh:
            fh.close()


def consume_hash(ds):
    """Batching-invariant per-column hash: accumulate to one table, hash each
    column's IPC bytes over the whole column (so arrow-rs's byte-budget batches
    and PyArrow's batches hash equal when the data matches)."""
    tbl = pa.concat_tables(list(ds.iter_batches(batch_format="pyarrow")))
    tbl = tbl.combine_chunks()
    out = {"__rows__": tbl.num_rows}
    for name in tbl.schema.names:
        rb = pa.record_batch([tbl.column(name).combine_chunks()], [name])
        sink = pa.BufferOutputStream()
        w = pa.ipc.new_stream(sink, rb.schema)
        w.write_batch(rb)
        w.close()
        out[name] = hashlib.blake2b(sink.getvalue().to_pybytes(),
                                    digest_size=16).hexdigest()
    return out


def _count_paths(trace_dir):
    native = fallback = 0
    for f in glob.glob(os.path.join(trace_dir, "path_*.log")):
        for line in open(f):
            if line.strip() == "native":
                native += 1
            elif line.strip() == "fallback":
                fallback += 1
    return native, fallback


def _run_dir(label):
    d = os.path.join(OUT, label)
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d)
    return d


def _warm(reader, trace_dir):
    fx.make_fixture("_warm", WARM)
    consume(ray.data.read_parquet(fx.make_fixture("_warm", WARM)), "iter_batches")
    time.sleep(0.5)
    # Drop the warmup's path-trace lines so path counts reflect only the measured
    # read (the USS csvs are kept — the measured window is selected by timestamp).
    for f in glob.glob(os.path.join(trace_dir, "path_*.log")):
        try:
            os.remove(f)
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# Axes
# --------------------------------------------------------------------------- #
def axis_layout():
    S = "wide_str"
    layouts = {
        "small_1grp":      {"rows": 200_000, "num_files": 1, "row_group_size": 200_000, "schema": S},
        "small_many_grp":  {"rows": 200_000, "num_files": 1, "row_group_size": 20_000, "schema": S},
        "one_large_grp":   {"rows": 2_000_000, "num_files": 1, "row_group_size": 2_000_000, "schema": S},
        "many_large_grp":  {"rows": 2_000_000, "num_files": 1, "row_group_size": 250_000, "schema": S},
        "mixed_grp":       {"rows": 2_000_000, "num_files": 1,
                            "row_group_sizes": [500_000, 20_000, 500_000, 20_000], "schema": S},
    }
    results = []
    for name, spec in layouts.items():
        path = fx.make_fixture(name, spec)
        for reader in ["pyarrow", "arrow_rs"]:
            for mode in ["iter_batches"]:
                label = f"layout__{name}__{mode}__{reader}"
                d = _run_dir(label)
                _fresh_session(reader, d)
                _warm(reader, d)
                t0 = time.time()
                rows = consume(ray.data.read_parquet(path), mode)
                t1 = time.time(); wall = t1 - t0
                ray.shutdown(); time.sleep(0.3)
                peak = _node_sum_peak_mb(d, t0, t1)
                incr = _node_sum_incr_peak_mb(d, t0, t1)
                nat, fb = _count_paths(d)
                wk = _worker_breakdown(d, t0, t1)
                results.append({"layout": name, "reader": reader, "mode": mode,
                                "wall_s": wall, "rows": rows, "t0": t0, "t1": t1,
                                "node_sum_peak_mb": peak, "node_sum_incr_mb": incr,
                                "native": nat, "fallback": fb, "workers": wk})
                print(f"  {label}: wall={wall:.3f}s abs={peak:.0f}MB incr={incr:.0f}MB "
                      f"rows={rows} native={nat} fallback={fb} "
                      f"workers={wk['n_grown']}/{wk['n_workers']} "
                      f"max_task={wk['max_worker_incr_mb']:.0f}MB")
    json.dump(results, open(os.path.join(OUT, "results_layout.json"), "w"), indent=2)
    return results


def axis_schema():
    N = 500_000
    results = []
    hashes = {}
    for schema in fx.SCHEMA_BUILDERS:
        spec = {"rows": N, "num_files": 1, "row_group_size": 100_000, "schema": schema}
        path = fx.make_fixture(f"schema_{schema}", spec)
        for reader in ["pyarrow", "arrow_rs"]:
            label = f"schema__{schema}__{reader}"
            d = _run_dir(label)
            _fresh_session(reader, d)
            _warm(reader, d)
            t0 = time.time()
            err = None
            try:
                h = consume_hash(ray.data.read_parquet(path))
            except Exception as e:  # native crate handed a type it can't decode
                h = {"__error__": f"{type(e).__name__}: {e}"[:200]}
                err = h["__error__"]
            wall = time.time() - t0
            ray.shutdown(); time.sleep(0.3)
            nat, fb = _count_paths(d)
            hashes[(schema, reader)] = h
            results.append({"schema": schema, "reader": reader, "wall_s": wall,
                            "expected": fx.expected_path(schema),
                            "native": nat, "fallback": fb, "error": err})
            print(f"  {label}: wall={wall:.3f}s native={nat} fallback={fb} "
                  f"(expected {fx.expected_path(schema)}) err={err}")
    # Parity: arrow_rs vs pyarrow per schema.
    for schema in fx.SCHEMA_BUILDERS:
        pa_h = hashes[(schema, "pyarrow")]
        rs_h = hashes[(schema, "arrow_rs")]
        match = pa_h == rs_h
        for r in results:
            if r["schema"] == schema:
                r["parity"] = match
        print(f"  PARITY {schema}: {'OK' if match else 'MISMATCH ' + str([k for k in pa_h if pa_h.get(k)!=rs_h.get(k)])}")
    json.dump(results, open(os.path.join(OUT, "results_schema.json"), "w"), indent=2)
    return results


def axis_tuning():
    spec = {"rows": 2_000_000, "num_files": 1, "row_group_size": 2_000_000, "schema": "wide_str"}
    path = fx.make_fixture("one_large_grp", spec)
    results = []
    # PyArrow baseline (no budget knob).
    d = _run_dir("tuning__pyarrow")
    _fresh_session("pyarrow", d)
    _warm("pyarrow", d)
    t0 = time.time(); rows = consume(ray.data.read_parquet(path), "iter_batches")
    wall = time.time() - t0; ray.shutdown(); time.sleep(0.3)
    results.append({"reader": "pyarrow", "budget_mb": None, "wall_s": wall})
    print(f"  tuning pyarrow: wall={wall:.3f}s")
    for budget_mb in [1, 2, 4, 8, 16, 32]:
        d = _run_dir(f"tuning__arrow_rs__{budget_mb}mb")
        _fresh_session("arrow_rs", d, budget_bytes=budget_mb * MB)
        _warm("arrow_rs", d)
        t0 = time.time(); rows = consume(ray.data.read_parquet(path), "iter_batches")
        wall = time.time() - t0; ray.shutdown(); time.sleep(0.3)
        results.append({"reader": "arrow_rs", "budget_mb": budget_mb, "wall_s": wall})
        print(f"  tuning arrow_rs {budget_mb}MB: wall={wall:.3f}s")
    json.dump(results, open(os.path.join(OUT, "results_tuning.json"), "w"), indent=2)
    return results


def axis_leak():
    spec = {"rows": 1_000_000, "num_files": 1, "row_group_size": 1_000_000, "schema": "wide_str"}
    path = fx.make_fixture("leak_1grp", spec)
    results = []
    for reader in ["pyarrow", "arrow_rs"]:
        d = _run_dir(f"leak__{reader}")
        _fresh_session(reader, d)
        _warm(reader, d)
        windows = []
        for i in range(8):
            t0 = time.time()
            rows = consume(ray.data.read_parquet(path), "iter_batches")
            t1 = time.time()
            windows.append({"i": i, "t_start": t0, "t_end": t1, "wall_s": t1 - t0})
            print(f"  leak {reader} iter {i}: wall={t1-t0:.3f}s")
        json.dump({"reader": reader, "windows": windows},
                  open(os.path.join(d, "windows.json"), "w"))
        ray.shutdown(); time.sleep(0.3)
        results.append({"reader": reader, "windows": windows})
    json.dump(results, open(os.path.join(OUT, "results_leak.json"), "w"), indent=2)
    return results


def axis_mixed():
    # 6 files, each a different schema with a different bytes/row, in one dataset
    # dir read as a single dataset. Tests (a) whether arrow-rs's per-group byte
    # budget adapts file-to-file against PyArrow's fixed sizing, and (b) that a
    # mixed dataset with a NON-native schema (struct) routes cleanly: arrow-rs
    # takes the 5 flat files natively and falls back to PyArrow for the struct
    # file, in one read, without breaking. Struct is the "one with structs" case.
    path = fx.make_mixed_fixture("mixed6_struct", per=400_000)
    results = []
    for reader in ["pyarrow", "arrow_rs"]:
        rd = _run_dir(f"mixed__{reader}")
        _fresh_session(reader, rd)
        _warm(reader, rd)
        t0 = time.time(); rows = consume(ray.data.read_parquet(path), "iter_batches")
        t1 = time.time(); wall = t1 - t0; ray.shutdown(); time.sleep(0.3)
        nat, fb = _count_paths(rd)
        peak = _node_sum_peak_mb(rd, t0, t1)
        incr = _node_sum_incr_peak_mb(rd, t0, t1)
        wk = _worker_breakdown(rd, t0, t1)
        results.append({"reader": reader, "wall_s": wall, "rows": rows,
                        "t0": t0, "t1": t1, "node_sum_peak_mb": peak,
                        "node_sum_incr_mb": incr, "native": nat, "fallback": fb,
                        "workers": wk})
        print(f"  mixed {reader}: wall={wall:.3f}s rows={rows} abs={peak:.0f}MB "
              f"incr={incr:.0f}MB native={nat} fallback={fb} "
              f"workers={wk['n_grown']}/{wk['n_workers']} "
              f"max_task={wk['max_worker_incr_mb']:.0f}MB")
    json.dump(results, open(os.path.join(OUT, "results_mixed.json"), "w"), indent=2)
    return results


def _node_sum_peak_mb(trace_dir, t0, t1):
    """Peak of the SUM across all workers' private-heap (USS), within the measured
    window — the node-level physical memory the concurrent read workers occupy."""
    import csv

    import numpy as np

    series = []
    for f in glob.glob(os.path.join(trace_dir, "uss_*.csv")):
        rows = list(csv.reader(open(f)))[1:]
        if not rows:
            continue
        series.append((np.array([float(r[0]) for r in rows]),
                       np.array([float(r[1]) for r in rows])))
    if not series:
        return 0.0
    grid = np.linspace(t0, t1, 500)
    total = np.zeros_like(grid)
    for ep, uss in series:
        idx = np.searchsorted(ep, grid, side="right") - 1
        held = np.where(idx >= 0, uss[np.clip(idx, 0, len(uss) - 1)], 0.0)
        # Alive-gate: don't forward-fill a worker's last sample past its final
        # sample — an exited worker would otherwise contribute a constant across
        # the rest of the window. (The §5.0 dead-worker double-count check from
        # the mac run, now applied by default; the incr variant already had it.)
        alive = (grid >= ep.min()) & (grid <= ep.max())
        total += np.where(alive, held, 0.0)
    return float(total.max()) / MB


def _node_sum_incr_peak_mb(trace_dir, t0, t1):
    """Like _node_sum_peak_mb, but each worker's USS at the START of the window (its
    warm baseline) is subtracted before summing — the peak EXTRA private heap the
    read itself caused, node-wide.

    This, not the absolute sum, is the number to compare across readers/knobs. On a
    many-core node Ray pre-starts one worker per core; all of them inherit our
    runtime_env and run the sampler, but only the few that actually decode grow their
    heap. The idle workers each hold ~tens of MB of imported libs — a large,
    reader-INDEPENDENT constant that, summed absolutely, swamps the decode signal and
    washes every ratio toward 1.0x. Subtracting each worker's own t0 baseline cancels
    that constant; a worker that never grows contributes ~0. (Absolute node-sum, for
    cross-checking the platform memory dashboard, stays in _node_sum_peak_mb.)
    """
    import csv

    import numpy as np

    series = []
    for f in glob.glob(os.path.join(trace_dir, "uss_*.csv")):
        rows = list(csv.reader(open(f)))[1:]
        if not rows:
            continue
        series.append((np.array([float(r[0]) for r in rows]),
                       np.array([float(r[1]) for r in rows])))
    if not series:
        return 0.0
    grid = np.linspace(t0, t1, 500)
    total = np.zeros_like(grid)
    for ep, uss in series:
        idx = np.searchsorted(ep, grid, side="right") - 1
        held = np.where(idx >= 0, uss[np.clip(idx, 0, len(uss) - 1)], np.nan)
        alive = (grid >= ep.min()) & (grid <= ep.max())
        held = np.where(alive, held, np.nan)
        valid = ~np.isnan(held)
        if not valid.any():
            continue
        # baseline = this worker's first in-window sample (already post-import)
        baseline = held[int(np.argmax(valid))]
        total += np.where(valid, held - baseline, 0.0)
    return float(total.max()) / MB


def _worker_breakdown(trace_dir, t0, t1):
    """Per-worker decomposition of the measured window — the disaggregation a
    node-sum scalar can't give. One 150 MB decoder plus idle workers must be
    distinguishable from several workers each paying a ~100 MB cold-start import
    ramp inside the window; a single incr number conflates them (that conflation
    is exactly how the layout small_many_grp anomaly slipped in).

    Reports: n_workers (traces present), n_grown (>5 MB windowed growth — the
    actual decoders), max_worker_incr_mb (largest single-worker windowed delta ≈
    the true per-task working set), max_worker_minmax_mb (largest full-trace
    min→max growth — the mac-methodology per-PID number, immune to the
    warm-baseline hiding that understates a reader whose warmup retained heap).
    """
    import csv

    import numpy as np

    n = grown = 0
    max_incr = max_minmax = 0.0
    for f in glob.glob(os.path.join(trace_dir, "uss_*.csv")):
        rows = list(csv.reader(open(f)))[1:]
        if not rows:
            continue
        ep = np.array([float(r[0]) for r in rows])
        uss = np.array([float(r[1]) for r in rows])
        n += 1
        max_minmax = max(max_minmax, float(uss.max() - uss.min()) / MB)
        in_w = (ep >= t0) & (ep <= t1)
        if not in_w.any():
            continue
        w = uss[in_w]
        delta = float(w.max() - w[0]) / MB
        if delta > 5.0:
            grown += 1
        max_incr = max(max_incr, delta)
    return {"n_workers": n, "n_grown": grown,
            "max_worker_incr_mb": round(max_incr, 1),
            "max_worker_minmax_mb": round(max_minmax, 1)}


def axis_scaling():
    """Is arrow-rs O(n) or O(n^2) on ONE big row group? If a per-batch reader
    rebuild (growing RowSelection skip) lurked, wall/row would GROW with row count.
    Sweep rows on a single-row-group file at a fixed budget and check wall/row is
    ~flat. Also sweep the budget (fewer/more batches) — O(n^2) would blow up as the
    budget shrinks (more rebuilds); O(n) per-batch-overhead does the opposite."""
    results = []
    for rows in [1_000_000, 2_000_000, 4_000_000, 8_000_000]:
        spec = {"rows": rows, "num_files": 1, "row_group_size": rows, "schema": "wide_str"}
        path = fx.make_fixture(f"scale_{rows}", spec)
        for reader in ["pyarrow", "arrow_rs"]:
            d = _run_dir(f"scale__{rows}__{reader}")
            _fresh_session(reader, d)
            _warm(reader, d)
            t0 = time.time()
            r = consume(ray.data.read_parquet(path), "iter_batches")
            wall = time.time() - t0
            ray.shutdown(); time.sleep(0.3)
            results.append({"rows": rows, "reader": reader, "wall_s": wall,
                            "us_per_row": wall / rows * 1e6})
            print(f"  scaling {reader} {rows//1_000_000}M: wall={wall:.3f}s "
                  f"({wall / rows * 1e6:.4f} us/row)")
    json.dump(results, open(os.path.join(OUT, "results_scaling.json"), "w"), indent=2)
    return results


def axis_concurrency():
    """The real single-node distributed test: N files each with ONE big row group,
    read concurrently across `num_cpus` workers. This is the scenario the OOM thesis
    is about — K concurrent big decodes, each holding a private-heap working set the
    scheduler doesn't count. Reports node-sum peak USS (physical memory the workers
    occupy together) and wall, both readers."""
    results = []
    # Two regimes: medium row group (per §5.2, ~parity even concurrent) and BIG
    # row group (the overcommit case — each worker holds a large private decode set,
    # so K concurrent big decodes are where PyArrow balloons the node).
    fixtures = {
        "medium_8x1M": {"rows": 8_000_000, "num_files": 8, "row_group_size": 1_000_000,
                        "schema": "wide_str"},
        "big_4x4M": {"rows": 16_000_000, "num_files": 4, "row_group_size": 4_000_000,
                     "schema": "wide_str"},
    }
    for fxname, spec in fixtures.items():
        path = fx.make_fixture(f"conc_{fxname}", spec)
        for reader in ["pyarrow", "arrow_rs"]:
            for ncpu in [2, 4]:
                d = _run_dir(f"conc__{fxname}__{reader}__cpu{ncpu}")
                _fresh_session(reader, d, num_cpus=ncpu)
                _warm(reader, d)
                t0 = time.time()
                rows = consume(ray.data.read_parquet(path), "iter_batches")
                t1 = time.time()
                ray.shutdown(); time.sleep(0.3)
                peak = _node_sum_peak_mb(d, t0, t1)
                incr = _node_sum_incr_peak_mb(d, t0, t1)
                nat, fb = _count_paths(d)
                results.append({"fixture": fxname, "reader": reader, "num_cpus": ncpu,
                                "wall_s": t1 - t0, "node_sum_peak_mb": peak,
                                "node_sum_incr_mb": incr, "rows": rows, "t0": t0,
                                "t1": t1, "native": nat, "fallback": fb})
                print(f"  conc {fxname} {reader} cpu={ncpu}: wall={t1-t0:.3f}s "
                      f"abs={peak:.0f}MB incr={incr:.0f}MB native={nat} fallback={fb}")
    json.dump(results, open(os.path.join(OUT, "results_concurrency.json"), "w"), indent=2)
    return results


def axis_showcase():
    """The 'show me the graphs' axis: 5 configs spanning the variables we varied
    (row-group size, file size, layout, file count), each read end-to-end with
    BOTH (a) 5 ms USS sampling and (b) a driver-side cumulative-rows-over-time
    trace, and the measured read window recorded. Feeds the 5-panel memory-vs-time
    grid and the speed-vs-time overlay. huge_str (3 cols x 48) so the decode
    transient is fat enough to see."""
    S = "huge_str"
    configs = {
        "small_many_rg": {"rows": 2_000_000, "num_files": 1, "row_group_size": 50_000, "schema": S},
        "medium_1rg": {"rows": 2_000_000, "num_files": 1, "row_group_size": 2_000_000, "schema": S},
        "large_1rg": {"rows": 8_000_000, "num_files": 1, "row_group_size": 8_000_000, "schema": S},
        "mixed_rg": {"rows": 4_000_000, "num_files": 1,
                     "row_group_sizes": [500_000, 20_000, 500_000, 20_000], "schema": S},
        "many_files_1rg": {"rows": 8_000_000, "num_files": 4, "row_group_size": 2_000_000, "schema": S},
    }
    results = []
    for name, spec in configs.items():
        path = fx.make_fixture(f"show_{name}", spec)
        for reader in ["pyarrow", "arrow_rs"]:
            d = _run_dir(f"show__{name}__{reader}")
            _fresh_session(reader, d, num_cpus=4)
            _warm(reader, d)
            prog = os.path.join(d, "progress.csv")
            t0 = time.time()
            rows = consume(ray.data.read_parquet(path), "iter_batches", prog)
            t1 = time.time()
            ray.shutdown(); time.sleep(0.3)
            peak = _node_sum_peak_mb(d, t0, t1)
            incr = _node_sum_incr_peak_mb(d, t0, t1)
            nat, fb = _count_paths(d)
            results.append({"config": name, "reader": reader, "wall_s": t1 - t0,
                            "node_sum_peak_mb": peak, "node_sum_incr_mb": incr,
                            "rows": rows, "t0": t0, "t1": t1,
                            "native": nat, "fallback": fb})
            print(f"  show {name} {reader}: wall={t1-t0:.3f}s abs={peak:.0f}MB "
                  f"incr={incr:.0f}MB rows={rows} native={nat} fallback={fb}")
    json.dump(results, open(os.path.join(OUT, "results_showcase.json"), "w"), indent=2)
    return results


def _run_sweep(name, levels, mode="iter_batches", num_cpus=4):
    """Run a one-variable sweep: `levels` is an ordered list of dicts, each
    {label, fixture_name, spec, budget_bytes?, num_cpus?}. For every level we read
    the fixture with BOTH readers, sampling USS at 5 ms, logging rows-over-time,
    and recording the measured read window. All 'same test, one thing varied', so
    the 5 panels of the resulting plot are directly comparable.
    -> results_sweep_<name>.json
    """
    results = []
    for lv in levels:
        spec = lv["spec"]
        budget = lv.get("budget_bytes", 8 * MB)
        ncpu = lv.get("num_cpus", num_cpus)
        path = fx.make_fixture(lv["fixture_name"], spec)
        for reader in ["pyarrow", "arrow_rs"]:
            d = _run_dir(f"sweep_{name}__{lv['label']}__{reader}")
            _fresh_session(reader, d, budget_bytes=budget, num_cpus=ncpu)
            _warm(reader, d)
            prog = os.path.join(d, "progress.csv")
            t0 = time.time()
            rows = consume(ray.data.read_parquet(path), mode, prog)
            t1 = time.time()
            ray.shutdown(); time.sleep(0.3)
            peak = _node_sum_peak_mb(d, t0, t1)
            incr = _node_sum_incr_peak_mb(d, t0, t1)
            nat, fb = _count_paths(d)
            wk = _worker_breakdown(d, t0, t1)
            results.append({"sweep": name, "level": lv["label"], "reader": reader,
                            "wall_s": t1 - t0, "node_sum_peak_mb": peak,
                            "node_sum_incr_mb": incr, "rows": rows,
                            "t0": t0, "t1": t1, "budget_mb": budget // MB,
                            "num_cpus": ncpu, "native": nat, "fallback": fb,
                            "workers": wk})
            print(f"  sweep[{name}] {lv['label']} {reader}: wall={t1-t0:.3f}s "
                  f"abs={peak:.0f}MB incr={incr:.0f}MB rows={rows} "
                  f"native={nat} fallback={fb} "
                  f"workers={wk['n_grown']}/{wk['n_workers']} "
                  f"max_task={wk['max_worker_incr_mb']:.0f}MB")
    json.dump(results, open(os.path.join(OUT, f"results_sweep_{name}.json"), "w"), indent=2)
    return results


def sweep_size():
    """Same schema (flat int64, one big row group), 5 decoded sizes from ~14 MB
    to ~1.4 GB. Shows the memory gap widening with size while arrow-rs stays flat."""
    # _ints => id + 8 int64 cols = 72 bytes/row. rows chosen for ~14/50/144/504/1440 MB.
    levels = [
        {"label": "14MB_200k", "fixture_name": "sw_size_200k",
         "spec": {"rows": 200_000, "num_files": 1, "row_group_size": 200_000, "schema": "int"}},
        {"label": "50MB_700k", "fixture_name": "sw_size_700k",
         "spec": {"rows": 700_000, "num_files": 1, "row_group_size": 700_000, "schema": "int"}},
        {"label": "144MB_2M", "fixture_name": "sw_size_2M",
         "spec": {"rows": 2_000_000, "num_files": 1, "row_group_size": 2_000_000, "schema": "int"}},
        {"label": "504MB_7M", "fixture_name": "sw_size_7M",
         "spec": {"rows": 7_000_000, "num_files": 1, "row_group_size": 7_000_000, "schema": "int"}},
        {"label": "1.4GB_20M", "fixture_name": "sw_size_20M",
         "spec": {"rows": 20_000_000, "num_files": 1, "row_group_size": 20_000_000, "schema": "int"}},
    ]
    return _run_sweep("size", levels)


def sweep_batch():
    """Same file (huge_str, 4 M rows, one ~400 MB row group), 5 decode budgets.
    arrow-rs's peak tracks the budget (memory is a knob); PyArrow ignores it
    (materializes the whole group) — a flat control line across all 5 panels."""
    spec = {"rows": 4_000_000, "num_files": 1, "row_group_size": 4_000_000, "schema": "huge_str"}
    levels = [
        {"label": f"{b}MB", "fixture_name": "sw_batch_4M", "spec": spec,
         "budget_bytes": b * MB}
        for b in [1, 4, 8, 16, 64]
    ]
    return _run_sweep("batch", levels)


def sweep_rowgroup():
    """Fixed total size (4 M rows huge_str, ~400 MB), vary how it's chopped into
    row groups: from many tiny groups (Ray's pool parallelizes → parity) to one
    whole-file group (the lone-big-fragment case → the gap). This is the
    'row-group size / number of row groups' axis."""
    S = "huge_str"
    R = 4_000_000
    levels = [
        {"label": f"rg_{rg//1000}k" if rg < R else "rg_whole_file",
         "fixture_name": f"sw_rg_{rg}",
         "spec": {"rows": R, "num_files": 1, "row_group_size": rg, "schema": S}}
        for rg in [50_000, 200_000, 1_000_000, 2_000_000, R]
    ]
    return _run_sweep("rowgroup", levels)


def sweep_files():
    """Fixed per-file layout (one ~200 MB row group, huge_str), vary the number of
    files read concurrently across 4 workers: 1,2,4,6,8. Shows node-sum USS (the
    physical RAM the concurrent decodes occupy) rising with the file count — the
    single-node overcommit that is the actual OOM mechanism."""
    S = "huge_str"
    levels = [
        {"label": f"{nf}_files", "fixture_name": f"sw_files_{nf}",
         "spec": {"rows": nf * 2_000_000, "num_files": nf,
                  "row_group_size": 2_000_000, "schema": S}, "num_cpus": 4}
        for nf in [1, 2, 4, 6, 8]
    ]
    return _run_sweep("files", levels)


def sweep_schema():
    """Same size (2 M rows, one big row group), vary the column dtype: the memory
    gap is biggest where cells are widest (wide strings) and smallest for fixed
    width numerics."""
    levels = [
        {"label": sc, "fixture_name": f"sw_schema_{sc}",
         "spec": {"rows": 2_000_000, "num_files": 1, "row_group_size": 2_000_000,
                  "schema": sc}}
        for sc in ["int", "float", "wide_str", "large_str", "huge_str"]
    ]
    return _run_sweep("schema", levels)


def sweep_batch_dd():
    """The decode budget sweep in decode_drop mode — output retention is removed,
    so arrow-rs's memory now TRACKS the budget (the knob that iter_batches masks),
    while PyArrow still holds the whole row group regardless."""
    spec = {"rows": 4_000_000, "num_files": 1, "row_group_size": 4_000_000, "schema": "huge_str"}
    levels = [
        {"label": f"{b}MB", "fixture_name": "sw_batch_4M", "spec": spec,
         "budget_bytes": b * MB}
        for b in [1, 4, 8, 16, 64]
    ]
    return _run_sweep("batch_dd", levels, mode="decode_drop")


def axis_workloads():
    """The real 'decode-heavy, output-light' cases (per the §3.3 critique): an
    aggregation (ds.sum) and a selective read-time filter, on one big row group.
    Both force a full decode but emit ~nothing, so they show the decode-heap gap
    on a realistic consumer (not the synthetic count-returning map_batches), and
    the filter honestly exercises the no-predicate-pushdown limitation (§4.7:
    arrow-rs decodes rows it then drops)."""
    spec = {"rows": 4_000_000, "num_files": 1, "row_group_size": 4_000_000, "schema": "int"}
    path = fx.make_fixture("wl_int_4M", spec)
    high = (1 << 30) - 2000  # selective: only a sliver of i0 exceeds this
    results = []
    for reader in ["pyarrow", "arrow_rs"]:
        # Aggregation: sum one column. Full decode, output = one scalar.
        d = _run_dir(f"wl__sum__{reader}")
        _fresh_session(reader, d)
        _warm(reader, d)
        t0 = time.time()
        total = ray.data.read_parquet(path).sum("i0")
        t1 = time.time()
        ray.shutdown(); time.sleep(0.3)
        peak = _node_sum_peak_mb(d, t0, t1)
        incr = _node_sum_incr_peak_mb(d, t0, t1)
        nat, fb = _count_paths(d)
        wk = _worker_breakdown(d, t0, t1)
        results.append({"workload": "sum(i0)", "reader": reader, "wall_s": t1 - t0,
                        "node_sum_peak_mb": peak, "node_sum_incr_mb": incr,
                        "out": str(total), "t0": t0, "t1": t1,
                        "native": nat, "fallback": fb, "workers": wk})
        print(f"  wl sum {reader}: wall={t1-t0:.3f}s abs={peak:.0f}MB "
              f"incr={incr:.0f}MB native={nat} fallback={fb} "
              f"workers={wk['n_grown']}/{wk['n_workers']} "
              f"max_task={wk['max_worker_incr_mb']:.0f}MB")
        # Selective read-time filter: decode all, keep a sliver. Non-empty
        # projection so the arrow-rs path runs (count() would empty-project → fallback).
        d = _run_dir(f"wl__filter__{reader}")
        _fresh_session(reader, d)
        _warm(reader, d)
        t0 = time.time()
        kept = consume(
            ray.data.read_parquet(path).filter(expr=f"i0 > {high}"), "iter_batches")
        t1 = time.time()
        ray.shutdown(); time.sleep(0.3)
        peak = _node_sum_peak_mb(d, t0, t1)
        incr = _node_sum_incr_peak_mb(d, t0, t1)
        nat, fb = _count_paths(d)
        wk = _worker_breakdown(d, t0, t1)
        results.append({"workload": "filter(i0>hi)", "reader": reader, "wall_s": t1 - t0,
                        "node_sum_peak_mb": peak, "node_sum_incr_mb": incr,
                        "kept_rows": kept, "t0": t0, "t1": t1,
                        "native": nat, "fallback": fb, "workers": wk})
        print(f"  wl filter {reader}: wall={t1-t0:.3f}s abs={peak:.0f}MB "
              f"incr={incr:.0f}MB kept={kept} native={nat} fallback={fb} "
              f"workers={wk['n_grown']}/{wk['n_workers']} "
              f"max_task={wk['max_worker_incr_mb']:.0f}MB")
    json.dump(results, open(os.path.join(OUT, "results_workloads.json"), "w"), indent=2)
    return results


def axis_s3():
    """Memory + speed on a REAL S3 dataset — the deferred perf-validation phase.

    **Env-gated and intentionally not run locally.** S3 speed/memory can only be
    judged on Linux + real S3: moto has no network latency (so it can't show the
    windowed-fetch memory win) and macOS USS is only directional. Set
    ``RAY_DATA_ARROW_RS_S3_BENCH_PATH=s3://bucket/prefix`` (with AWS creds in the
    environment) on the Linux box to run it. The single-big-row-group-per-file
    layout is the target (§0); write fixtures with ``write_page_index=True``.

    This sweeps the three knobs that govern the memory-first claim, since the goal
    is memory-parity-or-better at speed-parity (NOT max throughput):
      * ``fetch_window_mb`` — compressed bytes in flight per stream. Smaller =
        lower, flatter peak; the whole point is that S3 peak is this knob, not the
        row-group size. ``0`` = no window cap (fetch the whole range — the old
        behavior, kept as the upper-bound control). Swept at a fixed 2 MiB budget.
      * ``decode_budget_bytes`` — the decoded working-set floor (byte-budget
        batching). Swept {2, 8, 32} MiB at a fixed 16 MiB window to confirm the
        standalone finding that budget is the *floor* knob (small mem effect) while
        the window is the lever. Default is 2 MiB (was 8; lowered per the local win).
      * allocator — the retention lever (§7.8), all env-only / no recompile:
        ``MALLOC_ARENA_MAX=2`` (glibc arena cap) always runs; and if you export
        ``RAY_DATA_ARROW_RS_MIMALLOC_SO`` / ``RAY_DATA_ARROW_RS_JEMALLOC_SO`` to the
        respective ``.so``, an ``LD_PRELOAD`` run for each is added (swaps the whole
        worker's allocator — Python + Rust + PyArrow — sidestepping the compile-time
        mimalloc segfault). PyArrow itself always uses its bundled jemalloc.
    PyArrow is the baseline. K stays 1 for the per-file layout (Ray's pool
    parallelizes files); the lone-big-row-group K-split is a separate fixture.
    """
    s3_path = os.environ.get("RAY_DATA_ARROW_RS_S3_BENCH_PATH")
    if not s3_path:
        print("  (skip s3: set RAY_DATA_ARROW_RS_S3_BENCH_PATH=s3://... on a "
              "Linux+real-S3 box — S3 perf is not meaningful on macOS/moto)")
        return []

    # (reader, fetch_window_mb, budget_mb, malloc_arena_max, ld_preload) configs.
    # window/budget/arena/preload are ignored by the pyarrow baseline. The window
    # and budget sweeps cross at (window=16, budget=2) so that point is shared.
    configs = [
        ("pyarrow", None, None, None, None),
        # --- window sweep @ fixed 2 MiB budget (the memory lever) ---
        ("arrow_rs", 4, 2, None, None),
        ("arrow_rs", 16, 2, None, None),
        ("arrow_rs", 64, 2, None, None),
        ("arrow_rs", 0, 2, None, None),   # no window cap (control: shows win's size)
        # --- budget sweep @ fixed 16 MiB window (the floor knob) ---
        ("arrow_rs", 16, 8, None, None),
        ("arrow_rs", 16, 32, None, None),
        # --- allocator sweep @ window 16 / budget 2 (all env-only, no recompile) ---
        ("arrow_rs", 16, 2, 2, None),     # glibc + MALLOC_ARENA_MAX=2
    ]
    # LD_PRELOAD allocator A/B (#9): only added when the .so path is provided, so a
    # box without the lib installed just skips it. Find paths with e.g.
    #   dpkg -L libmimalloc2 | grep '\.so'   /   dpkg -L libjemalloc2 | grep '\.so'
    mi = os.environ.get("RAY_DATA_ARROW_RS_MIMALLOC_SO")
    je = os.environ.get("RAY_DATA_ARROW_RS_JEMALLOC_SO")
    if mi:
        configs.append(("arrow_rs", 16, 2, None, mi))    # LD_PRELOAD mimalloc
    if je:
        configs.append(("arrow_rs", 16, 2, None, je))    # LD_PRELOAD jemalloc
    results = []
    for reader, window, budget_mb, arena, preload in configs:
        alloc = ("mi" if preload and "mimalloc" in preload
                 else "je" if preload and "jemalloc" in preload
                 else f"arena{arena}" if arena else "sys")
        tag = (reader if reader == "pyarrow"
               else f"arrow_rs_w{window}_b{budget_mb}_{alloc}")
        d = _run_dir(f"s3__{tag}")
        _fresh_session(reader, d, budget_bytes=(budget_mb or 2) * MB,
                       fetch_window_mb=(window or 0), malloc_arena_max=arena,
                       ld_preload=preload)
        _warm(reader, d)  # warms worker imports on a tiny local fixture
        t0 = time.time()
        rows = consume(ray.data.read_parquet(s3_path), "iter_batches",
                       progress_path=os.path.join(d, "progress.csv"))
        t1 = time.time(); ray.shutdown(); time.sleep(0.3)
        peak = _node_sum_peak_mb(d, t0, t1)
        incr = _node_sum_incr_peak_mb(d, t0, t1)
        nat, fb = _count_paths(d)
        results.append({"reader": reader, "tag": tag, "fetch_window_mb": window,
                        "budget_mb": budget_mb, "malloc_arena_max": arena,
                        "ld_preload": preload, "alloc": alloc,
                        "wall_s": t1 - t0, "rows": rows, "t0": t0, "t1": t1,
                        "node_sum_peak_mb": peak, "node_sum_incr_mb": incr,
                        "native": nat, "fallback": fb})
        print(f"  s3 {tag}: wall={t1-t0:.3f}s abs_peak={peak:.0f}MB "
              f"incr_peak={incr:.0f}MB rows={rows} native={nat} fallback={fb}")
    json.dump(results, open(os.path.join(OUT, "results_s3.json"), "w"), indent=2)
    return results


AXES = {"layout": axis_layout, "schema": axis_schema, "tuning": axis_tuning,
        "leak": axis_leak, "mixed": axis_mixed, "scaling": axis_scaling,
        "concurrency": axis_concurrency, "showcase": axis_showcase,
        "sweep_size": sweep_size, "sweep_batch": sweep_batch,
        "sweep_rowgroup": sweep_rowgroup, "sweep_files": sweep_files,
        "sweep_schema": sweep_schema, "sweep_batch_dd": sweep_batch_dd,
        "workloads": axis_workloads, "s3": axis_s3}


def write_summary_csv():
    """Flatten every runs/results_*.json into ONE runs/summary.csv (also echoed to
    stdout): axis, config, reader, wall_s, abs_peak_mb, incr_peak_mb, rows, path.
    One row per measured read across ALL axes — the machine-readable digest to paste
    back for analysis. Pyarrow and arrow_rs rows share the same (axis, config), so
    the mem/speed ratios are a one-line pairing away; incr_peak_mb is the column that
    matters (baseline-subtracted; see _node_sum_incr_peak_mb)."""
    import csv as _csv

    def _config(axis, r):
        base = axis
        for k in ("tag", "config", "workload", "layout", "schema", "fixture"):
            if r.get(k) is not None:
                base = str(r[k])
                break
        if r.get("sweep") is not None and r.get("level") is not None:
            base = f"{r['sweep']}:{r['level']}"
        if axis == "scaling" and r.get("rows"):
            base = f"{r['rows'] // 1_000_000}M"
        extra = []
        if r.get("budget_mb") is not None:
            extra.append(f"b{r['budget_mb']}")
        if r.get("num_cpus") is not None:
            extra.append(f"cpu{r['num_cpus']}")
        return base + (("_" + "_".join(extra)) if extra else "")

    def _num(r, k):
        v = r.get(k)
        return round(v, 3) if isinstance(v, (int, float)) else ""

    fields = ["axis", "config", "reader", "wall_s", "abs_peak_mb", "incr_peak_mb",
              "max_task_mb", "n_grown", "n_workers", "rows", "native", "fallback"]
    out_rows = []
    for f in sorted(glob.glob(os.path.join(OUT, "results_*.json"))):
        axis = os.path.basename(f)[len("results_"):-len(".json")]
        try:
            data = json.load(open(f))
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        for r in data:
            if not isinstance(r, dict) or "reader" not in r:
                continue
            wk = r.get("workers") or {}
            out_rows.append({
                "axis": axis, "config": _config(axis, r),
                "reader": r.get("reader", ""), "wall_s": _num(r, "wall_s"),
                "abs_peak_mb": _num(r, "node_sum_peak_mb"),
                "incr_peak_mb": _num(r, "node_sum_incr_mb"),
                "max_task_mb": wk.get("max_worker_incr_mb", ""),
                "n_grown": wk.get("n_grown", ""),
                "n_workers": wk.get("n_workers", ""),
                "rows": r.get("rows", ""), "native": r.get("native", ""),
                "fallback": r.get("fallback", ""),
            })
    path = os.path.join(OUT, "summary.csv")
    with open(path, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    print(f"\n===== SUMMARY CSV ({len(out_rows)} rows) -> {path} =====")
    print(",".join(fields))
    for r in out_rows:
        print(",".join(str(r[k]) for k in fields))
    return path


def main():
    os.makedirs(OUT, exist_ok=True)
    which = sys.argv[1].split(",") if len(sys.argv) > 1 else list(AXES)
    for axis in which:
        print(f"\n===== AXIS: {axis} =====")
        AXES[axis]()
    write_summary_csv()
    print("\nALL DONE")


if __name__ == "__main__":
    main()
