"""arrow-rs vs PyArrow benchmark suite — the macOS-decisive axes.

Runs the axes whose verdict does NOT depend on absolute memory magnitude (which
is only directional on macOS because USS excludes shared pages). Those are:

  layout   — #1  wall time across the 5 file/row-group shapes (both readers)
  schema   — #2  path taken (native vs PyArrow fallback) + per-column parity,
                  across int/float/str/struct/list/tensor (coverage %)
  tuning   — #3  sweep decode_budget_bytes on one big single-row-group file
  leak     — #8  read the same file N times in ONE session; USS must return to
                  its floor each iteration (no ratchet)
  leak_multigrp — the proper arrow#39808 geometry: MANY small row groups in ONE
                  read task vs one-per-task, all 3 readers, pre_buffer on/off.
                  Shows the scanner's per-task working set tier above iter/arrow-rs
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

# The three readers every comparison axis sweeps, so the existing benchmark
# produces the three-reader figure (table + 3-line USS graph) with no new axis:
#   pyarrow       — Ray's normal V2 scanner path (fragment.scanner().scan_batches),
#                   the baseline that %Δ is measured against.
#   pyarrow_iter  — the SAME PyArrow pinned to pq.ParquetFile.iter_batches for every
#                   fragment (RAY_DATA_PARQUET_FORCE_ITER_BATCHES), lower peak but
#                   NOT Ray's default path.
#   arrow_rs      — the Rust reader.
READERS = ["pyarrow", "pyarrow_iter", "arrow_rs"]


def _fresh_session(
    reader,
    trace_dir,
    budget_bytes=8 * MB,
    k=1,
    num_cpus=4,
    fetch_window_mb=16,
    prefetch_windows=2,
    malloc_arena_max=None,
    ld_preload=None,
    pre_buffer=None,
    extra_env=None,
):
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
        # "pyarrow_v1" reproduces the pre-V2 read path (one whole-file scanner per
        # file — the ray#49158 surge geometry); every other reader is V2.
        "RAY_DATA_USE_DATASOURCE_V2": "0" if reader == "pyarrow_v1" else "1",
        "RAY_DATA_USE_ARROW_RS_PARQUET_READER": "1" if reader == "arrow_rs" else "0",
        # Third reader: PyArrow forced down the pq.iter_batches row-level path for
        # EVERY fragment (not just ARROW-5030 nested cases). "pyarrow" = the normal
        # V2 scanner path (Ray's default); "pyarrow_iter" = the same PyArrow but
        # pinned to iter_batches; "arrow_rs" = the Rust reader.
        "RAY_DATA_PARQUET_FORCE_ITER_BATCHES": "1" if reader == "pyarrow_iter" else "0",
        "RAY_DATA_ARROW_RS_K": str(k),
        "RAY_DATA_ARROW_RS_DECODE_BUDGET_BYTES": str(budget_bytes),
        "RAY_DATA_ARROW_RS_FETCH_WINDOW_MB": str(fetch_window_mb),
        "RAY_DATA_ARROW_RS_PREFETCH_WINDOWS": str(prefetch_windows),
    }
    # pyarrow pre_buffer toggle (both scanner and iter paths honor it). Only
    # injected when an axis explicitly sweeps it; left unset otherwise so every
    # other axis keeps each reader's historical default (scanner=True, iter=False).
    if pre_buffer is not None:
        env_vars["RAY_DATA_PARQUET_PRE_BUFFER"] = "1" if pre_buffer else "0"
    # Arbitrary reader-knob overrides (batch_readahead, buffer_size, iter
    # use_threads, fragment_readahead, ...) injected into the worker env so an
    # axis can sweep any reader setting without a new named parameter.
    if extra_env:
        env_vars.update({k: str(v) for k, v in extra_env.items()})
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
        ray.init(
            address="auto",
            ignore_reinit_error=True,
            log_to_driver=False,
            runtime_env=runtime_env,
        )
    else:
        ray.init(
            num_cpus=num_cpus,
            include_dashboard=False,
            ignore_reinit_error=True,
            log_to_driver=False,
            runtime_env=runtime_env,
        )
    from ray.data.context import DataContext

    ctx = DataContext.get_current()
    # "pyarrow_v1" runs the legacy V1 ParquetDatasource (whole-file scanner); all
    # other readers are V2. Set on the driver ctx too (propagated to workers) so
    # it agrees with the RAY_DATA_USE_DATASOURCE_V2 worker env above.
    ctx.use_datasource_v2 = reader != "pyarrow_v1"
    ctx.use_arrow_rs_parquet_reader = reader == "arrow_rs"
    ctx.execution_options.preserve_order = True  # so parity hashes are comparable
    # Record the run's config knobs next to the traces (target block size, decode
    # budget, S3 fetch window). These are descriptive metadata for the figures and
    # summary CSV — the per-task graph's only reference line is the MEASURED
    # warm-worker floor (task_mem.py), not an estimate built from these.
    _write_meta(
        trace_dir,
        {
            "reader": reader,
            "target_block_mb": ctx.target_max_block_size / MB,
            "budget_mb": budget_bytes / MB,
            "fetch_window_mb": fetch_window_mb,
            "prefetch_windows": prefetch_windows,
        },
    )
    return ctx


def _write_meta(trace_dir, updates):
    """Merge updates into trace_dir/meta.json (read-modify-write, driver-side)."""
    p = os.path.join(trace_dir, "meta.json")
    meta = {}
    if os.path.exists(p):
        try:
            meta = json.load(open(p))
        except Exception:
            meta = {}
    meta.update(updates)
    with open(p, "w") as fh:
        json.dump(meta, fh)


def _note_fixture(trace_dir, path):
    """Record what the fixture actually IS next to the traces — its shape (rows,
    files, row groups), the largest row group's size (compressed + uncompressed),
    its compression, and a short column/dtype summary — so every figure and the
    summary CSV can say *what was read*, not just how fast.

    The geometry is written to meta.json (read by task_mem.py / summarize.py for
    figure subtitles) AND cached in ``_GEOM[trace_dir]`` so the axis can attach it
    to its result row via :func:`_R` (→ summary.csv columns). ``max_rg_comp_mb``
    (the largest row group's on-disk size) appears in the figure subtitle so the
    reader can see how much a whole-group reader must materialize — but it is NOT
    used to synthesize a reference line; the per-task graph's only reference is the
    measured warm-worker floor (task_mem.py).
    Best-effort: local paths, dirs, and s3:// URIs all work via pyarrow.dataset;
    failure just skips the line and leaves the geometry empty."""
    geom = {"fixture": str(path)}
    try:
        import pyarrow.dataset as pds

        dset = pds.dataset(path, format="parquet")
        max_comp = max_uncomp = 0
        sum_uncomp = 0.0
        n_rg = rows_total = num_files = 0
        compression = None
        for frag in dset.get_fragments():
            num_files += 1
            md = frag.metadata
            rows_total += md.num_rows
            for i in range(md.num_row_groups):
                rg = md.row_group(i)
                comp = sum(
                    rg.column(j).total_compressed_size for j in range(rg.num_columns)
                )
                uncomp = rg.total_byte_size
                max_comp = max(max_comp, comp)
                max_uncomp = max(max_uncomp, uncomp)
                sum_uncomp += uncomp
                n_rg += 1
                if compression is None and rg.num_columns:
                    try:
                        compression = rg.column(0).compression
                    except Exception:
                        pass
        schema = dset.schema
        shown = [
            f"{schema.field(i).name}:{schema.field(i).type}"
            for i in range(min(4, len(schema)))
        ]
        schema_desc = ", ".join(shown) + (
            f", … ({len(schema)} cols)" if len(schema) > 4 else ""
        )
        geom.update(
            {
                "rows_total": rows_total,
                "num_files": num_files,
                "num_row_groups": n_rg,
                "max_rg_comp_mb": max_comp / MB,
                "max_rg_uncomp_mb": max_uncomp / MB,
                "avg_rg_uncomp_mb": (sum_uncomp / n_rg / MB) if n_rg else 0.0,
                "compression": compression,
                "schema_desc": schema_desc,
            }
        )
        _write_meta(trace_dir, geom)
    except Exception as e:
        print(f"  (fixture note skipped for {path}: {type(e).__name__}: {e})")
    _GEOM[trace_dir] = geom
    return geom


# Fixture geometry recorded by the most recent _note_fixture(trace_dir, ...),
# keyed by trace_dir, so _R can fold it into that config's result row.
_GEOM = {}

# Which geometry fields ride along into results (and thus into summary.csv).
_GEOM_RESULT_KEYS = (
    "rows_total",
    "num_files",
    "num_row_groups",
    "max_rg_uncomp_mb",
    "max_rg_comp_mb",
    "compression",
    "schema_desc",
)


def _R(trace_dir, result):
    """Fold the fixture geometry recorded for ``trace_dir`` into a result row, so
    summary.csv carries what was read alongside the timing. Existing result keys
    win (never clobbered)."""
    g = _GEOM.get(trace_dir, {})
    merged = dict(result)
    for k in _GEOM_RESULT_KEYS:
        if k in g and k not in merged:
            merged[k] = g[k]
    # Stamp the measured wall into this run's meta.json so task_mem's per-config
    # timing table (absolute wall + %Δ vs the pyarrow V2-scanner baseline) is
    # self-contained per run dir — one choke point covering every axis, since
    # they all route their result row (carrying "wall_s") through _R.
    if isinstance(merged.get("wall_s"), (int, float)):
        _write_meta(trace_dir, {"wall_s": merged["wall_s"]})
    return merged


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
                batch_format="pyarrow"
            ):
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
        out[name] = hashlib.blake2b(
            sink.getvalue().to_pybytes(), digest_size=16
        ).hexdigest()
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
    # Stamp the warm/measured boundary so per-task graphs can exclude the warmup
    # read's task windows (they're real tasks, but of the tiny warm fixture).
    _write_meta(trace_dir, {"warm_end": time.time()})


# --------------------------------------------------------------------------- #
# Axes
# --------------------------------------------------------------------------- #
def axis_layout():
    S = "wide_str"
    layouts = {
        "small_1grp": {
            "rows": 200_000,
            "num_files": 1,
            "row_group_size": 200_000,
            "schema": S,
        },
        "small_many_grp": {
            "rows": 200_000,
            "num_files": 1,
            "row_group_size": 20_000,
            "schema": S,
        },
        "one_large_grp": {
            "rows": 2_000_000,
            "num_files": 1,
            "row_group_size": 2_000_000,
            "schema": S,
        },
        "many_large_grp": {
            "rows": 2_000_000,
            "num_files": 1,
            "row_group_size": 250_000,
            "schema": S,
        },
        "mixed_grp": {
            "rows": 2_000_000,
            "num_files": 1,
            "row_group_sizes": [500_000, 20_000, 500_000, 20_000],
            "schema": S,
        },
    }
    results = []
    for name, spec in layouts.items():
        path = fx.make_fixture(name, spec)
        for reader in READERS:
            for mode in ["iter_batches"]:
                label = f"layout__{name}__{mode}__{reader}"
                d = _run_dir(label)
                _fresh_session(reader, d)
                _warm(reader, d)
                _note_fixture(d, path)
                t0 = time.time()
                rows = consume(ray.data.read_parquet(path), mode)
                t1 = time.time()
                wall = t1 - t0
                ray.shutdown()
                time.sleep(0.3)
                peak = _node_sum_peak_mb(d, t0, t1)
                incr = _node_sum_incr_peak_mb(d, t0, t1)
                nat, fb = _count_paths(d)
                wk = _worker_breakdown(d, t0, t1)
                results.append(
                    _R(
                        d,
                        {
                            "layout": name,
                            "reader": reader,
                            "mode": mode,
                            "wall_s": wall,
                            "rows": rows,
                            "t0": t0,
                            "t1": t1,
                            "node_sum_peak_mb": peak,
                            "node_sum_incr_mb": incr,
                            "native": nat,
                            "fallback": fb,
                            "workers": wk,
                        },
                    )
                )
                _report(
                    f"layout {name} {reader} ({mode})", wall, peak, incr, nat, fb,
                    extra={
                        "rows": rows,
                        "workers": f"{wk['n_grown']}/{wk['n_workers']} grown  "
                        f"(max task {wk['max_worker_incr_mb']:.0f} MB)",
                    },
                )
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
            _note_fixture(d, path)
            t0 = time.time()
            err = None
            try:
                h = consume_hash(ray.data.read_parquet(path))
            except Exception as e:  # native crate handed a type it can't decode
                h = {"__error__": f"{type(e).__name__}: {e}"[:200]}
                err = h["__error__"]
            wall = time.time() - t0
            ray.shutdown()
            time.sleep(0.3)
            nat, fb = _count_paths(d)
            hashes[(schema, reader)] = h
            results.append(
                _R(
                    d,
                    {
                        "schema": schema,
                        "reader": reader,
                        "wall_s": wall,
                        "expected": fx.expected_path(schema),
                        "native": nat,
                        "fallback": fb,
                        "error": err,
                    },
                )
            )
            print(
                f"  {label}: wall={wall:.3f}s native={nat} fallback={fb} "
                f"(expected {fx.expected_path(schema)}) err={err}"
            )
    # Parity: arrow_rs vs pyarrow per schema.
    for schema in fx.SCHEMA_BUILDERS:
        pa_h = hashes[(schema, "pyarrow")]
        rs_h = hashes[(schema, "arrow_rs")]
        match = pa_h == rs_h
        for r in results:
            if r["schema"] == schema:
                r["parity"] = match
        print(
            f"  PARITY {schema}: {'OK' if match else 'MISMATCH ' + str([k for k in pa_h if pa_h.get(k)!=rs_h.get(k)])}"
        )
    json.dump(results, open(os.path.join(OUT, "results_schema.json"), "w"), indent=2)
    return results


def axis_tuning():
    spec = {
        "rows": 2_000_000,
        "num_files": 1,
        "row_group_size": 2_000_000,
        "schema": "wide_str",
    }
    path = fx.make_fixture("one_large_grp", spec)
    results = []
    # PyArrow baseline (no budget knob).
    d = _run_dir("tuning__pyarrow")
    _fresh_session("pyarrow", d)
    _warm("pyarrow", d)
    _note_fixture(d, path)
    t0 = time.time()
    consume(ray.data.read_parquet(path), "iter_batches")
    wall = time.time() - t0
    ray.shutdown()
    time.sleep(0.3)
    results.append(_R(d, {"reader": "pyarrow", "budget_mb": None, "wall_s": wall}))
    print(f"  tuning pyarrow: wall={wall:.3f}s")
    for budget_mb in [1, 2, 4, 8, 16, 32]:
        d = _run_dir(f"tuning__arrow_rs__{budget_mb}mb")
        _fresh_session("arrow_rs", d, budget_bytes=budget_mb * MB)
        _warm("arrow_rs", d)
        _note_fixture(d, path)
        t0 = time.time()
        consume(ray.data.read_parquet(path), "iter_batches")
        wall = time.time() - t0
        ray.shutdown()
        time.sleep(0.3)
        results.append(
            _R(d, {"reader": "arrow_rs", "budget_mb": budget_mb, "wall_s": wall})
        )
        print(f"  tuning arrow_rs {budget_mb}MB: wall={wall:.3f}s")
    json.dump(results, open(os.path.join(OUT, "results_tuning.json"), "w"), indent=2)
    return results


def axis_leak():
    spec = {
        "rows": 1_000_000,
        "num_files": 1,
        "row_group_size": 1_000_000,
        "schema": "wide_str",
    }
    path = fx.make_fixture("leak_1grp", spec)
    results = []
    for reader in ["pyarrow", "arrow_rs"]:
        d = _run_dir(f"leak__{reader}")
        _fresh_session(reader, d)
        _warm(reader, d)
        _note_fixture(d, path)
        windows = []
        for i in range(8):
            t0 = time.time()
            consume(ray.data.read_parquet(path), "iter_batches")
            t1 = time.time()
            windows.append({"i": i, "t_start": t0, "t_end": t1, "wall_s": t1 - t0})
            print(f"  leak {reader} iter {i}: wall={t1-t0:.3f}s")
        json.dump(
            {"reader": reader, "windows": windows},
            open(os.path.join(d, "windows.json"), "w"),
        )
        ray.shutdown()
        time.sleep(0.3)
        results.append(_R(d, {"reader": reader, "windows": windows}))
    json.dump(results, open(os.path.join(OUT, "results_leak.json"), "w"), indent=2)
    return results


def axis_leak_multigrp():
    """Reproduce the apache/arrow#39808 / ray#49158 accumulation *inside Ray*, with
    the geometry the issue is actually about: MANY small row groups feeding ONE
    read task. ``axis_leak`` uses a single row group, where "whole file" == "one
    row group" so the leak signature can't even appear (see the leak-signature
    unit test's 16-group fixture); this axis fixes that.

    Fixture: 6M rows of fat strings, ``row_group_size=200_000`` => 30 row groups,
    one file (~900 MB decoded, ~30 MB per group).

    Two "relevant settings" are pinned so the READER's working set is what USS
    measures, not Ray's block packing (the confound in ``axis_leak``: a fixture
    that fits inside one output block is held whole by every reader regardless of
    how it decodes):
      1. ``ctx.parquet_chunker_target_chunk_size`` — how row groups map to read
         tasks (the knob the ParquetFileChunker resolves from; default falls back
         to ``target_min_block_size`` = 1 MiB):
           * ``one_chunk`` — target huge: all 30 groups land in ONE read task, so
             the V2 scanner spans the whole file in a single ``scan_batches`` — the
             #39808 leak geometry (also what a single fat row group or a raised
             block-size target hits).
           * ``per_group`` — target 0: one row group per read task — the bounded
             geometry V2 normally produces (the "fixed" case).
      2. ``ctx.target_max_block_size`` is dropped to 8 MiB and the consume is
         ``decode_drop`` (touch each batch, keep only the row count) so output
         blocks don't accumulate the whole file and mask the reader. What's left in
         USS is the reader's own decode transient: the scanner's whole-scan
         accumulation vs iter_batches' one-row-group floor vs arrow-rs' byte budget.

    For the first two readers (``pyarrow`` scanner and ``pyarrow_iter``) we sweep
    ``pre_buffer`` on/off: on holds the fragment's compressed column chunks
    alongside the decoded output, off streams them. ``arrow_rs`` runs once (its
    analog is the byte budget / fetch window, not pre_buffer). Peak + per-task
    incremental USS shows how each reacts.
    """
    spec = {
        "rows": 6_000_000,
        "num_files": 1,
        "row_group_size": 200_000,
        "schema": "huge_str",
    }
    path = fx.make_fixture("leak_multigrp_30rg", spec)
    small_block = 8 * MB  # shrink output blocks so the reader transient is visible
    # (reader, pre_buffer): the first two readers sweep pre_buffer; arrow_rs once.
    reader_configs = [
        ("pyarrow", True),
        ("pyarrow", False),
        ("pyarrow_iter", True),
        ("pyarrow_iter", False),
        ("arrow_rs", None),
    ]
    # chunk_mode -> target_chunk_size handed to the chunker via ctx. HUGE bundles
    # every row group into one read task; 0 forces one row group per task.
    HUGE = 8 * 1024 * MB
    chunk_modes = [("one_chunk", HUGE), ("per_group", 0)]
    results = []
    for chunk_mode, target in chunk_modes:
        for reader, pre_buffer in reader_configs:
            pb = "na" if pre_buffer is None else ("on" if pre_buffer else "off")
            label = f"leakmg__{chunk_mode}__{reader}__pb_{pb}"
            d = _run_dir(label)
            ctx = _fresh_session(reader, d, pre_buffer=pre_buffer)
            _warm(reader, d)
            _note_fixture(d, path)
            # Pin the row-group -> read-task grouping and shrink output blocks for
            # the measured read only (after warm, which uses ordinary settings).
            ctx.parquet_chunker_target_chunk_size = target
            ctx.target_max_block_size = small_block
            t0 = time.time()
            rows = consume(ray.data.read_parquet(path), "decode_drop")
            t1 = time.time()
            ray.shutdown()
            time.sleep(0.3)
            peak = _node_sum_peak_mb(d, t0, t1)
            incr = _node_sum_incr_peak_mb(d, t0, t1)
            nat, fb = _count_paths(d)
            wk = _worker_breakdown(d, t0, t1)
            results.append(
                _R(
                    d,
                    {
                        "config": f"{chunk_mode}_pb-{pb}",
                        "chunk_mode": chunk_mode,
                        "reader": reader,
                        "pre_buffer": pb,
                        "wall_s": t1 - t0,
                        "node_sum_peak_mb": peak,
                        "node_sum_incr_mb": incr,
                        "rows": rows,
                        "t0": t0,
                        "t1": t1,
                        "native": nat,
                        "fallback": fb,
                        # read-task geometry is fixed by chunk_mode (one_chunk = 1
                        # task spanning all groups; per_group = 1 task per group).
                        # n_grown is grown WORKERS (1 here — preserve_order
                        # serializes the tasks onto one worker at a time), not the
                        # task count.
                        "grown_workers": wk["n_grown"],
                        "max_task_mb": wk["max_worker_incr_mb"],
                        "workers": wk,
                    },
                )
            )
            _report(
                f"leakmg {chunk_mode} {reader} pb={pb}", t1 - t0, peak, incr, nat, fb,
                extra={
                    "rows": rows,
                    "grown": f"{wk['n_grown']}/{wk['n_workers']}",
                    "max task": f"{wk['max_worker_incr_mb']:.0f} MB",
                },
            )
    json.dump(
        results, open(os.path.join(OUT, "results_leak_multigrp.json"), "w"), indent=2
    )
    return results


def axis_reader_settings():
    """How the READER's own settings move its memory, in the one-chunk leak
    geometry (all 30 groups in one read task, so a whole-file-scale working set is
    possible). Same fixture and isolation as ``axis_leak_multigrp`` (8 MiB output
    blocks + ``decode_drop`` so what USS shows is the reader's decode transient,
    not Ray's block packing). Each config varies ONE knob off its default:

      pyarrow (scanner):
        * batch_readahead {1,2,8(default),32} — decoded batches held ahead per
          fragment. The prime suspect for the scanner's ~140 MB working set: it
          should scale ~linearly here.
        * pre_buffer {on(default),off} — compressed column chunks coalesced up
          front. Expected ~no local effect (it is an S3 I/O knob).
        * buffer_size {1,8(default),64 MiB} — per-stream compressed read buffer.
      pyarrow_iter:
        * pre_buffer {off(default),on}
        * use_threads {off(default),on} — threaded decode is faster but holds
          more of the row group at once.
      arrow_rs:
        * decode_budget {2,8(default),32 MiB} — its explicit working-set floor.
    """
    spec = {
        "rows": 6_000_000,
        "num_files": 1,
        "row_group_size": 200_000,
        "schema": "huge_str",
    }
    path = fx.make_fixture("leak_multigrp_30rg", spec)
    small_block = 8 * MB
    HUGE = 8 * 1024 * MB
    # (reader, setting-label, extra worker env, arrow_rs budget MB or None)
    configs = [
        ("pyarrow", "scanner_default", {}, None),
        ("pyarrow", "batch_readahead=1", {"RAY_DATA_ARROW_SCANNER_BATCH_READAHEAD": 1}, None),  # noqa: E501
        ("pyarrow", "batch_readahead=2", {"RAY_DATA_ARROW_SCANNER_BATCH_READAHEAD": 2}, None),  # noqa: E501
        ("pyarrow", "batch_readahead=32", {"RAY_DATA_ARROW_SCANNER_BATCH_READAHEAD": 32}, None),  # noqa: E501
        ("pyarrow", "pre_buffer=off", {"RAY_DATA_PARQUET_PRE_BUFFER": 0}, None),
        ("pyarrow", "buffer_size=1MB", {"RAY_DATA_PARQUET_FRAGMENT_BUFFER_SIZE": 1 * MB}, None),  # noqa: E501
        ("pyarrow", "buffer_size=64MB", {"RAY_DATA_PARQUET_FRAGMENT_BUFFER_SIZE": 64 * MB}, None),  # noqa: E501
        ("pyarrow_iter", "iter_default", {}, None),
        ("pyarrow_iter", "iter_pre_buffer=on", {"RAY_DATA_PARQUET_PRE_BUFFER": 1}, None),
        ("pyarrow_iter", "iter_use_threads=on", {"RAY_DATA_PARQUET_ITER_USE_THREADS": 1}, None),  # noqa: E501
        ("arrow_rs", "budget=2MB", {}, 2),
        ("arrow_rs", "budget=8MB", {}, 8),
        ("arrow_rs", "budget=32MB", {}, 32),
    ]
    results = []
    for reader, setting, extra_env, budget_mb in configs:
        d = _run_dir(f"rknob__{reader}__{setting}")
        kw = {"extra_env": extra_env}
        if budget_mb is not None:
            kw["budget_bytes"] = budget_mb * MB
        ctx = _fresh_session(reader, d, **kw)
        _warm(reader, d)
        _note_fixture(d, path)
        ctx.parquet_chunker_target_chunk_size = HUGE  # all groups -> one read task
        ctx.target_max_block_size = small_block
        t0 = time.time()
        rows = consume(ray.data.read_parquet(path), "decode_drop")
        t1 = time.time()
        ray.shutdown()
        time.sleep(0.3)
        peak = _node_sum_peak_mb(d, t0, t1)
        incr = _node_sum_incr_peak_mb(d, t0, t1)
        nat, fb = _count_paths(d)
        wk = _worker_breakdown(d, t0, t1)
        results.append(
            _R(
                d,
                {
                    "config": setting,
                    "reader": reader,
                    "setting": setting,
                    "wall_s": t1 - t0,
                    "node_sum_peak_mb": peak,
                    "node_sum_incr_mb": incr,
                    "max_task_mb": wk["max_worker_incr_mb"],
                    "rows": rows,
                    "t0": t0,
                    "t1": t1,
                    "native": nat,
                    "fallback": fb,
                    "workers": wk,
                },
            )
        )
        _report(
            f"rknob {reader} {setting}", t1 - t0, peak, incr, nat, fb,
            extra={"rows": rows, "max task": f"{wk['max_worker_incr_mb']:.0f} MB"},
        )
    json.dump(
        results, open(os.path.join(OUT, "results_reader_settings.json"), "w"), indent=2
    )
    return results


def axis_leak_rgsize():
    """The ray#49158 leak decomposed by row-group SIZE, V1 vs V2 vs iter vs arrow-rs.

    Reproduces the two mechanisms found on 2026-07-27 (see Agents.md), using the
    #49158 shape: FEW rows, HUGE binary cells (``blob`` schema, ~256 KiB/row). The
    same data is written two ways so only the row-group geometry changes:

      * ``many_tiny`` — ``row_group_size=16`` → ~200 tiny row groups. This is the
        churn case. A V1 whole-file scanner spans all 200 groups and SURGES to the
        whole-file working set; V2 fans one group per read task and stays bounded.
        It is ALSO where arrow-rs can read *high* on RSS despite a smaller live
        working set: 200 rapid decode/free cycles churn the crate's system
        allocator, which retains freed pages, while PyArrow's jemalloc releases
        them (a macOS-libmalloc artifact — on Linux glibc, re-run with
        ``MALLOC_ARENA_MAX=2`` or an ``LD_PRELOAD`` jemalloc via the env levers
        _fresh_session reads, and the gap should close).
      * ``few_large`` — ``row_group_size`` = whole file / 4 → 4 big (~200 MB)
        groups. This is the decode-FLOOR case: per-group fanning can't help because
        one group IS huge, so the PyArrow scanner must materialize the whole group
        while arrow-rs's byte budget streams it. arrow-rs's intended win.

    Readers: ``pyarrow_v1`` (legacy whole-file scanner — the actual #49158 path),
    ``pyarrow`` (V2 scanner), ``pyarrow_iter`` (V2 iter_batches), ``arrow_rs``.
    Output blocks are shrunk to 8 MiB and the consume is ``decode_drop`` so USS
    shows the reader's own transient, not Ray's retained output (same isolation as
    ``axis_leak_multigrp``). ``max_task_mb`` — the busiest worker's windowed USS
    growth — is the reader's working set; the node-sum trajectory over time
    (graphed by summarize.plot_leak_rgsize) shows the V1 surge vs the V2 plateau.
    """
    # ~256 KiB cells × 3200 rows ≈ 800 MB per fixture; identical data, two layouts.
    ROWS = 3200
    many_tiny = fx.make_fixture(
        "leakrg_blob_manytiny",
        {"rows": ROWS, "num_files": 1, "row_group_size": 16, "schema": "blob"},
    )
    few_large = fx.make_fixture(
        "leakrg_blob_fewlarge",
        {"rows": ROWS, "num_files": 1, "row_group_size": ROWS // 4, "schema": "blob"},
    )
    geometries = [("many_tiny", many_tiny), ("few_large", few_large)]
    readers = ["pyarrow_v1", "pyarrow", "pyarrow_iter", "arrow_rs"]
    small_block = 8 * MB
    results = []
    for geom, path in geometries:
        for reader in readers:
            d = _run_dir(f"leakrg__{geom}__{reader}")
            ctx = _fresh_session(reader, d)
            _warm(reader, d)
            _note_fixture(d, path)
            # Shrink output blocks for the measured read (after warm) so retained
            # output doesn't mask the reader's decode transient.
            ctx.target_max_block_size = small_block
            t0 = time.time()
            rows = consume(ray.data.read_parquet(path), "decode_drop")
            t1 = time.time()
            ray.shutdown()
            time.sleep(0.3)
            peak = _node_sum_peak_mb(d, t0, t1)
            incr = _node_sum_incr_peak_mb(d, t0, t1)
            nat, fb = _count_paths(d)
            wk = _worker_breakdown(d, t0, t1)
            results.append(
                _R(
                    d,
                    {
                        "config": f"{geom}_{reader}",
                        "geom": geom,
                        "reader": reader,
                        "wall_s": t1 - t0,
                        "node_sum_peak_mb": peak,
                        "node_sum_incr_mb": incr,
                        "max_task_mb": wk["max_worker_incr_mb"],
                        "rows": rows,
                        "t0": t0,
                        "t1": t1,
                        "native": nat,
                        "fallback": fb,
                        "workers": wk,
                    },
                )
            )
            _report(
                f"leakrg {geom} {reader}", t1 - t0, peak, incr, nat, fb,
                extra={
                    "rows": rows,
                    "grown": f"{wk['n_grown']}/{wk['n_workers']}",
                    "max task": f"{wk['max_worker_incr_mb']:.0f} MB",
                },
            )
    json.dump(
        results, open(os.path.join(OUT, "results_leak_rgsize.json"), "w"), indent=2
    )
    return results


def axis_mixed():
    # 7 files, each a different schema with a different bytes/row, in one dataset
    # dir read as a single dataset. Tests (a) whether arrow-rs's per-group byte
    # budget adapts file-to-file against PyArrow's fixed sizing, and (b) that a
    # mixed dataset with a NON-native schema routes cleanly: arrow-rs takes the
    # 6 flat/struct files natively (struct ungated 2026-07-21) and falls back to
    # PyArrow for the ray_tensor file, in one read, without breaking.
    path = fx.make_mixed_fixture("mixed7_tensor", per=400_000)
    results = []
    for reader in READERS:
        rd = _run_dir(f"mixed__{reader}")
        _fresh_session(reader, rd)
        _warm(reader, rd)
        _note_fixture(rd, path)
        t0 = time.time()
        rows = consume(ray.data.read_parquet(path), "iter_batches")
        t1 = time.time()
        wall = t1 - t0
        ray.shutdown()
        time.sleep(0.3)
        nat, fb = _count_paths(rd)
        peak = _node_sum_peak_mb(rd, t0, t1)
        incr = _node_sum_incr_peak_mb(rd, t0, t1)
        wk = _worker_breakdown(rd, t0, t1)
        results.append(
            _R(
                rd,
                {
                    "reader": reader,
                    "wall_s": wall,
                    "rows": rows,
                    "t0": t0,
                    "t1": t1,
                    "node_sum_peak_mb": peak,
                    "node_sum_incr_mb": incr,
                    "native": nat,
                    "fallback": fb,
                    "workers": wk,
                },
            )
        )
        _report(
            f"mixed {reader}", wall, peak, incr, nat, fb,
            extra={
                "rows": rows,
                "workers": f"{wk['n_grown']}/{wk['n_workers']} grown  "
                f"(max task {wk['max_worker_incr_mb']:.0f} MB)",
            },
        )
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
        series.append(
            (
                np.array([float(r[0]) for r in rows]),
                np.array([float(r[1]) for r in rows]),
            )
        )
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
        series.append(
            (
                np.array([float(r[0]) for r in rows]),
                np.array([float(r[1]) for r in rows]),
            )
        )
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
    return {
        "n_workers": n,
        "n_grown": grown,
        "max_worker_incr_mb": round(max_incr, 1),
        "max_worker_minmax_mb": round(max_minmax, 1),
    }


def _report(label, wall, peak=None, incr=None, native=None, fallback=None, extra=None):
    """One clean, uniform per-run block (mirrors mem3.py) so a sweep reads as a
    stack of labelled blocks instead of cramped one-liners.

    The ABSOLUTE node-sum peak is the metric of record here -- it's what the OOM
    killer / Ray's memory monitor act on, and the only memory column summary.csv
    keeps. ``incr`` (baseline-subtracted) is a secondary DIAGNOSTIC, not a ranking
    number: subtracting each worker's warm baseline systematically flatters
    whichever reader retains more warm heap (Agents.md §3.5), so the old one-liners
    that led with ``incr=`` were quietly disagreeing with the CSV. The decode-path
    line prints only when the arrow-rs trace wrote something (pyarrow leaves it
    blank -- absence + native=0 is itself the "PyArrow ran" signal)."""
    print(f"\n===== {label} =====")
    print(f"wall           : {wall:6.3f}s")
    if peak is not None:
        print(f"node-sum peak  : {peak:7.0f} MB   (absolute -- OOM killer / summary.csv)")
    if incr is not None:
        print(f"node-sum incr  : {incr:7.0f} MB   (read-caused delta -- diagnostic only)")
    if native or fallback:
        n, fb = native or 0, fallback or 0
        verdict = "ARROW-RS" if n and not fb else "MIXED/FALLBACK"
        print(f"decode path    : native={n}  fallback={fb}  ({verdict})")
    for k, v in (extra or {}).items():
        print(f"{k:<15}: {v}")


def axis_scaling():
    """Is arrow-rs O(n) or O(n^2) on ONE big row group? If a per-batch reader
    rebuild (growing RowSelection skip) lurked, wall/row would GROW with row count.
    Sweep rows on a single-row-group file at a fixed budget and check wall/row is
    ~flat. Also sweep the budget (fewer/more batches) — O(n^2) would blow up as the
    budget shrinks (more rebuilds); O(n) per-batch-overhead does the opposite."""
    results = []
    for rows in [1_000_000, 2_000_000, 4_000_000, 8_000_000]:
        spec = {
            "rows": rows,
            "num_files": 1,
            "row_group_size": rows,
            "schema": "wide_str",
        }
        path = fx.make_fixture(f"scale_{rows}", spec)
        for reader in READERS:
            d = _run_dir(f"scale__{rows}__{reader}")
            _fresh_session(reader, d)
            _warm(reader, d)
            _note_fixture(d, path)
            t0 = time.time()
            consume(ray.data.read_parquet(path), "iter_batches")
            wall = time.time() - t0
            ray.shutdown()
            time.sleep(0.3)
            results.append(
                _R(
                    d,
                    {
                        "rows": rows,
                        "reader": reader,
                        "wall_s": wall,
                        "us_per_row": wall / rows * 1e6,
                    },
                )
            )
            print(
                f"  scaling {reader} {rows//1_000_000}M: wall={wall:.3f}s "
                f"({wall / rows * 1e6:.4f} us/row)"
            )
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
        "medium_8x1M": {
            "rows": 8_000_000,
            "num_files": 8,
            "row_group_size": 1_000_000,
            "schema": "wide_str",
        },
        "big_4x4M": {
            "rows": 16_000_000,
            "num_files": 4,
            "row_group_size": 4_000_000,
            "schema": "wide_str",
        },
    }
    for fxname, spec in fixtures.items():
        path = fx.make_fixture(f"conc_{fxname}", spec)
        for reader in READERS:
            for ncpu in [2, 4]:
                d = _run_dir(f"conc__{fxname}__{reader}__cpu{ncpu}")
                _fresh_session(reader, d, num_cpus=ncpu)
                _warm(reader, d)
                _note_fixture(d, path)
                t0 = time.time()
                rows = consume(ray.data.read_parquet(path), "iter_batches")
                t1 = time.time()
                ray.shutdown()
                time.sleep(0.3)
                peak = _node_sum_peak_mb(d, t0, t1)
                incr = _node_sum_incr_peak_mb(d, t0, t1)
                nat, fb = _count_paths(d)
                results.append(
                    _R(
                        d,
                        {
                            "fixture": fxname,
                            "reader": reader,
                            "num_cpus": ncpu,
                            "wall_s": t1 - t0,
                            "node_sum_peak_mb": peak,
                            "node_sum_incr_mb": incr,
                            "rows": rows,
                            "t0": t0,
                            "t1": t1,
                            "native": nat,
                            "fallback": fb,
                        },
                    )
                )
                _report(
                    f"conc {fxname} {reader} cpu={ncpu}", t1 - t0, peak, incr, nat, fb,
                    extra={"rows": rows},
                )
    json.dump(
        results, open(os.path.join(OUT, "results_concurrency.json"), "w"), indent=2
    )
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
        "small_many_rg": {
            "rows": 2_000_000,
            "num_files": 1,
            "row_group_size": 50_000,
            "schema": S,
        },
        "medium_1rg": {
            "rows": 2_000_000,
            "num_files": 1,
            "row_group_size": 2_000_000,
            "schema": S,
        },
        "large_1rg": {
            "rows": 8_000_000,
            "num_files": 1,
            "row_group_size": 8_000_000,
            "schema": S,
        },
        "mixed_rg": {
            "rows": 4_000_000,
            "num_files": 1,
            "row_group_sizes": [500_000, 20_000, 500_000, 20_000],
            "schema": S,
        },
        "many_files_1rg": {
            "rows": 8_000_000,
            "num_files": 4,
            "row_group_size": 2_000_000,
            "schema": S,
        },
    }
    results = []
    for name, spec in configs.items():
        path = fx.make_fixture(f"show_{name}", spec)
        for reader in READERS:
            d = _run_dir(f"show__{name}__{reader}")
            _fresh_session(reader, d, num_cpus=4)
            _warm(reader, d)
            _note_fixture(d, path)
            prog = os.path.join(d, "progress.csv")
            t0 = time.time()
            rows = consume(ray.data.read_parquet(path), "iter_batches", prog)
            t1 = time.time()
            ray.shutdown()
            time.sleep(0.3)
            peak = _node_sum_peak_mb(d, t0, t1)
            incr = _node_sum_incr_peak_mb(d, t0, t1)
            nat, fb = _count_paths(d)
            results.append(
                _R(
                    d,
                    {
                        "config": name,
                        "reader": reader,
                        "wall_s": t1 - t0,
                        "node_sum_peak_mb": peak,
                        "node_sum_incr_mb": incr,
                        "rows": rows,
                        "t0": t0,
                        "t1": t1,
                        "native": nat,
                        "fallback": fb,
                    },
                )
            )
            _report(
                f"showcase {name} {reader}", t1 - t0, peak, incr, nat, fb,
                extra={"rows": rows},
            )
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
        for reader in READERS:
            d = _run_dir(f"sweep_{name}__{lv['label']}__{reader}")
            _fresh_session(reader, d, budget_bytes=budget, num_cpus=ncpu)
            _warm(reader, d)
            _note_fixture(d, path)
            prog = os.path.join(d, "progress.csv")
            t0 = time.time()
            rows = consume(ray.data.read_parquet(path), mode, prog)
            t1 = time.time()
            ray.shutdown()
            time.sleep(0.3)
            peak = _node_sum_peak_mb(d, t0, t1)
            incr = _node_sum_incr_peak_mb(d, t0, t1)
            nat, fb = _count_paths(d)
            wk = _worker_breakdown(d, t0, t1)
            results.append(
                _R(
                    d,
                    {
                        "sweep": name,
                        "level": lv["label"],
                        "reader": reader,
                        "wall_s": t1 - t0,
                        "node_sum_peak_mb": peak,
                        "node_sum_incr_mb": incr,
                        "rows": rows,
                        "t0": t0,
                        "t1": t1,
                        "budget_mb": budget // MB,
                        "num_cpus": ncpu,
                        "native": nat,
                        "fallback": fb,
                        "workers": wk,
                    },
                )
            )
            _report(
                f"sweep[{name}] {lv['label']} {reader}", t1 - t0, peak, incr, nat, fb,
                extra={
                    "rows": rows,
                    "workers": f"{wk['n_grown']}/{wk['n_workers']} grown  "
                    f"(max task {wk['max_worker_incr_mb']:.0f} MB)",
                },
            )
    json.dump(
        results, open(os.path.join(OUT, f"results_sweep_{name}.json"), "w"), indent=2
    )
    return results


def sweep_size():
    """Same schema (flat int64, one big row group), 5 decoded sizes from ~14 MB
    to ~1.4 GB. Shows the memory gap widening with size while arrow-rs stays flat."""
    # _ints => id + 8 int64 cols = 72 bytes/row. rows chosen for ~14/50/144/504/1440 MB.
    levels = [
        {
            "label": "14MB_200k",
            "fixture_name": "sw_size_200k",
            "spec": {
                "rows": 200_000,
                "num_files": 1,
                "row_group_size": 200_000,
                "schema": "int",
            },
        },
        {
            "label": "50MB_700k",
            "fixture_name": "sw_size_700k",
            "spec": {
                "rows": 700_000,
                "num_files": 1,
                "row_group_size": 700_000,
                "schema": "int",
            },
        },
        {
            "label": "144MB_2M",
            "fixture_name": "sw_size_2M",
            "spec": {
                "rows": 2_000_000,
                "num_files": 1,
                "row_group_size": 2_000_000,
                "schema": "int",
            },
        },
        {
            "label": "504MB_7M",
            "fixture_name": "sw_size_7M",
            "spec": {
                "rows": 7_000_000,
                "num_files": 1,
                "row_group_size": 7_000_000,
                "schema": "int",
            },
        },
        {
            "label": "1.4GB_20M",
            "fixture_name": "sw_size_20M",
            "spec": {
                "rows": 20_000_000,
                "num_files": 1,
                "row_group_size": 20_000_000,
                "schema": "int",
            },
        },
    ]
    return _run_sweep("size", levels)


def sweep_batch():
    """Same file (huge_str, 4 M rows, one ~400 MB row group), 5 decode budgets.
    arrow-rs's peak tracks the budget (memory is a knob); PyArrow ignores it
    (materializes the whole group) — a flat control line across all 5 panels."""
    spec = {
        "rows": 4_000_000,
        "num_files": 1,
        "row_group_size": 4_000_000,
        "schema": "huge_str",
    }
    levels = [
        {
            "label": f"{b}MB",
            "fixture_name": "sw_batch_4M",
            "spec": spec,
            "budget_bytes": b * MB,
        }
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
        {
            "label": f"rg_{rg//1000}k" if rg < R else "rg_whole_file",
            "fixture_name": f"sw_rg_{rg}",
            "spec": {"rows": R, "num_files": 1, "row_group_size": rg, "schema": S},
        }
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
        {
            "label": f"{nf}_files",
            "fixture_name": f"sw_files_{nf}",
            "spec": {
                "rows": nf * 2_000_000,
                "num_files": nf,
                "row_group_size": 2_000_000,
                "schema": S,
            },
            "num_cpus": 4,
        }
        for nf in [1, 2, 4, 6, 8]
    ]
    return _run_sweep("files", levels)


def sweep_schema():
    """Same size (2 M rows, one big row group), vary the column dtype: the memory
    gap is biggest where cells are widest (wide strings) and smallest for fixed
    width numerics."""
    levels = [
        {
            "label": sc,
            "fixture_name": f"sw_schema_{sc}",
            "spec": {
                "rows": 2_000_000,
                "num_files": 1,
                "row_group_size": 2_000_000,
                "schema": sc,
            },
        }
        for sc in ["int", "float", "wide_str", "large_str", "huge_str"]
    ]
    return _run_sweep("schema", levels)


def sweep_batch_dd():
    """The decode budget sweep in decode_drop mode — output retention is removed,
    so arrow-rs's memory now TRACKS the budget (the knob that iter_batches masks),
    while PyArrow still holds the whole row group regardless."""
    spec = {
        "rows": 4_000_000,
        "num_files": 1,
        "row_group_size": 4_000_000,
        "schema": "huge_str",
    }
    levels = [
        {
            "label": f"{b}MB",
            "fixture_name": "sw_batch_4M",
            "spec": spec,
            "budget_bytes": b * MB,
        }
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
    spec = {
        "rows": 4_000_000,
        "num_files": 1,
        "row_group_size": 4_000_000,
        "schema": "int",
    }
    path = fx.make_fixture("wl_int_4M", spec)
    high = (1 << 30) - 2000  # selective: only a sliver of i0 exceeds this
    results = []
    for reader in READERS:
        # Aggregation: sum one column. Full decode, output = one scalar.
        d = _run_dir(f"wl__sum__{reader}")
        _fresh_session(reader, d)
        _warm(reader, d)
        _note_fixture(d, path)
        t0 = time.time()
        total = ray.data.read_parquet(path).sum("i0")
        t1 = time.time()
        ray.shutdown()
        time.sleep(0.3)
        peak = _node_sum_peak_mb(d, t0, t1)
        incr = _node_sum_incr_peak_mb(d, t0, t1)
        nat, fb = _count_paths(d)
        wk = _worker_breakdown(d, t0, t1)
        results.append(
            _R(
                d,
                {
                    "workload": "sum(i0)",
                    "reader": reader,
                    "wall_s": t1 - t0,
                    "node_sum_peak_mb": peak,
                    "node_sum_incr_mb": incr,
                    "out": str(total),
                    "t0": t0,
                    "t1": t1,
                    "native": nat,
                    "fallback": fb,
                    "workers": wk,
                },
            )
        )
        _report(
            f"workload sum(i0) {reader}", t1 - t0, peak, incr, nat, fb,
            extra={
                "workers": f"{wk['n_grown']}/{wk['n_workers']} grown  "
                f"(max task {wk['max_worker_incr_mb']:.0f} MB)",
            },
        )
        # Selective read-time filter: decode all, keep a sliver. Non-empty
        # projection so the arrow-rs path runs (count() would empty-project → fallback).
        d = _run_dir(f"wl__filter__{reader}")
        _fresh_session(reader, d)
        _warm(reader, d)
        _note_fixture(d, path)
        t0 = time.time()
        kept = consume(
            ray.data.read_parquet(path).filter(expr=f"i0 > {high}"), "iter_batches"
        )
        t1 = time.time()
        ray.shutdown()
        time.sleep(0.3)
        peak = _node_sum_peak_mb(d, t0, t1)
        incr = _node_sum_incr_peak_mb(d, t0, t1)
        nat, fb = _count_paths(d)
        wk = _worker_breakdown(d, t0, t1)
        results.append(
            _R(
                d,
                {
                    "workload": "filter(i0>hi)",
                    "reader": reader,
                    "wall_s": t1 - t0,
                    "node_sum_peak_mb": peak,
                    "node_sum_incr_mb": incr,
                    "kept_rows": kept,
                    "t0": t0,
                    "t1": t1,
                    "native": nat,
                    "fallback": fb,
                    "workers": wk,
                },
            )
        )
        _report(
            f"workload filter(i0>hi) {reader}", t1 - t0, peak, incr, nat, fb,
            extra={
                "kept rows": kept,
                "workers": f"{wk['n_grown']}/{wk['n_workers']} grown  "
                f"(max task {wk['max_worker_incr_mb']:.0f} MB)",
            },
        )
    json.dump(results, open(os.path.join(OUT, "results_workloads.json"), "w"), indent=2)
    return results


def axis_kfan():
    """Does raising the intra-row-group split K help on a LONE big row group?

    K splits the single row group into K contiguous row-RANGES, each handled by
    its own concurrent worker that does its OWN byte-range fetch AND its own decode
    for that slice -- so K parallelizes I/O and decode together, as a data-parallel
    split (it is NOT pyarrow-style pipeline overlap; within a range decode is still
    sequential). Locally there is no network latency to hide, so the I/O half is
    moot; the only lever that can pay off is decode-parallelism across cores. The
    cost is memory: each of the K ranges holds its own decode transient, so peak
    grows toward ~K x the K=1 working set.

    This axis makes that trade explicit on the arrow-rs thesis fixture (one 4M-row
    row group > the block-size split threshold, so K actually engages): PyArrow as
    the reference, then arrow-rs at K=1,2,4. If peak climbs with K while wall stays
    flat, the K=1 memory win was partly "arrow-rs did less parallel work", and a
    tuned-up (K=4) arrow-rs lands closer to PyArrow's node-sum peak."""
    spec = {
        "rows": 4_000_000,
        "num_files": 1,
        "row_group_size": 4_000_000,
        "schema": "wide_str",
    }
    path = fx.make_fixture("kfan_4M_1rg", spec)
    results = []
    # PyArrow reference (K is a no-op for it), then arrow-rs across K.
    runs = [("pyarrow", 1)] + [("arrow_rs", k) for k in (1, 2, 4)]
    for reader, k in runs:
        d = _run_dir(f"kfan__k{k}__{reader}")
        _fresh_session(reader, d, k=k)
        _warm(reader, d)
        _note_fixture(d, path)
        t0 = time.time()
        rows = consume(ray.data.read_parquet(path), "iter_batches")
        t1 = time.time()
        ray.shutdown()
        time.sleep(0.3)
        peak = _node_sum_peak_mb(d, t0, t1)
        incr = _node_sum_incr_peak_mb(d, t0, t1)
        nat, fb = _count_paths(d)
        wk = _worker_breakdown(d, t0, t1)
        results.append(
            _R(
                d,
                {
                    "reader": reader,
                    "k": k,
                    "wall_s": t1 - t0,
                    "node_sum_peak_mb": peak,
                    "node_sum_incr_mb": incr,
                    "rows": rows,
                    "t0": t0,
                    "t1": t1,
                    "native": nat,
                    "fallback": fb,
                    "workers": wk,
                },
            )
        )
        _report(
            f"kfan K={k} {reader}", t1 - t0, peak, incr, nat, fb,
            extra={
                "rows": rows,
                "workers": f"{wk['n_grown']}/{wk['n_workers']} grown  "
                f"(max task {wk['max_worker_incr_mb']:.0f} MB)",
            },
        )
    json.dump(results, open(os.path.join(OUT, "results_kfan.json"), "w"), indent=2)
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
      * ``prefetch_windows`` — the latency lever. Consecutive fetch windows are
        pipelined this many deep: depth 1 fetches window N+1 only after decoding N
        (strictly serial); depth 2 issues N+1's GET while N decodes, hiding S3
        first-byte latency behind decode; depth 4 adds slack for fetch-time jitter.
        Swept {1, 2, 4} at window 16 / budget 2 / sys. Memory-bounded analog of
        PyArrow's whole-fragment ``pre_buffer``: recovers the latency PyArrow hides
        but at ``prefetch_windows * fetch_window`` compressed in flight (~2 windows),
        not the whole row group. Expect wall to drop 1->2 then flatten, peak to rise
        ~linearly in depth.
      * allocator — the retention lever (§7.8), all env-only / no recompile:
        ``MALLOC_ARENA_MAX=2`` (glibc arena cap) always runs; and if you export
        ``RAY_DATA_ARROW_RS_MIMALLOC_SO`` / ``RAY_DATA_ARROW_RS_JEMALLOC_SO`` to the
        respective ``.so``, an ``LD_PRELOAD`` run for each is added (swaps the whole
        worker's allocator — Python + Rust + PyArrow — sidestepping the compile-time
        mimalloc segfault). PyArrow itself always uses its bundled jemalloc.
    PyArrow is the baseline. K stays 1 for the per-file layout (Ray's pool
    parallelizes files); the lone-big-row-group K-split is a separate fixture.
    """
    # Points at ONE pre-existing dataset (not a prefix): this is the knob-sweep
    # axis. The self-generating geometry sweep (option B) is axis_s3_geom, which
    # owns RAY_DATA_ARROW_RS_S3_BENCH_PATH — so the two never collide in a full run.
    s3_path = os.environ.get("RAY_DATA_ARROW_RS_S3_DATASET")
    if not s3_path:
        print(
            "  (skip s3: set RAY_DATA_ARROW_RS_S3_DATASET=s3://bucket/one_dataset to "
            "sweep window/budget/allocator on a hand-made dataset; for the geometry "
            "sweep use axis s3_geom instead)"
        )
        return []

    # (reader, fetch_window_mb, budget_mb, malloc_arena_max, ld_preload,
    # prefetch_windows) configs. window/budget/arena/preload/prefetch are ignored by
    # the pyarrow baseline. The window, budget, and prefetch sweeps all cross at
    # (window=16, budget=2, prefetch=2, sys) so that point is shared across them.
    configs = [
        ("pyarrow", None, None, None, None, None),
        # --- window sweep @ fixed 2 MiB budget, prefetch 2 (the memory lever) ---
        ("arrow_rs", 4, 2, None, None, 2),
        ("arrow_rs", 16, 2, None, None, 2),
        ("arrow_rs", 64, 2, None, None, 2),
        ("arrow_rs", 0, 2, None, None, 2),  # no window cap (control: win's size)
        # --- budget sweep @ fixed 16 MiB window, prefetch 2 (the floor knob) ---
        ("arrow_rs", 16, 8, None, None, 2),
        ("arrow_rs", 16, 32, None, None, 2),
        # --- prefetch-depth sweep @ window 16 / budget 2 / sys (the latency lever):
        # depth 1 = strictly-serial windows (fetch W_{n+1} only after decode W_n);
        # depth 2 = one fetch hidden behind a decode; depth 4 = tail-jitter slack.
        # Expect wall to drop 1->2 and flatten 2->4, mem to rise ~linearly in depth
        # (each slot is one more window of compressed bytes in flight). ---
        ("arrow_rs", 16, 2, None, None, 1),
        ("arrow_rs", 16, 2, None, None, 4),
        # --- allocator sweep @ window 16 / budget 2 / prefetch 2 (env-only) ---
        ("arrow_rs", 16, 2, 2, None, 2),  # glibc + MALLOC_ARENA_MAX=2
    ]
    # LD_PRELOAD allocator A/B (#9): only added when the .so path is provided, so a
    # box without the lib installed just skips it. Find paths with e.g.
    #   dpkg -L libmimalloc2 | grep '\.so'   /   dpkg -L libjemalloc2 | grep '\.so'
    mi = os.environ.get("RAY_DATA_ARROW_RS_MIMALLOC_SO")
    je = os.environ.get("RAY_DATA_ARROW_RS_JEMALLOC_SO")
    if mi:
        configs.append(("arrow_rs", 16, 2, None, mi, 2))  # LD_PRELOAD mimalloc
    if je:
        configs.append(("arrow_rs", 16, 2, None, je, 2))  # LD_PRELOAD jemalloc
    results = []
    for reader, window, budget_mb, arena, preload, pf in configs:
        alloc = (
            "mi"
            if preload and "mimalloc" in preload
            else "je"
            if preload and "jemalloc" in preload
            else f"arena{arena}"
            if arena
            else "sys"
        )
        tag = (
            reader
            if reader == "pyarrow"
            else f"arrow_rs_w{window}_b{budget_mb}_pf{pf}_{alloc}"
        )
        d = _run_dir(f"s3__{tag}")
        _fresh_session(
            reader,
            d,
            budget_bytes=(budget_mb or 2) * MB,
            fetch_window_mb=(window or 0),
            prefetch_windows=(pf or 2),
            malloc_arena_max=arena,
            ld_preload=preload,
        )
        _warm(reader, d)  # warms worker imports on a tiny local fixture
        _note_fixture(d, s3_path)
        t0 = time.time()
        rows = consume(
            ray.data.read_parquet(s3_path),
            "iter_batches",
            progress_path=os.path.join(d, "progress.csv"),
        )
        t1 = time.time()
        ray.shutdown()
        time.sleep(0.3)
        peak = _node_sum_peak_mb(d, t0, t1)
        incr = _node_sum_incr_peak_mb(d, t0, t1)
        nat, fb = _count_paths(d)
        results.append(
            _R(
                d,
                {
                    "reader": reader,
                    "tag": tag,
                    "fetch_window_mb": window,
                    "budget_mb": budget_mb,
                    "prefetch_windows": pf,
                    "malloc_arena_max": arena,
                    "ld_preload": preload,
                    "alloc": alloc,
                    "wall_s": t1 - t0,
                    "rows": rows,
                    "t0": t0,
                    "t1": t1,
                    "node_sum_peak_mb": peak,
                    "node_sum_incr_mb": incr,
                    "native": nat,
                    "fallback": fb,
                },
            )
        )
        _report(
            f"s3 {tag}", t1 - t0, peak, incr, nat, fb,
            extra={"rows": rows},
        )
    json.dump(results, open(os.path.join(OUT, "results_s3.json"), "w"), indent=2)
    return results


def axis_s3_geom():
    """Option B: the KEY differentiating geometries on REAL S3, self-generating.

    The local axes prove correctness and CPU-bound parity fast, but they cannot
    show the one thing S3 adds — the windowed async fetch, where arrow-rs holds
    only ``fetch_window_mb`` of compressed bytes in flight instead of the whole
    object, so peak is a knob rather than a property of the file. This axis puts
    just the geometries where that matters onto S3 and reads each with PyArrow vs
    arrow-rs (windowed), producing the per-task memory figures (task_mem pairs the
    ``s3geom__<geom>__{pyarrow,arrow_rs}`` run dirs automatically).

    Set ``RAY_DATA_ARROW_RS_S3_BENCH_PATH=s3://bucket/prefix`` (a PREFIX, not a
    file). Fixtures are generated once under it via the normal generator with
    FIXTURE_ROOT temporarily repointed — the OTHER axes stay local. Idempotent:
    a prefix that already holds the parquet is reused, so re-runs don't re-upload.
    """
    root = os.environ.get("RAY_DATA_ARROW_RS_S3_BENCH_PATH")
    if not root or not root.startswith("s3://"):
        print(
            "  (skip s3_geom: set RAY_DATA_ARROW_RS_S3_BENCH_PATH=s3://bucket/prefix "
            "on a Linux+real-S3 box — this is where the windowed-fetch memory win "
            "actually shows, which local/moto cannot)"
        )
        return []

    # The geometries that separate a streaming reader from a whole-group one: a
    # lone big row group (the headline), a decode-heavy wide-string group, many
    # files x 1 group, many tiny groups (the no-win control), and a nested column.
    geometries = {
        "large_1rg": {
            "rows": 8_000_000,
            "num_files": 1,
            "row_group_size": 8_000_000,
            "schema": "wide_str",
        },
        "large_str_1rg": {
            "rows": 2_000_000,
            "num_files": 1,
            "row_group_size": 2_000_000,
            "schema": "large_str",
        },
        "many_files_1rg": {
            "rows": 8_000_000,
            "num_files": 4,
            "row_group_size": 2_000_000,
            "schema": "int",
        },
        "small_many_rg": {
            "rows": 4_000_000,
            "num_files": 1,
            "row_group_size": 100_000,
            "schema": "int",
        },
        "list_1rg": {
            "rows": 2_000_000,
            "num_files": 1,
            "row_group_size": 2_000_000,
            "schema": "list",
        },
    }
    saved_root = fx.FIXTURE_ROOT
    fx.FIXTURE_ROOT = root  # only for this axis; restored in finally
    results = []
    try:
        for name, spec in geometries.items():
            path = fx.make_fixture(f"s3geom_{name}", spec)  # uploads once, then reused
            for reader in READERS:
                d = _run_dir(f"s3geom__{name}__{reader}")
                _fresh_session(
                    reader, d, budget_bytes=2 * MB, fetch_window_mb=16, num_cpus=4
                )
                _warm(reader, d)
                _note_fixture(d, path)
                prog = os.path.join(d, "progress.csv")
                t0 = time.time()
                rows = consume(ray.data.read_parquet(path), "iter_batches", prog)
                t1 = time.time()
                ray.shutdown()
                time.sleep(0.3)
                peak = _node_sum_peak_mb(d, t0, t1)
                incr = _node_sum_incr_peak_mb(d, t0, t1)
                nat, fb = _count_paths(d)
                results.append(
                    _R(
                        d,
                        {
                            "config": name,
                            "reader": reader,
                            "wall_s": t1 - t0,
                            "node_sum_peak_mb": peak,
                            "node_sum_incr_mb": incr,
                            "rows": rows,
                            "t0": t0,
                            "t1": t1,
                            "native": nat,
                            "fallback": fb,
                            "fetch_window_mb": 16 if reader == "arrow_rs" else None,
                        },
                    )
                )
                _report(
                    f"s3geom {name} {reader}", t1 - t0, peak, incr, nat, fb,
                    extra={"rows": rows},
                )
    finally:
        fx.FIXTURE_ROOT = saved_root
    json.dump(results, open(os.path.join(OUT, "results_s3geom.json"), "w"), indent=2)
    return results


AXES = {
    "layout": axis_layout,
    "schema": axis_schema,
    "tuning": axis_tuning,
    "leak": axis_leak,
    "leak_multigrp": axis_leak_multigrp,
    "leak_rgsize": axis_leak_rgsize,
    "reader_settings": axis_reader_settings,
    "mixed": axis_mixed,
    "scaling": axis_scaling,
    "concurrency": axis_concurrency,
    "showcase": axis_showcase,
    "sweep_size": sweep_size,
    "sweep_batch": sweep_batch,
    "sweep_rowgroup": sweep_rowgroup,
    "sweep_files": sweep_files,
    "sweep_schema": sweep_schema,
    "sweep_batch_dd": sweep_batch_dd,
    "workloads": axis_workloads,
    "kfan": axis_kfan,
    "s3": axis_s3,
    "s3_geom": axis_s3_geom,
}


def write_summary_csv():
    """Flatten every runs/results_*.json into ONE runs/summary.csv (also echoed to
    stdout): one row per measured read across ALL axes — the machine-readable
    digest to paste back for analysis. Memory verdicts do NOT live here: the
    metric of record is the per-task absolute-USS-over-time graph vs Ray's
    expectation line (task_mem.py). This table keeps wall time + the raw
    node-sum absolute peak as a node-level sanity number only; the windowed
    incremental columns were retired (baseline subtraction systematically
    flattered whichever reader retains more warm heap — see Agents.md §3.5)."""
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
        if r.get("k") is not None:
            extra.append(f"k{r['k']}")
        if r.get("budget_mb") is not None:
            extra.append(f"b{r['budget_mb']}")
        if r.get("num_cpus") is not None:
            extra.append(f"cpu{r['num_cpus']}")
        return base + (("_" + "_".join(extra)) if extra else "")

    def _num(r, k):
        v = r.get(k)
        return round(v, 3) if isinstance(v, (int, float)) else ""

    fields = [
        "axis",
        "config",
        "reader",
        "wall_s",
        "abs_peak_mb",
        "rows",
        "files",
        "row_groups",
        "rg_uncomp_mb",
        "rg_comp_mb",
        "native",
        "fallback",
        "contents",
    ]
    out_rows = []
    for f in sorted(glob.glob(os.path.join(OUT, "results_*.json"))):
        axis = os.path.basename(f)[len("results_") : -len(".json")]
        try:
            data = json.load(open(f))
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        for r in data:
            if not isinstance(r, dict) or "reader" not in r:
                continue
            out_rows.append(
                {
                    "axis": axis,
                    "config": _config(axis, r),
                    "reader": r.get("reader", ""),
                    "wall_s": _num(r, "wall_s"),
                    "abs_peak_mb": _num(r, "node_sum_peak_mb"),
                    # rows read: axis-native count if present, else the fixture total.
                    "rows": r.get("rows") or r.get("rows_total", ""),
                    "files": r.get("num_files", ""),
                    "row_groups": r.get("num_row_groups", ""),
                    "rg_uncomp_mb": _num(r, "max_rg_uncomp_mb"),
                    "rg_comp_mb": _num(r, "max_rg_comp_mb"),
                    "native": r.get("native", ""),
                    "fallback": r.get("fallback", ""),
                    "contents": r.get("schema_desc", ""),
                }
            )
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
