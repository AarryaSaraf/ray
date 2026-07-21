"""One-command S3 memory/speed benchmark for the arrow-rs Parquet reader.

Run this on a **Linux box with real S3** (moto/macOS can't show the win — moto has
no network latency and macOS USS is only directional). It does the whole thing from
a single command:

    RAY_DATA_ARROW_RS_S3_BENCH_PATH=s3://your-bucket/some/prefix \\
        python run_s3_benchmark.py

Three modes:
  * `smoke`  — fast gate: write one small file, read both ways, assert the crate
    drives the read and row counts agree. ALWAYS run this first.
  * (no arg) — the full single-node memory/speed sweep (the precise headline; run on
    a single fat node — see below).
  * `distributed-check` — attach to the CURRENT multi-node cluster and verify the
    reader works under real distributed scheduling: same row count as PyArrow, crate
    drove every fragment on every node, plus a directional per-node memory readout.

    RAY_DATA_ARROW_RS_S3_BENCH_PATH=s3://your-bucket/some/prefix \\
        python run_s3_benchmark.py smoke              # 1. gate
    RAY_DATA_ARROW_RS_S3_BENCH_PATH=... python run_s3_benchmark.py                # 2. sweep
    RAY_DATA_ARROW_RS_S3_BENCH_PATH=... python run_s3_benchmark.py distributed-check  # 3. verify

The full run (no arg) will, in order:
  1. generate the target fixtures ON S3 if not already there — N files, each ONE big
     row group (the layout Ray sees most), `write_page_index=True`, snappy;
  2. run the sweep: a PyArrow baseline vs arrow-rs across fetch-window sizes
     {4, 16, 64, 0(=no cap)} MB (at a 2 MiB budget), decode-budget sizes {2, 8, 32} MB
     (at a 16 MB window), and a `MALLOC_ARENA_MAX` cap, each reading the same S3
     fixtures under a fresh Ray session with 5 ms node-sum USS sampling;
  3. write the overlaid figures `figs/s3_mem_time.png` (memory over time, PyArrow vs
     each arrow-rs config) and `figs/s3_speed_time.png` (throughput over time), plus a
     peak+wall summary table to stdout and `runs/results_s3.json`.

Prerequisites on the Linux box:
  * AWS credentials in the environment (env vars or an instance role) with read+write
    on the bucket; `AWS_REGION` (or `AWS_DEFAULT_REGION`) set. For MinIO/custom S3,
    also set `AWS_ENDPOINT_URL=http://host:port`.
  * The native crate built + installed:
      cd python/ray/data/_internal/datasource_v2/native/ray_data_arrow_rs
      maturin build --release
      uv pip install --force-reinstall --no-deps target/wheels/*.whl
    (optionally `--features jemalloc` to A/B PyArrow's allocator; see Agents.md §7.8).

Scale knobs (env, optional) — bump these on a big box to make the memory gap obvious:
  RAY_DATA_ARROW_RS_S3_BENCH_ROWS       total rows across all files   (default 8_000_000)
  RAY_DATA_ARROW_RS_S3_BENCH_NUM_FILES  number of files               (default 4)
  RAY_DATA_ARROW_RS_S3_BENCH_SCHEMA     fixtures.py schema key        (default huge_str)
"""
import os
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import fs as pafs

import bench_suite
import fixtures as fx
import summarize


def _s3_base():
    base = os.environ.get("RAY_DATA_ARROW_RS_S3_BENCH_PATH")
    if not base:
        sys.exit(
            "Set RAY_DATA_ARROW_RS_S3_BENCH_PATH=s3://bucket/prefix (with AWS creds "
            "in the environment). This benchmark is real-S3 only — see the module "
            "docstring."
        )
    if not base.startswith("s3://"):
        sys.exit(f"RAY_DATA_ARROW_RS_S3_BENCH_PATH must be an s3:// URI, got: {base}")
    return base.rstrip("/")


def _s3_filesystem():
    """Build a pyarrow S3FileSystem from the ambient env (region + optional custom
    endpoint for MinIO). Only used to WRITE fixtures; the read path builds its own
    filesystem from the URI, exactly as a real `ray.data.read_parquet` call would."""
    kw = {}
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if region:
        kw["region"] = region
    endpoint = os.environ.get("AWS_ENDPOINT_URL")
    if endpoint:
        kw["endpoint_override"] = endpoint
        kw["scheme"] = "http" if endpoint.startswith("http://") else "https"
    return pafs.S3FileSystem(**kw)


def ensure_fixtures(base):
    """Generate the target layout on S3 if absent; return the read URI (s3://...).

    Target = `num_files` files, each a single big row group of `schema` data, with
    `write_page_index=True`. Idempotent: if the prefix already holds .parquet files
    (a prior run), it is reused so re-runs are fast.
    """
    rows = int(os.environ.get("RAY_DATA_ARROW_RS_S3_BENCH_ROWS", 8_000_000))
    num_files = int(os.environ.get("RAY_DATA_ARROW_RS_S3_BENCH_NUM_FILES", 4))
    schema = os.environ.get("RAY_DATA_ARROW_RS_S3_BENCH_SCHEMA", "huge_str")
    if schema not in fx.SCHEMA_BUILDERS:
        sys.exit(f"unknown schema {schema!r}; choose from {sorted(fx.SCHEMA_BUILDERS)}")
    per = rows // num_files

    uri = f"{base}/fixtures/{schema}_{rows}_{num_files}f_1rg"
    key = uri[len("s3://") :]  # bucket/prefix/... (pyarrow fs path, no scheme)
    filesystem = _s3_filesystem()

    sel = pafs.FileSelector(key, allow_not_found=True, recursive=True)
    existing = [f for f in filesystem.get_file_info(sel) if f.path.endswith(".parquet")]
    if len(existing) >= num_files:
        print(f"  fixtures present ({len(existing)} parquet files) at {uri} — reusing")
        return uri

    print(
        f"  generating {num_files} files x {per} rows ({schema}, one big row group "
        f"each) -> {uri}"
    )
    rng = np.random.default_rng(0)
    build = fx.SCHEMA_BUILDERS[schema]
    for f in range(num_files):
        table = pa.table(build(rng, per))
        out_key = f"{key}/part-{f:04d}.parquet"
        # row_group_size == rows-per-file => exactly one row group per file (the case
        # Ray can't split, and where the windowed reader bounds what PyArrow would
        # pre-buffer whole).
        pq.write_table(
            table,
            out_key,
            filesystem=filesystem,
            row_group_size=per,
            write_page_index=True,
            compression="snappy",
        )
        print(f"    wrote s3://{out_key}")
    return uri


def smoke(base):
    """Tiny end-to-end check BEFORE the big suite: writes one small file to S3, reads
    it both ways under real Ray sessions, and asserts (a) the crate imported and drove
    the read (native > 0, fallback == 0 — i.e. we did NOT silently fall back to
    PyArrow), and (b) arrow-rs and PyArrow agree on the row count. Fast (~seconds) and
    cheap; if this fails, fix it before spending money/time on the full sweep."""
    rows = 200_000
    schema = os.environ.get("RAY_DATA_ARROW_RS_SMOKE_SCHEMA", "huge_str")
    uri = f"{base}/smoke/{schema}_{rows}_1f_1rg"
    key = uri[len("s3://") :]
    filesystem = _s3_filesystem()

    sel = pafs.FileSelector(key, allow_not_found=True, recursive=True)
    have = [f for f in filesystem.get_file_info(sel) if f.path.endswith(".parquet")]
    if not have:
        print(f"  writing smoke fixture -> {uri}")
        rng = np.random.default_rng(0)
        table = pa.table(fx.SCHEMA_BUILDERS[schema](rng, rows))
        pq.write_table(
            table,
            f"{key}/part-0000.parquet",
            filesystem=filesystem,
            row_group_size=rows,
            write_page_index=True,
            compression="snappy",
        )
    else:
        print(f"  smoke fixture present at {uri} — reusing")

    import ray

    def _read(reader):
        d = bench_suite._run_dir(f"smoke__{reader}")
        bench_suite._fresh_session(
            reader, d, budget_bytes=2 * bench_suite.MB, fetch_window_mb=16
        )
        n = bench_suite.consume(ray.data.read_parquet(uri), "iter_batches")
        nat, fb = bench_suite._count_paths(d)
        ray.shutdown()
        return n, nat, fb

    pa_rows, _, _ = _read("pyarrow")
    rs_rows, nat, fb = _read("arrow_rs")
    print(
        f"  pyarrow rows={pa_rows}  arrow_rs rows={rs_rows}  "
        f"native={nat} fallback={fb}"
    )
    problems = []
    if rs_rows != pa_rows:
        problems.append(f"row-count mismatch: arrow_rs={rs_rows} vs pyarrow={pa_rows}")
    if nat == 0:
        problems.append("arrow_rs took 0 native fragments (crate not driving the read)")
    if fb != 0:
        problems.append(
            f"arrow_rs fell back to PyArrow on {fb} fragment(s) "
            "(unsupported schema? the sweep would misattribute memory)"
        )
    if problems:
        print("\nSMOKE FAILED:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print(
        "\nSMOKE PASSED — crate drives the read, native-only, row counts agree. "
        "Safe to run the full suite (drop the `smoke` arg)."
    )


def _shared_trace_root():
    """A trace dir visible on EVERY node, so the multi-node check can aggregate
    across the cluster. On Anyscale, ``/mnt/cluster_storage`` is shared cluster-wide.
    Returns None if none is found (the check then reports head-node only)."""
    override = os.environ.get("RAY_DATA_ARROW_RS_SHARED_TRACE_DIR")
    if override:
        return override
    for cand in ("/mnt/cluster_storage", "/mnt/shared_storage"):
        if os.path.isdir(cand):
            return os.path.join(cand, "arrow_rs_bench_traces")
    return None


def _crate_wheel():
    """Path/URI of the built crate wheel to ship to worker nodes via runtime_env.

    Anyscale does NOT propagate a locally `pip install`-ed wheel to worker nodes
    ("Local packages are not supported across cluster"). Ray's own runtime_env DOES
    upload to every node, so for the multi-node check we ship the wheel that way.
    Set RAY_DATA_ARROW_RS_WHEEL to the built .whl (local path or an s3:// URI); if
    unset, auto-detect the newest wheel under the crate's target/wheels/.
    """
    env = os.environ.get("RAY_DATA_ARROW_RS_WHEEL")
    if env:
        return env
    from glob import glob as _glob

    here = os.path.dirname(os.path.abspath(__file__))
    # release/nightly_tests/dataset/arrow_rs_memtrace -> repo python/ray/data/...
    guess = os.path.normpath(
        os.path.join(
            here,
            "..",
            "..",
            "..",
            "..",
            "python",
            "ray",
            "data",
            "_internal",
            "datasource_v2",
            "native",
            "ray_data_arrow_rs",
            "target",
            "wheels",
            "*.whl",
        )
    )
    hits = sorted(_glob(guess))
    return hits[-1] if hits else None


def _per_node_peaks(trace_dir, t0, t1):
    """Group ``uss_<host>_<pid>.csv`` by host, return {host: peak-of-sum USS (MB)}
    within [t0, t1] — the per-node physical memory the concurrent read workers held.
    This is the multi-node analogue of bench_suite._node_sum_peak_mb, split per node."""
    import csv as _csv
    from collections import defaultdict

    import numpy as np

    from glob import glob as _glob

    by_host = defaultdict(list)
    for f in _glob(os.path.join(trace_dir, "uss_*.csv")):
        host = os.path.basename(f)[len("uss_") : -len(".csv")].rsplit("_", 1)[0]
        rows = list(_csv.reader(open(f)))[1:]
        if rows:
            by_host[host].append(
                (
                    np.array([float(r[0]) for r in rows]),
                    np.array([float(r[1]) for r in rows]),
                )
            )
    grid = np.linspace(t0, t1, 500)
    out = {}
    for host, series in by_host.items():
        total = np.zeros_like(grid)
        for ep, uss in series:
            idx = np.searchsorted(ep, grid, side="right") - 1
            total += np.where(idx >= 0, uss[np.clip(idx, 0, len(uss) - 1)], 0.0)
        out[host] = float(total.max()) / bench_suite.MB
    return out


def distributed_check(base):
    """Verification pass on the CURRENT (multi-node) cluster — NOT a fresh local one.

    Attaches to the running workspace cluster (``ray.init(address='auto')``), reads
    the same S3 fixtures both ways, and:
      * asserts correctness under real distributed scheduling — arrow-rs and PyArrow
        return the SAME row count, and the crate drove every fragment on every node
        (native > 0, fallback == 0), aggregated cluster-wide via a SHARED trace dir;
      * reports a per-node USS peak for each reader — a *directional* multi-node
        memory readout (coarser sampling; the precise headline stays the single fat
        node run). Cross-check it against the Anyscale/Ray memory dashboard.

    Needs a trace dir visible on all nodes (Anyscale /mnt/cluster_storage, or set
    RAY_DATA_ARROW_RS_SHARED_TRACE_DIR); without one, readouts are head-node only.
    """
    import glob as _globmod

    import ray

    read_uri = ensure_fixtures(base)  # same fixtures as the single-node run
    wheel = _crate_wheel()
    if wheel:
        print(f"  shipping crate to all nodes via runtime_env py_modules: {wheel}")
    else:
        print(
            "  WARNING: no crate wheel found to ship (set RAY_DATA_ARROW_RS_WHEEL or "
            "build it under target/wheels/). Worker nodes without the crate installed "
            "will fail the arrow_rs read — build the wheel first."
        )
    root = _shared_trace_root()
    if root is None:
        root = os.path.join(bench_suite.OUT, "dist_check")
        print(
            "  WARNING: no shared cluster storage found — path/memory readouts will "
            "reflect the HEAD node ONLY. Set RAY_DATA_ARROW_RS_SHARED_TRACE_DIR to a "
            "cluster-shared path for full-cluster aggregation. (Correctness/count "
            "check is still valid.)"
        )

    def _read(reader):
        ray.shutdown()
        d = os.path.join(root, reader)
        os.makedirs(d, exist_ok=True)
        for f in _globmod.glob(os.path.join(d, "uss_*.csv")) + _globmod.glob(
            os.path.join(d, "path_*.log")
        ):
            try:
                os.remove(f)
            except OSError:
                pass
        env_vars = {
            "RAY_MEM_TRACE_DIR": d,
            "RAY_MEM_TRACE_INTERVAL_S": "0.05",  # coarser: many nodes -> shared FS
            "RAY_DATA_ARROW_RS_PATH_TRACE": d,
            "RAY_DATA_USE_DATASOURCE_V2": "1",
            "RAY_DATA_USE_ARROW_RS_PARQUET_READER": "1"
            if reader == "arrow_rs"
            else "0",
            "RAY_DATA_ARROW_RS_K": "1",
            "RAY_DATA_ARROW_RS_DECODE_BUDGET_BYTES": str(2 * bench_suite.MB),
            "RAY_DATA_ARROW_RS_FETCH_WINDOW_MB": "16",
        }
        runtime_env = {
            "working_dir": bench_suite.HOOKDIR,
            "worker_process_setup_hook": "worker_mem_sampler.setup",
            "env_vars": env_vars,
        }
        # Ship the crate wheel to every node (Anyscale won't propagate a local
        # pip-installed wheel). py_modules accepts a local .whl or an s3:// URI and
        # uploads it to the cluster's package store, importable on all workers.
        if wheel and reader == "arrow_rs":
            runtime_env["py_modules"] = [wheel]
        ray.init(
            address="auto",
            ignore_reinit_error=True,
            log_to_driver=False,
            runtime_env=runtime_env,
        )
        from ray.data.context import DataContext

        ctx = DataContext.get_current()
        ctx.use_datasource_v2 = True
        ctx.use_arrow_rs_parquet_reader = reader == "arrow_rs"
        ctx.execution_options.preserve_order = True
        t0 = time.time()
        n = bench_suite.consume(ray.data.read_parquet(read_uri), "iter_batches")
        t1 = time.time()
        time.sleep(0.4)  # let line-buffered samples flush to the shared FS
        peaks = _per_node_peaks(d, t0, t1)
        nat, fb = bench_suite._count_paths(d)
        ray.shutdown()
        return n, nat, fb, peaks

    print("  reading via PyArrow across the cluster ...")
    pa_rows, _, _, pa_peaks = _read("pyarrow")
    print("  reading via arrow-rs across the cluster ...")
    rs_rows, nat, fb, rs_peaks = _read("arrow_rs")

    nodes = sorted(set(pa_peaks) | set(rs_peaks))
    print(
        f"\n  rows: pyarrow={pa_rows}  arrow_rs={rs_rows}   "
        f"native={nat} fallback={fb}   nodes seen={len(nodes)}"
    )
    print(
        f"\n  {'node':22s} {'pyarrow peak':>14s} {'arrow_rs peak':>14s} {'mem vs pa':>10s}"
    )
    for h in nodes:
        pk, rk = pa_peaks.get(h, 0.0), rs_peaks.get(h, 0.0)
        ratio = f"{pk / rk:.2f}x" if rk else "-"
        print(f"  {h:22s} {pk:11.0f}MB {rk:11.0f}MB {ratio:>10s}")

    problems = []
    if rs_rows != pa_rows:
        problems.append(f"row-count mismatch: arrow_rs={rs_rows} vs pyarrow={pa_rows}")
    if nat == 0:
        problems.append("0 native fragments (crate not driving the read on any node)")
    if fb != 0:
        problems.append(f"{fb} fragment(s) fell back to PyArrow (unsupported schema?)")
    if len(nodes) < 2:
        problems.append(
            f"only {len(nodes)} node(s) observed — either the cluster is single-node "
            "or traces aren't on shared storage (so this isn't verifying multi-node)"
        )
    if problems:
        print("\nDISTRIBUTED CHECK FAILED:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print(
        "\nDISTRIBUTED CHECK PASSED — correct row count under multi-node scheduling, "
        "crate drove every fragment on every node, per-node memory listed above "
        "(directional). Confirm the shape against the Anyscale/Ray memory dashboard."
    )


def main():
    base = _s3_base()
    os.makedirs(bench_suite.OUT, exist_ok=True)

    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        print("===== SMOKE TEST =====")
        smoke(base)
        return

    if len(sys.argv) > 1 and sys.argv[1] == "distributed-check":
        print("===== DISTRIBUTED CHECK (current cluster) =====")
        distributed_check(base)
        return

    print("===== FIXTURES =====")
    read_uri = ensure_fixtures(base)
    # axis_s3 reads exactly this path (see bench_suite.axis_s3).
    os.environ["RAY_DATA_ARROW_RS_S3_BENCH_PATH"] = read_uri

    print("\n===== S3 SWEEP (PyArrow baseline vs arrow-rs window/arena) =====")
    results = bench_suite.axis_s3()
    if not results:
        sys.exit("axis_s3 produced no results (did the read fall back / fail?).")

    print("\n===== SUMMARY + PLOTS =====")
    summarize.table_s3(results)
    summarize.plot_s3()
    bench_suite.write_summary_csv()
    print(
        "\nDONE. Figures in figs/s3_mem_time.png + figs/s3_speed_time.png; "
        "raw numbers in runs/results_s3.json; machine-readable digest in "
        "runs/summary.csv (paste it back for analysis)."
    )


if __name__ == "__main__":
    main()
