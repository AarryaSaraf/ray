"""One-command S3 memory/speed benchmark for the arrow-rs Parquet reader.

Run this on a **Linux box with real S3** (moto/macOS can't show the win — moto has
no network latency and macOS USS is only directional). It does the whole thing from
a single command:

    RAY_DATA_ARROW_RS_S3_BENCH_PATH=s3://your-bucket/some/prefix \\
        python run_s3_benchmark.py

which will, in order:
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


def main():
    base = _s3_base()
    os.makedirs(bench_suite.OUT, exist_ok=True)

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
    print(
        "\nDONE. Figures in figs/s3_mem_time.png + figs/s3_speed_time.png; "
        "raw numbers in runs/results_s3.json"
    )


if __name__ == "__main__":
    main()
