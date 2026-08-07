#!/usr/bin/env bash
# Shared environment for the arrow-rs single-machine experiments.
# Sourced by exp*.sh -- not meant to be run directly.
#
# Sizing assumes the documented box: 8 CPUs, 32 GB RAM. Both arms of every A/B
# get the SAME num_cpus and object-store budget, because Ray's defaults derive
# from *free* RAM, which drifts between runs and would silently change the
# comparison.
# Strict mode for scripts, but NOT when sourced into an interactive shell:
# `set -e` there means the next command that returns nonzero -- a failed grep, a
# completion probe -- closes the terminal.
case $- in
  *i*) ;;
  *) set -euo pipefail ;;
esac

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="$(cd "$HERE/.." && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
OUT_DIR="${OUT_DIR:-$HERE/out}"
mkdir -p "$OUT_DIR"

# --- our own S3 bucket --------------------------------------------------------
# Everything reads and writes here: fixtures, the staged TPC-H copy, and exp3's
# output. Using a bucket we own removes the permission question entirely (the
# shared `ray-benchmark-data` is readable but not writable, and its sibling
# `-internal-` bucket is not readable at all), and it means the S3 experiments
# run against REAL S3 -- real latency, real concurrency limits -- which is what
# the prefetch/serialisation hypothesis actually needs.
S3_BUCKET="${S3_BUCKET:-arrowrs-bench-21f6c795}"
S3_PREFIX="${S3_PREFIX:-arrow_rs_probe}"
S3_ROOT="s3://$S3_BUCKET/$S3_PREFIX"
# Both pyarrow and the crate must agree on the region; pyarrow can discover it
# per bucket, object_store cannot, so pin it.
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-west-2}"
export AWS_REGION="${AWS_REGION:-$AWS_DEFAULT_REGION}"

NUM_CPUS="${NUM_CPUS:-8}"
# ~8 GiB of a 32 GB box. Ray's default is 30% of *free* RAM; pinning it keeps the
# object store from being a moving target between the two arms.
OBJECT_STORE_MB="${OBJECT_STORE_MB:-8192}"
export RAY_object_store_memory=$((OBJECT_STORE_MB * 1024 * 1024))
export RAY_num_cpus="$NUM_CPUS"

# --- managed-workspace traps (each of these cost a day at least once) ---------
# A platform-injected runtime-env module that is not in this venv kills the
# runtime-env agent; the raylet fate-shares and ray.init() hangs with no error.
unset RAY_RUNTIME_ENV_HOOK RAY_RUNTIME_ENV_PLUGINS 2>/dev/null || true
# Never attach to the managed cluster: different Ray version, different reader.
export RAY_ADDRESS=local
# Some 2026-07 master nightlies SIGSEGV in the task-event aggregator flush,
# killing workers and the driver. We do not read task events.
export RAY_task_events_report_interval_ms=0

export RAY_DATA_USE_DATASOURCE_V2=1

# --- guard: are we running THIS checkout's Ray? -------------------------------
# Without the venv activated, `python` is the image's anaconda, whose Ray is the
# Anyscale runtime -- a different read path where this branch's reader does not
# exist. The run would succeed and mean nothing.
check_env() {
  python - <<'PYEOF'
import os, sys
import ray.data
repo = os.path.realpath(os.environ["REPO"])
resolved = os.path.realpath(ray.data.__file__)
if not resolved.startswith(repo + os.sep):
    if os.environ.get("RAY_DATA_BENCH_ALLOW_FOREIGN_RAY") != "1":
        sys.exit(
            f"FATAL: ray.data resolves to {resolved}, outside {repo}.\n"
            "Run: source $REPO/.venv/bin/activate  (or set "
            "RAY_DATA_BENCH_ALLOW_FOREIGN_RAY=1 if that is deliberate)"
        )
    print(f"WARNING: measuring a foreign Ray at {resolved}")
try:
    import ray_data_arrow_rs  # noqa: F401
except ImportError as exc:
    sys.exit(f"FATAL: native crate not importable ({exc}). Run setup.sh.")
# grpcio is a lazy import inside ray._private.grpc_utils, reached only when a
# driver asks for object-store/spill stats -- so a missing one does not fail at
# import time, it fails several minutes into the first benchmark.
try:
    import grpc  # noqa: F401
except ImportError:
    sys.exit(
        "FATAL: grpcio missing. Recent master dropped it from the base install,\n"
        "but every release driver calls get_memory_info_reply(). Run: "
        "uv pip install grpcio"
    )
print(f"env OK: ray.data={resolved}")
PYEOF
}
export REPO

# --- guard: can we actually reach the bucket, both ways? ----------------------
# Fails in seconds instead of ten minutes into a run. Checks WRITE too, because
# exp3 writes and a read-only credential would only surface at the end.
check_s3() {
  S3_BUCKET="$S3_BUCKET" S3_PREFIX="$S3_PREFIX" python - <<'PYEOF'
import os, sys
from pyarrow.fs import S3FileSystem
bucket, prefix = os.environ["S3_BUCKET"], os.environ["S3_PREFIX"]
region = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
probe = f"{bucket}/{prefix}/.probe"
try:
    fs = S3FileSystem(region=region)
    with fs.open_output_stream(probe) as f:
        f.write(b"ok")
    fs.delete_file(probe)
except Exception as exc:
    sys.exit(
        f"FATAL: cannot write s3://{probe} ({type(exc).__name__}: {exc}).\n"
        "On an EC2/Anyscale box credentials come from the instance role and no\n"
        "export is needed. Otherwise:  "
        'eval "$(aws configure export-credentials --format env)"'
    )
print(f"s3 OK: s3://{bucket}/{prefix} readable and writable in {region}")
PYEOF
}

# Run one command with the arrow-rs reader on (arg 1 = "arrow_rs"|"pyarrow"),
# profiling into a per-run directory. Remaining args are the command.
run_arm() {
  local reader="$1"; shift
  local tag="$1"; shift
  local prof="$OUT_DIR/prof/${tag}_${reader}"
  local mem="$OUT_DIR/mem/${tag}_${reader}.jsonl"
  mkdir -p "$prof" "$OUT_DIR/mem"
  echo "=== [$tag] reader=$reader -> $OUT_DIR/${tag}_${reader}.json"

  # Record whole-machine memory alongside Ray's own per-task numbers. The two
  # disagree -- on the release A/B, node memory confirmed write_parquet's
  # regression and refuted read_from_uris's -- so a run that captures only one
  # of them cannot be compared to release. See node_memory.py.
  python "$HERE/node_memory.py" record --out "$mem" --interval 1 &
  local mem_pid=$!
  # Stop the sampler even if the benchmark dies or the shell is interrupted.
  trap 'kill "$mem_pid" 2>/dev/null || true' RETURN INT TERM

  env \
    RAY_DATA_USE_ARROW_RS_PARQUET_READER="$([ "$reader" = arrow_rs ] && echo 1 || echo 0)" \
    RAY_DATA_ARROW_RS_PROFILE=1 \
    RAY_DATA_ARROW_RS_PROFILE_DIR="$prof" \
    TEST_OUTPUT_JSON="$OUT_DIR/${tag}_${reader}.json" \
    "$@" 2>&1 | tee "$OUT_DIR/${tag}_${reader}.log"

  kill "$mem_pid" 2>/dev/null || true
  wait "$mem_pid" 2>/dev/null || true
}
