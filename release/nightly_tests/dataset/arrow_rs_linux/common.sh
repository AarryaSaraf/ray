#!/usr/bin/env bash
# Shared environment for the arrow-rs single-machine experiments.
# Sourced by exp*.sh -- not meant to be run directly.
#
# Sizing assumes the documented box: 8 CPUs, 32 GB RAM. Both arms of every A/B
# get the SAME num_cpus and object-store budget, because Ray's defaults derive
# from *free* RAM, which drifts between runs and would silently change the
# comparison.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="$(cd "$HERE/.." && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
OUT_DIR="${OUT_DIR:-$HERE/out}"
mkdir -p "$OUT_DIR"

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
print(f"env OK: ray.data={resolved}")
PYEOF
}
export REPO

# Run one command with the arrow-rs reader on (arg 1 = "arrow_rs"|"pyarrow"),
# profiling into a per-run directory. Remaining args are the command.
run_arm() {
  local reader="$1"; shift
  local tag="$1"; shift
  local prof="$OUT_DIR/prof/${tag}_${reader}"
  mkdir -p "$prof"
  echo "=== [$tag] reader=$reader -> $OUT_DIR/${tag}_${reader}.json"
  env \
    RAY_DATA_USE_ARROW_RS_PARQUET_READER="$([ "$reader" = arrow_rs ] && echo 1 || echo 0)" \
    RAY_DATA_ARROW_RS_PROFILE=1 \
    RAY_DATA_ARROW_RS_PROFILE_DIR="$prof" \
    TEST_OUTPUT_JSON="$OUT_DIR/${tag}_${reader}.json" \
    "$@" 2>&1 | tee "$OUT_DIR/${tag}_${reader}.log"
}
