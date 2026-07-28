#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-shot environment setup for the arrow-rs Parquet memory benchmark suite.
#
# Brings a fresh box (Linux x86-64 or macOS/arm64) to the point where you can run
#   python bench_suite.py <axis>   and   python summarize.py
# by:
#   1. ensuring a Python 3.12 venv (uv-managed),
#   2. installing a Ray nightly wheel + symlinking THIS repo's python/ray over it
#      (so the local arrow-rs reader source is live) — skip with SKIP_RAY=1,
#   3. installing the Rust toolchain (rustup) + maturin,
#   4. building the native crate `ray_data_arrow_rs` into the venv,
#   5. installing the harness's Python deps (psutil, matplotlib, numpy),
#   6. verifying the arrow-rs read path actually engages end to end.
#
# Idempotent: re-running skips work that's already done. Everything it installs
# goes into the venv / ~/.cargo — nothing touches the system Python.
#
# Usage (from anywhere):
#   bash release/nightly_tests/dataset/arrow_rs_memtrace/setup.sh
#
# Knobs (env vars):
#   RAY_VENV=<path>        venv to use/create           (default: <repo>/.venv)
#   RAY_WHEEL_URL=<url>    Ray nightly wheel to install  (default: cp312 linux/mac)
#   SKIP_RAY=1            don't touch Ray (already installed + symlinked)
#   SKIP_APT=1            don't apt-get build deps (build-essential, python3-dev)
#   SKIP_CRATE=1          don't (re)build the Rust crate
# ---------------------------------------------------------------------------
set -euo pipefail

# --- locate the repo (this script lives at <repo>/release/nightly_tests/dataset/arrow_rs_memtrace) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
CRATE="$REPO/python/ray/data/_internal/datasource_v2/native/ray_data_arrow_rs"
RAY_VENV="${RAY_VENV:-$REPO/.venv}"
OS="$(uname -s)"; ARCH="$(uname -m)"

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

say "repo=$REPO  venv=$RAY_VENV  os=$OS/$ARCH"

# --- 0. system build deps (Linux only; the crate links libpython + needs a C toolchain) ---
if [ "$OS" = "Linux" ] && [ "${SKIP_APT:-0}" != "1" ] && command -v apt-get >/dev/null 2>&1; then
  say "apt: build-essential + python3-dev + curl (sudo)"
  sudo apt-get update -qq
  sudo apt-get install -y -qq build-essential python3-dev curl pkg-config
fi

# --- 1. uv + venv ---
if ! command -v uv >/dev/null 2>&1; then
  say "installing uv (official installer)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
if [ ! -x "$RAY_VENV/bin/python" ]; then
  say "creating venv at $RAY_VENV (python 3.12)"
  uv venv --python 3.12 "$RAY_VENV"
fi
PY="$RAY_VENV/bin/python"
PIP() { uv pip install --python "$PY" "$@"; }
say "python: $($PY --version)"

# --- 2. Ray nightly + local-source symlink ---
# CRITICAL: the wheel's compiled protobuf must match the branch's Python source, or
# `custom_types.py` asserts "out of sync" at import. setup-dev.py symlinks THIS repo's
# python/ray over the wheel, so we install the per-commit nightly built from the
# branch's base commit (merge-base with upstream/master), NOT "latest" — latest drifts.
BASE_SHA="$(git -C "$REPO" merge-base HEAD upstream/master 2>/dev/null \
            || git -C "$REPO" merge-base HEAD origin/master 2>/dev/null \
            || echo 7dc67bed3ba2f3504325b206a70adcc470422860)"
if [ "${SKIP_RAY:-0}" != "1" ]; then
  if [ -z "${RAY_WHEEL_URL:-}" ]; then
    if [ "$OS" = "Linux" ]; then
      PYTAG=cp312; PLAT=manylinux2014_x86_64
    elif [ "$ARCH" = "arm64" ]; then
      PYTAG=cp312; PLAT=macosx_11_0_arm64
    else
      PYTAG=cp312; PLAT=macosx_10_15_x86_64
    fi
    RAY_WHEEL_URL="https://s3-us-west-2.amazonaws.com/ray-wheels/master/${BASE_SHA}/ray-3.0.0.dev0-${PYTAG}-${PYTAG}-${PLAT}.whl"
  fi
  say "installing Ray nightly (base commit ${BASE_SHA:0:12}): $RAY_WHEEL_URL"
  # Wipe any prior ray install FIRST. A re-run over a setup-dev'd tree has symlinked
  # subpackages (ray/workflow -> local source); pip/uv --force-reinstall dies trying
  # to rmdir a symlink ("Not a directory"). Removing the dir just unlinks those
  # symlinks (never touches the repo source they point to), so a clean install lands
  # the commit-matched wheel — and swaps a same-version "latest" wheel too.
  SITE="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  rm -rf "$SITE/ray" "$SITE"/ray-*.dist-info "$SITE"/ray_*.dist-info 2>/dev/null || true
  PIP "ray[data] @ $RAY_WHEEL_URL"
  # Symlink THIS repo's python/ray over the installed wheel so the local
  # arrow-rs reader source is what actually runs (mirrors the mac dev setup).
  say "symlinking local python/ray via setup-dev.py"
  "$PY" "$REPO/python/ray/setup-dev.py" -y
else
  say "SKIP_RAY=1 — assuming Ray is installed and python/ray is symlinked"
fi

# --- 3. Rust toolchain + maturin ---
if ! command -v cargo >/dev/null 2>&1; then
  say "installing Rust via rustup (official installer)"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
export PATH="$HOME/.cargo/bin:$PATH"
say "rustc: $(rustc --version)"
PIP maturin

# --- 4. build the native crate into the venv ---
if [ "${SKIP_CRATE:-0}" != "1" ]; then
  say "building ray_data_arrow_rs (maturin develop --release) — this compiles arrow/parquet, ~2-5 min"
  # maturin refuses if BOTH VIRTUAL_ENV and CONDA_PREFIX are set (common when a
  # base conda env is active); unset CONDA_PREFIX for this build only.
  ( cd "$CRATE" && unset CONDA_PREFIX && VIRTUAL_ENV="$RAY_VENV" "$RAY_VENV/bin/maturin" develop --release )
else
  say "SKIP_CRATE=1 — assuming ray_data_arrow_rs is already built"
fi

# --- 5. harness Python deps ---
# aiohttp: NOT pulled in by the ray[data] extra, but the runtime-env agent imports
# it; without it the agent crashes and the raylet fate-shares (`ray start` hangs
# indefinitely) — cost a day on the 2026-07-27 workspace run.
say "installing harness deps (psutil, matplotlib, numpy, aiohttp)"
PIP psutil matplotlib numpy aiohttp

# --- 6. verify the arrow-rs path actually engages ---
say "verifying arrow-rs read path end to end"
# Force a private local cluster: on an Anyscale workspace, ray.init() would attach to
# the running (different-version) cluster and fail the version check. RAY_ADDRESS=local
# + a cleared platform hook make this self-contained.
RAY_ADDRESS=local RAY_DATA_USE_DATASOURCE_V2=1 RAY_DATA_USE_ARROW_RS_PARQUET_READER=1 \
  "$PY" - <<'PYEOF'
import os, tempfile
os.environ.pop("RAY_RUNTIME_ENV_HOOK", None)  # Anyscale platform hook not in this venv
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
import ray_data_arrow_rs  # noqa: F401  -> import must succeed (crate built)
import ray

d = tempfile.mkdtemp()
p = os.path.join(d, "t.parquet")
pq.write_table(pa.table({"a": np.arange(1000), "b": np.arange(1000) * 1.5}),
               p, write_page_index=True)
ray.init(address="local", include_dashboard=False,
         ignore_reinit_error=True, log_to_driver=False)
ds = ray.data.read_parquet(p)
assert ds.count() == 1000, ds.count()
assert ds.sum("a") == sum(range(1000))
print("OK  ray", ray.__version__, " arrow-rs read path verified (count + sum match)")
ray.shutdown()
PYEOF

say "DONE. Activate with:  source $RAY_VENV/bin/activate"
echo "Then, from $SCRIPT_DIR, run the suite with a private local cluster:"
echo "  export RAY_ADDRESS=local"
echo "  python bench_suite.py leak_rgsize && python summarize.py"
echo "If workers SIGSEGV in the core task-event aggregator (seen on 2026-07 master),"
echo "set RAY_task_events_report_interval_ms=0 as a workaround."
