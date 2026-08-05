#!/usr/bin/env bash
# Experiment 3 -- `write_parquet` with a decode-budget sweep.
#
# Why: this is the ONLY unambiguous per-task memory regression in the release A/B
# -- 2.28x max USS, 1.68x average USS, at wall time exactly 1.00x. Time-neutral
# and memory-negative is the signature of accumulation, not of a slow path.
# PyArrow's worst task is 1.07x its own average; ours is 1.46x, i.e. a heavy tail
# PyArrow does not have.
#
# What the sweep decides: H4/H7 predict the regression shrinks monotonically as
# the decode budget rises toward the block size (more rows per batch -> fewer,
# larger batches for BlockOutputBuffer to accumulate). If USS is flat across
# 2 / 32 / 128 MiB, the cause is elsewhere and H4 is dead.
#
# Scaled down from the release config (sf1000 -> sf10); the shape is what
# matters, not the volume.
#
# Needs: AWS credentials (input bucket) and ~10 GB of local disk for the output.
# The release target bucket is not writable outside the release account, hence
# RAY_DATA_BENCH_WRITE_ROOT.
#
# Runtime: ~20 min for four arms.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
check_env

DATA="${DATA:-s3://ray-benchmark-data/tpch/parquet/sf10/lineitem}"
export RAY_DATA_BENCH_WRITE_ROOT="${RAY_DATA_BENCH_WRITE_ROOT:-$OUT_DIR/write}"
mkdir -p "$RAY_DATA_BENCH_WRITE_ROOT"

cd "$DATASET_DIR"

run_arm pyarrow exp3_write \
  python read_and_consume_benchmark.py "$DATA" --format parquet --write

for budget in 2097152 33554432 134217728; do
  mib=$((budget / 1024 / 1024))
  echo "=== arrow_rs, decode_budget=${mib}MiB"
  # The knob goes through `env` inside the command rather than as a prefix on
  # run_arm: a prefix assignment on a shell function can leak into the caller.
  run_arm arrow_rs "exp3_write_budget${mib}" \
    env "RAY_DATA_ARROW_RS_DECODE_BUDGET_BYTES=$budget" \
    python read_and_consume_benchmark.py "$DATA" --format parquet --write
done

# Clean up the written data -- sf10 lineitem is a few GB per arm.
rm -rf "${RAY_DATA_BENCH_WRITE_ROOT:?}"/*

python "$HERE/compare_results.py" \
  "$OUT_DIR"/exp3_write_budget*_arrow_rs.json \
  "$OUT_DIR/exp3_write_pyarrow.json" \
  --baseline "$OUT_DIR/exp3_write_pyarrow.json" \
  | tee "$OUT_DIR/exp3_summary.txt"
