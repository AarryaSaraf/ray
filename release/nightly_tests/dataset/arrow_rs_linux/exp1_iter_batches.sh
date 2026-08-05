#!/usr/bin/env bash
# Experiment 1 -- `iter_batches_numpy`, the release test run VERBATIM.
#
# Why this one first: it is the only top-of-list release case whose input bucket
# we can actually read (s3://ray-benchmark-data/tpch/parquet/sf10/lineitem), and
# the release case already uses sf10 -- so this is the same data, same consume
# mode, same driver as build 103100/103101, just on one box.
#
# What it decides: whether the decode memory win exists in Ray at all. At
# 145 B/row the 2 MiB decode budget genuinely binds, so this is the best case for
# the premise; the release run measured 1.01x on per-task USS where the local
# control predicts ~0.3x. Either that gap reproduces here or it is a
# cluster-scale effect.
#
# Needs: AWS credentials in the environment (the TPC-H bucket is readable from
# an Anyscale account; it is not public).
#
# Runtime: ~10 min for both arms. Reads 2.84 GB compressed / ~8.5 GB decoded.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
check_env

DATA="${DATA:-s3://ray-benchmark-data/tpch/parquet/sf10/lineitem}"
FORMAT="${FORMAT:-numpy}"

cd "$DATASET_DIR"
for reader in pyarrow arrow_rs; do
  run_arm "$reader" "exp1_iter_batches_${FORMAT}" \
    python read_and_consume_benchmark.py "$DATA" --format parquet \
      --iter-batches "$FORMAT"
done

python "$HERE/compare_results.py" \
  "$OUT_DIR/exp1_iter_batches_${FORMAT}_arrow_rs.json" \
  "$OUT_DIR/exp1_iter_batches_${FORMAT}_pyarrow.json" \
  --prof "$OUT_DIR/prof/exp1_iter_batches_${FORMAT}_arrow_rs" \
  | tee "$OUT_DIR/exp1_summary.txt"
