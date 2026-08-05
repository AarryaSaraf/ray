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
# Needs: credentials for our bucket. Both the input and the output live there,
# so this exercises the real S3 write path the release case uses -- writing to
# local disk instead would change the very thing under test (the write task's
# memory profile), since a local write has no multipart upload buffering.
#
# Runtime: ~20 min for four arms.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
check_env
check_s3

SF="${SF:-10}"
DATA="${DATA:-$S3_ROOT/tpch/sf$SF/lineitem}"
export RAY_DATA_BENCH_WRITE_ROOT="${RAY_DATA_BENCH_WRITE_ROOT:-$S3_ROOT/write}"

python "$HERE/stage_data.py" --dst "$DATA" --sf "$SF" --region "$AWS_DEFAULT_REGION"

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

# Clean up the written data -- sf10 lineitem is a few GB per arm, four arms.
# Deliberately narrow: refuses to touch anything outside our own prefix, so a
# mis-set RAY_DATA_BENCH_WRITE_ROOT cannot delete a bucket root.
WRITE_ROOT="$RAY_DATA_BENCH_WRITE_ROOT" S3_PREFIX="$S3_PREFIX" python - <<'PYEOF'
import os, shutil
root, prefix = os.environ["WRITE_ROOT"], os.environ["S3_PREFIX"]
if not root.startswith("s3://"):
    shutil.rmtree(root, ignore_errors=True)
    raise SystemExit(0)
bucket, _, key = root[len("s3://") :].partition("/")
if not key or prefix not in key:
    raise SystemExit(f"refusing to delete s3://{bucket}/{key} -- not under {prefix}/")
import boto3
client = boto3.client("s3", region_name=os.environ.get("AWS_DEFAULT_REGION"))
deleted = 0
for page in client.get_paginator("list_objects_v2").paginate(
    Bucket=bucket, Prefix=key.rstrip("/") + "/"
):
    batch = [{"Key": o["Key"]} for o in page.get("Contents", [])]
    if batch:
        client.delete_objects(Bucket=bucket, Delete={"Objects": batch})
        deleted += len(batch)
print(f"cleaned {deleted} objects under s3://{bucket}/{key}")
PYEOF

python "$HERE/compare_results.py" \
  "$OUT_DIR"/exp3_write_budget*_arrow_rs.json \
  "$OUT_DIR/exp3_write_pyarrow.json" \
  --baseline "$OUT_DIR/exp3_write_pyarrow.json" \
  | tee "$OUT_DIR/exp3_summary.txt"
