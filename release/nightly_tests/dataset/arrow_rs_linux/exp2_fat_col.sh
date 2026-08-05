#!/usr/bin/env bash
# Experiment 2 -- the fat-column pair, against a local S3 endpoint.
#
# Why an S3 endpoint on localhost: the defect we are chasing (`plan_s3_units` /
# `partition_columns_by_budget` / the prefetch admission loop) exists ONLY in the
# crate's `read_row_groups_s3` entry point. The local-filesystem path has no
# units, no column groups and no fetch budget, so a file:// run provably cannot
# reproduce it -- which is exactly why no local benchmark we ever ran caught it.
#
# What it decides, with its own control:
#   fat_col       1 fat + 1 small column. `partition_columns_by_budget` returns
#                 >1 group ("a single column always fits"), so this takes the
#                 column-group/Hstack branch, which retains the WHOLE decoded row
#                 group -- PyArrow's exact behaviour, i.e. no win by construction.
#   fat_col_solo  the same bytes in ONE column. Cannot be split, so it must take
#                 the windowed branch.
# If arrow_rs loses on fat_col and wins on fat_col_solo, the mis-selection is
# confirmed and the fix is a threshold, not a rewrite.
#
# Also runs wide_5k: a local stand-in for wide_schema_pipeline_primitives (#1 /
# 4.90x), whose real input bucket we cannot read.
#
# Needs: nothing external. moto[server] is installed by setup.sh and the fake
# credentials below are scoped to this script.
#
# Runtime: ~15 min including fixture generation (~1.5 GB, held in moto's RAM).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
check_env

PORT="${MOTO_PORT:-5002}"
ENDPOINT="http://127.0.0.1:$PORT"
BUCKET="${BUCKET:-arrow-rs-bench}"

# Fake creds for moto. Exported so Ray workers inherit them; scoped to this
# script, so exp1/exp3's real credentials are untouched.
export AWS_ACCESS_KEY_ID=testing AWS_SECRET_ACCESS_KEY=testing
export AWS_SESSION_TOKEN=testing AWS_DEFAULT_REGION=us-east-1
export AWS_EC2_METADATA_DISABLED=true

cleanup() { [ -n "${MOTO_PID:-}" ] && kill "$MOTO_PID" 2>/dev/null || true; }
trap cleanup EXIT

echo "=== starting moto_server on $ENDPOINT"
moto_server -H 127.0.0.1 -p "$PORT" >"$OUT_DIR/moto.log" 2>&1 &
MOTO_PID=$!
python - <<PYEOF
import time, urllib.error, urllib.request
for _ in range(60):
    try:
        urllib.request.urlopen("$ENDPOINT", timeout=1)
        break
    except urllib.error.HTTPError:
        break            # any HTTP response means it is listening
    except Exception:
        time.sleep(0.5)
else:
    raise SystemExit("moto_server did not come up -- see $OUT_DIR/moto.log")
import boto3
boto3.client("s3", endpoint_url="$ENDPOINT").create_bucket(Bucket="$BUCKET")
print("moto up, bucket $BUCKET created")
PYEOF

cd "$DATASET_DIR"
FIXTURES="s3://$BUCKET/fixtures"
SHAPES="${SHAPES:-fat_col,fat_col_solo,wide_5k}"

echo "=== generating fixtures ($SHAPES) into $FIXTURES"
python arrow_rs_fixtures.py --out "$FIXTURES" --endpoint "$ENDPOINT" \
  --shapes "$SHAPES" 2>&1 | tee "$OUT_DIR/exp2_fixtures.log"

echo "=== probing (iter_bundles, 3 reps, both readers)"
python arrow_rs_probe.py \
  --data "$FIXTURES" --endpoint "$ENDPOINT" \
  --shapes "$SHAPES" --readers arrow_rs,pyarrow \
  --consume iter_bundles --repeat 3 \
  --num-cpus "$NUM_CPUS" --object-store-mb "$OBJECT_STORE_MB" \
  --prof-root "$OUT_DIR/prof/exp2" \
  > "$OUT_DIR/exp2_runs.jsonl" 2> "$OUT_DIR/exp2_summary.txt"

echo
echo "=== exp2 summary (also in $OUT_DIR/exp2_summary.txt)"
cat "$OUT_DIR/exp2_summary.txt"
