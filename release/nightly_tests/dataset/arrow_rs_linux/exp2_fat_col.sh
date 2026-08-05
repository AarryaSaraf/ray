#!/usr/bin/env bash
# Experiment 2 -- the fat-column pair, against real S3.
#
# Why S3 and not file://: the defect we are chasing (`plan_s3_units` /
# `partition_columns_by_budget` / the prefetch admission loop) exists ONLY in the
# crate's `read_row_groups_s3` entry point. The local-filesystem path has no
# units, no column groups and no fetch budget, so a file:// run provably cannot
# reproduce it -- which is exactly why no local benchmark we ever ran caught it.
#
# Why REAL S3 and not moto: half the hypothesis is about *latency*. An oversized
# unit that takes the whole prefetch semaphore serialises GET -> decode -> GET;
# against moto on localhost a serialised fetch costs microseconds, so the bug is
# invisible even when the (separately visible) retention half is present. Real
# S3 has ~20-60 ms per GET, which is what makes the serialisation measurable.
# Set MOTO=1 to fall back to a local moto server -- fine for the retention
# question, useless for the serialisation one.
#
# What it decides, with its own control:
#   fat_col       1 fat + 1 small column. `partition_columns_by_budget` returns
#                 >1 group ("a single column always fits"), so this takes the
#                 column-group/Hstack branch, which retains the WHOLE decoded row
#                 group -- PyArrow's exact behaviour, i.e. no win by construction.
#   fat_col_solo  the same bytes in ONE column. Cannot be split, so it must take
#                 the windowed branch.
# If arrow_rs loses on fat_col and wins on fat_col_solo, the mis-selection is
# confirmed and the fix is a threshold, not a rewrite. (Confirmed on macOS via
# profiling: fat_col retained 1.115x its whole row group; this measures the cost.)
#
# Also runs wide_5k: a stand-in for wide_schema_pipeline_primitives (#1 / 4.90x),
# whose real input bucket we cannot read.
#
# Runtime: ~15 min including fixture generation (~1.5 GB, written once and
# reused -- set REGEN=1 to rebuild them).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
check_env

SHAPES="${SHAPES:-fat_col,fat_col_solo,wide_5k}"
ENDPOINT_ARGS=()

if [ "${MOTO:-0}" = 1 ]; then
  PORT="${MOTO_PORT:-5002}"
  ENDPOINT="http://127.0.0.1:$PORT"
  BUCKET="${BUCKET:-arrow-rs-bench}"
  FIXTURES="s3://$BUCKET/fixtures"
  ENDPOINT_ARGS=(--endpoint "$ENDPOINT")

  # Fake creds for moto. Exported so Ray workers inherit them; scoped to this
  # script, so the real credentials are untouched.
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
else
  check_s3
  FIXTURES="$S3_ROOT/fixtures"
fi

cd "$DATASET_DIR"

# Fixtures are deterministic and a fixed cost; skip regeneration if the shapes
# are already up there, since 1.5 GB takes a few minutes to write.
NEED_GEN=1
if [ "${REGEN:-0}" != 1 ] && [ "${MOTO:-0}" != 1 ]; then
  if FIXTURES="$FIXTURES" SHAPES="$SHAPES" python - <<'PYEOF'
import os, sys
from pyarrow.fs import S3FileSystem, FileSelector
base = os.environ["FIXTURES"][len("s3://") :]
fs = S3FileSystem(region=os.environ.get("AWS_DEFAULT_REGION", "us-west-2"))
for shape in os.environ["SHAPES"].split(","):
    try:
        if not fs.get_file_info(FileSelector(f"{base}/{shape}", recursive=True)):
            sys.exit(1)
    except Exception:
        sys.exit(1)
PYEOF
  then
    echo "=== fixtures already present at $FIXTURES (REGEN=1 to rebuild)"
    NEED_GEN=0
  fi
fi

if [ "$NEED_GEN" = 1 ]; then
  echo "=== generating fixtures ($SHAPES) into $FIXTURES"
  python arrow_rs_fixtures.py --out "$FIXTURES" "${ENDPOINT_ARGS[@]}" \
    --shapes "$SHAPES" 2>&1 | tee "$OUT_DIR/exp2_fixtures.log"
fi

echo "=== probing (iter_bundles, 3 reps, both readers)"
python arrow_rs_probe.py \
  --data "$FIXTURES" "${ENDPOINT_ARGS[@]}" \
  --shapes "$SHAPES" --readers arrow_rs,pyarrow \
  --consume iter_bundles --repeat 3 \
  --num-cpus "$NUM_CPUS" --object-store-mb "$OBJECT_STORE_MB" \
  --prof-root "$OUT_DIR/prof/exp2" \
  > "$OUT_DIR/exp2_runs.jsonl" 2> "$OUT_DIR/exp2_summary.txt"

echo
echo "=== exp2 summary (also in $OUT_DIR/exp2_summary.txt)"
cat "$OUT_DIR/exp2_summary.txt"
