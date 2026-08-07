#!/usr/bin/env bash
# Experiment 4 -- the `read_parquet` SPEED regression.
#
# Why: re-reading the release A/B against the cluster's own per-node memory
# scrape (metrics.json.gz, builds 103100 vs 103101) collapsed ten suspected
# regressions to two, and they have opposite shapes:
#
#   write_parquet        3.36x memory rise, 1.00x wall  -> exp3
#   read_parquet         1.82x wall, 0.90x node memory  -> THIS
#
# read_parquet is the one nobody predicted: arrow-rs is nearly twice as slow
# while using *less* memory on the machine and 23% less per-task USS. A memory
# fix cannot explain it, so the two experiments must not be conflated.
#
# The release input is s3://ray-benchmark-data-internal-us-west-2/imagenet/parquet,
# which we cannot read (ACCESS_DENIED). `fat_rows` is the stand-in: 33 KiB
# binary cells plus an int64 label, which is imagenet-parquet's shape -- a few
# thousand large opaque blobs per file, no wide schema, nothing to prune. If the
# slowdown reproduces here, the cause is in how we decode big binary cells and
# has nothing to do with the real imagenet bytes.
#
# Consume mode is `--iter-bundles`, copied verbatim from the release entry in
# release_data_tests.yaml, because how the blocks are drained changes what the
# reader is allowed to overlap.
#
# Runtime: ~10 min, plus ~5 min to write ~1 GB of fixtures on the first run.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
check_env
check_s3

DATA="${DATA:-$S3_ROOT/fixtures/fat_rows}"

# Writing the fixture is the slow part and it never changes, so skip it if the
# prefix already holds parquet. A partial write from an interrupted run would
# pass this check -- delete the prefix by hand if a run died mid-write.
DATA="$DATA" python - <<'PYEOF'
import os
import sys

from pyarrow.fs import FileSelector, S3FileSystem

data = os.environ["DATA"]
path = data[len("s3://"):] if data.startswith("s3://") else data
fs = S3FileSystem(region=os.environ.get("AWS_DEFAULT_REGION", "us-west-2"))
try:
    found = [f for f in fs.get_file_info(FileSelector(path)) if f.path.endswith(".parquet")]
except OSError:
    found = []
print(f"fixture: {len(found)} parquet files at {data}")
sys.exit(0 if found else 3)
PYEOF
if [ $? -eq 3 ]; then
  echo "=== generating fat_rows fixture (imagenet-shaped: 33 KiB binary cells)"
  python "$DATASET_DIR/arrow_rs_fixtures.py" \
    --out "$S3_ROOT/fixtures" --shapes fat_rows --region "$AWS_DEFAULT_REGION"
fi

cd "$DATASET_DIR"

for reader in pyarrow arrow_rs; do
  run_arm "$reader" exp4_read \
    python read_and_consume_benchmark.py "$DATA" --format parquet --iter-bundles
done

echo
echo "=== Ray's own metrics (per-task USS, wall time)"
python "$HERE/compare_results.py" \
  "$OUT_DIR/exp4_read_arrow_rs.json" \
  "$OUT_DIR/exp4_read_pyarrow.json" \
  --baseline "$OUT_DIR/exp4_read_pyarrow.json" \
  | tee "$OUT_DIR/exp4_summary.txt"

echo
echo "=== whole-machine memory (the release-comparable metric)"
python "$HERE/node_memory.py" report \
  "$OUT_DIR/mem/exp4_read_arrow_rs.jsonl" \
  "$OUT_DIR/mem/exp4_read_pyarrow.jsonl" \
  --baseline "$OUT_DIR/mem/exp4_read_pyarrow.jsonl" \
  | tee -a "$OUT_DIR/exp4_summary.txt"

echo
echo "Expect: wall ~1.8x slower for arrow_rs, node memory at or below 1.0x."
echo "If wall is at parity here, fat_rows is the wrong stand-in for imagenet"
echo "and the next step is getting read access to the internal bucket."
