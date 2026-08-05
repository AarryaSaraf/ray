#!/usr/bin/env bash
# All three experiments, in the order that maximises what you learn per hour.
# Each is independently runnable; this just chains them and keeps going if one
# fails, so a missing AWS credential does not cost you the moto-only experiment.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-$HERE/out}"
mkdir -p "$OUT_DIR"

status=0
for exp in exp1_iter_batches exp2_fat_col exp3_write_parquet; do
  echo
  echo "################ $exp ################"
  if ! bash "$HERE/$exp.sh"; then
    echo "!!! $exp FAILED (continuing) -- see $OUT_DIR/${exp}*.log"
    status=1
  fi
done

echo
echo "################ summaries ################"
for f in "$OUT_DIR"/exp*_summary.txt; do
  [ -e "$f" ] || continue
  echo "--- $f"
  cat "$f"
done
exit "$status"
