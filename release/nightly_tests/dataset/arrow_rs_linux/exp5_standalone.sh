#!/usr/bin/env bash
# Experiment 5 -- the same read with NO Ray, local disk vs S3.
#
# Why: write_parquet is the one release regression confirmed by the cluster's
# own per-node memory scrape (3.36x on all ten machines). Both arms write with
# PyArrow, so the cause is on the read side -- but every measurement we have was
# taken inside Ray, so "the crate is heavy" and "our integration accumulates" are
# indistinguishable. This runs the read in a bare process.
#
# Two axes, eight runs:
#
#   reader     pyarrow | arrow_rs      -- is the crate heavier than the scanner?
#   mode       read    | write         -- does feeding a writer amplify it?
#   transport  local   | s3            -- is the S3 planner required to see it?
#
# The transport axis matters because the crate's S3 path has code the local path
# does not have at all: plan_s3_units, partition_columns_by_budget, and the
# prefetch admission loop. If the regression only appears over S3, those three
# functions are the whole search space.
#
# How to read the result:
#
#   arrow_rs >> pyarrow, both transports   -> the decoder itself; fix the crate
#   arrow_rs >> pyarrow, s3 only           -> the S3 planner; fix those 3 funcs
#   arrow_rs ~= pyarrow everywhere         -> NOT the crate. The bug is in the
#                                             Ray integration and exp3 is where
#                                             it will show. Stop tuning the crate.
#
# Runtime: ~15 min, plus a one-off local copy of the input.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
check_env

SF="${SF:-10}"
S3_DATA="${S3_DATA:-$S3_ROOT/tpch/sf$SF/lineitem}"
# Local copies live outside /tmp: these are multi-GB and /tmp is often a small
# tmpfs, which would silently turn the "local" arm into a RAM-disk read.
LOCAL_ROOT="${LOCAL_ROOT:-$HOME/arrow_rs_local}"
LOCAL_DATA="$LOCAL_ROOT/lineitem"
WRITE_DIR="$LOCAL_ROOT/write"
# Same bytes on both transports, and few enough that a local copy is quick.
MAX_FILES="${MAX_FILES:-4}"
RESULTS="$OUT_DIR/exp5"
mkdir -p "$RESULTS" "$LOCAL_ROOT"

check_s3
python "$HERE/stage_data.py" --dst "$S3_DATA" --sf "$SF" --region "$AWS_DEFAULT_REGION"

# --- localize the same bytes --------------------------------------------------
if [ ! -d "$LOCAL_DATA" ] || [ -z "$(ls -A "$LOCAL_DATA" 2>/dev/null)" ]; then
  echo "=== copying $MAX_FILES files to $LOCAL_DATA"
  mkdir -p "$LOCAL_DATA"
  # Copy only as many files as the benchmark will read, so the local and S3 arms
  # see the identical set. `sort` makes the selection deterministic, matching the
  # sorted order standalone_parquet_bench.py lists in.
  aws s3 ls "$S3_DATA/" | awk '{print $4}' | grep '\.parquet$' | sort \
    | head -n "$MAX_FILES" \
    | while read -r name; do
        [ -f "$LOCAL_DATA/$name" ] || aws s3 cp "$S3_DATA/$name" "$LOCAL_DATA/$name"
      done
fi
du -sh "$LOCAL_DATA"

# --- the matrix ---------------------------------------------------------------
# Each arm is a FRESH process: memory is the whole point here, and a second
# reader in the same interpreter would inherit the first one's allocator state.
for transport in local s3; do
  case "$transport" in
    local) src="$LOCAL_DATA" ;;
    s3)    src="$S3_DATA" ;;
  esac
  for reader in pyarrow arrow_rs; do
    for mode in read write; do
      tag="${transport}_${reader}_${mode}"
      echo "=== $tag"
      write_args=()
      [ "$mode" = write ] && write_args=(--write-to "$WRITE_DIR")
      python "$HERE/standalone_parquet_bench.py" \
        --source "$src" --reader "$reader" --max-files "$MAX_FILES" \
        "${write_args[@]}" --out "$RESULTS/$tag.json" \
        > "$RESULTS/$tag.log" 2>&1 \
        || echo "  FAILED -- see $RESULTS/$tag.log"
    done
  done
done

# --- pipelining control -------------------------------------------------------
# The default pyarrow arm keeps the scanner's fragment/batch readahead, so it
# overlaps I/O with decode; the crate's local path is straight-line synchronous.
# A wall-time gap between them is therefore ambiguous. This arm serializes the
# scanner: if pyarrow's wall time collapses toward the crate's here, the gap was
# pipelining, and the crate needs a local prefetch rather than a faster decoder.
for transport in local s3; do
  case "$transport" in
    local) src="$LOCAL_DATA" ;;
    s3)    src="$S3_DATA" ;;
  esac
  tag="${transport}_pyarrow-serial_read"
  echo "=== $tag"
  python "$HERE/standalone_parquet_bench.py" \
    --source "$src" --reader pyarrow --max-files "$MAX_FILES" --readahead 0 \
    --out "$RESULTS/$tag.json" > "$RESULTS/$tag.log" 2>&1 \
    || echo "  FAILED -- see $RESULTS/$tag.log"
done
rm -rf "$WRITE_DIR"

# --- summary ------------------------------------------------------------------
RESULTS="$RESULTS" python - <<'PYEOF' | tee "$OUT_DIR/exp5_summary.txt"
import glob
import json
import os

MiB = 1024 * 1024
rows = {}
for path in sorted(glob.glob(os.path.join(os.environ["RESULTS"], "*.json"))):
    with open(path) as handle:
        r = json.load(handle)
    tag = os.path.basename(path)[: -len(".json")]
    # pyarrow-serial shares reader="pyarrow" in the JSON; key off the filename.
    reader = "pyarrow-serial" if "pyarrow-serial" in tag else r["reader"]
    rows[(r["transport"], r["mode"], reader)] = r

head = (
    f"{'transport':<10}{'mode':<7}{'reader':<16}"
    f"{'USS rise':>11}{'ratio':>8}{'wall':>8}{'cpu':>8}{'cpu/wall':>10}"
    f"{'batches':>10}{'rows/batch':>12}"
)
print(head)
print("-" * len(head))
for transport in ("local", "s3"):
    for mode in ("read", "write"):
        base = rows.get((transport, mode, "pyarrow"))
        for reader in ("pyarrow", "pyarrow-serial", "arrow_rs"):
            r = rows.get((transport, mode, reader))
            if not r:
                continue
            uss = r["uss_rise"] / MiB
            ratio = (
                "baseline"
                if r is base
                else (
                    f"{uss / (base['uss_rise'] / MiB):.2f}x"
                    if base and base["uss_rise"] > 0
                    else "-"
                )
            )
            per_batch = r["rows"] // max(1, r["batches"])
            print(
                f"{transport:<10}{mode:<7}{reader:<16}{uss:>8.0f}MiB{ratio:>8}"
                f"{r['wall_s']:>8.1f}{r.get('cpu_s', 0):>8.1f}"
                f"{r.get('cpu_per_wall', 0) or 0:>10.2f}"
                f"{r['batches']:>10,}{per_batch:>12,}"
            )

# Row parity is the validity check: a memory ratio means nothing if the arms
# did not decode the same data.
counts = {}
for (transport, mode, reader), r in rows.items():
    counts.setdefault((transport, mode), set()).add(r["rows"])
bad = [k for k, v in counts.items() if len(v) > 1]
print(
    "\nROW PARITY: "
    + ("OK -- every arm decoded the same rows" if not bad else f"MISMATCH in {bad}")
)
print(
    "\nUSS rise  = private memory the read added above a post-import baseline."
    "\ncpu/wall  = cores used. pyarrow's scanner pipelines I/O against decode"
    "\n            even at use_threads=False; the crate's LOCAL path does not."
    "\n            If pyarrow > 1 and arrow_rs ~= 1, compare cpu, not wall."
    "\npyarrow-serial = the same scanner with readahead disabled: the decode-cost"
    "\n            comparison. If it lands near arrow_rs, the wall gap was"
    "\n            pipelining and the crate needs local prefetch, not a faster"
    "\n            decoder."
)
PYEOF
