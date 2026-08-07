#!/usr/bin/env bash
# Experiment 6 -- what is a read task holding on to?
#
# Standing puzzle: the identical read costs 23 MiB of private memory in a bare
# process (exp5) and ~1 GiB inside a Ray read task (exp3). Three explanations
# are already ruled out:
#
#   * the transport   -- local and S3 both show ~1 GiB (exp3 both ways)
#   * the decode budget -- 2 -> 128 MiB moves per-task USS by 7%
#   * crate retention -- `retained_bytes` is 20 MiB per row group, and the
#                        column-group path never ran (`col_group_rgs = 0`)
#
# What per-task USS *does* equal is one input file's decoded size: lineitem is
# ~6.0M rows x 172 B = 1.03 GiB, measured USS is 958-1035 MiB. So the task looks
# like it holds a whole file rather than streaming it.
#
# This sweeps target_max_block_size over 32x. Two outcomes, both decisive:
#
#   USS rises with block size  -> output blocks accumulate in the task. The fix
#                                 is in the block layer / output buffer, applies
#                                 to BOTH readers, and the knob is a workaround.
#   USS flat across the sweep  -> blocks are NOT what is retained. Next suspect
#                                 is the fragment loop materializing a whole
#                                 file per yielded table, which would be immune
#                                 to every knob we have tried -- and would be
#                                 ours to fix, since it is where the 40x went.
#
# Both readers are swept: if PyArrow is flat too, the retention is Ray's and
# predates us; if only arrow-rs is flat, it is the arrow-rs reader.
#
# Local disk only -- S3 adds a fetch path that is not under test here, and exp3
# already showed the two transports agree.
#
# Runtime: ~15 min for 8 arms.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
check_env

LOCAL_ROOT="${LOCAL_ROOT:-$HOME/arrow_rs_local}"
DATA="${DATA:-$LOCAL_ROOT/lineitem}"
[ -d "$DATA" ] || { echo "FATAL: $DATA missing -- run exp5_standalone.sh first"; exit 1; }
WRITE_DIR="$LOCAL_ROOT/exp6_write"
RESULTS="$OUT_DIR/exp6"
BLOCKS="${BLOCKS:-16 64 128 512}"
mkdir -p "$RESULTS"

# One process per arm: Ray reuses workers, and MemoryProfiler reads whole-process
# private memory, so a second arm in the same cluster inherits the first arm's
# high-water mark. That worker-reuse effect is the exact artifact that made
# read_from_uris look like a regression in the release A/B.
for reader in pyarrow arrow_rs; do
  for mib in $BLOCKS; do
    tag="${reader}_${mib}"
    echo "=== $tag"
    python "$HERE/block_retention_probe.py" \
      --source "$DATA" --reader "$reader" --block-mib "$mib" \
      --write-to "$WRITE_DIR" --out "$RESULTS/$tag.json" \
      > "$RESULTS/$tag.log" 2>&1 \
      || echo "  FAILED -- see $RESULTS/$tag.log"
  done
done
rm -rf "$WRITE_DIR"

RESULTS="$RESULTS" python - <<'PYEOF' | tee "$OUT_DIR/exp6_summary.txt"
import glob
import json
import os

MiB = 1024 * 1024
rows = {}
for path in sorted(glob.glob(os.path.join(os.environ["RESULTS"], "*.json"))):
    with open(path) as handle:
        r = json.load(handle)
    rows[(r["reader"], r["block_mib"])] = r

head = (
    f"{'reader':<10}{'block':>8}{'avg USS/task':>15}{'max USS/task':>15}"
    f"{'tasks':>7}{'rows/task':>12}{'wall':>8}"
)
print(head)
print("-" * len(head))
for reader in ("pyarrow", "arrow_rs"):
    series = []
    for (rd, mib), r in sorted(rows.items(), key=lambda kv: kv[0][1]):
        if rd != reader:
            continue
        avg = (r.get("avg_max_uss_per_task") or 0) / MiB
        mx = (r.get("max_uss_per_task") or 0) / MiB
        series.append(avg)
        print(
            f"{reader:<10}{mib:>7}M{avg:>12.0f}MiB{mx:>12.0f}MiB"
            f"{r.get('task_count') or 0:>7}"
            f"{(r.get('rows_per_task_mean') or 0):>12,.0f}{r['wall_s']:>8.1f}"
        )
    if len(series) > 1 and min(series) > 0:
        spread = max(series) / min(series)
        verdict = (
            "TRACKS the knob -> output blocks accumulate in the task"
            if spread > 1.5
            else "FLAT across the sweep -> blocks are not what is retained"
        )
        print(f"{'':<10}  spread {spread:.2f}x over the block sweep: {verdict}\n")

print(
    "A lineitem file is ~6.0M rows x 172 B = 1.03 GiB decoded. Per-task USS at or"
    "\nnear that number, unmoved by a 32x block sweep, means the task holds a whole"
    "\nfile. Standalone, the same read holds 23 MiB."
)
PYEOF
