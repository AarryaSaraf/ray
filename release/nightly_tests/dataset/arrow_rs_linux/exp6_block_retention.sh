#!/usr/bin/env bash
# Experiment 6 -- what is a read task holding, and does batch/block alignment fix it?
#
# Standing puzzle: the identical read costs 23 MiB of private memory in a bare
# process (exp5) and ~1 GiB inside a Ray read task (exp3). Three candidate
# explanations are already dead:
#
#   * the transport   -- local and S3 both show ~1 GiB (exp3 both ways)
#   * crate retention -- `retained_bytes` is 20 MiB per row group, and
#                        `col_group_rgs = 0`: the column-group path never ran
#   * the decode budget ALONE -- 2 -> 128 MiB moves per-task USS by 7%
#
# What per-task USS *does* equal is one input file's decoded size: lineitem is
# ~6.0M rows x 172 B = 1.03 GiB, measured 958-1035 MiB.
#
# Two phases, because the budget is probably only meaningful RELATIVE to Ray's
# streaming unit:
#
# Phase A -- block sweep, both readers, budget at the shipping default.
#   USS rises with block size  -> output blocks accumulate in the task. That is
#                                 the block layer, it hits PyArrow too, and the
#                                 knob is a workaround rather than a fix.
#   USS flat across 32x        -> blocks are not what is retained; the fragment
#                                 loop materializing a whole file per yielded
#                                 table is the next suspect, and that one is
#                                 ours.
#   PyArrow flat at ~1.2 GiB too -> the retention predates the arrow-rs reader.
#
# Phase B -- budget sweep at a FIXED 128 MiB block, arrow-rs only.
#   Ray's streaming unit is target_max_block_size. Handing the block builder
#   2 MiB batches means it accumulates ~64 of them and concatenates, and a
#   concat needs inputs and output alive at once. Emitting batches at or near
#   the block size should let them pass straight through. exp3 already hinted at
#   this: 32 MiB beat 2 MiB on every axis, on both transports.
#   USS falls as budget -> block  -> alignment is the fix; set the default from
#                                    target_block_size instead of a constant.
#   USS flat                      -> the block builder is not the accumulator.
#
# Local disk only: exp3 showed the transports agree, and S3 adds a fetch path
# that is not under test here.
#
# Runtime: ~20 min for 12 arms.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
check_env

LOCAL_ROOT="${LOCAL_ROOT:-$HOME/arrow_rs_local}"
DATA="${DATA:-$LOCAL_ROOT/lineitem}"
[ -d "$DATA" ] || { echo "FATAL: $DATA missing -- run exp5_standalone.sh first"; exit 1; }
WRITE_DIR="$LOCAL_ROOT/exp6_write"
RESULTS="$OUT_DIR/exp6"
BLOCKS="${BLOCKS:-16 64 128 512}"
BUDGETS="${BUDGETS:-8 32 64 128}"
# On-disk bytes per read task. lineitem compresses ~4x, so these are roughly
# 1 GiB / 256 MiB / 64 MiB of decoded data per task.
CHUNKS="${CHUNKS:-256 64 16}"
FIXED_BLOCK="${FIXED_BLOCK:-128}"
mkdir -p "$RESULTS"

# One process per arm: Ray reuses workers, and MemoryProfiler reads whole-process
# private memory, so a second arm in the same cluster inherits the first arm's
# high-water mark. That worker-reuse effect is the exact artifact that made
# read_from_uris look like a regression in the release A/B.
run_probe() {  # tag, then probe args
  local tag="$1"; shift
  echo "=== $tag"
  python "$HERE/block_retention_probe.py" \
    --source "$DATA" --write-to "$WRITE_DIR" --out "$RESULTS/$tag.json" "$@" \
    > "$RESULTS/$tag.log" 2>&1 \
    || echo "  FAILED -- see $RESULTS/$tag.log"
  # A silent stats failure prints as 0 MiB / 0 tasks, which reads like a
  # measurement. Surface it at run time instead of in the summary table.
  grep -h "WARNING: no Read operator" "$RESULTS/$tag.log" 2>/dev/null || true
}

for reader in pyarrow arrow_rs; do
  for mib in $BLOCKS; do
    run_probe "A_${reader}_blk${mib}" --reader "$reader" --block-mib "$mib"
  done
done

for budget in $BUDGETS; do
  run_probe "B_arrow_rs_bud${budget}" \
    --reader arrow_rs --block-mib "$FIXED_BLOCK" --decode-budget-mib "$budget"
done

# Phase C -- the axis that phases A and B could not distinguish.
#
# A flat block sweep does NOT mean "nothing is retained": if a task holds ALL of
# its output, the total is invariant to how that output is chunked, so flat is
# exactly what retention-of-everything looks like. The way to tell is to change
# how much each task produces.
#
# The knob is parquet_chunker_target_chunk_size, NOT override_num_blocks -- the
# V2 path ignores override_num_blocks entirely (no reference to it anywhere
# under datasource_v2/), which is why the first attempt at this phase reported
# 4 tasks for 4, 16 and 64 and therefore measured nothing. ParquetFileChunker
# splits a file only when its ON-DISK size exceeds the target, and the built-in
# default is 1 GiB, so ordinary files are never split: one task per file, each
# decoding that file whole. Our lineitem files decode to ~1.03 GiB, which is
# precisely the per-task USS we keep measuring.
#
#   USS falls ~proportionally  -> the task holds its output. Smaller read tasks
#                                 are then an immediate mitigation, and the real
#                                 fix is finding what keeps the blocks alive.
#   USS pinned near 1 GiB      -> nothing is held; peak USS is tracking the
#                                 allocator's high-water mark over the task's
#                                 lifetime churn, not live data. Then no amount
#                                 of streaming helps and the lever is the
#                                 allocator (arena count, trim, or task size).
#
# Chunk sizes are on-disk bytes; lineitem compresses ~4x, so 64 MiB on disk is
# roughly a 256 MiB decode.
for chunk in $CHUNKS; do
  for reader in pyarrow arrow_rs; do
    run_probe "C_${reader}_chunk${chunk}" \
      --reader "$reader" --block-mib "$FIXED_BLOCK" --chunk-mib "$chunk"
  done
done
rm -rf "$WRITE_DIR"

RESULTS="$RESULTS" FIXED_BLOCK="$FIXED_BLOCK" python - <<'PYEOF' | tee "$OUT_DIR/exp6_summary.txt"
import glob
import json
import os

MiB = 1024 * 1024
rows = []
for path in sorted(glob.glob(os.path.join(os.environ["RESULTS"], "*.json"))):
    with open(path) as handle:
        r = json.load(handle)
    r["tag"] = os.path.basename(path)[: -len(".json")]
    rows.append(r)

if any(r.get("stats_error") for r in rows):
    print("!! some arms reported no Read operator:")
    for r in rows:
        if r.get("stats_error"):
            print(f"   {r['tag']}: {r['stats_error']}")
    print()


def show(subset, label_of, title):
    head = (
        f"{title:<22}{'avg USS/task':>15}{'max USS/task':>15}"
        f"{'tasks':>7}{'rows/task':>12}{'wall':>8}"
    )
    print(head)
    print("-" * len(head))
    series = []
    for r in subset:
        avg = (r.get("avg_max_uss_per_task") or 0) / MiB
        mx = (r.get("max_uss_per_task") or 0) / MiB
        series.append(avg)
        print(
            f"{label_of(r):<22}{avg:>12.0f}MiB{mx:>12.0f}MiB"
            f"{r.get('task_count') or 0:>7}"
            f"{(r.get('rows_per_task_mean') or 0):>12,.0f}{r['wall_s']:>8.1f}"
        )
    if len(series) > 1 and min(series) > 0:
        print(f"{'':<22}  spread {max(series) / min(series):.2f}x\n")
    else:
        print()


for reader in ("pyarrow", "arrow_rs"):
    subset = sorted(
        (r for r in rows if r["tag"].startswith("A_") and r["reader"] == reader),
        key=lambda r: r["block_mib"],
    )
    if subset:
        show(subset, lambda r: f"{r['reader']} blk={r['block_mib']}M", "PHASE A block")

fixed = os.environ["FIXED_BLOCK"]
subset = sorted(
    (r for r in rows if r["tag"].startswith("B_")),
    key=lambda r: r["decode_budget_mib"] or 0,
)
if subset:
    show(
        subset,
        lambda r: f"budget={r['decode_budget_mib']}M",
        f"PHASE B blk={fixed}M",
    )

# Phase C: the input is fixed, so bytes-per-task falls as the task count rises.
# USS x tasks constant means USS tracks per-task volume (retention); USS flat
# while the task count climbs means it is pinned regardless of how little a task
# produces (allocator high-water).
for reader in ("pyarrow", "arrow_rs"):
    subset = sorted(
        (r for r in rows if r["tag"].startswith("C_") and r["reader"] == reader),
        key=lambda r: -(r["chunk_mib"] or 0),
    )
    if not subset:
        continue
    head = (
        f"{'PHASE C ' + reader:<22}{'avg USS/task':>15}{'tasks':>7}"
        f"{'USS x tasks':>14}{'wall':>8}"
    )
    print(head)
    print("-" * len(head))
    for r in subset:
        avg = (r.get("avg_max_uss_per_task") or 0) / MiB
        # uss_num_samples is one sample per task that ran -- an independent
        # count, and the check that the chunk knob actually took effect.
        n = r.get("uss_num_samples") or r.get("task_count") or 0
        print(
            f"{'chunk=' + str(r['chunk_mib']) + 'M':<22}{avg:>12.0f}MiB{n:>7}"
            f"{avg * n / 1024:>11.1f}GiB{r['wall_s']:>8.1f}"
        )
    if len({r.get("uss_num_samples") for r in subset}) == 1:
        print(
            f"{'':<22}  !! task count never changed -- the chunk knob did not "
            "take effect; this phase measured nothing"
        )
    print()

print(
    "A lineitem file is ~6.0M rows x 172 B = 1.03 GiB decoded; standalone, the"
    "\nsame read holds 23 MiB."
    "\n"
    "\nPhase A/B flat does NOT mean nothing is retained: if a task holds ALL its"
    "\noutput, the total is invariant to chunking, so flat is what that looks like."
    "\nPhase C is the discriminator. USS falling ~proportionally with task size"
    "\nmeans the task holds its output. USS pinned near 1 GiB while tasks shrink"
    "\n16x means it is not live data at all -- it is the allocator's high-water"
    "\nmark over lifetime churn, and streaming cannot fix it."
)
PYEOF
