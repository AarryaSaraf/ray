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
THREADS="${THREADS:-1 2 4 8}"
# Phase D runs at the SMALLEST task size, where the fixed floor is the dominant
# term and therefore easiest to see: at 62 MiB decoded per task the measured gap
# is 309 vs 242 MiB, so a ~82 MiB thread-pool cost is most of it.
FIXED_CHUNK="${FIXED_CHUNK:-16}"
FIXED_BLOCK="${FIXED_BLOCK:-128}"
# Which phases to run. Phases A-C are answered (see FINDINGS.md section 4); the
# open question is D, so `PHASES=D ./exp6_block_retention.sh` is the useful
# invocation now. The summary prints whatever results are on disk, so running D
# alone still shows A/B/C from the previous run underneath it.
PHASES="${PHASES:-A B C D}"
mkdir -p "$RESULTS"
has_phase() { case " $PHASES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

# A phase that re-runs must not read its own previous generation: an earlier
# Phase C keyed on override_num_blocks, and the summary cannot sort those
# alongside chunk_mib. Clear only the phases about to be rewritten, so a
# selective run leaves the other phases' results intact.
for p in $PHASES; do rm -f "$RESULTS/${p}_"*.json; done

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

if has_phase A; then
for reader in pyarrow arrow_rs; do
  for mib in $BLOCKS; do
    run_probe "A_${reader}_blk${mib}" --reader "$reader" --block-mib "$mib"
  done
done
fi

if has_phase B; then
for budget in $BUDGETS; do
  run_probe "B_arrow_rs_bud${budget}" \
    --reader arrow_rs --block-mib "$FIXED_BLOCK" --decode-budget-mib "$budget"
done
fi

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
# ANSWERED: neither, cleanly. USS does fall with task size (so it IS live data,
# and the allocator branch is dead), but the two readers fall along different
# lines -- see FINDINGS.md section 4. With D = MiB decoded per task,
#     PyArrow   USS ~= 183 + 0.96*D      (holds ~100% of what it decodes)
#     arrow-rs  USS ~= 287 + 0.69*D      (holds ~69%, on a 104 MiB higher floor)
# which crosses over at D ~= 385 MiB: arrow-rs wins on big tasks and loses on
# small ones. That fixed floor is what phase D goes after.
#
# Chunk sizes are on-disk bytes; lineitem compresses ~4x, so 64 MiB on disk is
# roughly a 256 MiB decode.
if has_phase C; then
for chunk in $CHUNKS; do
  for reader in pyarrow arrow_rs; do
    run_probe "C_${reader}_chunk${chunk}" \
      --reader "$reader" --block-mib "$FIXED_BLOCK" --chunk-mib "$chunk"
  done
done
fi

# Phase D -- what is the fixed per-task floor made of?
#
# Phase C says arrow-rs pays ~104 MiB more per task than PyArrow no matter how
# small the task gets. The leading suspect is the reader thread pool:
# _dispatch_fragment_reads (file_reader.py:462) decodes
# min(RAY_DATA_READ_FILES_NUM_THREADS, len(fragments)) fragments concurrently,
# the default is 4, and the crate's profiler measured 20.4 MiB retained per row
# group. Four in flight is ~82 MiB live at every instant regardless of task
# size, and even the 68-task arm still holds ~7 row groups, enough to keep all
# four threads fed. 82 against a measured 104 is a good fit.
#
# Run at the smallest chunk, where the floor dominates the linear term.
#
#   arrow_rs USS falls ~20 MiB per thread removed -> confirmed. The floor is
#       concurrency x row-group retention, and the fix is a per-reader thread
#       default (note arrow-rs SCALES with threads where PyArrow is saturated,
#       so the answer may be more threads with smaller per-thread retention, not
#       fewer threads).
#   arrow_rs USS flat across 1..8 threads        -> the floor is Rust-side
#       (tokio thread stacks, object_store client, allocator arenas) and the
#       search moves into the crate.
#   BOTH readers fall together                   -> shared cost, not ours; it
#       cancels in the ratio and the 104 MiB gap is still unexplained.
#
# The pyarrow arm is the control: the pool is shared, so if only arrow-rs moves,
# the retention per in-flight fragment is the crate's, not Ray's.
if has_phase D; then
for threads in $THREADS; do
  for reader in pyarrow arrow_rs; do
    run_probe "D_${reader}_thr${threads}" \
      --reader "$reader" --block-mib "$FIXED_BLOCK" --chunk-mib "$FIXED_CHUNK" \
      --threads "$threads"
  done
done
fi
rm -rf "$WRITE_DIR"

RESULTS="$RESULTS" FIXED_BLOCK="$FIXED_BLOCK" python - <<'PYEOF' | tee "$OUT_DIR/exp6_summary.txt"
import glob
import json
import os

MiB = 1024 * 1024
# 4 lineitem files x ~6.0M rows x 172 B/row. Phase C fixes the input and varies
# the task count, so bytes-per-task is this over the number of tasks -- that is
# the x-axis the USS fit is against.
TOTAL_DECODED_MIB = 4219
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
        # .get, not [] -- a result file written by an older generation of this
        # script lacks the key entirely, and a KeyError here would take out the
        # whole summary after the arms have already run.
        key=lambda r: -(r.get("chunk_mib") or 0),
    )
    if not subset:
        continue
    head = (
        f"{'PHASE C ' + reader:<22}{'avg USS/task':>15}{'max/avg':>9}{'tasks':>7}"
        f"{'MiB decoded/task':>18}{'wall':>8}"
    )
    print(head)
    print("-" * len(head))
    for r in subset:
        avg = (r.get("avg_max_uss_per_task") or 0) / MiB
        mx = (r.get("max_uss_per_task") or 0) / MiB
        # uss_num_samples is one sample per task that ran -- an independent
        # count, and the check that the chunk knob actually took effect.
        n = r.get("uss_num_samples") or r.get("task_count") or 0
        print(
            f"{'chunk=' + str(r.get('chunk_mib')) + 'M':<22}{avg:>12.0f}MiB"
            f"{(mx / avg if avg else 0):>9.2f}{n:>7}"
            f"{(TOTAL_DECODED_MIB / n if n else 0):>18,.0f}{r['wall_s']:>8.1f}"
        )
    if len({r.get("uss_num_samples") for r in subset}) == 1:
        print(
            f"{'':<22}  !! task count never changed -- the chunk knob did not "
            "take effect; this phase measured nothing"
        )
    print()

# Phase D: task size is FIXED, only the reader's fragment concurrency moves. Any
# USS that tracks the thread count is the per-in-flight-fragment cost, which is
# the part of the fixed floor that does not shrink when tasks shrink.
for reader in ("pyarrow", "arrow_rs"):
    subset = sorted(
        (r for r in rows if r["tag"].startswith("D_") and r["reader"] == reader),
        key=lambda r: r.get("threads") or 0,
    )
    if not subset:
        continue
    head = (
        f"{'PHASE D ' + reader:<22}{'avg USS/task':>15}{'vs 1 thread':>13}"
        f"{'tasks':>7}{'wall':>8}"
    )
    print(head)
    print("-" * len(head))
    base = (subset[0].get("avg_max_uss_per_task") or 0) / MiB
    for r in subset:
        avg = (r.get("avg_max_uss_per_task") or 0) / MiB
        print(
            f"{'threads=' + str(r.get('threads')):<22}{avg:>12.0f}MiB"
            f"{(avg - base):>+12.0f}M"
            f"{(r.get('uss_num_samples') or 0):>7}{r['wall_s']:>8.1f}"
        )
    print()

print(
    "A lineitem file is ~6.0M rows x 172 B = 1.03 GiB decoded; standalone, the"
    f"\nsame read holds 23 MiB. Four files = {TOTAL_DECODED_MIB:,} MiB decoded total."
    "\n"
    "\nPhase A/B flat does NOT mean nothing is retained: if a task holds ALL its"
    "\noutput, the total is invariant to chunking, so flat is what that looks like."
    "\n"
    "\nPhase C answered the live-data question -- USS moves with task size, so it"
    "\nis live data, not an allocator high-water mark. But the two readers fall"
    "\nalong different lines: PyArrow ~= 183 + 0.96*D, arrow-rs ~= 287 + 0.69*D"
    "\n(D = MiB decoded per task). arrow-rs holds less of what it decodes but"
    "\ncarries ~104 MiB more fixed cost, so they cross over at D ~= 385 MiB."
    "\n"
    "\nPhase D asks what that fixed cost is. Task size is held at the smallest"
    "\nchunk and only fragment concurrency moves. ~20 MiB per thread on the"
    "\narrow_rs arm and ~0 on the pyarrow arm means the floor is the thread pool"
    "\nholding one retained row group per in-flight fragment. Flat on both means"
    "\nit is Rust-side and the search moves into the crate."
)
PYEOF
