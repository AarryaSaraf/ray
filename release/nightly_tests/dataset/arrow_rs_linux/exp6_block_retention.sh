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
#   RE-RUN 2026-08-07, twice changed. (1) It now runs UNFUSED. Every earlier
#   phase B arm carried a write, and phase F later showed the writer held 589 of
#   1032 MiB at the default chunk -- so phase B was reading a knob's effect on
#   the decoder through 57% of unrelated writer memory, which is enough to hide
#   it. Phases F and G were already fixed; B was missed. (2) It now runs at
#   threads=1, matching F, G and the shipped default, so the four arms differ in
#   the budget and nothing else.
#
#   The reason to re-ask now is phase G. G swept the block size unfused and
#   found arrow-rs costs about `base + 2.4 x target_max_block_size` -- 167 MiB
#   at a 16 MiB block rising to 1051 at 512. A factor of 2.4 on a knob that is
#   supposed to bound one block in flight is the signature of a copying
#   concatenate: inputs plus output live at once is 2x, plus the block being
#   handed onward is the remaining 0.4. If that is what it is, then feeding the
#   builder batches the size of a block should collapse the 2.4 toward 1, and
#   the budget default should be derived from target_max_block_size rather than
#   pinned at a constant. The sweep spans the old default (2 MiB) through the
#   one just shipped (32) to a full block (128).
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
# Spans the old shipping default (2), the one just shipped (32), and a full
# block (128) -- the alignment point phase B exists to test. 8 keeps a point
# between 2 and 32 so a monotone trend can be distinguished from a step.
BUDGETS="${BUDGETS:-2 8 32 128}"
# On-disk bytes per read task. lineitem compresses ~4x, so these are roughly
# 1 GiB / 256 MiB / 64 MiB of decoded data per task.
CHUNKS="${CHUNKS:-256 64 16}"
THREADS="${THREADS:-1 2 4 8}"
# Phase D runs at the SMALLEST task size, where the fixed floor is the dominant
# term and therefore easiest to see: at 62 MiB decoded per task the measured gap
# is 309 vs 242 MiB, so a ~82 MiB thread-pool cost is most of it.
FIXED_CHUNK="${FIXED_CHUNK:-16}"
FIXED_BLOCK="${FIXED_BLOCK:-128}"
# Phase E: threads crossed with task size. Phase D showed threads=1 is free at
# the smallest chunk, but at that chunk a task holds only ~3 fragments -- the
# question is whether serial decode still costs nothing when a task holds 49.
E_CHUNKS="${E_CHUNKS:-16 64 256 0}"
E_THREADS="${E_THREADS:-1 4}"
# Which phases to run. A-D are answered (arrow_rs_docs/regression_testing.md
# section 8); the open question is E, so `PHASES=E ./exp6_block_retention.sh` is
# the useful invocation now. The summary prints whatever results are on disk, so
# running one phase still shows the others from the previous run underneath it.
PHASES="${PHASES:-A B C D E F G}"
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
#
# PROBE_WRITE=0 drops --write-to, so the read is consumed by
# iter_internal_ref_bundles instead of being fused into a write task. Phases A-E
# all ran fused, which means they measured read+write together -- see phase F.
run_probe() {  # tag, then probe args
  local tag="$1"; shift
  local write_args=(--write-to "$WRITE_DIR")
  [ "${PROBE_WRITE:-1}" = 0 ] && write_args=()
  echo "=== $tag"
  python "$HERE/block_retention_probe.py" \
    --source "$DATA" "${write_args[@]}" --out "$RESULTS/$tag.json" "$@" \
    > "$RESULTS/$tag.log" 2>&1 \
    || echo "  FAILED -- see $RESULTS/$tag.log"
  # A silent stats failure prints as 0 MiB / 0 tasks, which reads like a
  # measurement. Surface it at run time instead of in the summary table.
  grep -h "^WARNING:" "$RESULTS/$tag.log" 2>/dev/null || true
}

if has_phase A; then
for reader in pyarrow arrow_rs; do
  for mib in $BLOCKS; do
    run_probe "A_${reader}_blk${mib}" --reader "$reader" --block-mib "$mib"
  done
done
fi

if has_phase B; then
# PROBE_WRITE=0 and --threads 1: see the phase B header. Fused, the writer's
# 589 MiB sat on top of every arm; at four threads the fragment pool added a
# floor that moves with neither knob. Both had to go before the budget's own
# effect could be read.
for budget in $BUDGETS; do
  PROBE_WRITE=0 run_probe "B_arrow_rs_bud${budget}" \
    --reader arrow_rs --block-mib "$FIXED_BLOCK" --threads 1 \
    --decode-budget-mib "$budget"
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

# Phase E -- does serial decode stay free as the task gets bigger?
#
# ANSWERED by D: the whole ~104 MiB floor is the fragment thread pool. arrow-rs
# pays +96 MiB for it and PyArrow +18, and at threads=1 arrow-rs (213 MiB) beats
# PyArrow (223) even at the smallest task size, so the Phase C crossover is
# entirely an artifact of the pool.
#
# Two things D could not see, and E must, before "default the arrow-rs path to
# one fragment thread" becomes a patch:
#
#   1. D ran only at chunk=16M, where a task holds ~3 fragments (4 files / 68
#      tasks x 49 row groups per file). num_workers = min(threads, fragments),
#      so threads=4 and threads=8 both clamped to ~3 and the top of that sweep
#      measured nothing. At the default chunk a task holds 49 fragments, and
#      serial-over-49 is a completely different proposition from serial-over-3.
#   2. D's headline is that threads=1 costs no wall time. That is only
#      established for ~3 fragments.
#
# Note threads=1 is not merely less concurrency: `num_workers <= 1` returns
# early into _read_fragments_sequential (file_reader.py:481) and make_async_gen
# is never constructed. That is why 86 of the 96 MiB arrives on the SECOND
# worker rather than spreading evenly -- the second worker buys a concurrent
# decode and the whole queue apparatus at once. E cannot separate those two
# either; it only asks whether the trade stays good.
#
#   threads=1 wall stays flat as the chunk grows -> ship the serial default.
#   threads=1 wall degrades at big chunks        -> the win is task-size
#       dependent, and the fix is per-fragment retention (or a smaller default
#       chunk) rather than serialising the pool.
#
# chunk=0 means "leave the default" (1 GiB, i.e. one whole file per task).
if has_phase E; then
for chunk in $E_CHUNKS; do
  for threads in $E_THREADS; do
    chunk_args=()
    [ "$chunk" != 0 ] && chunk_args=(--chunk-mib "$chunk")
    run_probe "E_arrow_rs_chunk${chunk}_thr${threads}" \
      --reader arrow_rs --block-mib "$FIXED_BLOCK" --threads "$threads" \
      "${chunk_args[@]}"
  done
done
fi

# Phase F -- is the retention the reader's, or the writer's?
#
# Phase E settled the shipping question: at threads=1 arrow-rs is 0.85-0.88x
# PyArrow's per-task memory at EVERY task size, at wall parity. But that is a
# 13% win, and the standalone reader (exp5) held 23 MiB for the same 4219 MiB of
# decode -- 2%, against 82.5% here. The slope, not the floor, is where the
# missing win went, and nothing tried so far moves it: block size 1.00-1.01x
# across 32x, decode budget 1.04x across 16x.
#
# The confound: EVERY arm in phases A-E passed --write-to, so all of them
# measured `read_parquet -> write_parquet` fused into one task. The Parquet
# writer holds its own column buffers in that same process, so the 0.825 slope
# may be the writer's accumulation rather than the reader's retention. No
# experiment here has ever run the read unfused.
#
# F re-runs the E grid with the write dropped: the read is consumed by
# iter_internal_ref_bundles, which pulls blocks through without materializing
# them.
#
#   unfused slope collapses  -> the reader streams correctly and the retention
#       is the writer's. That reopens H7 in a new form (not "small batches make
#       the writer accumulate" -- the budget sweep killed that -- but "the
#       writer accumulates regardless of batch size"), and the next target is
#       the write path, which is shared with PyArrow.
#   unfused slope unchanged  -> the read task itself holds its output, and the
#       gap between 23 MiB standalone and ~0.8x-of-decode in Ray is the
#       reader/block layer after all.
#
# The pyarrow arm is the control: if BOTH readers' slopes collapse unfused, the
# writer is the accumulator and it is not an arrow-rs defect at all.
if has_phase F; then
for chunk in $E_CHUNKS; do
  for reader in pyarrow arrow_rs; do
    chunk_args=()
    [ "$chunk" != 0 ] && chunk_args=(--chunk-mib "$chunk")
    PROBE_WRITE=0 run_probe "F_${reader}_chunk${chunk}" \
      --reader "$reader" --block-mib "$FIXED_BLOCK" --threads 1 \
      "${chunk_args[@]}"
  done
done
fi

# Phase G -- what is the unfused ceiling made of?
#
# F answered its question: the writer was holding 589 MiB, and unfused arrow-rs
# SATURATES -- 185 / 326 / 422 / 443 MiB across a 17x rise in task size, while
# PyArrow climbs 209 / 347 / 471 / 901 with no ceiling. Saturation is the whole
# point of the project (a bigger file stops costing more), so the next question
# is what sets the ceiling, because that number is the OOM budget.
#
# 443 MiB is ~3.5 x the 128 MiB default target_max_block_size. If the ceiling is
# "a few finished output blocks waiting to be handed to the object store", then
# it is a knob and memory becomes something we set rather than something the
# input decides.
#
# Phase A already swept block size and got 1.00-1.01x -- but A was FUSED, and F
# then showed the writer's 589 MiB was the majority of that measurement. A knob
# worth ~50 MiB is invisible underneath it. So A has to be re-run without the
# write before its flatness means anything.
#
#   USS scales with block size -> the ceiling is retained output blocks. Ship a
#       smaller target_max_block_size (or find why they are not released) and
#       the saturation point drops with it.
#   USS flat across 32x        -> the ~443 MiB is something else: the decode
#       working set across in-flight fragments, Rust allocator arenas, or the
#       block builder's concatenate scratch. Next probe is the crate's own
#       retained_bytes against this number.
#
# Run at the DEFAULT chunk (one file, ~1055 MiB decoded per task): that is where
# both readers are furthest apart and where the ceiling is actually reached. At
# chunk=16M a task only produces 62 MiB total, less than one block, so every
# block setting would look identical for a trivial reason.
if has_phase G; then
for reader in pyarrow arrow_rs; do
  for mib in $BLOCKS; do
    PROBE_WRITE=0 run_probe "G_${reader}_blk${mib}" \
      --reader "$reader" --block-mib "$mib" --threads 1
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
            f"{('chunk=' + str(r['chunk_mib']) + 'M') if r.get('chunk_mib') else 'chunk=default(1G)':<22}{avg:>12.0f}MiB"
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

# Phase E: threads x task size, arrow_rs only. Phase D established that
# threads=1 is both lighter and no slower -- but only where a task held ~3
# fragments. The wall column is the one that matters here: if serialising stays
# free as the task grows to 49 fragments, the serial default ships.
e_rows = [r for r in rows if r["tag"].startswith("E_")]
if e_rows:
    by_chunk = {}
    for r in e_rows:
        by_chunk.setdefault(r.get("chunk_mib") or 0, {})[r.get("threads") or 0] = r
    head = (
        f"{'PHASE E arrow_rs':<22}{'USS thr=1':>12}{'USS thr=4':>12}{'mem saved':>11}"
        f"{'wall thr=1':>12}{'wall thr=4':>12}{'wall cost':>11}"
    )
    print(head)
    print("-" * len(head))
    for chunk in sorted(by_chunk, key=lambda c: (c == 0, c)):
        arms = by_chunk[chunk]
        lo, hi = arms.get(1), arms.get(4)
        if not (lo and hi):
            continue
        lo_u = (lo.get("avg_max_uss_per_task") or 0) / MiB
        hi_u = (hi.get("avg_max_uss_per_task") or 0) / MiB
        label = "chunk=default" if chunk == 0 else f"chunk={chunk}M"
        print(
            f"{label:<22}{lo_u:>9.0f}MiB{hi_u:>9.0f}MiB{(hi_u - lo_u):>+10.0f}M"
            f"{lo['wall_s']:>12.1f}{hi['wall_s']:>12.1f}"
            f"{(lo['wall_s'] / hi['wall_s'] if hi['wall_s'] else 0):>10.2f}x"
        )
    print(
        f"{'':<22}  mem saved > 0 and wall cost ~1.00x at every chunk -> serial\n"
        f"{'':<22}  is strictly better and the default should change.\n"
    )


def fit(points):
    """Least-squares USS = intercept + slope * D over [(D, USS)].

    The slope is the fraction of what a task decodes that it still holds at
    peak: ~1.0 means it keeps everything (PyArrow's scanner), and the standalone
    reader in exp5 managed 0.02. That number, not the intercept, is where the
    missing win lives.
    """
    n = len(points)
    if n < 2:
        return None, None
    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n
    denom = sum((p[0] - mx) ** 2 for p in points)
    if denom == 0:
        return None, None
    slope = sum((p[0] - mx) * (p[1] - my) for p in points) / denom
    return my - slope * mx, slope


# Phase F: the same grid unfused (no write). Phases A-E all fused a write onto
# the read, so they measured read+write in one task. If the slope collapses
# here, the accumulation was the writer's.
for reader in ("pyarrow", "arrow_rs"):
    subset = sorted(
        (r for r in rows if r["tag"].startswith("F_") and r["reader"] == reader),
        key=lambda r: r.get("uss_num_samples") or 0,
    )
    if not subset:
        continue
    head = (
        f"{'PHASE F ' + reader:<22}{'avg USS/task':>15}{'tasks':>7}"
        f"{'MiB decoded/task':>18}{'wall':>8}"
    )
    print(head)
    print("-" * len(head))
    points = []
    for r in subset:
        avg = (r.get("avg_max_uss_per_task") or 0) / MiB
        n = r.get("uss_num_samples") or 0
        d = TOTAL_DECODED_MIB / n if n else 0
        label = (
            f"chunk={r['chunk_mib']}M" if r.get("chunk_mib") else "chunk=default(1G)"
        )
        if r.get("stats_error"):
            print(f"{label:<22}{'NO STATS -- see the log':>50}")
            continue
        points.append((d, avg))
        print(f"{label:<22}{avg:>12.0f}MiB{n:>7}{d:>18,.0f}{r['wall_s']:>8.1f}")
    # Fitting through arms that reported nothing produces a confident-looking
    # slope out of zeros, which is worse than printing no slope at all.
    intercept, slope = fit(points)
    if slope is None:
        print(f"{'':<22}  too few valid arms to fit a slope")
    else:
        print(
            f"{'':<22}  unfused USS ~= {intercept:.0f} MiB + {slope:.3f} x D"
            f"   ({len(points)}/{len(subset)} arms)"
        )
        # This used to print "compare fused, threads=1: arrow-rs was 162 + 0.825
        # x D" as a hardcoded string under BOTH reader tables, so under the
        # pyarrow one it compared PyArrow's unfused fit to arrow-rs's fused fit
        # -- two readers, two configurations, no shared axis. The comparison
        # worth printing is the SAME reader on the other transport, which is
        # what phase S measures, so the reference values are per reader.
        s3_fit = {"pyarrow": (189, 0.614), "arrow_rs": (306, 0.361)}.get(reader)
        if s3_fit:
            di, ds = intercept - s3_fit[0], slope - s3_fit[1]
            print(
                f"{'':<22}  same reader over S3 (exp7 phase S): "
                f"{s3_fit[0]} + {s3_fit[1]:.3f} x D"
                f"\n{'':<22}  transport costs this reader {-di:+.0f} MiB fixed and "
                f"{-ds:+.3f} per MiB decoded."
                f"\n{'':<22}  A near-zero slope term means the transport is a "
                "constant, which is\n"
                f"{'':<22}  what PyArrow shows. A positive one means our S3 path "
                "retains MORE\n"
                f"{'':<22}  per byte, which no amount of shrinking the constant "
                "will fix."
            )
    print()

# Phase G: the block sweep again, but unfused and serial -- i.e. the conditions
# under which the ~443 MiB ceiling was measured. Phase A ran this fused, where
# the writer's 589 MiB swamped anything the block knob could do.
g_rows = [r for r in rows if r["tag"].startswith("G_")]
if g_rows:
    head = (
        f"{'PHASE G unfused blk':<22}{'pyarrow':>12}{'arrow_rs':>12}{'ratio':>8}"
        f"{'wall P':>9}{'wall R':>9}"
    )
    print(head)
    print("-" * len(head))
    by_blk = {}
    for r in g_rows:
        by_blk.setdefault(r["block_mib"], {})[r["reader"]] = r
    series = {"pyarrow": [], "arrow_rs": []}
    for blk in sorted(by_blk):
        arms = by_blk[blk]
        p, a = arms.get("pyarrow"), arms.get("arrow_rs")
        if not (p and a):
            continue
        pu = (p.get("avg_max_uss_per_task") or 0) / MiB
        au = (a.get("avg_max_uss_per_task") or 0) / MiB
        series["pyarrow"].append(pu)
        series["arrow_rs"].append(au)
        print(
            f"{'block=' + str(blk) + 'M':<22}{pu:>9.0f}MiB{au:>9.0f}MiB"
            f"{(au / pu if pu else 0):>7.2f}x{p['wall_s']:>9.1f}{a['wall_s']:>9.1f}"
        )
    for reader, vals in series.items():
        if len(vals) > 1 and min(vals) > 0:
            print(f"{'':<22}  {reader} spread {max(vals) / min(vals):.2f}x")
    print(
        f"\n{'':<22}  arrow_rs spread >> 1 -> the ~443 MiB ceiling is retained\n"
        f"{'':<22}  output blocks, and it is a knob. Flat -> it is not, and the\n"
        f"{'':<22}  next suspect is in-flight decode scratch inside the crate.\n"
    )

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
    "\nPhase D found that fixed cost: it is the fragment thread pool. arrow-rs"
    "\npays +96 MiB for it and PyArrow +18, and at threads=1 arrow-rs (213 MiB)"
    "\nbeats PyArrow (223) even at the smallest task -- so the C crossover is an"
    "\nartifact of the pool, not a property of the decoder. 86 of the 96 arrives"
    "\non the SECOND worker because num_workers <= 1 returns early into"
    "\n_read_fragments_sequential (file_reader.py:481) and make_async_gen is never"
    "\nbuilt; the second worker buys a concurrent decode and the whole queue"
    "\napparatus at once. Above 2 the sweep measured nothing: num_workers ="
    "\nmin(threads, fragments) and a chunk=16M task holds only ~3 fragments."
    "\n"
    "\nPhase E answered the ship question yes: wall cost is 0.95-1.02x at every"
    "\nchunk, so serial fragments are free on time even at 49 fragments per task."
    "\nIt also corrected the floor story -- the pool's cost is +96/+148/+11/-2 MiB"
    "\nas the chunk grows, so it is not a constant, it only surfaces while the"
    "\naccumulated output is small enough not to dominate the peak. Serialized,"
    "\narrow-rs fits 162 + 0.825*D against PyArrow's 183 + 0.96*D: lower on both"
    "\nterms, no crossover, a uniform 0.85-0.88x at wall parity."
    "\n"
    "\nPhase F chases the part that is still missing. 0.825 means a Ray read task"
    "\nholds 82.5% of everything it decodes; the same read standalone (exp5) held"
    "\n2%. Nothing tried moves that slope. But every arm in A-E fused a WRITE onto"
    "\nthe read, so all of them measured read+write in one task and the writer's"
    "\ncolumn buffers were never separated out. F re-runs the grid unfused. If the"
    "\nslope collapses, the reader streams fine and the accumulator is the write"
    "\npath -- which is shared with PyArrow, so it would not be an arrow-rs defect"
    "\nat all. If it does not move, the read task really does hold its output."
    "\n"
    "\nANSWERED: the writer. At the default chunk it held 589 of the 1032 MiB, and"
    "\nunfused the ratio goes 0.87x -> 0.49x. The important part is not the ratio"
    "\nbut the SHAPE: unfused arrow-rs goes 185/326/422/443 MiB across a 17x rise"
    "\nin task size -- increments of +141/+96/+21, flattening toward a ceiling --"
    "\nwhile PyArrow goes 209/347/471/901, +138/+124/+430, still accelerating. A"
    "\nceiling is the anti-OOM property; a slope is not. PyArrow's own slope also"
    "\nworsens when fused (0.675 -> 0.96), so the write path is shared-code work,"
    "\nnot an arrow-rs defect."
    "\n"
    "\nPhase G asks what SETS that ceiling, because the ceiling is the OOM budget."
    "\n443 MiB is ~3.5 default blocks; if it is retained output blocks it is a"
    "\nknob. Phase A said no, but A was fused and the writer's 589 MiB would have"
    "\nhidden anything smaller, so it has to be asked again without the write."
)
PYEOF
