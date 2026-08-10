#!/usr/bin/env bash
# The 2026-08-10 batch: one command, one output to paste.
#
# On a FRESH box, run setup.sh first (README "Setup"). This script assumes the
# venv is built, the crate is compiled, and `source ~/ray/.venv/bin/activate` has
# been run in this shell.
#
# What earlier batches settled, so this one does not re-ask it:
#
#   * Saturation survives S3. Marginal MiB of peak per MiB decoded falls
#     0.75 -> 0.48 -> 0.47 across a 17x rise in task size, and at the release
#     read shape (D = 1055 MiB/task) arrow-rs is 0.90x PyArrow's per-task memory
#     at wall parity. On the lone-big-row-group fixture it is 0.55-0.63x.
#   * There is NO fixed per-task S3 cost. At D ~= 0, arrow-rs on S3 is 114 MiB
#     against PyArrow's 117 -- 3 MiB BELOW. The "+117 MiB" carried for weeks was
#     the intercept of a straight line fitted to a concave curve, read off at an
#     x-value the data never visited.
#   * Neither S3 knob is a memory knob. prefetch_budget_mb spreads 1.26x on
#     lineitem (non-monotone, shipping default is the WORST arm) and 1.14x on
#     bigrg, which is the noise floor. fetch_window_mb plans one unit on every
#     layout measured. Serial fragments are both lighter and no slower.
#   * The block-size cost is Ray's block layer, at comparable marginal cost for
#     both readers: PyArrow pays 3.18 then 1.46 MiB of peak per MiB of block
#     where we pay 2.46 then 1.59. Our base is ~128 MiB, theirs ~447.
#
# TWO CORRECTIONS that reframe this batch (2026-08-10):
#
#   A. The release regressions are mostly SPEED, not memory. By metric:
#      wide_schema_pipeline_tensors 4.90x READ OP WALL; read_parquet_autoscaling
#      3.06x READ OP WALL (USS 1.01x); read_parquet_fixed_size 2.64x READ OP WALL
#      *while winning* on memory (USS 0.77x, obj 0.61x). The only unambiguous
#      memory regression is write_parquet's per-task max USS -- shared writer
#      code, not the reader. exp6/exp7 measured per-task USS almost exclusively.
#   B. The noise floor is NOT uniformly large. `fixed_size` cases average 1.02x
#      (max 1.82x); only `autoscaling` WALL TIME has the ~2.5x control floor. So
#      read_parquet_fixed_size at 2.64x is signal.
#
# ALL FOUR RAN 2026-08-10 (real S3, us-west-2, REPEAT=3). Verdicts, so the
# predictions below are read as history and not as open questions -- full data in
# regression_testing.md §8.11 and the corrections in §8.12:
#
#   1 (D)  DEAD. A 256x budget sweep moves USS DOWN 99 MiB; spread 1.20x against a
#          1.13x floor. A batch cannot exceed its row group, so the knob has no
#          range on this layout. 32 MiB is still the best memory arm.
#   2 (X)  Parity BEATEN -- 0.66x memory at wall parity, 16/16 row groups on the
#          branch, 0 oversized units. But the MECHANISM below is wrong: cf=0
#          retains LESS decoded data (26 vs 76 MiB) and uses MORE memory, because
#          all 24 of its fetch units exceed the prefetch budget and serialise
#          (25.6 s vs 5.0). Column grouping earns its keep on the FETCH side.
#   3 (C)  DEAD. arrow-rs is 0.73-0.79x read-op time at every num_cpus in
#          {1,2,4,8}; cpu/wall never exceeds 1.11, so no oversubscription. But
#          read time is SCHEMA-DEPENDENT: 1.04-1.22x on the 64-column fixtures and
#          1.29-1.32x on chunked files, so "the regression inverts" holds for
#          narrow schemas only.
#   4 (K)  Argmins DO differ (PyArrow's best read time at 128 MiB blocks, ours at
#          32) -- but target_max_block_size is a GLOBAL DataContext knob for every
#          operator, not a reader setting, and at equal config we win at all three
#          sizes (0.76 / 0.90 / 0.74x). So it is evidence, not an action item.
#
# The four questions as they were posed:
#
#   1. Is the decoded channel the per-byte S3 surcharge?  (exp7 phase D)
#      Local slope 0.229, S3 slope 0.503, worth ~335 MiB at D = 1055. Both
#      readers pay the same ~+16 MiB to cross to S3; only ours pays per byte. On
#      S3 each stream holds S3_CHANNEL_DEPTH = 2 decoded batches sized by
#      decode_budget (~64 MiB at the shipped 32) that the local path never
#      allocates -- local drives the sync reader one batch at a time. exp6 swept
#      this knob on LOCAL disk, where there is no channel and "flat" was correct.
#      Prediction: ~2 x delta-budget, so ~60 of the 335 MiB. Flat points at
#      allocator retention, whose instrument is MALLOC_ARENA_MAX, not a knob.
#
#   2. What does the column-group path cost inside Ray?  (exp7 phase X)
#      RgDecode::Hstack has never executed under measurement here -- lineitem and
#      bigrg are both 16 narrow columns. (An earlier version of this header said
#      "every arm reported col_group_rgs = 0". That was never a measurement: the
#      profile directory was set through the environment and never created, the
#      reader's writes failed into a blanket except-pass, and the empty column
#      read as a zero. Fixed 2026-08-10 -- the probe now creates the directory and
#      the tables print NOPROF instead of a blank.)
#      Mechanism already confirmed on moto -- it accumulates every
#      batch of every column group before emitting one row, retaining 51.4 MiB
#      against a 39.9 MiB row group where the row-window path holds one 25.7 MiB
#      batch. Prediction: PARITY, not regression, because holding a whole decoded
#      row group is exactly what PyArrow does.
#
#   3. Is task CONCURRENCY the read-time regression?  (exp8 phase C, new)
#      Every experiment here pinned num_cpus at 8. PyArrow overlaps S3 latency
#      with an 8-thread io pool plus readahead; we run 1 fragment thread and rely
#      on tokio inside the task. At 8 concurrent tasks those tokio threads
#      contend for the same cores -- the only mechanism proposed that makes
#      read-op time worse while memory gets better, i.e. correction A's shape.
#
#   4. Does any Ray knob have a DIFFERENT optimum per reader?  (exp8 phase K/F)
#      Every ratio ever quoted here was measured at defaults tuned for PyArrow
#      over years. K sweeps chunk size and block size with BOTH readers and
#      prints two argmins instead of a ratio. F asks whether the worst release
#      regression (wide_schema tensors) is even on our path. ANSWERED: it IS --
#      0 fallbacks, fully native, 0.63x memory at 1.04x read-op time. The premise
#      (a support gate rejecting extension types) was falsified by our own
#      2026-07-28 commit removing the per-type gate.
#
# Runtime ~90-110 min all in: fixtures ~15 (the `tensors` layout is new),
# exp7 D+X ~25-35, exp8 C+K+F ~40-55, all at REPEAT=3. Split it if you prefer:
#
#     STAGES="fixtures exp7" ./run_next.sh      # then
#     STAGES="exp8" ./run_next.sh
#
# Must run in us-west-2, or it measures the distance to Oregon rather than memory.
#
#     cd release/nightly_tests/dataset/arrow_rs_linux
#     source ~/ray/.venv/bin/activate
#     ./run_next.sh 2>&1 | tail -n 500
#
# Paste the block between the BEGIN/END PASTE markers. Also written to
# out/next_summary.txt.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
check_env
check_s3

REPEAT="${REPEAT:-3}"
COMBINED="$OUT_DIR/next_summary.txt"
# Skip a stage that already ran, e.g. STAGES="exp8" after exp7 finished.
STAGES="${STAGES:-fixtures exp7 exp8}"
# S/W/T/Z/P are all answered; add them back explicitly only to re-establish a
# baseline on a new box (PHASES="S D X" is the full picture, ~90 min alone).
EXP7_PHASES="${EXP7_PHASES:-D X}"
EXP8_PHASES="${EXP8_PHASES:-C K F}"
has_stage() { case " $STAGES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

start=$(date +%s)

if has_stage fixtures; then
  echo "### building fixtures (idempotent -- a rebuild only happens if the"
  echo "### recorded geometry differs from what is asked for)"
  python "$HERE/make_fixtures.py" --bucket "$S3_ROOT"
fi

if has_stage exp7; then
  echo
  echo "### exp7 phases $EXP7_PHASES  (REPEAT=$REPEAT)"
  PHASES="$EXP7_PHASES" REPEAT="$REPEAT" bash "$HERE/exp7_s3_retention.sh" || {
    echo "!! exp7 exited nonzero -- its summary below covers whatever completed"
  }
fi

if has_stage exp8; then
  echo
  echo "### exp8 phases $EXP8_PHASES  (REPEAT=$REPEAT)"
  PHASES="$EXP8_PHASES" REPEAT="$REPEAT" bash "$HERE/exp8_ray_tuning.sh" || {
    echo "!! exp8 exited nonzero -- its summary below covers whatever completed"
  }
fi

# exp6 is local disk, where the run-to-run spread has been ~2% rather than S3's
# 13%, so one replicate suffices. Nothing in this batch needs it, so it is off by
# default and kept wired for the next local question.
if has_stage exp6; then
  echo
  echo "### exp6 phases ${EXP6_PHASES:-B}  (local, unfused, threads=1)"
  PHASES="${EXP6_PHASES:-B}" bash "$HERE/exp6_block_retention.sh" || {
    echo "!! exp6 exited nonzero -- its summary below covers whatever completed"
  }
fi

elapsed=$(( $(date +%s) - start ))

{
  echo "===================== BEGIN PASTE ====================="
  echo "arrow-rs next-steps batch"
  echo "  host    : $(uname -srm), $(nproc 2>/dev/null || echo '?') cpus"
  echo "  region  : ${AWS_DEFAULT_REGION:-unset}"
  echo "  commit  : $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo '?')"
  echo "  repeat  : $REPEAT"
  echo "  stages  : $STAGES  (exp7: $EXP7_PHASES / exp8: $EXP8_PHASES)"
  echo "  elapsed : $((elapsed / 60))m$((elapsed % 60))s"
  echo
  # The defaults under test, echoed rather than assumed: a stale checkout or a
  # leftover RAY_DATA_ARROW_RS_* export in the shell would otherwise produce a
  # table that looks like a measurement of the shipped code and is not.
  python - <<'PYEOF'
import os

import ray.data._internal.datasource_v2.readers.arrow_rs_parquet_file_reader as m
from ray.data._internal.datasource_v2.readers import file_reader as fr

MiB = 1024 * 1024
print("reader defaults actually in effect:")
print(f"  decode_budget      {m._ARROW_RS_DECODE_BUDGET_BYTES / MiB:.0f} MiB   (want 32)")
print(f"  fetch_window_mb    {m._ARROW_RS_FETCH_WINDOW_MB} MiB   (want 16)")
print(f"  prefetch_budget_mb {m._ARROW_RS_PREFETCH_BUDGET_MB}      (want -1 = derived)")
print(f"  k                  {m._ARROW_RS_K}       (want 1)")
# exp7 phase X is entirely about this knob: it is the threshold that decides
# whether a row group takes RgDecode::Hstack, so a non-16 value here silently
# changes which branch the "shipping" arm measures.
print(f"  column_fetch_mb    {m._ARROW_RS_COLUMN_FETCH_MB} MiB   (want 16)")
# The other half of 30297b9b7f. Not an env var: the arrow-rs reader overrides
# FileReader._num_fragment_read_threads() to 1 unless the env var was set
# EXPLICITLY, so the thing to confirm is that the override exists and that this
# shell has not set the var (which would hand the default back to PyArrow's 4).
print(
    f"  fragment threads   arrow_rs={'1 (override present)' if hasattr(m.ArrowRsParquetFileReader, '_num_fragment_read_threads') else 'OVERRIDE MISSING'}"
    f", base={fr._DEFAULT_NUM_THREADS}"
    f", env explicit={m._READ_FILES_NUM_THREADS_IS_EXPLICIT}"
)
leaked = sorted(k for k in os.environ if k.startswith("RAY_DATA_ARROW_RS_"))
# The probe sets these per-arm in a child process; anything set in THIS shell
# overrides every arm silently and invalidates the whole run.
if leaked:
    print(f"  !! exported in this shell, overriding every arm: {leaked}")
PYEOF
  echo
  for f in exp7_summary.txt exp8_summary.txt exp6_summary.txt; do
    [ -f "$OUT_DIR/$f" ] || continue
    echo "############################## $f"
    cat "$OUT_DIR/$f"
    echo
  done
  echo "====================== END PASTE ======================"
} | tee "$COMBINED"

echo
echo "also saved to $COMBINED"
