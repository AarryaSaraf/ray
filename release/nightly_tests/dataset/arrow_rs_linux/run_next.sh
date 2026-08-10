#!/usr/bin/env bash
# The 2026-08-10 batch: the two questions left, one command, one output to paste.
#
# What the previous batch settled, so this one does not re-ask it:
#
#   * Saturation survives S3. Marginal MiB of peak per MiB decoded falls
#     0.75 -> 0.48 -> 0.47 across a 17x rise in task size, and at the release read
#     shape (D = 1055 MiB per task) arrow-rs is 0.90x PyArrow's per-task memory at
#     wall parity. On the lone-big-row-group fixture it is 0.55-0.63x.
#   * There is NO fixed per-task S3 cost. At D ~= 0, arrow-rs on S3 is 114 MiB
#     against PyArrow's 117 -- 3 MiB BELOW. The "+117 MiB" carried for weeks was
#     the intercept of a straight line fitted to a concave curve, read off at an
#     x-value the data never visited.
#   * Neither S3 knob is a memory knob. prefetch_budget_mb spreads 1.26x on
#     lineitem (non-monotone, with the shipping default as the WORST arm) and
#     1.14x on bigrg, which is the noise floor. fetch_window_mb plans one unit on
#     every layout measured. Serial fragments are both lighter and no slower.
#   * The block-size cost is Ray's block layer, not ours: PyArrow pays 3.18 then
#     1.46 MiB of peak per MiB of block where we pay 2.46 then 1.59.
#
# That leaves the deficit as a single number with a single shape: a per-BYTE S3
# surcharge. Local slope 0.229, S3 slope 0.503, worth ~335 MiB at D = 1055. Both
# readers pay the same ~+16 MiB to cross to S3; only ours pays per byte.
#
#   1. Is the decoded channel the per-byte surcharge?  (exp7 phase D, new)
#      The last structural difference between the two paths. On S3 each stream
#      holds S3_CHANNEL_DEPTH = 2 decoded batches (crate lib.rs:919, 1709), each
#      sized by decode_budget -- ~64 MiB resident at the shipped 32 MiB that the
#      local path never allocates, because local drives the sync reader one batch
#      at a time straight into Python. exp6 phase B swept this knob on LOCAL disk
#      and found it flat across 64x, which is the correct answer where there is no
#      channel to fill. It has never been swept on S3. Prediction: USS moves by
#      ~2x the budget delta, which would explain ~60 of the 335 MiB -- a fifth,
#      not the whole thing. A flat result points at allocator retention instead,
#      whose instrument is MALLOC_ARENA_MAX rather than any knob.
#
#   2. What does the column-group path cost inside Ray?  (exp7 phase X, new)
#      RgDecode::Hstack has never executed under any measurement in this
#      directory: every arm reported col_group_rgs = 0, because lineitem and the
#      bigrg fixture are both 16 narrow columns and the branch needs one row
#      group's projected COMPRESSED bytes to exceed column_fetch_mb (16 MiB). It
#      is also the branch the 2026-08-05 root-cause blamed for the release
#      regressions. The mechanism is already confirmed on moto: it accumulates
#      every batch of every column group before emitting one row, so it retains
#      the whole decoded row group (51.4 MiB against a 39.9 MiB group) where the
#      row-window path retains one batch (25.7 MiB). This phase measures only the
#      CONSEQUENCE -- per-task USS against PyArrow -- on a new 64-column fixture.
#      Prediction: PARITY, not regression, because holding the whole decoded row
#      group is exactly what PyArrow's scanner does. The cf=0 arm is the same
#      fixture with only the branch changed, which is what makes it an experiment.
#
# Runtime ~35-45 min: fixtures ~10 (the `wide` layout is new and must be built and
# synced), exp7 D+X at REPEAT=3 ~25-35. Must run in us-west-2, or it measures the
# distance to Oregon rather than memory.
#
#     cd release/nightly_tests/dataset/arrow_rs_linux
#     source ~/ray/.venv/bin/activate
#     ./run_next.sh 2>&1 | tail -n 400
#
# Paste the block between the BEGIN/END PASTE markers. It is also written to
# out/next_summary.txt.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
check_env
check_s3

REPEAT="${REPEAT:-3}"
COMBINED="$OUT_DIR/next_summary.txt"
# Skip a stage that already ran, e.g. STAGES="exp7" after a fixture failure.
STAGES="${STAGES:-fixtures exp7}"
# Which exp7 phases this batch runs. S/W/T/Z/P are all answered; add them back
# explicitly (PHASES="S D X") only to re-establish a baseline on a new box.
EXP7_PHASES="${EXP7_PHASES:-D X}"
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

# exp6 is local disk, where the run-to-run spread has been ~2% rather than S3's
# 13%, so one replicate suffices. Nothing in this batch needs it -- phase B is
# answered -- so it is off by default and kept wired for the next local question.
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
  echo "  stages  : $STAGES  (exp7 phases: $EXP7_PHASES)"
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
# Phase X is entirely about this knob: it is the threshold that decides whether a
# row group takes RgDecode::Hstack, so a non-16 value here silently changes which
# branch the "shipping" arm measures.
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
  for f in exp7_summary.txt exp6_summary.txt; do
    [ -f "$OUT_DIR/$f" ] || continue
    echo "############################## $f"
    cat "$OUT_DIR/$f"
    echo
  done
  echo "====================== END PASTE ======================"
} | tee "$COMBINED"

echo
echo "also saved to $COMBINED"
