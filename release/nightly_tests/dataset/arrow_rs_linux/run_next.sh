#!/usr/bin/env bash
# The 2026-08-07 batch: three open questions, one command, one output to paste.
#
# Everything here is a re-measurement or a follow-up to something already run.
# The three things it settles, in the order they matter:
#
#   1. Is the S3 story still what we measured?  (exp7 phase S, re-run)
#      Every S3 number on record was taken with the OLD reader defaults: a 2 MiB
#      decode budget and a 4-thread fragment pool. Both changed in
#      30297b9b7f -- 32 MiB and one thread -- which is cherry-picked onto this
#      branch, so this run measures the code that would actually ship. The
#      headline 0.78x at the release read shape, and the "we lose 10-30% below
#      D ~= 700 MiB" caveat, are both stale until this finishes.
#
#   2. What is the fixed per-task cost, really?  (exp7 phase Z, new)
#      The fits say arrow-rs starts 117 MiB behind PyArrow before decoding a
#      single byte, and that constant is the whole reason we lose on small
#      tasks. 117 is an extrapolation off a curve that is visibly concave, which
#      is a weak way to learn a constant. Z measures it directly by reading a
#      ~250 KiB fixture, on BOTH transports, which also splits it into the part
#      that is the S3 client and the part that is the reader itself. Only one of
#      those two is cheap to fix.
#
#   3. Does prefetch_budget_mb bind?  (exp7 phase P, new)
#      This replaces a default I was about to ship and then retracted. Phase W
#      showed USS falling 892 -> 618 MiB as fetch_window_mb went 16 -> 128, and
#      I attributed it to the number of in-flight units -- until the crate
#      showed the unit count is 1 in all of those arms, because lineitem's row
#      groups are smaller than any window we tried. P pins the window, sweeps
#      the budget directly, and adds a fixture where the window CAN bind: one
#      big incompressible row group per file, which is the layout this whole
#      project targets and has never been measured inside Ray.
#
#   4. Is the 2.4x-per-block cost a copying concatenate?  (exp6 phase B, re-run)
#      Cheap, local, and it decides whether the decode budget should be derived
#      from target_max_block_size instead of pinned. Phase B has always run
#      fused, so the writer's 589 MiB sat on top of every arm.
#
# Runtime ~60-75 min: fixtures ~5, exp7 S/Z/P at REPEAT=3 ~55, exp6 B ~5.
# Must run in us-west-2, or phases S/Z/P measure the distance to Oregon.
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
STAGES="${STAGES:-fixtures exp7 exp6}"
has_stage() { case " $STAGES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

start=$(date +%s)

if has_stage fixtures; then
  echo "### building fixtures (idempotent -- a rebuild only happens if the"
  echo "### recorded geometry differs from what is asked for)"
  python "$HERE/make_fixtures.py" --bucket "$S3_ROOT"
fi

# Phases in dependency order, not importance order: S re-establishes the
# baseline the other two are read against, so if the box is misconfigured it
# fails in the first ten minutes rather than the last.
if has_stage exp7; then
  echo
  echo "### exp7 phases S, Z, P  (REPEAT=$REPEAT)"
  PHASES="S Z P" REPEAT="$REPEAT" bash "$HERE/exp7_s3_retention.sh" || {
    echo "!! exp7 exited nonzero -- its summary below covers whatever completed"
  }
fi

# One replicate is enough here: exp6 is local disk, where the run-to-run spread
# has been ~2%, not S3's 16%.
if has_stage exp6; then
  echo
  echo "### exp6 phase B  (local, unfused, threads=1)"
  PHASES="B" bash "$HERE/exp6_block_retention.sh" || {
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
  echo "  stages  : $STAGES"
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
