#!/usr/bin/env bash
# Experiment 7 -- does the exp6 story hold over S3, and is our S3 path heavier?
#
# Why this exists
# ---------------
# Every phase of exp6 -- the two-line model, the fragment thread pool, the
# writer -- ran on LOCAL DISK. exp6's own header says so: "S3 adds a fetch path
# that is not under test here." That was a reasonable scoping decision when the
# question was "what is a Ray read task holding", but three of the conclusions
# now being turned into shipped defaults were never checked on the transport the
# release suite actually uses. Every regression we are trying to explain came
# from an S3 read.
#
# And there is direct evidence the S3 path is not free. exp5 ran the identical
# read with NO Ray on both transports:
#
#     transport   pyarrow    arrow_rs    ratio
#     local       1013 MiB     23 MiB    0.02x
#     s3           961 MiB    174 MiB    0.18x
#
# PyArrow barely moves between transports. Ours goes 23 -> 174: a 151 MiB
# surcharge in code PyArrow does not have. It is still 5.5x better than PyArrow,
# but it eats most of the headroom, and it is entirely ours.
#
# The local path memory-maps the file and has no fetch machinery at all. The S3
# path has three things it does not:
#
#   1. fetch_window_mb (16)        -- compressed bytes of one row group pulled
#                                     into RAM before decode starts.
#   2. prefetch_budget_mb (-1)     -- total compressed bytes admitted ahead of
#                                     the decoder across all in-flight units;
#                                     -1 derives to 4 x max(fetch_window,
#                                     column_fetch) = 64 MiB.
#   3. the column-group planner    -- partition_columns_by_budget, which
#                                     RETAINS a whole decoded row group. exp2
#                                     found col_group_rgs = 0 on lineitem, so it
#                                     should not fire here; the profiler counter
#                                     is checked anyway.
#
# 16 + 64 + decode scratch is the right order of magnitude for 151 MiB, which
# would make the surcharge a KNOB rather than a leak. That is phase W.
#
# Phases
# ------
#   S -- exp6 phase F re-run over S3: unfused, threads=1, chunk swept. Does the
#        saturation survive the transport? This is the phase that decides
#        whether the 0.49x headline can be quoted at all.
#   W -- fetch_window x prefetch_budget, arrow-rs only, at the default chunk.
#        Is the 151 MiB surcharge the fetch machinery, and what does shrinking
#        it cost in wall time?
#   T -- threads 1 vs 4 over S3. The serial-fragment default is about to ship on
#        the strength of a local measurement. On S3 each fragment also carries a
#        fetch window, so serialising should save MORE memory -- but it also
#        removes overlap between one fragment's network wait and another's
#        decode, which local disk cannot show. If serial costs real wall time
#        here, the default has to be transport-aware.
#   Z -- MEASURE the fixed per-task cost instead of extrapolating it. Both phase
#        S fits have an intercept -- pyarrow ~189 MiB, arrow-rs ~306 -- and that
#        +117 MiB gap is the whole reason arrow-rs loses below D ~= 700 MiB. Z
#        reads a ~250 KiB fixture, so D ~= 0 and per-task USS IS the intercept.
#        Run on both transports, because the local path builds no HTTP client,
#        no connection pool and no tokio runtime: (s3 - local) is the
#        transport's share of the constant and the remainder is the reader's.
#   P -- what phase W thought it was measuring. See the S_WINDOW note below: on
#        lineitem the fetch window CANNOT bind, so W's 892 -> 618 MiB was the
#        derived prefetch budget moving, not the window. P sweeps
#        prefetch_budget_mb DIRECTLY at a pinned window, over two layouts:
#        lineitem (window provably inert -> isolates the budget) and the bigrg
#        fixture (window binds: ~20 units at 16 MiB against ~2 at 128). bigrg is
#        also the lone-big-row-group shape this whole project targets, and no
#        knob measurement has ever been taken on it inside Ray.
#   D -- the decode budget, ON S3. Phases W, P, T and Z between them killed every
#        other candidate: the window plans one unit, the prefetch budget is flat
#        and non-monotone, serial threads are strictly better, and the fixed cost
#        is 3 MiB BELOW PyArrow's. What is left is a per-BYTE surcharge (slope
#        0.503 on S3 against 0.229 local, ~335 MiB at the release shape), and the
#        remaining structural difference between the paths is the decoded channel:
#        depth 2 on S3, absent locally. exp6 phase B swept this knob on local disk
#        and found it flat, which is the right answer where there is no channel.
#   X -- the COLUMN-group path (RgDecode::Hstack), which no arm of exp6 or exp7 has
#        ever executed -- every one reported col_group_rgs = 0, because lineitem
#        and bigrg are both 16 narrow columns. That branch accumulates every batch
#        of every column group before emitting one row, so it cannot saturate; it
#        holds the whole decoded row group, which is exactly what PyArrow does.
#        Needs the `wide` fixture. Prediction: parity, not regression.
#
# Phases Z, P and X need `make_fixtures.py` to have run; run_next.sh does that.
#
# Runtime: ~35-45 min for the 20 arms of S/W/T; add ~10 min for Z, ~20 for P,
# ~15 for D and ~10 for X at REPEAT=3. Run in us-west-2 or the numbers are
# latency, not memory.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
check_env
check_s3

SF="${SF:-10}"
S3_DATA="${S3_DATA:-$S3_ROOT/tpch/sf$SF/lineitem}"
RESULTS="$OUT_DIR/exp7"
# Must match the local arms exp6 measured, or the comparison is between two
# different amounts of data rather than between two transports. exp5 copies the
# first MAX_FILES files of this same prefix to local disk.
MAX_FILES="${MAX_FILES:-4}"
TOTAL_DECODED_MIB="${TOTAL_DECODED_MIB:-4219}"

# Fixtures make_fixtures.py builds. tiny = ~250 KiB/file (phase Z); bigrg = one
# incompressible row group per file (phase P). Both exist locally AND on S3 so
# phase Z can difference the transports over identical bytes.
S3_TINY="${S3_TINY:-$S3_ROOT/fixtures/tiny}"
S3_BIGRG="${S3_BIGRG:-$S3_ROOT/fixtures/bigrg}"
LOCAL_FIXTURES="${LOCAL_FIXTURES:-$HOME/arrow_rs_local/fixtures}"

FIXED_BLOCK="${FIXED_BLOCK:-128}"
S_CHUNKS="${S_CHUNKS:-16 64 256 0}"
# fetch_window_mb for the phase S grid.
#
# 0 = the shipping default (16), which is what the first run used. The second
# run used 128 because phase W had found 16 to be the worst setting -- and that
# reading is now RETRACTED. `window_rows_for` (crate lib.rs:955) converts the
# MiB into a row count by dividing by compressed bytes/row, and `plan_s3_units`
# clamps to the row group's length; lineitem is ~43 compressed B/row in
# 122,428-row groups, so a 16 MiB window asks for ~390,000 rows and clamps to
# the whole group. 16, 64 and 128 all plan the identical single unit. Whatever
# moved USS in phase W, it was not the window.
#
# So 0 is the right default here again: it is what ships. Phase P is where the
# knob question moved, on a fixture where the window can actually bind.
S_WINDOW="${S_WINDOW:-0}"
# Replicates per arm. Default 1 for a first look; use 3 before quoting anything.
REPEAT="${REPEAT:-1}"
# 0 = leave the shipping default (16). 4 and 64 bracket it; 128 is deliberately
# past the point where the window stops bounding anything, as the control.
WINDOWS="${WINDOWS:-4 16 64 128}"
# Paired with the window sweep: -1 means "derive 4x the window", so the two move
# together. A fixed 16 holds the total constant while the per-unit window grows,
# which separates "how much per unit" from "how many units in flight".
BUDGETS_MB="${BUDGETS_MB:-0 16}"
T_THREADS="${T_THREADS:-1 4}"
# Phase P. The window is PINNED at the shipping 16 so the budget is the only
# thing moving; the sweep brackets the derived default (4 x max(16,16) = 64) by
# 4x in each direction. 1024 is well past the whole bigrg row group, so it is
# the "budget removed" control: if USS is flat from 64 to 1024 the budget is not
# binding either and the S3 surcharge is somewhere neither knob reaches.
P_WINDOW="${P_WINDOW:-16}"
P_BUDGETS="${P_BUDGETS:-16 64 256 1024}"
# Which layouts phase P runs over. lineitem is the control where the window is
# provably inert; bigrg is the target layout where it binds.
P_LAYOUTS="${P_LAYOUTS:-lineitem bigrg}"
# Phase D. Brackets the shipped 32 MiB by 16x in each direction. The S3 driver
# holds S3_CHANNEL_DEPTH=2 decoded batches per stream, so the prediction is a
# ~2 x delta-budget swing -- 2 MiB should sit ~60 MiB under 32 MiB if the channel
# is the mechanism. 512 is past the point where one batch is a whole block, which
# is the "batching removed" control.
D_BUDGETS="${D_BUDGETS:-2 8 32 128 512}"
# Phase X. 4 files x 4 row groups x 64 MiB decoded = 256 MiB per file, and Ray's
# 1 GiB chunker default gives one read task per file, so D = 256 MiB per task.
S3_WIDE="${S3_WIDE:-$S3_ROOT/fixtures/wide}"
X_FILES="${X_FILES:-4}"
X_DECODED_MIB="${X_DECODED_MIB:-256}"
PHASES="${PHASES:-S W T}"

mkdir -p "$RESULTS"
has_phase() { case " $PHASES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
for p in $PHASES; do rm -f "$RESULTS/${p}_"*.json; done

# Stage the input once. Server-side copy, no-op on re-runs. This stages the
# WHOLE sf10 prefix (~11 files); the per-arm --max-files below is what makes the
# S3 arms read the same 4 files exp5 copied to local disk.
python "$HERE/stage_data.py" --dst "$S3_DATA" --sf "$SF" \
  --region "$AWS_DEFAULT_REGION"

# One process per arm -- Ray reuses workers and MemoryProfiler reads
# whole-process private memory, so arm 2 would inherit arm 1's high-water mark.
#
# Every arm here is UNFUSED (no --write-to). exp6 phase F showed the writer held
# 589 of 1032 MiB at the default chunk; measuring read+write together on S3 too
# would just re-measure the writer through a different transport. The write path
# is its own workstream.
#
# Profiling is on for every arm: col_group_rgs and s3_rg-retained_bytes are the
# only way to tell "the window is working" from "the column-group branch fired
# and retained the whole row group", and those two look identical in USS.
#
# REPEAT is not optional rigour here, it is a correction. The 2026-08-07 run
# happened to measure the SAME configuration (arrow_rs, default chunk, shipping
# knobs, threads=1) in three different phases and got 802 / 837 / 892 MiB at
# 11.5 / 7.1 / 7.0 s -- 11% spread on memory and 64% on wall. Phase S's ratios
# (1.33 / 1.17 / 1.11) are not separable from each other at that noise level, and
# a single arm cannot be quoted. The summary reports the median and the spread so
# the noise is visible in the table instead of inferred across phases.
# SRC / MAXF override the input for one call, for the phases that do not read
# lineitem. They are per-call env, not globals, so a phase cannot leak its
# fixture into the next one:  SRC="$S3_TINY" MAXF=4 run_probe ...
run_probe() {  # tag, then probe args
  local tag="$1"; shift
  local src="${SRC:-$S3_DATA}"
  local maxf="${MAXF:-$MAX_FILES}"
  local rep
  for rep in $(seq 1 "${REPEAT:-1}"); do
    local rtag="$tag"
    [ "${REPEAT:-1}" -gt 1 ] && rtag="${tag}_r${rep}"
    echo "=== $rtag"
    env RAY_DATA_ARROW_RS_PROFILE=1 \
        RAY_DATA_ARROW_RS_PROFILE_DIR="$RESULTS/prof/$rtag" \
      python "$HERE/block_retention_probe.py" \
        --source "$src" --max-files "$maxf" \
        --out "$RESULTS/$rtag.json" "$@" \
        > "$RESULTS/$rtag.log" 2>&1 \
      || echo "  FAILED -- see $RESULTS/$rtag.log"
    grep -h "^WARNING:" "$RESULTS/$rtag.log" 2>/dev/null || true
  done
}

# Fail early and by name if a fixture phase was asked for without the fixture.
# Otherwise the arms run, read nothing, and the summary shows a stats_error per
# arm -- ten minutes to learn one listing would have told us.
need_fixture() {  # human name, s3 uri
  python - "$2" <<'PYEOF' || {
import sys
from pyarrow.fs import FileSelector, FileSystem, FileType
fs, root = FileSystem.from_uri(sys.argv[1])
infos = fs.get_file_info(FileSelector(root, recursive=True, allow_not_found=True))
n = sum(1 for i in infos if i.type == FileType.File and i.path.endswith(".parquet"))
sys.exit(0 if n else 1)
PYEOF
    echo "FATAL: no .parquet under $2 -- run make_fixtures.py (or run_next.sh,"
    echo "       which does it for you) before PHASES=$1"
    exit 1
  }
}

# Phase S -- the exp6 phase F grid, over S3.
#
# Local, unfused, at threads=1, the answer was:
#     D (MiB/task)     62    211    527   1055
#     pyarrow         209    347    471    901
#     arrow_rs        185    326    422    443   <- saturating
#
#   arrow_rs saturates here too      -> the headline holds on the transport the
#       release suite uses, and 0.49x can go in a PR description.
#   arrow_rs tracks D linearly here  -> the saturation is a local-path property
#       (mmap, no fetch buffers) and does NOT transfer. Everything we are about
#       to ship is then justified only for local reads, and the S3 fetch path
#       becomes the main event rather than a follow-up.
#   BOTH readers shift up by a constant -> transport overhead, cancels in the
#       ratio, story unchanged.
#
# ANSWERED 2026-08-07, and it was the bad branch: increments +133 / +115 / +270,
# i.e. flat then accelerating. Saturation does NOT transfer. arrow-rs came out
# 1.33 / 1.17 / 1.11 / 0.96x -- WORSE than PyArrow at every task size but the
# largest, which is the release regression reproduced on one box.
#
# But phase W then showed that grid was run at the WORST available S3 setting:
# fetch_window_mb=16 costs 892 MiB at the default chunk where 128 costs 618, at
# wall parity. S_WINDOW re-runs the grid at a chosen window so the transport
# question gets asked at a setting we would actually ship. 0 = leave the default
# (which is what produced the numbers above).
if has_phase S; then
for chunk in $S_CHUNKS; do
  for reader in pyarrow arrow_rs; do
    chunk_args=()
    [ "$chunk" != 0 ] && chunk_args=(--chunk-mib "$chunk")
    win_args=()
    # pyarrow ignores the knob, but passing it only to one arm would put the two
    # arms on different env, so it goes to both or neither.
    [ "${S_WINDOW:-0}" != 0 ] && win_args=(--fetch-window-mb "$S_WINDOW")
    run_probe "S_${reader}_chunk${chunk}" \
      --reader "$reader" --block-mib "$FIXED_BLOCK" --threads 1 \
      "${chunk_args[@]}" "${win_args[@]}"
  done
done
fi

# Phase W -- is the 151 MiB S3 surcharge the fetch machinery?
#
# Run at the DEFAULT chunk, where phase S puts arrow-rs at its ceiling: that is
# where a knob worth ~100 MiB is a visible fraction, and it is the shape the
# release suite runs (one task per file).
#
#   USS falls with the window   -> the surcharge is bounded prefetch, it is a
#       knob, and the default should be set from a memory target rather than
#       from throughput. Read the wall column before celebrating: a window of 4
#       MiB that halves memory and doubles wall time is not a win, it is the
#       same trade PyArrow already makes.
#   USS flat across 32x         -> the surcharge is not the window. Next suspect
#       is the object_store client's own connection pool and per-request
#       buffers, which no knob here reaches.
#   budget=16 tracks window=16  -> the total, not the per-unit window, is what
#       binds; then prefetch_budget_mb is the knob to default differently and
#       fetch_window_mb can stay where it is.
if has_phase W; then
for budget in $BUDGETS_MB; do
  for window in $WINDOWS; do
    budget_args=()
    [ "$budget" != 0 ] && budget_args=(--prefetch-budget-mb "$budget")
    run_probe "W_arrow_rs_win${window}_bud${budget}" \
      --reader arrow_rs --block-mib "$FIXED_BLOCK" --threads 1 \
      --fetch-window-mb "$window" "${budget_args[@]}"
  done
done
fi

# Phase T -- does the serial-fragment default survive network latency?
#
# exp6 phase D/E is about to become a shipped default: one fragment thread on
# the arrow-rs path, worth ~96 MiB at 0.95-1.02x wall. Both of those numbers are
# from local disk, where a fragment read is a memcpy and there is nothing to
# overlap. On S3 the trade is different in both directions:
#
#   memory: each in-flight fragment now also holds a fetch window, so serial
#           should save MORE than 96 MiB.
#   time:   four threads overlap one fragment's network wait with another's
#           decode. Serial cannot, and S3 round trips are milliseconds, not
#           microseconds. This is the risk to the default.
#
# The pyarrow arm is the control -- the pool is shared, so if both readers lose
# the same wall time, it is Ray's overlap and not ours.
#
#   serial still ~1.0x wall  -> ship the default unconditionally.
#   serial much slower on S3 -> the default must be transport-aware, or the
#       right fix is per-fragment retention (a smaller window per thread) rather
#       than fewer threads.
if has_phase T; then
for threads in $T_THREADS; do
  for reader in pyarrow arrow_rs; do
    run_probe "T_${reader}_thr${threads}" \
      --reader "$reader" --block-mib "$FIXED_BLOCK" --threads "$threads"
  done
done
fi

# Phase Z -- measure the fixed per-task cost instead of extrapolating it.
#
# Phase S fits two lines through four points:
#     pyarrow    USS ~= 189 + 0.614 x D
#     arrow_rs   USS ~= 306 + 0.361 x D
# The SLOPE is the number that matters for OOMs -- it is the fraction of what a
# task decodes that the task still holds at its peak -- and ours is 1.7x better.
# The INTERCEPT is what a task costs before it decodes anything, and ours is
# +117 MiB worse. That constant is the entire reason arrow-rs loses to PyArrow
# below D ~= 700 MiB, which is where most read tasks live.
#
# Two problems with the 117. It is an extrapolation off a visibly concave curve,
# so the fitted line does not have to pass anywhere near D = 0. And it is one
# number for what should be at least two costs. Phase Z fixes both by reading a
# fixture of a few hundred KiB: D ~= 0, so per-task USS IS the constant, no fit
# involved. Running it on both transports splits it:
#
#     arrow_rs local           the reader's own floor -- crate, FFI, allocator
#                              baseline, Ray worker. The local path memory-maps
#                              and builds no HTTP client at all.
#     arrow_rs s3 - local      the transport's floor -- object_store's client,
#                              its connection pool, the shared tokio runtime's
#                              thread stacks. NO KNOB IN THIS SCRIPT REACHES IT.
#     arrow_rs - pyarrow       the number that has to shrink.
#
# The 44 MiB the fragment thread pool used to contribute is already gone: this
# runs on the cherry-picked shipping default of one fragment thread.
#
#   s3 - local is most of the gap  -> the client is the target. One shared
#       client across fragments, or a capped connection pool, and the 1.31x /
#       1.16x / 1.12x rows at small D move as a block.
#   local alone is most of it      -> the transport is innocent and the floor is
#       the crate or the allocator, which is a much harder fix and probably
#       means quoting the crossover honestly instead.
if has_phase Z; then
need_fixture Z "$S3_TINY"
for reader in pyarrow arrow_rs; do
  SRC="$S3_TINY" MAXF=4 run_probe "Z_${reader}_s3" \
    --reader "$reader" --block-mib "$FIXED_BLOCK" --threads 1
  SRC="$LOCAL_FIXTURES/tiny" MAXF=4 run_probe "Z_${reader}_local" \
    --reader "$reader" --block-mib "$FIXED_BLOCK" --threads 1
done
fi

# Phase P -- the experiment that replaced a default I was about to ship.
#
# Phase W measured USS falling from 892 to 618 MiB as fetch_window_mb went 16 ->
# 128 and I read that as "fewer, larger in-flight units cost less". Then reading
# the crate killed it. `window_rows_for` (lib.rs:955) turns the MiB into a ROW
# COUNT by dividing by the row group's compressed bytes per row, and
# `plan_s3_units` clamps each window to the group's length. lineitem is ~43
# compressed B/row in 122,428-row groups, so a 16 MiB window asks for ~390,000
# rows -- three times the group -- and clamps. 16, 64 and 128 MiB plan the SAME
# SINGLE UNIT. The unit count I claimed as the mechanism never changed.
#
# What did change is prefetch_budget_mb, because -1 derives it as
# 4 x max(fetch_window_mb, column_fetch_mb): 16 -> 64 MiB, 128 -> 512 MiB. And
# it did not move monotonically with USS (window 16/64/256/512 gave 665/890/703/
# 618), which is not what a knob that binds looks like. So phase P stops
# inferring and sweeps the budget directly, at a pinned window, replicated.
#
# Two layouts, because one of them cannot answer the question alone:
#
#   lineitem  the control. The window is provably inert here, so anything the
#             sweep moves is the budget and only the budget.
#   bigrg     16 float64 columns of uniform random doubles -- 128 B/row decoded
#             and incompressible, measured at 161 compressed B/row, 3.7x
#             lineitem's 43 -- in ONE row group per file. Now a 16 MiB window
#             really does plan ~20 units where a 128 MiB window plans ~2, and
#             the row group is 256 MiB rather than 20. This is also the layout the
#             entire project targets: the lone big row group where PyArrow has
#             to materialize the whole decoded group and we are supposed not to.
#             Every measurement of it so far has been in the standalone harness,
#             outside Ray, with no read task and no output blocks around it.
#
# The pyarrow arm on bigrg is not decoration -- it is the first in-Ray baseline
# for the shape the project exists to beat.
#
#   USS tracks the budget on both   -> prefetch_budget_mb is the S3 memory knob,
#       set the default from a memory target, and phase W's effect is explained.
#   USS tracks it only on bigrg     -> it binds only when the window fragments a
#       row group, so the default has to be layout-aware, or the window should
#       be derived from the row group rather than fixed in MiB.
#   flat on both                    -> neither knob is the S3 surcharge; phase
#       Z's client/runtime split is the only remaining lead.
if has_phase P; then
need_fixture P "$S3_BIGRG"
for layout in $P_LAYOUTS; do
  case "$layout" in
    lineitem) psrc="$S3_DATA"; pmaxf="$MAX_FILES" ;;
    bigrg)    psrc="$S3_BIGRG"; pmaxf=2 ;;
    *) echo "FATAL: unknown P_LAYOUTS entry '$layout'"; exit 1 ;;
  esac
  # Control first: it is the cheapest arm and a failure here means the fixture
  # itself is wrong, which is worth learning before four budget arms run.
  SRC="$psrc" MAXF="$pmaxf" run_probe "P_pyarrow_${layout}_bud0" \
    --reader pyarrow --block-mib "$FIXED_BLOCK" --threads 1
  for budget in $P_BUDGETS; do
    SRC="$psrc" MAXF="$pmaxf" run_probe "P_arrowrs_${layout}_bud${budget}" \
      --reader arrow_rs --block-mib "$FIXED_BLOCK" --threads 1 \
      --fetch-window-mb "$P_WINDOW" --prefetch-budget-mb "$budget"
  done
done
fi

# Phase D -- the one cell of the knob x transport matrix nobody filled.
#
# After phases W, P and T, three candidates for the S3 surcharge are dead: the
# fetch window (it plans one unit on every layout measured), prefetch_budget_mb
# (phase P: 1.26x on lineitem, non-monotone, with the shipping default as the
# WORST arm; 1.14x on bigrg, which is the noise floor), and the fragment pool
# (phase T: serial is both lighter and no slower). What is left is not a
# surcharge on a constant -- phase Z showed the constant is 3 MiB BELOW PyArrow's
# -- it is a surcharge per byte decoded:
#
#     arrow_rs local   USS ~= 238 + 0.229 x D        marginal 0.95 / 0.30 / 0.04
#     arrow_rs s3      USS ~= 254 + 0.503 x D        marginal 0.75 / 0.48 / 0.47
#     pyarrow  local   USS ~= 169 + 0.675 x D
#     pyarrow  s3      USS ~= 186 + 0.629 x D        <- slope FALLS on S3
#
# Both readers pay the same ~+16 MiB to cross to S3. Only ours pays per byte:
# +0.274 per MiB decoded, worth ~290 MiB at the release shape (D = 1055). Locally
# the marginal cost decays to 0.04 -- extra data is nearly free -- and on S3 it
# plateaus at 0.47 and stays there. That plateau IS the gap.
#
# The remaining structural difference between the paths is the decoded channel.
# The S3 driver gives each stream `mpsc::channel(S3_CHANNEL_DEPTH)` -- depth 2
# (crate lib.rs:919, 1709) -- and sizes each batch with
# byte_budget_rows(decode_budget). At the shipped 32 MiB that is ~64 MiB of
# DECODED batches resident that the local path never allocates: local drives the
# sync reader, one batch at a time, straight into Python.
#
# exp6 phase B swept this knob on LOCAL disk, where the channel does not exist,
# and correctly found it flat (443/489/462/463 MiB across 64x). It has never been
# swept on S3, where it is multiplied by the channel depth.
#
#   USS falls ~2 x delta-budget  -> confirmed, and the fix is either a shallower
#       channel or a budget default that accounts for the depth. Predicts ~60 MiB
#       of the 335, so a partial explanation even if it lands exactly.
#   USS flat as it is locally    -> the channel is not it either, and the next
#       suspect is allocator retention on the async fetch path (many transient
#       HTTP body buffers), whose discriminator is MALLOC_ARENA_MAX, not a knob.
#   USS RISES as the budget falls -> more, smaller batches cost more round trips
#       through the channel; then the knob is a throughput/memory trade rather
#       than free, and the 32 MiB default is already the right side of it.
#
# The pyarrow arm is the control: it ignores the knob entirely, so any movement in
# it is the noise floor for this grid, measured rather than assumed.
if has_phase D; then
run_probe "D_pyarrow" --reader pyarrow --block-mib "$FIXED_BLOCK" --threads 1
for budget in $D_BUDGETS; do
  run_probe "D_arrowrs_bud${budget}" \
    --reader arrow_rs --block-mib "$FIXED_BLOCK" --threads 1 \
    --decode-budget-mib "$budget"
done
fi

# Phase X -- the column-group path, which no measurement here has ever executed.
#
# `plan_s3_units` sends a row group down RgDecode::Hstack instead of
# RgDecode::Windows when its projected COMPRESSED bytes exceed column_fetch_mb
# (16 MiB). Every arm of exp6 and exp7 so far has reported col_group_rgs = 0:
# lineitem is 16 columns at ~5.3 MB compressed per group, an order of magnitude
# under the budget, and the bigrg fixture is 16 columns too. So the one branch the
# 2026-08-05 root-cause blamed for the release regressions has never run under a
# per-task USS measurement inside Ray.
#
# It matters because that branch cannot saturate, by construction. The Hstack arm
# decodes column group 0 to completion, then group 1, ... accumulating every batch
# of every group in `group_batches` before it emits the first stitched row (crate
# lib.rs ~1645). Peak therefore tracks the whole DECODED row group and
# decode_budget_bytes bounds nothing at all.
#
# Verified on moto before this phase was written, 64 columns x 65,536 rows in one
# row group, same data both ways:
#
#     column_fetch_mb=16   col_group_rgs=1, 3 groups   retained 51.4 MiB  (1.29x
#                                                      the 39.9 MiB row group)
#     column_fetch_mb=0    col_group_rgs=0, 3 windows  retained 25.7 MiB  (one
#                                                      batch, bounded by the budget)
#
# So the mechanism is settled and this phase is only for the consequence: what it
# costs in per-task USS against PyArrow, which is the number a PR can quote.
#
# The prediction is PARITY, NOT REGRESSION. Holding the whole decoded row group is
# exactly what PyArrow's scanner does, so on wide schemas arrow-rs should stop
# being memory-advantaged rather than become worse. The standalone Linux/S3 run
# agrees (5000 columns, cf=16 gave 4.30 GB against PyArrow's 6.78) -- but
# standalone drops its output where a Ray read task retains it, which is the whole
# reason to re-ask the question here.
#
# Four arms, and the cf0 arm is the one that makes this an experiment rather than
# an observation: it is the same fixture with only the branch changed.
#
#   arrow_rs cf16 ~= pyarrow          -> parity confirmed. The honest scope of the
#       memory claim is "narrow-to-medium schemas", said out loud, and the
#       streaming-hstack fix becomes a real improvement over PyArrow rather than a
#       regression repair.
#   arrow_rs cf16 >> pyarrow          -> worse than PyArrow on the layout the
#       release suite regressed on. That is the release regression, found.
#   cf0 much lighter than cf16        -> the accumulation is the cost and the fix
#       is to interleave the groups (pull batch i from all N streams, stitch, emit,
#       drop) instead of materialising them. Peak goes from the row group to one
#       full-width batch.
#   cf0 no lighter                    -> the cost is the fetch, not the hstack,
#       and column_fetch_mb is doing its job.
if has_phase X; then
need_fixture X "$S3_WIDE"
SRC="$S3_WIDE" MAXF="$X_FILES" run_probe "X_pyarrow" \
  --reader pyarrow --block-mib "$FIXED_BLOCK" --threads 1
SRC="$S3_WIDE" MAXF="$X_FILES" run_probe "X_arrowrs_cf16" \
  --reader arrow_rs --block-mib "$FIXED_BLOCK" --threads 1
# -1 encodes the crate's 0 ("disable column grouping"); the probe's own 0 means
# "leave the default", so the two cannot share an encoding.
SRC="$S3_WIDE" MAXF="$X_FILES" run_probe "X_arrowrs_cf0" \
  --reader arrow_rs --block-mib "$FIXED_BLOCK" --threads 1 \
  --column-fetch-mb -1
# Does the decode budget reach this path at all? The mechanism says no -- the
# accumulation is over every batch regardless of how big each one is. If USS moves
# here, the mechanism is wrong and worth re-reading before anything is fixed.
SRC="$S3_WIDE" MAXF="$X_FILES" run_probe "X_arrowrs_cf16_bud2" \
  --reader arrow_rs --block-mib "$FIXED_BLOCK" --threads 1 \
  --decode-budget-mib 2
fi

RESULTS="$RESULTS" TOTAL_DECODED_MIB="$TOTAL_DECODED_MIB" \
  P_WINDOW="$P_WINDOW" P_LAYOUTS="$P_LAYOUTS" \
  D_BUDGETS="$D_BUDGETS" X_DECODED_MIB="$X_DECODED_MIB" \
  python - <<'PYEOF' | tee "$OUT_DIR/exp7_summary.txt"
import glob
import json
import os
import re

MiB = 1024 * 1024
TOTAL_DECODED_MIB = float(os.environ["TOTAL_DECODED_MIB"])
RESULTS = os.environ["RESULTS"]

# exp6 phase F, local disk, unfused, threads=1. Printed alongside every S row so
# the transport comparison is on one line instead of across two summary files.
LOCAL_F = {62: (209, 185), 211: (347, 326), 527: (471, 422), 1055: (901, 443)}


def nearest_local(d):
    if not d:
        return None
    key = min(LOCAL_F, key=lambda k: abs(k - d))
    return LOCAL_F[key] if abs(key - d) / key < 0.25 else None


raw = []
for path in sorted(glob.glob(os.path.join(RESULTS, "*.json"))):
    with open(path) as handle:
        r = json.load(handle)
    r["tag"] = os.path.basename(path)[: -len(".json")]
    raw.append(r)

if not raw:
    raise SystemExit(f"no results in {RESULTS}")


def _median(vals):
    s = sorted(vals)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


def collapse(records):
    """Median over replicates of the same configuration, carrying the spread.

    Not cosmetic. The first run measured the same config in three phases and got
    802 / 837 / 892 MiB at 11.5 / 7.1 / 7.0 s: 11% on memory, 64% on wall. Any
    table that shows one number per arm invites reading a 1.11x as different from
    a 1.17x when the instrument cannot tell them apart. The spread column is the
    reader's warning that it cannot.
    """
    keys = (
        "reader",
        "block_mib",
        "chunk_mib",
        "threads",
        "fetch_window_mb",
        "prefetch_budget_mb",
        "decode_budget_mib",
        # Phase X's cf16 and cf0 arms differ in NOTHING else -- same fixture, same
        # block, same threads -- so leaving this out would median the two sides of
        # the A/B together and report their average as both.
        "column_fetch_mb",
        # Phases Z and P read different inputs at IDENTICAL knob settings --
        # Z_arrow_rs_s3 and Z_arrow_rs_local differ in nothing else. Without the
        # source in the key they collapse into one group and the median silently
        # averages the two transports whose difference is the entire finding.
        "source",
        "num_files",
    )
    groups = {}
    for r in records:
        # Strip the _rN replicate suffix so replicates land in one group and the
        # phase letter still distinguishes grids that share knob settings.
        base = re.sub(r"_r\d+$", "", r["tag"])
        groups.setdefault(
            (base.split("_")[0], tuple(r.get(k) for k in keys)), []
        ).append(r)
    out = []
    for (_, _), reps in groups.items():
        good = [r for r in reps if not r.get("stats_error")]
        merged = dict((good or reps)[0])
        merged["tag"] = re.sub(r"_r\d+$", "", merged["tag"])
        merged["n_reps"] = len(reps)
        if good:
            uss = [r["avg_max_uss_per_task"] for r in good]
            merged["avg_max_uss_per_task"] = _median(uss)
            merged["wall_s"] = _median([r["wall_s"] for r in good])
            merged["uss_spread"] = (max(uss) / min(uss)) if min(uss) else 0
        out.append(merged)
    return out


rows = collapse(raw)
spreads = [r["uss_spread"] for r in rows if r.get("uss_spread", 0) and r["n_reps"] > 1]
if spreads:
    print(
        f"replicates: {max(r['n_reps'] for r in rows)} per arm, worst within-arm "
        f"USS spread {max(spreads):.2f}x -- differences smaller than this are noise\n"
    )
else:
    print(
        "replicates: 1 per arm. The 2026-08-07 run showed ~1.11x run-to-run spread\n"
        "on identical configs, so treat any ratio difference under ~15% as noise.\n"
        "Re-run with REPEAT=3 before quoting a number.\n"
    )

# Every fit below divides TOTAL_DECODED_MIB by the task count, so a run that
# read a different number of files than that constant assumes silently produces
# a wrong x-axis and therefore a wrong slope. Cheap to check, expensive to miss.
#
# Scoped to phase S. Phases Z and P deliberately read OTHER fixtures with other
# file counts -- that is the point of them -- so a global check here would fire
# on every run and train us to ignore it.
counts = {
    r.get("num_files")
    for r in rows
    if r.get("num_files") and r["tag"].startswith("S_")
}
if len(counts) > 1:
    print(f"!! phase S arms read differing file counts {sorted(counts)}\n")
elif counts:
    print(
        f"input: {counts.pop()} files, assumed {TOTAL_DECODED_MIB:,.0f} MiB decoded "
        "total (set TOTAL_DECODED_MIB to override)\n"
    )

bad = [r for r in rows if r.get("stats_error")]
if bad:
    print("!! arms with no usable stats (excluded from every fit below):")
    for r in bad:
        print(f"   {r['tag']}: {r['stats_error']}")
    print()


def uss(r):
    return (r.get("avg_max_uss_per_task") or 0) / MiB


def decoded_per_task(r):
    n = r.get("uss_num_samples") or 0
    return TOTAL_DECODED_MIB / n if n else 0


def fit(points):
    """Least-squares USS = intercept + slope * D.

    The slope is the fraction of what a task decodes that it still holds at
    peak. Report it, but read the raw increments too: a saturating curve fits a
    line badly and the line will understate how good saturation is.
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


# --- phase S: does saturation survive S3? ------------------------------------
s_rows = [r for r in rows if r["tag"].startswith("S_") and not r.get("stats_error")]
if s_rows:
    by_chunk = {}
    for r in s_rows:
        by_chunk.setdefault(r.get("chunk_mib") or 0, {})[r["reader"]] = r
    head = (
        f"{'PHASE S s3 unfused':<20}{'D MiB':>8}{'pyarrow':>10}{'arrow_rs':>10}"
        # wall P as well as wall R: "less memory at wall parity" is the whole
        # claim, and the first version of this table printed only arrow-rs's wall,
        # which cannot support the second half of it.
        f"{'ratio':>8}{'local R':>9}{'s3-local':>10}{'wall P':>8}{'wall R':>8}"
    )
    print(head)
    print("-" * len(head))
    pts = {"pyarrow": [], "arrow_rs": []}
    for chunk in sorted(by_chunk, key=lambda c: (c == 0, c)):
        arms = by_chunk[chunk]
        p, a = arms.get("pyarrow"), arms.get("arrow_rs")
        if not (p and a):
            continue
        d = decoded_per_task(a)
        pu, au = uss(p), uss(a)
        pts["pyarrow"].append((decoded_per_task(p), pu))
        pts["arrow_rs"].append((d, au))
        loc = nearest_local(d)
        loc_r = f"{loc[1]:.0f}" if loc else "-"
        delta = f"{au - loc[1]:+.0f}M" if loc else "-"
        label = "chunk=default(1G)" if not chunk else f"chunk={chunk}M"
        print(
            f"{label:<20}{d:>8,.0f}{pu:>7.0f}MiB{au:>7.0f}MiB"
            f"{(au / pu if pu else 0):>7.2f}x{loc_r:>9}{delta:>10}"
            f"{p['wall_s']:>8.1f}{a['wall_s']:>8.1f}"
        )
    for reader in ("pyarrow", "arrow_rs"):
        intercept, slope = fit(pts[reader])
        if slope is not None:
            print(f"{'':<20}  {reader:<9} s3 USS ~= {intercept:.0f} + {slope:.3f} x D")
    # Saturation has to be judged on MARGINAL cost -- MiB of peak per additional
    # MiB decoded -- not on the raw USS increments.
    #
    # Two bugs lived on this line. The first asserted saturation unconditionally
    # whenever there were more than two points, and printed "holds on S3" over
    # increments of +133 / +115 / +270. The second (this fix) tested whether the
    # USS increments shrink, which is not the same claim: the chunk sweep does not
    # step D evenly. On 2026-08-07 D went 62 -> 211 -> 527 -> 1055, so the D steps
    # were +149 / +316 / +528 -- doubling. Constant USS increments over doubling D
    # increments IS saturating, but constant USS increments over EVEN D steps would
    # be plain linear, and the old test called both "saturating".
    ordered = sorted(pts["arrow_rs"])
    if len(ordered) > 2:
        deltas = [b[1] - a[1] for a, b in zip(ordered, ordered[1:])]
        marginal = [
            (b[1] - a[1]) / (b[0] - a[0]) if b[0] != a[0] else 0
            for a, b in zip(ordered, ordered[1:])
        ]
        # 1.0 = holds every extra byte it decodes; the standalone reader hit 0.02.
        saturating = all(b <= a * 1.05 for a, b in zip(marginal, marginal[1:]))
        verdict = (
            "marginal cost falling = saturating = the OOM property HOLDS"
            if saturating
            else "marginal cost NOT falling -> memory tracks task size, so "
            "saturation does NOT transfer"
        )
        print(
            f"{'':<20}  arrow_rs USS increments: "
            + " / ".join(f"{d:+.0f}" for d in deltas)
            + f" MiB\n{'':<20}  marginal MiB peak per MiB decoded: "
            + " / ".join(f"{m:.2f}" for m in marginal)
            + f"\n{'':<20}  {verdict}"
            f"\n{'':<20}  Local (exp6 phase F, block=128M) was 0.95 / 0.30 / 0.04."
        )
    print()

# --- phase W: is the S3 surcharge the fetch machinery? -----------------------
w_rows = [r for r in rows if r["tag"].startswith("W_") and not r.get("stats_error")]
if w_rows:
    head = (
        f"{'PHASE W arrow_rs':<20}{'avg USS/task':>15}{'vs win=16':>11}"
        f"{'wall':>8}{'vs win=16':>11}"
    )
    print(head)
    print("-" * len(head))
    for budget in sorted({r.get("prefetch_budget_mb") or 0 for r in w_rows}):
        arms = sorted(
            (r for r in w_rows if (r.get("prefetch_budget_mb") or 0) == budget),
            key=lambda r: r.get("fetch_window_mb") or 0,
        )
        base = next((r for r in arms if r.get("fetch_window_mb") == 16), None)
        blabel = "budget=default(64)" if not budget else f"budget={budget}M"
        print(f"  {blabel}")
        for r in arms:
            u, w = uss(r), r["wall_s"]
            du = f"{u - uss(base):+.0f}M" if base else "-"
            dw = f"{w / base['wall_s']:.2f}x" if base and base["wall_s"] else "-"
            print(
                f"{'    window=' + str(r.get('fetch_window_mb')) + 'M':<20}"
                f"{u:>12.0f}MiB{du:>11}{w:>8.1f}{dw:>11}"
            )
        vals = [uss(r) for r in arms if uss(r)]
        if len(vals) > 1:
            print(f"{'':<20}  spread {max(vals) / min(vals):.2f}x")
    print(
        f"\n{'':<20}  2026-08-07: memory FELL as the window GREW -- 892 MiB at the\n"
        f"{'':<20}  shipping 16 MiB against 618 at 128, wall-neutral, single-shot.\n"
        f"{'':<20}\n"
        f"{'':<20}  MECHANISM RETRACTED. This line used to say the cost was the\n"
        f"{'':<20}  NUMBER of concurrent units, and that a 16 MiB window carves a\n"
        f"{'':<20}  row group into many where 128 makes one. The crate says\n"
        f"{'':<20}  otherwise: window_rows_for (lib.rs:955) divides the MiB by the\n"
        f"{'':<20}  row group's COMPRESSED BYTES PER ROW to get a row count, and\n"
        f"{'':<20}  plan_s3_units clamps it to the group's length. lineitem is ~43\n"
        f"{'':<20}  compressed B/row in 122,428-row groups, so 16 MiB asks for\n"
        f"{'':<20}  ~390,000 rows and clamps to the whole group -- as do 64 and\n"
        f"{'':<20}  128. The unit count is 1 in every arm above. Whatever moved\n"
        f"{'':<20}  USS here, it was not the window.\n"
        f"{'':<20}\n"
        f"{'':<20}  The remaining suspect is the DERIVED budget (-1 gives\n"
        f"{'':<20}  4 x max(window, 16), so 16 -> 64 MiB and 128 -> 512), and it\n"
        f"{'':<20}  moved USS NON-MONOTONICALLY: 665 / 890 / 703 / 618 MiB across\n"
        f"{'':<20}  windows 16/64/256/512. That is not the shape of a knob that\n"
        f"{'':<20}  binds. Phase P stops inferring, pins the window, sweeps the\n"
        f"{'':<20}  budget itself, and repeats it on a fixture where the window is\n"
        f"{'':<20}  actually capable of binding.\n"
    )

# --- phase T: does the serial default survive latency? -----------------------
t_rows = [r for r in rows if r["tag"].startswith("T_") and not r.get("stats_error")]
if t_rows:
    head = (
        f"{'PHASE T threads':<20}{'USS thr=1':>12}{'USS thr=4':>12}{'mem saved':>11}"
        f"{'wall thr=1':>12}{'wall thr=4':>12}{'wall cost':>11}"
    )
    print(head)
    print("-" * len(head))
    for reader in ("pyarrow", "arrow_rs"):
        arms = {r.get("threads"): r for r in t_rows if r["reader"] == reader}
        lo, hi = arms.get(1), arms.get(4)
        if not (lo and hi):
            continue
        print(
            f"{reader:<20}{uss(lo):>9.0f}MiB{uss(hi):>9.0f}MiB"
            f"{(uss(hi) - uss(lo)):>+10.0f}M"
            f"{lo['wall_s']:>12.1f}{hi['wall_s']:>12.1f}"
            f"{(lo['wall_s'] / hi['wall_s'] if hi['wall_s'] else 0):>10.2f}x"
        )
    print(
        f"\n{'':<20}  Local: arrow_rs saved 96 MiB at 0.95-1.02x wall, pyarrow 18.\n"
        f"{'':<20}  On S3 serial also gives up fetch/decode overlap, so the wall\n"
        f"{'':<20}  cost column is what decides whether the default is safe.\n"
    )

# --- phase Z: the fixed per-task cost, measured rather than extrapolated -----
z_rows = [r for r in rows if r["tag"].startswith("Z_") and not r.get("stats_error")]
if z_rows:
    head = (
        f"{'PHASE Z ~0 MiB/task':<24}{'avg USS/task':>14}{'max/avg':>9}"
        f"{'tasks':>7}{'wall':>8}"
    )
    print(head)
    print("-" * len(head))
    # tag is Z_<reader>_<transport>; reader comes off the record, transport off
    # the tag, because "source" is a full URI and too long for a table.
    z = {}
    for r in z_rows:
        transport = r["tag"].rsplit("_", 1)[-1]
        z[(r["reader"], transport)] = r
        avg = uss(r)
        mx = (r.get("max_uss_per_task") or 0) / MiB
        print(
            f"{r['reader'] + ' ' + transport:<24}{avg:>11.0f}MiB"
            f"{(mx / avg if avg else 0):>9.2f}{r.get('uss_num_samples') or 0:>7}"
            f"{r['wall_s']:>8.1f}"
        )

    def zu(reader, transport):
        arm = z.get((reader, transport))
        return uss(arm) if arm else None

    ar_s3, ar_loc = zu("arrow_rs", "s3"), zu("arrow_rs", "local")
    pa_s3, pa_loc = zu("pyarrow", "s3"), zu("pyarrow", "local")
    print()
    if ar_s3 and ar_loc:
        print(
            f"{'':<4}arrow_rs transport surcharge (s3 - local): {ar_s3 - ar_loc:+.0f} MiB"
            "  <- object_store client, connection pool, tokio stacks"
        )
    if pa_s3 and pa_loc:
        print(
            f"{'':<4}pyarrow  transport surcharge (s3 - local): {pa_s3 - pa_loc:+.0f} MiB"
            "  <- the control: PyArrow pays a transport cost too"
        )
    if ar_s3 and pa_s3:
        print(
            f"{'':<4}the gap that has to shrink (arrow_rs - pyarrow, s3): "
            f"{ar_s3 - pa_s3:+.0f} MiB"
        )
        print(
            f"{'':<4}phase S's fitted intercepts predicted +117 MiB. A measured "
            "value far\n"
            f"{'':<4}below that means the fit was extrapolating a concave curve "
            "past its\n"
            f"{'':<4}data and the small-D deficit is smaller than we have been "
            "quoting."
        )
    if ar_s3 and ar_loc and pa_s3 and pa_loc:
        ours = (ar_s3 - ar_loc) - (pa_s3 - pa_loc)
        floor = (ar_loc - pa_loc) if pa_loc else 0
        print(
            f"\n{'':<4}decomposition of the arrow_rs deficit at D ~= 0:\n"
            f"{'':<6}{floor:+7.0f} MiB  reader floor (local arrow_rs - local pyarrow):"
            " the crate,\n"
            f"{'':<19}FFI, and allocator baseline. No knob in this script\n"
            f"{'':<19}reaches it; the fix would be in the crate.\n"
            f"{'':<6}{ours:+7.0f} MiB  transport (our s3 surcharge - PyArrow's): the S3\n"
            f"{'':<19}client, connection pool and tokio runtime we build and\n"
            f"{'':<19}they do not. THIS is the tractable one -- one shared\n"
            f"{'':<19}client, or a capped pool.\n"
            f"{'':<6}whichever term dominates is the one worth working on; if they are\n"
            f"{'':<6}comparable, neither alone closes the small-D gap.\n"
        )

# --- phase P: does prefetch_budget_mb bind, and on which layout? -------------
p_rows = [r for r in rows if r["tag"].startswith("P_") and not r.get("stats_error")]
if p_rows:
    head = (
        f"{'PHASE P budget':<22}{'avg USS/task':>14}{'vs pyarrow':>12}"
        f"{'MiB dec/task':>14}{'wall':>8}"
    )
    print(head)
    print("-" * len(head))
    # Order by P_LAYOUTS so the control layout prints above the one being
    # judged against it; glob order would put bigrg first purely alphabetically.
    wanted = os.environ.get("P_LAYOUTS", "lineitem bigrg").split()
    found = {r["tag"].split("_")[2] for r in p_rows}
    layouts = [x for x in wanted if x in found] + sorted(found - set(wanted))
    for lay in layouts:
        arms = [r for r in p_rows if r["tag"].split("_")[2] == lay]
        ctl = next((r for r in arms if r["reader"] == "pyarrow"), None)
        ours = sorted(
            (r for r in arms if r["reader"] == "arrow_rs"),
            key=lambda r: r.get("prefetch_budget_mb") or 0,
        )
        print(f"  {lay}  (window pinned at {os.environ.get('P_WINDOW', '16')} MiB)")
        # Decoded-per-task cannot come from TOTAL_DECODED_MIB here: that constant
        # describes lineitem's 4 files. The bigrg arms read a different fixture
        # entirely, so the per-task volume is reported from the fixture's own
        # geometry (rows/task x 128 B/row for bigrg's 16 float64 columns).
        for r in ([ctl] if ctl else []) + ours:
            u = uss(r)
            rel = f"{u / uss(ctl):.2f}x" if ctl and uss(ctl) else "-"
            rows_per = r.get("rows_per_task_mean") or 0
            dec = rows_per * 16 * 8 / MiB if lay == "bigrg" else 0
            dec_s = f"{dec:,.0f}" if dec else "-"
            label = (
                "    pyarrow (control)"
                if r["reader"] == "pyarrow"
                else f"    budget={r.get('prefetch_budget_mb')}M"
            )
            print(f"{label:<22}{u:>11.0f}MiB{rel:>12}{dec_s:>14}{r['wall_s']:>8.1f}")
        vals = [uss(r) for r in ours if uss(r)]
        if len(vals) > 1:
            print(
                f"{'':<22}  arrow_rs spread across the budget sweep: "
                f"{max(vals) / min(vals):.2f}x"
            )
    print(
        f"\n{'':<4}Read the spread against the replicate noise printed at the top. A\n"
        f"{'':<4}sweep spread below the noise floor means the budget does NOT bind on\n"
        f"{'':<4}that layout, whatever direction the numbers happen to point.\n"
        f"{'':<4}\n"
        f"{'':<4}lineitem is the control: the window provably cannot bind there (see\n"
        f"{'':<4}the phase W note), so a spread on lineitem is the budget alone. A\n"
        f"{'':<4}spread on bigrg but not lineitem means the budget only matters once\n"
        f"{'':<4}the window has fragmented a row group -- which would make the right\n"
        f"{'':<4}default derived from the row group, not a fixed number of MiB.\n"
        f"{'':<4}\n"
        f"{'':<4}The bigrg pyarrow control is the number to keep regardless of how\n"
        f"{'':<4}the sweep comes out: it is the first in-Ray measurement of the lone\n"
        f"{'':<4}big row group, the layout this reader exists for. Standalone, that\n"
        f"{'':<4}shape was 3.2-3.3x lighter than PyArrow.\n"
    )

# --- phase D: does the decode budget bind on S3, where it did not locally? ---
d_rows = [r for r in rows if r["tag"].startswith("D_") and not r.get("stats_error")]
if d_rows:
    head = (
        f"{'PHASE D decode budget':<24}{'avg USS/task':>14}{'vs 32M':>9}"
        f"{'vs pyarrow':>12}{'wall':>8}"
    )
    print(head)
    print("-" * len(head))
    ctl = next((r for r in d_rows if r["reader"] == "pyarrow"), None)
    ours = sorted(
        (r for r in d_rows if r["reader"] == "arrow_rs"),
        key=lambda r: r.get("decode_budget_mib") or 0,
    )
    # The shipped default is the reference, not the smallest arm: the question is
    # whether moving OFF 32 MiB buys memory, so every delta is against it.
    base = next((uss(r) for r in ours if (r.get("decode_budget_mib") or 0) == 32), 0)
    if ctl:
        print(
            f"{'  pyarrow (control)':<24}{uss(ctl):>11.0f}MiB{'-':>9}"
            f"{'1.00x':>12}{ctl['wall_s']:>8.1f}"
        )
    for r in ours:
        u = uss(r)
        vs_base = f"{u - base:+.0f}M" if base else "-"
        vs_ctl = f"{u / uss(ctl):.2f}x" if ctl and uss(ctl) else "-"
        print(
            f"{'  budget=' + str(r.get('decode_budget_mib')) + 'M':<24}"
            f"{u:>11.0f}MiB{vs_base:>9}{vs_ctl:>12}{r['wall_s']:>8.1f}"
        )
    vals = [uss(r) for r in ours if uss(r)]
    if len(vals) > 1:
        print(f"{'':<24}  spread across the sweep: {max(vals) / min(vals):.2f}x")
    print(
        f"\n{'':<4}Local (exp6 phase B, unfused, threads=1, 128 MiB blocks) was FLAT:\n"
        f"{'':<4}443 / 489 / 462 / 463 MiB across 2 / 8 / 32 / 128 -- 1.10x over a 64x\n"
        f"{'':<4}sweep, non-monotone, minimum on the OLD default. That is the correct\n"
        f"{'':<4}answer locally, because the local path drives the sync reader one batch\n"
        f"{'':<4}at a time and there is no channel for batches to sit in.\n"
        f"{'':<4}\n"
        f"{'':<4}On S3 each stream holds S3_CHANNEL_DEPTH=2 decoded batches, so the\n"
        f"{'':<4}prediction is a swing of about 2 x the budget difference: 2 MiB should\n"
        f"{'':<4}land ~60 MiB under 32 MiB. Compare the spread against the replicate\n"
        f"{'':<4}noise at the top before believing any of it.\n"
        f"{'':<4}\n"
        f"{'':<4}Even a clean hit only explains ~60 of the 335 MiB the S3 slope costs at\n"
        f"{'':<4}D = 1055. If the spread is at the noise floor, the channel is not it\n"
        f"{'':<4}either and the next instrument is MALLOC_ARENA_MAX, not a knob.\n"
    )

# --- phase X: the column-group path, finally executed inside Ray -------------
x_rows = [r for r in rows if r["tag"].startswith("X_") and not r.get("stats_error")]
if x_rows:
    xdec = float(os.environ.get("X_DECODED_MIB", "256"))
    head = (
        f"{'PHASE X wide (64 cols)':<26}{'avg USS/task':>14}{'vs pyarrow':>12}"
        f"{'col grp rgs':>13}{'retained':>11}{'wall':>8}"
    )
    print(head)
    print("-" * len(head))
    ctl = next((r for r in x_rows if r["reader"] == "pyarrow"), None)

    def _xlabel(r):
        if r["reader"] == "pyarrow":
            return "  pyarrow (control)"
        cf = r.get("column_fetch_mb")
        bud = r.get("decode_budget_mib")
        if cf and cf < 0:
            return "  cf=0 (grouping OFF)"
        return f"  cf=16 budget={bud}M" if bud else "  cf=16 (shipping)"

    # Explicit order: control, the shipping arm, then the two arms that exist to
    # explain it. Glob order puts cf=0 above cf=16, which reads as though the
    # disabled-grouping arm were the configuration under test.
    # shipping -> grouping OFF (the A/B) -> budget=2 (the control for the claim
    # that the budget cannot reach this path at all).
    def _xrank(r):
        cf, bud = r.get("column_fetch_mb"), r.get("decode_budget_mib")
        if cf and cf < 0:
            return 1
        return 2 if bud else 0

    order = ([ctl] if ctl else []) + sorted(
        (r for r in x_rows if r["reader"] == "arrow_rs"), key=_xrank
    )
    for r in order:
        u = uss(r)
        rel = f"{u / uss(ctl):.2f}x" if ctl and uss(ctl) else "-"
        # Straight off the profiler, per arm: an arm that did not actually take the
        # branch is not evidence about the branch, however its USS came out.
        cg = r.get("prof_col_group_rgs")
        cg_s = "-" if cg is None else str(cg)
        ret = r.get("prof_max_retained_mib")
        ret_s = "-" if ret is None else f"{ret:,.0f}MiB"
        print(
            f"{_xlabel(r):<26}{u:>11.0f}MiB{rel:>12}{cg_s:>13}{ret_s:>11}"
            f"{r['wall_s']:>8.1f}"
        )
    print(
        f"\n{'':<4}D is fixed at ~{xdec:,.0f} MiB decoded per task here, so read this as one\n"
        f"{'':<4}point on the phase S curve, not a sweep. For scale, phase S at the\n"
        f"{'':<4}nearest chunk (D = 211) had arrow-rs at 377 and PyArrow at 357 MiB on\n"
        f"{'':<4}16 NARROW columns, where the column-group branch cannot fire.\n"
        f"{'':<4}\n"
        f"{'':<4}The col-grp-rgs column is the arm's own proof it took the branch. Any\n"
        f"{'':<4}cf=16 row showing 0 there measures nothing about wide schemas -- check\n"
        f"{'':<4}the fixture's per-column compressed bytes before reading further.\n"
        f"{'':<4}\n"
        f"{'':<4}Expected, from moto (64 cols x 65,536 rows, one row group): cf=16 fires\n"
        f"{'':<4}3 groups and retains 51.4 MiB against a 39.9 MiB row group (1.29x);\n"
        f"{'':<4}cf=0 takes row windows and retains 25.7 MiB, one batch. The mechanism\n"
        f"{'':<4}is already settled -- Hstack accumulates every batch of every group\n"
        f"{'':<4}before emitting (crate lib.rs ~1645) -- so what is being measured here\n"
        f"{'':<4}is only what it COSTS against PyArrow inside a real read task.\n"
        f"{'':<4}\n"
        f"{'':<4}Prediction: PARITY. Holding the whole decoded row group is what\n"
        f"{'':<4}PyArrow's scanner does too, so wide schemas should show arrow-rs losing\n"
        f"{'':<4}its advantage rather than going backwards. If cf=16 is much WORSE than\n"
        f"{'':<4}PyArrow, that is the release regression located. If cf=0 is much\n"
        f"{'':<4}lighter than cf=16, the fix is to interleave the groups -- pull batch i\n"
        f"{'':<4}from all N streams, stitch, emit, drop -- which turns peak from the row\n"
        f"{'':<4}group into one full-width batch. The budget=2M arm is the control for\n"
        f"{'':<4}that claim: the mechanism says the budget cannot reach this path, so if\n"
        f"{'':<4}USS moves there, re-read the crate before fixing anything.\n"
    )

# --- the column-group check, which USS alone cannot make -------------------
# A read whose memory looks fine can still have taken the branch that retains a
# whole decoded row group -- it just did not happen to be the peak. exp2 found
# col_group_rgs = 0 on lineitem and this should confirm it, but "should" is why
# it is worth one grep.
counters = {}
for path in glob.glob(os.path.join(RESULTS, "prof", "*", "*.jsonl")):
    tag = os.path.basename(os.path.dirname(path))
    with open(path) as handle:
        for line in handle:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            for key in ("col_group_rgs", "oversized_units", "retained_bytes"):
                if key in rec:
                    slot = counters.setdefault(tag, {})
                    slot[key] = max(slot.get(key, 0), rec[key] or 0)
if counters:
    print(f"{'PROFILER':<34}{'col_group_rgs':>15}{'oversized':>11}{'max retained':>15}")
    print("-" * 75)
    for tag in sorted(counters):
        c = counters[tag]
        print(
            f"{tag:<34}{c.get('col_group_rgs', 0):>15}{c.get('oversized_units', 0):>11}"
            f"{c.get('retained_bytes', 0) / MiB:>12.0f}MiB"
        )
    print(
        "\nNon-zero col_group_rgs means the read took the branch that holds a whole"
        "\ndecoded row group -- PyArrow's exact behaviour, and the thing this reader"
        "\nexists to avoid. Expected 0 on lineitem (16 narrow columns).\n"
    )

print(
    "Everything above is UNFUSED: no write is attached, because exp6 phase F"
    "\nshowed the Parquet writer held 589 of 1032 MiB at the default chunk and"
    "\nwould dominate any S3 measurement the same way it dominated the local ones."
    "\nThe write path is a separate, shared-with-PyArrow workstream."
    "\n"
    "\nWhat this experiment can and cannot settle: one box, one region, and three"
    "\nschemas -- lineitem (16 columns, 172 B/row, 49 row groups per file), bigrg"
    "\n(16 columns, 128 B/row, ONE row group per file) and wide (64 columns, fat"
    "\nenough row groups to trip the column-group planner). It CAN settle:"
    "\n  S  does the saturation that makes this reader OOM-resistant survive S3"
    "\n  Z  what a read task costs before it decodes anything, and which half of"
    "\n     that is the transport rather than the reader"
    "\n  P  is prefetch_budget_mb a memory knob or a red herring"
    "\n  D  is the decoded channel the per-byte S3 surcharge"
    "\n  X  what the column-group path costs against PyArrow inside a read task"
    "\n"
    "\nIt still cannot speak to MULTI-NODE contention, which is where the release"
    "\nsuite runs and where a per-task number and a per-node number diverge; nor to"
    "\nschemas in the thousands of columns, where the standalone harness measured"
    "\n5000 and this tops out at 64; nor to anything fused, since every arm here is"
    "\nread-only on purpose."
)
PYEOF
