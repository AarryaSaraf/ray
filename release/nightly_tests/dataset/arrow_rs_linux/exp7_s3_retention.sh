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
#
# Runtime: ~35-45 min for 20 arms. Run in us-west-2 or the numbers are latency,
# not memory.
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

FIXED_BLOCK="${FIXED_BLOCK:-128}"
S_CHUNKS="${S_CHUNKS:-16 64 256 0}"
# fetch_window_mb for the phase S grid. 0 = the shipping default (16), which is
# what the first run used -- and phase W then found 16 is the WORST setting on
# this axis. `S_WINDOW=128 PHASES=S ./exp7_s3_retention.sh` re-asks the transport
# question at the best one.
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
run_probe() {  # tag, then probe args
  local tag="$1"; shift
  local rep
  for rep in $(seq 1 "${REPEAT:-1}"); do
    local rtag="$tag"
    [ "${REPEAT:-1}" -gt 1 ] && rtag="${tag}_r${rep}"
    echo "=== $rtag"
    env RAY_DATA_ARROW_RS_PROFILE=1 \
        RAY_DATA_ARROW_RS_PROFILE_DIR="$RESULTS/prof/$rtag" \
      python "$HERE/block_retention_probe.py" \
        --source "$S3_DATA" --max-files "$MAX_FILES" \
        --out "$RESULTS/$rtag.json" "$@" \
        > "$RESULTS/$rtag.log" 2>&1 \
      || echo "  FAILED -- see $RESULTS/$rtag.log"
    grep -h "^WARNING:" "$RESULTS/$rtag.log" 2>/dev/null || true
  done
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

RESULTS="$RESULTS" TOTAL_DECODED_MIB="$TOTAL_DECODED_MIB" \
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
counts = {r.get("num_files") for r in rows if r.get("num_files")}
if len(counts) > 1:
    print(f"!! arms read differing file counts {sorted(counts)} -- not comparable\n")
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
        f"\n{'':<20}  2026-08-07: memory FALLS as the window GROWS -- 892 MiB at\n"
        f"{'':<20}  the shipping 16 MiB against 618 at 128, wall-neutral. Note\n"
        f"{'':<20}  prefetch_budget_mb=-1 derives to 4 x max(window, 16), so the\n"
        f"{'':<20}  128 arm had a 512 MiB byte budget and still used the least:\n"
        f"{'':<20}  what costs is the NUMBER of concurrent units, not the bytes\n"
        f"{'':<20}  admitted. A 4 MiB window carves a row group into many small\n"
        f"{'':<20}  units and the budget admits ~16 at once, each with its own\n"
        f"{'':<20}  decode state; a 128 MiB window means one. The budget=16 rows\n"
        f"{'':<20}  confirm it from the other side -- capping the count directly\n"
        f"{'':<20}  makes the window irrelevant (spread 1.03x).\n"
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
    "\nWhat this experiment can and cannot settle: it is one box, one region, one"
    "\nschema (lineitem, 16 narrow columns at 172 B/row). It cannot speak to wide"
    "\nschemas, where the column-group planner actually fires, or to multi-node"
    "\ncontention. It CAN settle whether the saturation that makes this reader"
    "\nOOM-resistant is a local-filesystem artifact -- which is the one thing"
    "\nstanding between the current results and a PR description."
)
PYEOF
