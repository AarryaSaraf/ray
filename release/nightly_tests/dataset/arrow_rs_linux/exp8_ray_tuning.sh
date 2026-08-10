#!/usr/bin/env bash
# exp8 -- tune RAY for arrow-rs, and reproduce the release suite's SPEED problem.
#
# Why this exists, when exp6/exp7 already ran
# -------------------------------------------
# Two things became clear on 2026-08-10, and both say the previous experiments
# were aimed slightly wrong.
#
# 1. THE RELEASE REGRESSIONS ARE MOSTLY SPEED, NOT MEMORY. Re-reading the A/B
#    (regression_testing.md §2) by metric rather than by rank:
#
#      wide_schema_pipeline_tensors   4.90x  READ OP WALL   obj 1.11x
#      read_parquet_autoscaling       3.06x  READ OP WALL   avg USS 1.01x
#      read_parquet_fixed_size        2.64x  READ OP WALL   avg USS 0.77x, obj 0.61x  <- WINS on memory
#      write_parquet                  2.28x  max USS/task   wall 1.00x  (shared writer, not us)
#
#    The worst read-time regression in the entire run is simultaneously a memory
#    win on both memory metrics. exp6 and exp7 measured per-task USS almost
#    exclusively -- they were instrumented for the wrong axis, and they still
#    have not reproduced the read-time regression on one box even once.
#
#    Note also the noise floor is NOT uniformly large, contrary to what was
#    written for a while: `fixed_size` cases average 1.02x (max 1.82x). Only
#    `autoscaling` wall time has the ~2.5x control floor. So read_parquet_
#    fixed_size at 2.64x is real signal, not cluster variance.
#
# 2. EVERY RATIO WE HAVE EVER QUOTED WAS MEASURED AT PYARROW'S OPTIMUM. Ray Data
#    was built and tuned against PyArrow for years; its defaults encode PyArrow's
#    cost model. Two examples that are not hypothetical:
#
#      * ParquetFileChunker's 1 GiB default exists, per its own comment, so that
#        "if the compression ratio is high ... reading can OOM" -- i.e. it is
#        sized around PyArrow materialising a whole row group. We do not have
#        that failure mode, and we are on the WINNING side of the break-even
#        above D ~= 540 MiB, so the tuned-for-us value is plausibly LARGER.
#      * target_max_block_size = 128 MiB costs both readers ~2.4 MiB of peak per
#        MiB of block, but our block-independent base is ~128 MiB against
#        PyArrow's ~447 -- so the same block size is a far larger FRACTION of our
#        footprint. A smaller block is nearly free for us and expensive for them.
#
#    The fair experiment is not "arrow-rs vs PyArrow at Ray's defaults". It is
#    "each reader at ITS OWN best settings", and nobody has run that. Where the
#    two optima differ we get either a per-reader default (as already happened
#    with the fragment pool, 4 -> 1) or a documented recommendation.
#
# What this does NOT assume
# -------------------------
# A tempting hypothesis got demoted while writing this, and the demotion is
# recorded so it is not re-derived: on files LARGER than the chunk size, the
# reader builds one native fragment PER ROW GROUP
# (arrow_rs_parquet_file_reader.py:1171) rather than one per range, so a chunk of
# 16 row groups is 16 separate crate calls -- and with threads=1 they are serial,
# no cross-row-group fetch overlap. That looked like a clean explanation for
# "slower reads, better memory".
#
# But exp7 phase S ALREADY RAN THAT PATH. At chunk = 16 / 64 / 256 MiB against
# ~677 MiB lineitem files, chunk_metadata is not None, so every one of those arms
# used per-row-group fragments -- and wall time was at parity throughout
# (11.8 / 11.2 / 10.4 vs PyArrow's 11.9 / 11.0 / 10.5). Per-row-group dispatch
# costs nothing measurable at this scale. Phase K re-checks it as a side effect
# (it sweeps chunk size on both readers and prints read-op time, which phase S
# did not capture), but it is not the premise of any phase here.
#
# The phases
# ----------
#   C  CONCURRENCY. ray.init(num_cpus=N) for N in 1..8, both readers, S3. This is
#      the one release axis every experiment here has held fixed (common.sh pins
#      8). It is also where the readers are least alike: PyArrow overlaps S3
#      latency with an 8-thread io pool plus batch/fragment readahead, while we
#      run 1 fragment thread and rely on tokio INSIDE the task. At 1 concurrent
#      task that favours us; at 8, eight tasks' tokio threads contend for the
#      same cores. That is the only mechanism proposed so far that makes read-op
#      time worse while memory gets better, which is exactly the signature of
#      read_parquet_fixed_size. `read_cpu_sum_s / read_wall_sum_s` is the direct
#      oversubscription read-out.
#
#   K  KNOB ASYMMETRY. chunk size x reader, then block size x reader, both
#      readers at every setting, reporting BOTH memory and read-op time. The
#      output is not a ratio -- it is two argmins. If they differ, that is a
#      per-reader default or a recommendation; if they agree, we stop wondering
#      whether Ray's defaults are quietly PyArrow-shaped.
#
#   F  FALLBACK AUDIT. The `tensors` fixture (one pyarrow canonical
#      fixed_shape_tensor column among primitives) against the `wide` fixture of
#      the same geometry. wide_schema_pipeline_tensors is the worst regression in
#      the release run at 4.90x read-op wall, and the support gate rejects any
#      field with an extension_name -- so the first question is not "why is our
#      decode slow", it is "does our decoder run at all?" If the file falls back,
#      both release arms decoded through PyArrow and the 4.90x is the cost of
#      DECIDING to fall back (native footer read, then a pyarrow read of the same
#      file), which is a different bug in a different place. The profiler answers
#      it directly: `kind: "fallback"` records carry the reason.
#
# All phases are unfused (no --write-to): exp6 phase F showed a fused writer
# holds 589 of 1032 MiB, which would dominate any memory number here, and the
# write path is its own workstream.
#
#   PHASES="C K F" REPEAT=3 ./exp8_ray_tuning.sh
#
# ~40-55 min at REPEAT=3 for all three phases. Must run in us-west-2.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
check_env
check_s3

SF="${SF:-10}"
S3_DATA="${S3_DATA:-$S3_ROOT/tpch/sf${SF}/lineitem}"
MAX_FILES="${MAX_FILES:-4}"
# 4 sf10 lineitem files. Kept identical to exp6/exp7 so read-op times here are
# comparable with the USS numbers already recorded for the same shape.
TOTAL_DECODED_MIB="${TOTAL_DECODED_MIB:-4219}"
RESULTS="${RESULTS:-$OUT_DIR/exp8}"
REPEAT="${REPEAT:-1}"

FIXED_BLOCK="${FIXED_BLOCK:-128}"
# Phase C. 1 is the cleanest per-task measurement (no cross-task contention at
# all); 8 is what common.sh pins and what every prior experiment measured; 2 and
# 4 bracket the middle so a monotone trend is distinguishable from a cliff.
C_CPUS="${C_CPUS:-1 2 4 8}"
# Phase K. The chunker splits on ON-DISK size, and lineitem files are ~677 MiB,
# so 1024 (and anything above it) leaves files whole while 256/64 split them.
# 4096 is the "chunker removed" arm -- identical to 1024 on this input, and it is
# the control proving that the default is already in the no-split regime here.
K_CHUNKS="${K_CHUNKS:-64 256 1024 4096}"
K_BLOCKS="${K_BLOCKS:-32 128 512}"
# Phase F. Same geometry both fixtures (4 files x 4 row groups), so the only
# difference is one column's type.
S3_WIDE="${S3_WIDE:-$S3_ROOT/fixtures/wide}"
S3_TENSORS="${S3_TENSORS:-$S3_ROOT/fixtures/tensors}"
F_FILES="${F_FILES:-4}"

PHASES="${PHASES:-C K F}"

mkdir -p "$RESULTS"
has_phase() { case " $PHASES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
for p in $PHASES; do rm -f "$RESULTS/${p}_"*.json; done

python "$HERE/stage_data.py" --dst "$S3_DATA" --sf "$SF" \
  --region "$AWS_DEFAULT_REGION"

# One process per arm: Ray reuses workers, and MemoryProfiler reads whole-process
# private memory, so arm 2 would inherit arm 1's high-water mark.
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

# --- phase C: concurrency ----------------------------------------------------
# Note --threads is pinned to 1 for BOTH readers here, not left at the default.
# Leaving it at 4 for PyArrow and 1 for arrow-rs (which is what ships) would
# confound "how does this reader scale with task concurrency" with "how many
# fragment threads does each reader get" -- two different multiplications of the
# same cores. The shipped asymmetry is already measured (exp7 phase T).
if has_phase C; then
  for cpus in $C_CPUS; do
    for reader in pyarrow arrow_rs; do
      run_probe "C_${reader}_cpu${cpus}" \
        --reader "$reader" --block-mib "$FIXED_BLOCK" --threads 1 \
        --num-cpus "$cpus"
    done
  done
fi

# --- phase K: knob asymmetry -------------------------------------------------
if has_phase K; then
  for chunk in $K_CHUNKS; do
    for reader in pyarrow arrow_rs; do
      run_probe "K_${reader}_chunk${chunk}" \
        --reader "$reader" --block-mib "$FIXED_BLOCK" --threads 1 \
        --chunk-mib "$chunk"
    done
  done
  for block in $K_BLOCKS; do
    for reader in pyarrow arrow_rs; do
      # Tagged Kb_ rather than K_ so the summary can split the two sweeps
      # without parsing which knob moved.
      run_probe "Kb_${reader}_block${block}" \
        --reader "$reader" --block-mib "$block" --threads 1
    done
  done
fi

# --- phase F: fallback audit -------------------------------------------------
if has_phase F; then
  need_fixture F "$S3_WIDE"
  need_fixture F "$S3_TENSORS"
  for fx in wide tensors; do
    case "$fx" in
      wide) src="$S3_WIDE" ;;
      tensors) src="$S3_TENSORS" ;;
    esac
    for reader in pyarrow arrow_rs; do
      SRC="$src" MAXF="$F_FILES" run_probe "F_${reader}_${fx}" \
        --reader "$reader" --block-mib "$FIXED_BLOCK" --threads 1
    done
  done
fi

# --- summary -----------------------------------------------------------------
{
  RESULTS="$RESULTS" \
  python - <<'PYEOF' | tee "$OUT_DIR/exp8_summary.txt"
import glob
import json
import os
import re

RESULTS = os.environ["RESULTS"]

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
    """Median over replicates, carrying the spread on BOTH axes.

    exp7's version medianed USS and wall only. This one also medians read-op
    time and reports its spread separately, because that is the metric the whole
    experiment is about and memory noise says nothing about timing noise -- the
    2026-08-07 run measured one config three times at 11% spread on memory and
    64% on wall.
    """
    keys = (
        "reader",
        "block_mib",
        "chunk_mib",
        "threads",
        "num_cpus",
        "decode_budget_mib",
        "column_fetch_mb",
        # Phase F's two fixtures differ in nothing else, so without the source
        # the median would average `wide` and `tensors` together and report the
        # average as both -- the exact bug found in phases Z/P and again in X.
        "source",
        "num_files",
    )
    groups = {}
    for r in records:
        base = re.sub(r"_r\d+$", "", r["tag"])
        groups.setdefault(
            (base.split("_")[0], tuple(r.get(k) for k in keys)), []
        ).append(r)
    out = []
    for reps in groups.values():
        good = [r for r in reps if not r.get("stats_error")]
        merged = dict((good or reps)[0])
        merged["tag"] = re.sub(r"_r\d+$", "", merged["tag"])
        merged["n_reps"] = len(reps)
        if good:
            uss = [r["avg_max_uss_per_task"] for r in good]
            merged["avg_max_uss_per_task"] = _median(uss)
            merged["wall_s"] = _median([r["wall_s"] for r in good])
            merged["uss_spread"] = (max(uss) / min(uss)) if min(uss) else 0
            rd = [r["read_wall_sum_s"] for r in good if r.get("read_wall_sum_s")]
            if rd:
                merged["read_wall_sum_s"] = _median(rd)
                merged["read_spread"] = (max(rd) / min(rd)) if min(rd) else 0
            cp = [r["read_cpu_sum_s"] for r in good if r.get("read_cpu_sum_s")]
            if cp:
                merged["read_cpu_sum_s"] = _median(cp)
        out.append(merged)
    return out


rows = collapse(raw)
by_tag = {r["tag"]: r for r in rows}


def fmt(v, nd=0):
    return "-" if v is None else f"{v:,.{nd}f}"


def mib(r):
    v = r.get("avg_max_uss_per_task")
    return v / (1024 * 1024) if v else None


def ratio(num, den):
    return f"{num / den:.2f}x" if (num and den) else "-"


us = [r["uss_spread"] for r in rows if r.get("uss_spread", 0) and r["n_reps"] > 1]
rs = [r["read_spread"] for r in rows if r.get("read_spread", 0) and r["n_reps"] > 1]
print(f"replicates: {max(r['n_reps'] for r in rows)} per arm")
if us or rs:
    print(
        f"worst within-arm spread: USS {max(us or [0]):.2f}x, "
        f"read-op time {max(rs or [0]):.2f}x -- smaller differences are noise"
    )
else:
    print(
        "1 replicate per arm. Prior runs showed ~1.13x on S3 memory and much\n"
        "worse on timing; re-run with REPEAT=3 before quoting anything."
    )
print()

bad = [r for r in rows if r.get("stats_error")]
if bad:
    print("!! arms with no usable stats (excluded below):")
    for r in bad:
        print(f"   {r['tag']}: {r['stats_error']}")
    print()

# ---- phase C ----------------------------------------------------------------
crows = sorted(
    (r for r in rows if r["tag"].startswith("C_") and not r.get("stats_error")),
    key=lambda r: (r.get("num_cpus") or 0, r["reader"]),
)
if crows:
    print("=== phase C: concurrency (num_cpus), S3, threads=1 both readers ===")
    print(
        "The axis every prior experiment pinned at 8. If arrow-rs's read-op time\n"
        "degrades with concurrency faster than PyArrow's, that is the release\n"
        "read-time regression, and cpu/wall > 1 says why (tokio threads inside a\n"
        "1-CPU task). If both degrade alike, concurrency is not the mechanism and\n"
        "the next suspect is the fetch pipeline itself.\n"
    )
    print(
        f"{'cpus':>4} | {'USS P':>7} {'USS R':>7} {'ratio':>6} | "
        f"{'readop P':>9} {'readop R':>9} {'ratio':>6} | {'cpu/wall R':>10}"
    )
    print("-" * 78)
    for cpus in sorted({r.get("num_cpus") for r in crows if r.get("num_cpus")}):
        p = by_tag.get(f"C_pyarrow_cpu{cpus}")
        a = by_tag.get(f"C_arrow_rs_cpu{cpus}")
        if not (p and a):
            continue
        cw = (
            a["read_cpu_sum_s"] / a["read_wall_sum_s"]
            if a.get("read_cpu_sum_s") and a.get("read_wall_sum_s")
            else None
        )
        print(
            f"{cpus:>4} | {fmt(mib(p)):>7} {fmt(mib(a)):>7} "
            f"{ratio(mib(a), mib(p)):>6} | "
            f"{fmt(p.get('read_wall_sum_s'), 1):>9} "
            f"{fmt(a.get('read_wall_sum_s'), 1):>9} "
            f"{ratio(a.get('read_wall_sum_s'), p.get('read_wall_sum_s')):>6} | "
            f"{fmt(cw, 2):>10}"
        )
    print()

# ---- phase K ----------------------------------------------------------------
def argbest(prefix, knob, values, metric):
    """The setting each reader is best at, per metric. The POINT of phase K.

    A ratio table answers "who wins at Ray's default". These two argmins answer
    "is Ray's default the right one for each of them", which is the question the
    default was never chosen to answer.
    """
    out = {}
    for reader in ("pyarrow", "arrow_rs"):
        cands = []
        for v in values:
            r = by_tag.get(f"{prefix}_{reader}_{knob}{v}")
            if r and not r.get("stats_error") and r.get(metric):
                cands.append((r[metric], v))
        out[reader] = min(cands)[1] if cands else None
    return out


for prefix, knob, label, env in (
    ("K", "chunk", "parquet_chunker_target_chunk_size (MiB)", "K_CHUNKS"),
    ("Kb", "block", "target_max_block_size (MiB)", "K_BLOCKS"),
):
    vals = [
        int(m.group(1))
        for m in (
            re.match(rf"{prefix}_\w+?_{knob}(\d+)$", r["tag"]) for r in rows
        )
        if m
    ]
    vals = sorted(set(vals))
    if not vals:
        continue
    print(f"=== phase K: {label} ===")
    print(
        f"{knob:>6} | {'USS P':>7} {'USS R':>7} {'ratio':>6} | "
        f"{'readop P':>9} {'readop R':>9} {'ratio':>6} | {'tasks R':>7}"
    )
    print("-" * 76)
    for v in vals:
        p = by_tag.get(f"{prefix}_pyarrow_{knob}{v}")
        a = by_tag.get(f"{prefix}_arrow_rs_{knob}{v}")
        if not (p and a):
            continue
        print(
            f"{v:>6} | {fmt(mib(p)):>7} {fmt(mib(a)):>7} "
            f"{ratio(mib(a), mib(p)):>6} | "
            f"{fmt(p.get('read_wall_sum_s'), 1):>9} "
            f"{fmt(a.get('read_wall_sum_s'), 1):>9} "
            f"{ratio(a.get('read_wall_sum_s'), p.get('read_wall_sum_s')):>6} | "
            f"{fmt(a.get('task_count')):>7}"
        )
    for metric, name in (
        ("avg_max_uss_per_task", "memory"),
        ("read_wall_sum_s", "read-op time"),
    ):
        best = argbest(prefix, knob, vals, metric)
        verdict = (
            "SAME optimum -- Ray's default is not reader-specific on this axis"
            if best["pyarrow"] == best["arrow_rs"]
            else "DIFFERENT optima -> candidate per-reader default"
        )
        print(
            f"  best for {name:<13}: pyarrow={best['pyarrow']}, "
            f"arrow_rs={best['arrow_rs']}  ({verdict})"
        )
    print(
        f"  (sweep values come from ${env}; the argmin is only meaningful\n"
        "   inside the range swept, and only if the spread beats the noise line\n"
        "   at the top of this summary.)\n"
    )

# ---- phase F ----------------------------------------------------------------
frows = [r for r in rows if r["tag"].startswith("F_")]
if frows:
    print("=== phase F: does the wide/tensor shape reach the native path? ===")
    print(
        "wide_schema_pipeline_tensors is the worst release regression (4.90x\n"
        "read-op wall). The support gate rejects any field with an\n"
        "extension_name, so if `tensors` shows fallbacks, BOTH release arms\n"
        "decoded through PyArrow and that 4.90x cannot be our decoder -- it is\n"
        "the cost of deciding to fall back. `wide` is the same geometry with no\n"
        "extension column, so it is the control.\n"
    )
    print(
        f"{'fixture':>8} | {'USS P':>7} {'USS R':>7} {'ratio':>6} | "
        f"{'readop P':>9} {'readop R':>9} {'ratio':>6} | {'native?':>8}"
    )
    print("-" * 80)
    for fx in ("wide", "tensors"):
        p = by_tag.get(f"F_pyarrow_{fx}")
        a = by_tag.get(f"F_arrow_rs_{fx}")
        if not (p and a):
            continue
        fb = a.get("prof_fallbacks")
        # Three states, not two. A blank column used to mean either "zero
        # fallbacks" or "no records were ever written", and the 2026-08-10 run
        # spent its whole phase F in the second state without saying so.
        if a.get("profile_error"):
            native = "NO PROF"
        else:
            native = "FELL BACK" if fb else "native"
        print(
            f"{fx:>8} | {fmt(mib(p)):>7} {fmt(mib(a)):>7} "
            f"{ratio(mib(a), mib(p)):>6} | "
            f"{fmt(p.get('read_wall_sum_s'), 1):>9} "
            f"{fmt(a.get('read_wall_sum_s'), 1):>9} "
            f"{ratio(a.get('read_wall_sum_s'), p.get('read_wall_sum_s')):>6} | "
            f"{native:>8}"
        )
    for fx in ("wide", "tensors"):
        a = by_tag.get(f"F_arrow_rs_{fx}")
        if not a:
            continue
        reasons = a.get("prof_fallback_reasons")
        if a.get("profile_error"):
            print(f"  {fx}: NO PROFILING RECORDS -- {a['profile_error']}")
            print(
                "     every counter below is uncollected, NOT zero; this phase"
                " cannot answer the fallback question"
            )
            continue
        print(
            f"  {fx}: fallbacks={a.get('prof_fallbacks', 0)}, "
            f"col_group_rgs={a.get('prof_col_group_rgs')}, "
            f"row_window_rgs={a.get('prof_row_window_rgs')}"
        )
        if reasons:
            print(f"     reasons: {reasons}")
    print()

print("=== what this run can and cannot settle ===")
print(
    "CAN: whether task concurrency is the read-time regression (C); whether any\n"
    "  Ray knob has a different optimum per reader (K); whether the worst\n"
    "  release regression is even on the arrow-rs path (F).\n"
    "CANNOT: multi-node contention or autoscaling, which is where the release\n"
    "  suite's own noise lives; sf1000-scale file counts; anything fused; and\n"
    "  read-op time on shapes not fixtured here. A same-optimum result in K is\n"
    "  only as strong as the swept range -- it does not prove the default is\n"
    "  optimal, only that it is not reader-specific between the values tried."
)
PYEOF
} || echo "!! summary failed"

echo
echo "also saved to $OUT_DIR/exp8_summary.txt"
