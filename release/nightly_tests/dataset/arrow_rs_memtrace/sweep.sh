#!/usr/bin/env bash
# Run the full v1/v2/v3 memory sweep (3 tests) in one shot, each read in its own
# fresh process (isolation -- a prior run's leak must not contaminate the next),
# streaming full output to a log and printing a clean digest at the end.
#
#   bash sweep.sh
#
# Must run from the arrow_rs_memtrace dir (mem3.py imports bench_suite).
set -u

cd "$(dirname "$0")"
DATA="$PWD/leak_linux_data"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$PWD/runs/sweep_${TS}.log"
mkdir -p "$PWD/runs"
echo "logging to $LOG"

run() {  # run <label> -- <cmd...>   ; label is echoed into the log as a marker
  local label="$1"; shift; shift  # drop label and the literal "--"
  echo -e "\n##### $label #####" | tee -a "$LOG"
  "$@" >>"$LOG" 2>&1
}

# --- fixtures (generate once; gen.py is a no-op-fast rewrite if you rerun) -----
SMALL="$DATA/leak_1000000_250000.parquet"   # 80MB, 4 row groups (already present)
BIGRG="$DATA/leak_4000000_4000000.parquet"  # ~300MB, ONE giant row group
MULTI="$DATA/leak_4000000_500000.parquet"   # ~300MB, 8 row groups
[ -f "$BIGRG" ] || python gen.py 4000000 4000000
[ -f "$MULTI" ] || python gen.py 4000000 500000

# --- Test 1: lone-big-rg mode sweep (arrow-rs thesis). v3 first, v1 last. ------
for m in v3 v2 v1; do run "T1 bigrg $m" -- python mem3.py "$m" "$BIGRG"; done

# --- Test 2: v1 batch-size leak curve (only v1 responds; small file) ----------
for b in 131072 8192 128 8; do run "T2 leakcurve v1 bs=$b" -- python mem3.py v1 "$SMALL" "$b"; done

# --- Test 3: arrow-rs decode-budget sweep (v3, lone-big-rg) --------------------
for mb in 1 2 8 32; do
  run "T3 budget v3 ${mb}MB" -- \
    env RAY_DATA_ARROW_RS_DECODE_BUDGET_BYTES=$((mb*1024*1024)) python mem3.py v3 "$BIGRG"
done

# --- Test 4: arrow-rs intra-row-group K-split sweep (v3, lone-big-rg) ----------
# K splits the ONE big row group into K concurrent row-ranges, each doing its own
# fetch+decode. Locally there's no I/O latency to hide, so this isolates whether
# decode-parallelism across cores buys wall time -- and what it costs in peak
# memory (each range holds its own decode transient, so peak grows ~K x). v2 is
# in Test 1 as the reference; here we just walk arrow-rs K=1->2->4.
for k in 1 2 4; do
  run "T4 kfan v3 K=$k" -- \
    env RAY_DATA_ARROW_RS_K=$k python mem3.py v3 "$BIGRG"
done

# --- digest: pull just the result blocks out of the full log ------------------
echo -e "\n\n=================== SUMMARY ===================" | tee -a "$LOG"
grep -E "^#####|^=====|^wall|^node-sum|^decode path" "$LOG" | tee -a "$LOG"
echo -e "\nfull log: $LOG"
