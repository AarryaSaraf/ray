# arrow-rs reader — single-machine investigation

Experiments that reproduce, on one 8-CPU / 32 GB Linux box, what the multi-node
release A/B (builds 103100 vs 103101) could only show as aggregates. Everything
here is measurement scaffolding for the flag-gated arrow-rs Parquet reader; none
of it runs in CI.

> **Results live in the docs, not here.**
> `python/ray/data/arrow_rs_docs/regression_testing.md` **§8** carries what these
> runs found and which hypotheses they killed;
> `python/ray/data/arrow_rs_docs/TODO.md` §1 carries what is left to do. This
> file is only how to run things.
>
> Headline, so nobody re-derives it: the crate is **not** the problem (outside
> Ray it is 1.57× faster on 1.62× less CPU at equal cores, at 20–40× less
> memory). What remains is a **~104 MiB fixed cost per read task** that arrow-rs
> pays and PyArrow does not, which makes arrow-rs win on big read tasks and lose
> on small ones. Neither block size (32× sweep) nor decode budget (16× sweep)
> touches it.

## Why the first three

The release A/B found the arrow-rs reader mixed-to-negative on memory, which
contradicts every local measurement. Reading the code turned up three defects,
all of them in the crate's **S3** planner — `plan_s3_units`,
`partition_columns_by_budget`, and the prefetch admission loop. Those functions
exist only in `read_row_groups_s3`; the local-filesystem path has no units, no
column groups and no fetch budget. **That is why no benchmark we had ever run
touched them**, and why all three experiments go through S3.

Everything reads and writes `s3://arrowrs-bench-21f6c795/arrow_rs_probe/`
(us-west-2) — a bucket we own, so there is no permission ambiguity and exp3 has
somewhere to write. The TPC-H input for exps 1 and 3 is copied into it once,
server-side, by `stage_data.py`; the scripts call that themselves and it is a
no-op on re-runs.

| # | Experiment | Question it answers |
|---|---|---|
| 1 | `exp1_iter_batches.sh` | Does the decode memory win exist in Ray *at all*? This is the release test verbatim — same bucket, same sf10 data, same consume mode. At 145 B/row the 2 MiB decode budget genuinely binds, so it is the best case for the premise. The release run measured **1.01×** per-task USS where the local control predicts ~0.3×. |
| 2 | `exp2_fat_col.sh` | Is the column-group branch mis-selected? `fat_col` (1 fat + 1 small column) takes it, because "a single column always fits" makes `partition_columns_by_budget` return >1 group; that branch **retains the whole decoded row group**, i.e. PyArrow's exact behaviour. `fat_col_solo` has the same bytes in one column and cannot be split, so it must take the windowed path. Same bytes, opposite branch — the pair is its own control. |
| 3 | `exp3_write_parquet.sh` | The only unambiguous per-task memory regression in the release run: **2.28× max USS at 1.00× wall time**. Time-neutral and memory-negative is the signature of accumulation. Sweeping the decode budget 2 → 32 → 128 MiB is the direct test: if USS is flat, that hypothesis is dead. |

## Setup (once)

```bash
git clone https://github.com/AarryaSaraf/ray.git ~/ray && cd ~/ray
git remote add upstream https://github.com/ray-project/ray.git
git fetch upstream master --quiet          # setup.sh picks the commit-matched wheel
git checkout arrow-rs-linux-probe
bash release/nightly_tests/dataset/arrow_rs_linux/setup.sh
```

`setup.sh` is idempotent and takes ~10–15 min (most of it compiling arrow/parquet).
It creates a Python 3.12 venv, installs the Ray nightly built from **this
branch's merge-base** (a "latest" wheel drifts from the branch's compiled
protobufs and asserts "out of sync" at import), symlinks this checkout's
`python/ray` over it, installs Rust + maturin, builds the native crate, and
verifies end to end that a read actually served through the arrow-rs path.

Then, in **every** new shell:

```bash
source ~/ray/.venv/bin/activate
```

Skipping that is the single most common way to waste a run: without it `python`
is the image's anaconda, whose Ray is the Anyscale runtime — a different read
path where this branch's reader does not exist. Every script here refuses to
start in that state rather than producing meaningless numbers.

## Run

**Current batch (2026-08-07).** One command, one output to paste back:

```bash
cd ~/ray/release/nightly_tests/dataset/arrow_rs_linux
source ~/ray/.venv/bin/activate
./run_next.sh 2>&1 | tail -n 400        # ~60-75 min
```

It builds the two fixtures, re-runs exp7 phase S against the reader defaults
shipped in `30297b9b7f` (every S3 number on record predates them), adds phase Z
(the fixed per-task cost, measured at D ~= 0 on both transports instead of
extrapolated) and phase P (does `prefetch_budget_mb` bind, on a layout where the
fetch window actually can), and re-runs exp6 phase B unfused. Each script's
header says what each outcome would mean. The summary lands in
`out/next_summary.txt` between BEGIN/END PASTE markers.

Everything, ~45-60 min, keeps going if one experiment fails:

```bash
./run_all.sh
```

or individually:

```bash
./exp1_iter_batches.sh      # ~10 min  (stages 2.84 GB of TPC-H on first run)
./exp2_fat_col.sh           # ~15 min  (writes ~1.5 GB of fixtures on first run)
./exp3_write_parquet.sh     # ~20 min  (writes a few GB per arm, then deletes it)
```

**Credentials:** on an EC2/Anyscale box the instance role covers all of this and
you need to export nothing. Every script calls `check_s3` first, which does a
real write-then-delete against the bucket and fails in seconds with the exact
fix if it can't. If credentials live only in `~/.aws/credentials`, export them —
`eval "$(aws configure export-credentials --format env)"` — because the crate's
`object_store` is built with `features = ["aws"]` and reads env vars and IMDS
but *not* the profile file, while pyarrow reads all three. That asymmetry shows
up as the arrow-rs arm alone getting `ACCESS_DENIED`.

Knobs: `OUT_DIR` (default `./out`), `NUM_CPUS` (8), `OBJECT_STORE_MB` (8192),
`S3_BUCKET` / `S3_PREFIX` / `AWS_DEFAULT_REGION`, `SF` (TPC-H scale, 10),
`DATA` (overrides the input path), `REGEN=1` (rebuild exp2 fixtures),
`MOTO=1` (run exp2 against a local moto server instead of real S3).

Both arms of every A/B are pinned to the same `num_cpus` and object-store size,
because Ray derives both from *free* RAM by default — which drifts between runs
and would silently change the comparison.

## Reading the output

Everything lands in `out/`:

| File | What it is |
|---|---|
| `exp*_summary.txt` | the A/B table — ratios are **arrow-rs ÷ pyarrow, >1.00× means worse** |
| `exp*_<reader>.json` | the raw `Benchmark` result, including per-operator memory |
| `exp2_runs.jsonl` | one JSON record per probe run |
| `prof/<tag>/*.jsonl` | the reader's own profiling, one file per pid |
| `moto.log` | the local S3 server, if exp2 misbehaves |

The columns that decide the hypotheses, all printed in the summaries:

- **`read USS samples`** — if this is `0`, per-task memory was *not collected*;
  the blank memory rows are an instrumentation gap, not flat memory. (Linux
  reports USS; macOS cannot, which is why this box exists.)
- **`column-group row groups`** and **`oversized units`** — non-zero means the
  read took the branch that retains whole row groups. Expect non-zero on
  `fat_col` and **zero** on `fat_col_solo`; if they match, the mis-selection
  theory is wrong.
- **`max retained bytes`** vs the row-group size — the direct measurement of how
  much decoded data was held at once.
- **`fetch wait s` vs `decode s`** — if fetch and decode alternate instead of
  overlapping, an oversized unit took the whole prefetch semaphore and the reads
  serialised.
- **`floored row groups`** — the 2048-row batch floor overrode the byte budget,
  so `decode_budget_bytes` did nothing. Expect this on wide rows: the budget only
  binds below ~1 KiB/row (2 MiB ÷ 2048).

## Gotchas

- **Anyscale workspaces:** never attach to the managed cluster (different Ray
  version). The scripts export `RAY_ADDRESS=local`, unset the platform's
  runtime-env hook/plugins (they kill the runtime-env agent, the raylet
  fate-shares, and `ray.init()` hangs forever with no error), and disable
  task-event reporting (a 2026-07 nightly SIGSEGVs in the aggregator flush).
  If a run hangs anyway: `cat /tmp/ray/session_latest/logs/runtime_env_agent.err`.
- **Run in us-west-2.** The bucket is there; a box in another region turns every
  GET into a cross-region round trip, which is the exact variable exp2 measures.
- **Bucket access:** `s3://ray-benchmark-data/tpch/...` is readable from an
  Anyscale account (it is the staging source, read once).
  `s3://ray-benchmark-data-internal-us-west-2/...` is **not** readable — which is
  why exp2 uses generated fixtures rather than the real wide-schema input.
- **What's left in the bucket:** the staged TPC-H copy (2.84 GB at sf10) and the
  exp2 fixtures (~1.5 GB), both reused across runs. exp3's output is deleted at
  the end of the script, by a prefix-scoped delete that refuses to touch
  anything outside `$S3_PREFIX`. To reclaim it all:
  `aws s3 rm --recursive s3://arrowrs-bench-21f6c795/arrow_rs_probe/`.
- **`MOTO=1`** runs exp2 against a local moto server. It answers the *retention*
  half of the hypothesis but not the *serialisation* half — moto on localhost has
  no latency, so an oversized unit hogging the prefetch semaphore costs nothing
  there. Also note moto holds objects in RAM (~1.5 GB for these fixtures).
