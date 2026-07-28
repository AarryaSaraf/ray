# arrow-rs Parquet reader — setup & run commands (the runbook)

Everything operational for the arrow-rs reader lives here: bringing up a fresh
machine (macOS dev box or the Linux measurement box), building the native crate,
running the tests, and every benchmark/harness command. The design, results, and
rationale live in [Agents.md](Agents.md); this file is only *how to run things*.

---

## 1. Fresh machine, one command

`release/nightly_tests/dataset/arrow_rs_memtrace/setup.sh` brings a fresh box
(Linux x86-64 or macOS/arm64) to "the suite runs":

```bash
bash release/nightly_tests/dataset/arrow_rs_memtrace/setup.sh
```

It is idempotent and does, in order: uv + a Python 3.12 venv → a **commit-matched**
Ray nightly wheel + `setup-dev.py` symlink (so this branch's reader source is what
runs — a "latest" wheel drifts from the branch's compiled protobufs and asserts
"out of sync" at import) → rustup + maturin → `maturin develop --release` on the
crate → harness deps → an end-to-end verify that the arrow-rs path actually served
a read. Knobs: `RAY_VENV`, `RAY_WHEEL_URL`, `SKIP_RAY=1`, `SKIP_APT=1`,
`SKIP_CRATE=1` (see the script header).

Two facts to know before picking the box:

- The default wheel URL is **x86-64 manylinux** — an arm64 Linux box needs
  `RAY_WHEEL_URL` overridden (and re-verifying the crate build there).
- For S3 axes, put the box in the **same region as the bucket** (the harness
  measures memory, not your cross-region latency) and export AWS creds in the env.
- Disk: corpus + generated fixtures + run artifacts grew to **~26 GB** on the mac
  box; give the machine ≥100 GB free.

## 2. Anyscale-workspace pitfalls (learned 2026-07-27, cost a day)

Running a from-source master branch on a managed workspace image (2.56.1 at the
time) hits three traps; none are arrow-rs-related:

1. **Never attach to the managed cluster** — it is a different Ray version and
   fails the version check. Run a private local cluster: `export RAY_ADDRESS=local`
   (and if starting a head manually, use a non-default port — the managed cluster
   owns 6379). `setup.sh`'s verify step and the harness both assume this.
2. **`aiohttp` must be in the venv** — the `ray[data]` extra omits it, the
   runtime-env agent crashes on import, and the raylet fate-shares with it: the
   symptom is `ray start` hanging indefinitely. `setup.sh` now installs it.
3. **`RAY_task_events_report_interval_ms=0`** dodged a core task-event-aggregator
   SIGSEGV on the 2026-07 master commit. Only set it if workers segfault in the
   event aggregator; it disables task-event reporting.
4. **Unset the platform runtime-env extensions** (learned 2026-07-28, a different
   workspace image): the image injects the platform's cgroup module either as a
   driver hook (`RAY_RUNTIME_ENV_HOOK`) or as agent plugins
   (`RAY_RUNTIME_ENV_PLUGINS`). The module isn't in our venv; with the *plugins*
   variant the runtime-env agent dies on import
   (`ModuleNotFoundError: cgroup_runtime_plugin` in `runtime_env_agent.err`), the
   raylet fate-shares, and `ray.init()` hangs forever. `setup.sh`'s verify and
   `bench_suite.py` now pop both env vars in-process; for ad-hoc scripts,
   `unset RAY_RUNTIME_ENV_HOOK RAY_RUNTIME_ENV_PLUGINS` first. Diagnosis pattern
   for any such hang: `cat /tmp/ray/session_latest/logs/runtime_env_agent.err`.

## 3. Building the crate manually

```bash
cd python/ray/data/_internal/datasource_v2/native/ray_data_arrow_rs
maturin develop --release
# On a uv-managed venv (PEP 668 blocks `develop`'s pip step), build + install instead:
#   maturin build --release && uv pip install --force-reinstall --no-deps target/wheels/*.whl
```

No allocator build features exist anymore (Agents.md §6.9/§7.8: the retention
theory was disproven and a `#[global_allocator]` in the cdylib segfaulted Ray
workers); allocator A/B if ever needed = `LD_PRELOAD` a `.so`, or
`MALLOC_ARENA_MAX=2` in the worker env.

## 4. Tests

```bash
# Correctness + order parity (skips if the native module isn't importable).
# The S3 tests also need `moto[server]` on PATH (else they skip):
uv pip install "moto[server]"   # once, for the S3 tests
pytest python/ray/data/tests/datasource/test_arrow_rs_parquet_reader.py -v

# Rust unit tests (no Python involved):
cd python/ray/data/_internal/datasource_v2/native/ray_data_arrow_rs
cargo test --no-default-features
```

## 5. The large correctness run (TODO item 1 — the P0)

The decoder-replacement corpus + scenario suite is the `correctness` axis of the
memtrace harness (fixture builders in `corpus.py`; three oracles per scenario —
differential pyarrow-vs-arrow-rs, golden rows recomputed from the deterministic
builders, and chunked-vs-whole stability):

```bash
cd release/nightly_tests/dataset/arrow_rs_memtrace
export RAY_ADDRESS=local
python bench_suite.py correctness   # builds the corpus idempotently on first run,
                                    # then every scenario × both readers →
                                    # runs/results_correctness.json
python summarize.py                 # the parity/golden/stability digest
python task_mem.py corr__           # per-scenario worker-USS-over-time figures
# Subset while iterating: RAY_DATA_CORR_SCENARIOS=int96_coerce_ms,mega_main python bench_suite.py correctness
```

The corpus lands under `data/corpus_v1/` by default; to build/read it on S3
instead (exercising the crate's S3 path), export
`RAY_DATA_ARROW_RS_FIXTURE_ROOT=s3://<bucket>/<prefix>` **before** the run, with
AWS creds/region in the env.

Two caveats: the correctness axis is **separate from the perf axes** — run it
independently of `size`/`schema`/`showcase`; and read its wall/USS numbers as
diagnostic only (scenarios call `take_all()`, so they are not headline
speed/memory figures).

**On `RAY_DATA_ARROW_RS_STRICT=1`:** strict mode makes *any* fallback decision
raise instead of silently serving PyArrow bytes — but some corpus scenarios fall
back **by design** (e.g. the us-hint int96 file; rationale Agents.md §7.11), and
under strict those surface as error rows / parity DIFFs — expected and ignorable
for those scenarios. Run the axis normally (the per-scenario `fallback` count
already proves which path ran), and use strict for targeted checks where a read
is *expected* fully native — e.g. re-running a single scenario or a
production-shaped read where a silent fallback would invalidate the result.

**Known open divergence to re-check on this run:**
`coerce_int96_timestamp_unit="ms"` — pyarrow coerces to ms, the native path
appeared to honor the file's embedded per-column hints instead (TODO item 1).

## 6. Benchmarks

```bash
# One config, RSS-over-time (macOS-era harness; flip the reader via --readers):
RAY_DATA_USE_DATASOURCE_V2=1 python \
  release/nightly_tests/dataset/arrow_rs_read_benchmark.py \
  --readers pyarrow arrow_rs --consume iter_batches \
  --rows 8000000 --num-files 1 --row-group-size 8000000 \
  --str-cols 3 --str-width 48 --num-cpus 4
```

Consume modes: `iter_batches` (realistic), `decode_drop` (isolate decode
CPU/transient), `sum`, `materialize` (retained blocks). Knobs are env-driven at
worker import — the full list with defaults is Agents.md §8; the ones you'll touch
most: `RAY_DATA_ARROW_RS_DECODE_BUDGET_BYTES`, `RAY_DATA_ARROW_RS_K`,
`RAY_DATA_ARROW_RS_FETCH_WINDOW_MB` (S3 in-flight compressed bytes; the S3 memory
knob), `RAY_DATA_ARROW_RS_PREFETCH_WINDOWS` (S3 intra-unit look-ahead depth).

**Memory-over-time + the expanded suite** live in
`release/nightly_tests/dataset/arrow_rs_memtrace/`:

```bash
cd release/nightly_tests/dataset/arrow_rs_memtrace
# Per-worker private-heap (USS) over time, both readers overlaid (§3 figures):
python mem_over_time.py && python plot_mem.py
# The axes (§5.5–5.6): layout, schema coverage+parity, budget tuning, leak check,
# mixed-schema, scaling (O(n) proof), concurrency (single-node overcommit).
# Emits runs/results_<axis>.json; USS is sampled every 5 ms on EVERY run.
# S3 memory/speed (§5.9) — Linux + real S3 only; skips locally. ONE command does
# everything: generate fixtures on S3 (N files x one big row group, page index) →
# sweep PyArrow baseline vs arrow-rs across fetch_window_mb ∈ {4,16,64,0} and
# MALLOC_ARENA_MAX on/off → write figs/s3_mem_time.png + figs/s3_speed_time.png +
# a summary table. Needs AWS creds in env + the crate built/installed:
#   RAY_DATA_ARROW_RS_S3_BENCH_PATH=s3://bucket/prefix python run_s3_benchmark.py
# Scale via RAY_DATA_ARROW_RS_S3_BENCH_{ROWS,NUM_FILES,SCHEMA}. (bench_suite.py s3
# runs just the sweep against an existing RAY_DATA_ARROW_RS_S3_BENCH_PATH.)
# or one-variable sweeps (§5.0): bench_suite.py sweep_size,sweep_rowgroup,sweep_schema,
#   sweep_files,sweep_batch,sweep_batch_dd,workloads
python bench_suite.py            # runs all axes if no arg; every axis logs per-task
                                 # windows (tasks_*.csv) + USS (uss_*.csv) + meta.json
python summarize.py              # per-axis tables + THE memory graphs: per-task USS
                                 # vs the expected-without-decode line, one per config
                                 # (figs/task_mem/) + the §5.0 sweep gallery
python task_mem.py [substr]      # just the per-task graphs (optionally filtered)

# Postmortem / drill-down instruments (§3.5.1, §5.10):
python inspect_run.py runs/<dir> --plot   # per-worker base/peak/delta + USS-over-time
python micro_alloc_probe.py data/<fixture> [budget_mb]   # Ray-free crate-vs-pyarrow A/B
                                 # (modes incl. LD_PRELOAD allocator + import-cost)

# Ray-FREE raw-decode bench (§5.7) — iterate on the crate without Ray overhead;
# reports pyarrow at 1 thread (Ray-representative) AND full threads:
python standalone_decode_bench.py --rows 8000000 --schema huge_str --budgets 8,32 --ks 1,4
```

`bench_suite.py` sets `RAY_DATA_ARROW_RS_PATH_TRACE=<dir>`, which makes the reader
append `native`/`fallback` per fragment to `path_<pid>.log` in that dir — the
harness reads it to assert which path the support gate chose. The knob is inert
unless the env var is set.
