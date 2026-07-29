# arrow-rs Parquet migration — TODO

**Context (so this file stands alone).** Ray Data's V2 Parquet reader decodes through
PyArrow, whose decode working set is the whole row group — private worker heap that
Ray's scheduler never accounts for, the cause of node OOMs on big-row-group files
(ray#49158). We built a Rust reader — the `ray_data_arrow_rs` PyO3 crate +
`ArrowRsParquetFileReader`, behind the `DataContext.use_arrow_rs_parquet_reader` flag
(V2 only) — that streams byte-budgeted decode batches, so the working set is a knob we
set instead of a property of the file. Where it stands today ([Agents.md](Agents.md) is
the full design/results doc):

- **Functionality/parity is complete for local + S3**: supported files are read
  PyArrow-free end to end (footer, statistics pruning, decode). Anything unsupported
  falls back to PyArrow *by documented decision* — the per-case rationale is
  Agents.md §7.11 — and a fallback is never a correctness risk, only a
  no-memory-win-for-that-read.
- **The deciding memory runs are in** (Linux + real S3, Agents.md §5.10): peak memory
  at-or-below PyArrow in every paired config at wall parity-to-faster; 2.6–2.8× less
  on the target layout (a lone big row group as one S3 object).
- A **strict mode** exists for validation runs: `RAY_DATA_ARROW_RS_STRICT=1` makes any
  fallback decision raise (naming the reason) instead of silently serving PyArrow
  bytes — so a correctness harness can prove the native path produced what it checked.

**The plan (agreed 2026-07-28):** correctness at scale first → optimization → re-run
the memory/timing bench → Ray release tests → present. Packaging and the default flip
ride behind those.

**Ratings.** *Ease*: how cheap to build (High = quick/low-risk). *Importance*: how much
it moves "a real memory win users can turn on" (High = critical path). *Priority*: the
agreed order; `Parked` = not now, revive-trigger in the detail.

| # | Item | Ease | Importance | Priority |
|---|------|------|-----------|----------|
| 1 | Large-scale correctness run (Linux, strict mode) | Med | **High** | **P0 — next** |
| 2 | Optimization: in-decode `RowFilter` + late materialization | Low | Med | P1 |
| 3 | Optimization: bloom-filter row-group pruning (`=`/`IN`) | Med | Med | P1 |
| 4 | Optimization: wide-string decode kernel / K-split reorder | Med | Med | P1 |
| 5 | Fix the `decode_budget=32MB` → 431 MB memory cliff | Med | Med | P1 |
| 6 | Full bench re-run: memory + timing, incl. the `oom` axis | Med | **High** | P1 (after 2–5) |
| 7 | Ray release tests + present the results | Med | **High** | P2 |
| 8 | Packaging: ship the crate (wheel or optional dep) | Low–Med | **High** | P2 |
| 9 | Default flip (`use_arrow_rs_parquet_reader=True`) | High | High | P3 (blocked on 7+8) |
| 10 | GCS / Azure filesystems | Med | Low | Parked |
| 11 | Base-V2 `arrow_parquet_args` bug (separate PR) | Med | Med | Parked (own PR) |
| 12 | Allocator A/B on Linux | High | Low | Parked (theory disproven) |
| 13 | Native support for the deliberate fallbacks | Low | Low | Parked (see §7.11) |

## Details

### 1. Large-scale correctness run — P0 (user-driven)
On a Linux machine, run the decoder-replacement correctness suite at scale with the
flag on. The harness already exists: `release/nightly_tests/dataset/arrow_rs_memtrace/`
axis `correctness` — a deterministic corpus (`corpus.py` → `corpus_v1/`: every scalar
type with pinned extremes, deep nesting, exotic encodings/compressions, pathological
layouts, int96 hint/no-hint, schema evolution, tensors, pickle) driven through 17
scenarios per reader with differential + golden + stability oracles. Two additions for
this run: set `RAY_DATA_ARROW_RS_STRICT=1` so any silent fallback **errors with its
reason** ("the run passed" then means "the native path produced every byte checked").
The mac smoke run's three findings are all closed (2026-07-28): the on-disk-`path`
column drop was ruled expected V2 behavior (golden oracle now encodes it);
`coerce_int96_timestamp_unit` now routes INT96-decoding files to the per-file
fallback (a cast provably can't reproduce decode-time coercion on pre-1970 values —
Agents.md §7.11); the pickle-object guard now runs on the native path. Everything
downstream (optimization targets, the bench re-run, the release-test pitch) keys off
this run.

### 2. In-decode `RowFilter` + late materialization — P1
Today a pushed filter prunes whole row groups by footer statistics (native), but inside
a surviving group we still decode *every* row and drop non-matching ones in Python.
The Rust parquet crate's `RowFilter` enables the two-phase decode a query engine does:
decode only the predicate column(s), evaluate, build a `RowSelection`, then decode the
remaining (wide) columns **only for surviving rows** — and skip whole pages via the
page index where present. Full design, including how it reuses the K-split's existing
`RowSelection` machinery and what it needs from the predicate IR: Agents.md §7.9.
Ease is Low because it needs an expression-evaluation layer in Rust and careful
no-over-drop testing; it pays only on selective filters (the memory win for filters
already exists without it — this is the CPU/IO win).

### 3. Bloom-filter row-group pruning — P1
Min/max statistics only prune *range* predicates on sorted/clustered data. Parquet
**bloom filters** (optional, column-chunk level) prune equality/`IN` predicates on
*unclustered* high-cardinality columns — "is `x` definitely not in this row group?"
regardless of sort order. The Rust crate already reads them (`parquet::bloom_filter`,
the same path DataFusion uses), so this extends the existing native pruning step; the
IR already carries `in`. Smaller and more self-contained than #2 — a good first
optimization PR. Caveat: writers rarely emit bloom filters by default, so gate the
bench claim on fixtures that have them.

### 4. Wide-string decode kernel / K-split reorder — P1
The one real speed deficit (Agents.md §5.7): at the 1 thread a Ray worker gets, the
crate decodes moderate strings at parity but wide strings ~1.5× slower than PyArrow.
Two attack angles: profile/optimize the wide-string kernel itself, and/or redesign the
K-split's strictly-ordered depth-2 channels into a **bounded reorder buffer** so K
ranges make real concurrent progress while still emitting in row order (today they
serialize — that's why best-tuned K wins at 2 M rows but not 8 M). Rust-only work,
benchmarkable Ray-free via `standalone_decode_bench.py`.

### 5. The 32 MB budget cliff — P1
In the reader-settings sweep, `decode_budget=32 MiB` produced a 431 MB per-worker peak
— non-linear vs 39 MB at 8 MiB (Agents.md §5.12). Suspects: the byte-budget batcher
emitting an oversized batch over a many-group chunk, or a `fetch_window`/K interaction.
Must be understood before any budget-sweep number above 8 MB is quoted, and before
tuning guidance says "raise the budget for speed."

### 6. Full bench re-run (memory + timing) — P1, after 2–5
Re-run `arrow_rs_memtrace/bench_suite.py` (all axes, Linux + real S3) once the
optimizations land, so the presented numbers include them. Two additions over the last
run: the **`oom` axis** — a memory-ceilinged node where PyArrow's hidden decode heap
gets the read OOM-killed and arrow-rs finishes — the end-to-end demonstration of the
failure mode this project removes; and cross-checking each per-task USS line against
Ray's own `TaskExecWorkerStats.max_uss_bytes` to make the metric indisputable.
Optionally fold in the parked allocator A/B (#12) as one sweep axis.

### 7. Ray release tests + presentation — P2
Wire the reader into Ray's existing release/nightly Parquet benchmarks (the harness in
`release/nightly_tests/dataset/` already models this) and assemble the presentation:
the mechanism (Agents.md §3), the deciding numbers (§5.10 + the re-run), and the
honest caveats (§5.7 decode gap, deliberate fallbacks §7.11).

### 8. Packaging — P2
The crate must reach users somehow; today it's built locally with `maturin`. Two routes:

| Route | What it takes | Trade-off |
|---|---|---|
| **In Ray's wheel** | Rust toolchain in every CI build image; cargo wired into Ray's Bazel/wheel pipeline (`rules_rust` or a bolt-on maturin step); artifacts for each platform; dependency/license review; build-infra team ownership | One wheel, nothing extra to install; heaviest org lift — cross-team infra, long review latency |
| **Separate optional package** (`ray-data-arrow-rs` on PyPI) | Own repo/CI with `maturin generate-ci` (multi-platform wheels nearly free; PyO3 abi3 = one wheel per platform, all Python versions); Ray imports it when the flag is on (already fail-loud if missing) | Days of work, no Ray-CI coupling; but the default flip then needs the package as a `ray[data]` dependency (or the flip stays opt-in) |

Recommended default: **start with the optional package** — it unblocks real users and
the release tests immediately — and treat in-wheel as the eventual default-flip
requirement, negotiated with the build-infra owners in parallel. (This is why the table
says Low–Med, not Low: the *hard* part is only the in-wheel route.)

### 9. Default flip — P3 (blocked)
Flip `use_arrow_rs_parquet_reader` to True. Trivial code; blocked on #7 (field
evidence) and #8 (the crate actually installable everywhere Ray runs). Keep the flag
one release as the escape hatch.

### 10. GCS / Azure — Parked
Feasible (the crate's `object_store` dependency ships both backends; per backend ≈ one
Cargo feature + one `_s3_config`-style config bridge), but the credential bridge is the
risk: short-lived tokens / ambient application-default credentials often don't survive
`fs.__reduce__`, and a naive bridge fails the read rather than falling back. Full
analysis: Agents.md §7.10 + §7.11. **Revive trigger:** a user with GCS/Azure OOMs;
then build fail-safe (unbridgeable config → PyArrow fallback), test on
fake-gcs-server/Azurite, validate on a real bucket.

### 11. Base-V2 `arrow_parquet_args` dead-end — Parked (separate PR)
Not an arrow-rs item; tracked so it isn't lost. Top-level `arrow_parquet_args` on
`read_parquet(...)` — the exact thing the `dataset_kwargs` deprecation message tells
users to migrate to — are stored by `ParquetDatasourceV2.__init__` and **never read
again**: V2 silently ignores them for *both* readers (V1 honors them). The fix (honor
them, or reject loudly) belongs in its own PR against the V2 datasource.

### 12. Allocator A/B — Parked (theory disproven)
The old hypothesis — that the Rust system allocator retaining freed pages explained a
many-tiny-groups loss — was **disproven by direct measurement** (Agents.md §5.10/§7.8):
`LD_PRELOAD` jemalloc moved nothing, `MALLOC_ARENA_MAX=2` was inert on the S3 sweep,
and the "loss" was an artifact of a since-retired metric. The crate's allocator build
features were removed (§6.9 — a `#[global_allocator]` in the cdylib segfaulted Ray
workers; don't reintroduce one). What remains is a cheap optional check: an
`LD_PRELOAD` A/B as one axis of the #6 re-run, no crate changes. Do it later, if at all.

### 13. Native support for the deliberate fallbacks — Parked
Encryption, `page_checksum_verification=False`, option bags,
`arrow_extensions_enabled`: each stays on the PyArrow fallback by decision, with the
full per-case rationale in Agents.md §7.11 (in one line: the fallback is correct and
identical to today's behavior; native support would buy no memory that matters and
costs real engineering/security risk). **Revive trigger per case:** a real workload
that hits it *and* suffers memory pressure on it.
