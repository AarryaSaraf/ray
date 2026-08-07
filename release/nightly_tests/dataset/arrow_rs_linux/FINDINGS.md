# arrow-rs vs PyArrow: what the Linux probes actually measured

Status as of the exp6 Phase C run. This file is the running record: what is
settled, what is retracted, and what to do next. `README.md` explains how to run
the scripts; this explains what came back.

Convention throughout: **R** = arrow-rs, **P** = PyArrow, ratios are R/P, so
`>1.00x` means arrow-rs is worse.

---

## 0. The question

A multi-node release A/B (Buildkite `release` pipeline, build **103100** =
arrow-rs, **103101** = PyArrow) came back mixed-to-negative on memory. Every
controlled local measurement we had said the opposite. These probes exist to
resolve that contradiction.

Test data throughout: TPC-H `lineitem` sf10. 16 narrow columns, ~172 B/row,
~6.0M rows and **~1.03 GiB decoded per file**. Local arms use 4 files (**4219
MiB decoded total**, 490 row groups, 49 per file at ~122,420 rows each); S3 arms
use 10.

---

## 1. Settled: the crate is not the problem

exp5 runs the identical read in a bare process with no Ray at all. The first
version of this experiment was **wrong** and its conclusion was retracted:
`use_threads=False` disables only Arrow's CPU thread pool, and the
`pyarrow.dataset` scanner keeps `fragment_readahead=4` / `batch_readahead=16`
and its separate I/O pool. PyArrow was running at **5.49 cores** while arrow-rs
ran at 1.13, so the apparent "arrow-rs is 2.3x slower" was a core-count
comparison, not a decoder comparison.

Corrected, with `resource.getrusage` CPU accounting and a serialized-scanner
control arm:

| transport | mode | reader | USS rise | ratio | wall | cpu | cores |
|---|---|---|---|---|---|---|---|
| local | read | pyarrow | 1013 MiB | baseline | 3.1 | 16.8 | 5.49 |
| local | read | pyarrow-serial | 489 MiB | 0.48x | 9.6 | 11.2 | 1.17 |
| local | read | **arrow_rs** | **23 MiB** | **0.02x** | 6.1 | 6.9 | 1.13 |
| s3 | read | pyarrow | 961 MiB | baseline | 8.5 | 20.3 | 2.39 |
| s3 | read | pyarrow-serial | 451 MiB | 0.47x | 23.9 | 15.7 | 0.66 |
| s3 | read | **arrow_rs** | **174 MiB** | **0.18x** | 8.6 | 11.7 | 1.36 |
| local | write | pyarrow / arrow_rs | 1576 / **54** MiB | 0.03x | 19.1 / 28.6 | 40.9 / 32.4 | 2.14 / 1.13 |
| s3 | write | pyarrow / arrow_rs | 1448 / **224** MiB | 0.16x | 21.6 / 24.4 | 43.9 / 39.9 | 2.03 / 1.63 |

Row parity confirmed across every arm (196 x 122,428 == 2,539 x 9,450 == 24.0M
rows).

**At equal core count arrow-rs is 1.57x faster and uses 1.62x less CPU**, and it
uses 20-40x less memory. Outside Ray, the crate is not merely acceptable, it
dominates. The concurrency sweep agrees: arrow-rs scales with threads
(11.3 -> 8.5 -> 4.9 s at concurrency 1/2/4, costing 23 -> 40 -> 78 MiB) while
PyArrow is already saturated and does not (4.6 -> 3.9 -> 4.3 s, costing
908 -> 1409 -> 1808 MiB). *(That sweep contended with a concurrent exp3 run; the
ratios are directional, the absolute wall times are not. Re-run on an idle box.)*

**Therefore: everything below is about the Ray integration, not the decoder.**

---

## 2. Settled: 32 MiB is the right decode budget, not the shipping 2 MiB

exp3 sweeps `RAY_DATA_ARROW_RS_DECODE_BUDGET_BYTES` under a real
`write_parquet`. Values are R/P ratios:

| budget | avg USS | max USS | total wall |
|---|---|---|---|
| 2 MiB (shipping default) | 1.00x | **1.47x** | 0.99x |
| **32 MiB** | **0.83x** | **1.03x** | **0.87x** |
| 128 MiB | 0.90x | 1.29x | 1.28x |

Over local disk, 32 MiB gives 0.81x / 0.81x / 0.93x. **32 MiB dominates 2 MiB on
every axis on both transports**, and exp6 Phase B later showed 32 / 64 / 128 sit
within 4% of each other, so the value is not delicate. This is the one confirmed,
shippable win and it is independent of everything in section 4.

---

## 3. Retracted hypotheses

Recorded so nobody re-runs them.

1. **"The S3 column-group path retains the whole row group."** Falsified twice:
   the profiling dump reports `col_group_rgs = 0` and `mode = row_windows`, so
   that code never executed; and exp3 over local disk shows the same ~1 GiB with
   no S3 path at all.
2. **"Per-task USS is an allocator high-water mark, so streaming cannot fix
   it."** Falsified by Phase C: USS moves substantially when task size moves. It
   is live data.
3. **"Worker reuse ratchets arrow-rs specifically."** Falsified by the max/avg
   column: at 68 tasks, arrow-rs 1.16 vs PyArrow 1.13. Both readers spread their
   tasks by the same fraction.
4. **"`override_num_blocks` sizes read tasks."** The V2 path contains no
   reference to it. The knob is `DataContext.parquet_chunker_target_chunk_size`
   (`chunkers/file_chunker.py:174`), compared against **on-disk** file size with
   a 1 GiB default -- so ordinary files are never split and each read task
   decodes one whole file.
5. **"Flat under a block-size sweep exonerates block retention."** It does not.
   If a task holds *all* of its output, the total is invariant to how that output
   is chunked, so flat is exactly what retaining-everything looks like.

---

## 4. The open finding: a fixed per-task floor, ~104 MiB larger for arrow-rs

exp6 Phase C holds the input fixed and shrinks the read task, so bytes-per-task
falls as the task count rises. `D` below is MiB decoded per task = 4219 / tasks.

| chunk (on-disk) | tasks | D (MiB) | PyArrow USS | arrow-rs USS |
|---|---|---|---|---|
| default (1 GiB) | 4 | 1055 | 1192 | **1030** |
| 256 MiB | 8 | 527 | 690 | **596** |
| 64 MiB | 20 | 211 | **386** | 489 |
| 16 MiB | 68 | 62 | **242** | 309 |

Wall time is flat across the entire sweep for both readers (12.7-13.8 s), so
chunk size costs nothing in throughput here.

Least squares over those four points:

```
PyArrow   USS ~= 183 MiB + 0.96 * D     residuals +/- 3 MiB
arrow-rs  USS ~= 287 MiB + 0.69 * D     residuals +/- 57 MiB
```

Read plainly:

* **PyArrow holds ~100% of everything its task decodes**, on a ~183 MiB
  per-worker floor (interpreter + Ray worker + Arrow libs). That is the scanner's
  whole-row-group behaviour, confirmed from outside the process.
* **arrow-rs holds ~69%** -- genuinely less -- **but on a ~104 MiB higher
  floor.**

Setting them equal gives a **crossover at D ~= 385 MiB decoded per task**. Above
that, arrow-rs wins; below it, PyArrow does. The measured table crosses between
527 and 211, exactly as predicted.

**This is the whole regression story.** Nothing here is about the decoder. It is
104 MiB of fixed cost per read task, and it decides the sign of the result
whenever tasks are small.

### Leading candidate: the reader thread pool

`file_reader.py:462` `_dispatch_fragment_reads` runs
`num_workers = min(_DEFAULT_NUM_THREADS, len(fragments))` concurrent fragments,
and `_DEFAULT_NUM_THREADS = env_integer("RAY_DATA_READ_FILES_NUM_THREADS", 4)`.
The profiler measured the crate retaining `retained_bytes / 490 = 20.4 MiB` per
row group (cross-check: 122,420 rows x 172 B = 21.1 MB). Four fragments in
flight is then ~82 MiB live at any instant **regardless of how small the task
is** -- and even the 68-task arm still has ~7 row groups, enough to keep all four
threads busy. 82 MiB against a measured 104 MiB gap is a good fit.

Note this pool is shared by both readers, so if the theory is right the same
82 MiB exists on the PyArrow side but is already folded into its 0.96 slope
rather than sitting on top of it.

### Not yet ruled out

* Rust-side fixed cost: the shared tokio runtime's thread stacks, the
  `object_store` client, per-thread allocator arenas.
* Prefetch budgets sized in absolute bytes: `fetch_window_mb` 16,
  `column_fetch_mb` 16, `prefetch_budget_mb` = 4 x max(16,16) = 64. These are
  S3-path knobs, so they should not appear in a local arm -- but they are the
  right order of magnitude and worth confirming rather than assuming.

### Things this floor is provably NOT

* Not output blocks: `target_max_block_size` moved 32x (16 -> 512 MiB) and
  per-task USS moved 1.00-1.01x.
* Not the decode budget: 8 -> 128 MiB moved it 1.04x.
* Not the object store: the probe consumes via `iter_internal_ref_bundles()`
  without materializing.

### Why this cannot be fixed by shrinking tasks

Smaller chunks lower node memory for both readers -- but they lower PyArrow's
faster, and hand it the win. Concurrency here is ~8 (one task per CPU), so node
memory is per-task USS x 8: at 256 MiB chunks that is 4.8 GiB for arrow-rs vs
5.5 for PyArrow; at 16 MiB it is 2.5 vs 1.9. Chunk size is a mitigation for
*absolute* memory and an *anti*-fix for the ratio.

*(Disregard any earlier `USS x tasks` column. It sums over sequential tasks,
which is not a real quantity -- what matters for OOM is per-task USS times
concurrent tasks.)*

---

## 5. Queue

Ordered. Everything in section 4 is one question, and P0 answers it.

| # | What | Why now | Cost |
|---|---|---|---|
| **P0** | **exp6 Phase D: sweep `RAY_DATA_READ_FILES_NUM_THREADS` = 1/2/4/8 at `--chunk-mib 16`, both readers** | Directly tests the ~82 MiB thread-pool theory against the ~104 MiB measured floor. If arrow-rs at 1 thread drops toward PyArrow, the floor is identified and the fix is a per-reader thread default. If it does not move, the floor is Rust-side and the search moves into the crate. | ~15 min |
| **P0b** | Re-run the concurrency sweep on an idle box | The existing numbers contended with a live exp3; ratios survive, absolute wall times do not. | ~10 min |
| **P1** | **Ship the decode-budget default: 2 MiB -> 32 MiB** (`arrow_rs_parquet_file_reader.py`) | Section 2. Confirmed on both transports, insensitive to the exact value, independent of P0. | small patch |
| **P2** | Decide the reader thread default, gated on P0 | If P0 confirms, arrow-rs wants a different `num_threads` than PyArrow -- it scales where PyArrow is saturated (section 1), so the answer may be *more* threads with a smaller per-thread retention, not fewer. | depends |
| **P3** | exp4 read-speed run | The 1.82x read slowdown is now *expected* to be the core-count asymmetry from section 1 rather than a real regression. Confirm and close it. | ~15 min |
| **P4** | Experiment D: read fused with a heavy actor `map_batches` | The actual OOM shape. The release suite never covers it -- its map cases force `use_datasource_v2=False`, which makes them controls, and two controls disagreeing 2.5x is the autoscaling noise floor. | new script |
| **P5** | Docs: metric definitions | Five different things get called "memory" (process-tree peak RSS; per-worker USS self-sampled; Ray `max_uss_per_task`, which is worker-reuse contaminated; `object_store_memory_used_peak_gb`; node `memory_usage` at 60 s effective resolution). Relabel headlines as *worker private memory*. Fix the wrong `read_from_uris` and `read_large_parquet` rows in `arrow_rs_vs_pyarrow_release_comparison.md`; mark the four `wide_schema_*` rows unverified. | ~1 h |
| **P6** | One multi-node release re-run | Only worth spending once P0-P2 give a **numeric prediction** to test. | expensive |

### PR hygiene (branch `arrow-rs-parquet-reader-pr`, draft #65117)

* Cherry-pick the S3 env-credential fix (`native_metadata.py` + regression test)
  from `faf68b1241`.
* Decide whether the env-gated profiling instrumentation (~289 lines,
  `RAY_DATA_ARROW_RS_PROFILE`) ships or is dropped.
* Drop the six TESTING commits (`2d0fa73673`, `e7a41e3b74`, `c17f8837bd`,
  `ded173bf57`, `83cbfaa630`, `1b6c781216`) and reword `a7c580e12c` before
  draft -> ready.

---

## 6. Reference: knobs and where they live

| knob | default | file |
|---|---|---|
| `DataContext.parquet_chunker_target_chunk_size` | 1 GiB, vs **on-disk** size | `chunkers/file_chunker.py:174` |
| `DataContext.target_max_block_size` | 128 MiB | `context.py` |
| `RAY_DATA_READ_FILES_NUM_THREADS` | 4 | `readers/file_reader.py:44` |
| `RAY_DATA_ARROW_RS_DECODE_BUDGET_BYTES` | 2 MiB (**should be 32**) | `readers/arrow_rs_parquet_file_reader.py:190` |
| `RAY_DATA_ARROW_RS_K` | 1 | same |
| crate split threshold | 128 MiB, and it only applies to a **lone** row group -- it never engages on multi-row-group files like lineitem | same |
| `fetch_window_mb` / `column_fetch_mb` / `prefetch_budget_mb` | 16 / 16 / 64 (S3 path only) | crate |

Per-read overrides use `dataset_kwargs={"arrow_rs_*": ...}`; precedence is
kwarg > env > default.

Gotcha worth keeping: `ds.write_parquet()` executes an *internal* dataset stored
at `ds._write_ds` (`dataset.py:6153`), and only the **public**
`ds.get_stats_summary()` knows to look there (`dataset.py:7361`). The private
`_get_stats_summary()` reports the un-executed outer plan and returns zeros --
silently, because "no samples" and "no memory" render identically. That bug cost
one full 20-minute exp6 run.
