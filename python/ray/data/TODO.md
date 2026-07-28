# arrow-rs Parquet migration — TODO

Companion to [Agents.md](Agents.md) (the design/critique doc). This file tracks the
*open* work items and the decisions behind them; Agents.md records what's done and why.
Statuses: **next** (agreed, do it), **decision-needed** (blocked on an API/scope choice),
**deferred** (agreed to postpone, with the trigger that would revive it).

## Format-kwarg gaps (from the 2026-07-28 audit — currently PyArrow-fallback)

### 1. Native decryption (`decryption_properties`, `decryption_config`) — DECIDED 2026-07-28: keep the fallback
The Rust `parquet` crate (v59, `encryption` feature) fully supports read-side decryption,
but pyarrow's `FileDecryptionProperties` is an opaque Cython object (zero public attrs —
keys cannot be extracted) and `decryption_config` is a live KMS callback flow inside
Arrow C++, so config-faithful bridging is impossible. Decision: encrypted files keep the
PyArrow fallback (correct, just no memory win). Revive trigger: a real encrypted-parquet
user — then design a Ray-side key channel (`arrow_rs_decryption_keys` via
`dataset_kwargs` + a KMS-unwrap hook) with a security review (raw keys would ride task
args).

### 3. `page_checksum_verification` — DONE 2026-07-28
Crate rebuilt with parquet's `crc` feature: page CRCs are now verified during decode on
BOTH the sync/local and async/S3 paths (both funnel through `decode_page` in
`serialized_reader.rs`; verified by corrupt-page test). The feature is compile-time
(no runtime off-switch), so the gate is value-sensitive: `True` → native (semantics
match, corrupt page raises from both readers — parity-of-error), explicit `False` →
PyArrow fallback (the opt-out for reading *despite* corrupt checksums, which only
PyArrow can honor). Unset → native, which now verifies where pyarrow wouldn't —
divergence only on corrupt data, in the safe direction (reject instead of returning
garbage). Pyarrow doesn't write page CRCs by default, so most files are unaffected.
Pinned by `test_page_checksum_verification_true_native` / `_false_falls_back`.
**Found & fixed a base-V2 bug along the way** (`_arrow_scanner_kwargs`): the reader's
hardcoded `ParquetFragmentScanOptions` REPLACED the format's scan options wholesale,
silently dropping every scan-level `dataset_kwargs` option (`page_checksum_verification`,
user `pre_buffer`/`buffer_size`/`cache_options`, scan-time decryption). User values are
now merged over Ray's tuned defaults (`_FRAGMENT_SCAN_OPTION_KEYS`).

### 4. Option bags (`read_options`, `default_fragment_scan_options`) — deferred
Opaque pyarrow objects that can carry any of the other options inside them, so the
allowlist can't inspect what they'd change. Introspecting them is fragile
(version-dependent attribute sets). Stay on fallback; revive only if a real workload
passes them.

### 5. Schema-shaping kwargs (`binary_type`, `list_type`, `arrow_extensions_enabled`) — binary/list DONE 2026-07-28
- `binary_type` / `list_type`: implemented for the pinned-schema case (the only case
  the V2 pipeline produces). Empirical finding that made it trivial: the pinned unified
  schema — inferred by the listing via `pq.read_schema`, which is blind to these kwargs —
  is the output-type authority, and the base reader's pinned-schema cast **silently
  undoes** the kwargs (pyarrow 24, verified). So parity = "output the pinned schema",
  which the alignment's existing drift casts already guarantee → the kwargs are simply
  admitted when `self._file_dataset_schema is not None`, zero new cast machinery.
  Without a pinned schema they genuinely change output types (`binary`→`large_binary`,
  `string`→`large_string`, `list`→`large_list` on no-embedded-schema files) → still
  fallback. Pinned by `test_schema_shaped_kwargs_native_with_pinned_schema` /
  `_fall_back_without_schema`.
- `arrow_extensions_enabled`: changes whether canonical extension types are
  reconstructed at all — interacts with the extension-metadata passthrough the crate
  already does. Deferred until someone needs it (now the fallback exemplar in
  `test_unsupported_format_kwarg_falls_back`).

## Migration phases (from the 6-phase plan)

- **Phase 3 — GCS/Azure filesystems**: `object_store` already vendors `gcp`/`azure`
  features; per backend ≈ one Cargo feature + a `_gcs_config`/`_azure_config` bridge via
  `fs.__reduce__` + gate admission. Real risk is credentials (short-lived tokens/ADC
  don't round-trip through `__reduce__`) — bridge must be fail-safe (unrecoverable
  config → None → PyArrow fallback), emulator tests (fake-gcs-server / Azurite), then
  real-bucket validation.
- **Phase 5 — packaging/CI wheels**: the long pole for any default flip. The crate must
  build in Ray's CI for manylinux/macos and ship in the wheel (or as an optional dep).
  Nothing designed yet.
- **Phase 6 — default flip**: `use_arrow_rs_parquet_reader=True` by default. Blocked on
  phase 5 + the Linux/real-S3 deciding run (below).

## Other open items

- **Linux + real-S3 deciding run**: `bench_suite.py s3` on Linux against real S3 — the
  authoritative memory measurement (macOS numbers are directional; USS is
  Linux-only). Sweeps fetch_window × MALLOC_ARENA_MAX; validates the windowed-async
  port and the jemalloc feature.
- **In-decode `RowFilter`**: crate currently prunes row groups by stats and filters
  rows post-decode in Python; a native RowFilter would skip decoding non-matching rows.
  Deferred: it's a speed win we don't currently need — memory is the goal, and the
  post-decode filter is the correctness authority either way.
- **Base-V2 `arrow_parquet_args` dead-end (separate PR, not arrow-rs)**: top-level
  parquet args on `read_parquet` — exactly what the `dataset_kwargs` deprecation message
  tells users to pass — are stored in `ParquetDatasourceV2.__init__` and never read
  again; V2 silently drops them for BOTH readers. Fix (honor or reject loudly) needs its
  own PR against the V2 datasource.
- **Bench anomaly**: `decode_budget=32MB` → 431 MB per-worker peak (non-linear cliff
  from 39 MB at 8 MB budget) in the reader-settings axis. Suspect oversized byte-budget
  batch over a 30-group chunk, or fetch_window/K interaction. Investigate before quoting
  budget-sweep numbers above 8 MB.
- **Allocator retention (§7.8)**: jemalloc crate feature + `MALLOC_ARENA_MAX=2` worker
  env are built but unvalidated on Linux (mimalloc segfaulted and was removed). Validate
  in the deciding run.
