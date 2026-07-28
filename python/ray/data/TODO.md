# arrow-rs Parquet migration — TODO

Companion to [Agents.md](Agents.md) (the design/critique doc). This file tracks the
*open* work items and the decisions behind them; Agents.md records what's done and why.
Statuses: **next** (agreed, do it), **decision-needed** (blocked on an API/scope choice),
**deferred** (agreed to postpone, with the trigger that would revive it).

## Format-kwarg gaps (from the 2026-07-28 audit — currently PyArrow-fallback)

### 1. Native decryption (`decryption_properties`, `decryption_config`) — decision-needed
The Rust `parquet` crate (v59, `encryption` feature, via `ring`) fully supports read-side
Parquet modular encryption: `FileDecryptionProperties::builder().with_footer_key(...)
.with_column_key(...)`, plugged into both sync and async arrow readers. The crate half is
straightforward.

**Blocker:** pyarrow's `FileDecryptionProperties` is an opaque Cython object — `dir()`
shows zero public attributes, so the keys a user hands to `read_parquet` **cannot be
extracted** and re-passed to the crate. `decryption_config` is worse: it's a live KMS
flow (`CryptoFactory` + user `KmsClient` callbacks) that unwraps keys inside Arrow C++.
Config-faithful bridging (the standard we held for S3 creds) is impossible here.

Options:
- (a) Keep the fallback (today's state): encrypted files read correctly via PyArrow,
  they just don't get the memory win. Zero new API.
- (b) Add a Ray-side key channel (e.g. `arrow_rs_decryption_keys={"footer": b"...",
  "columns": {...}}` via `dataset_kwargs`): native decrypt works, but it's a NEW API
  that diverges from pyarrow's, and raw keys ride task args (pyarrow's object at least
  keeps them opaque). Security review needed before building this.
- Recommendation: (a) until a real encrypted-parquet user shows up; revisit with (b) + a
  KMS-unwrap hook then.

### 2. Thrift footer limits (`thrift_string_size_limit`, `thrift_container_size_limit`) — next
Agreed 2026-07-28: implement natively rather than fall back. Approach: these limits only
affect *metadata deserialization* (accept vs reject the footer), never decoded bytes. So
when they're set, the planned native read does a metadata-only footer probe through
`pq.ParquetFile(f, thrift_string_size_limit=..., thrift_container_size_limit=...)` per
file — the exact same C++ thrift parser the scanner would use, so accept/reject behavior
(and the raised `OSError`) is identical by construction — then decodes natively. Cost:
one footer read per file, only when the limits are actually set. Note this deliberately
relaxes "pyarrow never opens a supported file" for this one kwarg (probe is
metadata-only; decode stays native).

### 3. `page_checksum_verification` — feasible, small caveats
The `parquet` crate has a `crc` feature (crc32fast) that verifies page checksums during
decode. Caveats found reading parquet-59.1.0 source: (a) it's **compile-time** — once the
feature is on, the sync reader always verifies (no runtime toggle), so
`page_checksum_verification=False` (the pyarrow default) would still verify natively —
stricter, diverging only on corrupt files that pyarrow would happily read; (b) the
`#[cfg(feature = "crc")]` sites checked live in the *sync* `serialized_reader.rs` — the
async (S3) path's coverage must be verified before claiming parity. Plan: enable the
feature, keep `page_checksum_verification=True` native (semantics match), decide whether
always-verify-under-False is acceptable (probably yes — it only rejects corrupt data) or
keep False→fallback out of caution.

### 4. Option bags (`read_options`, `default_fragment_scan_options`) — deferred
Opaque pyarrow objects that can carry any of the other options inside them, so the
allowlist can't inspect what they'd change. Introspecting them is fragile
(version-dependent attribute sets). Stay on fallback; revive only if a real workload
passes them.

### 5. Schema-shaping kwargs (`binary_type`, `list_type`, `arrow_extensions_enabled`) — partially easy
- `binary_type` / `list_type` (pyarrow 21+): map cleanly onto the existing
  `_ColumnAlignment` cast machinery (e.g. `binary` → `large_binary` is a safe cast).
  One precedence trap to mirror exactly: when a file embeds an arrow schema (all
  Ray-written files do), pyarrow **ignores** these kwargs — the alignment must apply
  them only when the file lacks an embedded schema, or it would diverge in the
  opposite direction. Moderate effort, test-heavy.
- `arrow_extensions_enabled`: changes whether canonical extension types are
  reconstructed at all — interacts with the extension-metadata passthrough the crate
  already does. Deferred until someone needs it.

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
