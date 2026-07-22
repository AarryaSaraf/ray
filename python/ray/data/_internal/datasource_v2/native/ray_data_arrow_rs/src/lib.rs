//! Experimental arrow-rs Parquet reader for Ray Data (PyO3 extension).
//!
//! RECONSTRUCTION STATUS
//! ---------------------
//! Reconstructed from the surviving standalone benchmark (`main.rs`). The local
//! path (`read_row_groups`) ports two of `main.rs`'s modes:
//!
//! 1. **Byte-budgeted per-group streaming** (`row_group_loop_bb`, threads==1):
//!    read one row group at a time with a batch size computed *by bytes* from the
//!    footer (`byte_budget_rows`), so the decoded working set stays flat across
//!    schemas (wide-string groups get few rows/batch, numeric groups many). A
//!    single reader streams each group in file order and drops each batch, so peak
//!    memory ~= one budget while row order is preserved.
//!
//! 2. **Intra-fragment K-split** (`build_units` / `read_range_fixed`): when a call
//!    covers a *single* row group larger than `split_threshold_bytes`, split its
//!    rows into K contiguous ranges decoded by K threads and merge them back in
//!    range order (`ParallelRangeReader`). This is the case Ray can't parallelize
//!    (a big row group is a lone fragment → thread pool of 1), so PyArrow decodes
//!    it ~single-threaded; K gives us parallel decode without regressing speed.
//!    Multi-row-group / small-row-group calls use path 1 (K=1) because Ray's
//!    fragment thread pool already parallelizes those — so the two parallelism
//!    layers never multiply.
//!
//! Unlike `main.rs` (which only sums a commutative checksum, so range order is
//! irrelevant), we return real data, so the K-split merge is strictly order
//! preserving: one bounded channel per range, drained in range order.
//!
//! The S3 path (`read_row_groups_s3`) ports `main.rs`'s windowed-async reader
//! (`read_all_async` / `read_unit_windowed`) but tuned **memory-first**: peak RSS
//! is `≈ (fetch window compressed) + (decode budget)`, both knobs, flat regardless
//! of row-group size — not `main.rs`'s always-K-way fan-out (which multiplies
//! in-flight memory by K for speed). We fan out to K concurrent GET streams ONLY
//! for a lone row group above `split_threshold_bytes` (the case Ray's fragment
//! pool can't parallelize) — exactly mirroring the local K-split rule — so crate-K
//! and Ray's 4-thread pool never multiply. Every other layout is a single windowed
//! stream (K=1). Output is order-preserving (per-unit channels drained in order),
//! and the decode batch is byte-budgeted just like the local path.
//!
//! Public API (consumed by `ArrowRsParquetFileReader` on the Python side via
//! `pa.RecordBatchReader.from_stream(...)`):
//!
//!   read_row_groups(path, row_groups=None, columns=None, batch_size=131072,
//!                   decode_budget_bytes=2*1024*1024, k=1,
//!                   split_threshold_bytes=128*1024*1024)
//!   read_row_groups_s3(bucket, key, region, anonymous, endpoint=None, ...creds...,
//!                      row_groups=None, columns=None, batch_size=131072,
//!                      decode_budget_bytes=2*1024*1024, fetch_window_mb=16, k=1,
//!                      split_threshold_bytes=128*1024*1024)

use std::fs::File;
use std::sync::mpsc::{sync_channel, Receiver};
use std::thread;

use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use arrow::error::ArrowError;
use arrow::ffi_stream::FFI_ArrowArrayStream;
use arrow::record_batch::RecordBatchReader;
use parquet::arrow::arrow_reader::{
    ArrowReaderMetadata, ArrowReaderOptions, ParquetRecordBatchReader,
    ParquetRecordBatchReaderBuilder, RowSelection, RowSelector,
};
use parquet::arrow::async_reader::{ParquetObjectReader, ParquetRecordBatchStreamBuilder};
use parquet::arrow::ProjectionMask;
use parquet::errors::ParquetError;
use parquet::file::metadata::PageIndexPolicy;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use std::ffi::CString;

use futures::StreamExt;
use object_store::aws::AmazonS3Builder;
use object_store::path::Path as ObjPath;
use object_store::ObjectStore;
use std::sync::{Arc, OnceLock};
use tokio::sync::mpsc;

// NOTE on allocators: earlier prototypes carried optional mimalloc/jemalloc
// global-allocator features to chase a suspected allocator-retention gap vs
// PyArrow. Measurement killed the theory (jemalloc LD_PRELOAD inert, per-worker
// high-water lower than PyArrow's on the same fixtures), and mimalloc as a
// cdylib global allocator segfaulted Ray workers across the Arrow C-stream FFI
// boundary. Both features were removed to keep the dependency tree minimal;
// the system allocator is correct here. A/B experiments can still use
// LD_PRELOAD without recompiling.

// --------------------------------------------------------------------------- //
// Shared tokio runtime
// --------------------------------------------------------------------------- //
/// One process-wide multi-thread runtime, lazily built, shared by every
/// `read_row_groups_s3` call. Previously each fragment read built and tore down
/// its own 2-thread runtime — churn that scales with the file count. The async
/// work is IO-bound (awaiting range GETs), so a small fixed worker pool drives
/// many concurrent fetches. `worker_threads(4)` matches Ray's per-worker fragment
/// pool so decode never oversubscribes cores: either 4 fragments × K=1 unit, or
/// 1 lone fragment × K units — never both at once.
fn shared_runtime() -> &'static tokio::runtime::Runtime {
    static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .enable_all()
            .build()
            .expect("build shared tokio runtime")
    })
}

// --------------------------------------------------------------------------- //
// Byte-budget batch sizing (ported from main.rs `byte_budget_rows`)
// --------------------------------------------------------------------------- //
/// Choose a batch row count so `rows * bytes_per_row ~= budget_bytes`, using the
/// row group's uncompressed size / row count from the footer. `requested` is the
/// upper clamp (a narrow schema never grows past the caller's ask) and 2048 is the
/// lower clamp (a very wide schema never collapses to a pathologically tiny batch).
/// This is what keeps the decoded working set flat across schemas.
fn byte_budget_rows(
    uncompressed_bytes: i64,
    num_rows: i64,
    requested: usize,
    budget_bytes: u64,
) -> usize {
    if num_rows <= 0 {
        return requested;
    }
    let bpr = (uncompressed_bytes as f64 / num_rows as f64).max(1.0);
    let budget_rows = (budget_bytes as f64 / bpr) as usize;
    budget_rows.clamp(2048, requested.max(2048))
}

// --------------------------------------------------------------------------- //
// Column projection helper
// --------------------------------------------------------------------------- //
/// Build a leaf-column ProjectionMask from column names using the parquet schema
/// descriptor. Names not present are ignored (Python already resolved the read
/// set). Flat schemas only — the Python `_arrow_rs_supported` gate rejects nested
/// columns before we get here.
fn projection_mask(
    parquet_schema: &parquet::schema::types::SchemaDescriptor,
    columns: &Option<Vec<String>>,
) -> ProjectionMask {
    match columns {
        None => ProjectionMask::all(),
        Some(names) => {
            let root = parquet_schema.root_schema();
            let mut indices = Vec::new();
            for (i, f) in root.get_fields().iter().enumerate() {
                if names.iter().any(|n| n == f.name()) {
                    indices.push(i);
                }
            }
            ProjectionMask::roots(parquet_schema, indices)
        }
    }
}

/// Probe the projected output schema with an empty (zero row group) reader, so
/// `schema()` is available to the FFI stream before any batch is pulled.
fn probe_schema(
    path: &str,
    meta: &ArrowReaderMetadata,
    mask: &ProjectionMask,
) -> Result<SchemaRef, ParquetError> {
    Ok(
        ParquetRecordBatchReaderBuilder::new_with_metadata(File::open(path)?, meta.clone())
            .with_projection(mask.clone())
            .with_row_groups(vec![])
            .build()?
            .schema(),
    )
}

// --------------------------------------------------------------------------- //
// Arrow C-stream wrapper returned to Python
// --------------------------------------------------------------------------- //
/// Holds an FFI stream until Python pulls it out via `__arrow_c_stream__`.
#[pyclass]
struct ArrowStream {
    inner: Option<FFI_ArrowArrayStream>,
}

#[pymethods]
impl ArrowStream {
    /// PyCapsule protocol: PyArrow's `RecordBatchReader.from_stream` calls this.
    #[pyo3(signature = (_requested_schema=None))]
    fn __arrow_c_stream__<'py>(
        &mut self,
        py: Python<'py>,
        _requested_schema: Option<PyObject>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        let stream = self
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("stream already consumed"))?;
        let name = CString::new("arrow_array_stream").unwrap();
        PyCapsule::new_bound(py, stream, Some(name))
    }
}

fn into_py_stream(reader: Box<dyn RecordBatchReader + Send>) -> ArrowStream {
    ArrowStream {
        inner: Some(FFI_ArrowArrayStream::new(reader)),
    }
}

// --------------------------------------------------------------------------- //
// Local read (sync): per-group byte-budgeted sequential reader (K=1 path)
// --------------------------------------------------------------------------- //
/// Streams the selected row groups in order, building one `ParquetRecordBatchReader`
/// per group with a byte-budgeted batch size. Row order is preserved (single
/// reader, groups in ascending order) and peak memory stays ~one decode budget
/// because each batch is dropped as Python pulls the next.
struct RowGroupSeqReader {
    path: String,
    meta: ArrowReaderMetadata,
    mask: ProjectionMask,
    budget_bytes: u64,
    batch_clamp: usize,
    row_groups: Vec<usize>,
    pos: usize,
    current: Option<ParquetRecordBatchReader>,
    schema: SchemaRef,
}

impl RowGroupSeqReader {
    fn new(
        path: String,
        meta: ArrowReaderMetadata,
        mask: ProjectionMask,
        schema: SchemaRef,
        row_groups: Vec<usize>,
        batch_clamp: usize,
        budget_bytes: u64,
    ) -> Self {
        Self {
            path,
            meta,
            mask,
            budget_bytes,
            batch_clamp,
            row_groups,
            pos: 0,
            current: None,
            schema,
        }
    }

    fn build_group_reader(&self, rg: usize) -> Result<ParquetRecordBatchReader, ParquetError> {
        let rgm = self.meta.metadata().row_group(rg);
        let eff = byte_budget_rows(
            rgm.total_byte_size(),
            rgm.num_rows(),
            self.batch_clamp,
            self.budget_bytes,
        );
        ParquetRecordBatchReaderBuilder::new_with_metadata(
            File::open(&self.path)?,
            self.meta.clone(),
        )
        .with_batch_size(eff)
        .with_row_groups(vec![rg])
        .with_projection(self.mask.clone())
        .build()
    }
}

impl Iterator for RowGroupSeqReader {
    type Item = Result<RecordBatch, ArrowError>;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(reader) = self.current.as_mut() {
                match reader.next() {
                    Some(batch) => return Some(batch),
                    None => self.current = None,
                }
            }
            if self.pos >= self.row_groups.len() {
                return None;
            }
            let rg = self.row_groups[self.pos];
            self.pos += 1;
            match self.build_group_reader(rg) {
                Ok(reader) => self.current = Some(reader),
                Err(e) => return Some(Err(ArrowError::ExternalError(Box::new(e)))),
            }
        }
    }
}

impl RecordBatchReader for RowGroupSeqReader {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

// --------------------------------------------------------------------------- //
// Local read (sync): intra-fragment K-split for one big row group
// --------------------------------------------------------------------------- //
/// Splits one row group's rows into K contiguous ranges decoded by K threads,
/// merging them back in range order so output row order matches a sequential read.
/// Each range has its own bounded channel (backpressure), and the consumer drains
/// channels in ascending range order — so at most `k * channel_depth` batches are
/// resident and rows come out in file order. Requires the offset/page index so a
/// `RowSelection` fetches only its range's pages (else each worker would decode the
/// whole column chunk); the caller checks this before choosing this path.
struct ParallelRangeReader {
    schema: SchemaRef,
    receivers: Vec<Receiver<Result<RecordBatch, ArrowError>>>,
    cur: usize,
}

fn build_range_reader(
    path: &str,
    meta: &ArrowReaderMetadata,
    mask: &ProjectionMask,
    rg: usize,
    start: usize,
    len: usize,
    batch: usize,
) -> Result<ParquetRecordBatchReader, ParquetError> {
    let sel = RowSelection::from(vec![RowSelector::skip(start), RowSelector::select(len)]);
    ParquetRecordBatchReaderBuilder::new_with_metadata(File::open(path)?, meta.clone())
        .with_row_groups(vec![rg])
        .with_row_selection(sel)
        .with_batch_size(batch)
        .with_projection(mask.clone())
        .build()
}

impl ParallelRangeReader {
    fn spawn(
        path: String,
        meta: ArrowReaderMetadata,
        mask: ProjectionMask,
        schema: SchemaRef,
        rg: usize,
        total_rows: usize,
        k: usize,
        batch: usize,
    ) -> Self {
        let chunk = total_rows.div_ceil(k.max(1)).max(1);
        let mut receivers = Vec::new();
        let mut start = 0usize;
        while start < total_rows {
            let len = chunk.min(total_rows - start);
            // Depth 2: a worker may run one batch ahead of the consumer, no more.
            let (tx, rx) = sync_channel::<Result<RecordBatch, ArrowError>>(2);
            receivers.push(rx);
            let (path, meta, mask) = (path.clone(), meta.clone(), mask.clone());
            thread::spawn(move || {
                match build_range_reader(&path, &meta, &mask, rg, start, len, batch) {
                    Ok(reader) => {
                        for batch in reader {
                            if tx.send(batch).is_err() {
                                break; // consumer dropped
                            }
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(ArrowError::ExternalError(Box::new(e))));
                    }
                }
            });
            start += len;
        }
        Self {
            schema,
            receivers,
            cur: 0,
        }
    }
}

impl Iterator for ParallelRangeReader {
    type Item = Result<RecordBatch, ArrowError>;
    fn next(&mut self) -> Option<Self::Item> {
        while self.cur < self.receivers.len() {
            match self.receivers[self.cur].recv() {
                Ok(item) => return Some(item),
                Err(_) => self.cur += 1, // this range's channel closed → next range
            }
        }
        None
    }
}

impl RecordBatchReader for ParallelRangeReader {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

// --------------------------------------------------------------------------- //
// Local entry point: choose sequential vs K-split
// --------------------------------------------------------------------------- //
fn open_local_reader(
    path: String,
    row_groups: Option<Vec<usize>>,
    columns: Option<Vec<String>>,
    batch_size: usize,
    budget_bytes: u64,
    k: usize,
    split_threshold_bytes: u64,
) -> Result<Box<dyn RecordBatchReader + Send>, ParquetError> {
    // Lean footer parse (#6): the page index is only needed for the K-split
    // RowSelection (to skip pages by byte range). K-split can only fire when k > 1,
    // so for the common k == 1 local path we Skip the page index entirely — a
    // cheaper footer parse that matters on many-row-group files. When k > 1 we load
    // it Optional so the lone-big-row-group split can use it if present.
    let policy = if k > 1 {
        PageIndexPolicy::Optional
    } else {
        PageIndexPolicy::Skip
    };
    let opts = ArrowReaderOptions::new().with_page_index_policy(policy);
    let meta = ArrowReaderMetadata::load(&File::open(&path)?, opts)?;
    let mask = projection_mask(meta.metadata().file_metadata().schema_descr(), &columns);
    let selected: Vec<usize> = match row_groups {
        Some(v) => v,
        None => (0..meta.metadata().num_row_groups()).collect(),
    };
    let schema = probe_schema(&path, &meta, &mask)?;

    // K-split only for a *single* row group above the threshold, and only when the
    // page index is present (else each range would decode the whole column chunk).
    // This is exactly the lone-big-fragment case Ray's pool can't parallelize; every
    // other layout uses the sequential path so crate-K and Ray's pool never multiply.
    let split = k > 1
        && selected.len() == 1
        && meta.metadata().row_group(selected[0]).total_byte_size() as u64 >= split_threshold_bytes
        && meta.metadata().offset_index().is_some();

    if split {
        let rg = selected[0];
        let rgm = meta.metadata().row_group(rg);
        let total_rows = rgm.num_rows().max(0) as usize;
        let eff = byte_budget_rows(
            rgm.total_byte_size(),
            rgm.num_rows(),
            batch_size,
            budget_bytes,
        );
        Ok(Box::new(ParallelRangeReader::spawn(
            path, meta, mask, schema, rg, total_rows, k, eff,
        )))
    } else {
        Ok(Box::new(RowGroupSeqReader::new(
            path,
            meta,
            mask,
            schema,
            selected,
            batch_size,
            budget_bytes,
        )))
    }
}

#[pyfunction]
#[pyo3(signature = (path, row_groups=None, columns=None, batch_size=131072, decode_budget_bytes=2*1024*1024, k=1, split_threshold_bytes=134217728))]
fn read_row_groups(
    path: String,
    row_groups: Option<Vec<usize>>,
    columns: Option<Vec<String>>,
    batch_size: usize,
    decode_budget_bytes: u64,
    k: usize,
    split_threshold_bytes: u64,
) -> PyResult<ArrowStream> {
    let reader = open_local_reader(
        path,
        row_groups,
        columns,
        batch_size,
        decode_budget_bytes,
        k,
        split_threshold_bytes,
    )
    .map_err(to_py)?;
    Ok(into_py_stream(reader))
}

// --------------------------------------------------------------------------- //
// S3 read (async, windowed, byte-budgeted, order-preserving)
// --------------------------------------------------------------------------- //
/// Number of decoded batches a unit task may run ahead of the consumer. Depth 2
/// bounds resident memory while still letting a task fetch/decode one batch ahead.
const S3_CHANNEL_DEPTH: usize = 2;

/// A sync `RecordBatchReader` fed by K background tokio tasks (one per row-range
/// unit), each draining its unit into a bounded async channel. The consumer drains
/// channels in ascending unit order — so at most `k * S3_CHANNEL_DEPTH` batches are
/// resident and rows come out in file order (K units are contiguous ascending
/// ranges). `blocking_recv` is called from the Python thread (outside the runtime),
/// which is exactly what tokio's mpsc supports.
struct S3ChannelReader {
    schema: SchemaRef,
    receivers: Vec<mpsc::Receiver<Result<RecordBatch, ArrowError>>>,
    cur: usize,
}

impl Iterator for S3ChannelReader {
    type Item = Result<RecordBatch, ArrowError>;
    fn next(&mut self) -> Option<Self::Item> {
        while self.cur < self.receivers.len() {
            match self.receivers[self.cur].blocking_recv() {
                Some(item) => return Some(item),
                None => self.cur += 1, // this unit's channel closed → next unit
            }
        }
        None
    }
}

impl RecordBatchReader for S3ChannelReader {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

/// Rows per fetch window from a byte budget over the row group's *compressed*
/// bytes/row — this bounds IN-FLIGHT NETWORK bytes (what we fetch before decode).
/// 0 means "whole range in one shot" (no window cap).
fn window_rows_for(rgm: &parquet::file::metadata::RowGroupMetaData, fetch_window_mb: u64) -> usize {
    if fetch_window_mb == 0 {
        return 0;
    }
    let comp = rgm.compressed_size().max(1) as f64;
    let rows = rgm.num_rows().max(1) as f64;
    let comp_bpr = (comp / rows).max(1.0);
    (((fetch_window_mb as f64) * 1024.0 * 1024.0) / comp_bpr).max(1.0) as usize
}

/// Drive one unit (a list of contiguous `(rg, start, len)` sub-ranges, in order)
/// over the async object store, sending decoded batches to `tx`. Each unit's rows
/// are sliced into fetch windows; every window builds a fresh stream restricted to
/// that window's rows via a `RowSelection`, so only that window's pages are fetched
/// and buffered (the page index, loaded below, lets object_store skip unselected
/// pages by byte range). The decode batch is byte-budgeted (knob 1); the window is
/// compressed-byte-budgeted (knob 2). Backpressure via `tx.send(..).await` on the
/// bounded channel keeps at most `S3_CHANNEL_DEPTH` batches resident per unit.
#[allow(clippy::too_many_arguments)]
async fn drive_unit(
    store: Arc<dyn ObjectStore>,
    path: ObjPath,
    meta: ArrowReaderMetadata,
    mask: ProjectionMask,
    subranges: Vec<(usize, usize, usize)>,
    budget_bytes: u64,
    batch_clamp: usize,
    fetch_window_mb: u64,
    tx: mpsc::Sender<Result<RecordBatch, ArrowError>>,
) {
    for (rg, start, len) in subranges {
        let rgm = meta.metadata().row_group(rg);
        let batch_rows = byte_budget_rows(
            rgm.total_byte_size(),
            rgm.num_rows(),
            batch_clamp,
            budget_bytes,
        );
        let window_rows = window_rows_for(rgm, fetch_window_mb);
        let end = start + len;
        let step = if window_rows == 0 {
            len.max(1)
        } else {
            window_rows.max(1)
        };
        let mut w = start;
        while w < end {
            let wlen = step.min(end - w);
            // Select rows [w, w+wlen) WITHIN this row group (we restrict to `rg`).
            let sel = RowSelection::from(vec![RowSelector::skip(w), RowSelector::select(wlen)]);
            let reader = ParquetObjectReader::new(store.clone(), path.clone());
            let built = ParquetRecordBatchStreamBuilder::new_with_metadata(reader, meta.clone())
                .with_row_groups(vec![rg])
                .with_row_selection(sel)
                .with_batch_size(batch_rows)
                .with_projection(mask.clone())
                .build();
            let mut stream = match built {
                Ok(s) => s,
                Err(e) => {
                    let _ = tx.send(Err(ArrowError::ExternalError(Box::new(e)))).await;
                    return;
                }
            };
            while let Some(item) = stream.next().await {
                let is_err = item.is_err();
                let msg = item.map_err(|e| ArrowError::ExternalError(Box::new(e)));
                if tx.send(msg).await.is_err() {
                    return; // consumer dropped
                }
                if is_err {
                    return;
                }
            }
            w += wlen;
        }
    }
}

#[pyfunction]
#[pyo3(signature = (bucket, key, region, anonymous, endpoint=None, access_key_id=None,
                    secret_access_key=None, session_token=None, allow_http=false,
                    virtual_hosted_style=false, row_groups=None, columns=None,
                    batch_size=131072, decode_budget_bytes=2*1024*1024,
                    fetch_window_mb=16, k=1, split_threshold_bytes=134217728))]
#[allow(clippy::too_many_arguments)]
fn read_row_groups_s3(
    bucket: String,
    key: String,
    region: String,
    anonymous: bool,
    // Full S3 config, recovered from the pyarrow S3FileSystem on the Python side
    // (fs.__reduce__()[1][0]) so credentialed / custom-endpoint (MinIO, moto) /
    // anonymous buckets all decode identically to PyArrow. Empty/None → unset.
    endpoint: Option<String>,
    access_key_id: Option<String>,
    secret_access_key: Option<String>,
    session_token: Option<String>,
    allow_http: bool,
    virtual_hosted_style: bool,
    row_groups: Option<Vec<usize>>,
    columns: Option<Vec<String>>,
    batch_size: usize,
    decode_budget_bytes: u64,
    fetch_window_mb: u64,
    k: usize,
    split_threshold_bytes: u64,
) -> PyResult<ArrowStream> {
    let mut sb = AmazonS3Builder::new()
        .with_bucket_name(&bucket)
        .with_region(&region)
        .with_virtual_hosted_style_request(virtual_hosted_style);
    if let Some(ep) = endpoint.filter(|s| !s.is_empty()) {
        sb = sb.with_endpoint(ep);
    }
    if allow_http {
        sb = sb.with_allow_http(true);
    }
    if anonymous {
        // No signing — public buckets. Any creds are irrelevant.
        sb = sb.with_skip_signature(true);
    } else {
        // Explicit static creds if the S3FileSystem carried them; otherwise the
        // builder falls back to the AWS credential chain (env / IMDS role).
        if let Some(kid) = access_key_id.filter(|s| !s.is_empty()) {
            sb = sb.with_access_key_id(kid);
        }
        if let Some(s) = secret_access_key.filter(|s| !s.is_empty()) {
            sb = sb.with_secret_access_key(s);
        }
        if let Some(t) = session_token.filter(|s| !s.is_empty()) {
            sb = sb.with_token(t);
        }
    }
    let store: Arc<dyn ObjectStore> = Arc::new(sb.build().map_err(to_py)?);
    let obj_path = ObjPath::from(key);

    let rt = shared_runtime();

    // Load footer + page index ONCE (Optional so a window's RowSelection can skip
    // unselected pages by byte range), and build the projected output schema up
    // front from an empty stream (no network). Reporting the projected schema is
    // what keeps it matching the projected batches at the FFI boundary.
    let (meta, mask, schema) = rt
        .block_on(async {
            let opts = ArrowReaderOptions::new().with_page_index_policy(PageIndexPolicy::Optional);
            let mut probe = ParquetObjectReader::new(store.clone(), obj_path.clone());
            let meta = ArrowReaderMetadata::load_async(&mut probe, opts).await?;
            let mask = projection_mask(meta.metadata().file_metadata().schema_descr(), &columns);
            let schema = ParquetRecordBatchStreamBuilder::new_with_metadata(
                ParquetObjectReader::new(store.clone(), obj_path.clone()),
                meta.clone(),
            )
            .with_projection(mask.clone())
            .with_row_groups(vec![])
            .build()?
            .schema()
            .clone();
            Ok::<_, parquet::errors::ParquetError>((meta, mask, schema))
        })
        .map_err(to_py)?;

    let selected: Vec<usize> = match row_groups {
        Some(v) => v,
        None => (0..meta.metadata().num_row_groups()).collect(),
    };

    // K-split ONLY for a lone row group above the threshold with a page index —
    // the case Ray's fragment pool can't parallelize. Mirrors the local rule so
    // crate-K and Ray's pool never multiply. Otherwise a single windowed stream
    // (K=1) over all selected groups in order; Ray's pool parallelizes files.
    let split = k > 1
        && selected.len() == 1
        && meta.metadata().row_group(selected[0]).total_byte_size() as u64 >= split_threshold_bytes
        && meta.metadata().offset_index().is_some();

    // Build the per-unit sub-range lists (each becomes one task + one channel,
    // drained in order).
    let units: Vec<Vec<(usize, usize, usize)>> = if split {
        let rg = selected[0];
        let total_rows = meta.metadata().row_group(rg).num_rows().max(0) as usize;
        let chunk = total_rows.div_ceil(k.max(1)).max(1);
        let mut units = Vec::new();
        let mut start = 0usize;
        while start < total_rows {
            let len = chunk.min(total_rows - start);
            units.push(vec![(rg, start, len)]);
            start += len;
        }
        units
    } else {
        // One unit: every selected group, whole, in order.
        let subranges = selected
            .iter()
            .map(|&rg| {
                (
                    rg,
                    0usize,
                    meta.metadata().row_group(rg).num_rows().max(0) as usize,
                )
            })
            .collect();
        vec![subranges]
    };

    // Spawn one task per unit on the shared runtime; collect receivers in order.
    let mut receivers = Vec::with_capacity(units.len());
    for subranges in units {
        let (tx, rx) = mpsc::channel::<Result<RecordBatch, ArrowError>>(S3_CHANNEL_DEPTH);
        receivers.push(rx);
        let (store, path, meta, mask) =
            (store.clone(), obj_path.clone(), meta.clone(), mask.clone());
        rt.spawn(drive_unit(
            store,
            path,
            meta,
            mask,
            subranges,
            decode_budget_bytes,
            batch_size,
            fetch_window_mb,
            tx,
        ));
    }

    Ok(into_py_stream(Box::new(S3ChannelReader {
        schema,
        receivers,
        cur: 0,
    })))
}

fn to_py<E: std::fmt::Display>(e: E) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

#[pymodule]
fn ray_data_arrow_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_row_groups, m)?)?;
    m.add_function(wrap_pyfunction!(read_row_groups_s3, m)?)?;
    Ok(())
}
