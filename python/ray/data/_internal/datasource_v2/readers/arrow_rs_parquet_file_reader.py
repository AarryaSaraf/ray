"""Experimental arrow-rs Parquet reader (prototype).

Subclasses :class:`ParquetFileReader` and swaps *only* the per-fragment decode
step (:meth:`_iter_fragment_tables`) for the ``ray_data_arrow_rs`` PyO3
extension (a thin wrapper over the Rust ``parquet``/``arrow`` crates).
Everything above the seam — chunking / row-group fan-out, column projection
resolution, ``path`` / ``row_hash`` synthesis, ``limit`` slicing, block sizing,
per-fragment retry — is inherited unchanged from :class:`FileReader` /
:class:`ParquetFileReader`.

Selected via ``DataContext.use_arrow_rs_parquet_reader`` (only takes effect when
``use_datasource_v2`` is also set). Switched in
:meth:`ParquetScanner.create_reader`.

How it reads
------------
The native extension exposes two entry points, both returning an Arrow
C-stream (consumed zero-copy via ``pa.RecordBatchReader.from_stream``):

- ``read_row_groups(path, row_groups, columns, batch_size, ...)`` — local files.
- ``read_row_groups_s3(bucket, key, region, anonymous, ...creds..., row_groups,
  columns, batch_size, decode_budget_bytes, fetch_window_mb, k,
  split_threshold_bytes)`` — S3 via the Rust ``object_store`` crate, using a
  **windowed** fetch (only ``fetch_window_mb`` of compressed bytes in flight per
  stream) so S3 peak RSS is a knob, not a property of the row-group size.

Byte-budgeted decode (no reader-side accumulation)
--------------------------------------------------
The native reader sizes each decode batch *by bytes, not rows*: it reads each
row group's uncompressed size / row count from the footer and picks a row count
so ``rows × bytes_per_row ≈ decode_budget_bytes`` (~8 MiB,
:data:`_ARROW_RS_DECODE_BUDGET_BYTES`). A wide-string group gets few rows/batch,
a numeric group many — both land near the budget, so the decoded working set is
flat across schemas (this is *why* arrow-rs memory doesn't scale with the data
the way PyArrow's whole-row-group materialization does). The ``batch_size`` we
pass is only the upper *clamp*.

We yield each budget-sized batch straight through — exactly like the base
PyArrow path yields one table per scanner batch (:meth:`FileReader.
_iter_fragment_tables`). Coalescing to ``target_max_block_size`` is done once,
downstream, by the read op's :class:`BlockOutputBuffer`. Accumulating a full
block *here* as well (an earlier prototype did) just stacks a second
block-sized buffer on top of the output buffer's, roughly doubling per-worker
peak RSS relative to PyArrow — so we don't. The decode transient stays bounded
by the byte budget; the single ~128 MiB coalesce buffer lives downstream, shared
with the PyArrow path.

Prototype limitations (documented, not hidden)
----------------------------------------------
- The predicate is applied *post-decode* in Python via PyArrow, not pushed
  into arrow-rs. Row groups are still selected upstream by the chunker; only
  stat-based intra-chunk row-group skipping is skipped. This makes filtered-read
  benchmark numbers conservative for arrow-rs.
- :meth:`_arrow_rs_supported` restricts the native path to local **and S3**
  files with flat, non-nested / non-dictionary / non-extension columns and no
  ``int96`` coercion. Everything else transparently falls back to the PyArrow
  reader, so correctness is never at risk — but benchmarks must confirm the
  arrow-rs path actually ran (see the ``RAY_DATA_USE_ARROW_RS_PARQUET_READER``
  verification).
"""

import logging
import os
from typing import TYPE_CHECKING, Iterator, List, Optional

import pyarrow as pa
import pyarrow.dataset as pds
from typing_extensions import override

from ray._common.utils import env_integer
from ray.data._internal.datasource_v2.readers.file_reader import (
    _ARROW_DEFAULT_BATCH_SIZE,
)
from ray.data._internal.datasource_v2.readers.parquet_file_reader import (
    ParquetFileReader,
    _estimate_batch_size_from_metadata,
)
from ray.data._internal.util import MiB
from ray.util.annotations import DeveloperAPI

if TYPE_CHECKING:
    import pyarrow.compute as pc  # noqa: F401

logger = logging.getLogger(__name__)

# Byte budget for a single arrow-rs decode batch. Sizing decode batches by
# bytes (not a fixed row count) keeps the transient working set flat across
# schemas: a wide file gets few rows/batch, a narrow file gets many. Kept far
# below ``target_block_size`` so the decode transient is bounded while output
# blocks are still coalesced to the normal Ray block size. Default 2 MiB: the
# standalone benchmark found budget is the *floor* knob (it moves peak only ~12 MB
# across its range; the S3 fetch window is the real lever), and 2 MiB is the lowest
# that holds throughput — so we take the smaller working set. Swept on the Linux/S3
# run to confirm it holds under real integration (Agents.md §7.1).
_ARROW_RS_DECODE_BUDGET_BYTES = env_integer(
    "RAY_DATA_ARROW_RS_DECODE_BUDGET_BYTES", 2 * MiB
)

# Floor on the estimated decode batch size (rows), so a very wide schema can't
# collapse the batch to a handful of rows and starve throughput.
_ARROW_RS_MIN_DECODE_BATCH_ROWS = 2048

# Intra-fragment parallelism: when a fragment is a *single* row group larger than
# the block-size target (the lone-big-fragment case Ray's thread pool can't split),
# the native reader decodes it in ``K`` parallel row-range workers and merges them
# back in order. Every other layout (multiple / small row groups) uses K=1 because
# Ray's fragment thread pool already parallelizes those — so crate-K and Ray's pool
# never multiply.
#
# Default K=1: locally, K-split costs memory (each range holds its own decode
# transient) for ~no speed, since there is no network latency to hide (benchmarks:
# Agents.md §5.1, §6.3). K>1 is opt-in via the env var and is reserved for the S3
# phase, where concurrent range GETs hide request latency.
_ARROW_RS_K = env_integer("RAY_DATA_ARROW_RS_K", 1)

# Default single-row-group split threshold when the reader has no target block
# size (a row group smaller than this is left to the sequential path).
_ARROW_RS_DEFAULT_SPLIT_THRESHOLD_BYTES = 128 * MiB

# S3 fetch window (MiB of *compressed* bytes in flight per stream). This is the
# memory knob for the S3 path: the native reader slices each row group's rows into
# windows sized so only ~this many compressed bytes are fetched+buffered before
# decode, so peak RSS is `≈ fetch_window + decode_budget` — flat regardless of
# row-group size, instead of PyArrow's whole-row-group pre-buffer. 0 = no window
# cap (fetch the whole range at once). Swept on the Linux/S3 run (Agents.md §7.1).
_ARROW_RS_FETCH_WINDOW_MB = env_integer("RAY_DATA_ARROW_RS_FETCH_WINDOW_MB", 16)


def _trace_reader_path(supported: bool) -> None:
    """Benchmark instrumentation (inert unless ``RAY_DATA_ARROW_RS_PATH_TRACE``
    names a directory): append ``native``/``fallback`` for each fragment to a
    per-pid file so a harness can assert which path the support gate chose. Never
    raises into the read path.
    """
    trace_dir = os.environ.get("RAY_DATA_ARROW_RS_PATH_TRACE")
    if not trace_dir:
        return
    try:
        import socket

        # Namespace by hostname so nodes writing to a shared trace dir (multi-node
        # verification) don't collide on pid; the harness's ``path_*.log`` glob
        # still matches. Single-node is unaffected.
        line = "native\n" if supported else "fallback\n"
        fname = f"path_{socket.gethostname()}_{os.getpid()}.log"
        with open(os.path.join(trace_dir, fname), "a") as fh:
            fh.write(line)
    except Exception:
        pass


def _s3_config(fs: "pa.fs.S3FileSystem") -> dict:
    """Recover the full S3 connection config from a pyarrow ``S3FileSystem`` so
    the native crate connects *identically* — same endpoint, credentials, region,
    addressing style — instead of rebuilding a default client from the ambient env
    (which would silently ignore an explicit endpoint override or static creds and
    break MinIO / moto / custom-endpoint / credentialed buckets).

    pyarrow round-trips the whole config through ``__reduce__`` (verified to include
    ``secret_key``/``session_token``/``endpoint_override``/``scheme`` across the
    pyarrow versions Ray supports), so that is the source of truth. Empty strings
    (pyarrow's "unset" sentinel) are normalized to ``None``.
    """
    try:
        opts = fs.__reduce__()[1][0]
    except Exception:
        opts = {}

    def _val(key):
        v = opts.get(key)
        return v if v else None

    endpoint = _val("endpoint_override")
    # object_store refuses plain-HTTP endpoints unless explicitly allowed
    # (moto / MinIO are http). pyarrow's `scheme` defaults to "https" even when
    # the endpoint override is an http:// URL, so trust the endpoint URL first.
    allow_http = (str(endpoint).startswith("http://")) or opts.get("scheme") == "http"

    return {
        "region": _val("region") or "us-east-1",
        "anonymous": bool(opts.get("anonymous", False)),
        "endpoint": endpoint,
        "access_key_id": _val("access_key"),
        "secret_access_key": _val("secret_key"),
        "session_token": _val("session_token"),
        "allow_http": allow_http,
        "virtual_hosted_style": bool(opts.get("force_virtual_addressing", False)),
    }


@DeveloperAPI
class ArrowRsParquetFileReader(ParquetFileReader):
    """Parquet reader that decodes each fragment via the arrow-rs extension.

    See the module docstring for the design. Only :meth:`_iter_fragment_tables`,
    :meth:`_resolve_batch_size`, and :meth:`_on_batch_read` are overridden; the
    rest of the read pipeline is inherited from :class:`ParquetFileReader`.
    """

    @override
    def _resolve_batch_size(self, dataset: pds.Dataset) -> int:
        """Size the decode batch to the arrow-rs byte budget, not the block size.

        Priority: explicit ``batch_size`` > byte-budget estimate from row-group
        metadata > default. Unlike the base reader this targets
        :data:`_ARROW_RS_DECODE_BUDGET_BYTES` (~8 MiB) rather than
        ``target_block_size`` (~128 MiB), because each decode batch is yielded
        straight through in :meth:`_iter_fragment_tables` (the downstream
        ``BlockOutputBuffer`` does the coalescing to the block size).
        """
        if self._explicit_batch_size is not None:
            return self._explicit_batch_size

        if self._target_block_size is None:
            return _ARROW_DEFAULT_BATCH_SIZE

        first_fragment = next(dataset.get_fragments(), None)
        if first_fragment is None:
            return _ARROW_DEFAULT_BATCH_SIZE

        estimated = _estimate_batch_size_from_metadata(
            first_fragment, self._columns, _ARROW_RS_DECODE_BUDGET_BYTES
        )
        if estimated is None:
            return _ARROW_DEFAULT_BATCH_SIZE
        return max(estimated, _ARROW_RS_MIN_DECODE_BATCH_ROWS)

    @override
    def _on_batch_read(self, table: pa.Table) -> None:
        """No-op: the decode batch size is fixed by the byte budget, so there is
        nothing to refine from actual data (unlike the base reader)."""
        return None

    def _arrow_rs_supported(
        self,
        fragment: pds.ParquetFileFragment,
        read_columns: Optional[List[str]],
    ) -> bool:
        """Return True only for fragments the native path handles correctly.

        Conservative on purpose — anything not covered here falls back to the
        PyArrow reader in :meth:`_iter_fragment_tables`, so correctness is never
        at risk. Supports **local and S3** files with flat columns only; every
        other filesystem (GCS, ABFS, HTTP, …) falls back to PyArrow.
        """
        from pyarrow.fs import LocalFileSystem, S3FileSystem

        # Local and S3 are wired in `_iter_fragment_tables` (S3 uses the windowed,
        # byte-budgeted native path). Any other filesystem falls back to PyArrow.
        if not isinstance(self._filesystem, (LocalFileSystem, S3FileSystem)):
            return False

        # int96 coercion and forced dictionary decoding aren't mirrored by the
        # native reader.
        if self._parquet_format_kwargs.get("coerce_int96_timestamp_unit") is not None:
            return False
        if self._parquet_format_kwargs.get("dictionary_columns"):
            return False

        physical_schema = fragment.physical_schema
        unified_schema = self._file_dataset_schema
        names = (
            read_columns if read_columns is not None else list(physical_schema.names)
        )

        # Empty projection (count-style scan) — let PyArrow handle the
        # stub-column dance.
        if read_columns is not None and len(read_columns) == 0:
            return False

        # A dotted name means a nested-column projection, which the native
        # reader doesn't support.
        if any("." in name for name in names):
            return False

        for name in names:
            idx = physical_schema.get_field_index(name)
            if idx == -1:
                # Column absent from this fragment (schema evolution) — defer to
                # PyArrow's null-fill path.
                return False
            field_type = physical_schema.field(idx).type
            # Catch every extension type, not just custom subclasses:
            # pyarrow *canonical* extensions (e.g. fixed_shape_tensor) are not
            # `isinstance(pa.ExtensionType)` in some pyarrow versions, so check
            # for an `extension_name` too — otherwise a tensor column would slip
            # through to the native crate, which doesn't handle extensions.
            if (
                pa.types.is_nested(field_type)
                or pa.types.is_dictionary(field_type)
                or isinstance(field_type, pa.ExtensionType)
                or getattr(field_type, "extension_name", None) is not None
            ):
                return False
            if unified_schema is None:
                continue
            unified_idx = unified_schema.get_field_index(name)
            if unified_idx == -1:
                continue
            # A per-fragment type that differs from the unified type needs a
            # cast the native reader doesn't do — fall back.
            if unified_schema.field(unified_idx).type != field_type:
                return False
        return True

    @override
    def _iter_fragment_tables(
        self,
        fragment: pds.Fragment,
        scanner_kwargs: dict,
    ) -> "Iterator[pa.Table]":
        try:
            import ray_data_arrow_rs
        except ImportError as e:
            raise ImportError(
                "use_arrow_rs_parquet_reader=True requires the "
                "'ray_data_arrow_rs' extension. Build it with "
                "`maturin develop --release` from "
                "python/ray/data/_internal/datasource_v2/native/ray_data_arrow_rs/."
            ) from e

        from ray.data._internal.datasource.parquet_datasource import (
            _resolve_read_columns,
        )
        from ray.data._internal.planner.plan_expression.expression_visitors import (
            get_column_references,
        )

        columns = scanner_kwargs.get("columns")
        filter_expr = scanner_kwargs.get("filter")
        batch_size = scanner_kwargs.get("batch_size") or _ARROW_DEFAULT_BATCH_SIZE

        filter_columns = (
            get_column_references(self._predicate)
            if self._predicate is not None
            else None
        )
        read_columns = _resolve_read_columns(columns, filter_expr, filter_columns)

        supported = self._arrow_rs_supported(fragment, read_columns)
        _trace_reader_path(supported)
        if not supported:
            yield from super()._iter_fragment_tables(fragment, scanner_kwargs)
            return

        row_groups = (
            [rg.id for rg in fragment.row_groups]
            if fragment.row_groups is not None
            else None
        )

        from pyarrow.fs import S3FileSystem

        split_threshold = (
            self._target_block_size
            if self._target_block_size is not None
            else _ARROW_RS_DEFAULT_SPLIT_THRESHOLD_BYTES
        )

        fs = self._filesystem
        if isinstance(fs, S3FileSystem):
            bucket, _, key = fragment.path.partition("/")
            cfg = _s3_config(fs)
            reader = ray_data_arrow_rs.read_row_groups_s3(
                bucket,
                key,
                cfg["region"],
                cfg["anonymous"],
                endpoint=cfg["endpoint"],
                access_key_id=cfg["access_key_id"],
                secret_access_key=cfg["secret_access_key"],
                session_token=cfg["session_token"],
                allow_http=cfg["allow_http"],
                virtual_hosted_style=cfg["virtual_hosted_style"],
                row_groups=row_groups,
                columns=read_columns,
                batch_size=batch_size,
                decode_budget_bytes=_ARROW_RS_DECODE_BUDGET_BYTES,
                fetch_window_mb=_ARROW_RS_FETCH_WINDOW_MB,
                k=_ARROW_RS_K,
                split_threshold_bytes=split_threshold,
            )
        else:
            reader = ray_data_arrow_rs.read_row_groups(
                fragment.path,
                row_groups,
                read_columns,
                batch_size,
                _ARROW_RS_DECODE_BUDGET_BYTES,
                _ARROW_RS_K,
                split_threshold,
            )

        record_batch_reader = pa.RecordBatchReader.from_stream(reader)

        # Yield each budget-sized batch straight through. The read op's
        # BlockOutputBuffer coalesces to target_max_block_size downstream (same
        # as the PyArrow path) — accumulating a full block here too would just
        # stack a second block-sized buffer on top of it. See module docstring.
        for batch in record_batch_reader:
            table = pa.Table.from_batches([batch], schema=record_batch_reader.schema)
            if filter_expr is not None:
                table = table.filter(filter_expr)
                if table.num_rows == 0:
                    continue
            if columns is not None:
                table = table.select([c for c in columns if c in table.column_names])
            yield table
