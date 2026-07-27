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
  split_threshold_bytes, prefetch_windows)`` — S3 via the Rust ``object_store``
  crate, using a **windowed** fetch (only ``fetch_window_mb`` of compressed bytes
  in flight per stream) so S3 peak RSS is a knob, not a property of the row-group
  size. ``prefetch_windows`` pipelines consecutive windows (issue window N+1's GET
  while window N decodes) to hide S3 latency without staging the whole row group.

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
- Predicate handling prunes at row-group granularity *natively*: the pushed
  Ray ``Expr`` is lowered to a small JSON IR (:func:`_predicate_to_ir`) and
  handed to the crate, which drops row groups whose footer statistics prove no
  row can match (``predicate.rs``) before fetching or decoding them. This
  replaces PyArrow's ``fragment.subset(filter=...)``. Pruning is conservative
  (a missing column / absent stats / uncomparable type keeps the group), so it
  can only avoid IO/decode, never change results. *Row-level* filtering is then
  applied post-decode in Python via PyArrow (the final authority) — the crate
  has no in-decode ``RowFilter`` yet, so rows inside a surviving row group are
  decoded before being dropped.
- :meth:`_arrow_rs_supported` restricts the native path to local **and S3**
  files whose columns the crate decodes byte-identically to PyArrow: flat
  types, ``dictionary``, ``map``, and ``extension`` types (registered like
  Ray's tensor types or not — the crate passes the embedded arrow-schema field
  metadata straight through FFI, so pyarrow reconstructs them exactly as it
  would on its own read path), plus struct / list / map nesting of all those to
  any depth. Still gated: ``union`` / ``list_view`` and other exotic nesting
  (untested against the crate), ``int96`` coercion, forced ``dictionary_columns``
  reads, and nested-column *projection* (dotted names). Everything gated
  transparently falls back to the PyArrow reader, so correctness is never at
  risk — but benchmarks must confirm the arrow-rs path actually ran (see the
  ``RAY_DATA_USE_ARROW_RS_PARQUET_READER`` verification).
"""

import json
import logging
import os
from functools import cached_property
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, NamedTuple, Optional, Tuple

import pyarrow as pa
import pyarrow.dataset as pds
from typing_extensions import override

from ray._common.utils import env_integer
from ray.data._internal.datasource_v2.native_metadata import (
    read_native_metadata as _read_native_metadata_via_crate,
)
from ray.data._internal.datasource_v2.native_metadata import s3_config as _s3_config
from ray.data._internal.datasource_v2.readers.file_reader import (
    _ARROW_DEFAULT_BATCH_SIZE,
    _ARROW_SCANNER_BATCH_READAHEAD,
)
from ray.data._internal.datasource_v2.readers.parquet_file_reader import (
    ParquetFileReader,
    _estimate_batch_size_from_metadata,
)
from ray.data._internal.util import MiB
from ray.util.annotations import DeveloperAPI

if TYPE_CHECKING:
    import pyarrow.compute as pc  # noqa: F401

    from ray.data._internal.datasource_v2.listing.file_manifest import (  # noqa: F401
        FileManifest,
    )
    from ray.data.expressions import Expr  # noqa: F401

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

# S3 window prefetch depth: how many consecutive fetch windows the native reader
# keeps in flight per stream. With depth 2 the GET for window N+1 is issued while
# window N decodes, so S3 first-byte latency is hidden behind decode instead of
# paid serially between windows — the memory-bounded analog of pyarrow's
# whole-fragment pre_buffer (peak stays `≈ prefetch_windows * fetch_window_mb`
# compressed in flight, not the whole row group). 1 = strictly serial windows
# (no prefetch). Only affects the S3 path (local reads have no fetch latency to
# hide); reserved for and swept on the Linux/S3 run (Agents.md §7.1).
_ARROW_RS_PREFETCH_WINDOWS = env_integer("RAY_DATA_ARROW_RS_PREFETCH_WINDOWS", 2)


def _arrow_rs_type_supported(t: pa.DataType) -> bool:
    """True if the native path decodes this column type byte-identically to
    PyArrow: flat types, ``dictionary``, ``map``, and ``extension`` types, plus
    struct / list / map nesting of all those to any depth.

    Extension types are admitted (including Ray's ``ArrowTensorType`` /
    ``ArrowVariableShapedTensorType`` and pyarrow's canonical
    ``fixed_shape_tensor``): the crate carries the embedded arrow-schema field
    metadata (``ARROW:extension:name`` / ``:metadata``) straight through the C
    data interface, so pyarrow reconstructs a *registered* extension identically
    and surfaces an *unregistered* one as its storage type + metadata — exactly
    as PyArrow's own reader does. We recurse into the storage type so an
    extension over an unsupported nesting still falls back. (This is why a plain
    ``isinstance(pa.ExtensionType)`` rejection was dropped — empirically every
    extension case round-trips; see ``test_extension_types_native_parity``.)

    The remaining ``is_nested`` rejection (``union``, ``list_view`` /
    ``large_list_view``, ``run_end_encoded``, interval, …) is an unreachable
    safety net rather than a real limitation: those are Arrow *in-memory-only*
    types with no Parquet encoding — PyArrow itself refuses to write them
    ("Unhandled type for Arrow to Parquet schema conversion"), so a column read
    from a Parquet file can never have one. Every type Parquet *can* store
    (flat / dictionary / map / extension / list / struct / map nesting to any
    depth) is admitted above and decodes byte-identically to PyArrow (verified in
    ``test_extension_types_native_parity`` and the type probes behind it). The
    branch stays as defense in depth in case a future Arrow/Parquet version adds
    an encoding for one of them before the crate is verified against it.

    Note a *forced* dictionary read (the ``dictionary_columns`` read kwarg) is
    still gated separately in :meth:`_reader_level_supported` — the crate doesn't
    honor it; a *naturally* dictionary-typed column (embedded arrow dictionary
    type) is what this admits.
    """
    if isinstance(t, pa.ExtensionType):
        return _arrow_rs_type_supported(t.storage_type)
    # Canonical pyarrow extensions (e.g. ``fixed_shape_tensor``) are not always
    # ``isinstance(pa.ExtensionType)`` across pyarrow versions; catch them via
    # ``extension_name`` and recurse into their storage type just the same.
    if getattr(t, "extension_name", None) is not None:
        storage = getattr(t, "storage_type", None)
        return storage is not None and _arrow_rs_type_supported(storage)
    if pa.types.is_dictionary(t):
        return _arrow_rs_type_supported(t.value_type)
    if pa.types.is_struct(t):
        return all(_arrow_rs_type_supported(f.type) for f in t)
    if pa.types.is_map(t):
        return _arrow_rs_type_supported(t.key_type) and _arrow_rs_type_supported(
            t.item_type
        )
    if (
        pa.types.is_list(t)
        or pa.types.is_large_list(t)
        or pa.types.is_fixed_size_list(t)
    ):
        return _arrow_rs_type_supported(t.value_type)
    if pa.types.is_nested(t):
        return False  # union, list_view, ... — untested against the crate
    return True


def _is_pyarrow_int96_decode_type(t: pa.DataType) -> bool:
    """Whether ``t`` is the arrow type PyArrow produces for an INT96 timestamp
    column by default: ``timestamp[ns]`` with no timezone. parquet-rs matches
    this for a file with no embedded Arrow-schema hint, but honors a non-ns hint
    when present — so an INT96 column is only safe on the native path when the
    crate also lands on ns/no-tz (otherwise it would diverge from PyArrow, which
    always forces ns)."""
    return pa.types.is_timestamp(t) and t.unit == "ns" and t.tz is None


def _pyarrow_fragment_int96_roots(fragment: "pds.ParquetFileFragment") -> set:
    """Root (top-level) column names backing an INT96 leaf in a PyArrow Parquet
    fragment, read from the *parquet* schema (``fragment.metadata.schema``).

    The fragment's ``physical_schema`` is PyArrow's post-coercion Arrow schema, in
    which INT96 already shows up as ``timestamp[ns]`` — so it can't reveal which
    columns were INT96 on disk. The parquet schema descriptor can, via each leaf
    column's ``physical_type``. Used by the conservative re-gate so an INT96 file
    is never handed to the crate through the per-fragment path."""
    roots: set = set()
    try:
        schema = fragment.metadata.schema
        for i in range(len(schema)):
            col = schema.column(i)
            if col.physical_type == "INT96":
                roots.add(col.path.split(".", 1)[0])
    except Exception:  # noqa: BLE001 - missing/odd metadata => treat as none
        pass
    return roots


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


# ---------------------------------------------------------------------------
# Predicate lowering: Ray Expr -> native pruning IR (predicate pushdown, part 1)
# ---------------------------------------------------------------------------
# The native crate does statistics-based row-group pruning from a small JSON IR
# (parsed by ``predicate.rs``). We lower the *pushed* Ray ``Expr`` predicate into
# that IR here rather than translating the PyArrow expression, because the Ray
# AST (ColumnExpr / LiteralExpr / BinaryExpr / UnaryExpr) is directly
# introspectable.
#
# The lowering is **total**: any node it can't represent becomes ``{"t":
# "unknown"}``, which the crate treats as "keep this row group". So a partially
# understood predicate like ``a > 5 AND some_udf(b)`` still lowers to
# ``And[cmp(a>5), unknown]`` and prunes soundly on the ``a > 5`` conjunct instead
# of giving up. Pruning is conservative on the Rust side (a group is dropped only
# when provably empty), and the reader re-applies the full predicate post-decode,
# so this only ever avoids IO/decode — it can never change which rows are
# returned.

# Ray comparison Operation -> IR op string.
_CMP_OP_TO_IR = {
    "gt": "gt",
    "lt": "lt",
    "ge": "ge",
    "le": "le",
    "eq": "eq",
    "ne": "ne",
}
# When the column is on the *right* of a comparison (``5 < col``), flip the op so
# the IR always reads ``col OP literal``.
_CMP_OP_FLIP = {"gt": "lt", "lt": "gt", "ge": "le", "le": "ge", "eq": "eq", "ne": "ne"}

_IR_UNKNOWN: Dict[str, Any] = {"t": "unknown"}


def _literal_to_ir_value(value: Any) -> Optional[Dict[str, Any]]:
    """Lower a Python literal to a tagged IR value, or None if the crate can't
    order it for pruning (bytes, datetimes, decimals, ...), which makes the
    enclosing atom ``unknown``. ``bool`` is checked before ``int`` because
    ``bool`` is an ``int`` subclass."""
    if isinstance(value, bool):
        return {"vt": "bool", "v": value}
    if isinstance(value, int):
        return {"vt": "int", "v": value}
    if isinstance(value, float):
        return {"vt": "float", "v": value}
    if isinstance(value, str):
        return {"vt": "str", "v": value}
    if value is None:
        return {"vt": "null"}
    return None


def _predicate_to_ir(expr: "Expr") -> Dict[str, Any]:
    """Lower a Ray Data predicate ``Expr`` to the native pruning IR (see above).

    Total by construction: unrepresentable subtrees become ``_IR_UNKNOWN``.
    """
    from ray.data.expressions import (
        AliasExpr,
        BinaryExpr,
        ColumnExpr,
        LiteralExpr,
        Operation,
        UnaryExpr,
    )

    def unwrap(e: "Expr") -> "Expr":
        # Aliasing doesn't change the value being compared.
        while isinstance(e, AliasExpr):
            e = e.expr
        return e

    def lower(e: "Expr") -> Dict[str, Any]:
        e = unwrap(e)

        if isinstance(e, UnaryExpr):
            if e.op == Operation.NOT:
                return {"t": "not", "pred": lower(e.operand)}
            if e.op in (Operation.IS_NULL, Operation.IS_NOT_NULL):
                operand = unwrap(e.operand)
                if isinstance(operand, ColumnExpr):
                    tag = "is_null" if e.op == Operation.IS_NULL else "is_not_null"
                    return {"t": tag, "col": operand.name}
            return _IR_UNKNOWN

        if isinstance(e, BinaryExpr):
            if e.op in (Operation.AND, Operation.OR):
                tag = "and" if e.op == Operation.AND else "or"
                return {"t": tag, "preds": [lower(e.left), lower(e.right)]}

            if e.op in (Operation.IN, Operation.NOT_IN):
                col = unwrap(e.left)
                rhs = unwrap(e.right)
                if isinstance(col, ColumnExpr) and isinstance(rhs, LiteralExpr):
                    raw = rhs.value
                    raw = raw if isinstance(raw, list) else [raw]
                    values = [_literal_to_ir_value(v) for v in raw]
                    if all(v is not None for v in values):
                        return {
                            "t": "in",
                            "col": col.name,
                            "values": values,
                            "negated": e.op == Operation.NOT_IN,
                        }
                return _IR_UNKNOWN

            ir_op = _CMP_OP_TO_IR.get(e.op.value)
            if ir_op is not None:
                left = unwrap(e.left)
                right = unwrap(e.right)
                if isinstance(left, ColumnExpr) and isinstance(right, LiteralExpr):
                    val = _literal_to_ir_value(right.value)
                    if val is not None:
                        return {"t": "cmp", "col": left.name, "op": ir_op, "value": val}
                elif isinstance(left, LiteralExpr) and isinstance(right, ColumnExpr):
                    val = _literal_to_ir_value(left.value)
                    if val is not None:
                        return {
                            "t": "cmp",
                            "col": right.name,
                            "op": _CMP_OP_FLIP[ir_op],
                            "value": val,
                        }
            return _IR_UNKNOWN

        return _IR_UNKNOWN

    return lower(expr)


def _predicate_json(predicate: "Optional[Expr]") -> Optional[str]:
    """Serialize the pushed predicate's pruning IR for the crate, or None when
    there's nothing prunable (no predicate, or it lowered entirely to
    ``unknown``) so we skip the pushdown argument altogether."""
    if predicate is None:
        return None
    ir = _predicate_to_ir(predicate)
    if ir == _IR_UNKNOWN:
        return None
    return json.dumps(ir)


class _NativeParquetFragment(NamedTuple):
    """A native (pyarrow-free) unit of work for one file's row-group slice.

    The arrow-rs ``read()`` builds these instead of pyarrow ``ParquetFileFragment``
    objects for files the native reader handles, so pyarrow never opens a
    supported file. ``row_groups is None`` means "all row groups in the file"
    (whole-file read). Exposes ``.path`` so it flows through the same
    :meth:`FileReader._dispatch_fragment_reads` threading/retry machinery as a
    pyarrow fragment; :meth:`ArrowRsParquetFileReader._iter_fragment_tables`
    dispatches on the type.
    """

    path: str
    row_groups: Optional[List[int]]


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

    @override
    def read(self, input_split: "FileManifest") -> "Iterator[pa.Table]":
        """Pyarrow-free Parquet read for supported files.

        For files the native reader handles (local/S3, flat + struct/list types,
        no schema evolution / dictionary / extension), the footer, row-group
        layout, and statistics come from the crate's ``read_metadata`` and decode
        from ``read_row_groups`` — pyarrow never opens the file. Files (or whole
        splits) the native path can't handle transparently fall back to the base
        pyarrow ``read()`` / scanner, so correctness is never at risk. The
        format-agnostic finishing (limit, partition/``path``/``row_hash``
        synthesis, projection) is shared with the base reader via
        :meth:`_postprocess`.
        """
        if len(input_split) == 0:
            return

        # Reader-wide ineligibility (unsupported filesystem or format kwargs):
        # nothing native to do — use the base pyarrow read() unchanged.
        if not self._reader_level_supported():
            yield from super().read(input_split)
            return

        plan = self._plan_native_read(input_split)
        if plan is None:
            # A file's footer couldn't be read natively (corrupt / unsupported
            # footer); fall the whole split back to pyarrow rather than reason
            # about a partially-known layout.
            yield from super().read(input_split)
            return

        fragments_with_offsets, columns_to_synthesize, scanner_kwargs = plan
        triples = self._dispatch_fragment_reads(fragments_with_offsets, scanner_kwargs)
        yield from self._postprocess(triples, columns_to_synthesize)

    def _read_native_metadata(
        self, path: str
    ) -> Optional[Tuple[pa.Schema, List[int], List[str]]]:
        """Read one file's footer via the crate: ``(arrow schema, per-row-group
        row counts, int96 root columns)``, or ``None`` if the native footer read
        fails (caller then falls the whole split back to pyarrow). Does *not*
        swallow a missing extension — :meth:`_import_extension` raises that
        loudly. The int96 list lets the gate check that the crate's decoded unit
        for those columns matches PyArrow (see :func:`_is_pyarrow_int96_decode_type`)."""
        # Surfaces a missing extension loudly (import inside the crate call);
        # any *footer-read* failure below becomes a whole-split pyarrow fallback.
        self._import_extension()

        try:
            md = _read_native_metadata_via_crate(path, self._filesystem)
        except Exception as e:  # noqa: BLE001 - any footer failure => fallback
            logger.debug("arrow-rs native metadata read failed for %s: %s", path, e)
            return None
        return pa.schema(md), list(md.row_group_num_rows), list(md.int96_columns)

    def _plan_native_read(
        self, manifest: "FileManifest"
    ) -> Optional[Tuple[List[Tuple[Any, int]], Optional[set], dict]]:
        """Plan a native read: footer-read every file, decide native vs pyarrow
        per file, and build the ordered ``[(fragment, file_row_offset)]`` list
        plus the shared column split and scanner kwargs. Returns ``None`` to
        signal a whole-split pyarrow fallback (some file's footer read failed)."""
        import pyarrow.dataset as pds
        from pyarrow.fs import LocalFileSystem

        from ray.data._internal.datasource_v2.chunkers.parquet_file_chunking_utils import (  # noqa: E501
            _fragments_from_chunk_metadata,
        )

        unique_paths = list(dict.fromkeys(list(manifest.paths)))
        native_md: Dict[str, Tuple[pa.Schema, List[int], List[str]]] = {}
        for path in unique_paths:
            md = self._read_native_metadata(path)
            if md is None:
                return None  # whole-split pyarrow fallback
            native_md[path] = md

        # Column split from the union of on-disk names across files (matches the
        # base reader's ``dataset.schema.names``); partition / path / row_hash
        # columns aren't on disk anywhere and so land in the synthesize set.
        on_disk_names: set = set()
        for schema, _, _ in native_md.values():
            on_disk_names.update(schema.names)
        columns_to_read_from_file, columns_to_synthesize = self._split_columns(
            on_disk_names
        )

        scanner_kwargs = {
            "columns": columns_to_read_from_file,
            "filter": (
                self._predicate.to_pyarrow() if self._predicate is not None else None
            ),
            # The native decode re-derives its per-batch size by byte budget from
            # the footer, so this is only an upper clamp; pyarrow-fallback
            # fragments (nested/dictionary/extension) further clamp it themselves.
            "batch_size": self._explicit_batch_size or _ARROW_DEFAULT_BATCH_SIZE,
            "batch_readahead": _ARROW_SCANNER_BATCH_READAHEAD,
        }
        scanner_kwargs.update(self._arrow_scanner_kwargs())

        read_columns = self._resolve_read_columns_for(scanner_kwargs)

        # Per-file verdict: native decode vs pyarrow fallback.
        native_paths = {
            path
            for path, (schema, _, int96_cols) in native_md.items()
            if self._columns_supported(schema, read_columns, int96_cols)
        }
        fallback_paths = [p for p in unique_paths if p not in native_paths]

        # Build pyarrow fragments for the fallback files only (pyarrow never opens
        # native files). One dataset over the fallback paths; the per-file fan-out
        # reuses the base chunker helper so offsets / row-group slicing match the
        # base path exactly.
        fallback_fragment_by_path: dict = {}
        if fallback_paths:
            fb_dataset = pds.dataset(
                source=fallback_paths,
                format=self._make_format(),
                filesystem=self._filesystem or LocalFileSystem(),
                schema=self._file_dataset_schema,
                ignore_prefixes=self._ignore_prefixes,
            )
            fallback_fragment_by_path = {
                frag.path: frag for frag in fb_dataset.get_fragments()
            }

        fragments_with_offsets: List[Tuple[Any, int]] = []
        for path, chunk_metadata in zip(manifest.paths, manifest.file_chunk_metadatas):
            if path in native_paths:
                fragments_with_offsets.extend(
                    self._native_fragments_for_file(
                        path, chunk_metadata, native_md[path][1]
                    )
                )
            else:
                fragment = fallback_fragment_by_path[path]
                if chunk_metadata is None:
                    fragments_with_offsets.append((fragment, 0))
                else:
                    fragments_with_offsets.extend(
                        _fragments_from_chunk_metadata(fragment, chunk_metadata)
                    )

        return fragments_with_offsets, columns_to_synthesize, scanner_kwargs

    @staticmethod
    def _native_fragments_for_file(
        path: str,
        chunk_metadata: Optional[dict],
        row_group_num_rows: List[int],
    ) -> List[Tuple[_NativeParquetFragment, int]]:
        """Build native fragments for one file, matching the base reader's
        granularity so ``row_hash`` offsets are identical:

        - whole file (``chunk_metadata is None``) → one fragment over *all* row
          groups at offset 0 (the base emits one whole-file fragment);
        - a chunk → one fragment *per row group* in ``[start, end)``, each seeded
          with the pre-filter file row offset of that group's first row (mirrors
          :func:`_fragments_from_chunk_metadata`), so post-filter accumulation
          within a group starts from the right position.
        """
        if chunk_metadata is None:
            return [(_NativeParquetFragment(path, None), 0)]

        total = len(row_group_num_rows)
        start = min(chunk_metadata["row_group_start"], total)
        end = min(chunk_metadata["row_group_end"], total)
        fragments: List[Tuple[_NativeParquetFragment, int]] = []
        file_row_offset = sum(row_group_num_rows[:start])
        for rg in range(start, end):
            fragments.append((_NativeParquetFragment(path, [rg]), file_row_offset))
            file_row_offset += row_group_num_rows[rg]
        return fragments

    @cached_property
    def _pushdown_predicate_json(self) -> Optional[str]:
        """The pushed predicate lowered to the native pruning IR (JSON), or
        ``None`` when there's nothing prunable. Depends only on
        ``self._predicate``, so it's computed once and reused for every
        fragment. See :func:`_predicate_to_ir` for the (total, conservative)
        lowering and the soundness argument."""
        return _predicate_json(self._predicate)

    def _reader_level_supported(self) -> bool:
        """Reader-wide (fragment-independent) half of the support gate:
        filesystem + Parquet-format kwargs. Checked once per ``read()`` before
        touching any file, so an unsupported filesystem / int96 coercion /
        forced dictionary decode short-circuits the whole native path.
        """
        from pyarrow.fs import LocalFileSystem, S3FileSystem

        # Local and S3 are wired in `_iter_fragment_tables` / the native `read()`
        # (S3 uses the windowed, byte-budgeted native path). Any other
        # filesystem (GCS, ABFS, HTTP, …) falls back to PyArrow.
        if not isinstance(self._filesystem, (LocalFileSystem, S3FileSystem)):
            return False

        # int96 coercion and forced dictionary decoding aren't mirrored by the
        # native reader.
        if self._parquet_format_kwargs.get("coerce_int96_timestamp_unit") is not None:
            return False
        if self._parquet_format_kwargs.get("dictionary_columns"):
            return False
        return True

    def _columns_supported(
        self,
        physical_schema: pa.Schema,
        read_columns: Optional[List[str]],
        int96_columns: Optional[List[str]] = None,
    ) -> bool:
        """Per-file half of the support gate: does the native reader handle the
        columns we'd read from *this* file's physical schema?

        Takes a ``pa.Schema`` (the crate's ``read_metadata`` schema — i.e. what
        the crate will actually decode) plus, optionally, the root column names
        the crate reports as INT96-physical. Conservative — anything not covered
        here falls back to PyArrow, so correctness is never at risk:

        - empty projection (count scan) → PyArrow handles the stub-column dance;
        - dotted (nested-column) projection → unsupported;
        - a column absent from this file (schema evolution) → PyArrow null-fill;
        - a type the crate can't decode (:func:`_arrow_rs_type_supported`);
        - a per-file type that differs from the unified schema (needs a cast the
          native reader doesn't do);
        - an INT96 column the crate would decode to a non-ns unit (PyArrow always
          forces ns; see :func:`_is_pyarrow_int96_decode_type`).
        """
        unified_schema = self._file_dataset_schema
        int96 = set(int96_columns or ())
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
                # Column absent from this file (schema evolution) — defer to
                # PyArrow's null-fill path.
                return False
            field_type = physical_schema.field(idx).type
            if not _arrow_rs_type_supported(field_type):
                return False
            # INT96 column: only safe natively when the crate lands on the same
            # type PyArrow produces (ns / no tz). A file embedding a non-ns hint
            # decodes to us/ms/s here but ns in PyArrow — fall it back.
            if name in int96 and not _is_pyarrow_int96_decode_type(field_type):
                return False
            if unified_schema is None:
                continue
            unified_idx = unified_schema.get_field_index(name)
            if unified_idx == -1:
                continue
            # A per-file type that differs from the unified type needs a
            # cast the native reader doesn't do — fall back.
            if unified_schema.field(unified_idx).type != field_type:
                return False
        return True

    def _arrow_rs_supported(
        self,
        fragment: pds.ParquetFileFragment,
        read_columns: Optional[List[str]],
    ) -> bool:
        """Whole-gate verdict for a pyarrow fragment: reader-level checks plus
        the per-file column/type checks against the fragment's physical schema.
        Used by the per-fragment ``_iter_fragment_tables`` path.

        A fragment's ``physical_schema`` is PyArrow's *post-coercion* Arrow schema,
        so an INT96 column already reads as ``timestamp[ns]`` and can't reveal
        whether the crate would decode it differently (it honors an embedded
        non-ns hint). This re-gate can't see the crate's output, so it is
        conservative: any INT96-physical read column falls the fragment back to
        PyArrow. The authoritative plan-time gate (:meth:`_columns_supported` with
        the crate's ``int96_columns``) is what admits INT96→ns files to the native
        path; this path only ever *withholds*, never wrongly admits.
        """
        if not self._reader_level_supported():
            return False
        int96_roots = _pyarrow_fragment_int96_roots(fragment)
        if int96_roots:
            names = (
                read_columns
                if read_columns is not None
                else list(fragment.physical_schema.names)
            )
            if any(name in int96_roots for name in names):
                return False
        return self._columns_supported(fragment.physical_schema, read_columns)

    @staticmethod
    def _import_extension():
        """Import the native extension, raising a clear, actionable error if it
        isn't built. Called on every native entry point so a missing module
        surfaces loudly (never a silent fall back to PyArrow, which would
        corrupt benchmark attribution)."""
        try:
            import ray_data_arrow_rs

            return ray_data_arrow_rs
        except ImportError as e:
            raise ImportError(
                "use_arrow_rs_parquet_reader=True requires the "
                "'ray_data_arrow_rs' extension. Build it with "
                "`maturin develop --release` from "
                "python/ray/data/_internal/datasource_v2/native/ray_data_arrow_rs/."
            ) from e

    def _resolve_read_columns_for(self, scanner_kwargs: dict) -> Optional[List[str]]:
        """The set of columns the native decode must read from the file: the
        projected columns plus any columns referenced only by the pushed filter
        (which we still filter on post-decode). ``None`` means all columns."""
        from ray.data._internal.datasource.parquet_datasource import (
            _resolve_read_columns,
        )
        from ray.data._internal.planner.plan_expression.expression_visitors import (
            get_column_references,
        )

        columns = scanner_kwargs.get("columns")
        filter_expr = scanner_kwargs.get("filter")
        filter_columns = (
            get_column_references(self._predicate)
            if self._predicate is not None
            else None
        )
        return _resolve_read_columns(columns, filter_expr, filter_columns)

    @override
    def _iter_fragment_tables(
        self,
        fragment: pds.Fragment,
        scanner_kwargs: dict,
    ) -> "Iterator[pa.Table]":
        # Native front-end (arrow-rs ``read()``) hands us pyarrow-free work units.
        if isinstance(fragment, _NativeParquetFragment):
            _trace_reader_path(True)
            yield from self._iter_native_tables(
                fragment.path, fragment.row_groups, scanner_kwargs
            )
            return

        # Pyarrow fragment (used when ``read()`` is not overridden, e.g. the
        # reader-level-unsupported delegate, and by unit tests that drive this
        # method directly). Re-check the per-fragment gate and either decode
        # natively or fall back to the PyArrow scanner.
        read_columns = self._resolve_read_columns_for(scanner_kwargs)
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
        yield from self._iter_native_tables(fragment.path, row_groups, scanner_kwargs)

    def _iter_native_tables(
        self,
        path: str,
        row_groups: Optional[List[int]],
        scanner_kwargs: dict,
    ) -> "Iterator[pa.Table]":
        """Decode ``row_groups`` of ``path`` via the native crate and yield
        ``pa.Table`` batches.

        Row-group pruning is native (``predicate.rs``), replacing PyArrow's
        ``fragment.subset(filter=...)``: we hand the crate the row-group ids plus
        the pushed predicate lowered to a JSON IR, and it drops the groups whose
        footer statistics prove no row can match before fetching or decoding
        them. Pruning is conservative by construction — a missing column, absent
        stats, or an uncomparable type all *keep* the group — so it can only ever
        avoid IO/decode, never change which rows surface. Row-level filtering
        then runs post-decode here (the final authority), and a fully-pruned file
        simply yields nothing.
        """
        ray_data_arrow_rs = self._import_extension()

        from pyarrow.fs import S3FileSystem

        columns = scanner_kwargs.get("columns")
        filter_expr = scanner_kwargs.get("filter")
        batch_size = scanner_kwargs.get("batch_size") or _ARROW_DEFAULT_BATCH_SIZE
        read_columns = self._resolve_read_columns_for(scanner_kwargs)
        predicate_json = self._pushdown_predicate_json

        split_threshold = (
            self._target_block_size
            if self._target_block_size is not None
            else _ARROW_RS_DEFAULT_SPLIT_THRESHOLD_BYTES
        )

        fs = self._filesystem
        if isinstance(fs, S3FileSystem):
            bucket, _, key = path.partition("/")
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
                prefetch_windows=_ARROW_RS_PREFETCH_WINDOWS,
                predicate_json=predicate_json,
            )
        else:
            reader = ray_data_arrow_rs.read_row_groups(
                path,
                row_groups,
                read_columns,
                batch_size,
                _ARROW_RS_DECODE_BUDGET_BYTES,
                _ARROW_RS_K,
                split_threshold,
                predicate_json,
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
