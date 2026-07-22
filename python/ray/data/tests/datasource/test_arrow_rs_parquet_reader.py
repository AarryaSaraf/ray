"""Correctness + integration tests for the experimental arrow-rs Parquet reader.

These run only when the native ``ray_data_arrow_rs`` extension is importable
(built via ``maturin`` from the crate under
``_internal/datasource_v2/native/ray_data_arrow_rs/``); otherwise the whole
module is skipped. They confirm that:

- reading through the arrow-rs path yields byte-identical columns to PyArrow,
- the native decode path actually runs (not the PyArrow fallback), and
- unsupported schemas transparently fall back to PyArrow and stay correct.
"""
import os

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

import ray
from ray.data.context import DataContext
from ray.data.datasource.path_util import _unwrap_protocol

ray_data_arrow_rs = pytest.importorskip("ray_data_arrow_rs")


@pytest.fixture
def restore_ctx():
    ctx = DataContext.get_current()
    v2, arrow_rs = ctx.use_datasource_v2, ctx.use_arrow_rs_parquet_reader
    try:
        ctx.use_datasource_v2 = True
        yield ctx
    finally:
        ctx.use_datasource_v2 = v2
        ctx.use_arrow_rs_parquet_reader = arrow_rs


def _flat_table(num_rows=20_000):
    rng = np.random.default_rng(0)
    return pa.table(
        {
            "id": pa.array(np.arange(num_rows, dtype=np.int64)),
            "x": pa.array(rng.random(num_rows)),
            "label": pa.array((np.arange(num_rows) % 5).astype(np.int32)),
            "name": pa.array([f"row-{i}" for i in range(num_rows)]),
        }
    )


def _read_sorted(path, use_arrow_rs, restore_ctx, **read_kwargs):
    restore_ctx.use_arrow_rs_parquet_reader = use_arrow_rs
    ds = ray.data.read_parquet(str(path), **read_kwargs)
    return pa.Table.from_pandas(ds.to_pandas()).sort_by("id")


@pytest.mark.parametrize("row_group_size", [20_000, 5_000])
def test_arrow_rs_parity_full_scan(tmp_path, restore_ctx, row_group_size):
    """arrow-rs and PyArrow produce identical tables (full scan)."""
    path = tmp_path / "data.parquet"
    table = _flat_table()
    pq.write_table(
        table, str(path), write_page_index=True, row_group_size=row_group_size
    )

    pa_tbl = _read_sorted(path, False, restore_ctx)
    rs_tbl = _read_sorted(path, True, restore_ctx)

    assert pa_tbl.num_rows == rs_tbl.num_rows == table.num_rows
    assert pa_tbl.equals(rs_tbl)


def test_arrow_rs_parity_with_projection(tmp_path, restore_ctx):
    path = tmp_path / "data.parquet"
    table = _flat_table()
    pq.write_table(table, str(path), write_page_index=True)

    pa_tbl = _read_sorted(path, False, restore_ctx, columns=["id", "x"])
    rs_tbl = _read_sorted(path, True, restore_ctx, columns=["id", "x"])
    assert rs_tbl.column_names == ["id", "x"]
    assert pa_tbl.equals(rs_tbl)


@pytest.mark.parametrize("row_group_size", [20_000, 5_000])
def test_arrow_rs_parity_sum(tmp_path, restore_ctx, row_group_size):
    """The aggregation workload benchmarked in Agents.md §3.3 (``ds.sum()``) must
    return identical results via the arrow-rs decode path and PyArrow. This is a
    decode-heavy / output-light consumer: the read decodes every value and the
    aggregation collapses it to a scalar, so it exercises full-column decode
    correctness end-to-end through Ray's aggregation, not just a raw table read.
    """
    path = tmp_path / "data.parquet"
    table = _flat_table()
    pq.write_table(
        table, str(path), write_page_index=True, row_group_size=row_group_size
    )

    # Ground truth from the source table, independent of either reader.
    expected_id = pc.sum(table["id"]).as_py()
    expected_label = pc.sum(table["label"]).as_py()

    restore_ctx.use_arrow_rs_parquet_reader = False
    pa_sum = ray.data.read_parquet(str(path)).sum(["id", "label"])
    restore_ctx.use_arrow_rs_parquet_reader = True
    rs_sum = ray.data.read_parquet(str(path)).sum(["id", "label"])

    assert rs_sum == pa_sum
    assert rs_sum["sum(id)"] == expected_id
    assert rs_sum["sum(label)"] == expected_label


def _read_crate_stream(path, **kwargs):
    """Read a file straight through the crate (bypassing the reader) into a
    single table, so we can force the K-split path via ``split_threshold_bytes``
    / ``k`` explicitly."""
    stream = ray_data_arrow_rs.read_row_groups(str(path), **kwargs)
    return pa.RecordBatchReader.from_stream(stream).read_all()


@pytest.mark.parametrize("k", [2, 4, 8])
def test_kspilt_parity_and_order(tmp_path, k):
    """The intra-fragment K-split path (single big row group, forced via
    ``split_threshold_bytes=0``) must be byte-identical to both the sequential
    (k=1) crate path and PyArrow, and preserve row order across the K parallel
    range workers.

    Row order is the load-bearing property here: the split decodes K disjoint
    row ranges on separate threads and merges them back. A merge bug would
    surface as a shuffled ``id`` column even when the row *set* is correct, so
    we assert the ``id`` column is exactly ``0..n-1`` in order.
    """
    num_rows = 50_000
    path = tmp_path / "big_single_rg.parquet"
    table = _flat_table(num_rows)
    # One row group covering all rows → a lone fragment Ray's pool can't split.
    pq.write_table(table, str(path), write_page_index=True, row_group_size=num_rows)
    assert pq.ParquetFile(str(path)).num_row_groups == 1

    # k=1 sequential (never splits) vs forced K-split (threshold=0).
    seq = _read_crate_stream(path, k=1)
    split = _read_crate_stream(path, k=k, split_threshold_bytes=0)

    # Byte-identical to the sequential path and to the source table.
    assert split.equals(seq)
    assert split.equals(table)
    # Order preserved across ranges: id is exactly 0..n-1, not just the right set.
    assert split.column("id").to_pylist() == list(range(num_rows))


def test_native_path_actually_runs(tmp_path):
    """Directly exercise the reader and confirm it calls the native extension
    rather than silently falling back to PyArrow."""
    import pyarrow.dataset as pds
    from pyarrow.fs import LocalFileSystem

    from ray.data._internal.datasource_v2.readers.arrow_rs_parquet_file_reader import (
        ArrowRsParquetFileReader,
    )

    path = tmp_path / "data.parquet"
    table = _flat_table()
    pq.write_table(table, str(path), write_page_index=True)

    calls = {"n": 0}
    orig = ray_data_arrow_rs.read_row_groups

    def wrapped(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    ray_data_arrow_rs.read_row_groups = wrapped
    try:
        reader = ArrowRsParquetFileReader(
            filesystem=LocalFileSystem(), target_block_size=128 * 1024 * 1024
        )
        dataset = pds.dataset(str(path), format="parquet", filesystem=LocalFileSystem())
        fragment = next(dataset.get_fragments())
        scanner_kwargs = {
            "columns": None,
            "filter": None,
            "batch_size": reader._resolve_batch_size(dataset),
        }
        got = pa.concat_tables(
            list(reader._iter_fragment_tables(fragment, scanner_kwargs))
        )
    finally:
        ray_data_arrow_rs.read_row_groups = orig

    assert calls["n"] > 0, "native read_row_groups was not called (fell back)"
    assert got.sort_by("id").equals(table.sort_by("id"))


def test_filter_pushdown_prunes_row_groups(tmp_path, restore_ctx):
    """A pushed-down predicate must prune row groups via footer statistics
    BEFORE the crate fetches/decodes them (mirroring the PyArrow path's
    ``fragment.subset``). On a sorted ``id`` over 4 row groups, ``id >= 3000``
    must reach the crate as ``row_groups=[3]``; a predicate no row group can
    satisfy must not call the crate at all. Results stay byte-correct."""
    import pyarrow.dataset as pds
    from pyarrow.fs import LocalFileSystem

    from ray.data._internal.datasource_v2.readers.arrow_rs_parquet_file_reader import (
        ArrowRsParquetFileReader,
    )

    path = tmp_path / "sorted.parquet"
    n = 4000
    table = pa.table(
        {
            "id": pa.array(np.arange(n, dtype=np.int64)),
            "x": pa.array(np.arange(n) * 0.5),
        }
    )
    pq.write_table(table, str(path), write_page_index=True, row_group_size=1000)

    seen = []
    orig = ray_data_arrow_rs.read_row_groups

    def wrapped(path_, row_groups, *a, **k):
        seen.append(row_groups)
        return orig(path_, row_groups, *a, **k)

    reader = ArrowRsParquetFileReader(
        filesystem=LocalFileSystem(), target_block_size=128 * 1024 * 1024
    )
    dataset = pds.dataset(str(path), format="parquet", filesystem=LocalFileSystem())
    fragment = next(dataset.get_fragments())
    batch_size = reader._resolve_batch_size(dataset)

    ray_data_arrow_rs.read_row_groups = wrapped
    try:
        got = pa.concat_tables(
            list(
                reader._iter_fragment_tables(
                    fragment,
                    {
                        "columns": None,
                        "filter": pds.field("id") >= 3000,
                        "batch_size": batch_size,
                    },
                )
            )
        )
        pruned_all = list(
            reader._iter_fragment_tables(
                fragment,
                {
                    "columns": None,
                    "filter": pds.field("id") >= 10**9,
                    "batch_size": batch_size,
                },
            )
        )
    finally:
        ray_data_arrow_rs.read_row_groups = orig

    assert seen == [[3]], f"stats pruning didn't reach the crate: {seen}"
    assert got.sort_by("id").equals(table.slice(3000))
    assert pruned_all == [] and len(seen) == 1, "fully-pruned fragment hit the crate"


def test_filter_pushdown_e2e_parity(tmp_path, restore_ctx):
    """``ds.filter(expr=...)`` goes through the PredicatePushdown rule into the
    read; both readers must agree (and match ground truth) on a sorted
    multi-row-group file where pruning actually kicks in."""
    path = tmp_path / "sorted_e2e.parquet"
    n = 4000
    table = pa.table(
        {
            "id": pa.array(np.arange(n, dtype=np.int64)),
            "x": pa.array(np.arange(n) * 0.5),
        }
    )
    pq.write_table(table, str(path), write_page_index=True, row_group_size=1000)

    def _read(use_arrow_rs):
        restore_ctx.use_arrow_rs_parquet_reader = use_arrow_rs
        ds = ray.data.read_parquet(str(path)).filter(expr="id >= 3500")
        return pa.Table.from_pandas(ds.to_pandas()).sort_by("id")

    pa_tbl = _read(False)
    rs_tbl = _read(True)
    assert pa_tbl.num_rows == rs_tbl.num_rows == 500
    assert pa_tbl.equals(rs_tbl)


def _gate_verdict(path, read_columns=None):
    """The support gate's verdict for a file, via a real fragment."""
    import pyarrow.dataset as pds
    from pyarrow.fs import LocalFileSystem

    from ray.data._internal.datasource_v2.readers.arrow_rs_parquet_file_reader import (
        ArrowRsParquetFileReader,
    )

    reader = ArrowRsParquetFileReader(filesystem=LocalFileSystem())
    frag = next(
        pds.dataset(
            str(path), format="parquet", filesystem=LocalFileSystem()
        ).get_fragments()
    )
    return reader._arrow_rs_supported(frag, read_columns)


@pytest.mark.parametrize(
    "colname,builder",
    [
        ("vals", lambda n: pa.array([[i, i + 1, None] for i in range(n)])),
        (
            "st",
            lambda n: pa.StructArray.from_arrays(
                [pa.array(np.arange(n)), pa.array(np.arange(n) * 0.5)],
                names=["a", "b"],
            ),
        ),
        (
            "st_nested",
            lambda n: pa.array(
                [{"a": [i, i + 1], "b": {"c": f"row-{i}"}} for i in range(n)]
            ),
        ),
    ],
)
def test_nested_column_native_parity(tmp_path, restore_ctx, colname, builder):
    """List, struct, and deeper struct/list nesting decode NATIVELY (the gate
    admits them) and stay byte-identical to PyArrow."""
    path = tmp_path / f"{colname}.parquet"
    n = 2000
    table = pa.table(
        {"id": pa.array(np.arange(n, dtype=np.int64)), colname: builder(n)}
    )
    pq.write_table(table, str(path), write_page_index=True)

    assert _gate_verdict(path) is True, f"{colname} should be native now"
    pa_tbl = _read_sorted(path, False, restore_ctx)
    rs_tbl = _read_sorted(path, True, restore_ctx)
    assert pa_tbl.equals(rs_tbl)


def test_falls_back_for_map_and_nested_extension(tmp_path, restore_ctx):
    """Map columns, and extension types hiding INSIDE nesting, stay gated: the
    recursive type check must reject them anywhere in the type tree."""
    n = 500
    # Map column → fallback (still correct end-to-end).
    map_path = tmp_path / "map.parquet"
    map_table = pa.table(
        {
            "id": pa.array(np.arange(n, dtype=np.int64)),
            "m": pa.array(
                [[(f"k{i}", i)] for i in range(n)],
                type=pa.map_(pa.string(), pa.int64()),
            ),
        }
    )
    pq.write_table(map_table, str(map_path), write_page_index=True)
    assert _gate_verdict(map_path) is False
    pa_tbl = _read_sorted(map_path, False, restore_ctx)
    rs_tbl = _read_sorted(map_path, True, restore_ctx)
    assert pa_tbl.equals(rs_tbl)

    # Extension type nested inside a struct → the recursive check must catch it
    # (a top-level-only check would let it slip through to the crate).
    from ray.data._internal.datasource_v2.readers.arrow_rs_parquet_file_reader import (
        _arrow_rs_type_supported,
    )

    tensor = pa.fixed_shape_tensor(pa.float32(), [2])
    assert _arrow_rs_type_supported(pa.struct([("t", tensor)])) is False
    assert _arrow_rs_type_supported(pa.list_(tensor)) is False
    assert (
        _arrow_rs_type_supported(
            pa.struct([("a", pa.int64()), ("b", pa.list_(pa.string()))])
        )
        is True
    )


def test_falls_back_for_canonical_extension(tmp_path, restore_ctx):
    """A pyarrow *canonical* extension type (fixed_shape_tensor) is NOT
    ``isinstance(pa.ExtensionType)`` on some pyarrow versions, so it once slipped
    the support gate into the native crate. The gate must reject any type with an
    ``extension_name`` and fall back to PyArrow, staying correct."""
    if not hasattr(pa, "fixed_shape_tensor"):
        pytest.skip("pyarrow lacks fixed_shape_tensor")
    path = tmp_path / "tensor.parquet"
    n = 1000
    flat = pa.array(np.random.rand(n * 4).astype(np.float32), type=pa.float32())
    storage = pa.FixedSizeListArray.from_arrays(flat, 4)
    tarr = pa.ExtensionArray.from_storage(
        pa.fixed_shape_tensor(pa.float32(), [4]), storage
    )
    pq.write_table(
        pa.table({"id": pa.array(np.arange(n, dtype=np.int64)), "tns": tarr}),
        str(path),
        write_page_index=True,
    )

    import pyarrow.dataset as pds
    from pyarrow.fs import LocalFileSystem

    from ray.data._internal.datasource_v2.readers.arrow_rs_parquet_file_reader import (
        ArrowRsParquetFileReader,
    )

    reader = ArrowRsParquetFileReader(filesystem=LocalFileSystem())
    frag = next(
        pds.dataset(
            str(path), format="parquet", filesystem=LocalFileSystem()
        ).get_fragments()
    )
    # Gate must reject it (extension_name present) even though isinstance may be False.
    assert reader._arrow_rs_supported(frag, None) is False
    # And the end-to-end read stays byte-identical via the fallback.
    pa_tbl = _read_sorted(path, False, restore_ctx)
    rs_tbl = _read_sorted(path, True, restore_ctx)
    assert pa_tbl.equals(rs_tbl)


def test_arrow_rs_supported_gate(tmp_path):
    """Unit-check the fallback gate: local flat AND struct/list = supported;
    nested projection / empty projection / non-local = unsupported."""
    import pyarrow.dataset as pds
    from pyarrow.fs import LocalFileSystem

    from ray.data._internal.datasource_v2.readers.arrow_rs_parquet_file_reader import (
        ArrowRsParquetFileReader,
    )

    flat = tmp_path / "flat.parquet"
    pq.write_table(_flat_table(1000), str(flat), write_page_index=True)
    nested = tmp_path / "nested.parquet"
    pq.write_table(
        pa.table({"id": pa.array([1, 2]), "v": pa.array([[1], [2]])}), str(nested)
    )

    reader = ArrowRsParquetFileReader(filesystem=LocalFileSystem())

    flat_frag = next(
        pds.dataset(
            str(flat), format="parquet", filesystem=LocalFileSystem()
        ).get_fragments()
    )
    nested_frag = next(
        pds.dataset(
            str(nested), format="parquet", filesystem=LocalFileSystem()
        ).get_fragments()
    )

    assert reader._arrow_rs_supported(flat_frag, None) is True
    assert reader._arrow_rs_supported(flat_frag, ["id", "x"]) is True
    # Empty projection → fall back.
    assert reader._arrow_rs_supported(flat_frag, []) is False
    # List column → native (ungated 2026-07-21).
    assert reader._arrow_rs_supported(nested_frag, None) is True
    # Nested-column projection (dotted name) → fall back.
    assert reader._arrow_rs_supported(nested_frag, ["v.item"]) is False

    # Unknown filesystem (neither local nor S3) → fall back.
    reader_no_fs = ArrowRsParquetFileReader(filesystem=None)
    assert reader_no_fs._arrow_rs_supported(flat_frag, None) is False


# ---------------------------------------------------------------------------
# S3 (moto server). The crate reads S3 through the Rust `object_store` client,
# so it needs a real HTTP endpoint — Ray Data's `s3_server` fixture (a moto
# server) provides one. These prove the native path (a) connects with the same
# endpoint/credentials/region PyArrow used (recovered from the S3FileSystem via
# `_s3_config`), and (b) returns byte-identical data.
# ---------------------------------------------------------------------------


def _s3_write(table, s3_fs, s3_path, name="data.parquet"):
    """Write ``table`` as one file into the moto S3 dir, return the s3:// URI."""
    base = _unwrap_protocol(s3_path)  # strip s3:// → bucket/key
    key = os.path.join(base, name)
    pq.write_table(table, key, filesystem=s3_fs, write_page_index=True)
    return f"s3://{key}"


def _read_s3_sorted(uri, s3_fs, use_arrow_rs, restore_ctx, **read_kwargs):
    restore_ctx.use_arrow_rs_parquet_reader = use_arrow_rs
    ds = ray.data.read_parquet(uri, filesystem=s3_fs, **read_kwargs)
    return pa.Table.from_pandas(ds.to_pandas()).sort_by("id")


def test_s3_config_recovers_endpoint_and_creds(s3_fs):
    """`_s3_config` must recover the moto endpoint, static creds, region, and
    (critically) allow_http from an http:// endpoint whose `scheme` field still
    reads 'https' — otherwise the crate can't reach moto/MinIO."""
    from ray.data._internal.datasource_v2.readers.arrow_rs_parquet_file_reader import (
        _s3_config,
    )

    cfg = _s3_config(s3_fs)
    assert cfg["endpoint"] and cfg["endpoint"].startswith("http://")
    assert cfg["allow_http"] is True
    assert cfg["region"] == "us-west-2"
    assert cfg["access_key_id"] == "testing"
    assert cfg["secret_access_key"] == "testing"
    assert cfg["anonymous"] is False


def test_arrow_rs_s3_parity(s3_fs, s3_path, restore_ctx):
    """Full-scan parity reading from (moto) S3 via the native path vs PyArrow."""
    table = _flat_table()
    uri = _s3_write(table, s3_fs, s3_path)

    pa_tbl = _read_s3_sorted(uri, s3_fs, False, restore_ctx)
    rs_tbl = _read_s3_sorted(uri, s3_fs, True, restore_ctx)

    assert rs_tbl.num_rows == table.num_rows
    assert pa_tbl.equals(rs_tbl)


def test_arrow_rs_s3_parity_with_projection(s3_fs, s3_path, restore_ctx):
    table = _flat_table()
    uri = _s3_write(table, s3_fs, s3_path)

    pa_tbl = _read_s3_sorted(uri, s3_fs, False, restore_ctx, columns=["id", "x"])
    rs_tbl = _read_s3_sorted(uri, s3_fs, True, restore_ctx, columns=["id", "x"])
    assert rs_tbl.column_names == ["id", "x"]
    assert pa_tbl.equals(rs_tbl)


def test_arrow_rs_s3_native_path_runs(s3_fs, s3_path):
    """Confirm an S3 fragment actually goes through the native crate's S3 entry
    point (``read_row_groups_s3``), not a silent PyArrow fallback. Exercised via
    the reader directly (in-process) so the monkeypatch can observe the call —
    a ``ray.data.read_parquet`` read would run in a worker the driver can't patch.
    """
    import pyarrow.dataset as pds

    from ray.data._internal.datasource_v2.readers.arrow_rs_parquet_file_reader import (
        ArrowRsParquetFileReader,
    )

    table = _flat_table()
    _s3_write(table, s3_fs, s3_path)
    base = _unwrap_protocol(s3_path)

    calls = {"n": 0}
    orig = ray_data_arrow_rs.read_row_groups_s3

    def wrapped(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    ray_data_arrow_rs.read_row_groups_s3 = wrapped
    try:
        reader = ArrowRsParquetFileReader(
            filesystem=s3_fs, target_block_size=128 * 1024 * 1024
        )
        dataset = pds.dataset(base, format="parquet", filesystem=s3_fs)
        fragment = next(dataset.get_fragments())
        scanner_kwargs = {
            "columns": None,
            "filter": None,
            "batch_size": reader._resolve_batch_size(dataset),
        }
        got = pa.concat_tables(
            list(reader._iter_fragment_tables(fragment, scanner_kwargs))
        )
    finally:
        ray_data_arrow_rs.read_row_groups_s3 = orig

    assert calls["n"] > 0, "native read_row_groups_s3 was not called (fell back)"
    assert got.sort_by("id").equals(table.sort_by("id"))


def test_arrow_rs_s3_sum(s3_fs, s3_path, restore_ctx):
    """The `ds.sum()` aggregation workload (§3.3) over S3 must match PyArrow and
    ground truth — decode-heavy / output-light through the native S3 path."""
    table = _flat_table()
    uri = _s3_write(table, s3_fs, s3_path)
    expected = pc.sum(table["id"]).as_py()

    restore_ctx.use_arrow_rs_parquet_reader = False
    pa_sum = ray.data.read_parquet(uri, filesystem=s3_fs).sum("id")
    restore_ctx.use_arrow_rs_parquet_reader = True
    rs_sum = ray.data.read_parquet(uri, filesystem=s3_fs).sum("id")

    assert rs_sum == pa_sum == expected


@pytest.mark.parametrize("k", [2, 4, 8])
def test_arrow_rs_s3_kspilt_windowed_order(s3_fs, s3_path, k):
    """The windowed K-split S3 path (lone big row group → K concurrent GET streams,
    plus a small fetch window slicing each range into sub-windows) must return rows
    in EXACT file order — not just the right set. Forces the path by calling the
    native entry point directly with a tiny split threshold and window; a monotone
    `id` column then makes any range/window mis-ordering a hard failure.
    """
    from ray.data._internal.datasource_v2.readers.arrow_rs_parquet_file_reader import (
        _s3_config,
    )

    n = 50_000
    table = pa.table({"id": pa.array(np.arange(n, dtype=np.int64))})
    _s3_write(table, s3_fs, s3_path, name="mono.parquet")

    base = _unwrap_protocol(s3_path)
    bucket, _, key = os.path.join(base, "mono.parquet").partition("/")
    cfg = _s3_config(s3_fs)

    stream = ray_data_arrow_rs.read_row_groups_s3(
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
        row_groups=[0],
        columns=["id"],
        batch_size=4096,
        fetch_window_mb=1,  # force sub-window slicing within each range
        k=k,
        split_threshold_bytes=1,  # force the K-split path on a small file
    )
    got = pa.RecordBatchReader.from_stream(stream).read_all()
    assert got.num_rows == n
    # Exact order: id must be 0,1,2,...,n-1 with no reordering across K ranges.
    assert got["id"].to_pylist() == list(range(n))


def test_arrow_rs_s3_struct_parity(s3_fs, s3_path, restore_ctx):
    """A struct column over S3 decodes natively (the gate admits struct now,
    on S3 exactly as it does locally) and stays byte-identical to PyArrow."""
    table = pa.table(
        {
            "id": pa.array(np.arange(2000, dtype=np.int64)),
            "st": pa.StructArray.from_arrays(
                [pa.array(np.arange(2000)), pa.array(np.arange(2000) * 0.5)],
                names=["a", "b"],
            ),
        }
    )
    uri = _s3_write(table, s3_fs, s3_path, name="struct.parquet")

    restore_ctx.use_arrow_rs_parquet_reader = False
    pa_tbl = pa.Table.from_pandas(
        ray.data.read_parquet(uri, filesystem=s3_fs).to_pandas()
    ).sort_by("id")
    restore_ctx.use_arrow_rs_parquet_reader = True
    rs_tbl = pa.Table.from_pandas(
        ray.data.read_parquet(uri, filesystem=s3_fs).to_pandas()
    ).sort_by("id")
    assert pa_tbl.equals(rs_tbl)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
