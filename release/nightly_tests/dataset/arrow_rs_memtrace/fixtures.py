"""Parameterized Parquet fixture generator for the arrow-rs benchmark suite.

Factored across the axes the benchmark varies (schema dtype, file/row-group
layout, size). Every fixture writes ``write_page_index=True`` (required by the
arrow-rs K-split path) and snappy compression, matching the reader's assumptions.

A fixture is described by a small spec dict so the driver can build the whole
matrix declaratively:

    {
      "rows": int,              # total rows across all files
      "num_files": int,
      "row_group_size": int,    # rows per row group
      "schema": str,            # key into SCHEMA_BUILDERS
    }

Schema builders return an ordered dict of column-name -> pyarrow Array. Nested
(struct/list) and canonical-extension (fixed_shape_tensor) schemas exist to
exercise the reader's PyArrow *fallback* gate, not the native path.
"""
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# Where fixtures live. Default: a local ./data dir (macOS dev). Point this at an
# s3://bucket/prefix to write + read the WHOLE suite on S3 (the Linux/PR run) with
# no code change — every axis calls make_fixture(), which now writes through the
# right filesystem and returns an s3:// URI that ray.data.read_parquet reads
# natively. Idempotent: a prefix that already holds .parquet is reused, so re-runs
# don't re-upload gigabytes.
FIXTURE_ROOT = os.environ.get("RAY_DATA_ARROW_RS_FIXTURE_ROOT", DATA)


def _is_s3(root=None):
    return (root if root is not None else FIXTURE_ROOT).startswith("s3://")


def _s3_filesystem():
    """S3FileSystem from the ambient env (region + optional MinIO endpoint) — the
    same construction run_s3_benchmark uses for its fixtures."""
    import pyarrow.fs as pafs

    kw = {}
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if region:
        kw["region"] = region
    endpoint = os.environ.get("AWS_ENDPOINT_URL")
    if endpoint:
        kw["endpoint_override"] = endpoint
        kw["scheme"] = "http" if endpoint.startswith("http://") else "https"
    return pafs.S3FileSystem(**kw)


def _fs_for_root():
    """Return (filesystem, base_path_without_scheme, uri_scheme_prefix) for the
    active FIXTURE_ROOT. Local -> ('', abspath); S3 -> ('s3://', 'bucket/prefix')."""
    import pyarrow.fs as pafs

    if _is_s3():
        return _s3_filesystem(), FIXTURE_ROOT[len("s3://") :].rstrip("/"), "s3://"
    return pafs.LocalFileSystem(), os.path.abspath(FIXTURE_ROOT), ""


def _has_parquet(fs, path):
    import pyarrow.fs as pafs

    sel = pafs.FileSelector(path, allow_not_found=True, recursive=True)
    return any(info.path.endswith(".parquet") for info in fs.get_file_info(sel))


def _fixture_uri(name):
    """(filesystem, write_path, read_uri) for a fixture dir called `name` under the
    root. write_path is what pyarrow writers take (no scheme); read_uri is what
    ray.data.read_parquet takes (s3:// on S3, a plain dir locally)."""
    fs, base, scheme = _fs_for_root()
    path = f"{base}/{name}"
    return fs, path, (f"{scheme}{path}" if scheme else path)


def _ints(rng, n, ncols=8):
    cols = {"id": pa.array(np.arange(n, dtype=np.int64))}
    for i in range(ncols):
        cols[f"i{i}"] = pa.array(rng.integers(0, 1 << 30, n, dtype=np.int64))
    return cols


def _floats(rng, n, ncols=8):
    cols = {"id": pa.array(np.arange(n, dtype=np.int64))}
    for i in range(ncols):
        cols[f"f{i}"] = pa.array(rng.random(n))
    return cols


_LETTERS_U = np.array(list("abcdefghijklmnopqrstuvwxyz"), dtype="U1")


def _str_cols(rng, n, ncols, width):
    # Fully vectorized random fixed-width strings: index a unicode alphabet, then
    # view the (n, width) 'U1' block as (n,) 'U{width}' so pyarrow ingests it with
    # no Python-level per-row loop.
    cols = {"id": pa.array(np.arange(n, dtype=np.int64))}
    for c in range(ncols):
        idx = rng.integers(0, 26, size=(n, width))
        rows = np.ascontiguousarray(_LETTERS_U[idx]).view(f"U{width}").reshape(n)
        cols[f"s{c}"] = pa.array(rows)
    return cols


def _wide_str(rng, n):
    # Many short string columns (16 chars) — the "wide row" case.
    return _str_cols(rng, n, ncols=6, width=16)


def _large_str(rng, n):
    # Few very long string columns (256 chars) — the "fat cell" case.
    return _str_cols(rng, n, ncols=2, width=256)


def _huge_str(rng, n):
    # Matches the §5.1 headline fixture: 3 string cols x 48 chars (+ id).
    return _str_cols(rng, n, ncols=3, width=48)


def _struct(rng, n):
    a = pa.array(rng.integers(0, 1 << 20, n, dtype=np.int64))
    b = pa.array(rng.random(n))
    st = pa.StructArray.from_arrays([a, b], names=["a", "b"])
    return {"id": pa.array(np.arange(n, dtype=np.int64)), "st": st}


def _list(rng, n, ln=8):
    values = pa.array(rng.integers(0, 1 << 20, n * ln, dtype=np.int64))
    offsets = pa.array(np.arange(0, (n + 1) * ln, ln, dtype=np.int32))
    lst = pa.ListArray.from_arrays(offsets, values)
    return {"id": pa.array(np.arange(n, dtype=np.int64)), "lst": lst}


def _ray_tensor(rng, n, shape=(4,)):
    # Ray's own ArrowTensorType — isinstance(pa.ExtensionType) is True, so the
    # gate correctly falls back. This is what real Ray Data users have. Its
    # storage is a (variable) list, so we wrap a ListArray, not fixed_size_list.
    from ray.air.util.tensor_extensions.arrow import ArrowTensorType

    ln = int(np.prod(shape))
    typ = ArrowTensorType(shape, pa.float32())
    flat = pa.array(rng.random(n * ln).astype(np.float32), type=pa.float32())
    offsets = pa.array(np.arange(0, (n + 1) * ln, ln, dtype=np.int32))
    storage = pa.ListArray.from_arrays(offsets, flat)
    tarr = pa.ExtensionArray.from_storage(typ, storage)
    return {"id": pa.array(np.arange(n, dtype=np.int64)), "tns": tarr}


def _canonical_tensor(rng, n, shape=(4,)):
    # pyarrow canonical fixed_shape_tensor — NOT isinstance(pa.ExtensionType) in
    # pyarrow 24, nested=False, so the current gate MISSES it (latent bug). This
    # fixture exists to catch that empirically via the path trace.
    flat = pa.array(rng.random(n * shape[0]).astype(np.float32), type=pa.float32())
    storage = pa.FixedSizeListArray.from_arrays(flat, shape[0])
    tarr = pa.ExtensionArray.from_storage(
        pa.fixed_shape_tensor(pa.float32(), list(shape)), storage
    )
    return {"id": pa.array(np.arange(n, dtype=np.int64)), "tns": tarr}


SCHEMA_BUILDERS = {
    "int": _ints,
    "float": _floats,
    "wide_str": _wide_str,
    "large_str": _large_str,
    "huge_str": _huge_str,
    "struct": _struct,
    "list": _list,
    "ray_tensor": _ray_tensor,
    "canonical_tensor": _canonical_tensor,
}

# Schemas the native arrow-rs path is expected to handle: flat types plus
# struct/list nesting (ungated 2026-07-21). Extension types (both tensor
# flavors) still fall back to PyArrow.
NATIVE_SCHEMAS = {"int", "float", "wide_str", "large_str", "huge_str",
                  "struct", "list"}


def _write_mixed_row_groups(table, path, sizes, filesystem=None):
    """Write `table` split into row groups whose sizes cycle through `sizes`
    (the "many large and small groups" layout, which a single row_group_size
    can't express)."""
    with pq.ParquetWriter(path, table.schema, write_page_index=True,
                          compression="snappy", filesystem=filesystem) as w:
        off = 0
        i = 0
        n = table.num_rows
        while off < n:
            sz = sizes[i % len(sizes)]
            w.write_table(table.slice(off, sz), row_group_size=sz)
            off += sz
            i += 1


def make_fixture(name, spec):
    """Build (or reuse) a parquet fixture from a spec; return its read URI.

    The URI is a local dir path when FIXTURE_ROOT is local, or an ``s3://`` URI when
    it points at a bucket — ``ray.data.read_parquet`` handles both. Idempotent: a
    dir already holding .parquet is reused (so an S3 prefix isn't re-uploaded).

    spec keys: rows, num_files, schema, and EITHER row_group_size (uniform) OR
    row_group_sizes (a list cycled per group, for the mixed layout).
    """
    rows = spec["rows"]
    num_files = spec["num_files"]
    schema = spec["schema"]
    fs, path, uri = _fixture_uri(name)
    if _has_parquet(fs, path):
        return uri
    if not _is_s3():
        os.makedirs(path, exist_ok=True)  # S3 has no dirs; the object write makes it
    rng = np.random.default_rng(0)
    per = rows // num_files
    build = SCHEMA_BUILDERS[schema]
    for f in range(num_files):
        table = pa.table(build(rng, per))
        out = f"{path}/part-{f:04d}.parquet"
        if "row_group_sizes" in spec:
            _write_mixed_row_groups(table, out, spec["row_group_sizes"], filesystem=fs)
        else:
            pq.write_table(table, out, row_group_size=spec["row_group_size"],
                           write_page_index=True, compression="snappy", filesystem=fs)
    return uri


def make_mixed_fixture(name="mixed7_tensor", per=400_000):
    """The heterogeneous 7-file dataset in one dir: 6 native schemas (struct is
    native since the 2026-07-21 ungate) + 1 ray_tensor file, which is still
    extension-gated — so the axis keeps proving that a mixed dataset routes
    native and fallback files correctly in ONE read. Location-aware like
    make_fixture; kept here so it goes to S3 with the rest of the suite."""
    fs, path, uri = _fixture_uri(name)
    if _has_parquet(fs, path):
        return uri
    if not _is_s3():
        os.makedirs(path, exist_ok=True)
    rng = np.random.default_rng(0)
    specs = [("int", _ints), ("float", _floats), ("wide_str", _wide_str),
             ("large_str", _large_str), ("huge_str", _huge_str), ("struct", _struct),
             ("ray_tensor", _ray_tensor)]
    for i, (nm, build) in enumerate(specs):
        tbl = pa.table(build(rng, per))
        pq.write_table(tbl, f"{path}/part-{i:04d}_{nm}.parquet", row_group_size=per,
                       write_page_index=True, compression="snappy", filesystem=fs)
    return uri


def expected_path(schema):
    return "native" if schema in NATIVE_SCHEMAS else "fallback"
