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
import glob
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


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

# Schemas the native arrow-rs path is expected to handle (flat, non-extension).
# Everything else should fall back to PyArrow.
NATIVE_SCHEMAS = {"int", "float", "wide_str", "large_str", "huge_str"}


def _write_mixed_row_groups(table, path, sizes):
    """Write `table` split into row groups whose sizes cycle through `sizes`
    (the "many large and small groups" layout, which a single row_group_size
    can't express)."""
    with pq.ParquetWriter(path, table.schema, write_page_index=True,
                          compression="snappy") as w:
        off = 0
        i = 0
        n = table.num_rows
        while off < n:
            sz = sizes[i % len(sizes)]
            w.write_table(table.slice(off, sz), row_group_size=sz)
            off += sz
            i += 1


def make_fixture(name, spec):
    """Build (or reuse) a parquet fixture on disk from a spec; return its dir.

    spec keys: rows, num_files, schema, and EITHER row_group_size (uniform) OR
    row_group_sizes (a list cycled per group, for the mixed layout).
    """
    rows = spec["rows"]
    num_files = spec["num_files"]
    schema = spec["schema"]
    d = os.path.join(DATA, name)
    if os.path.isdir(d) and glob.glob(os.path.join(d, "*.parquet")):
        return d
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(0)
    per = rows // num_files
    build = SCHEMA_BUILDERS[schema]
    for f in range(num_files):
        table = pa.table(build(rng, per))
        out = os.path.join(d, f"part-{f:04d}.parquet")
        if "row_group_sizes" in spec:
            _write_mixed_row_groups(table, out, spec["row_group_sizes"])
        else:
            pq.write_table(table, out, row_group_size=spec["row_group_size"],
                           write_page_index=True, compression="snappy")
    return d


def expected_path(schema):
    return "native" if schema in NATIVE_SCHEMAS else "fallback"
