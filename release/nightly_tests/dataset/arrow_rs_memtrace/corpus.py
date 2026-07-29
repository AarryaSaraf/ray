"""Correctness corpus + comparison helpers for the `correctness` axis.

This is the *decoder-replacement* test corpus: every type Ray Data can intake
via Parquet, plus the layout/schema-evolution/encoding edge cases, generated
DETERMINISTICALLY (value = f(row id), no RNG) so golden rows can always be
recomputed from the builder code and compared against what a reader returns.

Layout under ``<FIXTURE_ROOT>/corpus_v<N>/`` (local dir or s3:// prefix — same
mechanism as fixtures.py, so pointing RAY_DATA_ARROW_RS_FIXTURE_ROOT at a
bucket puts the whole corpus on S3 with no code change):

  main/                     — the mega-read dir (heterogeneous schemas unified)
    scalars.parquet         — every scalar type; row groups [10, n-11, 1]
    nested.parquet          — deep struct/list/map/fixed_size_list + the
                              flat-column-literally-named-"a.b" ambiguity
    encodings.parquet       — dictionary (incl. overflow-to-plain), DELTA_*,
                              BYTE_STREAM_SPLIT, per-column compression,
                              statistics disabled on some columns
    patho.parquet           — ~200 row groups incl. zero-row groups, unicode/
                              space column names, on-disk `path` + `row_hash`
                              columns, occasional ~512 KiB string cells
    empty.parquet           — 0 rows, full schema
    shards/part-*.parquet   — 10 small same-schema shards + decoy files
                              (_SUCCESS, data.crc) that listing must skip
  int96/                    — INT96 timestamps: with_hint.parquet embeds an
                              ARROW:schema saying timestamp[us] (the non-ns
                              hint); no_hint.parquet has store_schema=False
                              (kept OUT of main/ so unit coercion is scenario-
                              controlled, not schema-unified away)
  evo/year=.../month=.../   — schema-evolution pair under hive dirs: int32 vs
                              int64 id, missing columns both ways, null-typed →
                              string promotion, reordered struct fields, an
                              on-disk column colliding with the hive key
  tensors/                  — same tensor column name with different fixed
                              shapes across two files (forces variable-shape
                              unification) + canonical fixed_shape_tensor +
                              Ray variable-shaped tensors
  pickle/                   — ArrowPythonObjectType column (env-gated read)

Golden rows (first row of each file + last row of every row group) are written
to ``golden.json`` at the corpus root on every build_corpus() call — cheap
because builders are deterministic — and checked by the axis against what the
readers actually return.

Everything the axis compares goes through :func:`norm` (a canonical, JSON-safe
representation: floats via repr with NaN spelled out, bytes as hex, tensors as
["nd", shape, dtype, values], datetimes as isoformat) so "equal" means
user-visible-value equal, independent of batch boundaries or chunking.
"""
import datetime
import decimal
import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import fixtures as fx

CORPUS_VERSION = 1

# id-space base per file so ids are globally unique across the corpus and a
# sorted-by-id comparison is well-defined for the mega read.
BASE = {
    "scalars": 0,
    "nested": 1_000_000,
    "encodings": 2_000_000,
    "patho": 4_000_000,
    "shards": 5_000_000,
    "evo_a": 6_000_000,
    "evo_b": 7_000_000,
    "tensors_fixed": 8_000_000,
    "tensors_ragged": 9_000_000,
    "pickle": 10_000_000,
    "int96": 11_000_000,
}


def corpus_root_uri():
    """(filesystem, write_path, read_uri) of the versioned corpus root."""
    return fx._fixture_uri(f"corpus_v{CORPUS_VERSION}")


def corpus_uri(rel):
    """Read URI for a corpus subdir (e.g. 'main', 'evo') — what read_parquet takes."""
    _, _, uri = corpus_root_uri()
    return f"{uri}/{rel}"


# --------------------------------------------------------------------------- #
# Canonical normalization — the single definition of "equal" for this axis.
# --------------------------------------------------------------------------- #
def norm(v):
    """Canonical JSON-safe representation of one value from take_all()/to_pylist()."""
    if v is None or isinstance(v, (bool, str, int)) and not isinstance(v, np.generic):
        return v
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        # NaN is equal to nothing. Not even itself.
        # https://stackoverflow.com/questions/20320022/why-in-numpy-nan-nan-is-false-while-nan-in-nan-is-true
        if f != f:
            return "NaN"
        return repr(f)  # repr keeps -0.0, inf, subnormals distinct and exact
    if isinstance(v, np.ndarray):
        return [
            "nd",
            list(v.shape),
            str(v.dtype),
            [norm(x) for x in v.ravel().tolist()],
        ]
    if isinstance(v, bytes):
        return v.hex()
    if isinstance(v, decimal.Decimal):
        return str(v)
    if isinstance(v, datetime.datetime):
        return v.isoformat()
    if isinstance(v, (datetime.date, datetime.time)):
        return v.isoformat()
    if isinstance(v, datetime.timedelta):
        return f"td:{v.days}d{v.seconds}s{v.microseconds}us"
    if isinstance(v, dict):
        return {k: norm(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [norm(x) for x in v]
    return repr(v)  # last resort: stable-ish; surfaces as a diff if readers disagree


def norm_rows(rows):
    """Normalize take_all() rows and sort by id when every row has one."""
    out = [{k: norm(v) for k, v in r.items()} for r in rows]
    if out and all("id" in r and isinstance(r["id"], int) for r in out):
        out.sort(key=lambda r: r["id"])
    return out


def diff_rows(a, b, max_report=3):
    """First few differences between two normalized row lists. '' == equal."""
    if len(a) != len(b):
        return f"row count {len(a)} vs {len(b)}"
    msgs = []
    for i, (ra, rb) in enumerate(zip(a, b)):
        if ra == rb:
            continue
        keys = set(ra) | set(rb)
        for k in sorted(keys):
            va, vb = ra.get(k, "<absent>"), rb.get(k, "<absent>")
            if va != vb:
                msgs.append(
                    f"row {i} (id={ra.get('id')}) col {k!r}: "
                    f"{str(va)[:80]} vs {str(vb)[:80]}"
                )
                if len(msgs) >= max_report:
                    return "; ".join(msgs)
    return "; ".join(msgs)


# --------------------------------------------------------------------------- #
# Builders — every column value is a deterministic function of the row id.
# --------------------------------------------------------------------------- #
def _nullable(ids, values):
    """Null out every id % 7 == 3 (the shared null pattern)."""
    return [None if i % 7 == 3 else v for i, v in zip(ids, values)]


def build_scalars(n=5_000):
    """Every scalar type Ray Data can hold, extreme values pinned at rows 0-4."""
    ids = list(range(BASE["scalars"], BASE["scalars"] + n))
    cols = {"id": pa.array(ids, type=pa.int64())}

    def put(name, mk, typ=None):
        # Per-column try: an exotic type unsupported by this pyarrow build is
        # dropped with a warning instead of killing the whole corpus.
        try:
            vals = _nullable(ids, [mk(i) for i in ids])
            cols[name] = pa.array(vals, type=typ)
        except Exception as e:
            print(f"  (scalars: dropping column {name}: {type(e).__name__}: {e})")

    put("b", lambda i: i % 2 == 0, pa.bool_())
    int_pins = {0: -(1 << 7), 1: (1 << 7) - 1}
    put("i8", lambda i: int_pins.get(i, (i % 200) - 100), pa.int8())
    put(
        "i16",
        lambda i: {0: -(1 << 15), 1: (1 << 15) - 1}.get(i, (i % 60000) - 30000),
        pa.int16(),
    )
    put(
        "i32",
        lambda i: {0: -(1 << 31), 1: (1 << 31) - 1}.get(i, i * 977 - 50_000),
        pa.int32(),
    )
    put(
        "i64",
        lambda i: {0: -(1 << 63), 1: (1 << 63) - 1}.get(i, i * 7 - 999_999),
        pa.int64(),
    )
    put("u8", lambda i: {1: 255}.get(i, i % 251), pa.uint8())
    put("u16", lambda i: {1: 65535}.get(i, i % 60013), pa.uint16())
    put(
        "u32",
        lambda i: {1: (1 << 32) - 1}.get(i, (i * 2654435761) % (1 << 32)),
        pa.uint32(),
    )
    put(
        "u64",
        lambda i: {1: (1 << 64) - 1}.get(i, (i * 2654435761) % (1 << 64)),
        pa.uint64(),
    )
    f_pins = {0: float("nan"), 1: float("inf"), 2: float("-inf"), 3: -0.0, 4: 5e-324}
    put("f64", lambda i: f_pins.get(i, i * 1.5), pa.float64())
    put(
        "f32",
        lambda i: {0: float("nan"), 1: -0.0, 2: 1e-45}.get(i, float(i % 1000) / 8),
        pa.float32(),
    )
    put("f16", lambda i: np.float16((i % 100) / 7), pa.float16())
    put("dec128", lambda i: decimal.Decimal(f"{i}.1234567890"), pa.decimal128(38, 10))
    put("dec256", lambda i: decimal.Decimal(f"{i}." + "3" * 20), pa.decimal256(60, 20))
    put(
        "s",
        lambda i: "" if i == 0 else f"row-{i}-" + "α" * (i % 5) + "x" * (i % 37),
        pa.string(),
    )
    put("ls", lambda i: f"large-{i}-" + "y" * (i % 53), pa.large_string())
    put("bin", lambda i: f"b{i}".encode(), pa.binary())
    put("lbin", lambda i: f"lb{i}".encode() * (i % 3 + 1), pa.large_binary())
    put("fbin", lambda i: f"{i:016d}".encode(), pa.binary(16))
    epoch = datetime.date(1970, 1, 1)
    put(
        "d32",
        lambda i: epoch + datetime.timedelta(days=(i % 20000) - 5000),
        pa.date32(),
    )
    put(
        "d64",
        lambda i: epoch + datetime.timedelta(days=(i % 20000) - 5000),
        pa.date64(),
    )
    put("t32", lambda i: (i * 997) % 86_400_000, pa.time32("ms"))
    put("t64", lambda i: (i * 997_001) % 86_400_000_000, pa.time64("us"))
    # timestamps straddle 1970 (negative values) in all four units, ± tz
    put("ts_s", lambda i: (i - 2500) * 3671, pa.timestamp("s"))
    put("ts_ms", lambda i: (i - 2500) * 3_671_003, pa.timestamp("ms"))
    put("ts_us", lambda i: (i - 2500) * 3_671_000_007, pa.timestamp("us"))
    put("ts_ns", lambda i: (i - 2500) * 3_671_000_000_011, pa.timestamp("ns"))
    put("ts_us_utc", lambda i: (i - 2500) * 3_671_000_007, pa.timestamp("us", tz="UTC"))
    put(
        "ts_ns_kol",
        lambda i: (i - 2500) * 3_671_000_000_011,
        pa.timestamp("ns", tz="Asia/Kolkata"),
    )
    put("dur", lambda i: (i - 2500) * 1_000_003, pa.duration("us"))
    cols["all_null_str"] = pa.nulls(n, pa.string())
    cols["all_null_untyped"] = pa.nulls(n)
    return pa.table(cols), [10, n - 11, 1], {}


def build_nested(n=3_000):
    """Deep nesting + null-vs-empty-vs-list-of-null + the dotted-name ambiguity."""
    ids = list(range(BASE["nested"], BASE["nested"] + n))
    deep_t = pa.struct(
        [
            (
                "l1",
                pa.struct(
                    [("l2", pa.struct([("l3", pa.struct([("l4", pa.int64())]))]))]
                ),
            )
        ]
    )
    deep = _nullable(ids, [{"l1": {"l2": {"l3": {"l4": i * 3}}}} for i in ids])
    lst_struct_t = pa.list_(
        pa.struct([("a", pa.list_(pa.int64())), ("b", pa.string())])
    )

    def mk_ls(i):
        if i % 5 == 0:
            return None
        if i % 5 == 1:
            return []
        return [{"a": [i, None, i + 1], "b": f"s{i}"} for _ in range(i % 3 + 1)]

    mp = [
        None if i % 11 == 5 else [(f"k{j}", i + j) for j in range(i % 4)] for i in ids
    ]
    # NOTE: parquet can't store fixed_size_list with OUTER nulls (pyarrow
    # ArrowNotImplementedError), so the null pattern goes on the elements.
    fsl = [
        [float(i), None if i % 7 == 3 else float(i) + 0.5, -1.0, float(i % 9)]
        for i in ids
    ]
    llst = [[f"v{i}"] * (i % 3) for i in ids]

    def mk_tri(i):
        return [None, [], [None], [i, None, i + 2]][i % 4]

    cols = {
        "id": pa.array(ids, type=pa.int64()),
        "deep": pa.array(deep, type=deep_t),
        "lst_struct": pa.array([mk_ls(i) for i in ids], type=lst_struct_t),
        "mp": pa.array(mp, type=pa.map_(pa.string(), pa.int64())),
        "fsl": pa.array(fsl, type=pa.list_(pa.float32(), 4)),
        "llst": pa.array(llst, type=pa.large_list(pa.large_string())),
        "tri": pa.array([mk_tri(i) for i in ids], type=pa.list_(pa.int64())),
        # struct "a" with child "b" NEXT TO a flat column literally named "a.b":
        # both have parquet leaf path "a.b" — projection must not confuse them.
        "a": pa.array(
            [{"b": i * 2, "c": f"c{i}"} for i in ids],
            type=pa.struct([("b", pa.int64()), ("c", pa.string())]),
        ),
        "a.b": pa.array([i * 1000 + 7 for i in ids], type=pa.int64()),
    }
    return pa.table(cols), [1_000, 1_500, 500], {"data_page_version": "2.0"}


def build_encodings(n=20_000):
    """Encoding axis: dictionary (incl. overflow), DELTA_*, BYTE_STREAM_SPLIT,
    per-column compression, statistics disabled on two columns."""
    ids = list(range(BASE["encodings"], BASE["encodings"] + n))
    cols = {
        "id": pa.array(ids, type=pa.int64()),
        "s_dict": pa.array([f"cat-{i % 20}" for i in ids]),  # stays dict
        "s_bigdict": pa.array([f"{i:032d}" for i in ids]),  # dict overflows
        "i_delta": pa.array([i * 3 for i in ids], type=pa.int64()),
        "s_delta": pa.array([f"prefix-common-{i:010d}" for i in ids]),
        "f_bss": pa.array([np.sin(i) for i in ids], type=pa.float64()),
        "nostat_s": pa.array([("Z" * 80) + f"{i:08d}" for i in ids]),
        "nostat_i": pa.array([i ^ 0xABCDEF for i in ids], type=pa.int64()),
    }
    stats_cols = [c for c in cols if not c.startswith("nostat")]
    wkw = {
        "use_dictionary": ["s_dict", "s_bigdict"],
        "dictionary_pagesize_limit": 4096,  # force s_bigdict's dict page to overflow
        "column_encoding": {
            "i_delta": "DELTA_BINARY_PACKED",
            "s_delta": "DELTA_BYTE_ARRAY",
            "f_bss": "BYTE_STREAM_SPLIT",
        },
        "compression": {
            "id": "NONE",
            "s_dict": "SNAPPY",
            "s_bigdict": "LZ4",
            "i_delta": "GZIP",
            "s_delta": "BROTLI",
            "f_bss": "ZSTD",
            "nostat_s": "SNAPPY",
            "nostat_i": "SNAPPY",
        },
        "write_statistics": stats_cols,
    }
    return pa.table(cols), [8_000, 8_000, 4_000], wkw


def build_patho(n=5_001):
    """~200 row groups (some zero-row), unicode/space names, on-disk `path` and
    `row_hash` columns, occasional ~512 KiB string cells."""
    ids = list(range(BASE["patho"], BASE["patho"] + n))
    cols = {
        "id": pa.array(ids, type=pa.int64()),
        "温度": pa.array(_nullable(ids, [i * 0.25 for i in ids]), type=pa.float64()),
        "col space": pa.array([i - 42 for i in ids], type=pa.int64()),
        "path": pa.array([f"on-disk-path-{i}" for i in ids]),
        "row_hash": pa.array([i % 100 for i in ids], type=pa.int32()),
        "hs": pa.array(["X" * (512 * 1024) if i % 997 == 0 else f"s{i}" for i in ids]),
    }
    sizes = [25] * 200 + [1]
    for pos in (150, 100, 50):  # zero-row row groups sprinkled in
        sizes.insert(pos, 0)
    return pa.table(cols), sizes, {}


def build_shard(f, per=500):
    base = BASE["shards"] + f * per
    ids = list(range(base, base + per))
    return pa.table(
        {
            "id": pa.array(ids, type=pa.int64()),
            "v": pa.array([i / 3.0 for i in ids], type=pa.float64()),
            "s": pa.array([f"shard{f}-{i}" for i in ids]),
        }
    )


def build_empty():
    return pa.table(
        {"id": pa.array([], type=pa.int64()), "s": pa.array([], type=pa.string())}
    )


def build_int96(n=2_000, hint=True):
    """INT96 timestamps. hint=True: physical INT96 whose embedded ARROW:schema
    claims ``timestamp[us]`` — the non-ns hint. pyarrow can't produce this
    naturally (its int96 writer coerces to ns BEFORE storing the schema), so the
    hint is hand-crafted: serialize the us-typed schema and attach it as the
    ``ARROW:schema`` key-value entry with ``store_schema=False``. pyarrow 24's
    reader IGNORES the hint (reads ns); a reader that honors it returns us —
    the differential comparison documents whichever way each reader goes.
    hint=False: no embedded schema at all — the bare-INT96 (Spark-ish) case."""
    import base64

    base = BASE["int96"] + (0 if hint else n)
    ids = list(range(base, base + n))
    # values stay within ns-representable range so the no-coerce read is valid
    us_vals = [(i % 4000 - 2000) * 86_400_000_000 + i for i in ids]  # ±~5.5 years
    cols = {
        "id": pa.array(ids, type=pa.int64()),
        "ts96_hint_us": pa.array(_nullable(ids, us_vals), type=pa.timestamp("us")),
        "ts96_ns": pa.array(
            _nullable(ids, [v * 1000 + 1 for v in us_vals]), type=pa.timestamp("ns")
        ),
    }
    table = pa.table(cols)
    if hint:
        # Pre-seeding ARROW:schema with the us-typed schema makes the writer
        # keep OUR entry (store_schema=True would otherwise write the
        # ns-coerced schema); store_schema=False would strip ALL kv metadata.
        b64 = base64.b64encode(table.schema.serialize().to_pybytes()).decode()
        table = table.replace_schema_metadata({"ARROW:schema": b64})
    wkw = {"use_deprecated_int96_timestamps": True, "store_schema": hint}
    return table, [n // 2, n - n // 2], wkw


def build_evo_a(n=2_000):
    ids = list(range(BASE["evo_a"], BASE["evo_a"] + n))
    return (
        pa.table(
            {
                "id": pa.array(ids, type=pa.int32()),  # int32 — B has int64
                "a_extra": pa.array([i * 3 for i in ids], type=pa.int64()),
                "only_a": pa.array(_nullable(ids, [f"a-{i}" for i in ids])),
                "late": pa.nulls(n),  # null-typed; B has string
                "s": pa.array(
                    [{"x": i, "y": f"y{i}"} for i in ids],
                    type=pa.struct([("x", pa.int64()), ("y", pa.string())]),
                ),
            }
        ),
        [n // 2, n - n // 2],
        {},
    )


def build_evo_b(n=2_000):
    ids = list(range(BASE["evo_b"], BASE["evo_b"] + n))
    return (
        pa.table(
            {
                "id": pa.array(ids, type=pa.int64()),
                "late": pa.array([f"late-{i}" for i in ids]),
                # struct fields REORDERED vs A
                "s": pa.array(
                    [{"y": f"y{i}", "x": i} for i in ids],
                    type=pa.struct([("y", pa.string()), ("x", pa.int64())]),
                ),
                "only_b": pa.array(
                    [decimal.Decimal(f"{i % 1000}.25") for i in ids],
                    type=pa.decimal128(10, 2),
                ),
                # collides with the hive key `month` — the path value must win
                "month": pa.array(["file-value"] * n),
            }
        ),
        [n],
        {},
    )


def _fixed_tensor(ids, shape, seedmul):
    from ray.air.util.tensor_extensions.arrow import ArrowTensorType

    ln = int(np.prod(shape))
    flat = np.array(
        [(i * seedmul + j) % 1000 / 8 for i in ids for j in range(ln)], dtype=np.float32
    )
    typ = ArrowTensorType(shape, pa.float32())
    offsets = pa.array(np.arange(0, (len(ids) + 1) * ln, ln, dtype=np.int32))
    return pa.ExtensionArray.from_storage(
        typ, pa.ListArray.from_arrays(offsets, pa.array(flat, type=pa.float32()))
    )


def _canon_tensor(ids, ln=4):
    flat = np.array(
        [(i + j) % 100 / 4 for i in ids for j in range(ln)], dtype=np.float32
    )
    storage = pa.FixedSizeListArray.from_arrays(pa.array(flat, type=pa.float32()), ln)
    return pa.ExtensionArray.from_storage(
        pa.fixed_shape_tensor(pa.float32(), [ln]), storage
    )


def _var_tensor(ids):
    from ray.data._internal.tensor_extensions.arrow import (
        ArrowVariableShapedTensorArray,
    )

    arrs = np.empty(len(ids), dtype=object)
    for k, i in enumerate(ids):
        arrs[k] = np.full((i % 3 + 1, 2), float(i % 50), dtype=np.float32)
    return ArrowVariableShapedTensorArray.from_numpy(arrs)


def build_tensors_fixed(n=1_000):
    ids = list(range(BASE["tensors_fixed"], BASE["tensors_fixed"] + n))
    cols = {
        "id": pa.array(ids, type=pa.int64()),
        "tns": _fixed_tensor(ids, (2, 3), 7),
        "canon": _canon_tensor(ids),
    }
    try:
        cols["vtns"] = _var_tensor(ids)
    except Exception as e:
        print(f"  (tensors: dropping vtns: {type(e).__name__}: {e})")
    return pa.table(cols), [n], {}


def build_tensors_ragged(n=1_000):
    ids = list(range(BASE["tensors_ragged"], BASE["tensors_ragged"] + n))
    cols = {
        "id": pa.array(ids, type=pa.int64()),
        "tns": _fixed_tensor(ids, (7,), 5),  # same name, DIFFERENT shape
        "canon": _canon_tensor(ids),
    }
    try:
        cols["vtns"] = _var_tensor(ids)
    except Exception as e:
        print(f"  (tensors: dropping vtns: {type(e).__name__}: {e})")
    return pa.table(cols), [n], {}


def build_pickle(n=200):
    from ray.data._internal.object_extensions.arrow import ArrowPythonObjectArray

    ids = list(range(BASE["pickle"], BASE["pickle"] + n))
    objs = [{"k": i, "l": [i, i + 1], "t": (i, f"o{i}")} for i in ids]
    return (
        pa.table(
            {
                "id": pa.array(ids, type=pa.int64()),
                "obj": ArrowPythonObjectArray.from_objects(objs),
            }
        ),
        [n],
        {},
    )


# (relpath, builder) — relpath is where the file lands under the corpus root.
_FILES = [
    ("main/scalars.parquet", build_scalars),
    ("main/nested.parquet", build_nested),
    ("main/encodings.parquet", build_encodings),
    ("main/patho.parquet", build_patho),
    ("int96/with_hint.parquet", lambda: build_int96(hint=True)),
    ("int96/no_hint.parquet", lambda: build_int96(hint=False)),
    ("evo/year=2024/month=1/a.parquet", build_evo_a),
    ("evo/year=2025/month=2/b.parquet", build_evo_b),
    ("tensors/fixed.parquet", build_tensors_fixed),
    ("tensors/ragged.parquet", build_tensors_ragged),
    ("pickle/objects.parquet", build_pickle),
]


def _write_file(fs, full, table, rg_sizes, wkw):
    kw = dict(compression="snappy", write_page_index=True)
    kw.update(wkw)
    # column_encoding and write_page_index are mutually exclusive in pyarrow
    # (the page-index write path doesn't support the non-dictionary encodings).
    if "column_encoding" in kw:
        kw.pop("write_page_index", None)

    def _do(kwargs):
        with pq.ParquetWriter(full, table.schema, filesystem=fs, **kwargs) as w:
            off = 0
            for sz in rg_sizes:
                w.write_table(table.slice(off, sz), row_group_size=max(sz, 1))
                off += sz
            assert off == table.num_rows, f"{full}: rg sizes {off} != {table.num_rows}"

    try:
        _do(kw)
    except Exception as e:
        # Exotic writer kwargs unsupported by this pyarrow → degrade to a plain
        # write so the corpus (and axis) still runs; the print marks the loss.
        print(
            f"  (corpus: {os.path.basename(full)} degraded write "
            f"({type(e).__name__}: {e}); retrying plain)"
        )
        _do(dict(compression="snappy"))


def _golden_for(table, rg_sizes):
    """Golden ids/rows: first row + last row of every non-empty row group."""
    if table.num_rows == 0:
        return {"ids": [], "rows": {}, "columns": table.column_names, "num_rows": 0}
    positions = {0}
    off = 0
    for sz in rg_sizes:
        off += sz
        if sz > 0:
            positions.add(off - 1)
    ids, rows = [], {}
    id_col = table.column("id")
    for p in sorted(positions):
        rid = id_col[p].as_py()
        row = {k: norm(v) for k, v in table.slice(p, 1).to_pylist()[0].items()}
        ids.append(rid)
        rows[str(rid)] = row
    return {
        "ids": ids,
        "rows": rows,
        "columns": table.column_names,
        "num_rows": table.num_rows,
    }


def build_corpus(verbose=True):
    """Idempotent: writes any missing corpus file, ALWAYS recomputes golden.json
    (cheap — builders are deterministic). Returns (root_uri, golden dict)."""
    import pyarrow.fs as pafs

    fs, root, uri = corpus_root_uri()
    golden = {"version": CORPUS_VERSION, "files": {}}
    for rel, builder in _FILES:
        table, rg_sizes, wkw = builder()
        golden["files"][rel] = _golden_for(table, rg_sizes)
        full = f"{root}/{rel}"
        info = fs.get_file_info(full)
        if info.type == pafs.FileType.NotFound:
            if not fx._is_s3():
                os.makedirs(os.path.dirname(full), exist_ok=True)
            if verbose:
                print(
                    f"  corpus: writing {rel} ({table.num_rows} rows, "
                    f"{len(rg_sizes)} rgs)"
                )
            _write_file(fs, full, table, rg_sizes, wkw)
    # empty file + shard files + listing decoys
    empty_full = f"{root}/main/empty.parquet"
    golden["files"]["main/empty.parquet"] = _golden_for(build_empty(), [])
    if fs.get_file_info(empty_full).type == pafs.FileType.NotFound:
        pq.write_table(build_empty(), empty_full, filesystem=fs)
    for f in range(10):
        t = build_shard(f)
        rel = f"main/shards/part-{f:04d}.parquet"
        golden["files"][rel] = _golden_for(t, [t.num_rows])
        full = f"{root}/{rel}"
        if fs.get_file_info(full).type == pafs.FileType.NotFound:
            if not fx._is_s3():
                os.makedirs(os.path.dirname(full), exist_ok=True)
            pq.write_table(
                t, full, filesystem=fs, row_group_size=t.num_rows, compression="snappy"
            )
    for decoy in ("main/shards/_SUCCESS", "main/shards/data.crc"):
        full = f"{root}/{decoy}"
        if fs.get_file_info(full).type == pafs.FileType.NotFound:
            with fs.open_output_stream(full) as out:
                out.write(b"decoy -- listing must skip this\n")
    with fs.open_output_stream(f"{root}/golden.json") as out:
        out.write(json.dumps(golden).encode())
    return uri, golden


def check_golden(payload_rows, golden, dirs=("main",)):
    """Compare an actually-read normalized row set against golden rows of every
    corpus file under `dirs`. Only the file's OWN columns are compared (the
    unified read null-fills the rest). Returns (n_checked, [mismatch strings])."""
    by_id = {r["id"]: r for r in payload_rows if isinstance(r.get("id"), int)}
    checked, bad = 0, []
    for rel, g in golden["files"].items():
        if not any(rel.startswith(d + "/") for d in dirs):
            continue
        for rid in g["ids"]:
            grow = g["rows"][str(rid)]
            got = by_id.get(rid)
            if got is None:
                bad.append(f"{rel}: id {rid} missing from read")
                continue
            for col in g["columns"]:
                checked += 1
                if got.get(col, "<absent>") != grow[col]:
                    bad.append(
                        f"{rel}: id {rid} col {col!r}: "
                        f"read={str(got.get(col, '<absent>'))[:60]} "
                        f"golden={str(grow[col])[:60]}"
                    )
    return checked, bad


# --------------------------------------------------------------------------- #
# Scenarios — each runs once per reader in its own run dir (=> USS graphs).
# --------------------------------------------------------------------------- #
SCENARIOS = [
    # The mega read: every file in main/ unified in one dataset; golden-checked.
    {"name": "mega_main", "dir": "main", "golden": True},
    # Dotted-name ambiguity: flat "a.b" vs struct a.b — must pick the flat column.
    {"name": "proj_dotted", "dir": "nested", "columns": ["id", "a.b"]},
    {"name": "proj_struct", "dir": "nested", "columns": ["id", "a"]},
    # Column present in one evo file, missing (→ null-fill) in the other.
    {"name": "proj_missing", "dir": "evo", "columns": ["id", "only_a"]},
    # Empty projection: pure count; row preservation via the stub column.
    {"name": "proj_empty_count", "dir": "main", "columns": [], "count_only": True},
    # Row-group pruning on the scalars file (3 rgs; predicate isolates the last).
    {
        "name": "filter_prune",
        "dir": "scalars_file",
        "filter": f"id >= {BASE['scalars'] + 4_989}",
    },
    # Filter referencing a column absent from evo_b: B must contribute 0 rows.
    {
        "name": "filter_missing_col",
        "dir": "evo",
        "filter": f"a_extra > {(BASE['evo_a'] + 1_000) * 3}",
    },
    # Filter on a hive partition key.
    {
        "name": "filter_partition",
        "dir": "evo",
        "hive": True,
        "filter": "year == '2025'",
    },
    # Limit pushdown (preserve_order is on session-wide → deterministic prefix).
    {"name": "limit_prefix", "dir": "main", "limit": 2_000},
    # INT96: default read (embedded non-ns hint must be honored) vs coerced unit.
    {"name": "int96_default", "dir": "int96"},
    {
        "name": "int96_coerce_ms",
        "dir": "int96",
        "dataset_kwargs": {"coerce_int96_timestamp_unit": "ms"},
    },
    # Dictionary-typed output requested via format kwargs.
    {
        "name": "dictionary_cols",
        "dir": "encodings",
        "dataset_kwargs": {"dictionary_columns": ["s_dict"]},
    },
    # Same tensor column name, different fixed shapes → variable-shape unification.
    {"name": "tensors_unify", "dir": "tensors"},
    # Hive partitions + path + row_hash synthesis (incl. on-disk `month` collision).
    {
        "name": "hive_full",
        "dir": "evo",
        "hive": True,
        "include_paths": True,
        "include_row_hash": True,
    },
    # Same read but one row group per read task: content + row hashes must be
    # IDENTICAL to hive_full (chunker offset bookkeeping).
    {
        "name": "hive_chunked",
        "dir": "evo",
        "hive": True,
        "include_paths": True,
        "include_row_hash": True,
        "chunk_per_group": True,
    },
    # Pickled-object column: guarded error by default, round-trip with the env var.
    {"name": "pickle_default", "dir": "pickle", "expect_error": "pickle"},
    {
        "name": "pickle_autoload",
        "dir": "pickle",
        "extra_env": {"RAY_DATA_AUTOLOAD_PICKLE_OBJECT_SCALAR": "1"},
    },
]

# hive_chunked must reproduce hive_full exactly, per reader.
STABILITY_PAIRS = [("hive_full", "hive_chunked")]


# Scenario "dir" aliases → actual corpus paths (single files live under main/).
_DIR_ALIASES = {
    "scalars_file": "main/scalars.parquet",
    "nested": "main/nested.parquet",
    "encodings": "main/encodings.parquet",
}


def scenario_path(sc):
    return corpus_uri(_DIR_ALIASES.get(sc["dir"], sc["dir"]))


def run_scenario(sc, path):
    """Execute one scenario read; return a comparable normalized payload."""
    import ray

    kw = {}
    if sc.get("columns") is not None:
        kw["columns"] = sc["columns"]
    if sc.get("hive"):
        from ray.data.datasource.partitioning import Partitioning

        kw["partitioning"] = Partitioning("hive")
    if sc.get("include_paths"):
        kw["include_paths"] = True
    if sc.get("include_row_hash"):
        kw["include_row_hash"] = True
    if sc.get("dataset_kwargs"):
        kw["dataset_kwargs"] = dict(sc["dataset_kwargs"])
    try:
        ds = ray.data.read_parquet(path, **kw)
        if sc.get("filter"):
            ds = ds.filter(expr=sc["filter"])
        if sc.get("limit"):
            ds = ds.limit(sc["limit"])
        if sc.get("count_only"):
            return {"count": ds.count()}
        schema_str = str(ds.schema())
        rows = norm_rows(ds.take_all())
        return {"count": len(rows), "schema": schema_str, "rows": rows}
    except Exception as e:
        # A RayTaskError's str() is a long traceback with the root cause at the
        # END, so truncating the head used to lose it (expected_error_hit then
        # searched 300 chars of traceback header). Record the unwrapped cause:
        # its type is the class parity compares, its message is what
        # expect_error substrings match against.
        cause = getattr(e, "cause", None)
        # RayTaskError's class name already embeds the cause class, e.g.
        # "RayTaskError(ValueError)" — no need to repeat it.
        msg = str(cause if cause is not None else e)[:300]
        return {"error": f"{type(e).__name__}: {msg}"}


def compare_payloads(a, b):
    """('OK'|'FAIL', detail) between two payloads from run_scenario."""
    if "error" in a or "error" in b:
        ea, eb = a.get("error"), b.get("error")
        if ea and eb:
            same = ea.split(":")[0] == eb.split(":")[0]
            return ("OK" if same else "FAIL"), f"both errored ({ea} | {eb})"
        return "FAIL", f"one reader errored: {ea or eb}"
    msgs = []
    if a.get("count") != b.get("count"):
        msgs.append(f"count {a.get('count')} vs {b.get('count')}")
    if a.get("schema") != b.get("schema"):
        msgs.append("schema differs")
    if "rows" in a and "rows" in b:
        d = diff_rows(a["rows"], b["rows"])
        if d:
            msgs.append(d)
    return ("OK" if not msgs else "FAIL"), "; ".join(msgs)
