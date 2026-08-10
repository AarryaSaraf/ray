#!/usr/bin/env python3
"""Four Parquet layouts the TPC-H input cannot produce, written local and to S3.

Why lineitem is not enough
--------------------------
Every S3 measurement so far ran on TPC-H ``lineitem`` sf10: ~6.0M rows per file
split into 49 row groups of ~122,428 rows and ~5.3 MB compressed each, i.e.
about **43 compressed bytes per row**. That single property silently disabled
the knob we spent a phase measuring. ``window_rows_for`` (crate ``lib.rs:955``)
turns ``fetch_window_mb`` into a ROW COUNT by dividing by compressed bytes/row,
then ``plan_s3_units`` clamps the window to the row group's length. At 43 B/row
even the smallest window we tried, 16 MiB, asks for ~390,000 rows against a
122,428-row group -- so it clamps to the whole group, and 16 / 64 / 128 MiB all
plan the **same one unit**. Phase W's 892 -> 618 MiB was real, but it cannot
have been the window doing it.

So this file builds the layouts that isolate the things lineitem confounds:

``bigrg`` -- ONE row group per file, high-entropy so it does not compress.
    16 float64 columns of uniform random bits: 128 bytes/row decoded, and Snappy
    cannot shrink random mantissas -- measured, it comes out at **161 compressed
    bytes/row**, slightly LARGER than decoded once page and index overhead is
    counted. That is 3.7x lineitem's 43, and the whole file is one row group of
    ~``--bigrg-mib``, so the window finally binds: at the 256 MiB default a
    16 MiB window plans ~20 units where a 128 MiB window plans ~2. This is also the
    layout the whole project targets -- the lone big row group where PyArrow
    must materialize the entire decoded group and the crate is supposed not to
    -- and no window or budget measurement has ever been taken on it inside
    Ray, only in the standalone harness.

``tiny`` -- the same 16 columns, ~2,000 rows per file, a few hundred KiB.
    A read task that decodes essentially nothing. Its per-task USS is the FIXED
    cost of being a read task at all, measured rather than extrapolated. It was
    built to check an apparent +117 MiB fixed S3 penalty (fitted intercepts of
    306 for arrow-rs against PyArrow's 189) and it **deleted** it: at D ~= 0,
    arrow-rs reads 114 MiB against PyArrow's 117 on S3, and 109 against 110
    locally. A straight line through four visibly concave points necessarily
    overshoots at the low end, so an intercept here is an extrapolation and not
    a measurement -- the lesson generalises to every fit in these docs. Writing
    the fixture to BOTH local disk and S3 also splits the constant in two,
    because the local path memory-maps and constructs no HTTP client, no
    connection pool and no tokio runtime: (arrow_rs s3 - arrow_rs local) is the
    transport's share (+4 MiB, against PyArrow's +6) and the rest is the
    reader's (-1 MiB).

``wide`` -- 64 columns, row groups deliberately fat enough to trip the
    COLUMN-group path. ``plan_s3_units`` sends a row group down
    ``RgDecode::Hstack`` instead of ``RgDecode::Windows`` when its projected
    compressed bytes exceed ``column_fetch_mb`` (16 MiB), and every measurement
    in exp6/exp7 so far has reported ``col_group_rgs = 0`` -- on lineitem the
    branch is unreachable (16 columns, ~5.3 MB compressed per group, an order of
    magnitude under the budget) and ``bigrg`` is 16 columns too. So the one
    branch the 2026-08-05 root-cause blamed has never executed under a per-task
    USS measurement inside Ray.

    Why it should matter: the Hstack arm decodes column group 0 to completion,
    then group 1, ... accumulating EVERY batch of EVERY group in
    ``group_batches`` before it emits the first stitched row
    (``lib.rs`` ~1645). Peak therefore tracks the whole DECODED row group and
    ``decode_budget_bytes`` bounds nothing at all -- which is precisely what
    PyArrow's scanner does, so the prediction is **parity, not regression**: on
    wide schemas arrow-rs should simply stop being memory-advantaged rather than
    become worse. The standalone Linux/S3 run agrees (5000 columns, cf=16 gave
    4.30 GB against PyArrow's 6.78; see the ``column_fetch_mb`` docstring), but
    standalone drops its output where a Ray read task retains it.

    Two things make this measurable rather than inferred. The profiler already
    emits ``col_group_rgs`` on the ``s3_plan`` record and both ``retained_bytes``
    and ``rg_uncompressed_bytes`` on ``s3_rg``, so "did the branch fire" and "did
    it hold the whole group" are read off directly. And ``column_fetch_mb=0``
    disables column grouping outright, giving a same-fixture A/B where the only
    difference is the branch taken.

``tensors`` -- the same geometry as ``wide`` with one column carrying pyarrow's
    canonical ``fixed_shape_tensor`` extension type. The support gate rejects any
    field with an ``extension_name``, so this fixture tests a question about the
    single worst release regression (``wide_schema_pipeline_tensors``, read op
    **4.90x**) that no primitive fixture can reach: whether that case runs on the
    arrow-rs path at all, or falls back to PyArrow in both arms -- in which case
    the gap is the cost of *deciding* to fall back, not of decoding. See
    :func:`_tensor_table`.

All four layouts get ``write_page_index=True``: the crate requires the page
index to plan byte-budgeted reads, and without it the read silently falls back.

Data is generated locally and pushed with ``aws s3 sync`` rather than written
through ``pyarrow.fs`` -- the same choice ``arrow_rs_probe/gen_s3_fixtures.py``
made, because the CLI picks up the instance role and region without the
per-filesystem plumbing.

Idempotent: an existing local fixture whose file count and row count already
match is left alone, so re-running before every experiment costs one listing.

    python make_fixtures.py --bucket s3://arrowrs-bench-21f6c795/arrow_rs_probe
    python make_fixtures.py --bucket s3://... --cases bigrg --bigrg-mib 512
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

MiB = 1024 * 1024
# 16 float64 columns = 128 bytes/row decoded. Deliberately NOT strings: a string
# column drawn from a pool dictionary-encodes down to almost nothing, which
# would put compressed bytes/row back near lineitem's 43 and re-disable the
# window this fixture exists to exercise. Uniform random doubles have full
# mantissa entropy, so Snappy returns them nearly unchanged.
N_COLS = 16
BYTES_PER_ROW = N_COLS * 8


def _table(n_rows: int, n_cols: int, seed: int) -> pa.Table:
    rng = np.random.default_rng(seed)
    return pa.table({f"c{i}": pa.array(rng.random(n_rows)) for i in range(n_cols)})


# Elements per tensor cell in the `tensors` fixture. 4x4 float64 = 128 B/cell, so
# one tensor column carries as many bytes per row as all 16 columns of `bigrg`.
TENSOR_SHAPE = (4, 4)


def _tensor_row_divisor(n_cols: int) -> int:
    """Per-row float64 slots in a ``tensors`` row: the tensor column's cells plus
    the ``n_cols - 1`` plain columns beside it. Dividing ``rows x n_cols`` by this
    gives the row count whose decoded bytes match a ``wide`` file of ``n_cols``
    float64 columns, which is the only way the two fixtures' ratios compare.
    """
    return int(np.prod(TENSOR_SHAPE)) + max(n_cols - 1, 0)


def _tensor_table(n_rows: int, n_cols: int, seed: int) -> pa.Table:
    """A schema mixing pyarrow canonical extension columns with plain float64.

    This exists to answer a question about the WORST regression in the release
    A/B (``wide_schema_pipeline_tensors``, read op 4.90x) that no experiment can
    answer with primitives: **is that case even on the arrow-rs path?**

    The reader's support gate rejects any field carrying an ``extension_name``
    -- including pyarrow's own canonical ``fixed_shape_tensor`` -- and routes the
    whole file to the PyArrow fallback by documented decision. If the release
    tensors case falls back, then both arms of that A/B decoded through PyArrow
    and a 4.90x gap cannot be the decoder: it would have to be the cost of
    deciding to fall back (a native footer read, then a pyarrow read of the same
    file). That is a very different bug from a slow decode, and the profiler
    reports which happened directly -- ``kind: "fallback"`` records carry the
    reason string.

    Mixed deliberately, one extension column among primitives: the gate is
    per-FILE, so a single extension column is enough to divert a file whose other
    columns the native path handles perfectly well. That is the realistic shape
    (nobody writes an all-tensor table) and it is the pessimistic one.
    """
    rng = np.random.default_rng(seed)
    cells = int(np.prod(TENSOR_SHAPE))
    data = {}
    # One extension column is sufficient to trip a per-file gate; a second only
    # costs generation time.
    values = rng.random(n_rows * cells).reshape((n_rows, *TENSOR_SHAPE))
    data["t0"] = pa.FixedShapeTensorArray.from_numpy_ndarray(values)
    for i in range(max(n_cols - 1, 0)):
        data[f"c{i}"] = pa.array(rng.random(n_rows))
    return pa.table(data)


def _manifest_path(d: str) -> str:
    return os.path.join(d, ".fixture.json")


def _already_built(d: str, want: dict) -> bool:
    """True when this directory already holds exactly the fixture asked for.

    Compares a recorded manifest rather than just counting files: a previous run
    with a different ``--bigrg-mib`` leaves the right NUMBER of parquet files
    holding the wrong amount of data, and every downstream fit is against MiB
    decoded per task, so silently reusing it would put a wrong x-axis under a
    right-looking table.
    """
    try:
        with open(_manifest_path(d)) as handle:
            return json.load(handle) == want
    except (OSError, ValueError):
        return False


def _write(
    d: str,
    n_files: int,
    rows_per_file: int,
    n_cols: int,
    row_group_rows: int,
    want: dict,
    kind: str = "plain",
):
    os.makedirs(d, exist_ok=True)
    builder = {"plain": _table, "tensor": _tensor_table}[kind]
    for f in range(n_files):
        path = os.path.join(d, f"part{f}.parquet")
        table = builder(rows_per_file, n_cols, seed=f)
        pq.write_table(
            table,
            path,
            write_page_index=True,
            compression="snappy",
            # `row_group_rows >= rows_per_file` gives exactly one row group.
            # pyarrow starts a new row group per write_table call regardless, so
            # the whole table has to be built and written at once; that is why
            # --bigrg-mib is bounded by RAM rather than by patience.
            row_group_size=row_group_rows,
        )
        # Read the decoded size off the table rather than computing
        # rows x cols x 8: a tensor cell is a whole sub-array, so the arithmetic
        # would understate `tensors` by the number of elements per cell.
        decoded = table.nbytes
        del table
        on_disk = os.path.getsize(path)
        meta = pq.read_metadata(path)
        rg0 = meta.row_group(0)
        # RowGroupMetaData.total_byte_size is UNCOMPRESSED; the compressed size
        # has to be summed over the column chunks. The distinction matters here
        # more than usual: partition_columns_by_budget packs against per-column
        # COMPRESSED bytes, so using the uncompressed number would over-predict
        # how many column groups the planner makes.
        rg0_compressed = sum(
            rg0.column(i).total_compressed_size for i in range(rg0.num_columns)
        )
        per_col_rg = rg0_compressed / max(n_cols, 1)
        print(
            f"  {os.path.basename(path)}: {rows_per_file:,} rows x {n_cols} cols, "
            f"{meta.num_row_groups} row group(s), "
            f"{decoded / MiB:,.0f} MiB decoded, "
            f"{on_disk / MiB:,.0f} MiB on disk "
            f"({on_disk / max(rows_per_file, 1):.0f} compressed B/row); "
            f"row group 0: {rg0_compressed / MiB:,.1f} MiB compressed, "
            f"{per_col_rg / 1024:,.0f} KiB per column",
            flush=True,
        )
    with open(_manifest_path(d), "w") as handle:
        json.dump(want, handle)


def sync_up(local_dir: str, s3_prefix: str):
    # --delete so a re-generated fixture replaces the old one instead of leaving
    # part4..partN from a larger previous run for --max-files to pick up.
    print(f"  aws s3 sync --delete {local_dir} -> {s3_prefix}", flush=True)
    subprocess.run(
        [
            "aws",
            "s3",
            "sync",
            "--delete",
            "--only-show-errors",
            "--exclude",
            ".fixture.json",
            local_dir,
            s3_prefix,
        ],
        check=True,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--bucket", required=True, help="s3://bucket/prefix to sync under")
    p.add_argument(
        "--cases",
        default="tiny,bigrg,wide,tensors",
        help="comma list of {tiny,bigrg,wide,tensors}",
    )
    p.add_argument(
        "--local-root",
        default=os.path.expanduser("~/arrow_rs_local/fixtures"),
        help="kept, not deleted: phase Z needs a LOCAL arm of the same bytes to "
        "separate the S3 client's fixed cost from the reader's.",
    )
    p.add_argument("--tiny-files", type=int, default=4)
    p.add_argument("--tiny-rows", type=int, default=2000)
    p.add_argument("--bigrg-files", type=int, default=2)
    p.add_argument(
        "--bigrg-mib",
        type=int,
        default=256,
        help="decoded MiB in the single row group of each file. 256 is a "
        "compromise: big enough that a 16 MiB fetch window plans ~20 units "
        "against a 128 MiB window's ~2, small enough that ~500 MiB crosses the "
        "network per arm and a 30-arm sweep is minutes rather than an hour.",
    )
    p.add_argument(
        "--wide-cols",
        type=int,
        default=64,
        help="float64 columns in the `wide` fixture. The column-group path fires "
        "only when ONE row group's projected compressed bytes exceed "
        "column_fetch_mb (16 MiB), and partition_columns_by_budget packs whole "
        "columns, so what matters is (per-column compressed bytes per row group) "
        "x cols > 16 MiB with more than one column's worth of slack. 64 columns "
        "at --wide-rg-mib 64 puts ~1 MiB of compressed data in each column per "
        "row group, so the greedy packer makes 4 groups of 16.",
    )
    p.add_argument(
        "--wide-rg-mib",
        type=int,
        default=64,
        help="decoded MiB per row group in the `wide` fixture. Must stay UNDER "
        "the crate's split_threshold (128 MiB): a row group past that gets "
        "K-split by rows, which sets `whole=false` in plan_s3_units and disables "
        "column grouping entirely (a K-split sub-range is a tall group being "
        "split by rows, not columns). 64 leaves a 2x margin.",
    )
    p.add_argument("--wide-rgs-per-file", type=int, default=4)
    p.add_argument(
        "--wide-files",
        type=int,
        default=4,
        help="4 files x 4 row groups x 64 MiB = 256 MiB decoded per file. Ray's "
        "1 GiB chunker default gives one read task per file, so D = 256 MiB per "
        "task -- on the steep part of the USS-vs-D curve, where a lost "
        "saturation advantage is visible rather than swamped by the floor.",
    )
    p.add_argument("--no-sync", action="store_true", help="build locally only")
    args = p.parse_args()

    bucket = args.bucket.rstrip("/")
    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    out = {}

    wide_rg_rows = args.wide_rg_mib * MiB // (args.wide_cols * 8)
    specs = {
        "tiny": dict(
            n_files=args.tiny_files,
            rows_per_file=args.tiny_rows,
            n_cols=N_COLS,
            row_group_rows=args.tiny_rows,
        ),
        "bigrg": dict(
            n_files=args.bigrg_files,
            rows_per_file=args.bigrg_mib * MiB // BYTES_PER_ROW,
            n_cols=N_COLS,
            row_group_rows=args.bigrg_mib * MiB // BYTES_PER_ROW,
        ),
        "wide": dict(
            n_files=args.wide_files,
            rows_per_file=wide_rg_rows * args.wide_rgs_per_file,
            n_cols=args.wide_cols,
            row_group_rows=wide_rg_rows,
        ),
        # Same geometry as `wide` so the two are comparable, but one column is a
        # pyarrow canonical extension type. `wide` measures the column-group
        # branch; `tensors` measures whether the file reaches the native path at
        # all -- so it has to be the same SIZE as `wide`, or its ratios are noise.
        #
        # The 2026-08-10 run proves that: dividing the row count by the tensor
        # cell size (16) built a 20 MiB fixture against wide's 256 MiB, and its
        # 2.1 s read told us nothing. That division was simply wrong arithmetic.
        # The tensor column REPLACES one float64 column, so it adds
        # (cells - 1) x 8 = 120 B/row, not a 16x factor:
        #   wide    row = n_cols x 8                    = 512 B at 64 cols
        #   tensors row = (cells + n_cols - 1) x 8      = 632 B at 64 cols
        # Equal bytes therefore wants rows x n_cols / (cells + n_cols - 1) --
        # about 0.81x the rows, not 0.06x.
        "tensors": dict(
            n_files=args.wide_files,
            rows_per_file=max(
                (wide_rg_rows * args.wide_rgs_per_file * args.wide_cols)
                // _tensor_row_divisor(args.wide_cols),
                1,
            ),
            n_cols=args.wide_cols,
            row_group_rows=max(
                (wide_rg_rows * args.wide_cols) // _tensor_row_divisor(args.wide_cols),
                1,
            ),
            kind="tensor",
        ),
    }

    for case in cases:
        if case not in specs:
            sys.exit(f"unknown case {case!r}; expected one of {sorted(specs)}")
        spec = specs[case]
        d = os.path.join(args.local_root, case)
        print(f"=== {case} ===", flush=True)
        if _already_built(d, spec):
            print(f"  up to date at {d} (delete it to force a rebuild)", flush=True)
        else:
            _write(d, want=spec, **spec)
        s3 = f"{bucket}/fixtures/{case}"
        if not args.no_sync:
            sync_up(d, s3)
        out[case] = (d, s3)

    print("\nfixtures ready:")
    for case, (local, s3) in out.items():
        print(f"  {case:<7} local={local}")
        print(f"  {case:<7} s3   ={s3}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
