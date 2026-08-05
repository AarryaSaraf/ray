"""Parquet fixtures that exercise the arrow-rs reader's *decision points*.

Each shape here exists to make one planner branch fire, so that a regression can
be attributed to a mechanism instead of to "S3 was slow". The branches, and the
shape that reaches each (see ``arrow_rs_docs/regression_testing.md``):

===========================  =====================================================
Shape                        Branch it forces
===========================  =====================================================
``fat_col``                  **The missing fixture.** One column whose *compressed
                             chunk exceeds* ``column_fetch_mb`` plus one tiny
                             column. ``partition_columns_by_budget`` emits
                             ``[[fat],[small]]`` -> ``len() > 1`` -> the 5000-column
                             COLUMN-GROUP path, and the fat unit is larger than the
                             whole prefetch budget, so it fetches alone with no
                             overlap against decode (H1 + H2).
``fat_col_solo``             The control for ``fat_col``: the *same bytes*, one
                             column only. ``cols.len() <= 1`` returns a single
                             group, so this takes the healthy ROW-WINDOW path.
                             ``fat_col`` slow + ``fat_col_solo`` fast == H2 proven;
                             both slow == H2 dead.
``wide_5k``                  5000 small columns: column groups, many units, every
                             unit under budget. Isolates H1 (whole-row-group
                             retention in the hstack) from H2 (oversized unit).
``narrow_big_rg``            18 numeric columns in one large row group: row
                             windows, budget binds, K-split eligible. The known-good
                             case where arrow-rs wins; a regression here means
                             something basic broke.
``narrow_many_rg``           Same columns, many small row groups: row windows,
                             Ray's fragment pool parallelizes, K=1.
``fat_rows``                 ~33 KB/row over few columns: the 2048-row batch floor
                             overrides ``decode_budget_bytes`` (H3). Locally
                             observable -- no S3 needed.
===========================  =====================================================

Usage::

    # local dir
    python arrow_rs_fixtures.py --out /data/arrow_rs_fixtures --shapes all
    # or into an S3-compatible endpoint (MinIO / moto) for the S3 planner
    python arrow_rs_fixtures.py --out s3://bench/fixtures --shapes fat_col,fat_col_solo \\
        --endpoint http://127.0.0.1:9000

Every file is written with ``write_page_index=True``: without the offset index the
row-window planner cannot split a row group by rows at all (it collapses to one
window), which silently turns off the mechanism under test.
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

MiB = 1024 * 1024

# Default fat-column chunk target. Must exceed ``column_fetch_mb`` (16 MiB
# default) by enough that it also exceeds ``prefetch_budget_mb`` (64 MiB
# default), which is what forces the serial-fetch branch rather than merely the
# column-group branch.
DEFAULT_FAT_CHUNK_MIB = 128


def _incompressible(rng: np.random.Generator, nbytes: int) -> bytes:
    """Random bytes: snappy cannot shrink these, so the *compressed* chunk size
    is predictable from the requested size. A compressible payload would leave
    the chunk under ``column_fetch_mb`` and quietly not test anything."""
    return rng.bytes(nbytes)


def _distinct_cells(rng: np.random.Generator, nrows: int, cell: int) -> List[bytes]:
    """``nrows`` genuinely distinct incompressible cells of ``cell`` bytes.

    Every cell must be unique. Slicing a rotating window out of one small buffer
    (the obvious cheap trick) yields only as many distinct values as the window
    has offsets, and Parquet then *dictionary-encodes* the column: the chunk comes
    out a fraction of the requested size and silently fails to exceed
    ``column_fetch_mb``, so the branch under test never fires. One large random
    buffer sliced non-overlapping keeps every value distinct.
    """
    buf = _incompressible(rng, nrows * cell)
    return [buf[i * cell : (i + 1) * cell] for i in range(nrows)]


def _fat_binary_column(
    rng: np.random.Generator, chunk_mib: int, cell_kib: int
) -> Tuple[pa.Array, int]:
    """A binary column of ~``chunk_mib`` MiB in ~``cell_kib`` KiB cells."""
    cell = cell_kib * 1024
    nrows = max(1, (chunk_mib * MiB) // cell)
    return pa.array(_distinct_cells(rng, nrows, cell), type=pa.binary()), nrows


def build_fat_col(rng, chunk_mib: int, cell_kib: int) -> pa.Table:
    fat, nrows = _fat_binary_column(rng, chunk_mib, cell_kib)
    return pa.table(
        {
            "payload": fat,
            # Deliberately tiny: it must land in its own column group so the
            # partition length is 2 and the wide path is selected.
            "label": pa.array(rng.integers(0, 1000, nrows), type=pa.int64()),
        }
    )


def build_fat_col_solo(rng, chunk_mib: int, cell_kib: int) -> pa.Table:
    fat, _ = _fat_binary_column(rng, chunk_mib, cell_kib)
    return pa.table({"payload": fat})


def build_wide_5k(rng, num_columns: int = 5000, num_rows: int = 4000) -> pa.Table:
    # ~40 B/col/row -> ~200 KB/row, comparable to the release wide_schema fixtures.
    return pa.table(
        {
            f"c{i:05d}": pa.array(
                [_incompressible(rng, 40) for _ in range(num_rows)], type=pa.binary()
            )
            for i in range(num_columns)
        }
    )


def build_narrow(rng, num_rows: int, num_columns: int = 18) -> pa.Table:
    return pa.table(
        {
            f"c{i:02d}": pa.array(
                rng.integers(0, 1 << 40, num_rows, dtype=np.int64), type=pa.int64()
            )
            for i in range(num_columns)
        }
    )


def build_fat_rows(rng, num_rows: int = 8000, cell_kib: int = 33) -> pa.Table:
    return pa.table(
        {
            "blob": pa.array(
                _distinct_cells(rng, num_rows, cell_kib * 1024), type=pa.binary()
            ),
            "label": pa.array(rng.integers(0, 1000, num_rows), type=pa.int64()),
        }
    )


# name -> (builder, row_group_size or None for "one row group", num_files)
SHAPES: Dict[str, dict] = {
    "fat_col": {"build": "fat_col", "row_group_size": None, "num_files": 4},
    "fat_col_solo": {"build": "fat_col_solo", "row_group_size": None, "num_files": 4},
    "wide_5k": {"build": "wide_5k", "row_group_size": None, "num_files": 2},
    "narrow_big_rg": {"build": "narrow_big", "row_group_size": None, "num_files": 4},
    "narrow_many_rg": {
        "build": "narrow_many",
        "row_group_size": 50_000,
        "num_files": 4,
    },
    "fat_rows": {"build": "fat_rows", "row_group_size": None, "num_files": 4},
}


def _build_table(name: str, rng, args) -> pa.Table:
    if name == "fat_col":
        return build_fat_col(rng, args.fat_chunk_mib, args.fat_cell_kib)
    if name == "fat_col_solo":
        return build_fat_col_solo(rng, args.fat_chunk_mib, args.fat_cell_kib)
    if name == "wide_5k":
        return build_wide_5k(rng, args.wide_columns, args.wide_rows)
    if name == "narrow_big":
        return build_narrow(rng, args.narrow_rows)
    if name == "narrow_many":
        return build_narrow(rng, args.narrow_rows)
    if name == "fat_rows":
        return build_fat_rows(rng)
    raise ValueError(name)


def _filesystem(out: str, endpoint: Optional[str]):
    """Return ``(filesystem, path)``. A local path gets ``None`` (pyarrow infers
    a LocalFileSystem); an ``s3://`` path with an endpoint gets an explicitly
    configured S3FileSystem so MinIO/moto works without ambient AWS config."""
    if not out.startswith("s3://"):
        os.makedirs(out, exist_ok=True)
        return None, out
    from pyarrow.fs import S3FileSystem

    path = out[len("s3://") :]
    kwargs = {}
    if endpoint:
        kwargs["endpoint_override"] = endpoint
        # MinIO/moto over plain HTTP; the real thing uses TLS and ignores this.
        kwargs["scheme"] = "http" if endpoint.startswith("http://") else "https"
    fs = S3FileSystem(**kwargs)
    return fs, path


def _chunk_report(local_or_buf, name: str) -> dict:
    """Read back the footer and report the facts the planner branches on."""
    md = pq.read_metadata(local_or_buf)
    rg = md.row_group(0)
    per_col = [
        (rg.column(i).path_in_schema, rg.column(i).total_compressed_size)
        for i in range(rg.num_columns)
    ]
    per_col.sort(key=lambda kv: -kv[1])
    unc = sum(rg.column(i).total_uncompressed_size for i in range(rg.num_columns))
    return {
        "shape": name,
        "num_rows": md.num_rows,
        "num_row_groups": md.num_row_groups,
        "num_columns": rg.num_columns,
        "uncompressed_bytes_per_row": round(unc / max(md.num_rows, 1), 1),
        "rg0_uncompressed_mib": round(unc / MiB, 1),
        "largest_column_compressed_mib": round(per_col[0][1] / MiB, 2),
        "largest_column": per_col[0][0],
        # The two predictions that make a fixture worth running.
        "exceeds_column_fetch_16mib": per_col[0][1] > 16 * MiB,
        "exceeds_prefetch_budget_64mib": per_col[0][1] > 64 * MiB,
    }


def write_shape(name: str, args) -> dict:
    spec = SHAPES[name]
    rng = np.random.default_rng(abs(hash(name)) % (2**31))
    fs, base = _filesystem(args.out, args.endpoint)
    report = None
    for file_index in range(spec["num_files"]):
        table = _build_table(spec["build"], rng, args)
        rgs = spec["row_group_size"] or table.num_rows
        rel = f"{name}/part_{file_index:05d}.parquet"
        target = f"{base}/{rel}"
        if fs is None:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            pq.write_table(
                table,
                target,
                compression=args.compression,
                row_group_size=rgs,
                write_page_index=True,
            )
            if report is None:
                report = _chunk_report(target, name)
        else:
            with fs.open_output_stream(target) as sink:
                pq.write_table(
                    table,
                    sink,
                    compression=args.compression,
                    row_group_size=rgs,
                    write_page_index=True,
                )
            if report is None:
                with fs.open_input_file(target) as src:
                    report = _chunk_report(src, name)
    report["num_files"] = spec["num_files"]
    report["path"] = f"{args.out}/{name}"
    return report


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True, help="Local dir or s3://bucket/prefix")
    p.add_argument(
        "--shapes",
        default="all",
        help=f"Comma-separated subset of {sorted(SHAPES)}, or 'all'",
    )
    p.add_argument("--endpoint", default=None, help="S3 endpoint (MinIO/moto)")
    p.add_argument("--compression", default="snappy")
    p.add_argument(
        "--fat-chunk-mib",
        type=int,
        default=DEFAULT_FAT_CHUNK_MIB,
        help="Target compressed size of the fat column chunk. Must exceed "
        "column_fetch_mb (16) to select the column-group path, and "
        "prefetch_budget_mb (64) to force the serial-fetch branch.",
    )
    p.add_argument("--fat-cell-kib", type=int, default=39, help="Bytes per fat cell")
    p.add_argument("--wide-columns", type=int, default=5000)
    p.add_argument("--wide-rows", type=int, default=4000)
    p.add_argument("--narrow-rows", type=int, default=2_000_000)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    names = sorted(SHAPES) if args.shapes == "all" else args.shapes.split(",")
    unknown = [n for n in names if n not in SHAPES]
    if unknown:
        raise SystemExit(f"unknown shapes {unknown}; known: {sorted(SHAPES)}")

    reports = []
    for name in names:
        report = write_shape(name, args)
        reports.append(report)
        print(json.dumps(report))

    # Fail loudly if the fixture that exists to trip the planner does not
    # actually trip it -- a silently-too-small chunk would make the whole
    # experiment come back "no regression found".
    for r in reports:
        if r["shape"] == "fat_col" and not r["exceeds_prefetch_budget_64mib"]:
            raise SystemExit(
                f"fat_col's largest column chunk is only "
                f"{r['largest_column_compressed_mib']} MiB compressed, which does "
                "not exceed the 64 MiB prefetch budget -- raise --fat-chunk-mib "
                "or lower the compression, else the serial-fetch branch is not "
                "under test."
            )


if __name__ == "__main__":
    main()
