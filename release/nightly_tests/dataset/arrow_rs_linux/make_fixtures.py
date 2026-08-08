#!/usr/bin/env python3
"""Two Parquet layouts the TPC-H input cannot produce, written local and to S3.

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

So this file builds the two layouts that isolate the things lineitem confounds:

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
    cost of being a read task at all, measured rather than extrapolated. Both
    fits of the S3 sweep have an intercept -- PyArrow ~189 MiB, arrow-rs ~306 --
    and that +117 MiB gap is the entire reason arrow-rs loses below D ~= 700
    MiB. A straight line through four points that are visibly concave is a weak
    way to learn a constant; this measures it at D ~= 0 directly. Writing the
    fixture to BOTH local disk and S3 also splits the constant in two, because
    the local path memory-maps and constructs no HTTP client, no connection
    pool and no tokio runtime: (arrow_rs s3 - arrow_rs local) is the transport's
    share and the rest is the reader's.

Both layouts get ``write_page_index=True``: the crate requires the page index to
plan byte-budgeted reads, and without it the read silently falls back.

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


def _table(n_rows: int, seed: int) -> pa.Table:
    rng = np.random.default_rng(seed)
    return pa.table({f"c{i}": pa.array(rng.random(n_rows)) for i in range(N_COLS)})


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


def _write(d: str, n_files: int, rows_per_file: int, one_row_group: bool, want: dict):
    os.makedirs(d, exist_ok=True)
    for f in range(n_files):
        path = os.path.join(d, f"part{f}.parquet")
        table = _table(rows_per_file, seed=f)
        pq.write_table(
            table,
            path,
            write_page_index=True,
            compression="snappy",
            # >= the row count in this single write_table call => exactly one
            # row group. pyarrow starts a new row group per call regardless, so
            # the whole table has to be built and written at once; that is why
            # --bigrg-mib is bounded by RAM rather than by patience.
            row_group_size=rows_per_file if one_row_group else 128 * 1024,
        )
        on_disk = os.path.getsize(path)
        meta = pq.read_metadata(path)
        print(
            f"  {os.path.basename(path)}: {rows_per_file:,} rows, "
            f"{meta.num_row_groups} row group(s), "
            f"{rows_per_file * BYTES_PER_ROW / MiB:,.0f} MiB decoded, "
            f"{on_disk / MiB:,.0f} MiB on disk "
            f"({on_disk / max(rows_per_file, 1):.0f} compressed B/row)",
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
    p.add_argument("--cases", default="tiny,bigrg", help="comma list of {tiny,bigrg}")
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
    p.add_argument("--no-sync", action="store_true", help="build locally only")
    args = p.parse_args()

    bucket = args.bucket.rstrip("/")
    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    out = {}

    specs = {
        "tiny": dict(
            n_files=args.tiny_files,
            rows_per_file=args.tiny_rows,
            one_row_group=True,
        ),
        "bigrg": dict(
            n_files=args.bigrg_files,
            rows_per_file=args.bigrg_mib * MiB // BYTES_PER_ROW,
            one_row_group=True,
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
