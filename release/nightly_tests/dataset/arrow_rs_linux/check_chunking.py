#!/usr/bin/env python3
"""Does any release Parquet input cross the chunk threshold? Metadata only.

Why this matters
----------------
``ParquetFileChunker`` splits a file only when its ON-DISK (compressed) size
exceeds ``parquet_chunker_target_chunk_size`` (default 1 GiB). That single
comparison selects between two very different code paths in the arrow-rs reader:

  file <= threshold  ->  ONE native fragment, ``row_groups=None``. One crate
                         call reads the whole file: one footer parse, one
                         object_store client, one prefetch window spanning
                         everything. Measured 0.71-0.79x PyArrow read-op time.

  file >  threshold  ->  chunked, and ``_native_fragments_for_file`` returns ONE
                         FRAGMENT PER ROW GROUP. At our shipped fragment-thread
                         default of 1 those run strictly serially, one crate call
                         each, and each call's prefetch window cannot span row
                         groups. Measured 1.29-1.32x PyArrow read-op time and
                         1.13x memory -- the only shape where arrow-rs loses on
                         both axes (regression_testing.md section 8.12.5).

So "do real inputs cross the threshold" decides whether that loss is a
real-workload problem worth fixing or an artefact only our chunk sweep produced.
This script answers it from footers alone: no decode, no cluster, seconds to run.

It also reports row groups per file REGARDLESS of size, because that is the
fragment count the chunked path would produce, and because a file with one huge
row group is the opposite failure mode (chunking cannot split it at all -- the
row group is the floor).

Usage
-----
    python check_chunking.py                       # the release inputs
    python check_chunking.py s3://bucket/prefix ...
    python check_chunking.py /local/dir --threshold-mib 256

Credentials: the release inputs live in ray-benchmark-data-internal-us-west-2,
which is readable from an Anyscale release cluster but NOT from a personal AWS
account. A prefix that 403s is reported as UNREADABLE and skipped rather than
failing the run, so the readable prefixes still produce a verdict. To cover the
rest, run this on a box that can see that bucket.
"""

import argparse
import statistics
import sys
from typing import List, Optional, Tuple

MiB = 1024 * 1024
GiB = 1024 * MiB

# The regressed cases from the 2026-08-05 multi-node A/B (builds 103100/103101),
# with the metric each one regressed on, so the output says which row matters.
RELEASE_INPUTS = [
    (
        "s3://ray-benchmark-data-internal-us-west-2/imagenet/parquet",
        "read_parquet_{fixed_size,autoscaling}: 2.64x / 3.06x read-op wall",
    ),
    (
        "s3://ray-benchmark-data-internal-us-west-2/large-parquet/",
        "read_large_parquet_*: yaml pins --memory 3.4 GiB as 'maximum heap observed'",
    ),
    (
        "s3://ray-benchmark-data-internal-us-west-2/wide_schema/tensors",
        "wide_schema_pipeline_tensors: 4.90x read-op wall, the worst in the run",
    ),
    (
        "s3://ray-benchmark-data-internal-us-west-2/imagenet/parquet_split",
        "read_from_uris / parquet_split",
    ),
    (
        "s3://ray-benchmark-data/tpch/parquet/sf10/lineitem",
        "control: the input every single-node experiment used (271 MiB/file)",
    ),
]


def _list_parquet(uri: str) -> Tuple[Optional[List[Tuple[str, int]]], Optional[str]]:
    """``([(path, size_bytes)], None)`` for the .parquet files under ``uri``, or
    ``(None, reason)`` if the prefix cannot be listed. Never raises: an
    unreadable prefix is a reportable outcome, not a crash."""
    try:
        from pyarrow.fs import FileSelector, FileSystem, FileType

        fs, root = FileSystem.from_uri(uri)
        infos = fs.get_file_info(
            FileSelector(root, recursive=True, allow_not_found=True)
        )
        files = [
            (i.path, i.size)
            for i in infos
            if i.type == FileType.File and i.path.endswith(".parquet")
        ]
        return sorted(files), None
    except Exception as exc:  # noqa: BLE001 - the reason is the result here
        msg = str(exc).split("\n", 1)[0][:160]
        return None, msg


def _row_groups(fs, path: str) -> Optional[Tuple[int, int]]:
    """``(num_row_groups, max_row_group_UNCOMPRESSED_bytes)`` from the footer only.

    ``read_metadata`` fetches the footer, not the data, so this costs one small
    ranged GET per file even on a multi-GiB object.

    Note the units: Parquet's ``RowGroup.total_byte_size`` is the **uncompressed**
    size, so it is NOT comparable to the file's on-disk size and can legitimately
    exceed it (a 25.2 MiB wide_schema/tensors file reports a 38.2 MiB row group).
    Uncompressed is the right currency anyway -- it is what bounds the DECODED
    working set, it is what the crate's K-split threshold and byte budget compare
    against, and it is what PR #64985's bin packer packs on.
    """
    try:
        import pyarrow.parquet as pq

        with fs.open_input_file(path) as handle:
            md = pq.read_metadata(handle)
        biggest = max(
            (md.row_group(i).total_byte_size for i in range(md.num_row_groups)),
            default=0,
        )
        return md.num_row_groups, biggest
    except Exception:  # noqa: BLE001 - a footer we cannot read is just unknown
        return None


def _fmt(n: int) -> str:
    if n >= GiB:
        return f"{n / GiB:,.2f} GiB"
    return f"{n / MiB:,.1f} MiB"


def _report(uri: str, why: str, threshold: int, max_footers: int) -> Optional[bool]:
    """Print one prefix's verdict. Returns True/False for "chunking fires", or
    None when the prefix could not be read at all."""
    print(f"\n=== {uri}")
    print(f"    ({why})")
    files, err = _list_parquet(uri)
    if files is None:
        print(f"    UNREADABLE -- {err}")
        print("    -> run this from a box that can see the bucket; skipped.")
        return None
    if not files:
        print("    no .parquet files found under this prefix")
        return None

    sizes = [s for _, s in files]
    over = [(p, s) for p, s in files if s > threshold]
    print(
        f"    {len(files):,} files, total {_fmt(sum(sizes))}, "
        f"largest {_fmt(max(sizes))}, median {_fmt(int(statistics.median(sizes)))}"
    )

    fires = bool(over)
    if fires:
        print(
            f"    *** {len(over)} file(s) EXCEED the {_fmt(threshold)} threshold "
            f"-> chunking fires -> per-row-group fragments"
        )
    else:
        print(
            f"    no file exceeds {_fmt(threshold)} -> chunk_metadata is None -> "
            f"ONE native fragment per file (the fast path)"
        )

    # Footers for the largest files: the row-group count is the fragment count the
    # chunked path would produce, and it is the number that makes the 1.29x
    # concrete. Read the largest first -- those are the ones that can chunk.
    from pyarrow.fs import FileSystem

    fs, _ = FileSystem.from_uri(uri)
    probe = sorted(files, key=lambda ps: -ps[1])[:max_footers]
    print(f"    footers for the {len(probe)} largest:")
    for path, size in probe:
        rg = _row_groups(fs, path)
        if rg is None:
            print(f"      {path.rsplit('/', 1)[-1]:<40} {_fmt(size):>12}  footer: n/a")
            continue
        n_rg, biggest_rg = rg
        # Mirror the chunker: ceil(size / threshold), then row groups are
        # distributed across chunks by COUNT (parquet_file_chunking_utils).
        n_chunks = -(-size // threshold)
        note = ""
        if n_chunks > 1:
            per_chunk = n_rg / n_chunks
            note = (
                f"  -> {n_chunks} chunks x ~{per_chunk:.1f} rgs = {n_rg} serial calls"
            )
        elif n_rg == 1:
            note = "  -> 1 row group: the floor; no chunk size can split it"
        print(
            f"      {path.rsplit('/', 1)[-1]:<40} {_fmt(size):>12}  "
            f"{n_rg:>4} rgs, largest rg {_fmt(biggest_rg):>11} uncompressed{note}"
        )
    return fires


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "prefixes",
        nargs="*",
        help="S3 URIs or local dirs. Default: the regressed release inputs.",
    )
    ap.add_argument(
        "--threshold-mib",
        type=int,
        default=1024,
        help="Chunk threshold to test against (default 1024, Ray's "
        "_DEFAULT_TARGET_CHUNK_SIZE).",
    )
    ap.add_argument(
        "--max-footers",
        type=int,
        default=5,
        help="Footers to read per prefix, largest files first (default 5).",
    )
    args = ap.parse_args()

    targets = (
        [(p, "user-supplied") for p in args.prefixes]
        if args.prefixes
        else RELEASE_INPUTS
    )
    threshold = args.threshold_mib * MiB

    print(f"chunk threshold under test: {_fmt(threshold)}")
    print("a file at or below it is read as ONE native fragment (fast path);")
    print("above it, one fragment PER ROW GROUP, serial at threads=1 (the 1.29x).")

    verdicts = [
        (uri, _report(uri, why, threshold, args.max_footers)) for uri, why in targets
    ]

    print("\n=== verdict")
    readable = [(u, v) for u, v in verdicts if v is not None]
    if not readable:
        print("  NOTHING was readable. This machine cannot answer the question;")
        print("  re-run where ray-benchmark-data-internal-us-west-2 is visible.")
        return 2
    firing = [u for u, v in readable if v]
    if firing:
        print("  Chunking FIRES on:")
        for u in firing:
            print(f"    - {u}")
        print("  -> the per-row-group-fragment loss is reachable in the release")
        print("     suite, and fixing it (TODO 1l) is justified.")
    else:
        print("  No readable prefix crosses the threshold.")
        print("  -> every release file is read as ONE native fragment, so the")
        print("     1.29-1.32x chunked-file loss did NOT contribute to the")
        print("     release regressions, and TODO 1l is not the explanation.")
    unread = [u for u, v in verdicts if v is None]
    if unread:
        print(f"  {len(unread)} prefix(es) unreadable here -- verdict is partial:")
        for u in unread:
            print(f"    - {u}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
