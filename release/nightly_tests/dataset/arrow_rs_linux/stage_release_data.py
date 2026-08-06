#!/usr/bin/env python3
"""Probe and stage the release-suite inputs behind the arrow-rs A/B regressions.

Why this exists
---------------
The multi-node release A/B (builds 103100 vs 103101) regressed on inputs we do
not own. Some live in `ray-benchmark-data` (readable); the rest in
`ray-benchmark-data-internal-us-west-2` (ACCESS_DENIED as of 2026-08-05). This
script answers, in one pass:

  1. which of them we can actually read today, and
  2. what SHAPE each one is -- row groups per file, bytes per row group,
     compression, and whether a page index was written.

(2) is the part that cannot be replaced by generating our own TPC-H. Scale we
can generate; layout we have to learn, because layout is what selects the code
path. A file with one 800 MiB row group and a file with forty 20 MiB row groups
are both "lineitem at scale factor N" and they exercise completely different
parts of the reader. And if the release inputs carry no page index, our reader
has been on a degraded path in every release run and a fast path in every local
benchmark we have ever written -- which would explain the whole disagreement.

Staging copies a bounded SLICE, not the whole input: enough files to keep 8 CPUs
busy for a few minutes. `read_large_parquet` alone is 809 GB and we need none of
that to reproduce a per-task memory or wall-time ratio. Copies are server-side
(`CopyObject`) and same-region, so no bytes cross the network and there is no
egress charge; re-runs skip objects already present at the right size.

Usage
-----
    python stage_release_data.py --probe            # read-only: access + shapes
    python stage_release_data.py --probe --json shapes.json
    python stage_release_data.py --stage            # copy the slices we can read
    python stage_release_data.py --stage --only write_parquet,read_parquet
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

BENCH = "ray-benchmark-data"
INTERNAL = "ray-benchmark-data-internal-us-west-2"

# Each entry is one release-suite input we need, keyed by the test that uses it.
#
# Size is expressed as `target_gb`, NOT a file count, because a file count picked
# before looking at the data means nothing: 32 objects is 9 GB if they are 280 MiB
# each and 160 GB if they are 5 GB each. The probe reports the actual sizes and
# the staging step then takes files until the budget is met.
#
# These budgets stage a SLICE for single-box mechanism work -- enough files to
# keep 8 CPUs busy for several waves so per-task memory has a distribution worth
# measuring. They are deliberately NOT a replication of the release, which ran
# full inputs across 10 nodes and can only be replicated on a cluster.
#
# `note` records what the release A/B measured, so the manifest doubles as the
# shortlist of what we are chasing.
MIN_FILES = 8  # >= one task per CPU, even when the budget would take fewer

SOURCES: Dict[str, Dict[str, Any]] = {
    "write_parquet": {
        "bucket": BENCH,
        "prefix": "tpch/parquet/sf1000/lineitem",
        "target_gb": 12,
        "note": "USS 1.68x avg / 2.28x max, RSS 2.19x, wall 1.00x -- the only "
        "per-task memory regression with more than one sample",
    },
    "iter_batches": {
        "bucket": BENCH,
        "prefix": "tpch/parquet/sf10/lineitem",
        "target_gb": 0,  # 0 = all of it; ~2.8 GB, and already staged
        "note": "max USS 1.10-1.12x; numpy variant reproduced at 0.91x locally",
    },
    "read_parquet": {
        "bucket": INTERNAL,
        "prefix": "imagenet/parquet",
        "target_gb": 12,
        "note": "wall 2.64-3.06x (worst speed loss); USS 0.77x and obj 0.61x on "
        "fixed_size but obj 1.25x on autoscaling -- same input, so treat the "
        "object-store numbers as confounded by cluster size",
    },
    "read_large_parquet": {
        "bucket": INTERNAL,
        "prefix": "large-parquet",
        "target_gb": 20,
        "note": "wall 0.51-0.55x (the headline WIN) but obj peak 1.75x -- faster "
        "and heavier at once; staged to keep the win and explain the peak",
    },
    "wide_schema_tensors": {
        "bucket": INTERNAL,
        "prefix": "wide_schema/tensors",
        "target_gb": 0,
        "note": "wall 4.90x, largest ratio in the run; obj peak 1.11x",
    },
    "wide_schema_primitives": {
        "bucket": INTERNAL,
        "prefix": "wide_schema/primitives",
        "target_gb": 0,
        "note": "obj peak 1.41x at wall parity",
    },
    "wide_schema_objects": {
        "bucket": INTERNAL,
        "prefix": "wide_schema/objects",
        "target_gb": 0,
        "note": "obj peak 1.32x at wall parity",
    },
    "wide_schema_nested_structs": {
        "bucket": INTERNAL,
        "prefix": "wide_schema/nested_structs",
        "target_gb": 0,
        "note": "obj peak 1.31x at wall parity",
    },
}

DEST_BUCKET = os.environ.get("S3_BUCKET", "arrowrs-bench-21f6c795")
DEST_PREFIX = os.environ.get("S3_STAGE_PREFIX", "arrow_rs_probe/release")
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")


def list_parquet(client, bucket: str, prefix: str) -> List[Dict[str, Any]]:
    """Every .parquet object under prefix, in S3's lexicographic order.

    Deterministic order matters: `select` then takes a prefix of this list, so
    two runs stage the same slice and a re-run is a no-op rather than a
    different sample of the input.
    """
    out: List[Dict[str, Any]] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".parquet"):
                out.append({"key": obj["Key"], "size": obj["Size"]})
    return out


def select(objs: List[Dict[str, Any]], target_gb: float) -> List[Dict[str, Any]]:
    """Take files until `target_gb` is met (all of them when target_gb=0).

    Always returns at least MIN_FILES so that a source made of very large
    objects still yields one task per CPU -- a single 20 GB file would satisfy
    any byte budget and then measure nothing about per-task distribution.
    """
    if not target_gb:
        return objs
    budget = target_gb * 1e9
    taken: List[Dict[str, Any]] = []
    used = 0.0
    for obj in objs:
        if used >= budget and len(taken) >= MIN_FILES:
            break
        taken.append(obj)
        used += obj["size"]
    return taken


def describe(bucket: str, key: str) -> Dict[str, Any]:
    """Footer-only characterisation of one Parquet file.

    `read_metadata` fetches the footer, not the data, so this is a couple of
    range GETs regardless of how large the object is.
    """
    import pyarrow.parquet as pq
    from pyarrow.fs import S3FileSystem

    fs = S3FileSystem(region=REGION)
    meta = pq.read_metadata(f"{bucket}/{key}", filesystem=fs)
    groups = [meta.row_group(i) for i in range(meta.num_row_groups)]
    sizes = [g.total_byte_size for g in groups] or [0]
    col0 = groups[0].column(0) if groups else None
    schema = meta.schema.to_arrow_schema()
    return {
        "rows": meta.num_rows,
        "columns": meta.num_columns,
        "row_groups": meta.num_row_groups,
        "rg_bytes_min": min(sizes),
        "rg_bytes_max": max(sizes),
        "rg_bytes_mean": sum(sizes) // len(sizes),
        "compression": col0.compression if col0 else None,
        # The offset index is half of the page index. Its absence means our
        # reader cannot do page-level range selection on this input.
        "page_index": bool(col0 and col0.offset_index_offset),
        "created_by": meta.created_by,
        "types": [f"{f.name}:{f.type}" for f in schema][:24],
    }


def probe(names: List[str], json_path: Optional[str]) -> Dict[str, Any]:
    import boto3
    from botocore.exceptions import ClientError

    client = boto3.client("s3", region_name=REGION)
    report: Dict[str, Any] = {}

    header = (
        f"{'source':<28}{'access':<10}{'files':>7}{'total GB':>10}"
        f"{'stage':>13}  shape"
    )
    print(header)
    print("-" * (len(header) + 40))

    for name in names:
        spec = SOURCES[name]
        entry: Dict[str, Any] = {"spec": spec}
        try:
            objs = list_parquet(client, spec["bucket"], spec["prefix"])
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "?")
            entry["access"] = code
            report[name] = entry
            print(f"{name:<28}{code:<10}{'-':>7}{'-':>10}{'-':>13}")
            continue
        if not objs:
            entry["access"] = "EMPTY"
            report[name] = entry
            print(f"{name:<28}{'EMPTY':<10}{'-':>7}{'-':>10}{'-':>13}")
            continue

        total = sum(o["size"] for o in objs)
        slice_ = select(objs, spec["target_gb"])
        slice_bytes = sum(o["size"] for o in slice_)
        entry["access"] = "ok"
        entry["num_files"] = len(objs)
        entry["total_bytes"] = total
        entry["stage_files"] = len(slice_)
        entry["stage_bytes"] = slice_bytes
        would = f"{len(slice_)}f/{slice_bytes / 1e9:.1f}GB"
        try:
            entry["shape"] = describe(spec["bucket"], objs[0]["key"])
            s = entry["shape"]
            shape = (
                f"{s['row_groups']} rg x {s['rg_bytes_mean'] / 2**20:.0f}MiB, "
                f"{s['columns']} cols, {s['compression']}, "
                f"{'page_index' if s['page_index'] else 'NO PAGE INDEX'}"
            )
        except Exception as exc:  # noqa: BLE001 - footer read is best-effort
            entry["shape_error"] = f"{type(exc).__name__}: {exc}"
            shape = f"footer unreadable ({type(exc).__name__})"
        report[name] = entry
        print(
            f"{name:<28}{'ok':<10}{len(objs):>7}{total / 1e9:>10.1f}"
            f"{would:>13}  {shape}"
        )

    denied = [n for n, e in report.items() if e.get("access") not in ("ok", None)]
    if denied:
        print(f"\nunreachable: {', '.join(denied)}")
        print("These carry 4 of the regressions and the headline win. Until access")
        print("lands they can only be approximated with synthetic fixtures.")
    missing_index = [
        n for n, e in report.items() if e.get("shape") and not e["shape"]["page_index"]
    ]
    if missing_index:
        print(f"\n!!! NO PAGE INDEX on: {', '.join(missing_index)}")
        print("Our reader's efficient path assumes one. Every local fixture we")
        print("wrote sets write_page_index=True, so this alone could explain a")
        print("fast path locally and a slow path in release. Investigate before")
        print("running anything else.")

    if json_path:
        with open(json_path, "w") as handle:
            json.dump(report, handle, indent=2, default=str)
        print(f"\nwrote {json_path}")
    return report


def stage(names: List[str]) -> None:
    import boto3
    from botocore.exceptions import ClientError

    client = boto3.client("s3", region_name=REGION)
    for name in names:
        spec = SOURCES[name]
        try:
            objs = select(
                list_parquet(client, spec["bucket"], spec["prefix"]),
                spec["target_gb"],
            )
        except ClientError as exc:
            print(f"{name}: skipped ({exc.response.get('Error', {}).get('Code')})")
            continue
        if not objs:
            print(f"{name}: skipped (no .parquet objects)")
            continue

        copied = skipped = 0
        for obj in objs:
            leaf = obj["key"][len(spec["prefix"]) :].lstrip("/")
            dest = f"{DEST_PREFIX}/{name}/{leaf}"
            try:
                head = client.head_object(Bucket=DEST_BUCKET, Key=dest)
                if head["ContentLength"] == obj["size"]:
                    skipped += 1
                    continue
            except ClientError:
                pass
            # `copy` (not `copy_object`) so objects over the 5 GB single-part
            # limit are copied multipart automatically -- large-parquet needs it.
            client.copy(
                {"Bucket": spec["bucket"], "Key": obj["key"]},
                DEST_BUCKET,
                dest,
            )
            copied += 1
        total = sum(o["size"] for o in objs)
        print(
            f"{name}: {copied} copied, {skipped} already present, "
            f"{total / 1e9:.1f} GB -> s3://{DEST_BUCKET}/{DEST_PREFIX}/{name}/"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--probe", action="store_true", help="access + shape only")
    parser.add_argument("--stage", action="store_true", help="copy the slices")
    parser.add_argument("--json", help="write the probe report here")
    parser.add_argument("--only", help="comma-separated subset of the manifest")
    parser.add_argument(
        "--target-gb",
        type=float,
        help="override every source's slice budget (0 = stage everything). "
        "Raise it once the probe shows the real file sizes and you know how "
        "much a run can chew through.",
    )
    args = parser.parse_args()

    if args.target_gb is not None:
        for spec in SOURCES.values():
            spec["target_gb"] = args.target_gb

    names = list(SOURCES)
    if args.only:
        names = args.only.split(",")
        unknown = [n for n in names if n not in SOURCES]
        if unknown:
            raise SystemExit(f"unknown source(s) {unknown}; known: {list(SOURCES)}")
    if not (args.probe or args.stage):
        raise SystemExit("pass --probe and/or --stage")
    if args.probe:
        probe(names, args.json)
    if args.stage:
        if args.probe:
            print()
        stage(names)


if __name__ == "__main__":
    main()
    sys.exit(0)
