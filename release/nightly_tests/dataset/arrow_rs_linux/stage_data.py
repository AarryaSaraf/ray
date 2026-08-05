#!/usr/bin/env python3
"""Stage the TPC-H input for exps 1 and 3 into our own S3 bucket.

Why copy instead of reading the shared bucket directly: exp1 is meant to be the
release case *verbatim*, so the bytes must be the same TPC-H data the release
run used -- but we also want every experiment pointed at a bucket we own, so
there is no ambiguity about who can read what and exp3 has somewhere to write.
Copying once gives us both.

The copy is **server-side** (S3 ``CopyObject``): the data never transits this
box, so a 2.84 GB stage takes seconds and costs no egress when the buckets share
a region. Re-running is a no-op for objects that already match in size, so this
is safe to put at the top of a run script.

Usage::

    python stage_data.py --dst s3://arrowrs-bench-21f6c795/arrow_rs_probe/tpch/sf10/lineitem
    python stage_data.py --dst ... --sf 1        # smaller: 1 file, 0.28 GB
"""

import argparse
import sys
from typing import List, Optional, Tuple

DEFAULT_SRC_BUCKET = "ray-benchmark-data"
DEFAULT_SRC_PREFIX = "tpch/parquet/sf{sf}/lineitem"


def _split(uri: str) -> Tuple[str, str]:
    if not uri.startswith("s3://"):
        raise SystemExit(f"expected an s3:// URI, got {uri!r}")
    bucket, _, key = uri[len("s3://") :].partition("/")
    return bucket, key.rstrip("/")


def _bucket_region(client, bucket: str, default: str) -> str:
    try:
        loc = client.get_bucket_location(Bucket=bucket)["LocationConstraint"]
    except Exception:
        return default
    # The API's quirk: us-east-1 is reported as None.
    return loc or "us-east-1"


def _list(client, bucket: str, prefix: str) -> List[dict]:
    objs: List[dict] = []
    for page in client.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix=prefix + "/"
    ):
        for o in page.get("Contents", []):
            if not o["Key"].endswith("/"):
                objs.append(o)
    return objs


def stage(src: str, dst: str, region: str, dry_run: bool = False) -> int:
    import boto3

    src_bucket, src_prefix = _split(src)
    dst_bucket, dst_prefix = _split(dst)

    probe = boto3.client("s3", region_name=region)
    src_region = _bucket_region(probe, src_bucket, region)
    src_client = (
        probe if src_region == region else boto3.client("s3", region_name=src_region)
    )
    # CopyObject is issued against the *destination* bucket's endpoint.
    dst_client = boto3.client("s3", region_name=region)

    try:
        source = _list(src_client, src_bucket, src_prefix)
    except Exception as exc:
        raise SystemExit(
            f"cannot list s3://{src_bucket}/{src_prefix} ({type(exc).__name__}: {exc}).\n"
            "If this account cannot read the shared benchmark bucket, generate a\n"
            "stand-in instead:  python arrow_rs_fixtures.py --out <s3 uri> --shapes narrow"
        )
    if not source:
        raise SystemExit(f"no objects under s3://{src_bucket}/{src_prefix}/")

    existing = {
        o["Key"]: o["Size"] for o in _list(dst_client, dst_bucket, dst_prefix or "")
    }
    total = sum(o["Size"] for o in source)
    print(
        f"source: {len(source)} objects, {total / 1e9:.2f} GB "
        f"in s3://{src_bucket}/{src_prefix} ({src_region})"
    )

    copied = skipped = 0
    for obj in sorted(source, key=lambda o: o["Key"]):
        name = obj["Key"][len(src_prefix) :].lstrip("/")
        dst_key = f"{dst_prefix}/{name}" if dst_prefix else name
        if existing.get(dst_key) == obj["Size"]:
            skipped += 1
            continue
        print(f"  copy {name}  ({obj['Size'] / 1e6:.0f} MB)")
        if not dry_run:
            dst_client.copy_object(
                Bucket=dst_bucket,
                Key=dst_key,
                CopySource={"Bucket": src_bucket, "Key": obj["Key"]},
            )
        copied += 1

    print(f"staged -> {dst}  ({copied} copied, {skipped} already present)")
    return 0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dst", required=True, help="s3://bucket/prefix to stage into")
    p.add_argument("--src", default=None, help="Override the source URI")
    p.add_argument("--sf", type=int, default=10, help="TPC-H scale factor (1 or 10)")
    p.add_argument("--region", default="us-west-2")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    src = args.src or (
        f"s3://{DEFAULT_SRC_BUCKET}/{DEFAULT_SRC_PREFIX.format(sf=args.sf)}"
    )
    sys.exit(stage(src, args.dst, args.region, args.dry_run))


if __name__ == "__main__":
    main()
