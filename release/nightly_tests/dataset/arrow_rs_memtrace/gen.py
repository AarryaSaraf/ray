"""Generate a Parquet fixture for the v1/v2/v3 memory sweep.

    python gen.py <rows> <row_group_size> [out_path]

7 columns (2 int/float + 4 wide strings) so decoded bytes/row >> on-disk bytes
-- that gap is what makes a whole-row-group materialization (PyArrow) hurt and a
byte-budgeted streaming decode (arrow-rs) win. Always writes the page index
(required by the arrow-rs crate). Chunked row-group writes keep driver RSS ~flat
regardless of total rows, so this never OOMs the box while generating.

Layout knob = row_group_size:
    row_group_size == rows      -> ONE giant row group  (arrow-rs's thesis case:
                                   PyArrow must materialize the whole group;
                                   arrow-rs K-splits it under a byte budget)
    row_group_size <  rows      -> many row groups       (Ray's fragment pool can
                                   already parallelize; arrow-rs ~ parity)
"""
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _strcol(rng, n, width):
    """n zero-padded numeric strings, `width` chars each (int64-safe)."""
    hi = 10 ** min(width, 15)  # 10**24 overflows int64; cap the numeric range
    base = rng.integers(0, hi, size=n)
    return pa.array(np.char.zfill(base.astype("U"), width))


def _chunk_table(rng, n):
    return pa.table(
        {
            "id": pa.array(np.arange(n, dtype=np.int64)),
            "f0": pa.array(rng.random(n)),
            "f1": pa.array(rng.random(n)),
            "s0": _strcol(rng, n, 16),
            "s1": _strcol(rng, n, 16),
            "s2": _strcol(rng, n, 24),
            "s3": _strcol(rng, n, 24),
        }
    )


def main():
    rows = int(sys.argv[1])
    rg = int(sys.argv[2])
    out = sys.argv[3] if len(sys.argv) > 3 else f"leak_linux_data/leak_{rows}_{rg}.parquet"
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    rng = np.random.default_rng(0)
    schema = _chunk_table(rng, 1).schema
    written = 0
    # Write one row group at a time so we never hold more than `rg` rows in the
    # driver at once (a lone-big-rg fixture still buffers only that one group).
    with pq.ParquetWriter(out, schema, compression="snappy",
                          write_page_index=True) as w:
        while written < rows:
            n = min(rg, rows - written)
            w.write_table(_chunk_table(rng, n), row_group_size=n)
            written += n
            print(f"  ... {written:,}/{rows:,} rows", flush=True)

    md = pq.ParquetFile(out).metadata
    print(f"generated {out}\n  rows={md.num_rows:,}  row_groups={md.num_row_groups}  "
          f"on-disk={os.path.getsize(out) / 1024 / 1024:.0f}MB")


if __name__ == "__main__":
    main()
