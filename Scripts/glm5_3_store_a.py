#!/usr/bin/env python3
"""Store A = exp(A_log) in a GLM-5.3-Flash s4nnc checkpoint written before GLM5_3Converter did.

Each KDA layer's `__text_model__[t-layers.N.self_attn.A_log-0-0]` (FP32, 64 values) becomes
`...self_attn.A-0-0]` holding exp(A_log), rounded from double precision as GLM5_3Converter does.
The values are rewritten where they lie, in SQLite or in the trailer past PRAGMA user_version, and
the rows are renamed in one transaction. The SQLite pages must not grow into the trailer, so the
script refuses to finish if they would; run it on a copy (an APFS clone, `cp -c`) and replace the
original once the copy checks out.

Usage: glm5_3_store_a.py CHECKPOINT
"""

import math
import os
import sqlite3
import struct
import sys

FP32 = 0x4000  # CCV_32F


def main(path):
    con = sqlite3.connect(path, isolation_level=None)
    page_size = con.execute("PRAGMA page_size").fetchone()[0]
    pages_before = con.execute("PRAGMA page_count").fetchone()[0]
    trailer = con.execute("PRAGMA user_version").fetchone()[0]
    size_before = os.path.getsize(path)
    if trailer and pages_before * page_size > trailer:
        raise SystemExit("SQLite already overlaps the trailer")
    if con.execute("SELECT count(*) FROM tensors WHERE name LIKE '%.self_attn.A-0-0]'").fetchone()[0]:
        raise SystemExit("already has A tensors")
    rows = con.execute(
        "SELECT name, datatype, data FROM tensors WHERE name LIKE '%.self_attn.A_log-0-0]'").fetchall()
    if not rows:
        raise SystemExit("no A_log tensors")
    con.execute("BEGIN IMMEDIATE")
    with open(path, "r+b") as f:
        for name, datatype, data in rows:
            if datatype != FP32:
                raise SystemExit(f"{name}: datatype {datatype:#x}, not FP32")
            if len(data) in (16, 24):
                # External: the tensor lies in the trailer; codecs with a block size lead with it.
                offset, length = struct.unpack_from("<QQ", data, len(data) - 16)
                f.seek(trailer + offset)
                raw = f.read(length)
                if len(raw) != length or length % 4:
                    raise SystemExit(f"{name}: short read")
                values = struct.unpack(f"<{length // 4}f", raw)
                packed = struct.pack(f"<{len(values)}f", *(math.exp(v) for v in values))
                f.seek(trailer + offset)
                f.write(packed)
                new_data = data
            else:
                if len(data) % 4:
                    raise SystemExit(f"{name}: inline blob of {len(data)} bytes")
                values = struct.unpack(f"<{len(data) // 4}f", data)
                new_data = struct.pack(f"<{len(values)}f", *(math.exp(v) for v in values))
            con.execute(
                "UPDATE tensors SET name = ?, data = ? WHERE name = ?",
                (name.replace(".A_log-0-0]", ".A-0-0]"), new_data, name))
            print(f"{name} -> A: {len(values)} values, {min(values):.4f}..{max(values):.4f} -> "
                  f"{math.exp(min(values)):.4f}..{math.exp(max(values)):.4f}")
        f.flush()
        os.fsync(f.fileno())
    pages_after = con.execute("PRAGMA page_count").fetchone()[0]
    if trailer and pages_after * page_size > trailer:
        con.execute("ROLLBACK")
        raise SystemExit(f"SQLite would grow to {pages_after} pages, into the trailer at {trailer}; "
                         "rolled back the rows, but the trailer values are already rewritten: discard this copy")
    con.execute("COMMIT")
    check = con.execute("PRAGMA integrity_check").fetchone()[0]
    con.close()
    if check != "ok" or os.path.getsize(path) != size_before:
        raise SystemExit(f"integrity {check}, size {size_before} -> {os.path.getsize(path)}")
    print(f"{len(rows)} tensors; SQLite {pages_before} -> {pages_after} pages of {page_size}, "
          f"trailer at {trailer}; size unchanged")


if __name__ == "__main__":
    main(sys.argv[1])
