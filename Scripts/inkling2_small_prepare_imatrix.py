#!/usr/bin/env python3
"""Audit an Inkling2SmallImatrix routed-expert imatrix and attach it to an Inkling2SmallConverter
recipe CSV, as Scripts/glm5_3_prepare_imatrix.py does for GLM 5.3 Flash.

The collector writes a legacy llama.cpp ``.dat`` keyed by the converter's logical names
(``layers.N.mlp.experts.{gate,up,down}_proj.weight``) with expert-major per-expert means
(``ncall`` 1): ``[384, 4608]`` for gate/up (the squared normalized MLP input) and ``[384, 2304]``
for down (the squared weighted SwiGLU output). An expert no calibration token reached is all
zeros; with ``--fill-unobserved`` it gets ones, which quantizes it as without an imatrix.
Observed values are kept byte-identical. The recipe is a ``name,format`` CSV covering every
tensor; the output adds the base64 imatrix of each routed bank, and an audit report beside it.

Requires NumPy.
"""

import argparse
import base64
import csv
import hashlib
import json
from pathlib import Path
import re
import struct

import numpy as np

EXPERTS = 384
LAYERS = 48
COLUMNS = {"gate_proj": 4608, "up_proj": 4608, "down_proj": 2304}
NAME = re.compile(r"layers\.(\d+)\.mlp\.experts\.(gate_proj|up_proj|down_proj)\.weight")


def prepare(source, recipe, output, fill_unobserved=False):
    with recipe.open(newline="") as file:
        rows = list(csv.DictReader(file))
    by_name = {row["name"]: row for row in rows}
    if len(by_name) != len(rows):
        raise ValueError("duplicate recipe keys")
    data = source.read_bytes()
    offset = 0

    def integer():
        nonlocal offset
        value = struct.unpack_from("<i", data, offset)[0]
        offset += 4
        return value

    def string():
        nonlocal offset
        size = integer()
        if size < 0 or offset + size > len(data):
            raise ValueError("invalid string extent")
        value = data[offset:offset + size].decode("utf-8")
        offset += size
        return value

    expected = {f"layers.{layer}.mlp.experts.{projection}.weight"
                for layer in range(LAYERS) for projection in COLUMNS}
    if integer() != len(expected):
        raise ValueError(f"expected {len(expected)} routed banks")
    entries, missing, stats = {}, {}, {}
    for _ in expected:
        name = string()
        if name not in expected or name in entries:
            raise ValueError(f"unexpected or duplicate bank: {name}")
        if name not in by_name or by_name[name]["format"] in ("F16", "F32"):
            raise ValueError(f"{name}: the recipe must quantize it")
        columns = COLUMNS[NAME.fullmatch(name).group(2)]
        calls, count = integer(), integer()
        if calls != 1 or count != EXPERTS * columns:
            raise ValueError(f"{name}: expected per-expert means, expert-major")
        if offset + count * 4 > len(data):
            raise ValueError(f"{name}: truncated vector")
        values = np.frombuffer(data, dtype="<f4", count=count, offset=offset)
        entries[name] = dict(offset=offset, count=count, sha256=hashlib.sha256(values).hexdigest())
        offset += count * 4
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{name}: invalid importance values")
        vectors = values.reshape(EXPERTS, columns)
        unseen = np.flatnonzero(~np.any(vectors, axis=1)).tolist()
        missing[name] = unseen
        if unseen and not fill_unobserved:
            raise ValueError(f"{name}: unobserved experts {unseen}; use --fill-unobserved")
        observed = vectors[np.any(vectors, axis=1)]
        # Spread of importance across input columns: max over median per expert. A flat vector
        # (ratio ~1) would make the imatrix a no-op.
        ratio = observed.max(axis=1) / np.maximum(np.median(observed, axis=1), 1e-30)
        stats[name] = dict(mean=float(observed.mean()), max=float(observed.max()),
                           zero_columns=int((observed == 0).sum()),
                           max_over_median=[float(ratio.min()), float(np.median(ratio)),
                                            float(ratio.max())])
    for layer in range(LAYERS):
        gate, up, down = (f"layers.{layer}.mlp.experts.{p}.weight" for p in COLUMNS)
        if entries[gate]["sha256"] != entries[up]["sha256"]:
            raise ValueError(f"layer {layer}: gate/up statistics differ")
        if not missing[gate] == missing[up] == missing[down]:
            raise ValueError(f"layer {layer}: inconsistent expert coverage")
    chunks, dataset = integer(), string()
    if offset != len(data) or chunks <= 0:
        raise ValueError("unexpected footer")
    report = dict(source=str(source.resolve()), sha256=hashlib.sha256(data).hexdigest(),
                  recipe=str(recipe.resolve()), output=str(output.resolve()),
                  entries=len(entries), collection_chunks=chunks, dataset=dataset,
                  unobserved={name: ids for name, ids in missing.items() if ids},
                  unobserved_policy="ones" if fill_unobserved else "reject",
                  statistics=stats)
    with output.open("x", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["name", "format", "imatrix"])
        for row in rows:
            name = row["name"]
            payload = ""
            if name in entries:
                entry = entries[name]
                values = np.frombuffer(data, dtype="<f4", count=entry["count"],
                                       offset=entry["offset"]).copy()
                values.reshape(EXPERTS, -1)[missing[name], :] = 1
                payload = base64.b64encode(values.astype("<f4").tobytes()).decode("ascii")
            writer.writerow([name, row["format"], payload])
    with output.with_suffix(output.suffix + ".json").open("x") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    summary = {key: value for key, value in report.items() if key != "statistics"}
    summary["unobserved"] = {name: len(ids) for name, ids in missing.items() if ids}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fill-unobserved", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(args.output.suffix + ".json").exists():
        parser.error("output or audit report already exists")
    prepare(args.source, args.recipe, args.output, args.fill_unobserved)
