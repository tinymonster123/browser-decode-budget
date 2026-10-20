#!/usr/bin/env python3
"""Record initializer bytes grouped by the op that consumes them.

Gate 7 uses these bytes to turn a per-op time into an effective bandwidth.
Summing per node would double-count shared tensors (the rotary cos/sin caches
are referenced by many layers), so each op type's set of initializer names is
deduplicated before summing.

Usage:
    python tools/weight_bytes_by_op.py \
        --model ~/Projects/gemma-browser-lab/models/onnx-community/gemma-4-E2B-it-ONNX/onnx/decoder_model_merged_q4f16.onnx \
        --json runs/2026-10-20-exp7-weight-bytes-by-op.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import onnx
from onnx import TensorProto


DTYPE_BYTES = {
    TensorProto.FLOAT: 4,
    TensorProto.DOUBLE: 8,
    TensorProto.FLOAT16: 2,
    TensorProto.BFLOAT16: 2,
    TensorProto.INT8: 1,
    TensorProto.UINT8: 1,
    TensorProto.INT16: 2,
    TensorProto.UINT16: 2,
    TensorProto.INT32: 4,
    TensorProto.UINT32: 4,
    TensorProto.INT64: 8,
    TensorProto.UINT64: 8,
}


def tensor_bytes(tensor: onnx.TensorProto) -> int:
    count = 1
    for dim in tensor.dims:
        count *= dim
    try:
        width = DTYPE_BYTES[tensor.data_type]
    except KeyError as error:
        raise ValueError(
            f"tensor {tensor.name!r} has unsupported dtype "
            f"{TensorProto.DataType.Name(tensor.data_type)}"
        ) from error
    return int(math.ceil(count * width))


def group_by_op(graph: onnx.GraphProto) -> dict:
    """Unique initializer bytes per consuming op type.

    A tensor referenced by several nodes of the same op counts once. A tensor
    referenced by two different op types counts in both, because the question
    is what each op class reads, not how the storage is shared.
    """
    sizes = {tensor.name: tensor_bytes(tensor) for tensor in graph.initializer}

    names_by_op: dict[str, set[str]] = defaultdict(set)
    consumer_count: Counter[str] = Counter()
    for node in graph.node:
        for name in set(node.input):
            if name in sizes:
                names_by_op[node.op_type].add(name)
                consumer_count[name] += 1

    total = sum(sizes.values())
    rows = []
    for op_type, names in names_by_op.items():
        used = sum(sizes[name] for name in names)
        rows.append(
            {
                "opType": op_type,
                "initializers": len(names),
                "bytes": used,
                "share": used / total if total else None,
            }
        )
    rows.sort(key=lambda row: row["bytes"], reverse=True)

    shared = {
        name: count for name, count in consumer_count.items() if count > 1
    }
    return {
        "totalInitializerBytes": total,
        "initializers": len(sizes),
        "sharedInitializers": len(shared),
        "sharedBytes": sum(sizes[name] for name in shared),
        "rows": rows,
    }


def tensor_fingerprints_for_op(graph: onnx.GraphProto, op_type: str) -> list[dict]:
    """Distinct initializer-size fingerprints for one op type.

    A node like MatMulNBits binds several initializers (packed weight, scales,
    zero points). Their sorted sizes are the fingerprint; nodes with the same
    fingerprint read the same number of bytes.
    """
    sizes = {}
    dims = {}
    for tensor in graph.initializer:
        sizes[tensor.name] = tensor_bytes(tensor)
        dims[tensor.name] = list(tensor.dims)

    fingerprints: dict[tuple[int, ...], dict] = {}
    for node in graph.node:
        if node.op_type != op_type:
            continue
        names = [name for name in set(node.input) if name in sizes]
        if not names:
            continue
        values = sorted(sizes[name] for name in names)
        key = tuple(values)
        entry = fingerprints.get(key)
        if entry is None:
            # The packed weight is the largest initializer whose name says
            # "weight"; equal-sized scale tensors would otherwise win the
            # tie. Kept as an example, not as a per-node claim.
            weight_names = [name for name in names if "weight" in name]
            largest = max(weight_names or names, key=lambda name: sizes[name])
            entry = {
                "sizes": list(key),
                "bytes": sum(key),
                "nodes": 0,
                "sample": node.name,
                "dims": dims[largest],
            }
        entry["nodes"] += 1
        fingerprints[key] = entry

    return sorted(fingerprints.values(), key=lambda entry: -entry["bytes"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument(
        "--op",
        help="also record the sorted per-tensor sizes for this op type",
    )
    args = parser.parse_args()

    graph = onnx.load(args.model, load_external_data=False).graph
    grouped = group_by_op(graph)
    payload = {
        "name": Path(args.model).stem,
        "model": str(Path(args.model).resolve()),
        **grouped,
    }

    if args.op:
        payload["fingerprints"] = {
            "opType": args.op,
            "entries": tensor_fingerprints_for_op(graph, args.op),
        }

    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")

    print(
        f"total initializer bytes: {grouped['totalInitializerBytes']:,} "
        f"({grouped['initializers']} tensors)"
    )
    print(
        f"shared tensors: {grouped['sharedInitializers']} "
        f"({grouped['sharedBytes']:,} bytes)"
    )
    for row in grouped["rows"]:
        print(
            f"{row['opType']:32s} {row['bytes']:>14,} "
            f"{100 * row['share']:5.1f}%  {row['initializers']} tensors"
        )
    if args.op:
        print(
            f"{args.op}: {len(payload['fingerprints']['entries'])} fingerprints"
        )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
