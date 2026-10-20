#!/usr/bin/env python3
"""Record the bytes the LM head MatMulNBits reads.

This is a one-file data extraction, not a model run. It exists so the Gate 6
utilization calculation uses a recorded weight size instead of the rounded
201 MB from the Gate 5 report.

The LM head is the `MatMulNBits` node under `/lm_head/`. All of its initializer
inputs are counted: the packed 4-bit weight, its scales, and its zero points.

Usage:
    python tools/lm_head_weights.py \
        --model ~/Projects/gemma-browser-lab/models/onnx-community/gemma-4-E2B-it-ONNX/onnx/decoder_model_merged_q4f16.onnx \
        --json runs/2026-10-20-lm-head-weights.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import onnx
from onnx import TensorProto


# Bytes per element for the dtypes this export uses. 4-bit tensors arrive
# packed into uint8, so they do not appear here.
LM_HEAD_MARKER = "/lm_head/"

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


def total_initializer_bytes(model: onnx.ModelProto) -> int:
    """All weight bytes in the graph, for the whole-model roofline."""
    return sum(tensor_bytes(tensor) for tensor in model.graph.initializer)


def find_lm_head(model: onnx.ModelProto) -> dict:
    """Return the MatMulNBits node under /lm_head/ and its initializer bytes."""
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    candidates = []

    for node in model.graph.node:
        if node.op_type != "MatMulNBits":
            continue

        tensors = []
        for name in node.input:
            tensor = initializers.get(name)
            if tensor is None:
                continue
            tensors.append(
                {
                    "name": tensor.name,
                    "dims": list(tensor.dims),
                    "dataType": int(tensor.data_type),
                    "dataTypeName": TensorProto.DataType.Name(tensor.data_type),
                    "bytes": tensor_bytes(tensor),
                }
            )

        candidates.append(
            {
                "node": node.name,
                "inputs": list(node.input),
                "outputs": list(node.output),
                "tensors": tensors,
                "totalBytes": sum(tensor["bytes"] for tensor in tensors),
            }
        )

    if not candidates:
        raise ValueError("the graph has no MatMulNBits node")

    named = [
        candidate
        for candidate in candidates
        if LM_HEAD_MARKER in candidate["node"]
        or any(LM_HEAD_MARKER in name for name in candidate["inputs"])
    ]
    if len(named) == 1:
        return {**named[0], "selection": "named"}

    # Fall back to the largest weight when the naming convention changes, and
    # record that the choice was positional rather than by name.
    return {
        **max(candidates, key=lambda candidate: candidate["totalBytes"]),
        "selection": "largest",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="q4 decoder ONNX file")
    parser.add_argument("--json", required=True, help="output JSON path")
    parser.add_argument("--name", help="artifact name (base, gate1, gate2)")
    args = parser.parse_args()

    model_path = Path(args.model).expanduser()
    model = onnx.load(str(model_path), load_external_data=False)

    lm_head = find_lm_head(model)
    record = {
        "name": args.name or model_path.stem,
        "model": str(model_path),
        "graphInitializerBytes": total_initializer_bytes(model),
        **lm_head,
    }

    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")

    print(f"artifact: {record['name']} ({record['selection']})")
    print(f"node: {record['node']}")
    for tensor in record["tensors"]:
        print(
            f"  {tensor['name']}: {tensor['dims']} {tensor['dataTypeName']} "
            f"= {tensor['bytes']} bytes"
        )
    print(f"total: {record['totalBytes']} bytes")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
