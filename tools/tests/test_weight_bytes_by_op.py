"""Tests for initializer bytes grouped by the consuming op.

The fixture is synthetic: sharing and dtype width are the only things the
grouping relies on, so a tiny graph keeps it honest.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import onnx
from onnx import TensorProto, helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from weight_bytes_by_op import group_by_op  # noqa: E402


def raw_initializer(name, dims, dtype, width):
    count = 1
    for dim in dims:
        count *= dim
    return helper.make_tensor(name, dtype, dims, bytes(count * width), raw=True)


def build_fixture() -> onnx.ModelProto:
    initializers = [
        # Shared by two MatMul nodes: counted once for the op type.
        raw_initializer("shared_weight", [4, 4], TensorProto.FLOAT16, 2),
        # Four rotary caches, each referenced by several layers.
        raw_initializer("cos_cache", [8], TensorProto.FLOAT, 4),
        raw_initializer("norm_weight", [4], TensorProto.FLOAT16, 2),
    ]
    nodes = [
        helper.make_node("MatMul", ["shared_weight", "x"], ["y0"]),
        helper.make_node("MatMul", ["x", "shared_weight"], ["y1"]),
        helper.make_node("RotaryEmbedding", ["cos_cache", "x"], ["y2"]),
        helper.make_node("RotaryEmbedding", ["cos_cache", "x"], ["y3"]),
        helper.make_node("SimplifiedLayerNormalization", ["x", "norm_weight"], ["y4"]),
    ]
    graph = helper.make_graph(
        nodes,
        "weight-bytes-fixture",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [4, 4])],
        [helper.make_tensor_value_info("y4", TensorProto.FLOAT16, [4, 4])],
        initializer=initializers,
    )
    return helper.make_model(graph)


class WeightBytesByOpTest(unittest.TestCase):
    def test_shared_initializer_counts_once_per_op_type(self) -> None:
        grouped = group_by_op(build_fixture().graph)
        rows = {row["opType"]: row for row in grouped["rows"]}

        # 4*4 fp16 = 32 bytes, used by two MatMul nodes.
        self.assertEqual(rows["MatMul"]["bytes"], 32)
        self.assertEqual(rows["MatMul"]["initializers"], 1)
        # 8 fp32 = 32 bytes, used by two RotaryEmbedding nodes.
        self.assertEqual(rows["RotaryEmbedding"]["bytes"], 32)
        # 4 fp16 = 8 bytes.
        self.assertEqual(rows["SimplifiedLayerNormalization"]["bytes"], 8)

    def test_totals_cover_every_initializer(self) -> None:
        grouped = group_by_op(build_fixture().graph)

        self.assertEqual(grouped["totalInitializerBytes"], 72)
        self.assertEqual(grouped["initializers"], 3)
        self.assertEqual(grouped["sharedInitializers"], 2)
        self.assertEqual(grouped["sharedBytes"], 64)

    def test_rows_are_sorted_by_bytes(self) -> None:
        grouped = group_by_op(build_fixture().graph)
        values = [row["bytes"] for row in grouped["rows"]]

        self.assertEqual(values, sorted(values, reverse=True))


if __name__ == "__main__":
    unittest.main()
