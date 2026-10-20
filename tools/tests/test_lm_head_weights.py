"""Tests for the LM head byte extraction.

The fixture is synthetic: the naming convention and the packed-uint8 shapes are
the only things the extractor relies on, so a tiny graph keeps it honest.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import onnx
from onnx import TensorProto, helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm_head_weights import find_lm_head, tensor_bytes  # noqa: E402


def _raw_initializer(name, dims, dtype, width):
    count = 1
    for dim in dims:
        count *= dim
    return helper.make_tensor(name, dtype, dims, bytes(count * width), raw=True)


def build_fixture() -> onnx.ModelProto:
    initializers = [
        # lm_head: packed q4 weight [64, 3, 16] uint8 = 3072 bytes,
        # scales [64, 3] fp16 = 384 bytes, zero points [64, 2] uint8 = 128 bytes.
        _raw_initializer("lm_head_weight", [64, 3, 16], TensorProto.UINT8, 1),
        _raw_initializer("lm_head_scales", [64, 3], TensorProto.FLOAT16, 2),
        _raw_initializer("lm_head_zp", [64, 2], TensorProto.UINT8, 1),
        # a smaller decoy that must not win
        _raw_initializer("layer_weight", [8, 3, 16], TensorProto.UINT8, 1),
    ]

    nodes = [
        helper.make_node(
            "MatMulNBits",
            ["x", "layer_weight"],
            ["hidden"],
            name="/model/layers.0/mlp/gate_up_proj/MatMul_Quant",
        ),
        helper.make_node(
            "MatMulNBits",
            ["hidden", "lm_head_weight", "lm_head_scales", "lm_head_zp"],
            ["logits"],
            name="/lm_head/MatMul_Quant",
        ),
    ]

    graph = helper.make_graph(
        nodes,
        "lm-head-fixture",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 64])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT16, [1, 64])],
        initializer=initializers,
    )
    return helper.make_model(graph, producer_name="test")


class LmHeadWeightsTest(unittest.TestCase):
    def test_sums_every_initializer_the_lm_head_reads(self) -> None:
        lm_head = find_lm_head(build_fixture())

        self.assertEqual(lm_head["node"], "/lm_head/MatMul_Quant")
        self.assertEqual(lm_head["selection"], "named")
        self.assertEqual(lm_head["totalBytes"], 3072 + 384 + 128)
        self.assertEqual(
            {tensor["name"]: tensor["bytes"] for tensor in lm_head["tensors"]},
            {
                "lm_head_weight": 3072,
                "lm_head_scales": 384,
                "lm_head_zp": 128,
            },
        )

    def test_packed_uint8_counts_one_byte_per_element(self) -> None:
        tensor = _raw_initializer("w", [262144, 48, 16], TensorProto.UINT8, 1)
        self.assertEqual(tensor_bytes(tensor), 262144 * 48 * 16)

    def test_a_graph_without_matmulnbits_is_rejected(self) -> None:
        graph = helper.make_graph([], "empty", [], [])
        model = helper.make_model(graph, producer_name="test")

        with self.assertRaisesRegex(ValueError, "no MatMulNBits"):
            find_lm_head(model)


if __name__ == "__main__":
    unittest.main()
