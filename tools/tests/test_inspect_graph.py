"""Tests for the Gate 0 census.

The fixture is intentionally synthetic: two transformer layers with the exact
naming conventions the real export uses. This keeps the classifier honest
without loading a 673 KB model in every test.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import onnx
from onnx import TensorProto, helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inspect_graph import census  # noqa: E402


def _node(op: str, name: str, inputs: list[str], outputs: list[str]) -> onnx.NodeProto:
    return helper.make_node(op, inputs, outputs, name=name)


def build_fixture() -> onnx.ModelProto:
    nodes = [
        # layer 0: already fused GQA
        _node(
            "com.microsoft.GroupQueryAttention",
            "/model/layers.0/attn/GroupQueryAttention",
            ["q0", "k0", "v0"],
            ["attn0"],
        ),
        _node("MatMul", "/model/layers.0/attn/o_proj/MatMul_Quant", ["attn0", "w0"], ["o0"]),
        # layer 1: manual attention with q/k norm kept outside the fused core
        _node("Reshape", "/model/layers.1/attn/manual/q_norm/Reshape_1", ["q1", "s"], ["qn1"]),
        _node(
            "SimplifiedLayerNormalization",
            "/model/layers.1/attn/manual/q_norm/SimplifiedLayerNormalization",
            ["qn1", "g"],
            ["qnn1"],
        ),
        _node("Reshape", "/model/layers.1/attn/manual/q_norm/Reshape_2", ["qnn1", "s"], ["qn2"]),
        _node("Reshape", "/model/layers.1/attn/manual/k_norm/Reshape_1", ["k1", "s"], ["kn1"]),
        _node(
            "SimplifiedLayerNormalization",
            "/model/layers.1/attn/manual/k_norm/SimplifiedLayerNormalization",
            ["kn1", "g"],
            ["knn1"],
        ),
        _node("Reshape", "/model/layers.1/attn/manual/k_norm/Reshape_2", ["knn1", "s"], ["kn2"]),
        _node("Reshape", "/model/layers.1/attn/manual/q_reshape", ["qn2", "s"], ["qr"]),
        _node("Transpose", "/model/layers.1/attn/manual/q_transpose", ["qr"], ["qt"]),
        _node("MatMul", "/model/layers.1/attn/manual/scores/MatMul", ["qt", "kt"], ["scores"]),
        _node("Softmax", "/model/layers.1/attn/manual/softmax/Softmax", ["scores"], ["probs"]),
        _node("MatMul", "/model/layers.1/attn/manual/output/MatMul", ["probs", "v1"], ["out"]),
        _node("Reshape", "/model/layers.1/attn/manual/output/Reshape", ["out", "s"], ["ores"]),
        _node("MatMul", "/model/layers.1/attn/o_proj/MatMul_Quant", ["ores", "w1"], ["o1"]),
        # final norm is named layers.2, beyond the declared 2-layer stack
        _node(
            "SimplifiedLayerNormalization",
            "/model/layers.2/final_norm_layernorm/LayerNorm",
            ["o1", "g"],
            ["final"],
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "fixture",
        [helper.make_tensor_value_info("q0", TensorProto.FLOAT, [1, 1, 8])],
        [helper.make_tensor_value_info("final", TensorProto.FLOAT, [1, 1, 8])],
        initializer=[helper.make_tensor("g", TensorProto.FLOAT, [8], [1.0] * 8)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    return model


class CensusTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.model_path = Path(self.tmp.name) / "decoder.onnx"
        onnx.save(build_fixture(), self.model_path)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_classifies_layers_and_counts_conservative_savings(self) -> None:
        report = census(self.model_path, config_path=None)

        self.assertEqual(report["totals"]["gqa_layers"], 1)
        self.assertEqual(report["totals"]["manual_layers"], 1)
        self.assertEqual(report["totals"]["final_norm_layers"], 1)

        # Manual core is 12 nodes; 6 are q/k norm, so 12 - 6 - 1 = 5.
        self.assertEqual(report["fusion"]["conservative_saved"], 5)
        # Aggressive folds the norms in: 12 - 1 = 11.
        self.assertEqual(report["fusion"]["aggressive_saved"], 11)

        manual = next(row for row in report["layers"] if row["index"] == 1)
        self.assertEqual(manual["kind"], "manual")
        self.assertEqual(manual["core_attn_nodes"], 12)
        self.assertEqual(manual["norm_nodes_kept"], 6)

    def test_final_norm_is_not_a_block(self) -> None:
        report = census(self.model_path, config_path=None)
        final = next(row for row in report["layers"] if row["index"] == 2)
        self.assertEqual(final["kind"], "final-norm")
        self.assertEqual(final["saved_conservative"], 0)


if __name__ == "__main__":
    unittest.main()
