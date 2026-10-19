"""Tests for the generic pieces of the Gate 1 surgery."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import onnx
from onnx import TensorProto, helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fuse_manual_attention import FusionError, toposort, validate_graph  # noqa: E402


def node(op: str, name: str, inputs: list[str], outputs: list[str]) -> onnx.NodeProto:
    return helper.make_node(op, inputs, outputs, name=name)


class ToposortTest(unittest.TestCase):
    def test_orders_dependencies(self) -> None:
        nodes = [
            node("Identity", "c", ["b"], ["out"]),
            node("Identity", "a", ["x"], ["a_out"]),
            node("Identity", "b", ["a_out"], ["b"]),
        ]
        ordered = [item.name for item in toposort(nodes)]
        self.assertLess(ordered.index("a"), ordered.index("b"))
        self.assertLess(ordered.index("b"), ordered.index("c"))

    def test_cycle_is_rejected(self) -> None:
        nodes = [
            node("Identity", "a", ["b"], ["a_out"]),
            node("Identity", "b", ["a_out"], ["b"]),
        ]
        with self.assertRaises(FusionError):
            toposort(nodes)


class ValidateGraphTest(unittest.TestCase):
    def test_dangling_input_is_rejected(self) -> None:
        graph = helper.make_graph(
            [node("Identity", "bad", ["missing"], ["out"])],
            "bad",
            [],
            [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1])],
        )
        model = helper.make_model(graph)
        with self.assertRaises(FusionError):
            validate_graph(model)

    def test_complete_graph_passes(self) -> None:
        graph = helper.make_graph(
            [node("Identity", "ok", ["x"], ["out"])],
            "ok",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1])],
        )
        model = helper.make_model(graph)
        validate_graph(model)


if __name__ == "__main__":
    unittest.main()
