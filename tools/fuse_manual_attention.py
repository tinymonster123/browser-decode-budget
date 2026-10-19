#!/usr/bin/env python3
"""Gate 1: replace decomposed full-attention blocks with GroupQueryAttention.

Scope is deliberately narrow: only layers whose attention is written as
`attn/manual/*` and which own their `past_key_values.<n>` cache. In the Gemma 4
E2B export those are layers 4, 9 and 14.

The rewrite keeps q_norm / k_norm / rotary outside the fused op, so the only
numerical change is the attention kernel's accumulation order.

Usage:
    python tools/fuse_manual_attention.py \
        --model ~/Projects/gemma-browser-lab/models/onnx-community/gemma-4-E2B-it-ONNX/onnx/decoder_model_merged_q4f16.onnx \
        --layers 4,9,14 \
        --out-dir models/derived/gemma4-e2b-q4f16-gate1
"""

from __future__ import annotations

import argparse
import heapq
import json
import sys
from pathlib import Path

import onnx
from onnx import helper

from artifact import write_artifact

NUM_HEADS = 8
KV_NUM_HEADS = 1
SCALE = 1.0

SEQ_LENS = "/model/attn_mask_reformat/attn_mask_subgraph/Expand/Cast/output_0"
TOTAL_SEQ = "/model/attn_mask_reformat/attn_mask_subgraph/Gather/Cast/output_0"
ATTENTION_BIAS = "/model/gqa_attention_bias/Expand/output_0"


class FusionError(RuntimeError):
    pass


def toposort(nodes: list[onnx.NodeProto]) -> list[onnx.NodeProto]:
    """Stable topological sort, preserving the original order among ready nodes."""
    producers: dict[str, int] = {}
    for index, node in enumerate(nodes):
        for output in node.output:
            if output in producers:
                raise FusionError(f"duplicate producer for {output!r}")
            producers[output] = index

    adjacency: list[set[int]] = [set() for _ in nodes]
    indegree = [0] * len(nodes)
    for index, node in enumerate(nodes):
        deps = {
            producers[value]
            for value in node.input
            if value and value in producers and producers[value] != index
        }
        indegree[index] = len(deps)
        for dep in deps:
            adjacency[dep].add(index)

    ready = [index for index, degree in enumerate(indegree) if degree == 0]
    heapq.heapify(ready)
    ordered: list[onnx.NodeProto] = []
    while ready:
        index = heapq.heappop(ready)
        ordered.append(nodes[index])
        for consumer in sorted(adjacency[index]):
            indegree[consumer] -= 1
            if indegree[consumer] == 0:
                heapq.heappush(ready, consumer)

    if len(ordered) != len(nodes):
        raise FusionError("graph has a cycle after fusion")
    return ordered


def validate_graph(model: onnx.ModelProto) -> None:
    available = {value.name for value in model.graph.input}
    available |= {tensor.name for tensor in model.graph.initializer}
    produced: set[str] = set()
    for node in model.graph.node:
        for output in node.output:
            if not output:
                continue
            if output in produced:
                raise FusionError(f"output {output!r} is produced twice")
            produced.add(output)
    available |= produced

    for node in model.graph.node:
        for value in node.input:
            if value and value not in available:
                raise FusionError(f"node {node.name!r} reads dangling value {value!r}")
    for value in model.graph.output:
        if value.name not in available:
            raise FusionError(f"graph output {value.name!r} is not produced")


def _require(by_name: dict[str, onnx.NodeProto], name: str) -> onnx.NodeProto:
    node = by_name.get(name)
    if node is None:
        raise FusionError(f"expected node not found: {name}")
    return node


def _single_output(node: onnx.NodeProto) -> str:
    if len(node.output) != 1:
        raise FusionError(f"expected one output on {node.name!r}, got {list(node.output)}")
    return node.output[0]


def fuse_layer(model: onnx.ModelProto, layer: int) -> dict:
    by_name = {node.name: node for node in model.graph.node}
    prefix = f"/model/layers.{layer}/attn/"

    q_rotary = _require(by_name, prefix + "q_rotary/RotaryEmbedding")
    k_rotary = _require(by_name, prefix + "k_rotary/RotaryEmbedding")
    v_norm = _require(by_name, prefix + "v_norm/SimplifiedLayerNormalization")
    output_reshape = _require(by_name, prefix + "manual/output/Reshape")
    o_proj = _require(by_name, prefix + "o_proj/MatMul_Quant")

    # q/k norm stay outside the fused op; make sure they exist so we never
    # silently remove a layer's normalization.
    _require(by_name, prefix + "manual/q_norm/Reshape_1")
    _require(by_name, prefix + "manual/k_norm/Reshape_1")

    past_key = f"past_key_values.{layer}.key"
    past_value = f"past_key_values.{layer}.value"
    present_key = f"present.{layer}.key"
    present_value = f"present.{layer}.value"

    graph_inputs = {value.name for value in model.graph.input}
    graph_outputs = {value.name for value in model.graph.output}
    for name in (past_key, past_value):
        if name not in graph_inputs:
            raise FusionError(f"layer {layer}: missing graph input {name}")
    for name in (present_key, present_value):
        if name not in graph_outputs:
            raise FusionError(f"layer {layer}: missing graph output {name}")

    # Remove the decomposed core, but keep q_norm / k_norm.
    #
    # `mask/pos_unsq` is shared by the donor mask subgraph for the sliding
    # shared-KV layers, so it has to survive even though it lives under
    # `attn/manual/`.
    removed = [
        node
        for node in model.graph.node
        if prefix + "manual/" in node.name
        and prefix + "manual/q_norm/" not in node.name
        and prefix + "manual/k_norm/" not in node.name
        and prefix + "manual/mask/pos_unsq" not in node.name
    ]
    removed_names = {node.name for node in removed}
    if not removed:
        raise FusionError(f"layer {layer}: no manual attention nodes found")

    gqa_output = prefix + "GroupQueryAttention/output_0"
    gqa = helper.make_node(
        "GroupQueryAttention",
        inputs=[
            _single_output(q_rotary),
            _single_output(k_rotary),
            _single_output(v_norm),
            past_key,
            past_value,
            SEQ_LENS,
            TOTAL_SEQ,
            "",
            "",
            "position_ids",
            ATTENTION_BIAS,
        ],
        outputs=[gqa_output, present_key, present_value],
        name=prefix + "GroupQueryAttention",
        domain="com.microsoft",
        num_heads=NUM_HEADS,
        kv_num_heads=KV_NUM_HEADS,
        scale=SCALE,
        local_window_size=-1,
        softcap=0.0,
        do_rotary=0,
        rotary_interleaved=0,
        causal=1,
    )

    # Rewire the consumer of the old attention output.
    old_output = _single_output(output_reshape)
    rewired = 0
    for node in model.graph.node:
        if node.name in removed_names:
            continue
        for index, value in enumerate(node.input):
            if value == old_output:
                node.input[index] = gqa_output
                rewired += 1
    if rewired == 0:
        raise FusionError(f"layer {layer}: no consumer found for {old_output!r}")

    # Layer 14's transposed key is reused by the four shared-KV full layers.
    # Keep a transpose of the fused present key so those layers still work.
    k_scores_node = by_name.get(prefix + "manual/k_scores_transpose")
    extra_nodes: list[onnx.NodeProto] = []
    reused_key = False
    if k_scores_node is not None:
        k_scores_out = _single_output(k_scores_node)
        external_consumers = [
            node
            for node in model.graph.node
            if node.name not in removed_names
            and node.name != k_scores_node.name
            and k_scores_out in node.input
        ]
        if external_consumers:
            extra_nodes.append(
                helper.make_node(
                    "Transpose",
                    [present_key],
                    [k_scores_out],
                    name=prefix + "fused_k_transpose/Transpose",
                    perm=[0, 1, 3, 2],
                )
            )
            reused_key = True

    kept = [node for node in model.graph.node if node.name not in removed_names]
    kept.extend([gqa, *extra_nodes])
    model.graph.ClearField("node")
    model.graph.node.extend(toposort(kept))

    return {
        "layer": layer,
        "removed": len(removed),
        "added": 1 + len(extra_nodes),
        "net_saved": len(removed) - 1 - len(extra_nodes),
        "rewired_consumers": rewired,
        "reused_key_transpose": reused_key,
    }


def fuse_model(model: onnx.ModelProto, layers: list[int]) -> dict:
    # Validate shared inputs once, before touching anything.
    node_names = {node.name for node in model.graph.node}
    for name in (SEQ_LENS, TOTAL_SEQ, ATTENTION_BIAS):
        # These are produced by nodes, so check via outputs.
        producer = any(
            name in node.output for node in model.graph.node
        )
        if not producer:
            raise FusionError(f"shared attention input not produced: {name}")
    if not node_names:
        raise FusionError("graph has no named nodes")

    reports = [fuse_layer(model, layer) for layer in layers]
    validate_graph(model)
    return {
        "layers": reports,
        "total_removed": sum(item["removed"] for item in reports),
        "total_added": sum(item["added"] for item in reports),
        "net_saved": sum(item["net_saved"] for item in reports),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--layers", default="4,9,14")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("models/derived/gemma4-e2b-q4f16-gate1"),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", type=Path, help="write the fusion report as JSON")
    args = parser.parse_args()

    layers = [int(value) for value in args.layers.split(",") if value.strip()]
    model = onnx.load(str(args.model), load_external_data=False)
    report = fuse_model(model, layers)

    out_model = write_artifact(
        model,
        args.model,
        args.out_dir,
        args.force,
        tool="tools/fuse_manual_attention.py",
        producer={"layers": layers},
        fusion=report,
    )
    validate_graph(onnx.load(str(out_model), load_external_data=False))

    print(f"wrote {out_model}")
    print(f"wrote {args.out_dir / 'manifest.json'}")
    for item in report["layers"]:
        print(
            f"  layer {item['layer']:>2}: removed {item['removed']:>2}, "
            f"added {item['added']}, net saved {item['net_saved']}"
            + (" (kept key transpose for shared layers)" if item["reused_key_transpose"] else "")
        )
    print(f"  net saved: {report['net_saved']} nodes")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FusionError as exc:
        print(f"fusion failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
