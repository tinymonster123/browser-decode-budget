#!/usr/bin/env python3
"""Gate 0 census: what is in the decoder graph, and what could GQA absorb?

This tool is deliberately read-only. It answers one question:

    How many per-token nodes could an attention fusion remove?

It does NOT claim a speedup. `dispatch_cost.py` turns the census into a
projection using a measured baseline.

Usage:
    python tools/inspect_graph.py \
        --model ~/Projects/gemma-browser-lab/models/onnx-community/gemma-4-E2B-it-ONNX/onnx/decoder_model_merged_q4f16.onnx \
        --config ~/Projects/gemma-browser-lab/models/onnx-community/gemma-4-E2B-it-ONNX/config.json \
        --json runs/2026-10-19-gate0-census.json
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import onnx

from artifact import sha256_file

LAYER_RE = re.compile(r"/model/layers\.(\d+)/")
PAST_RE = re.compile(r"^past_key_values\.(\d+)\.(key|value)$")

MANUAL_MARKER = "/attn/manual/"
SHARED_MARKER = "/attn/shared/"
GQA_MARKER = "/attn/GroupQueryAttention"

# Nodes that are likely shape/index bookkeeping rather than GPU dispatches.
# We report them separately but keep the total as the conservative denominator.
SHAPE_ONLY_OPS = {
    "Shape",
    "Gather",
    "Range",
    "Less",
    "GreaterOrEqual",
    "Cast",
    "Unsqueeze",
    "Squeeze",
    "Concat",
    "Slice",
    "Expand",
    "Constant",
    "ConstantOfShape",
}


def node_label(node: onnx.NodeProto) -> str:
    """Human-readable name; fall back to the first output for unnamed nodes."""
    if node.name:
        return node.name
    return node.output[0] if node.output else "<unnamed>"


def load_config(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return {
        "num_hidden_layers": payload.get("num_hidden_layers"),
        "hidden_size": payload.get("hidden_size"),
        "num_attention_heads": payload.get("num_attention_heads"),
        "num_key_value_heads": payload.get("num_key_value_heads"),
        "head_dim": payload.get("head_dim"),
        "vocab_size": payload.get("vocab_size"),
        "sliding_window": payload.get("sliding_window"),
        "layer_types": payload.get("layer_types"),
        **(payload.get("text_config") or {}),
    }


def classify_layer(nodes: list[onnx.NodeProto]) -> str:
    labels = [node_label(node) for node in nodes]
    if any(GQA_MARKER in label for label in labels):
        return "gqa"
    if any(MANUAL_MARKER in label for label in labels):
        return "manual"
    if any(SHARED_MARKER in label for label in labels):
        return "shared-kv"
    return "other"


def count_core_attention(nodes: list[onnx.NodeProto]) -> tuple[int, int]:
    """Return (core nodes, norm nodes) for a decomposed attention layer.

    Core nodes are the ones under `attn/manual/` or `attn/shared/`. Norm nodes
    (q_norm / k_norm) can stay outside a fused GQA, so the conservative saving
    keeps them. The aggressive saving folds them in.
    """
    core = 0
    norm = 0
    for node in nodes:
        label = node_label(node)
        if MANUAL_MARKER not in label and SHARED_MARKER not in label:
            continue
        core += 1
        if "norm" in label.lower():
            norm += 1
    return core, norm


def count_gqa_foldable(nodes: list[onnx.NodeProto]) -> int:
    """Q/K norm + rotary nodes that a GQA *could* fold in.

    Subject to runtime support: the GQA spec says q_norm_weight/k_norm_weight
    are honored by CUDA and native WebGPU EPs, and JSEP WebGPU/JS must reject
    them. Treated as upside only, never in the conservative number.
    """
    markers = (
        "/attn/q_norm/",
        "/attn/k_norm/",
        "/attn/q_rotary/",
        "/attn/k_rotary/",
    )
    return sum(1 for node in nodes if any(marker in node_label(node) for marker in markers))


def census(model_path: Path, config_path: Path | None = None) -> dict:
    model = onnx.load(str(model_path), load_external_data=False)
    config = load_config(config_path)

    nodes = list(model.graph.node)
    op_counts = collections.Counter(node.op_type for node in nodes)

    layers: dict[int, list[onnx.NodeProto]] = collections.defaultdict(list)
    for node in nodes:
        match = LAYER_RE.search(node_label(node))
        if match:
            layers[int(match.group(1))].append(node)

    layer_types = config.get("layer_types") or []
    attention_layers = [
        index
        for index, layer_nodes in layers.items()
        if classify_layer(layer_nodes) in {"gqa", "manual", "shared-kv"}
    ]
    num_layers = config.get("num_hidden_layers") or (
        max(attention_layers) + 1 if attention_layers else max(layers) + 1
    )
    rows: list[dict] = []
    totals = collections.Counter()
    saved_conservative = 0
    saved_aggressive = 0

    for index in sorted(layers):
        layer_nodes = layers[index]
        # Indices beyond the declared stack are graph-level nodes such as the
        # final norm, not transformer blocks.
        kind = classify_layer(layer_nodes) if index < num_layers else "final-norm"
        core, norm = count_core_attention(layer_nodes)

        if kind == "gqa":
            foldable = count_gqa_foldable(layer_nodes)
            conservative = 0
            aggressive = foldable
        elif kind == "final-norm":
            foldable = 0
            conservative = 0
            aggressive = 0
        else:
            foldable = 0
            # One fused GQA replaces the decomposed core.
            conservative = max(core - norm - 1, 0)
            aggressive = max(core - 1, 0)

        saved_conservative += conservative
        saved_aggressive += aggressive
        totals[kind] += 1

        rows.append(
            {
                "index": index,
                "kind": kind,
                "layer_type": layer_types[index] if index < len(layer_types) else None,
                "nodes": len(layer_nodes),
                "core_attn_nodes": core,
                "norm_nodes_kept": norm,
                "gqa_foldable": foldable,
                "saved_conservative": conservative,
                "saved_aggressive": aggressive,
            }
        )

    past_inputs = sorted(
        {
            int(match.group(1))
            for value in model.graph.input
            if (match := PAST_RE.match(value.name))
        }
    )

    shape_only = sum(count for op, count in op_counts.items() if op in SHAPE_ONLY_OPS)

    total_nodes = len(nodes)
    return {
        "model": {
            "path": str(model_path),
            "size_bytes": model_path.stat().st_size,
            "sha256": sha256_file(model_path),
            "opset": [[op.domain or "", op.version] for op in model.opset_import],
        },
        "config": config,
        "graph": {
            "total_nodes": total_nodes,
            "shape_only_nodes": shape_only,
            "dispatch_candidate_nodes": total_nodes - shape_only,
            "op_counts": dict(op_counts.most_common()),
            "past_key_value_indices": past_inputs,
            "past_key_value_count": len(past_inputs),
            "graph_inputs": [value.name for value in model.graph.input],
        },
        "layers": rows,
        "totals": {
            "layers": len(rows),
            "gqa_layers": totals["gqa"],
            "manual_layers": totals["manual"],
            "shared_kv_layers": totals["shared-kv"],
            "other_layers": totals["other"],
            "final_norm_layers": totals["final-norm"],
        },
        "fusion": {
            "conservative_saved": saved_conservative,
            "conservative_pct_of_all_nodes": round(100 * saved_conservative / total_nodes, 2),
            "aggressive_saved": saved_aggressive,
            "aggressive_pct_of_all_nodes": round(100 * saved_aggressive / total_nodes, 2),
            "notes": [
                "conservative: keep q_norm/k_norm/rotary outside GQA",
                "aggressive: fold q/k norm and rotary into GQA; runtime support must be verified",
                "shared-kv layers need external K/V; mapping to GQA is unproven (Gate 2)",
            ],
        },
    }


def print_report(report: dict) -> None:
    graph = report["graph"]
    totals = report["totals"]
    fusion = report["fusion"]

    print(f"model      : {report['model']['path']}")
    print(f"sha256     : {report['model']['sha256'][:16]}…")
    print(f"opset      : {report['model']['opset']}")
    print(f"nodes      : {graph['total_nodes']} total, {graph['shape_only_nodes']} shape-only")
    print(f"KV caches  : {graph['past_key_value_count']} past_key_values inputs")
    print()

    print("layers")
    print(f"  gqa (already fused) : {totals['gqa_layers']}")
    print(f"  manual              : {totals['manual_layers']}")
    print(f"  shared-kv           : {totals['shared_kv_layers']}")
    print(f"  other               : {totals['other_layers']}")
    print(f"  final norm (non-block): {totals['final_norm_layers']}")
    print()

    print("per-layer attention (non-gqa only)")
    header = f"  {'layer':>5}  {'kind':<10} {'type':<17} {'nodes':>5} {'core':>5} {'keep':>5} {'save':>5}"
    print(header)
    for row in report["layers"]:
        if row["kind"] in {"gqa", "final-norm"}:
            continue
        print(
            f"  {row['index']:>5}  {row['kind']:<10} {str(row['layer_type']):<17}"
            f" {row['nodes']:>5} {row['core_attn_nodes']:>5}"
            f" {row['norm_nodes_kept']:>5} {row['saved_conservative']:>5}"
        )
    print()

    print("top ops")
    for op, count in list(graph["op_counts"].items())[:12]:
        print(f"  {count:>5}  {op}")
    print()

    print("fusion estimate (nodes per token)")
    print(
        f"  conservative : -{fusion['conservative_saved']}"
        f"  ({fusion['conservative_pct_of_all_nodes']}% of all nodes)"
    )
    print(
        f"  aggressive   : -{fusion['aggressive_saved']}"
        f"  ({fusion['aggressive_pct_of_all_nodes']}% of all nodes)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="decoder ONNX file")
    parser.add_argument("--config", type=Path, help="config.json (optional)")
    parser.add_argument("--json", type=Path, help="write the full census as JSON")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="only write JSON, do not print the human-readable report",
    )
    args = parser.parse_args()

    if not args.model.exists():
        print(f"model not found: {args.model}", file=sys.stderr)
        return 2

    report = census(args.model, args.config)
    if not args.quiet:
        print_report(report)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
