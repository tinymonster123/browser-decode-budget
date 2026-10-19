#!/usr/bin/env python3
"""Gate 2: fuse the shared-KV attention blocks into GroupQueryAttention.

The 20 shared layers do not own a KV cache. They reuse another layer's cache:

    15-18, 20-23, 25-28, 30-33  ->  present.13.key / present.13.value  (sliding)
    19, 24, 29, 34              ->  present.14.key / present.14.value  (full)

GQA expects `past_key` (BNSH) plus optional new `key`/`value` (BSH). The shared
cache already contains the current tokens, so we split it once per donor:

    past = cache[:, :, :T-S, :]
    new  = cache[:, :, T-S:, :]  -> reshape to (B, S, D)

The split is built once and shared by every layer in the group. Each fused
layer then adds a single GQA node.

Usage:
    python tools/fuse_shared_kv.py \
        --model models/derived/gemma4-e2b-q4f16-gate1/onnx/decoder_model_merged_q4f16.onnx \
        --out-dir models/derived/gemma4-e2b-q4f16-gate2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import onnx
from onnx import TensorProto, helper

from fuse_manual_attention import (
    ATTENTION_BIAS,
    NUM_HEADS,
    SEQ_LENS,
    SCALE,
    TOTAL_SEQ,
    FusionError,
    _clone_or_copy,
    _single_output,
    toposort,
    validate_graph,
)

KV_NUM_HEADS = 1
INT_MAX = 2**31 - 1


@dataclass(frozen=True)
class Group:
    name: str
    layers: tuple[int, ...]
    cache_key: str
    cache_value: str
    head_dim: int
    window: int


GROUPS = (
    Group(
        name="13",
        layers=(15, 16, 17, 18, 20, 21, 22, 23, 25, 26, 27, 28, 30, 31, 32, 33),
        cache_key="present.13.key",
        cache_value="present.13.value",
        head_dim=256,
        window=512,
    ),
    Group(
        name="14",
        layers=(19, 24, 29, 34),
        cache_key="present.14.key",
        cache_value="present.14.value",
        head_dim=512,
        window=-1,
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _add_initializer(model: onnx.ModelProto, name: str, values: list[int]) -> str:
    tensor = helper.make_tensor(name, TensorProto.INT64, [len(values)], values)
    model.graph.initializer.append(tensor)
    return name


def _build_split(
    model: onnx.ModelProto,
    group: Group,
    q_name: str,
) -> dict[str, str]:
    """Add the shared slice/reshape subgraph and return its value names."""
    prefix = f"/model/shared_kv_{group.name}"
    axes = _add_initializer(model, prefix + "/axes", [2])
    steps = _add_initializer(model, prefix + "/steps", [1])
    zero = _add_initializer(model, prefix + "/zero", [0])
    int_max = _add_initializer(model, prefix + "/int_max", [INT_MAX])
    index_one = _add_initializer(model, prefix + "/index_one", [1])
    reshape = _add_initializer(
        model, prefix + "/reshape_bsh", [0, -1, group.head_dim]
    )

    values = {
        "q_shape": prefix + "/q_shape",
        "seq_len": prefix + "/seq_len",
        "neg_seq_len": prefix + "/neg_seq_len",
        "new_k_4d": prefix + "/new_k_4d",
        "new_v_4d": prefix + "/new_v_4d",
        "past_k": prefix + "/past_k",
        "past_v": prefix + "/past_v",
        "new_k": prefix + "/new_k",
        "new_v": prefix + "/new_v",
    }

    nodes = [
        helper.make_node("Shape", [q_name], [values["q_shape"]], name=prefix + "/Shape"),
        helper.make_node(
            "Gather",
            [values["q_shape"], index_one],
            [values["seq_len"]],
            name=prefix + "/GatherSeqLen",
            axis=0,
        ),
        helper.make_node(
            "Neg", [values["seq_len"]], [values["neg_seq_len"]], name=prefix + "/NegSeqLen"
        ),
        # new = cache[:, :, -S:, :]
        helper.make_node(
            "Slice",
            [group.cache_key, values["neg_seq_len"], int_max, axes, steps],
            [values["new_k_4d"]],
            name=prefix + "/SliceNewK",
        ),
        helper.make_node(
            "Slice",
            [group.cache_value, values["neg_seq_len"], int_max, axes, steps],
            [values["new_v_4d"]],
            name=prefix + "/SliceNewV",
        ),
        # past = cache[:, :, :-S, :]
        helper.make_node(
            "Slice",
            [group.cache_key, zero, values["neg_seq_len"], axes, steps],
            [values["past_k"]],
            name=prefix + "/SlicePastK",
        ),
        helper.make_node(
            "Slice",
            [group.cache_value, zero, values["neg_seq_len"], axes, steps],
            [values["past_v"]],
            name=prefix + "/SlicePastV",
        ),
        helper.make_node(
            "Reshape",
            [values["new_k_4d"], reshape],
            [values["new_k"]],
            name=prefix + "/ReshapeNewK",
        ),
        helper.make_node(
            "Reshape",
            [values["new_v_4d"], reshape],
            [values["new_v"]],
            name=prefix + "/ReshapeNewV",
        ),
    ]
    model.graph.node.extend(nodes)
    return values


def fuse_group(model: onnx.ModelProto, group: Group) -> dict:
    by_name = {node.name: node for node in model.graph.node}
    first_prefix = f"/model/layers.{group.layers[0]}/attn/"
    q_rotary = by_name.get(first_prefix + "q_rotary/RotaryEmbedding")
    if q_rotary is None:
        raise FusionError(f"group {group.name}: missing q_rotary on layer {group.layers[0]}")
    q_name = _single_output(q_rotary)

    values = _build_split(model, group, q_name)
    removed_total = 0

    for layer in group.layers:
        prefix = f"/model/layers.{layer}/attn/"
        q_rotary_layer = by_name.get(prefix + "q_rotary/RotaryEmbedding")
        o_proj = by_name.get(prefix + "o_proj/MatMul_Quant")
        output_reshape = by_name.get(prefix + "shared/output/Reshape")
        if q_rotary_layer is None or o_proj is None or output_reshape is None:
            raise FusionError(f"layer {layer}: expected shared attention pattern not found")

        # Keep q_norm; remove the decomposed attention core.
        removed = [
            node
            for node in model.graph.node
            if prefix + "shared/" in node.name
            and prefix + "shared/q_norm/" not in node.name
        ]
        removed_names = {node.name for node in removed}
        if not removed:
            raise FusionError(f"layer {layer}: no shared attention nodes found")
        removed_total += len(removed)

        old_output = _single_output(output_reshape)
        gqa_output = prefix + "GroupQueryAttention/output_0"
        gqa = helper.make_node(
            "GroupQueryAttention",
            inputs=[
                _single_output(q_rotary_layer),
                values["new_k"],
                values["new_v"],
                values["past_k"],
                values["past_v"],
                SEQ_LENS,
                TOTAL_SEQ,
                "",
                "",
                "position_ids",
                ATTENTION_BIAS,
            ],
            # The CPU/WebGPU kernel only reads past_key/past_value when the
            # optional present outputs are declared on the node. They stay
            # unused here; declaring them is what makes the cache path work.
            outputs=[
                gqa_output,
                prefix + "GroupQueryAttention/present_key",
                prefix + "GroupQueryAttention/present_value",
            ],
            name=prefix + "GroupQueryAttention",
            domain="com.microsoft",
            num_heads=NUM_HEADS,
            kv_num_heads=KV_NUM_HEADS,
            scale=SCALE,
            local_window_size=group.window,
            softcap=0.0,
            do_rotary=0,
            rotary_interleaved=0,
            causal=1,
        )

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

        kept = [node for node in model.graph.node if node.name not in removed_names]
        kept.append(gqa)
        model.graph.ClearField("node")
        model.graph.node.extend(kept)

    return {
        "group": group.name,
        "layers": list(group.layers),
        "removed": removed_total,
        "gqa_nodes": len(group.layers),
        "shared_split_nodes": 9,
        "net_saved": removed_total - len(group.layers) - 9,
    }


def drop_unused_layer14_transpose(model: onnx.ModelProto) -> int:
    """The 4 full shared layers no longer need layer 14's pre-transposed key."""
    target = "/model/layers.14/attn/fused_k_transpose/Transpose"
    node = next((item for item in model.graph.node if item.name == target), None)
    if node is None:
        return 0
    value = node.output[0]
    consumed = any(
        value in other.input for other in model.graph.node if other.name != target
    )
    if consumed:
        return 0
    model.graph.node.remove(node)
    return 1


def fuse_model(model: onnx.ModelProto, layers: list[int]) -> dict:
    requested = set(layers)
    groups = [
        Group(
            name=group.name,
            layers=tuple(layer for layer in group.layers if layer in requested),
            cache_key=group.cache_key,
            cache_value=group.cache_value,
            head_dim=group.head_dim,
            window=group.window,
        )
        for group in GROUPS
        if any(layer in requested for layer in group.layers)
    ]
    unknown = requested - {layer for group in GROUPS for layer in group.layers}
    if unknown:
        raise FusionError(f"not shared-KV layers: {sorted(unknown)}")

    reports = [fuse_group(model, group) for group in groups]
    dropped = drop_unused_layer14_transpose(model)
    ordered = toposort(list(model.graph.node))
    model.graph.ClearField("node")
    model.graph.node.extend(ordered)
    validate_graph(model)
    return {
        "groups": reports,
        "dropped_unused_transpose": dropped,
        "net_saved": sum(item["net_saved"] for item in reports) + dropped,
    }


def materialize(
    model: onnx.ModelProto,
    model_path: Path,
    out_dir: Path,
    force: bool,
    report: dict,
) -> Path:
    if out_dir.exists() and any(out_dir.iterdir()):
        if not force:
            raise FusionError(f"{out_dir} is not empty; pass --force to replace")
        shutil.rmtree(out_dir)

    onnx_dir = out_dir / "onnx"
    onnx_dir.mkdir(parents=True)

    base_root = model_path.parent.parent
    for item in base_root.iterdir():
        if item.name == "onnx":
            continue
        _clone_or_copy(item, out_dir / item.name)
    for item in model_path.parent.iterdir():
        if item.name == model_path.name:
            continue
        _clone_or_copy(item, onnx_dir / item.name)

    out_model = onnx_dir / model_path.name
    onnx.save(model, out_model)

    manifest = {
        "id": out_dir.name,
        "base": {
            "path": str(model_path),
            "decoder_sha256": sha256_file(model_path),
        },
        "producer": {
            "kind": "surgery",
            "tool": "tools/fuse_shared_kv.py",
            "strategy": "split shared cache into past + new key/value",
        },
        "dtype": "q4f16",
        "opset": [[op.domain or "", op.version] for op in model.opset_import],
        "files": {f"onnx/{model_path.name}": sha256_file(out_model)},
        "fusion": report,
        "runtime": {"browser_equivalence": "pending"},
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )
    return out_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument(
        "--layers",
        default=",".join(str(layer) for group in GROUPS for layer in group.layers),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("models/derived/gemma4-e2b-q4f16-gate2"),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    layers = [int(value) for value in args.layers.split(",") if value.strip()]
    model = onnx.load(str(args.model), load_external_data=False)
    report = fuse_model(model, layers)
    out_model = materialize(model, args.model, args.out_dir, args.force, report)

    print(f"wrote {out_model}")
    print(f"wrote {args.out_dir / 'manifest.json'}")
    for item in report["groups"]:
        print(
            f"  group {item['group']}: {len(item['layers'])} layers, "
            f"removed {item['removed']}, added {item['gqa_nodes'] + item['shared_split_nodes']}, "
            f"net saved {item['net_saved']}"
        )
    print(f"  dropped unused layer-14 transpose: {report['dropped_unused_transpose']}")
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
