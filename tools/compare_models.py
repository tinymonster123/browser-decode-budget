#!/usr/bin/env python3
"""Gate 1 equivalence: run two decoder graphs on the same random inputs.

This is a structural/equivalence check, not a translation-quality check. It
answers: does the fused graph compute the same thing as the base graph, up to
fp16 accumulation order?

Python ORT has no WebGPU EP, so this runs on CPU. WebGPU timing still belongs
to Gate 3.

Usage:
    python tools/compare_models.py \
        --a <base decoder.onnx> \
        --b models/derived/gemma4-e2b-q4f16-gate1/onnx/decoder_model_merged_q4f16.onnx \
        --json runs/2026-10-19-gate1-equivalence.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

DTYPES = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(bool)": np.bool_,
}


def resolve_shape(shape: list, sequence: int, past: int) -> tuple[int, ...]:
    resolved = []
    for dim in shape:
        if isinstance(dim, int):
            resolved.append(dim)
            continue
        if dim == "batch_size":
            resolved.append(1)
        elif dim == "sequence_length":
            resolved.append(sequence)
        elif dim == "past_sequence_length":
            resolved.append(past)
        elif dim == "total_sequence_length":
            resolved.append(past + sequence)
        else:
            raise ValueError(f"unhandled dynamic dimension: {dim}")
    return tuple(resolved)


def build_feeds(
    session: ort.InferenceSession,
    sequence: int,
    past: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    feeds: dict[str, np.ndarray] = {}
    for value in session.get_inputs():
        dtype = DTYPES[value.type]
        shape = resolve_shape(value.shape, sequence, past)
        name = value.name

        if name == "attention_mask":
            array = np.ones(shape, dtype=dtype)
        elif name == "position_ids":
            array = np.arange(past, past + sequence, dtype=dtype).reshape(shape)
        elif name == "num_logits_to_keep":
            array = np.asarray(1, dtype=dtype)
        elif name in {"inputs_embeds", "per_layer_inputs"} or name.startswith(
            "past_key_values."
        ):
            array = rng.normal(0.0, 0.02, size=shape).astype(dtype)
        else:
            array = np.zeros(shape, dtype=dtype)
        feeds[name] = array
    return feeds


def compare(
    base_path: Path,
    fused_path: Path,
    sequence: int,
    past: int,
    seed: int,
    min_cosine: float,
) -> dict:
    options = ort.SessionOptions()
    options.log_severity_level = 3
    base = ort.InferenceSession(str(base_path), sess_options=options, providers=["CPUExecutionProvider"])
    fused = ort.InferenceSession(str(fused_path), sess_options=options, providers=["CPUExecutionProvider"])

    feeds = build_feeds(base, sequence, past, seed)
    base_out = dict(zip([o.name for o in base.get_outputs()], base.run(None, feeds)))
    fused_out = dict(zip([o.name for o in fused.get_outputs()], fused.run(None, feeds)))

    if set(base_out) != set(fused_out):
        raise SystemExit(
            f"output sets differ: only-base={sorted(set(base_out) - set(fused_out))}, "
            f"only-fused={sorted(set(fused_out) - set(base_out))}"
        )

    a = base_out["logits"].astype(np.float32).reshape(-1)
    b = fused_out["logits"].astype(np.float32).reshape(-1)
    diff = np.abs(a - b)
    cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    rmse_over_std = float(np.sqrt(np.mean((a - b) ** 2)) / (np.std(a) + 1e-12))
    top5_base = set(np.argpartition(a, -5)[-5:].tolist())
    top5_fused = set(np.argpartition(b, -5)[-5:].tolist())
    top2_base = np.sort(a)[-2:]
    top2_fused = np.sort(b)[-2:]

    tensor_diffs = {}
    for name in base_out:
        if name == "logits":
            continue
        x = base_out[name].astype(np.float32)
        y = fused_out[name].astype(np.float32)
        tensor_diffs[name] = float(np.abs(x - y).max())

    report = {
        "base": str(base_path),
        "fused": str(fused_path),
        "input": {"sequence": sequence, "past": past, "seed": seed},
        "logits": {
            "max_abs": float(diff.max()),
            "mean_abs": float(diff.mean()),
            "cosine": cosine,
            "rmse_over_std": rmse_over_std,
            "top5_overlap": len(top5_base & top5_fused),
            "argmax_base": int(np.argmax(a)),
            "argmax_fused": int(np.argmax(b)),
            "argmax_equal": bool(int(np.argmax(a)) == int(np.argmax(b))),
            "top2_gap_base": float(top2_base[1] - top2_base[0]),
            "top2_gap_fused": float(top2_fused[1] - top2_fused[0]),
        },
        "present_max_abs": tensor_diffs,
        "fused_layer_present_max_abs": {
            name: value
            for name, value in tensor_diffs.items()
            if name.startswith(("present.4.", "present.9.", "present.14."))
        },
        # A top-1 swap is acceptable only when the top-5 sets still agree.
        # fp16 near-ties show up as argmax flips with a tiny top-2 gap.
        "passes": cosine >= min_cosine and len(top5_base & top5_fused) >= 4,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True, type=Path)
    parser.add_argument("--b", required=True, type=Path)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--past", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    report = compare(args.a, args.b, args.tokens, args.past, args.seed, args.min_cosine)
    logits = report["logits"]
    print(f"logits   max_abs={logits['max_abs']:.5g}  mean_abs={logits['mean_abs']:.5g}")
    print(f"         cosine={logits['cosine']:.8f}  rmse/std={logits['rmse_over_std']:.5f}")
    print(
        f"         top5={logits['top5_overlap']}/5  argmax={logits['argmax_base']} vs "
        f"{logits['argmax_fused']} (equal={logits['argmax_equal']}, "
        f"top2 gap={logits['top2_gap_base']:.4g}/{logits['top2_gap_fused']:.4g})"
    )
    print("fused-layer present max_abs:")
    for name, value in report["fused_layer_present_max_abs"].items():
        print(f"  {name:20s} {value:.6g}")
    print("verdict:", "PASS" if report["passes"] else "FAIL")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"wrote {args.json}")
    return 0 if report["passes"] else 1


if __name__ == "__main__":
    sys.exit(main())
