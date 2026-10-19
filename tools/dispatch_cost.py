#!/usr/bin/env python3
"""Turn a Gate 0 census into a projection.

The model is intentionally boring:

    T_token ≈ N_nodes × C_node

C_node comes from a *measured* baseline, not from a benchmark of this script.
The projection is only as good as the assumption that removing nodes removes
time linearly. Gate 1/Gate 3 exist to falsify that assumption.

Usage:
    python tools/dispatch_cost.py --census runs/2026-10-19-gate0-census.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SENSITIVITY_US = (15, 20, 25, 30, 40, 50)


def load_census(path: Path) -> dict:
    return json.loads(path.read_text())


def project(
    total_nodes: int,
    saved_nodes: int,
    decode_seconds: float,
    prefill_seconds: float,
    fixed_seconds: float,
) -> dict:
    ratio = (total_nodes - saved_nodes) / total_nodes
    new_decode = decode_seconds * ratio
    new_total = new_decode + prefill_seconds + fixed_seconds
    baseline_total = decode_seconds + prefill_seconds + fixed_seconds
    return {
        "saved_nodes": saved_nodes,
        "new_nodes": total_nodes - saved_nodes,
        "node_ratio": ratio,
        "new_decode_s": new_decode,
        "new_total_s": new_total,
        "baseline_total_s": baseline_total,
        "decode_gain_pct": 100 * (1 - ratio),
        "total_gain_pct": 100 * (1 - new_total / baseline_total),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census", required=True, type=Path)
    parser.add_argument("--ms-per-token", type=float, help="measured decode ms/token (preferred)")
    parser.add_argument("--decode-s", type=float, default=8.5, help="measured decode seconds")
    parser.add_argument("--decode-tokens", type=int, default=305)
    parser.add_argument("--prefill-s", type=float, default=2.6)
    parser.add_argument("--fixed-s", type=float, default=0.2)
    parser.add_argument(
        "--min-removable-pct",
        type=float,
        default=10.0,
        help="kill criterion: stop if the conservative saving is below this",
    )
    args = parser.parse_args()

    census = load_census(args.census)
    total_nodes = census["graph"]["total_nodes"]
    conservative = census["fusion"]["conservative_saved"]
    aggressive = census["fusion"]["aggressive_saved"]

    if args.ms_per_token is not None:
        per_token_ms = args.ms_per_token
        decode_seconds = per_token_ms * args.decode_tokens / 1000
    else:
        per_token_ms = args.decode_s * 1000 / args.decode_tokens
        decode_seconds = args.decode_s

    per_node_us = per_token_ms * 1000 / total_nodes

    print(f"census          : {args.census}")
    print(f"nodes           : {total_nodes}")
    print(f"baseline decode : {per_token_ms:.2f} ms/token × {args.decode_tokens} = {decode_seconds:.2f}s")
    print(f"baseline total  : {decode_seconds + args.prefill_s + args.fixed_s:.2f}s"
          f"  (decode {decode_seconds:.2f} + prefill {args.prefill_s:.2f} + fixed {args.fixed_s:.2f})")
    print(f"implied C_node  : {per_node_us:.1f} µs/dispatch")
    print()

    header = (
        f"{'scenario':<14} {'saved':>6} {'new':>6} {'ratio':>7}"
        f" {'new decode':>11} {'new total':>10} {'decode Δ':>9} {'total Δ':>8}"
    )
    print(header)
    scenarios = {
        "conservative": conservative,
        "aggressive": aggressive,
        f"break-even {args.min_removable_pct:.0f}%": int(total_nodes * args.min_removable_pct / 100),
    }
    for name, saved in scenarios.items():
        p = project(total_nodes, saved, decode_seconds, args.prefill_s, args.fixed_s)
        print(
            f"{name:<14} {saved:>6} {p['new_nodes']:>6} {p['node_ratio']:>7.3f}"
            f" {p['new_decode_s']:>10.2f}s {p['new_total_s']:>9.2f}s"
            f" {p['decode_gain_pct']:>8.1f}% {p['total_gain_pct']:>7.1f}%"
        )
    print()

    print("sensitivity: time saved per token at different dispatch costs")
    print(f"  {'C_node':>8} {'conservative':>14} {'aggressive':>12}")
    for cost_us in SENSITIVITY_US:
        cons_ms = conservative * cost_us / 1000
        aggr_ms = aggressive * cost_us / 1000
        print(f"  {cost_us:>6}µs {cons_ms:>11.2f} ms {aggr_ms:>9.2f} ms")
    print()

    conservative_pct = 100 * conservative / total_nodes
    print(f"kill criterion  : conservative saving {conservative_pct:.1f}%"
          f" vs threshold {args.min_removable_pct:.1f}%")
    if conservative_pct < args.min_removable_pct:
        print("verdict         : STOP — not enough removable dispatches to justify fusion")
        return 1
    print("verdict         : PROCEED — enough removable dispatches for the next fusion gate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
