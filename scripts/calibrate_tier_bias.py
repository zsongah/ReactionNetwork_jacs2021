#!/usr/bin/env python3
"""
Diagnose tier-bias settings by comparing the cost of FlowER-endorsed
reactions vs the cost they would have under fragrec-only routing.

For each v2 cache entry (reactants known), each FlowER prediction is
treated as a candidate reaction with three observable quantities:

  * P_FlowER  — count_i / sum_counts (per-entry normalized)
  * tier      — FLOWER_HIGH (since FlowER produced it)
  * proxy ΔG  — a constant baseline, since we don't recompute thermo
                here. The bias's effect is what we want to isolate.

For each (FLOWER_HIGH bias, FRAGREC_ORGANIC bias) grid point, we
compute:

  cost_endorsed   = softplus(ΔG) − λ·log(P_FlowER)         + bias_HIGH
  cost_unendorsed = softplus(ΔG) − λ·log(p_floor)          + bias_ORG

The question: under each setting, what fraction of FlowER-endorsed
reactions beat the unendorsed baseline (lower cost = preferred)?
A healthy calibration gives FLOWER_HIGH steps a clear majority.

Output: a markdown table the user reads to choose values. No fake
ground truth is fabricated; the script just makes the bias's
mechanical effect visible.

Usage:
    python -m scripts.calibrate_tier_bias
    python -m scripts.calibrate_tier_bias --lam 0.5 --p-floor 0.01
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.thermo import softplus  # noqa: E402

DEFAULT_CACHE = REPO_ROOT / "cache" / "flower"


def _load_v2_entries(cache_dir: Path) -> list[dict]:
    out: list[dict] = []
    for path in sorted(cache_dir.glob("*.json")):
        try:
            with path.open() as f:
                raw = json.load(f)
        except json.JSONDecodeError:
            continue
        if isinstance(raw, dict) and "products" in raw:
            out.append(raw)
    return out


def _flatten_priors(entries: list[dict]) -> list[float]:
    """Per-product P_FlowER values, one per (entry, product) pair."""
    probs: list[float] = []
    for e in entries:
        prods = e.get("products", [])
        counts = [int(c) for _, c in prods]
        total = sum(counts)
        if total == 0:
            continue
        for _, c in prods:
            probs.append(c / total)
    return probs


def calibrate(
    priors: list[float],
    lam: float,
    p_floor: float,
    delta_g: float,
    softplus_scale: float,
    bias_high_grid: list[float],
    bias_org_grid: list[float],
) -> list[dict]:
    """Grid-evaluate (bias_high, bias_org) pairs against the prior distribution.

    Returns a list of dicts, one per grid cell.
    """
    sp = softplus(delta_g, softplus_scale)
    cost_unendorsed_floor = sp - lam * math.log(p_floor)
    rows: list[dict] = []
    for bh in bias_high_grid:
        for bo in bias_org_grid:
            wins = 0
            margin_sum = 0.0
            for p in priors:
                cost_high = sp - lam * math.log(max(p, p_floor)) + bh
                cost_org = cost_unendorsed_floor + bo
                margin = cost_org - cost_high  # positive = FLOWER_HIGH wins
                margin_sum += margin
                if margin > 0:
                    wins += 1
            n = len(priors)
            rows.append({
                "bias_high": bh,
                "bias_org": bo,
                "win_rate": wins / n if n else 0.0,
                "mean_margin": margin_sum / n if n else 0.0,
                "n": n,
            })
    return rows


def _format_table(rows: list[dict]) -> str:
    lines = [
        "| bias_HIGH | bias_ORG | win-rate | mean cost margin |",
        "|----------:|---------:|---------:|-----------------:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['bias_high']:+.2f} | {r['bias_org']:+.2f} | "
            f"{r['win_rate']*100:.1f}% | {r['mean_margin']:+.3f} |"
        )
    return "\n".join(lines)


def _recommend(rows: list[dict]) -> dict | None:
    """Pick the smallest-magnitude bias pair with win_rate >= 0.95."""
    eligible = [r for r in rows if r["win_rate"] >= 0.95]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda r: abs(r["bias_high"]) + abs(r["bias_org"]),
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--p-floor", type=float, default=0.01)
    p.add_argument("--delta-g", type=float, default=0.0,
                   help="Constant proxy ΔG; bias effect is independent of "
                        "this value when both tiers share it.")
    p.add_argument("--softplus-scale", type=float, default=0.5)
    p.add_argument("--bias-high-min", type=float, default=-1.0)
    p.add_argument("--bias-high-max", type=float, default=0.0)
    p.add_argument("--bias-high-step", type=float, default=0.25)
    p.add_argument("--bias-org-min", type=float, default=0.0)
    p.add_argument("--bias-org-max", type=float, default=0.6)
    p.add_argument("--bias-org-step", type=float, default=0.15)
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args()

    if not args.cache_dir.is_dir():
        print(f"error: cache dir not found: {args.cache_dir}", file=sys.stderr)
        return 1

    entries = _load_v2_entries(args.cache_dir)
    priors = _flatten_priors(entries)
    if not priors:
        print(
            "error: no v2 cache entries with priors found. Run "
            "scripts/backfill_cache.py first to upgrade legacy entries.",
            file=sys.stderr,
        )
        return 2

    def _grid(lo: float, hi: float, step: float) -> list[float]:
        n = int(round((hi - lo) / step))
        return [round(lo + i * step, 4) for i in range(n + 1)]

    bias_high_grid = _grid(args.bias_high_min, args.bias_high_max, args.bias_high_step)
    bias_org_grid = _grid(args.bias_org_min, args.bias_org_max, args.bias_org_step)

    rows = calibrate(
        priors,
        lam=args.lam,
        p_floor=args.p_floor,
        delta_g=args.delta_g,
        softplus_scale=args.softplus_scale,
        bias_high_grid=bias_high_grid,
        bias_org_grid=bias_org_grid,
    )

    print(f"Loaded {len(entries)} v2 entries, {len(priors)} (entry, product) priors")
    print(f"P_FlowER stats: min={min(priors):.3f} median={sorted(priors)[len(priors)//2]:.3f} max={max(priors):.3f}")
    print(f"Grid: {len(bias_high_grid)} × {len(bias_org_grid)} = {len(rows)} cells")
    print(f"λ={args.lam}, p_floor={args.p_floor}, ΔG={args.delta_g}, softplus_scale={args.softplus_scale}")
    print()
    print(_format_table(rows))
    print()

    rec = _recommend(rows)
    if rec is None:
        print("No grid cell achieves >=95% win-rate; consider widening the search.")
    else:
        print(
            f"Recommendation (smallest |bias|, win-rate >= 95%): "
            f"FLOWER_HIGH = {rec['bias_high']:+.2f}, "
            f"FRAGREC_ORGANIC = {rec['bias_org']:+.2f}, "
            f"win-rate = {rec['win_rate']*100:.1f}%"
        )

    if args.json:
        args.json.write_text(json.dumps({
            "config": {
                "lam": args.lam, "p_floor": args.p_floor,
                "delta_g": args.delta_g, "softplus_scale": args.softplus_scale,
            },
            "rows": rows,
            "recommendation": rec,
        }, indent=2))
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
