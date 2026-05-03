#!/usr/bin/env python3
"""
Analyze the FlowER prior cache and report:

  * Total entries (and how many carry reactant SMILES vs are legacy
    bare-list files).
  * Probability distribution of FlowER predictions across all entries.
  * Per-reactant-size and per-element histograms (only over entries
    whose reactants are recoverable, i.e. the new dict schema).
  * Top-N most "confident" reactions (highest single-product P).
  * Reactions where FlowER returned the reactant unchanged
    (P_self >= 0.95) — these are FlowER saying "no productive
    reaction", which is itself useful prior information.

Usage:
    python -m scripts.analyze_cache               # uses cache/flower
    python -m scripts.analyze_cache --top 20
    python -m scripts.analyze_cache --json out.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = REPO_ROOT / "cache" / "flower"


def _normalize_entry(raw: Any) -> tuple[str | None, list[tuple[str, int]]]:
    """Return (reactants_or_None, products) for either schema."""
    if isinstance(raw, dict):
        return raw.get("reactants"), list(raw.get("products", []))
    if isinstance(raw, list):
        return None, list(raw)
    return None, []


def _canonicalize_smiles(smi: str) -> str:
    return ".".join(sorted(smi.split(".")))


def _count_elements(smi: str) -> Counter:
    """Cheap element count from atom-mapped SMILES (uppercase letters)."""
    elems: Counter = Counter()
    i = 0
    while i < len(smi):
        c = smi[i]
        if c.isupper():
            j = i + 1
            if j < len(smi) and smi[j].islower() and smi[j] not in "h":
                elems[smi[i:j+1]] += 1
                i = j + 1
                continue
            elems[c] += 1
        i += 1
    return elems


def _reactant_size(reactants_smi: str) -> int:
    return len(reactants_smi.split("."))


def analyze(cache_dir: Path, top_n: int = 10) -> dict[str, Any]:
    files = sorted(cache_dir.glob("*.json"))
    n_total = len(files)
    n_v2 = 0  # entries with recoverable reactants
    n_legacy = 0
    n_empty = 0  # FlowER returned no products at all

    all_probs: list[float] = []
    self_reactions = 0  # P(reactant unchanged) >= 0.95
    by_size: Counter = Counter()
    by_size_with_products: Counter = Counter()
    by_element: Counter = Counter()

    top_predictions: list[tuple[float, str, str]] = []  # (P, reactants, product)

    for path in files:
        try:
            with path.open() as f:
                raw = json.load(f)
        except json.JSONDecodeError:
            print(f"warn: skipping malformed cache file {path.name}", file=sys.stderr)
            continue

        reactants, products = _normalize_entry(raw)
        if reactants is None:
            n_legacy += 1
        else:
            n_v2 += 1
            r_canon = _canonicalize_smiles(reactants)
            size = _reactant_size(r_canon)
            by_size[size] += 1
            if products:
                by_size_with_products[size] += 1
            for elem, cnt in _count_elements(r_canon).items():
                by_element[elem] += cnt

        if not products:
            n_empty += 1
            continue

        # Probability = count / sample_size (here sample_size=64 by
        # default; we don't have it per-file in legacy entries, but the
        # ratio is preserved). Use raw counts as the prior weight; for
        # a probability-like number, normalize by the maximum count
        # observed in this entry.
        counts = [int(c) for _, c in products]
        total = sum(counts)
        if total == 0:
            continue
        for prod_smi, c in products:
            p = c / total
            all_probs.append(p)
            top_predictions.append((p, reactants or "(legacy)", prod_smi))
            if reactants is not None and _canonicalize_smiles(prod_smi) == _canonicalize_smiles(reactants) and p >= 0.95:
                self_reactions += 1

    top_predictions.sort(reverse=True, key=lambda t: t[0])

    return {
        "cache_dir": str(cache_dir),
        "n_total": n_total,
        "n_v2_with_reactants": n_v2,
        "n_legacy": n_legacy,
        "n_empty_predictions": n_empty,
        "n_self_reactions": self_reactions,
        "probability_stats": _prob_stats(all_probs),
        "by_reactant_size": dict(sorted(by_size.items())),
        "by_reactant_size_productive": dict(sorted(by_size_with_products.items())),
        "by_element_atom_count": dict(by_element.most_common()),
        "top_predictions": [
            {"p": round(p, 4), "reactants": r, "product": pr}
            for p, r, pr in top_predictions[:top_n]
        ],
    }


def _prob_stats(probs: list[float]) -> dict[str, float | int]:
    if not probs:
        return {"n": 0}
    return {
        "n": len(probs),
        "mean": round(statistics.fmean(probs), 4),
        "median": round(statistics.median(probs), 4),
        "stdev": round(statistics.stdev(probs), 4) if len(probs) > 1 else 0.0,
        "min": round(min(probs), 4),
        "max": round(max(probs), 4),
    }


def _print_report(report: dict[str, Any]) -> None:
    print(f"FlowER cache report — {report['cache_dir']}")
    print("=" * 60)
    print(f"Total entries:           {report['n_total']}")
    print(f"  v2 (with reactants):   {report['n_v2_with_reactants']}")
    print(f"  legacy (bare list):    {report['n_legacy']}")
    print(f"Empty-prediction entries:{report['n_empty_predictions']}")
    print(f"Self-reaction (P>=0.95): {report['n_self_reactions']}")
    print()
    print("Probability stats over all (product, count) pairs:")
    for k, v in report["probability_stats"].items():
        print(f"  {k:8s} {v}")
    print()
    if report["by_reactant_size"]:
        print("Entries by reactant size (v2 only):")
        for size, n in report["by_reactant_size"].items():
            prod = report["by_reactant_size_productive"].get(size, 0)
            print(f"  size={size}: {n}  (productive: {prod})")
        print()
    if report["by_element_atom_count"]:
        print("Element atom count over reactants (v2 only):")
        for elem, n in list(report["by_element_atom_count"].items())[:10]:
            print(f"  {elem:3s} {n}")
        print()
    print(f"Top {len(report['top_predictions'])} predictions by P:")
    for entry in report["top_predictions"]:
        print(f"  P={entry['p']:.3f}")
        print(f"    R: {entry['reactants']}")
        print(f"    P: {entry['product']}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--json", type=Path, default=None,
                   help="Also write the report as JSON to this path.")
    args = p.parse_args()

    if not args.cache_dir.is_dir():
        print(f"error: cache dir not found: {args.cache_dir}", file=sys.stderr)
        return 1

    report = analyze(args.cache_dir, top_n=args.top)
    _print_report(report)
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=2))
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
