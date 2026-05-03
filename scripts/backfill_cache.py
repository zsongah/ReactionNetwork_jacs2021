#!/usr/bin/env python3
"""
Backfill legacy FlowER cache entries with reactant SMILES.

Old cache files (schema v1) stored only [[product_smiles, count], ...].
Schema v2 wraps that list in {"reactants": ..., "products": ...} so
offline analyses can recover what was asked.

This script re-runs the same combo enumeration the live pipeline would
generate from the seed species, hashes each combo with the FlowER
backend's _cache_key, and rewrites any matching legacy file in v2
format. Files whose reactants we cannot recover (e.g. combos that were
only reachable through iterative pool expansion) are left untouched.

Usage:
    python -m scripts.backfill_cache
    python -m scripts.backfill_cache --dry-run

The cache_dir / model_name / sample_size must match the values the
demo used so that hashes line up. Defaults track FlowERConfig.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.flower_backend import FlowERBackend, FlowERConfig  # noqa: E402
from src.network import build_species_pool, enumerate_combos  # noqa: E402
from src.smiles_bridge import mol_to_smiles  # noqa: E402
from data.seed_species import SEEDS  # noqa: E402


def _is_legacy(path: Path) -> bool:
    try:
        with path.open() as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return False
    return isinstance(data, list)


def _load_legacy(path: Path) -> list:
    with path.open() as f:
        return json.load(f)


def _write_v2(path: Path, reactants: str, products: list, model: str, sample_size: int) -> None:
    payload = {
        "reactants": reactants,
        "model": model,
        "sample_size": sample_size,
        "products": products,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f)
    tmp.replace(path)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "cache" / "flower")
    p.add_argument("--combo-sizes", type=str, default="1,2",
                   help="Same value used for the demo (default 1,2).")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cfg = FlowERConfig(cache_dir=args.cache_dir)
    backend = FlowERBackend(cfg)

    seeds = list(SEEDS.values())
    pool, _ = build_species_pool(seeds, n_frag_steps=1)  # match demo's N_FRAG_STEPS
    sizes = tuple(int(s) for s in args.combo_sizes.split(","))
    combos = list(enumerate_combos(pool, sizes=sizes))
    print(f"Enumerated {len(combos)} combos from {len(pool)} pool species, sizes={sizes}")

    # Build hash -> reactant_smiles map
    hash_to_smi: dict[str, str] = {}
    for combo in combos:
        smi = ".".join(sorted(mol_to_smiles(m) for m in combo))
        key = backend._cache_key(smi)  # deterministic, pure function
        hash_to_smi.setdefault(key, smi)

    print(f"Built {len(hash_to_smi)} unique hash->reactant mappings")

    n_legacy = 0
    n_recovered = 0
    n_unrecovered = 0
    for path in sorted(args.cache_dir.glob("*.json")):
        if not _is_legacy(path):
            continue
        n_legacy += 1
        key = path.stem
        smi = hash_to_smi.get(key)
        if smi is None:
            n_unrecovered += 1
            continue
        products = _load_legacy(path)
        if args.dry_run:
            print(f"would rewrite {path.name}: reactants={smi}")
        else:
            _write_v2(path, smi, products, cfg.model_name, cfg.sample_size)
        n_recovered += 1

    print(f"\nLegacy files inspected: {n_legacy}")
    print(f"  Recovered (v2 written): {n_recovered}")
    print(f"  Unrecovered (no hash match): {n_unrecovered}")
    if args.dry_run:
        print("(dry run, no files were modified)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
