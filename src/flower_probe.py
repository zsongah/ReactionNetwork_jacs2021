"""Probe FlowER on a curated set of LEDC-relevant combos.

Used to assess whether FlowER's chemical priors are useful for the
JACS-2021 LEDC pipeline, before committing to a full demo run.

Run from the project root with the *jacs2021* env active and FlowER
configured (see docs/FLOWER_LOCAL_DEPLOY.md §4).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from src.flower_backend import FlowERBackend, FlowERConfig, route_combo
from src.smiles_bridge import dot_smiles_to_mols, mols_to_dot_smiles


# (label, reactant SMILES separated by '.', commentary)
COMBOS: list[tuple[str, str, str]] = [
    ("EC + H2O",          "[H]C1([H])OC(=O)OC1([H])[H].[H]O[H]",
     "control: should not react without catalyst"),
    ("EC (alone)",        "[H]C1([H])OC(=O)OC1([H])[H]",
     "ring-opening / auto-decomp"),
    ("EC + EC",           "[H]C1([H])OC(=O)OC1([H])[H].[H]C1([H])OC(=O)OC1([H])[H]",
     "two-EC concerted reaction"),
    ("VC + H2O",          "C1=COC(=O)O1.O",
     "VC = SEI additive vinylene carbonate"),
    ("EC + MeOH",         "[H]C1([H])OC(=O)OC1([H])[H].CO",
     "alcoholysis of cyclic carbonate"),
    ("EC + EC + H2O",     "[H]C1([H])OC(=O)OC1([H])[H].[H]C1([H])OC(=O)OC1([H])[H].O",
     "3-body: EC dimer + water"),
    ("CO2 + glycol",      "O=C=O.OCCO",
     "reverse direction: CO2 + ethylene glycol"),
    ("HCHO + H2O",        "C=O.O",
     "trivial control: formaldehyde hydration"),
    ("EC + MeNH2",        "[H]C1([H])OC(=O)OC1([H])[H].CN",
     "amine attack on carbonyl (aminolysis)"),
    ("DMC + H2O",         "COC(=O)OC.O",
     "open-chain carbonate hydrolysis"),
]


def main() -> None:
    if not os.environ.get("FLOWER_REPO_PATH"):
        print("ERROR: set FLOWER_REPO_PATH and FLOWER_MODEL_PATH first.")
        sys.exit(1)

    cfg = FlowERConfig.from_env()
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    backend = FlowERBackend(cfg)
    print(f"Backend available: {backend.is_available()}")
    print(f"Model            : {cfg.model_path}/{cfg.model_name}")
    print(f"Cache dir        : {cfg.cache_dir}")
    print()

    total_start = time.perf_counter()
    for label, smi, note in COMBOS:
        try:
            combo = tuple(dot_smiles_to_mols(smi))
        except Exception as e:
            print(f"[{label}]  ❌ parse error: {e}")
            continue

        route = route_combo(combo)
        print(f"=== {label} ===")
        print(f"    {smi}")
        print(f"    note   : {note}")
        print(f"    routed : {route}")
        if route != "flower":
            print(f"    (skipped — not routed to FlowER)\n")
            continue

        t0 = time.perf_counter()
        preds = backend.expand([combo], min_probability=0.0)
        dt = time.perf_counter() - t0
        print(f"    {len(preds)} predictions in {dt:.1f} s")
        for i, p in enumerate(preds, 1):
            try:
                prod = mols_to_dot_smiles(list(p.products))
            except Exception:
                prod = "<unrenderable>"
            formula = ".".join(m.formula for m in p.products)
            print(f"      [{i}] P={p.probability:.3f}  {formula}  ::  {prod}")
        print()

    total = time.perf_counter() - total_start
    print(f"Total wall time: {total:.1f} s  ({total/60:.1f} min)")


if __name__ == "__main__":
    main()
