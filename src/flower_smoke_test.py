"""Smoke-test the local FlowER deployment on a single reactant pair.

Run from the project root with the *jacs2021* env active::

    conda activate jacs2021
    export FLOWER_REPO_PATH="$PWD/vendor/FlowER"
    export FLOWER_MODEL_PATH="$PWD/vendor/FlowER/checkpoints/<exp>/<exp>/model.<step>_<idx>.pt"
    export PYTHON_EXECUTABLE="$HOME/miniforge3/envs/flower/bin/python"
    python -m src.flower_smoke_test

The script tries one canonical organic combo (EC + H2O) and prints the
top product sets that FlowER returns. Cache is forced cold the first
run; subsequent runs hit ``cache/flower/`` instantly.

If the script fails, see ``docs/FLOWER_LOCAL_DEPLOY.md`` §3.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from data.seed_species import make_EC, make_H2O
from src.flower_backend import (
    FlowERBackend,
    FlowERConfig,
    FlowERUnavailable,
    route_combo,
)
from src.smiles_bridge import mols_to_dot_smiles


def _check_env() -> None:
    repo = os.environ.get("FLOWER_REPO_PATH")
    model = os.environ.get("FLOWER_MODEL_PATH")
    pyx = os.environ.get("PYTHON_EXECUTABLE")
    missing = []
    if not repo:
        missing.append("FLOWER_REPO_PATH")
    if not model:
        missing.append("FLOWER_MODEL_PATH")
    if missing:
        print("ERROR: missing env var(s):", ", ".join(missing))
        print("See docs/FLOWER_LOCAL_DEPLOY.md §4 for the export block.")
        sys.exit(1)
    if not Path(repo).exists():
        print(f"ERROR: FLOWER_REPO_PATH does not exist: {repo}")
        sys.exit(1)
    if not Path(model).exists():
        print(f"ERROR: FLOWER_MODEL_PATH does not exist: {model}")
        print("Did you unzip checkpoints.zip into vendor/FlowER/?")
        sys.exit(1)
    if pyx and not Path(pyx).exists():
        print(f"WARN: PYTHON_EXECUTABLE not found at {pyx}; "
              f"falling back to system 'python'.")


def main() -> None:
    _check_env()

    cfg = FlowERConfig.from_env()
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    backend = FlowERBackend(cfg)

    print(f"FlowER repo : {cfg.repo_path}")
    print(f"Checkpoint  : {cfg.model_path}")
    print(f"Cache dir   : {cfg.cache_dir}")
    print(f"Interpreter : {cfg.python_executable}")
    print(f"Available?  : {backend.is_available()}")
    print()

    ec, water = make_EC(), make_H2O()
    combo = (ec, water)
    print(f"Reactants   : {mols_to_dot_smiles(list(combo))}")
    print(f"Routed to   : {route_combo(combo)}")
    print()

    t0 = time.perf_counter()
    try:
        preds = backend.expand([combo], min_probability=0.0)
    except FlowERUnavailable as e:
        print(f"ERROR FlowERUnavailable: {e}")
        sys.exit(2)
    dt = time.perf_counter() - t0

    print(f"Got {len(preds)} predictions in {dt:.1f} s.")
    print()
    for i, p in enumerate(preds, 1):
        prod_smi = mols_to_dot_smiles(list(p.products))
        prod_formula = ".".join(m.formula for m in p.products)
        print(f"  [{i}] P={p.probability:.3f}  →  {prod_smi}")
        print(f"        formula: {prod_formula}")

    if not preds:
        print("(empty — try lowering min_probability or check stderr)")
        sys.exit(3)


if __name__ == "__main__":
    main()
