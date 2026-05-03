"""End-to-end demo: EC + Li+ + H2O + e- ----> LEDC.

Pipeline (matches paper Sec 2.1-2.2):
    1. Build species pool by 2-step fragmentation + 1-step recombination
       of the seed molecules (EC, Li+, H2O), filtered by a BonDNet
       stand-in for ΔG_recomb.
       Optionally augment with FlowER (organic-only) — see ``--backend``.
    2. Assign mock free energies (per-atom heuristics — replace with DFT).
    3. Enumerate all stoichiometrically-balanced reactions with
       chemical distance ≤ MAX_BOND_CHANGES.
    4. Build a bipartite pathfinding graph with hybrid cost
       ``softplus(ΔG) − λ · log P_FlowER``.
    5. Find the top-K shortest reaction paths to LEDC.

Run from project root::

    python -m src.run_demo                         # original pipeline
    python -m src.run_demo --backend both --lam 0.5  # add FlowER prior
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random

from data.seed_species import SEEDS, TARGET, make_LEDC
from src.molecule import MoleculeGraph, dedupe
from src.network import build_species_pool, enumerate_reactions
from src.pathfind import build_pathfinding_graph, k_shortest_paths
from src.thermo import MU_E_LI_METAL


# ---------------------------------------------------------------- params
# Trimmed aggressively for laptop runtime. The paper's network is 570
# species / 9M reactions; this demo aims for ~hundreds / ~thousands so
# the whole pipeline runs in under a minute.
MAX_BOND_CHANGES = 3
N_FRAG_STEPS = 1            # 1-step is plenty for EC + H2O fragments
MAX_POOL_SIZE = 80          # cap on species kept after fragrec
MAX_HEAVY_ATOMS = 12        # discard recombinants larger than LEDC heavy-frame
N_PATHS = 5
RANDOM_SEED = 0


def mock_free_energy(mol: MoleculeGraph) -> float:
    """Heuristic G in eV: per-atom + per-bond contributions + charge cost.

    Calibrated only loosely; meant to give a plausible energy landscape
    so the path-search demo is non-trivial. To do quantitative work,
    replace this with G values from the published JCESR / Materials
    Project quantum-chemistry dataset.
    """
    g = 0.0
    atomic = {"H": -0.5, "C": -1.0, "O": -2.0, "Li": -0.2, "N": -1.5, "F": -2.5}
    bond = {
        frozenset(("C", "H")): -2.1,
        frozenset(("C", "C")): -1.8,
        frozenset(("C", "O")): -1.9,
        frozenset(("O", "H")): -2.4,
        frozenset(("Li", "O")): -1.7,
        frozenset(("Li", "C")): -1.2,
        frozenset(("Li", "H")): -1.2,
    }
    for _, d in mol.graph.nodes(data=True):
        g += atomic.get(d["element"], -1.0)
    for u, v in mol.graph.edges():
        ea = mol.graph.nodes[u]["element"]
        eb = mol.graph.nodes[v]["element"]
        g += bond.get(frozenset((ea, eb)), -1.5)
    g += 0.4 * abs(mol.charge)        # rough Born-style charging penalty
    g += 0.2 * mol.spin               # radical destabilisation
    return g


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--backend", choices=["fragrec", "flower", "both"], default="fragrec",
        help="Species-pool backend. 'flower' and 'both' require a FlowER "
             "checkpoint; see docs/FLOWER_INTEGRATION.md.",
    )
    p.add_argument(
        "--lam", type=float, default=0.0,
        help="Weight of the FlowER mechanistic-prior term in the path "
             "cost: cost = softplus(ΔG) − λ · log P_FlowER. λ=0 reproduces "
             "the paper's original cost.",
    )
    p.add_argument(
        "--p-floor", type=float, default=0.01,
        help="Minimum P_FlowER for reactions FlowER did not produce. "
             "Pinning this away from 0 prevents −log explosion.",
    )
    p.add_argument(
        "--max-bond-changes", type=int, default=MAX_BOND_CHANGES,
        help="CD upper bound for reaction enumeration (paper uses 5).",
    )
    p.add_argument("--n-paths", type=int, default=N_PATHS)
    p.add_argument("--verbose", action="store_true")

    # ---- FlowER-as-generator options (opt-in) --------------------------
    gen = p.add_argument_group("FlowER-as-generator (opt-in)")
    gen.add_argument(
        "--flower-as-generator", action="store_true",
        help="Use FlowER as a primary product generator (in addition to "
             "fragrec) and feed its products back into the species pool. "
             "Requires --backend flower or both.",
    )
    gen.add_argument(
        "--max-pool-iterations", type=int, default=1,
        help="Number of FlowER expansion rounds when "
             "--flower-as-generator is on (default 1).",
    )
    gen.add_argument(
        "--flower-prob-threshold", type=float, default=0.05,
        help="Discard FlowER predictions with probability below this "
             "threshold (default 0.05).",
    )
    gen.add_argument(
        "--combo-sizes", type=str, default="1,2",
        help="Comma-separated reactant combo sizes to feed FlowER "
             "(default '1,2'; size 3 supported but slower).",
    )
    gen.add_argument(
        "--use-tier-bias", action="store_true",
        help="Add per-tier additive bias to reaction cost "
             "(FLOWER_HIGH discounted, FRAGREC_ORGANIC penalized).",
    )
    return p.parse_args()


def maybe_make_flower_backend(backend: str):
    """Construct a FlowERBackend if requested; otherwise return None.

    The backend is returned even when the FlowER install is missing, as
    long as a cache directory exists — this supports cache-only runs on
    GPU-less machines. ``expand()`` will raise ``FlowERUnavailable`` if
    a cache miss forces a real subprocess invocation.
    """
    if backend == "fragrec":
        return None
    from src.flower_backend import FlowERBackend, FlowERConfig

    cfg = FlowERConfig.from_env()
    fb = FlowERBackend(cfg)
    if not fb.is_available() and not any(cfg.cache_dir.glob("*.json")):
        raise RuntimeError(
            "FlowER backend requested but neither a working install "
            "nor a populated cache was found. Set FLOWER_REPO_PATH and "
            "FLOWER_MODEL_PATH (and download the checkpoint), or copy a "
            "cache/flower/ directory from a host that has run FlowER. "
            "See docs/FLOWER_INTEGRATION.md."
        )
    return fb


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    random.seed(RANDOM_SEED)
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.normpath(os.path.join(here, "..", "results"))
    os.makedirs(out_dir, exist_ok=True)

    flower_backend = maybe_make_flower_backend(args.backend)

    # Validate generator-mode flag.
    if args.flower_as_generator and args.backend == "fragrec":
        raise SystemExit(
            "--flower-as-generator requires --backend flower or both."
        )
    combo_sizes = tuple(
        int(s) for s in args.combo_sizes.split(",") if s.strip()
    )

    # ---- 1. species pool -------------------------------------------------
    seeds = list(SEEDS.values())
    print(f"Seeds:                        {[s.name for s in seeds]}")
    print(f"Backend:                      {args.backend}")
    if args.flower_as_generator:
        print(f"FlowER-as-generator:          on "
              f"(iters={args.max_pool_iterations}, "
              f"p≥{args.flower_prob_threshold}, "
              f"combo_sizes={combo_sizes})")

    pool, flower_priors = build_species_pool(
        seeds, n_frag_steps=N_FRAG_STEPS, keep_endergonic_recomb=False,
        backend=args.backend, flower_backend=flower_backend,
        flower_as_generator=args.flower_as_generator,
        max_pool_iterations=args.max_pool_iterations,
        flower_prob_threshold=args.flower_prob_threshold,
        combo_sizes=combo_sizes,
    )
    print(f"  raw pool:                   {len(pool)}")
    if flower_priors:
        print(f"  FlowER priors collected:    {len(flower_priors)}")

    # Filter out absurdly large recombinants and cap pool size.
    pool = [m for m in pool if sum(
        1 for _, d in m.graph.nodes(data=True) if d["element"] != "H"
    ) <= MAX_HEAVY_ATOMS]
    print(f"  after heavy-atom filter:    {len(pool)}")

    # Make sure the target is in the pool.
    target = make_LEDC()
    if not any(target.is_isomorphic(m) for m in pool):
        pool.append(target)
    # Always keep seeds.
    seed_hashes = {s.canonical_hash() for s in seeds}
    seed_pool = [m for m in pool if m.canonical_hash() in seed_hashes]
    other_pool = [m for m in pool if m.canonical_hash() not in seed_hashes
                  and not target.is_isomorphic(m)]
    if len(other_pool) > MAX_POOL_SIZE - len(seed_pool) - 1:
        other_pool = other_pool[: MAX_POOL_SIZE - len(seed_pool) - 1]
    pool = dedupe(seed_pool + other_pool + [target])
    print(f"Species pool size (final):    {len(pool)}")

    # ---- 2. free energies ------------------------------------------------
    G = {m.canonical_hash(): mock_free_energy(m) for m in pool}
    name_of = {m.canonical_hash(): (m.name or m.formula) for m in pool}

    # ---- 3. reaction enumeration ----------------------------------------
    reactions = enumerate_reactions(
        pool, G,
        max_bond_changes=args.max_bond_changes,
        max_reactants=2, max_products=2,
        mu_e=MU_E_LI_METAL,
        flower_priors=flower_priors,
    )
    n_with_prior = sum(1 for r in reactions if r.p_flower is not None)
    print(f"Reactions (CD ≤ {args.max_bond_changes}):           {len(reactions)} "
          f"({n_with_prior} carry a FlowER prior)")

    # ---- 4. pathfinding graph -------------------------------------------
    starting = {m.canonical_hash() for m in seeds}
    pf_graph = build_pathfinding_graph(
        reactions, starting_pool=starting, scale=0.5,
        lam=args.lam, p_floor=args.p_floor,
        use_tier_bias=args.use_tier_bias,
    )
    target_h = target.canonical_hash()
    print(f"Path graph: {pf_graph.number_of_nodes()} nodes, "
          f"{pf_graph.number_of_edges()} edges")
    print(f"Cost weights: scale=0.5, lam={args.lam}, p_floor={args.p_floor}, "
          f"tier_bias={args.use_tier_bias}")

    # ---- 5. K-shortest paths --------------------------------------------
    paths = k_shortest_paths(pf_graph, target_h, k=args.n_paths)
    if not paths:
        print("No path to LEDC found — try increasing --max-bond-changes "
              "or expanding the species pool.")
        return

    print(f"\nTop {len(paths)} shortest pathways to LEDC")
    print("=" * 60)
    summaries = []
    for i, p in enumerate(paths, 1):
        print(f"\n--- Path {i} ---")
        print(p.pretty(name_of))
        summaries.append({
            "rank": i,
            "cost": p.cost,
            "n_steps": len(p.reactions),
            "steps": [
                {
                    "reactants": [name_of.get(h, h[:8]) for h in r.reactants],
                    "products": [name_of.get(h, h[:8]) for h in r.products],
                    "dG_eV": r.dG,
                    "n_electrons": r.n_electrons,
                    "bond_changes": r.bond_changes,
                    "p_flower": r.p_flower,
                    "source": r.source,
                    "tier": r.tier.value if r.tier is not None else None,
                }
                for r in p.reactions
            ],
        })

    out_path = os.path.join(out_dir, "ledc_paths.json")
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "paths": summaries}, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
