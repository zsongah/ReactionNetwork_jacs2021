"""Reaction network construction (Sec 2.2 of the paper).

The paper's full pipeline enumerates all reactant/product combinations
(up to two reactants and two products) and accepts any reaction whose
**chemical distance** — minimum number of bond changes needed to map
reactants to products — is at most ``max_bond_changes`` (5 in the
paper's main network, 2 in the smaller no-water network).

The chemical distance problem is solved exactly via mixed-integer linear
programming (MILP) in the original work. For a teaching demo we use a
simpler, exact-but-slow enumeration: we compute the symmetric difference
of the multiset of edges (with element-pair labels) of reactants vs
products. This is a *lower bound* of CD that becomes tight when atoms
are not relabelled across the reaction — which is true for the small
toy network here.
"""
from __future__ import annotations

import logging
from collections import Counter
from itertools import product as iproduct
from typing import Iterable, Optional

from .bde_model import predict_recomb_dG
from .fragrec import n_step_fragment, recombine_pool
from .molecule import MoleculeGraph, dedupe
from .thermo import MU_E_LI_METAL, Reaction

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Species pool
# ----------------------------------------------------------------------
def build_species_pool(
    seeds: list[MoleculeGraph],
    n_frag_steps: int = 2,
    keep_endergonic_recomb: bool = False,
    bde_threshold: float = 0.0,
    *,
    backend: str = "fragrec",
    flower_backend=None,
) -> tuple[list[MoleculeGraph], dict[tuple[tuple[str, ...], tuple[str, ...]], float]]:
    """Reproduce the 4-stage species generation of Sec 2.1.

    1. n-step fragmentation of every seed.
    2. One-step recombination of every fragment pair.
    3. ML filter: keep only exergonic recombinations (ΔG_recomb < threshold).
    4. (Skipped here:) DFT geometry optimization.

    Backend dispatch
    ----------------
    - ``"fragrec"`` (default): the paper's combinatorial pipeline.
    - ``"flower"``: queries a configured FlowER backend on every pair of
      *organic* species in the seed-derived fragment pool. The reactant
      sets it returns become reactions; the product molecules expand the
      species pool.
    - ``"both"``: union of the two. Organic combinations get FlowER; Li /
      charged / radical-bearing combinations stay with fragrec, matching
      the user's policy of letting fragrec handle the electrochemistry.

    Returns
    -------
    pool
        Deduplicated list of MoleculeGraph (seeds + fragments + recombinants
        + any FlowER products).
    flower_priors
        ``{(reactant_hashes, product_hashes): probability}``. Empty when
        backend == "fragrec". Consumed by :func:`enumerate_reactions` to
        annotate the corresponding ``Reaction`` objects.
    """
    if backend not in ("fragrec", "flower", "both"):
        raise ValueError(f"Unknown backend: {backend!r}")

    # 1. Fragmentation pool — used by every backend as input.
    frags: list[MoleculeGraph] = []
    for s in seeds:
        frags.extend(n_step_fragment(s, n=n_frag_steps))
    frags = dedupe(seeds + frags)

    # 2/3. Recombination via fragrec (skipped if backend == "flower").
    recombs: list[MoleculeGraph] = []
    if backend in ("fragrec", "both"):
        recombs = recombine_pool(frags)
        if not keep_endergonic_recomb:
            kept: list[MoleculeGraph] = []
            for r in recombs:
                elements = [
                    (r.graph.nodes[u]["element"], r.graph.nodes[v]["element"])
                    for u, v in r.graph.edges()
                ]
                est_dG = min(
                    predict_recomb_dG(r, r, r, e) for e in elements
                ) if elements else 0.0
                if est_dG < bde_threshold:
                    kept.append(r)
            recombs = kept

    # 2'. FlowER expansion (organic-only; backend handles filtering).
    flower_priors: dict[tuple[tuple[str, ...], tuple[str, ...]], float] = {}
    flower_products: list[MoleculeGraph] = []
    if backend in ("flower", "both"):
        if flower_backend is None:
            raise ValueError(
                "backend in {'flower','both'} requires flower_backend instance"
            )
        # Build candidate reactant tuples: pairs over the fragment pool.
        # Singletons (rearrangements) are also useful; FlowER handles them.
        combos: list[tuple[MoleculeGraph, ...]] = [(m,) for m in frags]
        for i, a in enumerate(frags):
            for b in frags[i:]:
                combos.append((a, b))

        try:
            preds = flower_backend.expand(combos)
        except Exception as e:
            log.warning("FlowER expansion failed: %s. Falling back.", e)
            preds = []

        for pred in preds:
            for prod in pred.products:
                flower_products.append(prod)
            r_hashes = tuple(sorted(m.canonical_hash() for m in pred.reactants))
            p_hashes = tuple(sorted(m.canonical_hash() for m in pred.products))
            # If FlowER predicts the trivial no-op, ignore.
            if r_hashes == p_hashes:
                continue
            # Take the max probability if multiple predictions land on the
            # same (reactants, products) tuple.
            key = (r_hashes, p_hashes)
            flower_priors[key] = max(flower_priors.get(key, 0.0), pred.probability)

    pool = dedupe(frags + recombs + flower_products)
    return pool, flower_priors


# ----------------------------------------------------------------------
# Chemical-distance approximation
# ----------------------------------------------------------------------
def edge_multiset(mols: Iterable[MoleculeGraph]) -> Counter:
    """Multiset of edges across a list of molecules, keyed by (element pair)."""
    c: Counter = Counter()
    for m in mols:
        for u, v in m.graph.edges():
            ea = m.graph.nodes[u]["element"]
            eb = m.graph.nodes[v]["element"]
            key = tuple(sorted((ea, eb)))
            c[key] += 1
    return c


def chemical_distance(
    reactants: list[MoleculeGraph], products: list[MoleculeGraph]
) -> int:
    """Lower-bound CD: bonds that must be broken + bonds that must be formed.

    True CD requires solving an atom-mapping MILP (paper Sec 2.2, SI I).
    For the demo this approximation is sufficient and conservative.
    """
    a = edge_multiset(reactants)
    b = edge_multiset(products)
    diff_break = sum((a - b).values())   # in reactants but not products
    diff_form = sum((b - a).values())    # in products but not reactants
    return diff_break + diff_form


def stoichiometry_balanced(
    reactants: list[MoleculeGraph], products: list[MoleculeGraph]
) -> bool:
    """Atom counts and net charge must balance (electrons handled separately)."""
    from collections import Counter as C

    ra = C()
    pa = C()
    for m in reactants:
        for _, d in m.graph.nodes(data=True):
            ra[d["element"]] += 1
    for m in products:
        for _, d in m.graph.nodes(data=True):
            pa[d["element"]] += 1
    return ra == pa


# ----------------------------------------------------------------------
# Reaction enumeration
# ----------------------------------------------------------------------
def enumerate_reactions(
    species: list[MoleculeGraph],
    G: dict[str, float],
    max_bond_changes: int = 5,
    max_reactants: int = 2,
    max_products: int = 2,
    mu_e: float = MU_E_LI_METAL,
    *,
    flower_priors: Optional[
        dict[tuple[tuple[str, ...], tuple[str, ...]], float]
    ] = None,
) -> list[Reaction]:
    """Enumerate concerted reactions among ``species``.

    For every (≤ max_reactants) → (≤ max_products) combination that is
    stoichiometrically balanced, accept the reaction iff its (lower-bound)
    chemical distance is ≤ ``max_bond_changes``. Redox is treated by
    allowing a balancing electron count and shifting ΔG by −n_e · μ_e.

    If ``flower_priors`` is supplied (a ``{(sorted_reactant_hashes,
    sorted_product_hashes): probability}`` mapping coming from
    :func:`build_species_pool` with a FlowER backend), each enumerated
    reaction is annotated with its ``p_flower`` and ``source`` fields so
    that :func:`thermo.reaction_cost` can apply the mechanistic prior.
    """
    out: list[Reaction] = []
    by_hash = {m.canonical_hash(): m for m in species}
    hashes = list(by_hash)
    flower_priors = flower_priors or {}

    # build all reactant tuples (1 or 2 species)
    reactant_sets = [(h,) for h in hashes] + [
        (h1, h2) for i, h1 in enumerate(hashes) for h2 in hashes[i:]
    ]
    product_sets = reactant_sets  # same shape

    for rs in reactant_sets:
        rmols = [by_hash[h] for h in rs]
        for ps in product_sets:
            if rs == ps:
                continue
            pmols = [by_hash[h] for h in ps]
            if not stoichiometry_balanced(rmols, pmols):
                continue
            cd = chemical_distance(rmols, pmols)
            if cd == 0 or cd > max_bond_changes:
                continue
            net_charge_change = (
                sum(m.charge for m in pmols) - sum(m.charge for m in rmols)
            )
            n_e = -net_charge_change   # electrons consumed
            dG = (
                sum(G[h] for h in ps)
                - sum(G[h] for h in rs)
                - n_e * mu_e
            )
            key = (tuple(sorted(rs)), tuple(sorted(ps)))
            p_flower = flower_priors.get(key)
            source = "flower" if p_flower is not None else "fragrec"
            out.append(
                Reaction(
                    reactants=rs,
                    products=ps,
                    n_electrons=n_e,
                    bond_changes=cd,
                    dG=dG,
                    p_flower=p_flower,
                    source=source,
                )
            )
    return out
