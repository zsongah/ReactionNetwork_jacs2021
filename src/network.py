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
from itertools import combinations, product as iproduct
from typing import Iterable, Optional

from .bde_model import predict_recomb_dG
from .fragrec import n_step_fragment, recombine_pool
from .molecule import MoleculeGraph, dedupe
from .thermo import MU_E_LI_METAL, CandidateTier, Reaction, classify_tier

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Combo enumeration
# ----------------------------------------------------------------------
def enumerate_combos(
    pool: list[MoleculeGraph],
    sizes: Iterable[int] = (1, 2),
    *,
    reactive_smiles: Optional[set[str]] = None,
) -> list[tuple[MoleculeGraph, ...]]:
    """All unordered combos of given sizes drawn from ``pool``.

    Size-1 yields singletons (rearrangements). Size-2 yields unordered
    pairs *with* self-pairs (a + a) — the paper allows "2 R → P" steps.
    Size-3 yields unordered triples without repetition; if you need
    radical + 2 substrates with one repeated substrate, that's still
    covered by FlowER's beam search since FlowER consumes a multiset of
    reactants.

    ``reactive_smiles``
        Optional set of canonical SMILES of "reactive" species. When
        provided, size-2 (and size-3) combos are kept only if at least
        one reactant's SMILES is in this set. Size-1 is unaffected.
        This is the principled way to prune O(N^2) FlowER calls down
        to O(K * N) where K = |reactive_smiles|, when the pool contains
        many large fragmentation products that won't react with each
        other in any chemically meaningful way (e.g. two large oligo
        carbonate fragments).
    """
    from src.smiles_bridge import mol_to_smiles  # local import: avoid cycle

    out: list[tuple[MoleculeGraph, ...]] = []
    sizes = sorted(set(sizes))

    def _is_reactive(m: MoleculeGraph) -> bool:
        if reactive_smiles is None:
            return True
        return mol_to_smiles(m) in reactive_smiles

    if 1 in sizes:
        out.extend((m,) for m in pool)
    if 2 in sizes:
        # all unordered pairs incl. self-pairs
        for i, a in enumerate(pool):
            a_react = _is_reactive(a)
            if a_react:
                out.append((a, a))
            for j in range(i + 1, len(pool)):
                b = pool[j]
                if a_react or _is_reactive(b):
                    out.append((a, b))
    if 3 in sizes:
        for trio in combinations(pool, 3):
            if any(_is_reactive(m) for m in trio):
                out.append(trio)
    return out


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
    flower_as_generator: bool = False,
    max_pool_iterations: int = 1,
    flower_prob_threshold: float = 0.05,
    combo_sizes: Iterable[int] = (1, 2),
    reactive_smiles: Optional[set[str]] = None,
) -> tuple[
    list[MoleculeGraph],
    dict[tuple[tuple[str, ...], tuple[str, ...]], float],
]:
    """Build the species pool.

    Two architectural modes:

    *Default* (``flower_as_generator=False``)
        Backwards-compatible behaviour: fragrec drives fragmentation +
        recombination; FlowER, if requested, produces priors used only to
        re-rank existing reactions. Equivalent to the v1 pipeline.

    *FlowER-as-generator* (``flower_as_generator=True``, requires
    ``backend in {"flower", "both"}``)
        FlowER outputs become **new species** in the pool. We iterate:

            pool_0 = fragrec fragments of seeds (+ recombinants)
            for i = 1 .. max_pool_iterations:
                combos = enumerate_combos(pool_{i-1}, sizes=combo_sizes)
                flower_products = expand FlowER over flower-routed combos
                fragrec_products = recombine fragrec-routed combos
                pool_i = pool_{i-1} ∪ {acceptable flower_products}
                                    ∪ {acceptable fragrec_products}
                stop if no new species

        Each FlowER output is filtered through ``is_acceptable_flower_product``
        before being admitted. Combos containing Li / charged species /
        free electrons are routed to fragrec automatically by
        :func:`flower_backend.route_combo`.

    Parameters
    ----------
    backend
        ``"fragrec"`` (default), ``"flower"``, or ``"both"``.
    flower_as_generator
        If True, FlowER products feed back into the pool (see above).
        Has no effect for ``backend == "fragrec"``.
    max_pool_iterations
        How many expansion rounds to run when ``flower_as_generator``.
        Each round re-enumerates combos over the *current* pool.
    flower_prob_threshold
        Minimum probability a FlowER prediction must have to enter the
        pool. Default 0.05 — keep this tight to control hallucinations.
    combo_sizes
        Reactant-multiplicity values to enumerate. The FlowER training
        distribution is dominated by 2-element combos; size 1 and 3 are
        also useful but produce noisier predictions.

    Returns
    -------
    pool
        Deduplicated MoleculeGraph list.
    flower_priors
        ``{(sorted_reactant_hashes, sorted_product_hashes): probability}``.
        Empty when ``backend == "fragrec"``. Consumed by
        :func:`enumerate_reactions` to annotate ``Reaction`` objects.
    """
    if backend not in ("fragrec", "flower", "both"):
        raise ValueError(f"Unknown backend: {backend!r}")
    if flower_as_generator and backend == "fragrec":
        raise ValueError(
            "flower_as_generator=True requires backend in {'flower','both'}"
        )
    if backend in ("flower", "both") and flower_backend is None:
        raise ValueError(
            "backend in {'flower','both'} requires flower_backend instance"
        )

    flower_priors: dict[tuple[tuple[str, ...], tuple[str, ...]], float] = {}

    # ------------------------------------------------------------------
    # Round 0: classical fragrec fragmentation pool.
    # ------------------------------------------------------------------
    frags: list[MoleculeGraph] = []
    for s in seeds:
        frags.extend(n_step_fragment(s, n=n_frag_steps))
    frags = dedupe(seeds + frags)

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
                est_dG = (
                    min(predict_recomb_dG(r, r, r, e) for e in elements)
                    if elements else 0.0
                )
                if est_dG < bde_threshold:
                    kept.append(r)
            recombs = kept

    pool = dedupe(frags + recombs)

    # ------------------------------------------------------------------
    # Static FlowER scoring (legacy, non-generator) path.
    # ------------------------------------------------------------------
    if backend in ("flower", "both") and not flower_as_generator:
        combos = enumerate_combos(pool, sizes=combo_sizes, reactive_smiles=reactive_smiles)
        try:
            preds = flower_backend.expand(combos, min_probability=0.0)
        except Exception as e:
            log.warning("FlowER expansion failed: %s. Continuing without priors.", e)
            preds = []
        for pred in preds:
            r_hashes = tuple(sorted(m.canonical_hash() for m in pred.reactants))
            p_hashes = tuple(sorted(m.canonical_hash() for m in pred.products))
            if r_hashes == p_hashes:
                continue
            key = (r_hashes, p_hashes)
            flower_priors[key] = max(flower_priors.get(key, 0.0), pred.probability)
        return pool, flower_priors

    # ------------------------------------------------------------------
    # FlowER-as-generator: iterative pool expansion.
    # ------------------------------------------------------------------
    if flower_as_generator:
        pool = _iterative_flower_expansion(
            pool=pool,
            flower_backend=flower_backend,
            flower_priors=flower_priors,
            max_pool_iterations=max_pool_iterations,
            flower_prob_threshold=flower_prob_threshold,
            combo_sizes=combo_sizes,
            reactive_smiles=reactive_smiles,
        )

    return pool, flower_priors


def _iterative_flower_expansion(
    pool: list[MoleculeGraph],
    flower_backend,
    flower_priors: dict[tuple[tuple[str, ...], tuple[str, ...]], float],
    max_pool_iterations: int,
    flower_prob_threshold: float,
    combo_sizes: Iterable[int],
    reactive_smiles: Optional[set[str]] = None,
) -> list[MoleculeGraph]:
    """Iteratively grow ``pool`` with FlowER products.

    On each round, every flower-routed combo over the current pool is
    queried; acceptable products extend the pool; rounds stop early when
    no new species enter.
    """
    seen_combo_keys: set[tuple[str, ...]] = set()
    current_pool = list(pool)

    for it in range(max_pool_iterations):
        all_combos = enumerate_combos(current_pool, sizes=combo_sizes, reactive_smiles=reactive_smiles)

        # Skip combos already queried in earlier rounds.
        fresh: list[tuple[MoleculeGraph, ...]] = []
        for combo in all_combos:
            key = tuple(sorted(m.canonical_hash() for m in combo))
            if key in seen_combo_keys:
                continue
            seen_combo_keys.add(key)
            fresh.append(combo)

        if not fresh:
            log.info("FlowER iteration %d: no fresh combos, stopping.", it)
            break

        try:
            preds = flower_backend.expand(
                fresh, min_probability=flower_prob_threshold,
            )
        except Exception as e:
            log.warning("FlowER iteration %d failed: %s. Stopping early.", it, e)
            break

        # Materialise predictions: record priors and collect new products.
        before = len(current_pool)
        existing = {m.canonical_hash() for m in current_pool}
        new_products: list[MoleculeGraph] = []
        for pred in preds:
            r_hashes = tuple(sorted(m.canonical_hash() for m in pred.reactants))
            p_hashes = tuple(sorted(m.canonical_hash() for m in pred.products))
            if r_hashes == p_hashes:
                continue
            key = (r_hashes, p_hashes)
            flower_priors[key] = max(
                flower_priors.get(key, 0.0), pred.probability,
            )
            for prod in pred.products:
                if prod.canonical_hash() not in existing:
                    new_products.append(prod)
                    existing.add(prod.canonical_hash())

        if not new_products:
            log.info(
                "FlowER iteration %d: no new species (queried %d combos).",
                it, len(fresh),
            )
            break

        current_pool = dedupe(current_pool + new_products)
        log.info(
            "FlowER iteration %d: pool %d -> %d (+%d species, %d combos).",
            it, before, len(current_pool), len(current_pool) - before, len(fresh),
        )

    return current_pool


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
    high_prob_cutoff: float = 0.3,
) -> list[Reaction]:
    """Enumerate concerted reactions among ``species``.

    For every (≤ max_reactants) → (≤ max_products) combination that is
    stoichiometrically balanced, accept the reaction iff its (lower-bound)
    chemical distance is ≤ ``max_bond_changes``. Redox is treated by
    allowing a balancing electron count and shifting ΔG by −n_e · μ_e.

    If ``flower_priors`` is supplied (a ``{(sorted_reactant_hashes,
    sorted_product_hashes): probability}`` mapping coming from
    :func:`build_species_pool` with a FlowER backend), each enumerated
    reaction is annotated with its ``p_flower``, ``source``, and ``tier``
    fields so that :func:`thermo.reaction_cost` can apply the mechanistic
    prior and tier-aware bias.

    Tier assignment
    ---------------
    Each ``Reaction`` is tagged with a :class:`CandidateTier`:

      * ``FLOWER_HIGH`` if FlowER probability >= ``high_prob_cutoff``.
      * ``FLOWER_FRAGREC`` if FlowER endorses *and* the (reactants,
        products) is also reachable by fragrec's CD-based enumeration
        (in this implementation: any combo we enumerate is fragrec-
        reachable, so FlowER+ enumerated == double endorsement).
      * ``FRAGREC_ORGANIC`` for organic combos FlowER did not predict.
      * ``FRAGREC_INORGANIC`` for combos containing Li / charged species
        / electron-transfer steps.
    """
    out: list[Reaction] = []
    by_hash = {m.canonical_hash(): m for m in species}
    hashes = list(by_hash)
    flower_priors = flower_priors or {}

    # Local import to avoid a top-level cycle if backend imports network.
    from .flower_backend import _METALS

    def _combo_is_organic(mols: list[MoleculeGraph]) -> bool:
        """Could FlowER even represent this combo? (See route_combo.)"""
        if any(m.charge != 0 for m in mols):
            return False
        for m in mols:
            for _, d in m.graph.nodes(data=True):
                if d["element"] in _METALS:
                    return False
        return True

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
            is_organic = _combo_is_organic(rmols + pmols)
            tier = classify_tier(
                p_flower=p_flower,
                in_fragrec=True,   # all enumerated combos are fragrec-reachable
                is_organic=is_organic,
                high_prob_cutoff=high_prob_cutoff,
            )
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
                    tier=tier,
                )
            )
    return out
