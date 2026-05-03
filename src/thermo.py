"""Thermochemistry helpers (Sec 2.2 of the paper).

The paper sets U = 0 vs Li/Li⁺, equivalent to an electron chemical
potential μ_e = −1.40 eV. Reaction free energy of a redox half-reaction
``A + e⁻ → A⁻`` is then ``G(A⁻) − G(A) − μ_e``.

For the demo we keep a flat dict of mock free energies (eV) for each
species hash. Replace ``MOCK_G`` with a JSON loaded from real DFT data
to do a quantitative reproduction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from .molecule import MoleculeGraph

# Electron chemical potential at U = 0 vs Li/Li+
MU_E_LI_METAL = -1.40  # eV (paper Sec 2.2)
MU_E_GRAPHITE = -1.60  # eV, for U = 0.2 V vs Li/Li+


# ----------------------------------------------------------------------
# Candidate tiers
# ----------------------------------------------------------------------
class CandidateTier(str, Enum):
    """How confident are we in this reaction?

    Tiers are assigned by :func:`network.enumerate_reactions` based on
    whether FlowER endorsed the step and how strongly. Path search uses
    the tier to apply different cost discounts.

    The tier hierarchy (most → least trusted):

    FLOWER_HIGH
        FlowER probability > ``high_prob_cutoff`` (default 0.3). The
        mechanism is well within FlowER's training distribution and the
        beam search converged tightly. Path search rewards these.

    FLOWER_FRAGREC
        FlowER and fragrec both produced this exact (reactants → products)
        combination. Double endorsement; even if probabilities are modest,
        independent agreement is strong evidence.

    FRAGREC_ORGANIC
        Pure organic step that fragrec found but FlowER did not endorse
        (or did not score). Possibly a fragrec hallucination of the kind
        FlowER's filter would catch, so it pays a mild penalty.

    FRAGREC_INORGANIC
        Step containing Li / charged species / free electron — the
        electrochemistry FlowER cannot represent. Its absence from FlowER
        is *not* evidence against it; we trust ΔG alone.
    """

    FLOWER_HIGH = "flower_high"
    FLOWER_FRAGREC = "flower_fragrec"
    FRAGREC_ORGANIC = "fragrec_organic"
    FRAGREC_INORGANIC = "fragrec_inorganic"


@dataclass(frozen=True)
class Reaction:
    reactants: tuple[str, ...]   # canonical hashes
    products: tuple[str, ...]
    n_electrons: int             # +k means k electrons consumed (reduction)
    bond_changes: int
    dG: float                    # eV, including electron μ if any
    # FlowER mechanistic prior. Defaults to None when fragrec is the only
    # source for this reaction; callers that mix backends use this to
    # discount well-supported steps in the path-graph cost (see
    # ``reaction_cost`` below).
    p_flower: float | None = None
    source: str = "fragrec"      # "fragrec", "flower", or "both"
    tier: CandidateTier = CandidateTier.FRAGREC_INORGANIC

    def __repr__(self) -> str:
        arrow = " + ".join(self.reactants) + " -> " + " + ".join(self.products)
        tag = f"{self.source}/{self.tier.value}"
        if self.p_flower is not None:
            tag += f" P={self.p_flower:.2f}"
        return (
            f"<Rxn ΔG={self.dG:+.2f} eV, Δb={self.bond_changes}, "
            f"ne={self.n_electrons}, {tag}>"
        )


def reaction_dG(
    reactant_Gs: list[float],
    product_Gs: list[float],
    n_electrons: int = 0,
    mu_e: float = MU_E_LI_METAL,
) -> float:
    """ΔG = ΣG(products) − ΣG(reactants) − n_e · μ_e.

    Convention: ``n_electrons > 0`` means the reaction *consumes* electrons.
    """
    return sum(product_Gs) - sum(reactant_Gs) - n_electrons * mu_e


# ----------------------------------------------------------------------
# Tier classification helper
# ----------------------------------------------------------------------
def classify_tier(
    *,
    p_flower: float | None,
    in_fragrec: bool,
    is_organic: bool,
    high_prob_cutoff: float = 0.3,
) -> CandidateTier:
    """Map (p_flower, in_fragrec, is_organic) -> CandidateTier.

    Used by :func:`network.enumerate_reactions` per reaction.

    Parameters
    ----------
    p_flower
        FlowER probability; None if FlowER did not predict this reaction.
    in_fragrec
        True if fragrec also produced this exact (reactants, products).
    is_organic
        True if the reaction is pure organic / metal-free / closed-shell.
        False for Li/electron-transfer steps that FlowER cannot represent.
    high_prob_cutoff
        FlowER probability above which the step is "FLOWER_HIGH".
    """
    if not is_organic:
        return CandidateTier.FRAGREC_INORGANIC
    if p_flower is not None and p_flower >= high_prob_cutoff:
        return CandidateTier.FLOWER_HIGH
    if p_flower is not None and in_fragrec:
        return CandidateTier.FLOWER_FRAGREC
    return CandidateTier.FRAGREC_ORGANIC


# ----------------------------------------------------------------------
# Hybrid path cost
# ----------------------------------------------------------------------
def softplus(x: float, scale: float = 0.5) -> float:
    """Smooth, monotone-in-ΔG cost used by the original paper (Sec 2.3).

    ``scale`` controls how aggressively endergonic steps are penalised
    relative to flat thermoneutral ones.
    """
    return scale * math.log1p(math.exp(x / scale))


# Per-tier multiplicative weights applied to the −λ·log P term, plus a
# fixed bias. Higher tier => smaller penalty (i.e. cheaper edge).
_TIER_BIAS: dict[CandidateTier, float] = {
    CandidateTier.FLOWER_HIGH:        -0.5,   # discount: mechanistically endorsed
    CandidateTier.FLOWER_FRAGREC:     -0.2,   # mild discount: dual endorsement
    CandidateTier.FRAGREC_ORGANIC:    +0.3,   # mild penalty: organic step FlowER didn't see
    CandidateTier.FRAGREC_INORGANIC:   0.0,   # neutral: trust ΔG alone
}


def reaction_cost(
    rxn: Reaction,
    *,
    scale: float = 0.5,
    lam: float = 0.0,
    p_floor: float = 0.01,
    use_tier_bias: bool = False,
) -> float:
    """Edge weight for path search.

    ``cost = softplus(ΔG; scale) − λ · log P_FlowER + tier_bias``

    Components
    ----------
    softplus(ΔG)
        The paper's thermodynamic term. Always present.
    −λ · log P_FlowER
        Mechanistic-prior discount. Only kicks in when FlowER predicted
        this step. Reactions invisible to FlowER use ``p_floor`` so they
        pay a fixed, mild penalty rather than infinite cost.
    tier_bias
        Optional per-tier additive bias (see :data:`_TIER_BIAS`). Off by
        default to preserve backwards-compatibility with the v1 cost.

    Set ``lam=0`` and ``use_tier_bias=False`` to recover the paper's
    original cost exactly.
    """
    base = softplus(rxn.dG, scale=scale)
    cost = base
    if lam > 0.0:
        p = rxn.p_flower if rxn.p_flower is not None else p_floor
        p = max(p, p_floor)            # guard log
        cost = cost - lam * math.log(p)
    if use_tier_bias:
        cost = cost + _TIER_BIAS.get(rxn.tier, 0.0)
    return cost
