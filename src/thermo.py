"""Thermochemistry helpers (Sec 2.2 of the paper).

The paper sets U = 0 vs Li/Li⁺, equivalent to an electron chemical
potential μ_e = −1.40 eV. Reaction free energy of a redox half-reaction
``A + e⁻ → A⁻`` is then ``G(A⁻) − G(A) − μ_e``.

For the demo we keep a flat dict of mock free energies (eV) for each
species hash. Replace ``MOCK_G`` with a JSON loaded from real DFT data
to do a quantitative reproduction.
"""
from __future__ import annotations

from dataclasses import dataclass

from .molecule import MoleculeGraph

# Electron chemical potential at U = 0 vs Li/Li+
MU_E_LI_METAL = -1.40  # eV (paper Sec 2.2)
MU_E_GRAPHITE = -1.60  # eV, for U = 0.2 V vs Li/Li+


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

    def __repr__(self) -> str:
        arrow = " + ".join(self.reactants) + " -> " + " + ".join(self.products)
        tag = f"{self.source}"
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
# Hybrid path cost
# ----------------------------------------------------------------------
import math


def softplus(x: float, scale: float = 0.5) -> float:
    """Smooth, monotone-in-ΔG cost used by the original paper (Sec 2.3).

    ``scale`` controls how aggressively endergonic steps are penalised
    relative to flat thermoneutral ones.
    """
    return scale * math.log1p(math.exp(x / scale))


def reaction_cost(
    rxn: Reaction,
    *,
    scale: float = 0.5,
    lam: float = 0.0,
    p_floor: float = 0.01,
) -> float:
    """Edge weight for path search.

    ``cost = softplus(ΔG; scale) − λ · log P_FlowER``

    Components
    ----------
    - softplus(ΔG): the paper's thermodynamic term. Always present.
    - −λ · log P_FlowER: mechanistic-prior discount. Only kicks in when
      FlowER predicted this step. Reactions invisible to FlowER (e.g.
      Li-coordination steps it was not trained on) use ``p_floor`` so
      they pay a fixed, mild penalty rather than infinite cost.

    Set ``lam=0`` to recover the paper's original cost exactly.
    """
    base = softplus(rxn.dG, scale=scale)
    if lam <= 0.0:
        return base
    p = rxn.p_flower if rxn.p_flower is not None else p_floor
    p = max(p, p_floor)            # guard log
    return base - lam * math.log(p)
