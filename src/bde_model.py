"""Mock BonDNet-style recombination ΔG predictor.

The real paper uses BonDNet (Wen et al., Chem. Sci. 2021, 12, 1858) — a
graph neural network trained on DFT-computed bond dissociation energies.
It predicts ΔG of recombination so that endergonic recombinations can be
filtered out *before* spending DFT cycles on geometry optimization.

Here we provide a deterministic, chemistry-aware **stand-in** based on
typical bond dissociation energies (eV). It is intentionally crude but
gives the correct *sign* often enough for the demo to be illustrative.

To plug in real BonDNet predictions, replace ``predict_recomb_dG`` with
a call to the trained model and keep the same signature.
"""
from __future__ import annotations

from .molecule import MoleculeGraph

# Approximate homolytic BDEs in eV (very rough; for demo only).
# Sources: standard physical-chemistry references rounded to 0.05 eV.
TYPICAL_BDE = {
    frozenset(("C", "C")): 3.6,
    frozenset(("C", "H")): 4.3,
    frozenset(("C", "O")): 3.7,
    frozenset(("C", "N")): 3.1,
    frozenset(("O", "H")): 4.8,
    frozenset(("O", "O")): 1.5,
    frozenset(("N", "H")): 4.0,
    frozenset(("Li", "O")): 3.5,    # ionic; very approximate
    frozenset(("Li", "C")): 2.5,
    frozenset(("Li", "H")): 2.4,
    frozenset(("H", "H")): 4.5,
}
DEFAULT_BDE = 2.5


def predict_bond_bde(el_a: str, el_b: str) -> float:
    return TYPICAL_BDE.get(frozenset((el_a, el_b)), DEFAULT_BDE)


def predict_recomb_dG(
    a: MoleculeGraph, b: MoleculeGraph, recomb: MoleculeGraph,
    new_edge: tuple[str, str],
) -> float:
    """ΔG (eV) for ``a + b → recomb``.

    By microscopic reversibility, ΔG_recomb ≈ −BDE of the new bond.
    Returns a negative number for an exergonic recombination (favourable).
    """
    bde = predict_bond_bde(*new_edge)
    return -bde
