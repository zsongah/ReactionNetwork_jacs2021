"""Unit tests for the layered FlowER routing/acceptance filters.

These do NOT exercise the FlowER subprocess. They cover the pure-Python
helpers added in the FlowER-as-generator refactor:

  Layer 0 :  is_flower_compatible          (per-species)
  reactivity heuristic : has_reactive_site (per-species)
  Layer 1 :  is_reactive_combo             (per-combo)
  Layer 1 :  route_combo                   (FlowER vs fragrec)
  Layer 2 :  is_acceptable_flower_product  (post-output sanity)
  tier    :  thermo.classify_tier          (mapping endorsement → tier)
"""
from __future__ import annotations

import pytest

from data.seed_species import make_EC, make_H2O, make_Li_cation
from src.flower_backend import (
    has_reactive_site,
    is_acceptable_flower_product,
    is_flower_compatible,
    is_reactive_combo,
    route_combo,
)
from src.smiles_bridge import smiles_to_mol
from src.thermo import CandidateTier, classify_tier


# ----------------------------------------------------------------------
# Layer 0: is_flower_compatible
# ----------------------------------------------------------------------
def test_compatible_neutral_organic() -> None:
    assert is_flower_compatible(make_EC()) is True
    assert is_flower_compatible(make_H2O()) is True


def test_compatible_rejects_lithium() -> None:
    assert is_flower_compatible(make_Li_cation()) is False


def test_compatible_rejects_charged_anion() -> None:
    # Hydroxide is small and metal-free, but charged → reject.
    oh_minus = smiles_to_mol("[OH-]")
    assert is_flower_compatible(oh_minus) is False


def test_compatible_accepts_neutral_radical() -> None:
    # CH3• is a neutral radical: spin > 0, charge 0 → architecturally fine.
    methyl = smiles_to_mol("[CH3]")
    assert is_flower_compatible(methyl) is True


# ----------------------------------------------------------------------
# has_reactive_site
# ----------------------------------------------------------------------
def test_reactive_site_radical() -> None:
    assert has_reactive_site(smiles_to_mol("[CH3]")) is True


def test_reactive_site_double_bond() -> None:
    assert has_reactive_site(smiles_to_mol("C=C")) is True


def test_reactive_site_heteroatom() -> None:
    assert has_reactive_site(make_H2O()) is True


def test_reactive_site_saturated_alkane_is_inert() -> None:
    # Methane: no radical, no multibond, only C/H → inert by heuristic.
    methane = smiles_to_mol("C")
    assert has_reactive_site(methane) is False


# ----------------------------------------------------------------------
# Layer 1: is_reactive_combo
# ----------------------------------------------------------------------
def test_combo_accepts_ec_water() -> None:
    assert is_reactive_combo([make_EC(), make_H2O()]) is True


def test_combo_rejects_oversize() -> None:
    # 4-mer combo trips max_size = 3.
    w = make_H2O()
    assert is_reactive_combo([w, w, w, w]) is False


def test_combo_rejects_three_identical() -> None:
    w = make_H2O()
    assert is_reactive_combo([w, w, w]) is False


def test_combo_rejects_pure_alkane_pair() -> None:
    methane = smiles_to_mol("C")
    assert is_reactive_combo([methane, methane]) is False


# ----------------------------------------------------------------------
# route_combo
# ----------------------------------------------------------------------
def test_route_organic_pair_to_flower() -> None:
    assert route_combo([make_EC(), make_H2O()]) == "flower"


def test_route_lithium_to_fragrec() -> None:
    assert route_combo([make_Li_cation(), make_H2O()]) == "fragrec"


def test_route_charged_to_fragrec() -> None:
    cation = smiles_to_mol("[NH4+]")
    assert route_combo([cation]) == "fragrec"


def test_route_inert_pair_to_fragrec() -> None:
    methane = smiles_to_mol("C")
    assert route_combo([methane, methane]) == "fragrec"


# ----------------------------------------------------------------------
# Layer 2: is_acceptable_flower_product
# ----------------------------------------------------------------------
def test_acceptable_mass_balanced() -> None:
    # EC (6 heavy) + H2O (1 heavy) → ethylene glycol (4) + CO2 (3) = 7. OK.
    reactants = [make_EC(), make_H2O()]
    products = [smiles_to_mol("OCCO"), smiles_to_mol("O=C=O")]
    assert is_acceptable_flower_product(products, reactants, 0.5) is True


def test_acceptable_rejects_low_probability() -> None:
    reactants = [make_EC(), make_H2O()]
    products = [smiles_to_mol("OCCO"), smiles_to_mol("O=C=O")]
    assert is_acceptable_flower_product(
        products, reactants, 0.01, min_probability=0.05
    ) is False


def test_acceptable_rejects_heavy_atom_violation() -> None:
    # EC + H2O = 7 heavy; products with only 4 heavy → ΔH = 3, rejected.
    reactants = [make_EC(), make_H2O()]
    products = [smiles_to_mol("CO"), smiles_to_mol("OC=O")]  # 2 + 3 = 5
    # |5 - 7| = 2 > 1 slack → reject
    assert is_acceptable_flower_product(products, reactants, 0.5) is False


def test_acceptable_rejects_metal_introduction() -> None:
    reactants = [make_EC(), make_H2O()]
    # Pretend FlowER hallucinated a Li atom.
    products = [make_Li_cation(), smiles_to_mol("OCCO"), smiles_to_mol("O=C=O")]
    assert is_acceptable_flower_product(products, reactants, 0.9) is False


# ----------------------------------------------------------------------
# thermo.classify_tier
# ----------------------------------------------------------------------
def test_tier_flower_high_above_cutoff() -> None:
    assert classify_tier(
        p_flower=0.6, in_fragrec=True, is_organic=True,
        high_prob_cutoff=0.3,
    ) is CandidateTier.FLOWER_HIGH


def test_tier_flower_fragrec_below_cutoff() -> None:
    assert classify_tier(
        p_flower=0.1, in_fragrec=True, is_organic=True,
        high_prob_cutoff=0.3,
    ) is CandidateTier.FLOWER_FRAGREC


def test_tier_fragrec_organic_no_flower() -> None:
    assert classify_tier(
        p_flower=None, in_fragrec=True, is_organic=True,
    ) is CandidateTier.FRAGREC_ORGANIC


def test_tier_fragrec_inorganic_no_flower() -> None:
    assert classify_tier(
        p_flower=None, in_fragrec=True, is_organic=False,
    ) is CandidateTier.FRAGREC_INORGANIC
