"""Round-trip and shape tests for src.smiles_bridge.

These run on the laptop conda env (no FlowER, no GPU). They guard against
regressions in the SMILES <-> MoleculeGraph adapter that FlowER depends
on.

Run with::

    python -m pytest tests/test_smiles_bridge.py -q
"""
from __future__ import annotations

import pytest
from rdkit import Chem

from data.seed_species import make_LEDC, make_EC, make_H2O, make_Li_cation
from src.smiles_bridge import (
    BridgeError,
    dot_smiles_to_mols,
    mol_to_smiles,
    mols_to_dot_smiles,
    smiles_to_mol,
)


def _canon(smi: str) -> str:
    """RDKit canonicalisation, so we compare apples to apples."""
    m = Chem.MolFromSmiles(smi)
    assert m is not None, f"RDKit could not parse {smi!r}"
    return Chem.MolToSmiles(m, canonical=True)


# ----------------------------------------------------------------------
# SMILES -> MoleculeGraph -> SMILES round-trip
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "smi",
    [
        "O",                       # H2O
        "[Li+]",                   # bare Li+
        "[OH-]",                   # hydroxide
        "C=O",                     # formaldehyde
        "CC=O",                    # acetaldehyde
        "CO",                      # methanol
        "C(=O)(O)O",               # carbonic acid
        "O=C1OCCO1",               # ethylene carbonate (EC)
    ],
)
def test_round_trip_neutral_and_simple_ions(smi: str) -> None:
    mol = smiles_to_mol(smi)
    out = mol_to_smiles(mol)
    assert _canon(out) == _canon(smi), f"{smi} -> graph -> {out}"


def test_round_trip_LEDC_seed() -> None:
    """LEDC built directly from seed_species (with explicit bond_orders)
    must serialise to a SMILES that RDKit recognises as LEDC."""
    ledc = make_LEDC()
    out = mol_to_smiles(ledc)
    parsed = Chem.MolFromSmiles(out)
    assert parsed is not None, f"emitted SMILES not parseable: {out!r}"
    # LEDC structure: Li-O-CO-O-CH2-CH2-O-CO-O-Li → C4H4Li2O6 (heavy=12).
    counts: dict[str, int] = {}
    for a in Chem.AddHs(parsed).GetAtoms():
        counts[a.GetSymbol()] = counts.get(a.GetSymbol(), 0) + 1
    assert counts.get("Li", 0) == 2
    assert counts.get("C", 0) == 4
    assert counts.get("O", 0) == 6
    assert counts.get("H", 0) == 4


# ----------------------------------------------------------------------
# Charge/spin placement
# ----------------------------------------------------------------------
def test_bare_lithium_cation_is_plus_one() -> None:
    mol = make_Li_cation()
    smi = mol_to_smiles(mol)
    assert smi == "[Li+]"


def test_hydroxide_round_trip() -> None:
    mol = smiles_to_mol("[OH-]")
    assert mol.charge == -1
    # RDKit may emit either "[OH-]" or "[H][O-]"; canonicalise both sides.
    assert _canon(mol_to_smiles(mol)) == _canon("[OH-]")


def test_methyl_radical_preserves_spin() -> None:
    mol = smiles_to_mol("[CH3]")
    assert mol.spin == 1
    out = mol_to_smiles(mol)
    # RDKit may emit either [CH3] forms; canonicalise.
    assert _canon(out) == _canon("[CH3]")


# ----------------------------------------------------------------------
# Multi-species (dot) SMILES
# ----------------------------------------------------------------------
def test_dot_smiles_concat_and_split() -> None:
    ec = make_EC()
    water = make_H2O()
    li = make_Li_cation()
    s = mols_to_dot_smiles([ec, water, li])
    parts = s.split(".")
    assert len(parts) == 3
    assert "[Li+]" in parts

    fragments = dot_smiles_to_mols(s)
    assert len(fragments) == 3
    formulas = sorted(f.formula for f in fragments)
    assert formulas == sorted([ec.formula, water.formula, li.formula])


def test_invalid_smiles_raises_bridge_error() -> None:
    with pytest.raises(BridgeError):
        smiles_to_mol("not a molecule")
    with pytest.raises(BridgeError):
        dot_smiles_to_mols("$$$")
