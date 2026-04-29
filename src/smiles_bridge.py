"""SMILES <-> MoleculeGraph bridge.

Used to exchange molecules with external models (notably FlowER) that
speak SMILES. The bridge is intentionally lossy with respect to bond
orders: our internal MoleculeGraph stores only a connectivity skeleton
(matching the paper's Sec 2.1 graph definition). To recover bond orders
we round-trip through RDKit, which performs valence/aromaticity
perception.

Round-trip caveat
-----------------
``mol_to_smiles(smiles_to_mol(smi)) == smi`` is *not* guaranteed for
arbitrary SMILES, because we collapse to a skeleton in the middle. It
*is* designed to be safe for:

  * the seed molecules in this project (EC, H2O, Li+, LEDC, ...)
  * organic radicals/anions/cations RDKit can sanitize cleanly

For pathological cases the caller will see ``BridgeError``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rdkit import Chem

from .molecule import MoleculeGraph


class BridgeError(ValueError):
    """Raised when a SMILES <-> MoleculeGraph conversion fails."""


# ----------------------------------------------------------------------
# MoleculeGraph -> SMILES
# ----------------------------------------------------------------------
def mol_to_smiles(mol: MoleculeGraph, *, kekulize: bool = False) -> str:
    """Convert a MoleculeGraph to an RDKit-perceived SMILES string.

    Strategy: build an RWMol with all bonds as SINGLE, then let RDKit's
    sanitizer infer aromaticity / multi-bond orders from element-level
    valence rules and the explicit charge/spin we set.
    """
    rw = Chem.RWMol()
    idx_map: dict[int, int] = {}
    for n, data in mol.graph.nodes(data=True):
        a = Chem.Atom(data["element"])
        a.SetNoImplicit(False)  # let RDKit add Hs to satisfy valence
        idx_map[n] = rw.AddAtom(a)

    for u, v, data in mol.graph.edges(data=True):
        order = data.get("order", 1) if isinstance(data, dict) else 1
        bt = {
            1: Chem.BondType.SINGLE,
            2: Chem.BondType.DOUBLE,
            3: Chem.BondType.TRIPLE,
        }.get(order, Chem.BondType.SINGLE)
        rw.AddBond(idx_map[u], idx_map[v], bt)

    # Distribute the molecular charge: place it on the most electronegative
    # atom that can accept it. For the simple ions we deal with (Li+, OH-,
    # CO3^2-) this is good enough; for everything else we fall back to atom 0.
    if mol.charge != 0:
        target = _pick_charge_carrier(mol, sign=1 if mol.charge > 0 else -1)
        rw.GetAtomWithIdx(idx_map[target]).SetFormalCharge(mol.charge)

    # Distribute unpaired electrons (spin) onto a heavy atom carrier. The
    # placement is heuristic; for our small radicals (•CH3, •OR) it is
    # adequate, and downstream consumers care only that the radical count
    # round-trips.
    if mol.spin > 0:
        # Prefer C, then O, then any heavy atom that is not Li.
        elems = {n: d["element"] for n, d in mol.graph.nodes(data=True)}
        carrier = None
        for pref in ("C", "O", "N"):
            for n, e in elems.items():
                if e == pref:
                    carrier = n
                    break
            if carrier is not None:
                break
        if carrier is None:
            for n, e in elems.items():
                if e not in ("H", "Li"):
                    carrier = n
                    break
        if carrier is not None:
            rw.GetAtomWithIdx(idx_map[carrier]).SetNumRadicalElectrons(mol.spin)

    rdmol = rw.GetMol()
    try:
        Chem.SanitizeMol(
            rdmol,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
            ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
        )
    except Exception as e:
        # Try once more with our skeleton hint: drop aromaticity perception.
        try:
            rdmol = rw.GetMol()
            Chem.SanitizeMol(
                rdmol,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_FINDRADICALS
                | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
                | Chem.SanitizeFlags.SANITIZE_ADJUSTHS,
            )
        except Exception:
            raise BridgeError(f"RDKit could not sanitize MoleculeGraph: {e}")

    return Chem.MolToSmiles(rdmol, kekuleSmiles=kekulize, canonical=True)


def _pick_charge_carrier(mol: MoleculeGraph, sign: int) -> int:
    """Pick an atom likely to carry the formal charge."""
    # Prefer Li for +; O/N for -. Fall back to first atom.
    elems = {n: d["element"] for n, d in mol.graph.nodes(data=True)}
    if sign > 0:
        for n, e in elems.items():
            if e == "Li":
                return n
    else:
        for pref in ("O", "N", "F", "C"):
            for n, e in elems.items():
                if e == pref:
                    return n
    return next(iter(elems))


# ----------------------------------------------------------------------
# SMILES -> MoleculeGraph
# ----------------------------------------------------------------------
def smiles_to_mol(
    smiles: str,
    *,
    name: Optional[str] = None,
    keep_bond_orders: bool = True,
) -> MoleculeGraph:
    """Parse SMILES into a heavy-atom MoleculeGraph.

    Hydrogens are made *explicit* before graph construction so that the
    skeleton matches the convention used elsewhere in this project
    (heavy + explicit H atoms as nodes).

    Parameters
    ----------
    keep_bond_orders : bool
        If True, store the perceived bond order on each edge as ``order``
        attribute (1/2/3, aromatic mapped to 1 with a flag). Internal
        fragrec ignores this attribute, but the bridge uses it on the
        return trip to recover multi-bond products.
    """
    rdmol = Chem.MolFromSmiles(smiles)
    if rdmol is None:
        raise BridgeError(f"Invalid SMILES: {smiles!r}")
    rdmol = Chem.AddHs(rdmol)

    # Force kekulization so we get integer bond orders, not aromatic.
    try:
        Chem.Kekulize(rdmol, clearAromaticFlags=True)
    except Exception:
        # Some radicals/ions cannot be kekulized; keep aromatic and map below.
        pass

    import networkx as nx

    g = nx.Graph()
    for atom in rdmol.GetAtoms():
        g.add_node(atom.GetIdx(), element=atom.GetSymbol())

    for bond in rdmol.GetBonds():
        order_f = bond.GetBondTypeAsDouble()  # 1.0/1.5/2.0/3.0
        order = int(round(order_f)) if order_f >= 1.0 else 1
        if keep_bond_orders:
            g.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), order=order)
        else:
            g.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    charge = Chem.GetFormalCharge(rdmol)
    n_radicals = sum(a.GetNumRadicalElectrons() for a in rdmol.GetAtoms())

    return MoleculeGraph(graph=g, charge=charge, spin=n_radicals, name=name or "")


# ----------------------------------------------------------------------
# Composition utilities (multiple species in one SMILES)
# ----------------------------------------------------------------------
def mols_to_dot_smiles(mols: list[MoleculeGraph]) -> str:
    """Join multiple MoleculeGraphs into the dot-separated SMILES used by
    FlowER beam_predict (``A.B.C``)."""
    return ".".join(mol_to_smiles(m) for m in mols)


def dot_smiles_to_mols(smi: str) -> list[MoleculeGraph]:
    """Inverse of mols_to_dot_smiles. Splits on '.' at the SMILES level
    via RDKit (NOT a naive string split, which would fail inside ring
    notation if any)."""
    rdmol = Chem.MolFromSmiles(smi)
    if rdmol is None:
        raise BridgeError(f"Invalid combined SMILES: {smi!r}")
    frags = Chem.GetMolFrags(rdmol, asMols=True)
    return [smiles_to_mol(Chem.MolToSmiles(f)) for f in frags]


@dataclass(frozen=True)
class ReactionLine:
    """Single line of FlowER beam_predict input."""

    reactant_smiles: str
    product_smiles: tuple[str, ...] = ()

    def render(self) -> str:
        if self.product_smiles:
            prods = "|".join(self.product_smiles)
            return f"{self.reactant_smiles}>>{prods}"
        return self.reactant_smiles
