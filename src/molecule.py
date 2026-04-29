"""Molecular graph representation.

Per Sec 2.1 of the paper: molecules are treated as graphs where atoms are
nodes (labelled with element + charge) and bonds are edges. Two molecules
are considered the same iff their graphs are isomorphic with matching
node labels.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable

import networkx as nx
from networkx.algorithms import isomorphism


def _node_match(a: dict, b: dict) -> bool:
    return a.get("element") == b.get("element")


@dataclass
class MoleculeGraph:
    """Light wrapper around a nx.Graph with element-labelled nodes.

    Attributes
    ----------
    graph : nx.Graph
        Atoms are integer node ids; each node has an ``element`` attribute.
    charge : int
        Net molecular charge. Tracked separately from the graph.
    spin : int
        Number of unpaired electrons (0 = singlet, 1 = doublet/radical).
    name : str
        Optional human-readable label used for diagnostics.
    """

    graph: nx.Graph
    charge: int = 0
    spin: int = 0
    name: str = ""

    # ------------------------------------------------------------------
    # constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_atoms_bonds(
        cls,
        atoms: list[str],
        bonds: list[tuple[int, int]],
        charge: int = 0,
        spin: int = 0,
        name: str = "",
        bond_orders: list[int] | None = None,
    ) -> "MoleculeGraph":
        g = nx.Graph()
        for i, el in enumerate(atoms):
            g.add_node(i, element=el)
        if bond_orders is None:
            g.add_edges_from(bonds)
        else:
            if len(bond_orders) != len(bonds):
                raise ValueError("bond_orders length must match bonds")
            for (u, v), o in zip(bonds, bond_orders):
                g.add_edge(u, v, order=int(o))
        return cls(g, charge=charge, spin=spin, name=name)

    # ------------------------------------------------------------------
    # identity
    # ------------------------------------------------------------------
    def is_isomorphic(self, other: "MoleculeGraph") -> bool:
        if self.charge != other.charge or self.spin != other.spin:
            return False
        if self.formula != other.formula:
            return False
        gm = isomorphism.GraphMatcher(self.graph, other.graph, node_match=_node_match)
        return gm.is_isomorphic()

    def canonical_hash(self) -> str:
        """A reasonably fast, isomorphism-aware fingerprint.

        Uses the Weisfeiler-Lehman hash from NetworkX, augmented with
        charge and spin so charge states are kept distinct.
        """
        wl = nx.weisfeiler_lehman_graph_hash(
            self.graph, node_attr="element", iterations=4
        )
        payload = f"{wl}|q={self.charge}|s={self.spin}"
        return hashlib.sha1(payload.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # bookkeeping helpers
    # ------------------------------------------------------------------
    @property
    def formula(self) -> str:
        from collections import Counter

        c = Counter(d["element"] for _, d in self.graph.nodes(data=True))
        return "".join(f"{el}{n if n > 1 else ''}" for el, n in sorted(c.items()))

    @property
    def n_atoms(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def n_bonds(self) -> int:
        return self.graph.number_of_edges()

    def copy(self) -> "MoleculeGraph":
        return MoleculeGraph(
            self.graph.copy(), charge=self.charge, spin=self.spin, name=self.name
        )

    def __repr__(self) -> str:
        tag = self.name or self.formula
        q = f"{'+' if self.charge > 0 else ''}{self.charge}" if self.charge else ""
        return f"<Mol {tag}{q} s={self.spin}>"


# ----------------------------------------------------------------------
# valence sanity check (Sec 2.1 step 2: discard absurd valences)
# ----------------------------------------------------------------------
MAX_VALENCE = {"H": 1, "Li": 1, "C": 4, "N": 3, "O": 2, "F": 1}


def has_valid_valence(mol: MoleculeGraph) -> bool:
    for n, data in mol.graph.nodes(data=True):
        el = data["element"]
        if el not in MAX_VALENCE:
            continue
        deg = mol.graph.degree(n)
        if deg > MAX_VALENCE[el]:
            return False
    return True


def dedupe(mols: Iterable[MoleculeGraph]) -> list[MoleculeGraph]:
    """Remove graph-isomorphic duplicates (with charge/spin distinguishing)."""
    seen: dict[str, MoleculeGraph] = {}
    for m in mols:
        h = m.canonical_hash()
        if h not in seen:
            seen[h] = m
    return list(seen.values())
