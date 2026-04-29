"""Seed molecules for the LEDC reproduction.

Connectivity only (no 3D geometry). The paper's reaction-network step is
graph-only, so this is sufficient. Atoms are listed in a stable order so
edge tuples are easy to read.

EC (ethylene carbonate)         C3H4O3, cyclic
H2O                             H2O
Li+                             single Li atom, +1 charge
LEDC (lithium ethylene
       dicarbonate)              (CH2OC(O)OLi)2 — two carbonate ends
                                  bridged by ethylene
"""
from __future__ import annotations

from src.molecule import MoleculeGraph


def make_EC() -> MoleculeGraph:
    # Atom indices:
    #   0: C carbonyl
    #   1: O carbonyl (=O)
    #   2: O ring  (-O-)
    #   3: O ring  (-O-)
    #   4: C methylene
    #   5: C methylene
    #   6-9: H on methylenes
    atoms = ["C", "O", "O", "O", "C", "C", "H", "H", "H", "H"]
    bonds = [
        (0, 1),  # C=O carbonyl
        (0, 2), (0, 3),  # C-O ring
        (2, 4), (3, 5),  # O-CH2
        (4, 5),          # CH2-CH2
        (4, 6), (4, 7), (5, 8), (5, 9),  # C-H
    ]
    bond_orders = [
        2,
        1, 1,
        1, 1,
        1,
        1, 1, 1, 1,
    ]
    return MoleculeGraph.from_atoms_bonds(
        atoms, bonds, charge=0, name="EC", bond_orders=bond_orders
    )


def make_H2O() -> MoleculeGraph:
    return MoleculeGraph.from_atoms_bonds(
        ["O", "H", "H"], [(0, 1), (0, 2)], charge=0, name="H2O"
    )


def make_Li_cation() -> MoleculeGraph:
    return MoleculeGraph.from_atoms_bonds(["Li"], [], charge=1, name="Li+")


def make_LEDC() -> MoleculeGraph:
    """Linear LEDC: Li-O-C(=O)-O-CH2-CH2-O-C(=O)-O-Li.

    The paper notes (Sec 3.2) that the linear conformer is what the
    reaction network "sees"; conformational refinement happens later in
    DFT. We mirror that here.
    """
    atoms = [
        "Li",  # 0
        "O",   # 1  Li-O
        "C",   # 2  carbonate C
        "O",   # 3  =O
        "O",   # 4  -O-CH2
        "C",   # 5  CH2
        "C",   # 6  CH2
        "O",   # 7  CH2-O-
        "C",   # 8  carbonate C
        "O",   # 9  =O
        "O",   # 10 O-Li
        "Li",  # 11
        "H", "H", "H", "H",  # 12-15 on methylenes
    ]
    bonds = [
        (0, 1), (1, 2),
        (2, 3), (2, 4),
        (4, 5), (5, 6), (5, 12), (5, 13),
        (6, 7), (6, 14), (6, 15),
        (7, 8), (8, 9), (8, 10), (10, 11),
    ]
    bond_orders = [
        1, 1,
        2, 1,         # C=O carbonyl, C-O ring
        1, 1, 1, 1,
        1, 1, 1,
        1, 2, 1, 1,   # O-C, C=O carbonyl, C-O, O-Li
    ]
    return MoleculeGraph.from_atoms_bonds(
        atoms, bonds, charge=0, name="LEDC", bond_orders=bond_orders
    )


SEEDS = {
    "EC": make_EC(),
    "H2O": make_H2O(),
    "Li+": make_Li_cation(),
}

TARGET = make_LEDC()
