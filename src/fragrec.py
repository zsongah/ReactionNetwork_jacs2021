"""Fragmentation and recombination procedures (Sec 2.1 of the paper).

Fragmentation: break one edge in a molecular graph; collect the resulting
1- or 2-fragment products. n-step fragmentation iterates this.

Recombination: connect one new edge between two specified atoms across
two molecules (or within one molecule), provided valences remain sane.

A graph-isomorphism dedup is applied at every stage.
"""
from __future__ import annotations

from itertools import combinations, product

import networkx as nx

from .molecule import MoleculeGraph, dedupe, has_valid_valence


# ----------------------------------------------------------------------
# Fragmentation
# ----------------------------------------------------------------------
def one_step_fragment(mol: MoleculeGraph) -> list[MoleculeGraph]:
    """Break each edge in turn; return the resulting molecule(s).

    The paper's "fragmentation" returns one or two fragments per cut.
    Charge is left on the parent for both children for now; downstream
    pathfinding can branch over charge states.
    """
    products: list[MoleculeGraph] = []
    for u, v in list(mol.graph.edges()):
        g = mol.graph.copy()
        g.remove_edge(u, v)
        components = [g.subgraph(c).copy() for c in nx.connected_components(g)]
        # remap node labels to be contiguous within each component
        for comp in components:
            mapping = {old: new for new, old in enumerate(comp.nodes())}
            comp_relabel = nx.relabel_nodes(comp, mapping)
            # spin bumps by 1 for each fragment of a homolytic cut
            products.append(
                MoleculeGraph(
                    comp_relabel,
                    charge=0,  # let recomb stage assign charges
                    spin=(mol.spin + 1) % 2,
                    name=f"frag({mol.name})",
                )
            )
    return [m for m in dedupe(products) if has_valid_valence(m)]


def n_step_fragment(mol: MoleculeGraph, n: int) -> list[MoleculeGraph]:
    """Iterate one_step_fragment n times, collecting all unique fragments."""
    pool: list[MoleculeGraph] = [mol]
    all_frags: list[MoleculeGraph] = []
    for _ in range(n):
        next_pool: list[MoleculeGraph] = []
        for m in pool:
            children = one_step_fragment(m)
            next_pool.extend(children)
        all_frags.extend(next_pool)
        pool = next_pool
    return dedupe(all_frags)


# ----------------------------------------------------------------------
# Recombination
# ----------------------------------------------------------------------
def one_step_recombine_pair(
    a: MoleculeGraph, b: MoleculeGraph
) -> list[MoleculeGraph]:
    """Form one new edge between every (atom_in_a, atom_in_b) pair.

    Returns valence-feasible, deduplicated recombinant molecular graphs.
    """
    out: list[MoleculeGraph] = []
    n_a = a.graph.number_of_nodes()
    # build a disjoint union with b nodes shifted by n_a
    for i, j in product(a.graph.nodes(), b.graph.nodes()):
        g = nx.disjoint_union(a.graph, b.graph)
        g.add_edge(i, j + n_a)
        recomb = MoleculeGraph(
            g,
            charge=a.charge + b.charge,
            spin=(a.spin + b.spin) % 2,
            name=f"recomb({a.name},{b.name})",
        )
        if has_valid_valence(recomb) and nx.is_connected(g):
            out.append(recomb)
    return dedupe(out)


def recombine_pool(species: list[MoleculeGraph]) -> list[MoleculeGraph]:
    """All pairwise one-step recombinations of a species pool.

    Includes self-pairs (a + a) per the paper.
    """
    out: list[MoleculeGraph] = []
    for a, b in combinations(species, 2):
        out.extend(one_step_recombine_pair(a, b))
    for a in species:
        out.extend(one_step_recombine_pair(a, a.copy()))
    return dedupe(out)
