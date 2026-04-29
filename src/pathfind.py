"""Pathfinding on the reaction network (Sec 2.2 of the paper).

The paper builds a directed bipartite graph (species nodes ↔ reaction
nodes) and uses Dijkstra's algorithm with a softplus cost on reaction
free energies::

    cost(rxn) = ln(1 + exp(ΔG / kT_eff))

This is differentiable, monotonic in ΔG, and tolerates slightly
endergonic reactions (cost ~ ln 2 at ΔG = 0). For pathfinding to a
target compound the search runs in a "grand canonical" ensemble: any
species in the starting pool can be consumed without cost, all others
must be produced en route.

Here we provide:
    - softplus cost
    - Dijkstra single-shortest-path (target-oriented)
    - Yen's K-shortest loopless paths
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import networkx as nx

from .thermo import Reaction, reaction_cost


def softplus_cost(dG: float, scale: float = 0.026) -> float:
    """ln(1 + exp(ΔG / scale)).

    ``scale`` ~ kT at room temperature in eV; the paper uses an effective
    scale that makes mildly exergonic reactions ~ free and strongly
    endergonic reactions exponentially expensive.

    Kept for backwards compatibility; new code should use
    :func:`thermo.reaction_cost` which can also include the FlowER
    mechanistic prior.
    """
    x = dG / scale
    # numerically stable
    if x > 30:
        return x
    if x < -30:
        return math.exp(x)
    return math.log1p(math.exp(x))


# ----------------------------------------------------------------------
# Bipartite graph: species nodes + reaction nodes
# ----------------------------------------------------------------------
def build_pathfinding_graph(
    reactions: list[Reaction],
    starting_pool: set[str],
    scale: float = 0.5,
    *,
    lam: float = 0.0,
    p_floor: float = 0.01,
) -> nx.DiGraph:
    """Construct the directed graph used by Dijkstra.

    Edges:
        starting species --(0)--> reaction node, for every reaction whose
            *non-starting* reactants are also resolvable
        reaction node --(reaction_cost(rxn))--> each product species
        species --(0)--> reaction (when species used as reactant)

    Edge weights come from :func:`thermo.reaction_cost`. With ``lam=0``
    this is the paper's ``softplus(ΔG)`` exactly; with ``lam>0`` and
    reactions carrying ``p_flower`` it discounts mechanistically
    well-supported steps.
    """
    G = nx.DiGraph()
    for i, rxn in enumerate(reactions):
        rnode = f"R{i}"
        G.add_node(rnode, kind="rxn", dG=rxn.dG, rxn=rxn)
        cost = reaction_cost(rxn, scale=scale, lam=lam, p_floor=p_floor)
        for r in rxn.reactants:
            G.add_node(r, kind="species")
            G.add_edge(r, rnode, weight=0.0)
        for p in rxn.products:
            G.add_node(p, kind="species")
            G.add_edge(rnode, p, weight=cost)

    # Virtual super-source connected to the starting pool with 0 cost.
    G.add_node("SOURCE", kind="source")
    for s in starting_pool:
        if s in G:
            G.add_edge("SOURCE", s, weight=0.0)
    return G


# ----------------------------------------------------------------------
# Path search
# ----------------------------------------------------------------------
@dataclass
class PathResult:
    cost: float
    species_sequence: list[str]
    reactions: list[Reaction]

    def pretty(self, name_of: dict[str, str]) -> str:
        lines = [f"total cost = {self.cost:.3f}"]
        for rxn in self.reactions:
            lhs = " + ".join(name_of.get(h, h[:8]) for h in rxn.reactants)
            rhs = " + ".join(name_of.get(h, h[:8]) for h in rxn.products)
            ne = f"  (+ {rxn.n_electrons} e⁻)" if rxn.n_electrons else ""
            lines.append(f"  {lhs}  ->  {rhs}{ne}    ΔG = {rxn.dG:+.2f} eV")
        return "\n".join(lines)


def shortest_path_to(graph: nx.DiGraph, target: str) -> PathResult | None:
    if target not in graph or not nx.has_path(graph, "SOURCE", target):
        return None
    cost, nodes = nx.single_source_dijkstra(graph, "SOURCE", target=target)
    rxns = [graph.nodes[n]["rxn"] for n in nodes if graph.nodes[n].get("kind") == "rxn"]
    species = [n for n in nodes if graph.nodes[n].get("kind") == "species"]
    return PathResult(cost=cost, species_sequence=species, reactions=rxns)


def k_shortest_paths(
    graph: nx.DiGraph, target: str, k: int = 5
) -> list[PathResult]:
    """Yen-style K-shortest paths via NetworkX's simple-path generator."""
    if target not in graph:
        return []
    gen = nx.shortest_simple_paths(graph, "SOURCE", target, weight="weight")
    out: list[PathResult] = []
    for i, path in enumerate(gen):
        if i >= k:
            break
        cost = sum(
            graph[u][v]["weight"] for u, v in zip(path[:-1], path[1:])
        )
        rxns = [graph.nodes[n]["rxn"] for n in path if graph.nodes[n].get("kind") == "rxn"]
        species = [n for n in path if graph.nodes[n].get("kind") == "species"]
        out.append(PathResult(cost=cost, species_sequence=species, reactions=rxns))
    return out
