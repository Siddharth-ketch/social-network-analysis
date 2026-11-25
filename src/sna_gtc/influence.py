"""Combinatorial optimization and diffusion models for social influence."""

from __future__ import annotations

import random
from collections import deque
from typing import Any

from .graph import SocialGraph


def greedy_vertex_cover_2approx(graph: SocialGraph) -> set[Any]:
    """Compute a 2-approximate minimum vertex cover by iteratively covering uncovered edges."""
    cover: set[Any] = set()
    for u, v in graph.edges():
        if u not in cover and v not in cover:
            cover.add(u)
            cover.add(v)
    return cover


def greedy_dominating_set(graph: SocialGraph) -> set[Any]:
    """Compute a greedy dominating set D such that every vertex is in D or adjacent to D."""
    nodes = set(graph.nodes())
    dominated: set[Any] = set()
    dom_set: set[Any] = set()
    adj = {
        u: set(graph.all_neighbors(u) if graph.directed else graph.neighbors(u))
        for u in nodes
    }

    while len(dominated) < len(nodes):
        best_node = max(
            nodes - dom_set,
            key=lambda u: len(({u} | adj[u]) - dominated),
        )
        dom_set.add(best_node)
        dominated |= {best_node} | adj[best_node]
    return dom_set


def independent_cascade(
    graph: SocialGraph,
    seeds: list[Any] | set[Any],
    prob: float = 0.2,
    simulations: int = 100,
    seed: int = 42,
) -> float:
    """Estimate expected influence spread under the Independent Cascade (IC) model."""
    if not seeds:
        return 0.0
    if not (0.0 <= prob <= 1.0):
        raise ValueError("Activation probability must be in [0, 1].")
    if simulations <= 0:
        raise ValueError("Number of simulations must be positive.")

    rng = random.Random(seed)
    total_activated = 0
    seed_set = set(seeds)

    for _ in range(simulations):
        activated = set(seed_set)
        frontier = deque(seed_set)
        while frontier:
            u = frontier.popleft()
            for v in graph.neighbors(u):
                if v not in activated and rng.random() <= prob:
                    activated.add(v)
                    frontier.append(v)
        total_activated += len(activated)

    return total_activated / float(simulations)


def greedy_influence_maximization(
    graph: SocialGraph,
    k: int,
    prob: float = 0.2,
    simulations: int = 50,
    seed: int = 42,
) -> tuple[list[Any], float]:
    """Select k seed nodes maximizing expected IC spread via submodular greedy hill-climbing."""
    if k < 0:
        raise ValueError("k must be non-negative.")
    nodes = graph.nodes()
    k = min(k, len(nodes))
    selected: list[Any] = []
    remaining = set(nodes)
    current_spread = 0.0

    for step in range(k):
        best_node = None
        best_spread = -1.0
        for candidate in remaining:
            spread = independent_cascade(
                graph,
                selected + [candidate],
                prob=prob,
                simulations=simulations,
                seed=seed + step,
            )
            if spread > best_spread:
                best_spread = spread
                best_node = candidate
        if best_node is None:
            break
        selected.append(best_node)
        remaining.remove(best_node)
        current_spread = best_spread

    return selected, current_spread
