"""Community detection, modularity, and Burt's structural holes."""

from __future__ import annotations

import random
from collections import Counter, deque
from typing import Any

from .graph import SocialGraph


def modularity(graph: SocialGraph, communities: list[set[Any]]) -> float:
    """Compute Newman-Girvan modularity Q for a partition of vertices."""
    m = graph.number_of_edges()
    if m == 0:
        return 0.0

    two_m = 2.0 * m
    node_to_comm: dict[Any, int] = {}
    for cid, comm in enumerate(communities):
        for node in comm:
            node_to_comm[node] = cid

    nodes = graph.nodes()
    deg = {u: len(graph.all_neighbors(u) if graph.directed else graph.neighbors(u)) for u in nodes}
    q = 0.0
    for u in nodes:
        cu = node_to_comm.get(u)
        if cu is None:
            continue
        for v in nodes:
            if node_to_comm.get(v) != cu:
                continue
            a_uv = 1.0 if (graph.has_edge(u, v) or (graph.directed and graph.has_edge(v, u))) else 0.0
            q += a_uv - (deg[u] * deg[v]) / two_m
    return q / two_m


def _edge_betweenness(graph: SocialGraph) -> dict[tuple[Any, Any], float]:
    """Compute undirected edge betweenness centrality via Brandes' algorithm."""
    nodes = graph.nodes()
    eb: dict[tuple[Any, Any], float] = {}
    for u, v in graph.edges():
        key = (u, v) if str(u) <= str(v) else (v, u)
        eb[key] = 0.0

    for s in nodes:
        stack: list[Any] = []
        preds: dict[Any, list[Any]] = {v: [] for v in nodes}
        sigma: dict[Any, float] = {v: 0.0 for v in nodes}
        dist: dict[Any, int] = {v: -1 for v in nodes}
        sigma[s] = 1.0
        dist[s] = 0
        queue: deque[Any] = deque([s])

        while queue:
            v = queue.popleft()
            stack.append(v)
            for w in graph.neighbors(v):
                if dist[w] < 0:
                    queue.append(w)
                    dist[w] = dist[v] + 1
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    preds[w].append(v)

        delta: dict[Any, float] = {v: 0.0 for v in nodes}
        while stack:
            w = stack.pop()
            for v in preds[w]:
                if sigma[w] > 0:
                    c = (sigma[v] / sigma[w]) * (1.0 + delta[w])
                    key = (v, w) if str(v) <= str(w) else (w, v)
                    if key in eb:
                        eb[key] += c
                    delta[v] += c

    for key in eb:
        eb[key] /= 2.0
    return eb


def girvan_newman(
    graph: SocialGraph, max_levels: int | None = None
) -> list[tuple[list[set[Any]], float]]:
    """Run divisive Girvan-Newman community detection.

    Returns a list of (communities, modularity_Q) across hierarchical splits.
    """
    working = SocialGraph(directed=False)
    for node in graph.nodes():
        working.add_node(node)
    for u, v in graph.edges():
        working.add_edge(u, v)

    results: list[tuple[list[set[Any]], float]] = []
    initial = working.connected_components()
    results.append((initial, modularity(graph, initial)))

    prev_count = len(initial)
    levels = 0
    while working.number_of_edges() > 0:
        if max_levels is not None and levels >= max_levels:
            break
        eb = _edge_betweenness(working)
        if not eb:
            break
        max_eb = max(eb.values())
        for (u, v), val in eb.items():
            if val >= max_eb - 1e-12:
                working.remove_edge(u, v)

        comps = working.connected_components()
        if len(comps) > prev_count:
            results.append((comps, modularity(graph, comps)))
            prev_count = len(comps)
            levels += 1
    return results


def best_girvan_newman_partition(
    graph: SocialGraph, max_levels: int | None = 10
) -> tuple[list[set[Any]], float]:
    """Return the Girvan-Newman partition maximizing Newman-Girvan modularity Q."""
    levels = girvan_newman(graph, max_levels=max_levels)
    return max(levels, key=lambda item: item[1])


def label_propagation_communities(
    graph: SocialGraph, seed: int = 42, max_iter: int = 100
) -> list[set[Any]]:
    """Detect communities using asynchronous Label Propagation."""
    rng = random.Random(seed)
    nodes = graph.nodes()
    labels = {node: idx for idx, node in enumerate(nodes)}

    for _ in range(max_iter):
        changed = False
        order = list(nodes)
        rng.shuffle(order)
        for u in order:
            nbrs = list(graph.all_neighbors(u) if graph.directed else graph.neighbors(u))
            if not nbrs:
                continue
            counts = Counter(labels[v] for v in nbrs)
            max_freq = max(counts.values())
            best_labels = [lbl for lbl, cnt in counts.items() if cnt == max_freq]
            chosen = rng.choice(best_labels)
            if chosen != labels[u]:
                labels[u] = chosen
                changed = True
        if not changed:
            break

    groups: dict[int, set[Any]] = {}
    for node, lbl in labels.items():
        groups.setdefault(lbl, set()).add(node)
    communities = list(groups.values())
    communities.sort(key=len, reverse=True)
    return communities


def burt_structural_holes(graph: SocialGraph) -> dict[Any, dict[str, float]]:
    """Compute Burt's structural hole metrics (effective_size, constraint) per node."""
    nodes = graph.nodes()
    adj = {
        u: set(graph.all_neighbors(u) if graph.directed else graph.neighbors(u))
        for u in nodes
    }
    metrics: dict[Any, dict[str, float]] = {}

    for i in nodes:
        nbrs = adj[i]
        deg = len(nbrs)
        if deg == 0:
            metrics[i] = {"effective_size": 0.0, "constraint": 0.0}
            continue

        # Unweighted symmetric tie proportions p_ij = 1 / deg(i)
        p_i = 1.0 / deg
        redundancy = 0.0
        constraint = 0.0

        for j in nbrs:
            # Redundancy between i and j through shared neighbors q in N(i) intersect N(j)
            shared = nbrs & adj[j]
            redundancy_j = sum(p_i * (1.0 / len(adj[q])) if len(adj[q]) > 0 else 0.0 for q in shared)
            # Effective size contribution: 1 - sum_q p_iq * m_jq (here m_jq = 1 for unweighted)
            # Standard unweighted effective size simplifies to deg - (2 * edges_among_nbrs) / deg
            redundancy += len(shared) / deg
            indirect = sum(p_i * p_i for _ in shared)
            c_ij = (p_i + indirect) ** 2
            constraint += c_ij

        effective_size = float(deg) - redundancy
        metrics[i] = {
            "effective_size": effective_size,
            "constraint": constraint,
        }
    return metrics
