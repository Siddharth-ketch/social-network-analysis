"""Centrality measures, PageRank, and clustering metrics for social networks."""

from __future__ import annotations

import math
from collections import deque
from typing import Any

from .graph import SocialGraph


def degree_centrality(graph: SocialGraph) -> dict[Any, float]:
    """Compute normalized degree centrality C_D(v) = deg(v) / (n - 1)."""
    n = graph.number_of_nodes()
    if n <= 1:
        return {node: 0.0 for node in graph.nodes()}
    denom = float(n - 1)
    return {node: graph.degree(node) / denom for node in graph.nodes()}


def closeness_centrality(graph: SocialGraph, wf_improved: bool = True) -> dict[Any, float]:
    """Compute closeness centrality using Wasserman-Faust normalization for disconnected graphs."""
    nodes = graph.nodes()
    n = len(nodes)
    if n <= 1:
        return {node: 0.0 for node in nodes}

    result: dict[Any, float] = {}
    for node in nodes:
        dists = graph.bfs_distances(node)
        reachable = len(dists) - 1
        total_dist = sum(dists.values())
        if total_dist == 0 or reachable == 0:
            result[node] = 0.0
            continue
        base = reachable / float(total_dist)
        if wf_improved and n > 1:
            base *= reachable / float(n - 1)
        result[node] = base
    return result


def harmonic_centrality(graph: SocialGraph, normalized: bool = True) -> dict[Any, float]:
    """Compute harmonic centrality sum_{v != u} 1 / d(u, v), optionally normalized by (n - 1)."""
    nodes = graph.nodes()
    n = len(nodes)
    if n <= 1:
        return {node: 0.0 for node in nodes}

    result: dict[Any, float] = {}
    for node in nodes:
        dists = graph.bfs_distances(node)
        score = sum(1.0 / d for target, d in dists.items() if target != node and d > 0)
        result[node] = score / (n - 1) if normalized else score
    return result


def betweenness_centrality(graph: SocialGraph, normalized: bool = True) -> dict[Any, float]:
    """Compute exact shortest-path betweenness centrality using Brandes' O(VE) algorithm."""
    nodes = graph.nodes()
    cb: dict[Any, float] = {node: 0.0 for node in nodes}
    n = len(nodes)
    if n <= 2:
        return cb

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
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                cb[w] += delta[w]

    if not graph.directed:
        for node in cb:
            cb[node] /= 2.0

    if normalized:
        scale = 1.0 / ((n - 1) * (n - 2)) if graph.directed else 2.0 / ((n - 1) * (n - 2))
        for node in cb:
            cb[node] *= scale
    return cb


def eigenvector_centrality(
    graph: SocialGraph, max_iter: int = 500, tol: float = 1e-6
) -> dict[Any, float]:
    """Compute eigenvector centrality via power iteration with L2 normalization."""
    nodes = graph.nodes()
    n = len(nodes)
    if n == 0:
        return {}
    if graph.number_of_edges() == 0:
        val = 1.0 / math.sqrt(n)
        return {node: val for node in nodes}

    scores = {node: 1.0 / math.sqrt(n) for node in nodes}
    for _ in range(max_iter):
        next_scores = {node: 0.0 for node in nodes}
        for u in nodes:
            # For directed graphs, incoming neighbors contribute prestige; for undirected, symmetric.
            sources = graph.predecessors(u) if graph.directed else graph.neighbors(u)
            next_scores[u] = sum(scores[v] for v in sources) + scores[u] * 1e-9

        norm = math.sqrt(sum(val * val for val in next_scores.values()))
        if norm == 0.0:
            return {node: 0.0 for node in nodes}
        for node in nodes:
            next_scores[node] /= norm

        diff = sum(abs(next_scores[node] - scores[node]) for node in nodes)
        scores = next_scores
        if diff < tol:
            break
    return scores


def pagerank(
    graph: SocialGraph,
    damping: float = 0.85,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> dict[Any, float]:
    """Compute PageRank distribution over graph vertices, handling dangling nodes."""
    nodes = graph.nodes()
    n = len(nodes)
    if n == 0:
        return {}

    rank = {node: 1.0 / n for node in nodes}
    out_deg = {node: graph.out_degree(node) for node in nodes}
    teleport = (1.0 - damping) / n

    for _ in range(max_iter):
        dangling_sum = sum(rank[node] for node in nodes if out_deg[node] == 0)
        dangling_contrib = damping * dangling_sum / n
        next_rank: dict[Any, float] = {}
        diff = 0.0

        for u in nodes:
            incoming = graph.predecessors(u) if graph.directed else graph.neighbors(u)
            inbound_sum = sum(rank[v] / out_deg[v] for v in incoming if out_deg[v] > 0)
            val = teleport + dangling_contrib + damping * inbound_sum
            next_rank[u] = val
            diff += abs(val - rank[u])

        rank = next_rank
        if diff < tol:
            break
    return rank


def local_clustering_coefficient(graph: SocialGraph, node: Any) -> float:
    """Compute local clustering coefficient C(v) = 2 * edges_in_neighborhood / (k * (k - 1))."""
    nbrs = set(graph.all_neighbors(node) if graph.directed else graph.neighbors(node))
    k = len(nbrs)
    if k < 2:
        return 0.0

    nbr_list = list(nbrs)
    links = 0
    for i, u in enumerate(nbr_list):
        for v in nbr_list[i + 1 :]:
            if graph.has_edge(u, v) or (graph.directed and graph.has_edge(v, u)):
                links += 1
    return (2.0 * links) / (k * (k - 1))


def average_clustering_coefficient(graph: SocialGraph) -> float:
    """Compute mean local clustering coefficient across all nodes."""
    nodes = graph.nodes()
    if not nodes:
        return 0.0
    return sum(local_clustering_coefficient(graph, node) for node in nodes) / len(nodes)


def global_transitivity(graph: SocialGraph) -> float:
    """Compute global transitivity = 3 * triangles / connected_triples."""
    triangles = 0
    triples = 0
    for node in graph.nodes():
        nbrs = list(graph.all_neighbors(node) if graph.directed else graph.neighbors(node))
        k = len(nbrs)
        if k < 2:
            continue
        triples += (k * (k - 1)) // 2
        for i, u in enumerate(nbrs):
            for v in nbrs[i + 1 :]:
                if graph.has_edge(u, v) or (graph.directed and graph.has_edge(v, u)):
                    triangles += 1
    if triples == 0:
        return 0.0
    # Each triangle is counted once per vertex (3 times total) in the loop above.
    return triangles / float(triples)
