"""Canonical social network benchmarks and combinatorial random graph generators."""

from __future__ import annotations

import random

from .graph import SocialGraph


def zachary_karate_club() -> SocialGraph:
    """Return Zachary's Karate Club social network (34 nodes, 78 edges).

    Nodes are labeled 1..34 with `faction` attribute ('Mr. Hi' or 'Officer').
    """
    mr_hi = {1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 17, 18, 20, 22}
    edges = [
        (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9),
        (1, 11), (1, 12), (1, 13), (1, 14), (1, 18), (1, 20), (1, 22), (1, 32),
        (2, 3), (2, 4), (2, 8), (2, 14), (2, 18), (2, 20), (2, 22), (2, 31),
        (3, 4), (3, 8), (3, 9), (3, 10), (3, 14), (3, 28), (3, 29), (3, 33),
        (4, 8), (4, 13), (4, 14),
        (5, 7), (5, 11),
        (6, 7), (6, 11), (6, 17),
        (7, 17),
        (9, 31), (9, 33), (9, 34),
        (10, 34),
        (14, 34),
        (15, 33), (15, 34),
        (16, 33), (16, 34),
        (19, 33), (19, 34),
        (20, 34),
        (21, 33), (21, 34),
        (23, 33), (23, 34),
        (24, 26), (24, 28), (24, 30), (24, 33), (24, 34),
        (25, 26), (25, 28), (25, 32),
        (26, 32),
        (27, 30), (27, 34),
        (28, 34),
        (29, 32), (29, 34),
        (30, 33), (30, 34),
        (31, 33), (31, 34),
        (32, 33), (32, 34),
        (33, 34),
    ]
    g = SocialGraph(directed=False)
    for node in range(1, 35):
        g.add_node(node, faction="Mr. Hi" if node in mr_hi else "Officer")
    g.add_edges_from(edges)
    return g


def florentine_families() -> SocialGraph:
    """Return Padgett's Florentine Families marriage network (15 families, 20 edges)."""
    edges = [
        ("Acciaiuoli", "Medici"),
        ("Albizzi", "Ginori"),
        ("Albizzi", "Guadagni"),
        ("Albizzi", "Medici"),
        ("Barbadori", "Castellani"),
        ("Barbadori", "Medici"),
        ("Bischeri", "Guadagni"),
        ("Bischeri", "Peruzzi"),
        ("Bischeri", "Strozzi"),
        ("Castellani", "Peruzzi"),
        ("Castellani", "Strozzi"),
        ("Guadagni", "Lamberteschi"),
        ("Guadagni", "Tornabuoni"),
        ("Medici", "Ridolfi"),
        ("Medici", "Salviati"),
        ("Medici", "Tornabuoni"),
        ("Pazzi", "Salviati"),
        ("Peruzzi", "Strozzi"),
        ("Ridolfi", "Strozzi"),
        ("Ridolfi", "Tornabuoni"),
    ]
    g = SocialGraph(directed=False)
    g.add_edges_from(edges)
    return g


def erdos_renyi_gnp(n: int, p: float, seed: int = 42) -> SocialGraph:
    """Generate an Erdos-Renyi G(n, p) random social network."""
    if n < 0 or not (0.0 <= p <= 1.0):
        raise ValueError("Require n >= 0 and 0 <= p <= 1.")
    rng = random.Random(seed)
    g = SocialGraph(directed=False)
    g.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                g.add_edge(i, j)
    return g


def watts_strogatz(n: int, k: int, beta: float, seed: int = 42) -> SocialGraph:
    """Generate a Watts-Strogatz small-world network (n nodes, k even ring degree, rewiring beta)."""
    if k >= n or k % 2 != 0:
        raise ValueError("k must be an even integer strictly less than n.")
    if not (0.0 <= beta <= 1.0):
        raise ValueError("beta must be in [0, 1].")

    rng = random.Random(seed)
    g = SocialGraph(directed=False)
    g.add_nodes_from(range(n))
    half_k = k // 2

    for u in range(n):
        for step in range(1, half_k + 1):
            v = (u + step) % n
            g.add_edge(u, v)

    for u in range(n):
        for step in range(1, half_k + 1):
            v = (u + step) % n
            if rng.random() < beta:
                candidates = [w for w in range(n) if w != u and not g.has_edge(u, w)]
                if candidates:
                    g.remove_edge(u, v)
                    g.add_edge(u, rng.choice(candidates))
    return g


def barabasi_albert(n: int, m: int, seed: int = 42) -> SocialGraph:
    """Generate a Barabasi-Albert scale-free network via preferential attachment."""
    if m < 1 or m >= n:
        raise ValueError("Require 1 <= m < n.")
    rng = random.Random(seed)
    g = SocialGraph(directed=False)

    # Start with a complete clique K_m
    for i in range(m):
        g.add_node(i)
    for i in range(m):
        for j in range(i + 1, m):
            g.add_edge(i, j)

    repeated_nodes: list[int] = []
    for i in range(m):
        repeated_nodes.extend([i] * (m - 1))

    for new_node in range(m, n):
        g.add_node(new_node)
        targets: set[int] = set()
        while len(targets) < m:
            if repeated_nodes:
                targets.add(rng.choice(repeated_nodes))
            else:
                targets.add(rng.randrange(new_node))
        for target in targets:
            g.add_edge(new_node, target)
            repeated_nodes.append(new_node)
            repeated_nodes.append(target)
    return g


def bipartite_affiliation_sample() -> tuple[SocialGraph, set[str], set[str]]:
    """Return a bipartite user-to-interest-group affiliation graph for matching/Hall checks."""
    users = {"Alice", "Bob", "Carol", "Dave", "Eve"}
    groups = {"GraphTheory", "Combinatorics", "DataViz", "Algorithms", "ML"}
    edges = [
        ("Alice", "GraphTheory"),
        ("Alice", "Combinatorics"),
        ("Bob", "GraphTheory"),
        ("Bob", "Algorithms"),
        ("Carol", "Combinatorics"),
        ("Carol", "DataViz"),
        ("Dave", "Algorithms"),
        ("Dave", "ML"),
        ("Eve", "DataViz"),
        ("Eve", "ML"),
    ]
    g = SocialGraph(directed=False)
    for u in users:
        g.add_node(u, kind="user")
    for grp in groups:
        g.add_node(grp, kind="group")
    g.add_edges_from(edges)
    return g, users, groups
