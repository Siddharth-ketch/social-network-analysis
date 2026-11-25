"""Combinatorial algorithms and extremal bounds for Social Network Analysis."""

from __future__ import annotations

import math
from collections import deque
from typing import Any

from .graph import SocialGraph


def _undirected_adj(graph: SocialGraph) -> dict[Any, set[Any]]:
    """Build a symmetric neighbor mapping for combinatorial structure routines."""
    return {
        node: set(graph.all_neighbors(node) if graph.directed else graph.neighbors(node))
        for node in graph.nodes()
    }


def count_triangles_and_wedges(graph: SocialGraph) -> dict[str, Any]:
    """Count total triangles (K_3), open wedges (2-paths), and per-node triangles.

    Uses node ordering by degree to count triangles efficiently in O(m^{3/2}) worst-case.
    """
    adj = _undirected_adj(graph)
    nodes = sorted(adj.keys(), key=lambda u: (len(adj[u]), str(u)))
    rank = {u: idx for idx, u in enumerate(nodes)}

    per_node = {u: 0 for u in nodes}
    total_triangles = 0

    for u in nodes:
        higher_nbrs = [v for v in adj[u] if rank[v] > rank[u]]
        for i, v in enumerate(higher_nbrs):
            for w in higher_nbrs[i + 1 :]:
                if w in adj[v]:
                    total_triangles += 1
                    per_node[u] += 1
                    per_node[v] += 1
                    per_node[w] += 1

    connected_triples = sum((len(adj[u]) * (len(adj[u]) - 1)) // 2 for u in nodes)
    open_wedges = connected_triples - 3 * total_triangles
    return {
        "triangles": total_triangles,
        "connected_triples": connected_triples,
        "open_wedges": open_wedges,
        "per_node_triangles": per_node,
    }


def triadic_census(graph: SocialGraph) -> dict[str, int]:
    """Compute undirected triadic census over all C(n, 3) vertex triples.

    Returns counts for 0-edge (empty), 1-edge (single_pair), 2-edge (open_triad),
    and 3-edge (closed_triangle) subgraphs, whose sum equals C(n, 3).
    """
    n = graph.number_of_nodes()
    total_triples = math.comb(n, 3)
    if n < 3:
        return {
            "total_triples": total_triples,
            "empty_0_edge": 0,
            "single_1_edge": 0,
            "open_2_edge": 0,
            "closed_3_edge": 0,
        }

    tri_info = count_triangles_and_wedges(graph)
    closed_3 = tri_info["triangles"]
    open_2 = tri_info["open_wedges"]

    adj = _undirected_adj(graph)
    m = sum(len(nbrs) for nbrs in adj.values()) // 2
    # Each edge participates in (n - 2) triples.
    # Summing edge participations counts 1-edge triads once, 2-edge twice, 3-edge thrice.
    single_1 = m * (n - 2) - 2 * open_2 - 3 * closed_3
    empty_0 = total_triples - single_1 - open_2 - closed_3
    return {
        "total_triples": total_triples,
        "empty_0_edge": empty_0,
        "single_1_edge": single_1,
        "open_2_edge": open_2,
        "closed_3_edge": closed_3,
    }


def find_maximal_cliques(graph: SocialGraph) -> list[set[Any]]:
    """Enumerate all maximal cliques using Bron-Kerbosch with Tomita pivoting."""
    adj = _undirected_adj(graph)
    cliques: list[set[Any]] = []

    def bron_kerbosch(r: set[Any], p: set[Any], x: set[Any]) -> None:
        if not p and not x:
            if r:
                cliques.append(set(r))
            return
        # Choose pivot u in P union X maximizing |P intersect N(u)|.
        union_px = p | x
        pivot = max(union_px, key=lambda u: len(p & adj[u]))
        candidates = p - adj[pivot]
        for v in list(candidates):
            bron_kerbosch(r | {v}, p & adj[v], x & adj[v])
            p.remove(v)
            x.add(v)

    bron_kerbosch(set(), set(adj.keys()), set())
    cliques.sort(key=len, reverse=True)
    return cliques


def clique_number(graph: SocialGraph) -> int:
    """Return clique number omega(G), the size of the largest clique."""
    cliques = find_maximal_cliques(graph)
    return max((len(c) for c in cliques), default=0)


def k_core_decomposition(graph: SocialGraph) -> dict[Any, int]:
    """Compute core number for every vertex via Batagelj-Zaversnik peeling."""
    adj = _undirected_adj(graph)
    deg = {u: len(nbrs) for u, nbrs in adj.items()}
    core: dict[Any, int] = {}
    remaining = set(adj.keys())

    while remaining:
        min_node = min(remaining, key=lambda u: deg[u])
        current_core = deg[min_node]
        # Peel all vertices whose current degree <= current_core
        queue = deque([u for u in remaining if deg[u] <= current_core])
        while queue:
            u = queue.popleft()
            if u not in remaining:
                continue
            remaining.remove(u)
            core[u] = current_core
            for v in adj[u]:
                if v in remaining:
                    deg[v] -= 1
                    if deg[v] <= current_core:
                        queue.append(v)
    return core


def greedy_coloring(graph: SocialGraph, strategy: str = "largest_first") -> dict[Any, int]:
    """Color graph vertices greedily so adjacent actors receive distinct colors.

    Provides a constructive upper bound chi(G) <= colors_used <= Delta(G) + 1.
    """
    adj = _undirected_adj(graph)
    if strategy == "largest_first":
        order = sorted(adj.keys(), key=lambda u: (-len(adj[u]), str(u)))
    elif strategy == "smallest_last":
        rem = {u: set(nbrs) for u, nbrs in adj.items()}
        order_rev: list[Any] = []
        while rem:
            u = min(rem.keys(), key=lambda x: (len(rem[x]), str(x)))
            order_rev.append(u)
            for v in rem[u]:
                rem[v].discard(u)
            del rem[u]
        order = list(reversed(order_rev))
    else:
        raise ValueError(f"Unknown coloring strategy: {strategy!r}")

    coloring: dict[Any, int] = {}
    for u in order:
        used = {coloring[v] for v in adj[u] if v in coloring}
        color = 0
        while color in used:
            color += 1
        coloring[u] = color
    return coloring


def havel_hakimi_is_graphical(degree_seq: list[int]) -> bool:
    """Test whether a degree sequence is realizable as a simple undirected graph."""
    if not degree_seq:
        return True
    if any(d < 0 or d >= len(degree_seq) for d in degree_seq):
        return False
    if sum(degree_seq) % 2 != 0:
        return False

    seq = sorted((d for d in degree_seq if d > 0), reverse=True)
    while seq:
        d = seq.pop(0)
        if d > len(seq):
            return False
        for i in range(d):
            seq[i] -= 1
            if seq[i] < 0:
                return False
        seq = sorted((x for x in seq if x > 0), reverse=True)
    return True


def _mat_mul(a: list[list[int]], b: list[list[int]]) -> list[list[int]]:
    n = len(a)
    res = [[0] * n for _ in range(n)]
    for i in range(n):
        row_a = a[i]
        row_res = res[i]
        for k, aik in enumerate(row_a):
            if aik == 0:
                continue
            row_b = b[k]
            for j in range(n):
                row_res[j] += aik * row_b[j]
    return res


def count_walks_of_length(graph: SocialGraph, k: int) -> tuple[list[Any], list[list[int]]]:
    """Compute exact number of walks of length k between all pairs via A^k."""
    if k < 0:
        raise ValueError("Walk length k must be non-negative.")
    nodes, float_adj = graph.adjacency_matrix(weighted=False)
    n = len(nodes)
    adj = [[int(val) for val in row] for row in float_adj]

    # Identity matrix for k = 0
    result = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    base = adj
    power = k
    while power > 0:
        if power & 1:
            result = _mat_mul(result, base)
        power >>= 1
        if power > 0:
            base = _mat_mul(base, base)
    return nodes, result


def turan_max_edges(n: int, r: int) -> int:
    """Return exact Turan bound ex(n, K_r): max edges in an n-vertex K_r-free graph."""
    if r <= 1:
        raise ValueError("r must be at least 2 for K_r.")
    if n <= 0:
        return 0
    parts = r - 1
    q, rem = divmod(n, parts)
    # Partition into `rem` parts of size q+1 and `parts - rem` parts of size q.
    within_edges = rem * math.comb(q + 1, 2) + (parts - rem) * math.comb(q, 2)
    return math.comb(n, 2) - within_edges


def stirling_second(n: int, k: int) -> int:
    """Compute Stirling number of the second kind S(n, k): partitions of n actors into k non-empty groups."""
    if n < 0 or k < 0:
        raise ValueError("n and k must be non-negative.")
    if k > n:
        return 0
    if n == 0 and k == 0:
        return 1
    if k == 0:
        return 0
    dp = [0] * (k + 1)
    dp[0] = 1
    for i in range(1, n + 1):
        for j in range(min(i, k), 0, -1):
            dp[j] = j * dp[j] + dp[j - 1]
        dp[0] = 0
    return dp[k]


def bell_number(n: int) -> int:
    """Compute Bell number B_n: total number of ways to partition n actors into communities."""
    if n < 0:
        raise ValueError("n must be non-negative.")
    return sum(stirling_second(n, k) for k in range(n + 1))


def erdos_renyi_expected_metrics(n: int, p: float) -> dict[str, float]:
    """Compute exact combinatorial expectations for G(n, p) random social networks."""
    if n < 0 or not (0.0 <= p <= 1.0):
        raise ValueError("Require n >= 0 and 0 <= p <= 1.")
    exp_edges = math.comb(n, 2) * p
    exp_triangles = math.comb(n, 3) * (p ** 3)
    exp_open_wedges = math.comb(n, 3) * 3.0 * (p ** 2) * (1.0 - p)
    conn_threshold = math.log(n) / n if n > 1 else 0.0
    return {
        "expected_edges": float(exp_edges),
        "expected_degree": float((n - 1) * p) if n > 0 else 0.0,
        "expected_triangles": float(exp_triangles),
        "expected_open_wedges": float(exp_open_wedges),
        "expected_clustering": float(p),
        "connectivity_threshold_p": float(conn_threshold),
    }


def bipartite_maximum_matching(
    graph: SocialGraph, left_nodes: set[Any], right_nodes: set[Any]
) -> dict[Any, Any]:
    """Find a maximum cardinality matching in a bipartite graph using Kuhn's augmenting paths.

    Returns mapping {left_node: matched_right_node}. By Hall's Marriage Theorem,
    a left-saturating matching exists iff |N(S)| >= |S| for every S subset of left_nodes.
    """
    if left_nodes & right_nodes:
        raise ValueError("Left and right partitions must be disjoint.")

    adj = _undirected_adj(graph)
    match_right: dict[Any, Any] = {}

    def dfs(u: Any, visited_right: set[Any]) -> bool:
        for v in adj.get(u, set()):
            if v not in right_nodes or v in visited_right:
                continue
            visited_right.add(v)
            if v not in match_right or dfs(match_right[v], visited_right):
                match_right[v] = u
                return True
        return False

    for u in left_nodes:
        dfs(u, set())

    return {u: v for v, u in match_right.items()}
