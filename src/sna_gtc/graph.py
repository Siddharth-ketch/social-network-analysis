"""Core graph data structure and shortest-path / structural graph-theory primitives."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass
class SocialGraph:
    """Graph ADT supporting undirected and directed social networks.

    Stores adjacency lists with optional edge weights and node attributes.
    """

    directed: bool = False
    _adj: dict[Any, dict[Any, float]] = field(default_factory=dict)
    _in_adj: dict[Any, dict[Any, float]] = field(default_factory=dict)
    _node_attrs: dict[Any, dict[str, Any]] = field(default_factory=dict)

    def add_node(self, node: Any, **attrs: Any) -> None:
        """Insert a node with optional metadata attributes."""
        if node not in self._adj:
            self._adj[node] = {}
            self._in_adj[node] = {}
            self._node_attrs[node] = {}
        if attrs:
            self._node_attrs[node].update(attrs)

    def add_nodes_from(self, nodes: Iterable[Any]) -> None:
        """Insert multiple nodes."""
        for node in nodes:
            self.add_node(node)

    def add_edge(self, u: Any, v: Any, weight: float = 1.0) -> None:
        """Add an edge (u, v) with an optional positive weight.

        Self-loops are ignored for simple-graph social network analysis.
        """
        if u == v:
            self.add_node(u)
            return
        self.add_node(u)
        self.add_node(v)
        self._adj[u][v] = float(weight)
        self._in_adj[v][u] = float(weight)
        if not self.directed:
            self._adj[v][u] = float(weight)
            self._in_adj[u][v] = float(weight)

    def add_edges_from(self, edges: Iterable[tuple[Any, ...]]) -> None:
        """Insert edges from (u, v) or (u, v, weight) tuples."""
        for edge in edges:
            if len(edge) == 2:
                self.add_edge(edge[0], edge[1])
            elif len(edge) == 3:
                self.add_edge(edge[0], edge[1], float(edge[2]))
            else:
                raise ValueError(f"Expected 2 or 3 elements in edge tuple, got {len(edge)}")

    def remove_edge(self, u: Any, v: Any) -> None:
        """Remove edge (u, v) if present."""
        if u in self._adj:
            self._adj[u].pop(v, None)
        if v in self._in_adj:
            self._in_adj[v].pop(u, None)
        if not self.directed:
            if v in self._adj:
                self._adj[v].pop(u, None)
            if u in self._in_adj:
                self._in_adj[u].pop(v, None)

    def has_node(self, node: Any) -> bool:
        """Return True if node exists in the graph."""
        return node in self._adj

    def has_edge(self, u: Any, v: Any) -> bool:
        """Return True if directed/undirected edge (u, v) exists."""
        return u in self._adj and v in self._adj[u]

    def edge_weight(self, u: Any, v: Any) -> float:
        """Return weight of edge (u, v)."""
        return self._adj[u][v]

    def node_attrs(self, node: Any) -> dict[str, Any]:
        """Return attribute dictionary for a node."""
        return dict(self._node_attrs.get(node, {}))

    def nodes(self) -> list[Any]:
        """Return list of all nodes."""
        return list(self._adj.keys())

    def number_of_nodes(self) -> int:
        """Return |V|."""
        return len(self._adj)

    def edges(self, with_weights: bool = False) -> list[tuple[Any, ...]]:
        """Return list of edges; deduplicated for undirected graphs."""
        result: list[tuple[Any, ...]] = []
        if self.directed:
            for u, nbrs in self._adj.items():
                for v, w in nbrs.items():
                    result.append((u, v, w) if with_weights else (u, v))
            return result

        seen: set[frozenset[Any]] = set()
        for u, nbrs in self._adj.items():
            for v, w in nbrs.items():
                key = frozenset((u, v))
                if key not in seen:
                    seen.add(key)
                    result.append((u, v, w) if with_weights else (u, v))
        return result

    def number_of_edges(self) -> int:
        """Return |E|."""
        if self.directed:
            return sum(len(nbrs) for nbrs in self._adj.values())
        return sum(len(nbrs) for nbrs in self._adj.values()) // 2

    def neighbors(self, node: Any) -> list[Any]:
        """Return outgoing neighbors (or all neighbors if undirected)."""
        return list(self._adj.get(node, {}).keys())

    def predecessors(self, node: Any) -> list[Any]:
        """Return incoming neighbors (identical to neighbors if undirected)."""
        return list(self._in_adj.get(node, {}).keys())

    def all_neighbors(self, node: Any) -> set[Any]:
        """Return union of outgoing and incoming neighbors."""
        return set(self._adj.get(node, {}).keys()) | set(self._in_adj.get(node, {}).keys())

    def degree(self, node: Any) -> int:
        """Return vertex degree (out-degree + in-degree for directed, neighbors for undirected)."""
        if self.directed:
            return len(self._adj[node]) + len(self._in_adj[node])
        return len(self._adj[node])

    def out_degree(self, node: Any) -> int:
        """Return out-degree of a vertex."""
        return len(self._adj[node])

    def in_degree(self, node: Any) -> int:
        """Return in-degree of a vertex."""
        return len(self._in_adj[node])

    def degree_sequence(self, descending: bool = True) -> list[int]:
        """Return the degree sequence of the graph."""
        seq = [self.degree(node) for node in self._adj]
        seq.sort(reverse=descending)
        return seq

    def density(self) -> float:
        """Compute graph density |E| / max_possible_edges."""
        n = self.number_of_nodes()
        if n < 2:
            return 0.0
        max_edges = n * (n - 1) if self.directed else (n * (n - 1)) // 2
        return self.number_of_edges() / max_edges

    def copy(self) -> SocialGraph:
        """Return a deep copy of the graph."""
        g = SocialGraph(directed=self.directed)
        for node, attrs in self._node_attrs.items():
            g.add_node(node, **attrs)
        for u, v, w in self.edges(with_weights=True):
            g.add_edge(u, v, weight=w)
        return g

    def subgraph(self, nodes: Iterable[Any]) -> SocialGraph:
        """Return the induced subgraph G[S] on the given subset of nodes."""
        node_set = set(nodes)
        sub = SocialGraph(directed=self.directed)
        for node in self._adj:
            if node in node_set:
                sub.add_node(node, **self._node_attrs.get(node, {}))
        for u, v, w in self.edges(with_weights=True):
            if u in node_set and v in node_set:
                sub.add_edge(u, v, weight=w)
        return sub

    def complement(self) -> SocialGraph:
        """Return the complement graph G_bar for an undirected simple graph."""
        if self.directed:
            raise ValueError("Graph complement is defined here for undirected simple graphs.")
        comp = SocialGraph(directed=False)
        node_list = self.nodes()
        for node in node_list:
            comp.add_node(node, **self._node_attrs.get(node, {}))
        for i, u in enumerate(node_list):
            for v in node_list[i + 1 :]:
                if not self.has_edge(u, v):
                    comp.add_edge(u, v)
        return comp

    def adjacency_matrix(self, weighted: bool = False) -> tuple[list[Any], list[list[float]]]:
        """Return (ordered_nodes, adjacency_matrix)."""
        node_list = self.nodes()
        index = {node: idx for idx, node in enumerate(node_list)}
        n = len(node_list)
        matrix = [[0.0] * n for _ in range(n)]
        for u, nbrs in self._adj.items():
            i = index[u]
            for v, w in nbrs.items():
                j = index[v]
                matrix[i][j] = w if weighted else 1.0
        return node_list, matrix

    def bfs_distances(self, source: Any) -> dict[Any, int]:
        """Compute unweighted shortest-path distances from source via BFS."""
        if source not in self._adj:
            raise KeyError(f"Source node {source!r} not in graph.")
        dist: dict[Any, int] = {source: 0}
        queue: deque[Any] = deque([source])
        while queue:
            u = queue.popleft()
            next_dist = dist[u] + 1
            for v in self._adj[u]:
                if v not in dist:
                    dist[v] = next_dist
                    queue.append(v)
        return dist

    def shortest_path_length(self, source: Any, target: Any) -> int | None:
        """Return shortest-path distance between source and target, or None if unreachable."""
        if source == target:
            return 0 if source in self._adj else None
        return self.bfs_distances(source).get(target)

    def all_pairs_shortest_paths(self) -> dict[Any, dict[Any, int]]:
        """Compute unweighted shortest-path distances between all reachable pairs."""
        return {node: self.bfs_distances(node) for node in self._adj}

    def connected_components(self) -> list[set[Any]]:
        """Return connected components (weakly connected if directed), largest first."""
        visited: set[Any] = set()
        components: list[set[Any]] = []
        for start in self._adj:
            if start in visited:
                continue
            comp: set[Any] = set()
            queue: deque[Any] = deque([start])
            visited.add(start)
            while queue:
                u = queue.popleft()
                comp.add(u)
                neighbors = self.all_neighbors(u) if self.directed else self._adj[u]
                for v in neighbors:
                    if v not in visited:
                        visited.add(v)
                        queue.append(v)
            components.append(comp)
        components.sort(key=len, reverse=True)
        return components

    def is_connected(self) -> bool:
        """Return True if the graph consists of a single connected component."""
        if not self._adj:
            return False
        return len(self.connected_components()) == 1

    def eccentricity(self, node: Any) -> int:
        """Return maximum shortest-path distance from node to all other reachable nodes.

        Raises ValueError if the graph is disconnected or empty.
        """
        if self.number_of_nodes() == 0:
            raise ValueError("Eccentricity is undefined for an empty graph.")
        dists = self.bfs_distances(node)
        if len(dists) != self.number_of_nodes():
            raise ValueError("Eccentricity requires a connected (or strongly connected) graph.")
        return max(dists.values())

    def diameter(self) -> int:
        """Return max eccentricity over all vertices."""
        return max(self.eccentricity(node) for node in self._adj)

    def radius(self) -> int:
        """Return min eccentricity over all vertices."""
        return min(self.eccentricity(node) for node in self._adj)

    def center(self) -> list[Any]:
        """Return vertices whose eccentricity equals the graph radius."""
        eccs = {node: self.eccentricity(node) for node in self._adj}
        rad = min(eccs.values())
        return [node for node, val in eccs.items() if val == rad]

    def periphery(self) -> list[Any]:
        """Return vertices whose eccentricity equals the graph diameter."""
        eccs = {node: self.eccentricity(node) for node in self._adj}
        diam = max(eccs.values())
        return [node for node, val in eccs.items() if val == diam]
