"""Unit tests for centrality, PageRank, and clustering metrics."""

import pytest

from sna_gtc import (
    SocialGraph,
    average_clustering_coefficient,
    betweenness_centrality,
    closeness_centrality,
    degree_centrality,
    eigenvector_centrality,
    florentine_families,
    global_transitivity,
    harmonic_centrality,
    local_clustering_coefficient,
    pagerank,
)


def test_star_graph_centralities():
    # Center node 0 connected to leaves 1..4
    g = SocialGraph()
    g.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4)])

    deg = degree_centrality(g)
    assert deg[0] == pytest.approx(1.0)
    assert deg[1] == pytest.approx(0.25)

    bet = betweenness_centrality(g)
    assert bet[0] == pytest.approx(1.0)
    assert bet[1] == pytest.approx(0.0)

    clo = closeness_centrality(g)
    assert clo[0] == pytest.approx(1.0)
    assert clo[1] < clo[0]

    harm = harmonic_centrality(g)
    assert harm[0] == pytest.approx(1.0)
    assert harm[1] == pytest.approx((1.0 + 3 * 0.5) / 4.0)


def test_pagerank_sums_to_one():
    g = florentine_families()
    pr = pagerank(g)
    assert sum(pr.values()) == pytest.approx(1.0, rel=1e-4)
    # Medici is the most prominent family in Padgett's network
    assert max(pr, key=pr.get) == "Medici"

    eig = eigenvector_centrality(g)
    assert max(eig, key=eig.get) == "Medici"


def test_clustering_and_transitivity():
    # K_4 has clustering = 1.0 and transitivity = 1.0
    g = SocialGraph()
    for i in range(4):
        for j in range(i + 1, 4):
            g.add_edge(i, j)
    assert local_clustering_coefficient(g, 0) == pytest.approx(1.0)
    assert average_clustering_coefficient(g) == pytest.approx(1.0)
    assert global_transitivity(g) == pytest.approx(1.0)
