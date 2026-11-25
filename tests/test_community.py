"""Unit tests for community detection, modularity, and structural holes."""

import pytest

from sna_gtc import (
    SocialGraph,
    best_girvan_newman_partition,
    burt_structural_holes,
    label_propagation_communities,
    modularity,
)


def _barbell_two_triangles() -> SocialGraph:
    g = SocialGraph()
    g.add_edges_from([(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)])
    return g


def test_girvan_newman_splits_bridge():
    g = _barbell_two_triangles()
    communities, mod_q = best_girvan_newman_partition(g, max_levels=3)
    comm_sets = {frozenset(c) for c in communities}
    assert comm_sets == {frozenset({0, 1, 2}), frozenset({3, 4, 5})}
    assert mod_q > 0.35


def test_modularity_and_label_propagation():
    g = _barbell_two_triangles()
    lpa = label_propagation_communities(g, seed=42)
    assert sum(len(c) for c in lpa) == 6
    assert modularity(g, [{0, 1, 2, 3, 4, 5}]) == pytest.approx(0.0)


def test_burt_structural_holes_brokerage():
    # Center 0 bridges two unconnected pairs {1,2} and {3,4}
    g = SocialGraph()
    g.add_edges_from([(0, 1), (0, 2), (1, 2), (0, 3), (0, 4), (3, 4)])
    holes = burt_structural_holes(g)
    # Node 0 has lowest constraint compared to nodes inside closed triads
    assert holes[0]["constraint"] < holes[1]["constraint"]
    assert holes[0]["effective_size"] > holes[1]["effective_size"]
