"""Unit tests for SocialGraph primitives and metric invariants."""

import pytest

from sna_gtc import SocialGraph, florentine_families, zachary_karate_club


def test_graph_basic_operations_and_complement():
    g = SocialGraph()
    g.add_edges_from([("A", "B"), ("B", "C")])
    assert g.number_of_nodes() == 3
    assert g.number_of_edges() == 2
    assert g.degree("B") == 2
    assert g.degree_sequence() == [2, 1, 1]
    assert pytest.approx(g.density()) == 2 / 3

    comp = g.complement()
    assert comp.number_of_edges() == 1
    assert comp.has_edge("A", "C")
    assert not comp.has_edge("A", "B")


def test_shortest_paths_and_metric_invariants():
    g = SocialGraph()
    # Path graph P_5: 0-1-2-3-4
    g.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
    assert g.is_connected()
    assert g.diameter() == 4
    assert g.radius() == 2
    assert g.center() == [2]
    assert set(g.periphery()) == {0, 4}
    assert g.shortest_path_length(0, 4) == 4


def test_disconnected_graph_components():
    g = SocialGraph()
    g.add_edges_from([(1, 2), (3, 4), (4, 5)])
    comps = g.connected_components()
    assert len(comps) == 2
    assert comps[0] == {3, 4, 5}
    assert not g.is_connected()
    with pytest.raises(ValueError):
        g.diameter()


def test_canonical_datasets_sizes():
    karate = zachary_karate_club()
    assert karate.number_of_nodes() == 34
    assert karate.number_of_edges() == 78
    assert karate.is_connected()

    florentine = florentine_families()
    assert florentine.number_of_nodes() == 15
    assert florentine.number_of_edges() == 20
    assert florentine.is_connected()
