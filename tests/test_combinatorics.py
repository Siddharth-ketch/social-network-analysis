"""Unit tests for combinatorial SNA algorithms and identities."""

import math

from sna_gtc import (
    SocialGraph,
    bell_number,
    bipartite_affiliation_sample,
    bipartite_maximum_matching,
    clique_number,
    count_triangles_and_wedges,
    count_walks_of_length,
    erdos_renyi_expected_metrics,
    find_maximal_cliques,
    greedy_coloring,
    havel_hakimi_is_graphical,
    k_core_decomposition,
    stirling_second,
    triadic_census,
    turan_max_edges,
    zachary_karate_club,
)


def test_triangle_trace_identity_and_triadic_census():
    g = zachary_karate_club()
    tri_info = count_triangles_and_wedges(g)
    _, a3 = count_walks_of_length(g, 3)
    trace_a3 = sum(a3[i][i] for i in range(len(a3)))
    assert trace_a3 // 6 == tri_info["triangles"]

    census = triadic_census(g)
    assert census["closed_3_edge"] == tri_info["triangles"]
    assert census["open_2_edge"] == tri_info["open_wedges"]
    total = (
        census["empty_0_edge"]
        + census["single_1_edge"]
        + census["open_2_edge"]
        + census["closed_3_edge"]
    )
    assert total == math.comb(g.number_of_nodes(), 3) == census["total_triples"]


def test_cliques_cores_and_coloring():
    g = SocialGraph()
    # K_4 attached to a leaf 4
    for i in range(4):
        for j in range(i + 1, 4):
            g.add_edge(i, j)
    g.add_edge(3, 4)

    cliques = find_maximal_cliques(g)
    assert {0, 1, 2, 3} in cliques
    assert {3, 4} in cliques
    assert clique_number(g) == 4

    cores = k_core_decomposition(g)
    assert cores[0] == 3
    assert cores[4] == 1

    coloring = greedy_coloring(g)
    for u, v in g.edges():
        assert coloring[u] != coloring[v]
    assert max(coloring.values()) + 1 == 4


def test_havel_hakimi_and_extremal_combinatorics():
    g = zachary_karate_club()
    assert havel_hakimi_is_graphical(g.degree_sequence())
    assert havel_hakimi_is_graphical([2, 2, 2])
    assert not havel_hakimi_is_graphical([3, 3, 3, 1])
    assert not havel_hakimi_is_graphical([3, 1, 1])

    # Turan bound ex(4, K_3) = floor(4^2 / 4) = 4 (C_4)
    assert turan_max_edges(4, 3) == 4
    assert turan_max_edges(5, 3) == 6

    assert stirling_second(4, 2) == 7
    assert bell_number(4) == 15
    assert bell_number(5) == 52

    er = erdos_renyi_expected_metrics(10, 0.5)
    assert er["expected_edges"] == 22.5


def test_bipartite_maximum_matching():
    g, users, groups = bipartite_affiliation_sample()
    matching = bipartite_maximum_matching(g, users, groups)
    assert len(matching) == len(users)
    assert len(set(matching.values())) == len(users)
    for u, grp in matching.items():
        assert g.has_edge(u, grp)
