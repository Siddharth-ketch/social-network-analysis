"""Unit tests for combinatorial monitoring sets and influence maximization."""

from sna_gtc import (
    SocialGraph,
    greedy_dominating_set,
    greedy_influence_maximization,
    greedy_vertex_cover_2approx,
    independent_cascade,
    zachary_karate_club,
)


def test_vertex_cover_and_dominating_set():
    g = zachary_karate_club()
    cover = greedy_vertex_cover_2approx(g)
    for u, v in g.edges():
        assert u in cover or v in cover

    dom = greedy_dominating_set(g)
    covered = set(dom)
    for u in dom:
        covered.update(g.neighbors(u))
    assert covered == set(g.nodes())


def test_independent_cascade_and_influence_maximization():
    g = SocialGraph()
    # Star centered at 0
    g.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)])
    spread_zero = independent_cascade(g, [], prob=0.5, simulations=20)
    assert spread_zero == 0.0

    seeds, spread = greedy_influence_maximization(g, k=1, prob=0.8, simulations=50, seed=7)
    assert seeds == [0]
    assert spread > 3.5
