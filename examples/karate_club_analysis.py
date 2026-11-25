"""Example: Full graph-theoretic and combinatorial analysis of Zachary's Karate Club."""

from sna_gtc import (
    best_girvan_newman_partition,
    betweenness_centrality,
    burt_structural_holes,
    clique_number,
    count_triangles_and_wedges,
    degree_centrality,
    find_maximal_cliques,
    global_transitivity,
    greedy_coloring,
    greedy_influence_maximization,
    k_core_decomposition,
    triadic_census,
    zachary_karate_club,
)


def main() -> None:
    g = zachary_karate_club()
    print(f"Zachary's Karate Club: |V|={g.number_of_nodes()}, |E|={g.number_of_edges()}")
    print(f"Density={g.density():.4f}, Diameter={g.diameter()}, Radius={g.radius()}")
    print(f"Center nodes={g.center()}, Global transitivity={global_transitivity(g):.4f}")

    tri = count_triangles_and_wedges(g)
    print(f"Triangles={tri['triangles']}, Open wedges={tri['open_wedges']}")
    print(f"Triadic census={triadic_census(g)}")
    print(f"Maximal cliques={len(find_maximal_cliques(g))}, Clique number omega(G)={clique_number(g)}")

    cores = k_core_decomposition(g)
    print(f"Max k-core={max(cores.values())}, Coloring bound={max(greedy_coloring(g).values()) + 1}")

    deg = degree_centrality(g)
    bet = betweenness_centrality(g)
    print("Top 5 by degree centrality:", sorted(deg.items(), key=lambda x: -x[1])[:5])
    print("Top 5 by betweenness:", sorted(bet.items(), key=lambda x: -x[1])[:5])

    comms, q = best_girvan_newman_partition(g, max_levels=6)
    print(f"Best Girvan-Newman partition: {len(comms)} communities, Q={q:.4f}")

    holes = burt_structural_holes(g)
    brokers = sorted(holes.items(), key=lambda x: x[1]["constraint"])[:5]
    print("Top 5 structural-hole brokers (lowest constraint):", [(n, round(m["constraint"], 4)) for n, m in brokers])

    seeds, spread = greedy_influence_maximization(g, k=3, prob=0.2, simulations=60)
    print(f"Top 3 IC influence seeds={seeds}, expected spread={spread:.2f}")


if __name__ == "__main__":
    main()
