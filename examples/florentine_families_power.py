"""Example: Medici brokerage and combinatorial power structure in Renaissance Florence."""

from sna_gtc import (
    betweenness_centrality,
    burt_structural_holes,
    closeness_centrality,
    eigenvector_centrality,
    florentine_families,
    pagerank,
)


def main() -> None:
    g = florentine_families()
    bet = betweenness_centrality(g)
    clo = closeness_centrality(g)
    eig = eigenvector_centrality(g)
    pr = pagerank(g)
    holes = burt_structural_holes(g)

    print(f"Florentine Families Network: |V|={g.number_of_nodes()}, |E|={g.number_of_edges()}")
    print(f"{'Family':<14} {'Deg':>4} {'Between':>9} {'Closeness':>10} {'Eigenv':>8} {'PageRank':>9} {'Constraint':>11}")
    for family in sorted(g.nodes(), key=lambda n: -bet[n]):
        print(
            f"{family:<14} {g.degree(family):>4} {bet[family]:>9.4f} "
            f"{clo[family]:>10.4f} {eig[family]:>8.4f} {pr[family]:>9.4f} "
            f"{holes[family]['constraint']:>11.4f}"
        )


if __name__ == "__main__":
    main()
