"""Example: Demonstrating combinatorial identities & extremal bounds on social graphs."""

import math

from sna_gtc import (
    bell_number,
    bipartite_affiliation_sample,
    bipartite_maximum_matching,
    count_triangles_and_wedges,
    count_walks_of_length,
    erdos_renyi_expected_metrics,
    erdos_renyi_gnp,
    havel_hakimi_is_graphical,
    stirling_second,
    turan_max_edges,
    zachary_karate_club,
)


def main() -> None:
    g = zachary_karate_club()

    # 1. Verify trace(A^3) / 6 == triangle count
    _, a3 = count_walks_of_length(g, 3)
    trace_a3 = sum(a3[i][i] for i in range(len(a3)))
    tri = count_triangles_and_wedges(g)["triangles"]
    print(f"Combinatorial trace identity: tr(A^3)/6 = {trace_a3 // 6}, exact triangles = {tri}")

    # 2. Havel-Hakimi graphicality check
    seq = g.degree_sequence()
    print(f"Karate degree sequence graphical (Havel-Hakimi): {havel_hakimi_is_graphical(seq)}")
    print(f"Non-graphical sequence [3, 3, 3, 1]: {havel_hakimi_is_graphical([3, 3, 3, 1])}")

    # 3. Turan extremal bound for K_6-free graphs on 34 nodes
    print(f"Turan bound ex(34, K_6) = {turan_max_edges(34, 6)} vs Karate edges = {g.number_of_edges()}")

    # 4. Community partition combinatorics (Stirling & Bell numbers)
    n_small = 10
    print(f"Ways to partition {n_small} users into 3 groups S(10,3) = {stirling_second(n_small, 3)}")
    print(f"Total partitions B_10 = {bell_number(n_small)}")

    # 5. Erdos-Renyi G(n, p) expected vs realized combinatorics
    n, p = 40, 0.15
    er = erdos_renyi_gnp(n, p, seed=7)
    exp = erdos_renyi_expected_metrics(n, p)
    er_tri = count_triangles_and_wedges(er)["triangles"]
    print(f"G({n},{p}): expected edges={exp['expected_edges']:.1f}, actual={er.number_of_edges()}")
    print(f"G({n},{p}): expected triangles={exp['expected_triangles']:.1f}, actual={er_tri}")

    # 6. Bipartite matching & Hall's condition
    bg, users, groups = bipartite_affiliation_sample()
    matching = bipartite_maximum_matching(bg, users, groups)
    print(f"Bipartite maximum matching size: {len(matching)}/{len(users)} -> {matching}")


if __name__ == "__main__":
    main()
