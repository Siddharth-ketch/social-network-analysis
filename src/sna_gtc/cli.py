"""Command-line interface for running Social Network Analysis reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .centrality import (
    average_clustering_coefficient,
    betweenness_centrality,
    closeness_centrality,
    degree_centrality,
    eigenvector_centrality,
    global_transitivity,
    pagerank,
)
from .combinatorics import (
    bell_number,
    clique_number,
    count_triangles_and_wedges,
    find_maximal_cliques,
    greedy_coloring,
    havel_hakimi_is_graphical,
    k_core_decomposition,
    triadic_census,
    turan_max_edges,
)
from .community import best_girvan_newman_partition, burt_structural_holes
from .datasets import (
    barabasi_albert,
    erdos_renyi_gnp,
    florentine_families,
    watts_strogatz,
    zachary_karate_club,
)
from .graph import SocialGraph
from .influence import greedy_influence_maximization
from .visualization import plot_degree_distribution, plot_network


def _top_k(scores: dict[Any, float], k: int = 5) -> list[tuple[str, float]]:
    ranked = sorted(scores.items(), key=lambda item: (-item[1], str(item[0])))
    return [(str(node), round(val, 4)) for node, val in ranked[:k]]


def build_dataset(name: str) -> SocialGraph:
    """Instantiate a built-in benchmark or synthetic social network."""
    if name == "karate":
        return zachary_karate_club()
    if name == "florentine":
        return florentine_families()
    if name == "erdos_renyi":
        return erdos_renyi_gnp(30, 0.18, seed=42)
    if name == "watts_strogatz":
        return watts_strogatz(30, 4, 0.25, seed=42)
    if name == "barabasi_albert":
        return barabasi_albert(30, 2, seed=42)
    raise ValueError(f"Unknown dataset: {name!r}")


def analyze_network(graph: SocialGraph, name: str) -> dict[str, Any]:
    """Compute a complete graph-theoretic and combinatorial SNA summary."""
    deg_c = degree_centrality(graph)
    bet_c = betweenness_centrality(graph)
    clo_c = closeness_centrality(graph)
    eig_c = eigenvector_centrality(graph)
    pr_c = pagerank(graph)

    tri = count_triangles_and_wedges(graph)
    census = triadic_census(graph)
    cliques = find_maximal_cliques(graph)
    omega = max((len(c) for c in cliques), default=0)
    cores = k_core_decomposition(graph)
    coloring = greedy_coloring(graph)
    deg_seq = graph.degree_sequence()

    communities, mod_q = best_girvan_newman_partition(graph, max_levels=8)
    holes = burt_structural_holes(graph)
    lowest_constraint = sorted(
        ((str(n), round(v["constraint"], 4)) for n, v in holes.items() if v["constraint"] > 0),
        key=lambda item: item[1],
    )[:5]

    seeds, expected_spread = greedy_influence_maximization(graph, k=3, prob=0.2, simulations=40)
    n = graph.number_of_nodes()

    return {
        "dataset": name,
        "nodes": n,
        "edges": graph.number_of_edges(),
        "density": round(graph.density(), 4),
        "connected": graph.is_connected(),
        "diameter": graph.diameter() if graph.is_connected() else None,
        "radius": graph.radius() if graph.is_connected() else None,
        "average_clustering": round(average_clustering_coefficient(graph), 4),
        "global_transitivity": round(global_transitivity(graph), 4),
        "combinatorics": {
            "triangles": tri["triangles"],
            "open_wedges": tri["open_wedges"],
            "triadic_census": census,
            "maximal_cliques_count": len(cliques),
            "clique_number_omega": omega,
            "max_k_core": max(cores.values(), default=0),
            "greedy_chromatic_bound": (max(coloring.values()) + 1) if coloring else 0,
            "turan_bound_for_k_omega_plus_1": turan_max_edges(n, max(omega + 1, 2)) if n > 0 else 0,
            "degree_sequence_graphical": havel_hakimi_is_graphical(deg_seq),
            "bell_number_for_n_if_small": bell_number(n) if n <= 15 else f"B_{n} (>10^9)",
        },
        "top_centrality": {
            "degree": _top_k(deg_c),
            "betweenness": _top_k(bet_c),
            "closeness": _top_k(clo_c),
            "eigenvector": _top_k(eig_c),
            "pagerank": _top_k(pr_c),
        },
        "communities": {
            "count": len(communities),
            "modularity_Q": round(mod_q, 4),
            "sizes": [len(c) for c in communities],
        },
        "structural_holes_lowest_constraint": lowest_constraint,
        "influence_maximization_k3": {
            "seeds": [str(s) for s in seeds],
            "expected_spread": round(expected_spread, 2),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Social Network Analysis Using Graph Theory and Combinatorics"
    )
    parser.add_argument(
        "--dataset",
        choices=["karate", "florentine", "erdos_renyi", "watts_strogatz", "barabasi_albert"],
        default="karate",
        help="Built-in or synthetic network dataset to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to save JSON report and PNG plots.",
    )
    args = parser.parse_args(argv)

    graph = build_dataset(args.dataset)
    report = analyze_network(graph, args.dataset)
    print(json.dumps(report, indent=2))

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = args.output_dir / f"{args.dataset}_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        communities, _ = best_girvan_newman_partition(graph, max_levels=8)
        scores = betweenness_centrality(graph)
        try:
            plot_network(
                graph,
                args.output_dir / f"{args.dataset}_network.png",
                communities=communities,
                scores=scores,
                title=f"{args.dataset.title()} Network (Communities & Betweenness)",
            )
            plot_degree_distribution(
                graph,
                args.output_dir / f"{args.dataset}_degrees.png",
                title=f"{args.dataset.title()} Degree Distribution",
            )
        except ImportError as exc:
            print(f"Skipping plots: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
