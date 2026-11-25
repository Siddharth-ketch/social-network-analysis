"""Pure-Python network layouts and optional Matplotlib visualization helpers."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

from .graph import SocialGraph


def circular_layout(graph: SocialGraph, radius: float = 1.0) -> dict[Any, tuple[float, float]]:
    """Place vertices evenly on a circle of the given radius."""
    nodes = graph.nodes()
    n = len(nodes)
    if n == 0:
        return {}
    return {
        node: (
            radius * math.cos(2.0 * math.pi * idx / n),
            radius * math.sin(2.0 * math.pi * idx / n),
        )
        for idx, node in enumerate(nodes)
    }


def spring_layout(
    graph: SocialGraph, iterations: int = 80, seed: int = 42
) -> dict[Any, tuple[float, float]]:
    """Compute Fruchterman-Reingold force-directed 2D node coordinates."""
    nodes = graph.nodes()
    n = len(nodes)
    if n == 0:
        return {}
    if n == 1:
        return {nodes[0]: (0.0, 0.0)}

    rng = random.Random(seed)
    pos = {u: [rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)] for u in nodes}
    k = math.sqrt(4.0 / n)
    temp = 0.5

    for it in range(iterations):
        disp = {u: [0.0, 0.0] for u in nodes}
        # Repulsive forces
        for i, u in enumerate(nodes):
            for v in nodes[i + 1 :]:
                dx = pos[u][0] - pos[v][0]
                dy = pos[u][1] - pos[v][1]
                dist = math.hypot(dx, dy) + 1e-6
                force = (k * k) / dist
                fx, fy = (dx / dist) * force, (dy / dist) * force
                disp[u][0] += fx
                disp[u][1] += fy
                disp[v][0] -= fx
                disp[v][1] -= fy

        # Attractive forces
        for u, v in graph.edges():
            dx = pos[u][0] - pos[v][0]
            dy = pos[u][1] - pos[v][1]
            dist = math.hypot(dx, dy) + 1e-6
            force = (dist * dist) / k
            fx, fy = (dx / dist) * force, (dy / dist) * force
            disp[u][0] -= fx
            disp[u][1] -= fy
            disp[v][0] += fx
            disp[v][1] += fy

        # Apply bounded displacement
        cooling = temp * (1.0 - it / float(iterations))
        for u in nodes:
            dx, dy = disp[u]
            dist = math.hypot(dx, dy) + 1e-6
            step = min(dist, max(cooling, 0.01))
            pos[u][0] += (dx / dist) * step
            pos[u][1] += (dy / dist) * step

    return {u: (pos[u][0], pos[u][1]) for u in nodes}


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting. Install with `pip install matplotlib`."
        ) from exc
    return plt


def plot_network(
    graph: SocialGraph,
    output_path: str | Path,
    communities: list[set[Any]] | None = None,
    scores: dict[Any, float] | None = None,
    title: str = "Social Network Graph",
) -> Path:
    """Render a network graph colored by community and sized by node score."""
    plt = _require_matplotlib()
    pos = spring_layout(graph)

    node_comm: dict[Any, int] = {}
    if communities:
        for idx, comm in enumerate(communities):
            for node in comm:
                node_comm[node] = idx

    fig, ax = plt.subplots(figsize=(9, 7))
    for u, v in graph.edges():
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y2], color="#94a3b8", linewidth=0.9, alpha=0.65, zorder=1)

    nodes = graph.nodes()
    xs = [pos[u][0] for u in nodes]
    ys = [pos[u][1] for u in nodes]
    colors = [node_comm.get(u, 0) for u in nodes]
    sizes = [240 + 1800 * (scores.get(u, 0.0) if scores else 0.1) for u in nodes]

    ax.scatter(xs, ys, c=colors, s=sizes, cmap="tab10", edgecolors="#0f172a", linewidths=1.0, zorder=2)
    for u in nodes:
        x, y = pos[u]
        ax.text(x, y, str(u), fontsize=8, ha="center", va="center", color="#0f172a", zorder=3)

    ax.set_title(title)
    ax.axis("off")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_degree_distribution(
    graph: SocialGraph,
    output_path: str | Path,
    title: str = "Degree Distribution",
) -> Path:
    """Plot vertex degree histogram."""
    plt = _require_matplotlib()
    degrees = graph.degree_sequence(descending=False)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(degrees, bins=range(min(degrees, default=0), max(degrees, default=1) + 2), align="left", edgecolor="black", color="#3b82f6")
    ax.set_xlabel("Degree k")
    ax.set_ylabel("Number of Nodes")
    ax.set_title(title)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
