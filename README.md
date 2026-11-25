# Social Network Analysis Using Graph Theory and Combinatorics (`sna-gtc`)

A clean, modular Python library and CLI for analyzing social networks through **graph theory** and **enumerative/extremal combinatorics**. Every core algorithm is implemented transparently in pure Python (zero required runtime dependencies) with optional Matplotlib visualization.

## Features

### 1. Graph Theory Primitives (`sna_gtc.graph`)
- Directed and undirected weighted `SocialGraph` ADT
- Induced subgraphs $G[S]$, graph complements $\overline{G}$, and adjacency matrices $A$
- BFS shortest paths, all-pairs distances, connected components
- Metric invariants: eccentricity, diameter, radius, center, periphery, density, degree sequence

### 2. Centrality, Prestige & Clustering (`sna_gtc.centrality`)
- Degree, Closeness (Wasserman–Faust), and Harmonic centrality
- Exact shortest-path Betweenness centrality via **Brandes' $O(|V||E|)$ algorithm**
- Eigenvector centrality (power iteration) and **PageRank** (with dangling-node redistribution)
- Local clustering coefficient, average clustering, and global transitivity

### 3. Combinatorial Network Analysis (`sna_gtc.combinatorics`)
- **Triangle & Wedge Enumeration**: degree-ordered $O(m^{3/2})$ triangle counting and full **Triadic Census** across all $\binom{n}{3}$ triples
- **Walk Counting via Matrix Powers**: exact $A^k$ walk counts (satisfying $\operatorname{tr}(A^3)/6 = \text{triangles}$)
- **Cliques & Cores**: Bron–Kerbosch with Tomita pivoting (`find_maximal_cliques`, clique number $\omega(G)$) and Batagelj–Zaveršnik $k$-core decomposition
- **Graph Coloring**: greedy Welsh–Powell / smallest-last coloring ($\chi(G) \le \Delta(G)+1$)
- **Degree Realizability**: **Havel–Hakimi theorem** check
- **Extremal & Enumerative Combinatorics**:
  - **Turán's theorem** bound $\operatorname{ex}(n, K_r)$
  - **Stirling numbers of the second kind** $S(n, k)$ and **Bell numbers** $B_n$ for community partition counting
  - **Erdős–Rényi** $G(n, p)$ combinatorial expectations and connectivity threshold $\frac{\ln n}{n}$
  - **Bipartite Maximum Matching** (Kuhn's augmenting paths / Hall's Marriage Theorem)

### 4. Communities, Brokerage & Diffusion (`sna_gtc.community`, `sna_gtc.influence`)
- **Girvan–Newman** divisive community detection and Newman–Girvan modularity $Q$
- **Label Propagation** community detection
- **Burt's Structural Holes** (effective size and constraint)
- **Independent Cascade** diffusion simulation and submodular **Greedy Influence Maximization**
- **2-Approximate Vertex Cover** and **Greedy Dominating Set** for network monitoring

### 5. Benchmark Datasets & Visualization (`sna_gtc.datasets`, `sna_gtc.visualization`)
- Canonical **Zachary's Karate Club** (34 nodes, 78 edges)
- Canonical **Padgett's Florentine Families** (15 nodes, 20 edges)
- Synthetic **Erdős–Rényi** $G(n, p)$, **Watts–Strogatz** small-world, **Barabási–Albert** scale-free, and bipartite affiliation networks
- Pure-Python circular & Fruchterman–Reingold spring layouts + Matplotlib plotting

## Installation

```bash
python -m venv .venv
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Install package in editable mode with dev/visualization extras:
pip install -e ".[dev]"
```

## Quick Start

Run the CLI analysis on Zachary's Karate Club or Florentine Families:

```bash
python -m sna_gtc.cli --dataset karate
python -m sna_gtc.cli --dataset florentine --output-dir outputs
```

Or run the example scripts:

```bash
python examples/karate_club_analysis.py
python examples/florentine_families_power.py
python examples/combinatorial_bounds_demo.py
```

## Running Tests

```bash
pytest -v
```
