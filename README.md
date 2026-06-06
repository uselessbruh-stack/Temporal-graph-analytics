# Dynamic Social Influence Ranking System

Tracking how influence shifts in social networks using temporal graph analytics — PageRank, community detection, and node embeddings on a real-world Bitcoin trust network.

## What This Does

Takes the [Bitcoin OTC trust network](https://snap.stanford.edu/data/soc-sign-bitcoinotc.html) (5,881 users, 35,592 timestamped trust ratings), slices it into 10 temporal snapshots, and analyzes how influence, communities, and structural roles evolve over time.

**Techniques used:**
- **PageRank** — recursive influence ranking at each time step
- **Louvain community detection** — tracking how communities form, merge, and split
- **Node2Vec** — 64-dim node embeddings with Procrustes alignment for cross-snapshot comparison
- **NDCG@10, Kendall's τ, NMI** — quantitative evaluation of ranking quality and stability

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the notebook

```bash
jupyter notebook Dynamic_Social_Influence_Ranking.ipynb
```

Run all cells — the dataset downloads automatically from Stanford SNAP on first run (~1 MB).

### 3. Outputs

- **Figures** → `output/figures/` (23 visualizations)
- **Metrics** → `output/metrics.csv`

## Project Structure

```
something-graph/
├── Dynamic_Social_Influence_Ranking.ipynb   # everything lives here
├── Report.md                                # full project report with analysis
├── requirements.txt                         # python dependencies
├── data/                                    # auto-downloaded dataset
│   ├── soc-sign-bitcoinotc.csv.gz
│   └── soc-sign-bitcoinotc.csv
└── output/
    ├── metrics.csv                          # evaluation metrics per snapshot
    └── figures/                             # all generated plots
        ├── network_t0.png
        ├── network_t4.png
        ├── network_t9.png
        ├── influence_heatmap.png
        ├── degree_distribution.png
        ├── pagerank_distribution.png
        ├── degree_vs_pagerank.png
        ├── centrality_correlation.png
        ├── top_node_trajectories.png
        ├── community_evolution.png
        ├── community_size_boxplot.png
        ├── modularity_over_time.png
        ├── embedding_drift.png
        ├── embedding_drift_distribution.png
        ├── drift_vs_pagerank_change.png
        ├── rank_stability.png
        ├── ndcg_scores.png
        ├── evaluation_overview.png
        ├── top_influencers.png
        ├── most_volatile_nodes.png
        ├── risers_and_fallers.png
        ├── graph_stats_over_time.png
        └── dataset_overview.png
```

## Requirements

- Python 3.10+
- networkx, python-louvain, node2vec, numpy, pandas, matplotlib, seaborn, scikit-learn, scipy

See `requirements.txt` for exact version constraints.

## Dataset

**Bitcoin OTC Trust Network** from [Stanford SNAP](https://snap.stanford.edu/data/soc-sign-bitcoinotc.html)

| | |
|---|---|
| Nodes | 5,881 |
| Edges | 35,592 |
| Type | Directed, signed, weighted |
| Ratings | -10 to +10 |
| Time span | Nov 2010 – Jan 2016 |

> S. Kumar, F. Spezzano, V.S. Subrahmanian, C. Faloutsos. *Edge Weight Prediction in Weighted Signed Networks.* IEEE ICDM, 2016.

## Key Results

| Metric | Mean | What it means |
|---|---|---|
| NDCG@10 | 0.988 | PageRank captures ground-truth influence almost perfectly |
| Kendall's τ | 0.912 | Influence rankings are highly stable across snapshots |
| NMI | 0.602 | Communities show moderate persistence with ongoing evolution |

## Sample Outputs

### Network at t=9
![Network](output/figures/network_t9.png)

### Influence Heatmap
![Heatmap](output/figures/influence_heatmap.png)

### Evaluation Overview
![Eval](output/figures/evaluation_overview.png)
