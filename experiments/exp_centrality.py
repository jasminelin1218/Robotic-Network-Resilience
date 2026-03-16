"""
experiments/exp_centrality.py
===============================
Centrality Analysis — What makes a node dangerous or protective?

This answers a core network science question (Topic 1 asks for centrality
analysis, Topic 6 benefits from knowing which nodes matter most):

  Which nodes in the network are the most influential for infection spread?
  Which are the best candidates for calibration placement?

Computes 4 centrality measures for SBM networks at varying h:
  1. Degree centrality          — how connected is each node?
  2. Betweenness centrality     — how often does a node sit on shortest paths?
  3. Eigenvector centrality     — how well-connected are a node's neighbors?
  4. Closeness centrality       — how quickly can a node reach all others?

Key findings expected:
  - At high h: betweenness centrality concentrates on inter-group bridge nodes
  - Bridge nodes (high betweenness, cross-group) are best calibration candidates
  - This connects directly to Topic 6: conductance bottlenecks = high-betweenness bridges
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.simulation import generate_network
from core.paths import get_output_dir, is_interactive

N = 100
K = 4
RUNS_PER_H = 10   # networks to average over


def run(runs: int = 100, verbose: bool = True) -> dict:
    p_out_configs = [
        ('Low h=4',   0.6, 0.15),
        ('Mid h=12',  0.6, 0.05),
        ('High h=30', 0.6, 0.02),
    ]

    results = {}
    for label, p_in, p_out in p_out_configs:
        deg_buf, bet_buf, eig_buf, clo_buf = [], [], [], []
        bridge_scores = []

        for _ in range(RUNS_PER_H):
            G = generate_network(N, K, p_in, p_out)
            deg = nx.degree_centrality(G)
            bet = nx.betweenness_centrality(G, normalized=True)
            eig = nx.eigenvector_centrality(G, max_iter=500, tol=1e-4)
            clo = nx.closeness_centrality(G)

            # Bridge score: betweenness × (1 - fraction of same-group neighbors)
            # = nodes that sit between groups are highest
            bridge = {}
            for nd in G.nodes():
                nb = list(G.neighbors(nd))
                if not nb:
                    bridge[nd] = 0; continue
                cross = sum(1 for n in nb if G.nodes[n]['group'] != G.nodes[nd]['group'])
                bridge[nd] = bet[nd] * (cross / len(nb))

            deg_buf.append(list(deg.values()))
            bet_buf.append(list(bet.values()))
            eig_buf.append(list(eig.values()))
            clo_buf.append(list(clo.values()))
            bridge_scores.append(list(bridge.values()))

        h = p_in / p_out
        results[label] = {
            'h': h, 'p_out': p_out,
            'deg': np.mean(deg_buf, axis=0),
            'bet': np.mean(bet_buf, axis=0),
            'eig': np.mean(eig_buf, axis=0),
            'clo': np.mean(clo_buf, axis=0),
            'bridge': np.mean(bridge_scores, axis=0),
        }
        if verbose:
            d = results[label]
            top_bridge = np.argsort(d['bridge'])[-3:]
            print(f"  {label}: top bridge nodes = {top_bridge}, "
                  f"max bridge score = {d['bridge'].max():.4f}")

    return results


def plot(data: dict, save: bool = True) -> None:
    labels = list(data.keys())
    colors = ['#3dd98a', '#ffb340', '#ff4d6a']
    K_size = N // K

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for col, (label, color) in enumerate(zip(labels, colors)):
        d = data[label]

        # Row 0: centrality distributions (violin-style via histogram)
        ax = axes[0, col]
        metrics = ['deg', 'bet', 'eig', 'clo']
        metric_names = ['Degree', 'Betweenness', 'Eigenvector', 'Closeness']
        mc = ['#4d7eff', '#ff4d6a', '#9b6dff', '#3dd9c4']
        for i, (m, mn, mc_) in enumerate(zip(metrics, metric_names, mc)):
            vals = d[m]
            # Normalize for comparison
            v = (np.array(vals) - np.min(vals)) / (np.max(vals) - np.min(vals) + 1e-9)
            ax.hist(v, bins=20, alpha=0.5, color=mc_, label=mn, density=True)

        ax.set_title(f'{label}\nCentrality distributions (normalised)',
                     fontweight='bold', fontsize=10)
        ax.set_xlabel('Normalised centrality value')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Row 1: Bridge score by group — which groups have the best bridge nodes?
        ax = axes[1, col]
        by_group = [[] for _ in range(K)]
        for nd_i, bs in enumerate(d['bridge']):
            g = nd_i // K_size
            if g < K:
                by_group[g].append(bs)

        group_means = [np.mean(bg) if bg else 0 for bg in by_group]
        group_maxes = [np.max(bg) if bg else 0 for bg in by_group]
        x = np.arange(K)
        ax.bar(x - 0.2, group_means, 0.35, label='Mean bridge score',
               color=color, alpha=0.7)
        ax.bar(x + 0.2, group_maxes, 0.35, label='Max bridge score',
               color=color, alpha=0.4, edgecolor=color)
        ax.set_xticks(x)
        ax.set_xticklabels([f'Squad {g}' for g in range(K)])
        ax.set_title(f'{label}\nBridge node score by squad\n'
                     '(betweenness × cross-group fraction)',
                     fontweight='bold', fontsize=10)
        ax.set_ylabel('Bridge score')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Centrality Analysis — Which nodes bridge communities?\n'
                 'Bridge score = betweenness × cross-group fraction\n'
                 '(Best calibration candidates = highest bridge score)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(get_output_dir(), 'exp_centrality.png'),
                    dpi=150, bbox_inches='tight')
    if is_interactive(): plt.show()
    plt.close('all')


if __name__ == '__main__':
    print("Running Centrality Analysis...")
    data = run()
    plot(data)
    print("Centrality analysis complete ✓")
