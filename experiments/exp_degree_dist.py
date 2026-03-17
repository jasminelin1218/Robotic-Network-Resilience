"""
experiments/exp_degree_dist.py
================================
Degree Distribution Analysis — Is our network scale-free or Poisson?

This is a direct requirement of Topic 1 (Network Analysis):
  "Analyze the degree distribution. Does it follow a power law
   or a Poisson degree distribution? Why or why not?"

Applied here to the SBM at varying homophily:
  - Low h: degree distribution should be approximately Poisson
    (random connectivity, similar to ER graph)
  - High h: bimodal distribution (many intra-group edges, few inter)
  - Compares against ER, BA (power law), WS graphs

Also computes:
  - Average shortest path length
  - Network diameter
  - Clustering coefficient
These are all listed as requirements in Topic 1.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import scipy.stats as stats
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.simulation import generate_network
from core.paths import get_output_dir, is_interactive

N = 100
K = 4


def run(runs: int = 100, verbose: bool = True) -> dict:
    # SBM parameters chosen so expected degree matches avg_deg ≈ 6.
    # For N=100, K=4 (group size 25): E[deg] = 24·p_in + 75·p_out = 6
    #   h=4:  p_out = 6/(24·4+75) = 6/171 ≈ 0.035,  p_in = 4·p_out ≈ 0.140
    #   h=20: p_out = 6/(24·20+75) = 6/555 ≈ 0.0108, p_in = 20·p_out ≈ 0.216
    configs = {
        'SBM h=4':   (0.140, 0.035,  'sbm'),
        'SBM h=20':  (0.216, 0.0108, 'sbm'),
        'Erdos-Renyi': (None, None, 'er'),
        'Barabasi-Albert': (None, None, 'ba'),
        'Watts-Strogatz': (None, None, 'ws'),
    }

    results = {}
    avg_deg = 6.0  # matched across all graph types

    for label, (p_in, p_out, gtype) in configs.items():
        degrees, paths, clusts, diams = [], [], [], []

        for trial in range(15):
            if gtype == 'sbm':
                G = generate_network(N, K, p_in, p_out)
            elif gtype == 'er':
                G = nx.erdos_renyi_graph(N, avg_deg/(N-1))
                for nd in G.nodes(): G.nodes[nd]['group'] = nd // (N//K)
            elif gtype == 'ba':
                G = nx.barabasi_albert_graph(N, int(avg_deg/2))
                for nd in G.nodes(): G.nodes[nd]['group'] = nd // (N//K)
            elif gtype == 'ws':
                G = nx.watts_strogatz_graph(N, int(avg_deg), 0.1)
                for nd in G.nodes(): G.nodes[nd]['group'] = nd // (N//K)

            degs = [d for _, d in G.degree()]
            degrees.append(degs)

            # Only compute path lengths on connected component (expensive)
            cc = max(nx.connected_components(G), key=len)
            Gcc = G.subgraph(cc)
            if len(cc) > 10:
                paths.append(nx.average_shortest_path_length(Gcc))
                diams.append(nx.diameter(Gcc))
            clusts.append(nx.average_clustering(G))

        results[label] = {
            'degrees': np.array(degrees),
            'mean_degree': float(np.mean([np.mean(d) for d in degrees])),
            'avg_path': float(np.mean(paths)) if paths else 0,
            'diameter': float(np.mean(diams)) if diams else 0,
            'clustering': float(np.mean(clusts)),
        }
        if verbose:
            r = results[label]
            print(f"  {label}: mean_deg={r['mean_degree']:.2f}, "
                  f"avg_path={r['avg_path']:.2f}, diam={r['diameter']:.1f}, "
                  f"clust={r['clustering']:.3f}")

    return results


def plot(data: dict, save: bool = True) -> None:
    labels = list(data.keys())
    colors = ['#4d7eff', '#ff4d6a', '#3dd98a', '#ffb340', '#9b6dff']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 0: degree distributions
    for i, (label, color) in enumerate(zip(labels[:3], colors[:3])):
        ax = axes[0, i]
        all_degs = data[label]['degrees'].flatten()
        ax.hist(all_degs, bins=range(0, 30), density=True, alpha=0.7,
                color=color, edgecolor='white', linewidth=0.3)

        # Fit Poisson
        mu = np.mean(all_degs)
        from scipy.stats import poisson
        k_range = np.arange(0, 30)
        ax.plot(k_range, poisson.pmf(k_range, mu), 'w--', linewidth=1.5,
                alpha=0.7, label=f'Poisson(μ={mu:.1f})')

        ax.set_title(f'{label}\nDegree distribution',
                     fontweight='bold', fontsize=10)
        ax.set_xlabel('Degree k')
        ax.set_ylabel('P(degree = k)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Row 0 col 3: all degree distributions overlaid
    ax = axes[0, 2]
    ax.clear()
    for label, color in zip(labels, colors):
        all_degs = data[label]['degrees'].flatten()
        vals, bins = np.histogram(all_degs, bins=range(0,30), density=True)
        ax.plot(bins[:-1], vals, 'o-', color=color, linewidth=2,
                markersize=4, label=label, alpha=0.8)
    ax.set_title('Degree distributions — all graph types', fontweight='bold', fontsize=10)
    ax.set_xlabel('Degree k')
    ax.set_ylabel('P(degree = k)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Row 1: network properties comparison bar charts
    props = ['mean_degree', 'avg_path', 'diameter', 'clustering']
    prop_names = ['Mean degree', 'Avg shortest path', 'Diameter', 'Clustering coeff']

    for j, (prop, pname) in enumerate(zip(props[:3], prop_names[:3])):
        ax = axes[1, j]
        vals = [data[l][prop] for l in labels]
        bars = ax.bar(range(len(labels)), vals,
                      color=colors[:len(labels)], alpha=0.8, edgecolor='white',
                      linewidth=0.5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([l.replace(' ', '\n') for l in labels], fontsize=8)
        ax.set_title(pname, fontweight='bold', fontsize=10)
        ax.set_ylabel(pname)
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    plt.suptitle('Network Property Analysis — Degree Distribution, Path Length, Clustering\n'
                 'SBM vs ER vs BA vs WS (N=100, matched average degree ≈ 6)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(get_output_dir(), 'exp_degree_dist.png'),
                    dpi=150, bbox_inches='tight')
    if is_interactive(): plt.show()
    plt.close('all')


if __name__ == '__main__':
    print("Running Degree Distribution Analysis...")
    data = run()
    plot(data)
    print("Degree distribution analysis complete ✓")
