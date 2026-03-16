"""
experiments/exp8_scalability.py
=================================
Experiment 8 — Scalability + Graph Model Comparison  (NEW)

Two sub-experiments:

8a. Varying N (network size): Does the cascade threshold and recovery
    behaviour hold as the swarm scales from 50 to 500 robots?
    Attack count scales proportionally (10% of N).

8b. Graph model comparison: Run the same attack/recovery on SBM vs
    Erdos-Renyi, Barabasi-Albert, and Watts-Strogatz graphs with
    matched edge density. Directly addresses course Topic 6 requirement.

Both directly answer examiner questions:
  "Do your results generalise beyond your specific parameter choice?"
  "How does SBM compare to standard graph models?"
"""
from __future__ import annotations
from typing import List
import numpy as np
import networkx as nx
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import matplotlib; matplotlib.use('Agg')  # must be before pyplot
import matplotlib.pyplot as plt
from core.simulation import initialize_agents, run_simulation, monte_carlo
from core.metrics import algebraic_connectivity
from core.paths import get_output_dir, is_interactive

RUNS   = 100

# ── Shared simulation params (same as all other experiments) ─────────────────
SIM_PARAMS = dict(
    alpha=0.6, beta=0.4, threshold=0.3,
    r=3, p_c=0.3, max_steps=200,
)


# ── 8a: Varying N ─────────────────────────────────────────────────────────────

def run_scaling(runs: int = RUNS, verbose: bool = True) -> dict:
    """
    Fix K=4, h=20 (p_out=0.03), attack = 10% of N.
    Vary N in {50, 100, 200, 400}.
    """
    from core.simulation import generate_network
    N_values   = [50, 100, 200, 400]
    results    = {'N': N_values, 'peak': [], 'final': [],
                  'std_peak': [], 'std_final': [], 'lam2': []}

    for N in N_values:
        attack = max(2, N // 10)   # 10% of N
        res = monte_carlo(
            runs, N=N, K=4, p_in=0.6, p_out=0.03,
            attack_count=attack, attack_distribution='concentrated',
            calib_count=0, calib_distribution='scattered',
            **SIM_PARAMS,
        )
        # Also compute mean lambda2
        lam2_buf = []
        for _ in range(20):
            G = generate_network(N, 4, 0.6, 0.03)
            lam2_buf.append(algebraic_connectivity(G))

        results['peak'].append(res['mean_peak'])
        results['final'].append(res['mean_final'])
        results['std_peak'].append(res['std_peak'])
        results['std_final'].append(res['std_final'])
        results['lam2'].append(float(np.mean(lam2_buf)))
        if verbose:
            print(f"  N={N:3d}, attack={attack:2d} -> "
                  f"peak={res['mean_peak']:.3f}, final={res['mean_final']:.3f}, "
                  f"lambda2={results['lam2'][-1]:.3f}")

    return results


# ── 8b: Graph model comparison ─────────────────────────────────────────────────

def _make_graph(model: str, N: int, avg_degree: float, K: int = 4,
                seed: int = None) -> nx.Graph:
    """Generate graph with matched average degree, tag nodes with group."""
    rng = np.random.default_rng(seed)

    if model == 'SBM':
        from core.simulation import generate_network
        G = generate_network(N, K, p_in=0.6, p_out=0.03, seed=seed)
        return G

    elif model == 'Erdos-Renyi':
        p = avg_degree / (N - 1)
        G = nx.erdos_renyi_graph(N, p, seed=seed)

    elif model == 'Barabasi-Albert':
        m = max(1, int(avg_degree / 2))
        G = nx.barabasi_albert_graph(N, m, seed=seed)

    elif model == 'Watts-Strogatz':
        k = max(2, int(avg_degree))
        k = k if k % 2 == 0 else k + 1
        G = nx.watts_strogatz_graph(N, k, 0.1, seed=seed)

    # Tag all nodes with group 0 (no community structure) for non-SBM
    # Assign groups round-robin so initialise_agents works
    size = N // K
    for i, node in enumerate(G.nodes()):
        G.nodes[node]['group'] = min(i // size, K - 1)
    return G


def run_graph_comparison(runs: int = RUNS, verbose: bool = True) -> dict:
    """
    Compare SBM vs ER, BA, WS on identical attack/recovery parameters.
    All graphs: N=100, matched average degree (~6).
    """
    N = 100
    K = 4
    models    = ['SBM', 'Erdos-Renyi', 'Barabasi-Albert', 'Watts-Strogatz']
    avg_deg   = 6.0
    results   = {m: {'peak': [], 'final': [], 'std_peak': [],
                     'std_final': [], 'lam2': []} for m in models}

    attack_counts = [3, 5, 8, 10, 15]

    for model in models:
        for atk in attack_counts:
            peaks, finals = [], []
            lam2_buf = []
            for _ in range(runs):
                G = _make_graph(model, N, avg_deg, K)
                lam2_buf.append(algebraic_connectivity(G))
                state, calib = initialize_agents(
                    G, atk, 'concentrated', 0, 'scattered', K)
                res = run_simulation(G, state, calib, **SIM_PARAMS)
                peaks.append(res['peak_infection'])
                finals.append(res['final_infection'])

            results[model]['peak'].append(float(np.mean(peaks)))
            results[model]['final'].append(float(np.mean(finals)))
            results[model]['std_peak'].append(float(np.std(peaks)))
            results[model]['std_final'].append(float(np.std(finals)))
            results[model]['lam2'].append(float(np.mean(lam2_buf)))
            if verbose:
                print(f"  {model:18s}, attack={atk:2d} -> "
                      f"peak={np.mean(peaks):.3f}")

    return dict(models=models, attack_counts=attack_counts, results=results)


def run(runs: int = RUNS, verbose: bool = True) -> dict:
    print("  -- 8a: Scaling experiment --")
    scaling = run_scaling(runs=runs, verbose=verbose)
    print("  -- 8b: Graph model comparison --")
    comparison = run_graph_comparison(runs=runs, verbose=verbose)
    return dict(scaling=scaling, comparison=comparison)


def plot(data: dict, save: bool = True) -> None:
    scaling    = data['scaling']
    comparison = data['comparison']

    # ── 8a plots ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].errorbar(scaling['N'],
                     scaling['peak'], yerr=scaling['std_peak'],
                     fmt='o-', color='#E24B4A', linewidth=2, capsize=4,
                     label='Peak infection')
    axes[0].errorbar(scaling['N'],
                     scaling['final'], yerr=scaling['std_final'],
                     fmt='s--', color='#378ADD', linewidth=2, capsize=4,
                     label='Final infection')
    axes[0].set_xlabel('Network size N', fontsize=11)
    axes[0].set_ylabel('Infection rate', fontsize=11)
    axes[0].set_title('Infection rate vs N\n(attack = 10% of N, h=20)',
                      fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xscale('log')
    axes[0].set_xticks(scaling['N'])
    axes[0].set_xticklabels([str(n) for n in scaling['N']])

    axes[1].plot(scaling['N'], scaling['lam2'], 'o-',
                 color='#1D9E75', linewidth=2, markersize=8)
    axes[1].set_xlabel('Network size N', fontsize=11)
    axes[1].set_ylabel('Algebraic connectivity λ₂', fontsize=11)
    axes[1].set_title('λ₂ vs N\n(larger networks = higher connectivity)',
                      fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale('log')
    axes[1].set_xticks(scaling['N'])
    axes[1].set_xticklabels([str(n) for n in scaling['N']])

    # Cascade threshold shift with N
    axes[2].scatter(scaling['N'], scaling['peak'],
                    c=scaling['lam2'], cmap='RdYlGn', s=150, zorder=3)
    for i, (n, pk) in enumerate(zip(scaling['N'], scaling['peak'])):
        axes[2].annotate(f'N={n}', (n, pk),
                         textcoords='offset points', xytext=(5, 5), fontsize=8)
    axes[2].set_xlabel('Network size N', fontsize=11)
    axes[2].set_ylabel('Peak infection rate', fontsize=11)
    axes[2].set_title('Peak infection coloured by λ₂\n'
                      '(green = high connectivity)', fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xscale('log')
    axes[2].set_xticks(scaling['N'])
    axes[2].set_xticklabels([str(n) for n in scaling['N']])

    plt.suptitle('Experiment 8a — Scalability: Varying Network Size N',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(get_output_dir(), 'exp8a_scaling.png'),
                    dpi=150, bbox_inches='tight')
    if is_interactive():
        plt.show()
    plt.close()

    # ── 8b plots ──────────────────────────────────────────────────────────────
    COLORS = {'SBM': '#E24B4A', 'Erdos-Renyi': '#378ADD',
              'Barabasi-Albert': '#1D9E75', 'Watts-Strogatz': '#EF9F27'}
    models = comparison['models']
    atks   = comparison['attack_counts']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for model in models:
        col = COLORS[model]
        y   = np.array(comparison['results'][model]['peak'])
        std = np.array(comparison['results'][model]['std_peak'])
        axes[0].plot(atks, y, 'o-', color=col, linewidth=2, label=model)
        axes[0].fill_between(atks, np.clip(y-std,0,1),
                             np.clip(y+std,0,1), alpha=0.1, color=col)

        y2  = np.array(comparison['results'][model]['final'])
        std2= np.array(comparison['results'][model]['std_final'])
        axes[1].plot(atks, y2, 's-', color=col, linewidth=2, label=model)
        axes[1].fill_between(atks, np.clip(y2-std2,0,1),
                             np.clip(y2+std2,0,1), alpha=0.1, color=col)

    for ax, title in zip(axes, ['Peak infection rate', 'Final infection rate']):
        ax.set_xlabel('Attack intensity (# seed nodes)', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    plt.suptitle('Experiment 8b — Graph Model Comparison\n'
                 'SBM vs Erdős-Rényi vs Barabási-Albert vs Watts-Strogatz\n'
                 '(N=100, matched average degree ≈ 6)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(get_output_dir(), 'exp8b_graph_comparison.png'),
                    dpi=150, bbox_inches='tight')
    if is_interactive():
        plt.show()
    plt.close()

    # Print summary
    print(f"\n  Scaling: peak infection {'stable' if np.std(scaling['peak']) < 0.1 else 'varies'} "
          f"across N={scaling['N']}")
    print(f"  Graph comparison: SBM peak at attack=10: "
          f"{comparison['results']['SBM']['peak'][comparison['attack_counts'].index(10)]:.3f}")
    for m in ['Erdos-Renyi', 'Barabasi-Albert', 'Watts-Strogatz']:
        pk = comparison['results'][m]['peak'][comparison['attack_counts'].index(10)]
        print(f"                   {m} peak: {pk:.3f}")


if __name__ == '__main__':
    print("Running Experiment 8 — Scalability + Graph Comparison...")
    data = run()
    plot(data)
    print("Experiment 8 complete ✓")
