"""
experiments/exp1_topology.py
=============================
Experiment 1 — Topological Resilience (K × h Resilience Surface)

Narrative position  —  Act 1, step 3  (final Act 1 result)
------------------------------------------------------------
Motivation (from Exp 7):
  Exp 7 showed that λ₂ is a reliable predictor of infection spread — it
  is the mechanism linking topology to vulnerability.  Now we need to
  translate that into actionable design guidance: for a robot swarm
  designer who controls K (number of squads) and p_out (inter-squad
  link probability), what combination minimises spread?

Hypothesis:
  There exists an optimal K* and a threshold h* below which cascades are
  suppressed.  The resilience surface (peak infection vs K vs h) will
  reveal a clear "safe zone" that a practitioner can target.

What we measure:
  2-D grid sweep of K ∈ {2,4,6,8} and h = p_in/p_out ∈ {4, …, 40}.
  Both peak and final infection are reported as heatmaps.

Handoff → Act 2 (Exp 2):
  We now know WHICH topology is safest.  The next question is:
  given that topology, can we make the system even MORE robust using
  calibration nodes and active defenses?

Parameters confirmed to show clear dynamics:
  alpha=0.6, beta=0.4, threshold=0.3, p_c=0.3, attack=10, no calib nodes
  p_out range gives h = 4 to 40 — clear transition from recovery to collapse
"""
from __future__ import annotations
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import matplotlib; matplotlib.use('Agg')  # must be before pyplot
import matplotlib.pyplot as plt
from core.paths import get_output_dir, is_interactive
from core.simulation import monte_carlo

RUNS   = 100
N      = 100


def run(runs: int = RUNS, verbose: bool = True) -> dict:
    K_list      = [2, 4, 6, 8, 10]
    p_out_range = np.array([0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.15])

    heat_peak     = np.zeros((len(K_list), len(p_out_range)))
    heat_final    = np.zeros((len(K_list), len(p_out_range)))
    heat_conv     = np.zeros((len(K_list), len(p_out_range)))
    heat_std_peak = np.zeros((len(K_list), len(p_out_range)))

    for i, K in enumerate(K_list):
        for j, p_out in enumerate(p_out_range):
            res = monte_carlo(
                runs, N=N, K=K, p_in=0.6, p_out=p_out,
                attack_count=10, attack_distribution='concentrated',
                calib_count=0, calib_distribution='scattered',
                alpha=0.6, beta=0.4, threshold=0.3,
                r=3, p_c=0.3, max_steps=200,
            )
            heat_peak[i, j]     = res['mean_peak']
            heat_final[i, j]    = res['mean_final']
            heat_conv[i, j]     = res['mean_conv']
            heat_std_peak[i, j] = res['std_peak']
            if verbose:
                h = 0.6 / p_out
                print(f"  K={K:2d}, h={h:5.1f} -> "
                      f"peak={res['mean_peak']:.3f}, final={res['mean_final']:.3f}")

    return dict(K_list=K_list, p_out_range=p_out_range,
                heat_peak=heat_peak, heat_final=heat_final, heat_conv=heat_conv,
                heat_std_peak=heat_std_peak, runs=runs)


def plot(data: dict, save: bool = True) -> None:
    K_list, p_out_range = data['K_list'], data['p_out_range']
    h_labels = [f'{0.6/v:.0f}' for v in p_out_range]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, mat, title in zip(
        axes,
        [data['heat_peak'], data['heat_final'], data['heat_conv']],
        ['Peak infection rate', 'Final infection rate', 'Convergence step'],
    ):
        vmax = 1.0 if 'infection' in title else mat.max()
        im = ax.imshow(mat, aspect='auto', origin='lower',
                       cmap='RdYlGn_r', vmin=0, vmax=vmax)
        ax.set_xticks(range(len(p_out_range)))
        ax.set_xticklabels(h_labels, rotation=0, fontsize=9)
        ax.set_yticks(range(len(K_list)))
        ax.set_yticklabels(K_list)
        ax.set_xlabel('Homophily ratio h = p_in / p_out')
        ax.set_ylabel('K (number of squads)')
        ax.set_title(title, fontweight='bold')
        plt.colorbar(im, ax=ax)
        for ii in range(len(K_list)):
            for jj in range(len(p_out_range)):
                v = mat[ii, jj]
                ax.text(jj, ii, f'{v:.2f}', ha='center', va='center',
                        fontsize=7, color='white' if v > 0.5 else 'black')

    plt.suptitle('Experiment 1 — Topological Resilience Heatmaps',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(get_output_dir(), 'exp1_heatmaps.png'),
                    dpi=150, bbox_inches='tight')
    if is_interactive(): plt.show()

    # Optimal-K line chart
    best_peak_per_K = data['heat_peak'].min(axis=1)
    best_h_per_K    = [0.6 / p_out_range[np.argmin(data['heat_peak'][i])]
                       for i in range(len(K_list))]
    K_star = K_list[int(np.argmin(best_peak_per_K))]

    runs = data.get('runs', RUNS)
    if 'heat_std_peak' in data:
        best_std_per_K = [data['heat_std_peak'][i, np.argmin(data['heat_peak'][i])]
                          for i in range(len(K_list))]
        ci = 1.96 * np.array(best_std_per_K) / np.sqrt(runs)
    else:
        ci = None

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(K_list, best_peak_per_K, 'o-', color='steelblue',
            linewidth=2.5, markersize=9, label='Min peak infection (optimal h per K)')
    if ci is not None:
        ax.fill_between(K_list,
                        np.clip(best_peak_per_K - ci, 0, 1),
                        np.clip(best_peak_per_K + ci, 0, 1),
                        alpha=0.15, color='steelblue', label='95% CI')
    ax.axvline(K_star, color='crimson', linestyle='--', linewidth=2,
               label=f'Optimal K* = {K_star}')
    for K_val, pk, h_val in zip(K_list, best_peak_per_K, best_h_per_K):
        ax.annotate(f'best h={h_val:.0f}', (K_val, pk),
                    textcoords='offset points', xytext=(0, 12),
                    ha='center', fontsize=8, color='gray')
    ax.set_xlabel('K (number of squads)', fontsize=12)
    ax.set_ylabel('Minimum achievable peak infection rate', fontsize=12)
    ax.set_title(f'Optimal K* = {K_star} minimises peak infection', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(get_output_dir(), 'exp1_optimal_K.png'),
                    dpi=150, bbox_inches='tight')
    if is_interactive(): plt.show()
    plt.close('all')
    print(f"  -> Optimal K* = {K_star}")


if __name__ == '__main__':
    print("Running Experiment 1 — Topological Resilience...")
    data = run()
    plot(data, save=(RUNS >= 100))
    print("Experiment 1 complete")
