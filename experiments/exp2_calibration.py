"""
experiments/exp2_calibration.py
================================
Experiment 2 — Calibration Node Deployment Strategy

Parameters: h=30 (p_out=0.02), attack=10 — infected regime where
calibration nodes make a measurable difference.
"""
from __future__ import annotations
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import matplotlib; matplotlib.use('Agg')  # must be before pyplot
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from core.paths import get_output_dir, is_interactive
from core.simulation import monte_carlo

RUNS   = 100
N      = 100

DISTRIBUTIONS = ['concentrated', 'scattered', 'attack_group', 'other_group']
COLORS = {
    'concentrated': '#E24B4A',
    'scattered':    '#378ADD',
    'attack_group': '#EF9F27',
    'other_group':  '#1D9E75',
}


def run(runs: int = RUNS, verbose: bool = True) -> dict:
    calib_counts = [0, 1, 2, 4, 6, 10]
    results = {d: {'conv': [], 'peak': [], 'final': [],
                   'std_peak': [], 'std_final': []}
               for d in DISTRIBUTIONS}

    for dist in DISTRIBUTIONS:
        for cnt in calib_counts:
            res = monte_carlo(
                runs, N=N, K=4, p_in=0.6, p_out=0.02,
                attack_count=10, attack_distribution='concentrated',
                calib_count=cnt, calib_distribution=dist,
                alpha=0.6, beta=0.4, threshold=0.3,
                r=3, p_c=0.3, max_steps=200,
            )
            results[dist]['conv'].append(res['mean_conv'])
            results[dist]['peak'].append(res['mean_peak'])
            results[dist]['final'].append(res['mean_final'])
            results[dist]['std_peak'].append(res['std_peak'])
            results[dist]['std_final'].append(res['std_final'])
            if verbose:
                print(f"  dist={dist:13s}, calib={cnt:2d} -> "
                      f"peak={res['mean_peak']:.3f}, final={res['mean_final']:.3f}")

    return dict(calib_counts=calib_counts, results=results, runs=runs)


def plot(data: dict, save: bool = True) -> None:
    calib_counts = data['calib_counts']
    results      = data['results']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [
        ('conv',  None,          'Convergence step'),
        ('peak',  'std_peak',    'Peak infection rate'),
        ('final', 'std_final',   'Final infection rate'),
    ]
    for ax, (metric, std_key, ylabel) in zip(axes, metrics):
        for dist in DISTRIBUTIONS:
            y   = np.array(results[dist][metric])
            col = COLORS[dist]
            ax.plot(calib_counts, y, 'o-',
                    label=dist.replace('_', ' ').capitalize(),
                    color=col, linewidth=2)
            if std_key:
                std = np.array(results[dist][std_key])
                ax.fill_between(calib_counts,
                                np.clip(y - std, 0, 1),
                                np.clip(y + std, 0, 1),
                                alpha=0.12, color=col)
        ax.set_xlabel('Number of calibration nodes')
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if 'rate' in ylabel:
            ax.set_ylim(0, 1.05)

    plt.suptitle('Experiment 2 — Calibration Node Deployment Strategy\n'
                 '(high-homophily regime h=30, attack=10)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(get_output_dir(), 'exp2_calibration.png'),
                    dpi=150, bbox_inches='tight')
    if is_interactive(): plt.show()


    # Welch's t-test: best vs worst strategy at max calibration count
    n = data.get('runs', RUNS)
    best_dist  = min(DISTRIBUTIONS, key=lambda d: results[d]['peak'][-1])
    worst_dist = max(DISTRIBUTIONS, key=lambda d: results[d]['peak'][-1])
    m_b, s_b = results[best_dist]['peak'][-1],  results[best_dist]['std_peak'][-1]
    m_w, s_w = results[worst_dist]['peak'][-1], results[worst_dist]['std_peak'][-1]
    se = np.sqrt(s_b**2 / n + s_w**2 / n)
    if se > 0:
        t_stat = (m_w - m_b) / se
        df_w   = (s_b**2/n + s_w**2/n)**2 / (
                  (s_b**2/n)**2/(n-1) + (s_w**2/n)**2/(n-1))
        p_val  = 2 * scipy_stats.t.sf(abs(t_stat), df=df_w)
        sig    = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
        label  = (f'{best_dist.replace("_"," ")} vs {worst_dist.replace("_"," ")}\n'
                  f't = {t_stat:.1f},  p = {p_val:.4f}  {sig}')
        axes[1].text(0.98, 0.95, label,
                     transform=axes[1].transAxes, fontsize=7.5,
                     ha='right', va='top',
                     bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.85))
        print(f"  Best strategy: {best_dist} (peak={m_b:.3f})")
        print(f"  Worst strategy: {worst_dist} (peak={m_w:.3f})")
        print(f"  Welch's t-test: t={t_stat:.2f}, p={p_val:.4f} {sig}")

    plt.close('all')

if __name__ == '__main__':
    print("Running Experiment 2 — Calibration Node Strategy...")
    data = run()
    plot(data, save=(RUNS >= 100))
    print("Experiment 2 complete")
