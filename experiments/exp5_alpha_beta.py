"""
experiments/exp5_alpha_beta.py
================================
Experiment 5 — In/Out-group Weight Sensitivity

High alpha (0.8,0.2): errors stay local — need high in-group infection to spread
High beta  (0.2,0.8): out-group signals dominate — errors and corrections spread faster

Clear dynamics visible at thresholds 0.15–0.4.
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

ALPHA_BETA_PAIRS = [(0.8, 0.2), (0.5, 0.5), (0.2, 0.8)]
COLORS           = ['#E24B4A', '#378ADD', '#1D9E75']
THRESHOLDS       = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]


def run(runs: int = RUNS, verbose: bool = True) -> dict:
    results = {}
    for (a, b) in ALPHA_BETA_PAIRS:
        peaks, finals, std_p, std_f = [], [], [], []
        for thr in THRESHOLDS:
            res = monte_carlo(
                runs, N=N, K=4, p_in=0.6, p_out=0.04,
                attack_count=8, attack_distribution='concentrated',
                calib_count=0, calib_distribution='scattered',
                alpha=a, beta=b, threshold=thr,
                r=3, p_c=0.3, max_steps=200,
            )
            peaks.append(res['mean_peak'])
            finals.append(res['mean_final'])
            std_p.append(res['std_peak'])
            std_f.append(res['std_final'])
            if verbose:
                print(f"  a={a}, b={b}, thr={thr:.2f} -> "
                      f"peak={res['mean_peak']:.3f}")
        results[(a, b)] = dict(peaks=peaks, finals=finals,
                               std_peaks=std_p, std_finals=std_f)
    return dict(results=results)


def plot(data: dict, save: bool = True) -> None:
    results = data['results']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for (a, b), color in zip(ALPHA_BETA_PAIRS, COLORS):
        d   = results[(a, b)]
        lbl = f'alpha={a}, beta={b}'
        for ax, metric, std_key in zip(
            axes,
            ['peaks', 'finals'],
            ['std_peaks', 'std_finals'],
        ):
            y   = np.array(d[metric])
            std = np.array(d[std_key])
            ax.plot(THRESHOLDS, y, 'o-', color=color, linewidth=2, label=lbl)
            ax.fill_between(THRESHOLDS,
                            np.clip(y-std, 0, 1),
                            np.clip(y+std, 0, 1),
                            alpha=0.10, color=color)

    for ax, title in zip(axes, ['Peak infection rate', 'Final infection rate']):
        ax.set_xlabel('Infection threshold')
        ax.set_ylabel(title)
        ax.set_title(title, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    plt.suptitle('Experiment 5 — In/Out-group Trust Weight (alpha, beta) Sensitivity\n'
                 'High alpha = in-group echo chamber; high beta = out-group influence',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(get_output_dir(), 'exp5_alpha_beta.png'),
                    dpi=150, bbox_inches='tight')
    if is_interactive(): plt.show()


    plt.close('all')

if __name__ == '__main__':
    print("Running Experiment 5 — alpha/beta Sensitivity...")
    data = run()
    plot(data, save=(RUNS >= 100))
    print("Experiment 5 complete")
