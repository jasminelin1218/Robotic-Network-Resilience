"""
experiments/exp3_attack.py
===========================
Experiment 3 — Attack Robustness & Cascade Threshold

Cascade occurs between 8-12 seeds at p_out=0.05 — clearly visible.
"""
from __future__ import annotations
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import matplotlib; matplotlib.use('Agg')  # must be before pyplot
import matplotlib.pyplot as plt
from core.paths import get_output_dir, is_interactive
from core.simulation import monte_carlo
from core.metrics import find_cascade_threshold

RUNS   = 100
N      = 100

ATTACK_DISTRIBUTIONS = ['concentrated', 'scattered']
COLORS = {'concentrated': '#E24B4A', 'scattered': '#378ADD'}


def run(runs: int = RUNS, verbose: bool = True) -> dict:
    attack_intensities = [1, 3, 5, 8, 10, 12, 15, 20, 25]
    results = {d: {'peak': [], 'final': [], 'conv': [],
                   'std_peak': [], 'std_final': []}
               for d in ATTACK_DISTRIBUTIONS}

    for dist in ATTACK_DISTRIBUTIONS:
        for intensity in attack_intensities:
            res = monte_carlo(
                runs, N=N, K=4, p_in=0.6, p_out=0.05,
                attack_count=intensity, attack_distribution=dist,
                calib_count=0, calib_distribution='scattered',
                alpha=0.6, beta=0.4, threshold=0.3,
                r=3, p_c=0.3, max_steps=200,
            )
            results[dist]['peak'].append(res['mean_peak'])
            results[dist]['final'].append(res['mean_final'])
            results[dist]['conv'].append(res['mean_conv'])
            results[dist]['std_peak'].append(res['std_peak'])
            results[dist]['std_final'].append(res['std_final'])
            if verbose:
                print(f"  dist={dist:13s}, attack={intensity:2d} -> "
                      f"peak={res['mean_peak']:.3f}, final={res['mean_final']:.3f}")

    return dict(attack_intensities=attack_intensities, results=results)


def plot(data: dict, save: bool = True) -> None:
    ai      = data['attack_intensities']
    results = data['results']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for dist in ATTACK_DISTRIBUTIONS:
        col = COLORS[dist]
        for ax, metric, std_key, title in zip(
            axes,
            ['peak', 'final'],
            ['std_peak', 'std_final'],
            ['Peak infection rate', 'Final infection rate'],
        ):
            y   = np.array(results[dist][metric])
            std = np.array(results[dist][std_key])
            ax.plot(ai, y, 's-', label=dist.capitalize(),
                    color=col, linewidth=2)
            ax.fill_between(ai,
                            np.clip(y - std, 0, 1),
                            np.clip(y + std, 0, 1),
                            alpha=0.12, color=col)
            thresh = find_cascade_threshold(list(y), ai, cutoff=0.5)
            if thresh:
                ax.axvline(thresh, color=col, linestyle=':', linewidth=1.5, alpha=0.7)

    for ax, title in zip(axes, ['Peak infection rate', 'Final infection rate']):
        ax.axhline(0.5, color='gray', linestyle='--', linewidth=1,
                   alpha=0.5, label='Cascade threshold (0.5)')
        ax.set_xlabel('Attack intensity (# seed nodes)')
        ax.set_ylabel(title)
        ax.set_title(title, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    plt.suptitle('Experiment 3 — Attack Robustness\n'
                 '(dotted verticals = cascade threshold per strategy)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(get_output_dir(), 'exp3_attack.png'),
                    dpi=150, bbox_inches='tight')
    if is_interactive(): plt.show()
    plt.close('all')

    for dist in ATTACK_DISTRIBUTIONS:
        ct = find_cascade_threshold(results[dist]['peak'], ai)
        if ct:
            print(f"  Cascade threshold ({dist}): ~{ct:.1f} seeds")
        else:
            print(f"  Cascade threshold ({dist}): not reached in tested range")


if __name__ == '__main__':
    print("Running Experiment 3 — Attack Robustness...")
    data = run()
    plot(data, save=(RUNS >= 100))
    print("Experiment 3 complete")
