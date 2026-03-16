"""
experiments/exp6_phase_diagram.py
===================================
Experiment 6 — Phase Diagram (NEW)

2D sweep of homophily h x attack intensity.
Shows exact recovery/collapse boundary — the h_crit curve.
"""
from __future__ import annotations
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import matplotlib; matplotlib.use('Agg')  # must be before pyplot
import matplotlib.pyplot as plt
from core.paths import get_output_dir, is_interactive
from core.simulation import monte_carlo

RUNS   = 60
N      = 100


def run(runs: int = RUNS, verbose: bool = True) -> dict:
    p_out_values  = np.array([0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.12, 0.15])
    attack_values = [1, 3, 5, 8, 10, 12, 15, 20]
    h_values      = 0.6 / p_out_values

    grid_final = np.zeros((len(p_out_values), len(attack_values)))
    grid_peak  = np.zeros((len(p_out_values), len(attack_values)))

    for i, p_out in enumerate(p_out_values):
        for j, atk in enumerate(attack_values):
            res = monte_carlo(
                runs, N=N, K=4, p_in=0.6, p_out=p_out,
                attack_count=atk, attack_distribution='concentrated',
                calib_count=0, calib_distribution='scattered',
                alpha=0.6, beta=0.4, threshold=0.3,
                r=3, p_c=0.3, max_steps=200,
            )
            grid_final[i, j] = res['mean_final']
            grid_peak[i, j]  = res['mean_peak']
            if verbose:
                print(f"  h={0.6/p_out:5.1f}, attack={atk:2d} -> "
                      f"final={res['mean_final']:.3f}")

    return dict(h_values=h_values, attack_values=attack_values,
                p_out_values=p_out_values,
                grid_final=grid_final, grid_peak=grid_peak)


def plot(data: dict, save: bool = True) -> None:
    h_vals  = data['h_values']
    a_vals  = data['attack_values']
    gf      = data['grid_final']
    gp      = data['grid_peak']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, grid, title, cmap in zip(
        axes,
        [gf, gp],
        ['Final infection rate', 'Peak infection rate'],
        ['RdYlGn_r', 'OrRd'],
    ):
        im = ax.imshow(grid, aspect='auto', origin='lower',
                       cmap=cmap, vmin=0, vmax=1,
                       extent=[min(a_vals)-0.5, max(a_vals)+0.5,
                               min(h_vals)-1,   max(h_vals)+1])
        plt.colorbar(im, ax=ax, label=title)

        # Phase boundary contours
        try:
            cs = ax.contour(a_vals, h_vals, grid,
                            levels=[0.10, 0.30, 0.50],
                            colors=['white', 'yellow', 'red'],
                            linewidths=[2.5, 1.5, 1.5],
                            linestyles=['-', '--', '--'])
            ax.clabel(cs, fmt='%.2f', fontsize=9, inline=True)
        except Exception:
            pass

        ax.set_xlabel('Attack intensity (# seed nodes)', fontsize=11)
        ax.set_ylabel('Homophily ratio h = p_in / p_out', fontsize=11)
        ax.set_title(title, fontweight='bold', fontsize=12)

    plt.suptitle('Experiment 6 — Phase Diagram: Homophily x Attack Intensity\n'
                 'White contour = 10% final infection (recovery boundary)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(get_output_dir(), 'exp6_phase_diagram.png'),
                    dpi=150, bbox_inches='tight')
    if is_interactive(): plt.show()

    # h_crit vs attack level
    # For each attack intensity, find the HIGHEST h at which final infection < 0.10
    # i.e. the largest h where the system still self-recovers
    # h_vals is DESCENDING [40, 30, 20, ...], so scan from LOW h (end) to HIGH h (start)
    fig, ax = plt.subplots(figsize=(9, 5))
    h_crits = []
    for j, atk in enumerate(a_vals):
        col  = gf[:, j]   # infection at each h value for this attack level
        crit = None
        # Scan from smallest h to largest h (reverse of array order)
        # Find the highest h where infection is still < 0.10
        # = last index (from the high-h end) where col < 0.10
        for i in range(len(h_vals)):  # i=0 is h=40 (highest), i=-1 is h=4 (lowest)
            if col[i] > 0.10:
                # This h is above critical — h_crit is just below here
                if i > 0:
                    # Interpolate between h_vals[i-1] (still infected) and h_vals[i]
                    frac = (0.10 - col[i-1]) / (col[i] - col[i-1] + 1e-9)
                    crit = h_vals[i-1] + frac * (h_vals[i] - h_vals[i-1])
                break
        h_crits.append(crit)

    valid_a  = [atk for atk, h in zip(a_vals, h_crits) if h is not None]
    valid_h  = [h   for h    in h_crits                if h is not None]

    if valid_a:
        ax.plot(valid_a, valid_h, 'o-', color='crimson', linewidth=2.5,
                markersize=8, label='h_crit (self-recovery boundary)')
        ax.fill_between(valid_a, 0, valid_h,
                        alpha=0.12, color='green', label='Recovery zone (h < h_crit)')
        ax.fill_between(valid_a, valid_h, max(h_vals) + 2,
                        alpha=0.08, color='red',   label='Collapse zone (h > h_crit)')
        # Annotate each point
        for a, h in zip(valid_a, valid_h):
            ax.annotate(f'h={h:.0f}', (a, h),
                        textcoords='offset points', xytext=(4, 4),
                        fontsize=8, color='crimson')
    else:
        ax.text(0.5, 0.5, 'No h_crit found\n(system collapses at all h values tested)',
                transform=ax.transAxes, ha='center', va='center', fontsize=11)

    ax.set_xlabel('Attack intensity (# seed nodes)', fontsize=12)
    ax.set_ylabel('Critical homophily h_crit', fontsize=12)
    ax.set_ylim(0, max(h_vals) + 3)
    ax.set_title('h_crit as a function of attack intensity\n'
                 'System self-recovers only when h < h_crit', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(get_output_dir(), 'exp6_hcrit_curve.png'),
                    dpi=150, bbox_inches='tight')
    if is_interactive(): plt.show()


    plt.close('all')

if __name__ == '__main__':
    print("Running Experiment 6 — Phase Diagram...")
    data = run()
    plot(data, save=(RUNS >= 100))
    print("Experiment 6 complete")
