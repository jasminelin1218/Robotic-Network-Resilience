"""
experiments/exp4_dynamic_defense.py
=====================================
Experiment 4 — Dynamic Defense: Adaptive Edge Severance

Narrative position  —  Act 2, step 4
--------------------------------------
Motivation (from Exp 5):
  Exp 5 showed that the trust weight balance (α vs β) shifts the
  cascade threshold, but does not eliminate cascades.  We need a
  runtime intervention.  The model supports dynamic defense:
  when the out-group infected fraction exceeds a threshold h_sev,
  the squad severs its cross-squad edges for the remainder of the run.
  This mirrors quarantine protocols in biological or cyber networks.

Hypothesis:
  Dynamic severance at h_sev ≈ 0.3 will cut peak infection by ≥40%
  relative to a static (non-severing) network, because it prevents
  cascade propagation across squad boundaries before it reaches a
  tipping point.  The optimal h_sev will be low enough to trigger
  early, but not so low that it also severs recovery pathways.

What we measure:
  Sweep severance threshold h_sev ∈ [0.3, 0.9].
  Compare peak and final infection against static baseline.

Handoff → Exp 6:
  Dynamic defense works — but only in certain (h, attack) regimes.
  Exp 6 maps the exact boundary: the h_crit curve that separates
  recovery from collapse across the full 2-D parameter space.

At h=20 (p_out=0.03), attack=10: static gives peak~0.74.
Dynamic severance at h_sev=0.3 cuts peak to ~0.29 — strong effect.
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


def run(runs: int = RUNS, verbose: bool = True) -> dict:
    severance_thresholds = np.linspace(0.3, 0.9, 7)
    base_kwargs = dict(
        N=N, K=4, p_in=0.6, p_out=0.03,
        attack_count=10, attack_distribution='concentrated',
        calib_count=0, calib_distribution='scattered',
        alpha=0.6, beta=0.4, threshold=0.3,
        r=3, p_c=0.3, max_steps=200,
    )

    static_res = monte_carlo(runs, use_dynamic_defense=False, **base_kwargs)
    if verbose:
        print(f"  Static baseline -> peak={static_res['mean_peak']:.3f}, "
              f"final={static_res['mean_final']:.3f}")

    dyn = {'peak': [], 'final': [], 'conv': [], 'std_peak': [], 'std_final': []}
    for h_sev in severance_thresholds:
        res = monte_carlo(runs, use_dynamic_defense=True,
                          severance_h=float(h_sev), **base_kwargs)
        dyn['peak'].append(res['mean_peak'])
        dyn['final'].append(res['mean_final'])
        dyn['conv'].append(res['mean_conv'])
        dyn['std_peak'].append(res['std_peak'])
        dyn['std_final'].append(res['std_final'])
        if verbose:
            print(f"  h_sev={h_sev:.2f} -> peak={res['mean_peak']:.3f}, "
                  f"final={res['mean_final']:.3f}")

    return dict(severance_thresholds=severance_thresholds,
                dyn=dyn, static_res=static_res, runs=runs)


def plot(data: dict, save: bool = True) -> None:
    sv     = data['severance_thresholds']
    dyn    = data['dyn']
    static = data['static_res']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, std_key, title in zip(
        axes,
        ['peak', 'final'],
        ['std_peak', 'std_final'],
        ['Peak infection rate', 'Final infection rate'],
    ):
        y   = np.array(dyn[metric])
        std = np.array(dyn[std_key])
        ax.plot(sv, y, 'o-', color='steelblue', linewidth=2,
                label='Dynamic defense')
        ax.fill_between(sv, np.clip(y-std,0,1), np.clip(y+std,0,1),
                        alpha=0.12, color='steelblue')
        stat_val = static[f'mean_{metric}']
        ax.axhline(stat_val, color='crimson', linestyle='--',
                   linewidth=2, label=f'Static baseline ({stat_val:.2f})')

        best_idx = int(np.argmin(y))
        ax.axvline(sv[best_idx], color='steelblue', linestyle=':',
                   linewidth=1.5, label=f'Optimal h_sev={sv[best_idx]:.2f}')

        ax.set_xlabel('Severance threshold h_sev')
        ax.set_ylabel(title)
        ax.set_title(title, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    plt.suptitle('Experiment 4 — Dynamic Defense vs Static Network\n'
                 '(h=20 network, attack=10 seeds)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(get_output_dir(), 'exp4_dynamic_defense.png'),
                    dpi=150, bbox_inches='tight')
    if is_interactive(): plt.show()
    plt.close('all')

    best_idx = int(np.argmin(dyn['peak']))
    best_h   = sv[best_idx]
    improve  = static['mean_peak'] - min(dyn['peak'])
    print(f"  Optimal severance: h_sev = {best_h:.2f}")
    print(f"  Peak improvement vs static: {improve:.3f} "
          f"({improve/static['mean_peak']*100:.0f}% reduction)")

    # Welch's t-test: static baseline vs best dynamic threshold
    n = data.get('runs', RUNS)
    m_s, s_s = static['mean_peak'], static['std_peak']
    m_d, s_d = dyn['peak'][best_idx], dyn['std_peak'][best_idx]
    se = np.sqrt(s_s**2 / n + s_d**2 / n)
    if se > 0:
        t_stat = (m_s - m_d) / se
        df_w   = (s_s**2/n + s_d**2/n)**2 / (
                  (s_s**2/n)**2/(n-1) + (s_d**2/n)**2/(n-1))
        p_val  = 2 * scipy_stats.t.sf(abs(t_stat), df=df_w)
        sig    = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
        label  = f'Static vs best dynamic\nt = {t_stat:.1f},  p = {p_val:.4f}  {sig}'
        axes[0].text(0.98, 0.95, label,
                     transform=axes[0].transAxes, fontsize=8,
                     ha='right', va='top',
                     bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.85))
        print(f"  Welch's t-test: t={t_stat:.2f}, p={p_val:.4f} {sig}")


if __name__ == '__main__':
    print("Running Experiment 4 — Dynamic Defense...")
    data = run()
    plot(data, save=(RUNS >= 100))
    print("Experiment 4 complete")
