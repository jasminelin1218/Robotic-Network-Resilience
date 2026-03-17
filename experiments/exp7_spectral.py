"""
experiments/exp7_spectral.py
==============================
Experiment 7 — Spectral Analysis (λ₂ as Vulnerability Predictor)

Narrative position  —  Act 1, step 2
--------------------------------------
Motivation (from Exp 8b):
  Exp 8b confirmed SBM captures qualitatively different dynamics than
  unstructured graphs — community structure IS the mechanism.
  But WHAT structural quantity determines vulnerability?

Hypothesis:
  Algebraic connectivity λ₂ (second-smallest Laplacian eigenvalue)
  measures global network cohesion.  We predict a NEGATIVE correlation:
  lower λ₂ → slower mixing → faster local collapse → higher peak/final
  infection.  If confirmed, λ₂ is the single-number "vulnerability dial"
  and designers can tune p_out (hence λ₂) as a control knob.

What we measure:
  Sweep p_out (homophily h = p_in/p_out).  At each h compute λ₂ and
  run Monte Carlo infection simulations.  Report Pearson r between λ₂
  and both peak and final infection.

Handoff → Exp 1:
  λ₂ predicts spread; now map the full K × h space to find the
  safe operating zone — which is exactly what Exp 1 does.

Uses same parameter regime (alpha=0.6, beta=0.4, threshold=0.3, p_c=0.3)
so lambda2 results are directly comparable to all other experiments.
"""
from __future__ import annotations
from typing import List
import numpy as np
import scipy.stats as stats
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import matplotlib; matplotlib.use('Agg')  # must be before pyplot
import matplotlib.pyplot as plt
from core.paths import get_output_dir, is_interactive
from core.simulation import generate_network, initialize_agents, run_simulation
from core.metrics import algebraic_connectivity, cheeger_constant

N      = 100


def run(trials: int = 30, verbose: bool = True) -> dict:
    p_out_range = np.array([0.01, 0.015, 0.02, 0.03, 0.04,
                            0.05, 0.07, 0.10, 0.13, 0.17, 0.20])
    K = 4

    lam2_list, peak_list, final_list, cheeger_list = [], [], [], []

    for p_out in p_out_range:
        lam2_buf, pk_buf, fin_buf, cheeger_buf = [], [], [], []
        for _ in range(trials):
            G = generate_network(N, K, p_in=0.6, p_out=p_out)
            lam2 = algebraic_connectivity(G)

            state, calib_set = initialize_agents(
                G, 10, 'concentrated', 0, 'scattered', K)
            res = run_simulation(G, state, calib_set,
                                 alpha=0.6, beta=0.4, threshold=0.3,
                                 r=3, p_c=0.3, max_steps=200)

            lam2_buf.append(lam2)
            pk_buf.append(res['peak_infection'])
            fin_buf.append(res['final_infection'])
            cheeger_buf.append(cheeger_constant(G))

        lam2_list.append(float(np.mean(lam2_buf)))
        peak_list.append(float(np.mean(pk_buf)))
        final_list.append(float(np.mean(fin_buf)))
        cheeger_list.append(float(np.mean(cheeger_buf)))
        if verbose:
            print(f"  p_out={p_out:.3f} -> lambda2={lam2_list[-1]:.4f}, "
                  f"peak={peak_list[-1]:.3f}, final={final_list[-1]:.3f}")

    r_peak,  p_peak  = stats.pearsonr(lam2_list, peak_list)
    r_final, p_final = stats.pearsonr(lam2_list, final_list)
    print(f"\n  Pearson r (lambda2 vs peak):  r={r_peak:+.4f}  (p={p_peak:.4f})")
    print(f"  Pearson r (lambda2 vs final): r={r_final:+.4f}  (p={p_final:.4f})")

    return dict(p_out_range=p_out_range,
                lam2_list=lam2_list, peak_list=peak_list, final_list=final_list,
                cheeger_list=cheeger_list,
                r_peak=r_peak, p_peak=p_peak,
                r_final=r_final, p_final=p_final)


def plot(data: dict, save: bool = True) -> None:
    lam2  = np.array(data['lam2_list'])
    peaks = np.array(data['peak_list'])
    fins  = np.array(data['final_list'])
    p_out = data['p_out_range']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, y, r_val, p_val, ylabel, color in zip(
        axes,
        [peaks, fins],
        [data['r_peak'], data['r_final']],
        [data['p_peak'], data['p_final']],
        ['Peak infection rate', 'Final infection rate'],
        ['#E24B4A', '#378ADD'],
    ):
        sc = ax.scatter(lam2, y, c=p_out, cmap='viridis',
                        s=100, zorder=3, edgecolors='white', linewidths=0.5)
        plt.colorbar(sc, ax=ax, label='p_out')

        m, b = np.polyfit(lam2, y, 1)
        x_fit = np.linspace(lam2.min(), lam2.max(), 100)
        sig = ('***' if p_val < 0.001 else
               '**'  if p_val < 0.01  else
               '*'   if p_val < 0.05  else 'n.s.')
        ax.plot(x_fit, m * x_fit + b, '--', color=color, linewidth=2,
                label=f'Linear fit  r={r_val:+.3f} {sig}')

        ax.set_xlabel('Algebraic connectivity lambda2 (Fiedler value)', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f'lambda2 vs {ylabel}\n'
                     f'Pearson r = {r_val:+.4f}  (p = {p_val:.4f})',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    plt.suptitle('Experiment 7 — Spectral Analysis: Algebraic Connectivity vs Infection',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(get_output_dir(), 'exp7_spectral.png'),
                    dpi=150, bbox_inches='tight')
    if is_interactive(): plt.show()

    # Fiedler interpretation: mixing time 1/lambda2 vs peak
    fig, ax = plt.subplots(figsize=(8, 4))
    mixing_norm = (1.0 / (lam2 + 1e-6))
    mixing_norm = mixing_norm / mixing_norm.max()
    ax2 = ax.twinx()
    ax.plot(p_out, peaks, 'o-', color='#E24B4A', linewidth=2,
            label='Peak infection')
    ax2.plot(p_out, mixing_norm, 's--', color='gray', linewidth=1.5,
             alpha=0.7, label='Mixing time 1/lambda2 (normalised)')
    ax.set_xlabel('p_out (inter-group edge probability)', fontsize=11)
    ax.set_ylabel('Peak infection rate', color='#E24B4A', fontsize=11)
    ax2.set_ylabel('Normalised mixing time 1/lambda2', color='gray', fontsize=11)
    ax.set_title('Fiedler theorem: slower mixing localises errors\n'
                 '(high h = low lambda2 = slow mixing = errors trapped in squad)',
                 fontsize=11)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(get_output_dir(), 'exp7_fiedler.png'),
                    dpi=150, bbox_inches='tight')
    if is_interactive(): plt.show()


    # ── Cheeger constant analysis (Topic 6 requirement) ─────────────────────
    cheeger = np.array(data['cheeger_list'])
    import scipy.stats as _stats
    r_ch, p_ch = _stats.pearsonr(cheeger, peaks)
    r_cl, p_cl = _stats.pearsonr(cheeger, lam2)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Cheeger vs peak infection
    sc = axes[0].scatter(cheeger, peaks, c=p_out, cmap='viridis',
                         s=100, zorder=3, edgecolors='white', linewidths=0.5)
    plt.colorbar(sc, ax=axes[0], label='p_out')
    try:
        m, b = np.polyfit(cheeger, peaks, 1)
    except Exception:
        m, b = 0, np.mean(peaks)
    x_fit = np.linspace(cheeger.min(), cheeger.max(), 100)
    sig = ('***' if p_ch < 0.001 else '**' if p_ch < 0.01 else
           '*' if p_ch < 0.05 else 'n.s.')
    axes[0].plot(x_fit, m*x_fit+b, '--', color='#E24B4A', linewidth=2,
                 label=f'r={r_ch:+.3f} {sig}')
    axes[0].set_xlabel('Cheeger constant (conductance)', fontsize=11)
    axes[0].set_ylabel('Peak infection rate', fontsize=11)
    axes[0].set_title(f'Cheeger vs peak infection\nPearson r = {r_ch:+.4f}  p={p_ch:.4f}',
                      fontweight='bold')
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

    # Panel 2: Cheeger vs lambda2
    axes[1].scatter(cheeger, lam2, c=p_out, cmap='viridis',
                    s=100, zorder=3, edgecolors='white', linewidths=0.5)
    try:
        m2, b2 = np.polyfit(cheeger, lam2, 1)
    except Exception:
        m2, b2 = 0, np.mean(lam2)
    axes[1].plot(x_fit, m2*x_fit+b2, '--', color='#378ADD', linewidth=2,
                 label=f'r={r_cl:+.3f}')
    axes[1].set_xlabel('Cheeger constant (conductance)', fontsize=11)
    axes[1].set_ylabel('Algebraic connectivity lambda2', fontsize=11)
    axes[1].set_title(f'Cheeger vs lambda2\nr={r_cl:+.3f} — both measure bottlenecks',
                      fontweight='bold')
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

    # Panel 3: infection + both metrics vs p_out
    lam2_n    = (lam2 - lam2.min())    / (lam2.max()    - lam2.min() + 1e-9)
    cheeger_n = (cheeger - cheeger.min()) / (cheeger.max() - cheeger.min() + 1e-9)
    ax3  = axes[2]
    ax3b = ax3.twinx()
    ax3.plot(p_out, peaks, 'o-', color='#E24B4A', linewidth=2, label='Peak infection')
    ax3b.plot(p_out, lam2_n,    's--', color='#378ADD', linewidth=1.5,
              alpha=0.8, label='lambda2 (normalised)')
    ax3b.plot(p_out, cheeger_n, '^:',  color='#1D9E75', linewidth=1.5,
              alpha=0.8, label='Cheeger (normalised)')
    ax3.set_xlabel('p_out', fontsize=11)
    ax3.set_ylabel('Peak infection rate', color='#E24B4A', fontsize=11)
    ax3b.set_ylabel('Normalised metric', fontsize=11)
    ax3.set_title('Infection vs lambda2 and Cheeger\nboth decrease as p_out increases',
                  fontweight='bold')
    lines1, lab1 = ax3.get_legend_handles_labels()
    lines2, lab2 = ax3b.get_legend_handles_labels()
    ax3.legend(lines1+lines2, lab1+lab2, fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.suptitle('Experiment 7 — Cheeger Constant Analysis (Topic 6 requirement)\n'
                 'Conductance measures information flow bottlenecks in homophilic networks',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(get_output_dir(), 'exp7_cheeger.png'),
                    dpi=150, bbox_inches='tight')
    if is_interactive(): plt.show()
    plt.close()
    print(f"  Cheeger vs peak: r={r_ch:+.4f}  (p={p_ch:.4f})")
    print(f"  Cheeger vs lambda2: r={r_cl:+.4f}  (both measure bottlenecks)")

if __name__ == '__main__':
    print("Running Experiment 7 — Spectral Analysis...")
    data = run()
    plot(data, save=(RUNS >= 100))
    print("Experiment 7 complete")
