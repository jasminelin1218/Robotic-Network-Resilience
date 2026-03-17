"""
experiments/exp9_disparity.py
==============================
Experiment 9 — Inter-Group Disparity: The Gaming Gap  (Topic 6 Core)

Narrative position  —  Act 3, step 2
--------------------------------------
Motivation (from Exp 8a):
  Exp 8a confirmed that the cascade dynamics scale well with N — the
  fraction infected stays roughly constant.  But "fraction infected"
  is a system-wide average that hides an important question: are all
  squads equally affected?  Homophily was designed to CONTAIN infection,
  but containment means Squad 0 (the attacked squad) bears the cost
  while other squads are protected.  Is that gap large and does it grow?

Gaming gap — formal definition introduced here:
  Δ(h) = infection_rate(Squad 0) − mean infection_rate(other squads)
  A positive Δ means Squad 0 is disproportionately burdened.
  This metric is used in Exp 10 to track fairness across feedback rounds.

Topic 6 mapping
---------------
Topic 6 asks for the "Gaming Gap":  ΔU = U_bar_C1 - U_bar_C2
  — how differently Community A and Community B are affected as h increases.

In our robotics model:
  C1 = Squad 0  (the attacked group  — analogous to Community A who has the info)
  C2 = all other squads  (analogous to Community B who is topologically isolated)

  ΔInfection(h) = infection_rate(Squad 0) - infection_rate(other squads)

Topic 6 hypothesis: as h increases, ΔInfection should INCREASE because
high homophily TRAPS the error inside Squad 0 (it cannot cross the community
boundary). At low h, errors diffuse everywhere equally.

This directly answers the core Topic 6 question:
  "Does increasing homophily linearly or exponentially increase the disparity?"

Required plots (from Topic 6 Phase 4)
--------------------------------------
Plot 1: ΔInfection vs homophily ratio h
Plot 2: ΔInfection vs λ₂ (algebraic connectivity)  — phase transition visible
Plot 3: Per-group infection curves (Squad 0 vs others) at several h values
"""
from __future__ import annotations
from typing import List, Dict
import numpy as np
import scipy.stats as stats
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import matplotlib; matplotlib.use('Agg')  # must be before pyplot
import matplotlib.pyplot as plt
from core.simulation import generate_network, initialize_agents, run_simulation
from core.metrics import algebraic_connectivity, cheeger_constant
from core.paths import get_output_dir, is_interactive

RUNS = 100
N    = 100
K    = 4       # 4 squads: Squad 0 is attacked, squads 1-3 are "other community"


def _run_single(N: int, K: int, p_in: float, p_out: float,
                attack_count: int) -> Dict[str, float]:
    """
    Run one simulation and return per-group infection rates.
    Returns:
        squad0_final  : final infection rate in Squad 0 (attacked group)
        others_final  : mean final infection rate in squads 1..K-1
        squad0_peak   : peak infection rate in Squad 0
        others_peak   : mean peak infection rate in squads 1..K-1
        delta_final   : squad0_final - others_final  (the Gaming Gap ΔU)
        delta_peak    : squad0_peak  - others_peak
    """
    G = generate_network(N, K, p_in, p_out)
    state, calib_set = initialize_agents(
        G, attack_count, 'concentrated', 0, 'scattered', K)

    res = run_simulation(G, state, calib_set,
                         alpha=0.6, beta=0.4, threshold=0.3,
                         r=3, p_c=0.3, max_steps=200)

    # Compute per-group infection at convergence
    final_state = state.copy()
    # Re-run to get final per-node state (run_simulation gives aggregate curve)
    cur = state.copy()
    for _ in range(200):
        nxt = cur.copy()
        changed = False
        for node in G.nodes():
            grp = G.nodes[node]['group']
            nb  = list(G.neighbors(node))
            in_nb  = [n for n in nb if G.nodes[n]['group'] == grp]
            out_nb = [n for n in nb if G.nodes[n]['group'] != grp]
            if cur[node] == 0:
                ir  = sum(cur[n] for n in in_nb)  / len(in_nb)  if in_nb  else 0
                or_ = sum(cur[n] for n in out_nb) / len(out_nb) if out_nb else 0
                score = (0.6+0.4)*ir if not out_nb else (
                         (0.6+0.4)*or_ if not in_nb else 0.6*ir + 0.4*or_)
                if score >= 0.3:
                    nxt[node] = 1; changed = True
            else:
                hi = sum(1 for n in in_nb if cur[n] == 0)
                if hi >= 3:
                    if np.random.random() < 0.3:
                        nxt[node] = 0; changed = True
        if not changed:
            break
        cur = nxt

    # Group sizes
    from collections import defaultdict
    by_group: Dict[int, List] = defaultdict(list)
    for node in G.nodes():
        by_group[G.nodes[node]['group']].append(node)

    # Squad 0 infection rate
    s0_nodes = by_group[0]
    s0_final = sum(cur[n] for n in s0_nodes) / len(s0_nodes) if s0_nodes else 0

    # Other squads infection rate
    other_nodes = [n for g, ns in by_group.items() if g != 0 for n in ns]
    other_final = sum(cur[n] for n in other_nodes) / len(other_nodes) if other_nodes else 0

    # Peak from infection_over_time curve — approximate per group using final ratio
    peak_total = res['peak_infection']

    return {
        'squad0_final': s0_final,
        'others_final': other_final,
        'delta_final':  s0_final - other_final,
        'total_peak':   peak_total,
        'total_final':  res['final_infection'],
    }


def run(runs: int = RUNS, verbose: bool = True) -> dict:
    """
    Sweep h = p_in/p_out and measure inter-group disparity ΔInfection at each h.
    Also compute λ₂ and Cheeger for the spectral correlation plots.
    """
    p_out_range  = np.array([0.015, 0.02, 0.03, 0.04, 0.05,
                              0.07,  0.10, 0.13, 0.17, 0.20])
    p_in         = 0.6
    h_values     = p_in / p_out_range
    attack_count = 8

    results = {
        'h':             [],
        'p_out':         [],
        'lam2':          [],
        'cheeger':       [],
        'squad0_final':  [],   # infection in attacked group
        'others_final':  [],   # infection in other groups
        'delta_final':   [],   # ΔInfection = Squad0 - Others  (the Gaming Gap)
        'std_delta':     [],
        'std_squad0':    [],
        'std_others':    [],
    }

    for p_out in p_out_range:
        h = p_in / p_out

        # Compute spectral metrics (averaged over multiple network samples)
        lam2_buf, cheeger_buf = [], []
        for _ in range(min(runs, 30)):
            G = generate_network(N, K, p_in, p_out)
            lam2_buf.append(algebraic_connectivity(G))
            cheeger_buf.append(cheeger_constant(G))

        # Monte Carlo simulation for disparity
        s0_buf, ot_buf, dt_buf = [], [], []
        for _ in range(runs):
            r = _run_single(N, K, p_in, p_out, attack_count)
            s0_buf.append(r['squad0_final'])
            ot_buf.append(r['others_final'])
            dt_buf.append(r['delta_final'])

        results['h'].append(h)
        results['p_out'].append(p_out)
        results['lam2'].append(float(np.mean(lam2_buf)))
        results['cheeger'].append(float(np.mean(cheeger_buf)))
        results['squad0_final'].append(float(np.mean(s0_buf)))
        results['others_final'].append(float(np.mean(ot_buf)))
        results['delta_final'].append(float(np.mean(dt_buf)))
        results['std_delta'].append(float(np.std(dt_buf)))
        results['std_squad0'].append(float(np.std(s0_buf)))
        results['std_others'].append(float(np.std(ot_buf)))

        if verbose:
            print(f"  h={h:5.1f}  lambda2={results['lam2'][-1]:.4f}  "
                  f"cheeger={results['cheeger'][-1]:.4f}  "
                  f"Squad0={results['squad0_final'][-1]:.3f}  "
                  f"Others={results['others_final'][-1]:.3f}  "
                  f"Delta={results['delta_final'][-1]:+.3f}")

    # Convert to arrays
    for k in results:
        results[k] = np.array(results[k])

    # Pearson correlations
    r_delta_h,   p_delta_h   = stats.pearsonr(results['h'],
                                               results['delta_final'])
    r_delta_lam, p_delta_lam = stats.pearsonr(results['lam2'],
                                               results['delta_final'])
    r_delta_ch,  p_delta_ch  = stats.pearsonr(results['cheeger'],
                                               results['delta_final'])

    print(f"\n  Pearson r (delta vs h):       r={r_delta_h:+.4f}  (p={p_delta_h:.4f})")
    print(f"  Pearson r (delta vs lambda2): r={r_delta_lam:+.4f}  (p={p_delta_lam:.4f})")
    print(f"  Pearson r (delta vs cheeger): r={r_delta_ch:+.4f}  (p={p_delta_ch:.4f})")

    results['r_delta_h']   = r_delta_h
    results['p_delta_h']   = p_delta_h
    results['r_delta_lam'] = r_delta_lam
    results['p_delta_lam'] = p_delta_lam
    results['r_delta_ch']  = r_delta_ch
    results['p_delta_ch']  = p_delta_ch
    results['attack_count'] = attack_count

    return results


def plot(data: dict, save: bool = True) -> None:
    h        = data['h']
    lam2     = data['lam2']
    cheeger  = data['cheeger']
    s0       = data['squad0_final']
    ot       = data['others_final']
    delta    = data['delta_final']
    std_d    = data['std_delta']
    std_s0   = data['std_squad0']
    std_ot   = data['std_others']

    # ── Plot 1: Per-group infection rates AND delta vs h ─────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel A: both groups vs h
    axes[0].plot(h, s0, 'o-', color='#E24B4A', linewidth=2.5,
                 label='Squad 0 (attacked group)')
    axes[0].fill_between(h, np.clip(s0-std_s0,0,1), np.clip(s0+std_s0,0,1),
                          alpha=0.12, color='#E24B4A')
    axes[0].plot(h, ot, 's-', color='#378ADD', linewidth=2.5,
                 label='Squads 1–3 (other groups)')
    axes[0].fill_between(h, np.clip(ot-std_ot,0,1), np.clip(ot+std_ot,0,1),
                          alpha=0.12, color='#378ADD')
    axes[0].set_xlabel('Homophily ratio h = p_in / p_out', fontsize=11)
    axes[0].set_ylabel('Final infection rate', fontsize=11)
    axes[0].set_title('Per-group infection vs h\n'
                       'High h traps errors in Squad 0', fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1.05)

    # Panel B: ΔInfection vs h  — THE Gaming Gap plot
    sig_h = ('***' if data['p_delta_h'] < 0.001 else
              '**'  if data['p_delta_h'] < 0.01  else
              '*'   if data['p_delta_h'] < 0.05  else 'n.s.')
    axes[1].plot(h, delta, 'D-', color='#1D9E75', linewidth=2.5, markersize=8,
                 label=f'ΔInfection = Squad0 − Others\nr={data["r_delta_h"]:+.3f} {sig_h}')
    axes[1].fill_between(h, np.clip(delta-std_d,-1,1),
                          np.clip(delta+std_d,-1,1),
                          alpha=0.15, color='#1D9E75')
    axes[1].axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    axes[1].set_xlabel('Homophily ratio h = p_in / p_out', fontsize=11)
    axes[1].set_ylabel('ΔInfection (Squad 0 − Others)', fontsize=11)
    axes[1].set_title(f'Inter-group disparity vs h\n'
                       f'Topic 6 "Gaming Gap" ΔU = Ū_C1 − Ū_C2',
                       fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # Panel C: ΔInfection vs λ₂ — spectral correlation
    sig_lam = ('***' if data['p_delta_lam'] < 0.001 else
               '**'  if data['p_delta_lam'] < 0.01  else
               '*'   if data['p_delta_lam'] < 0.05  else 'n.s.')
    sc = axes[2].scatter(lam2, delta, c=h, cmap='RdYlGn_r',
                          s=120, zorder=3, edgecolors='white', linewidths=0.5)
    plt.colorbar(sc, ax=axes[2], label='h (homophily)')
    try:
        m, b = np.polyfit(lam2, delta, 1)
        x_fit = np.linspace(lam2.min(), lam2.max(), 100)
        axes[2].plot(x_fit, m*x_fit+b, '--', color='#1D9E75', linewidth=2,
                     label=f'r={data["r_delta_lam"]:+.3f} {sig_lam}')
    except Exception:
        pass
    axes[2].set_xlabel('Algebraic connectivity λ₂', fontsize=11)
    axes[2].set_ylabel('ΔInfection (Squad 0 − Others)', fontsize=11)
    axes[2].set_title(f'Disparity vs λ₂\n'
                       f'Phase transition below critical λ₂',
                       fontweight='bold')
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Experiment 9 — Inter-Group Information Disparity\n'
                 'Topic 6 "Gaming Gap": ΔU = Ū_C1 − Ū_C2  '
                 f'(attack={data["attack_count"]} seeds in Squad 0)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(get_output_dir(), 'exp9_disparity.png'),
                    dpi=150, bbox_inches='tight')
    if is_interactive(): plt.show()
    plt.close('all')

    # ── Plot 2: Cheeger vs delta + comparison panel ───────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Cheeger vs disparity
    sig_ch = ('***' if data['p_delta_ch'] < 0.001 else
              '**'  if data['p_delta_ch'] < 0.01  else
              '*'   if data['p_delta_ch'] < 0.05  else 'n.s.')
    sc2 = axes[0].scatter(cheeger, delta, c=h, cmap='RdYlGn_r',
                           s=120, zorder=3, edgecolors='white', linewidths=0.5)
    plt.colorbar(sc2, ax=axes[0], label='h (homophily)')
    try:
        m2, b2 = np.polyfit(cheeger, delta, 1)
        xf2 = np.linspace(cheeger.min(), cheeger.max(), 100)
        axes[0].plot(xf2, m2*xf2+b2, '--', color='#534AB7', linewidth=2,
                     label=f'r={data["r_delta_ch"]:+.3f} {sig_ch}')
    except Exception:
        pass
    axes[0].set_xlabel('Cheeger constant (conductance)', fontsize=11)
    axes[0].set_ylabel('ΔInfection (Squad 0 − Others)', fontsize=11)
    axes[0].set_title('Disparity vs Cheeger constant\n'
                       'Lower conductance → higher disparity',
                       fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Summary: both spectral metrics vs h on one axis
    ax_r = axes[1].twinx()
    lam2_n    = (lam2-lam2.min())/(lam2.max()-lam2.min()+1e-9)
    cheeger_n = (cheeger-cheeger.min())/(cheeger.max()-cheeger.min()+1e-9)
    delta_n   = (delta-delta.min())/(delta.max()-delta.min()+1e-9)

    axes[1].plot(h, delta, 'D-', color='#1D9E75', linewidth=2.5,
                 label='ΔInfection (disparity)')
    ax_r.plot(h, lam2_n,    's--', color='#378ADD', linewidth=1.5,
              alpha=0.8, label='λ₂ (normalised)')
    ax_r.plot(h, cheeger_n, '^:',  color='#534AB7', linewidth=1.5,
              alpha=0.8, label='Cheeger (normalised)')

    axes[1].set_xlabel('Homophily ratio h', fontsize=11)
    axes[1].set_ylabel('ΔInfection disparity', color='#1D9E75', fontsize=11)
    ax_r.set_ylabel('Normalised spectral metric', fontsize=11)
    axes[1].set_title('Disparity and spectral metrics vs h\n'
                       'As h↑: λ₂↓, Cheeger↓, Disparity↑',
                       fontweight='bold')
    l1, lb1 = axes[1].get_legend_handles_labels()
    l2, lb2 = ax_r.get_legend_handles_labels()
    axes[1].legend(l1+l2, lb1+lb2, fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Experiment 9 — Spectral Properties vs Inter-Group Disparity\n'
                 'Lower λ₂ and Cheeger constant → stronger information bottleneck → '
                 'higher disparity',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(get_output_dir(), 'exp9_disparity_spectral.png'),
                    dpi=150, bbox_inches='tight')
    if is_interactive(): plt.show()
    plt.close('all')

    # Print summary
    max_delta_idx = int(np.argmax(delta))
    print(f"\n  Max disparity: ΔInfection = {delta[max_delta_idx]:+.3f} "
          f"at h = {h[max_delta_idx]:.1f}")
    print(f"  At h={h[0]:.0f} (low): Squad0={s0[0]:.3f}, "
          f"Others={ot[0]:.3f}, Δ={delta[0]:+.3f}")
    print(f"  At h={h[-1]:.0f} (high): Squad0={s0[-1]:.3f}, "
          f"Others={ot[-1]:.3f}, Δ={delta[-1]:+.3f}")
    print(f"  Interpretation: "
          f"{'high h traps errors in Squad 0 (disparity increases)' if delta[-1] > delta[0] else 'uniform spread regardless of h'}")


if __name__ == '__main__':
    print("Running Experiment 9 — Inter-Group Disparity...")
    data = run()
    plot(data, save=(RUNS >= 100))
    print("Experiment 9 complete")
