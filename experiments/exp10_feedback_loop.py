"""
experiments/exp10_feedback_loop.py
====================================
Experiment 10 — Adversarial Retraining Loop  (Topic 6 Phase 3, revised)

Narrative position  —  Act 3, step 3  (Act 3 + project conclusion)
-------------------------------------------------------------------
Motivation (from Exp 9):
  Exp 9 showed that the gaming gap Δ(h) is real and statistically
  significant — high-homophily networks protect other squads at the
  expense of Squad 0.  But the original Exp 10 had a critical flaw:
  the adversary always attacked Squad 0 across every round, so it
  could never exploit the defence's blind spots.

Revision (from SVG diagram):
  The new loop gives the adversary memory.  After each round t the
  adversary records ρₖ for ALL K squads and re-targets the least-
  infected one:
      k*(t+1) = argmin_k ρₖ(t)
  This models an adaptive attacker that learns which squad is currently
  least protected and pivots accordingly.

Two defence conditions:
  (A) Static defence:   fixed threshold=0.3 every round.
  (B) Adaptive defence: threshold tightens each round proportional to
      the infection seen in k* (the "retraining" analogue).

Four conditions compared (2×2):
  static_fixed    : static defence, adversary always attacks Squad 0
  adaptive_fixed  : adaptive defence, adversary always attacks Squad 0
  static_learning : static defence, adversary re-targets k*
  adaptive_learning: adaptive defence, adversary re-targets k*  ← new

New metric (from SVG):
  System-wide disparity = max_k(ρₖ) − min_k(ρₖ) over all rounds.
  This replaces the old ΔU = Squad0 − Others, which was only meaningful
  for a fixed-target adversary.

Four research questions (from SVG):
  Q1. Does adaptive defence still contain the disparity gap when the
      adversary shifts targets each round?
  Q2. Which squad ends up worst off — originally attacked Squad 0, or
      whichever is currently least defended?
  Q3. Does adaptive threshold learning generalise across squads, or only
      protect the squad it was learned on?
  Q4. Is there a steady-state where adversary and defence reach
      equilibrium, or does one side always win?

Topic 6 mapping
---------------
Topic 6 Phase 3 requires a T-round feedback loop:
  Round t: Seed → Propagate → React → Retrain → Repeat

In our revised model:
  Round t: k* = argmin ρₖ → Seeds injected into k*
           → Errors propagate (threshold contagion)
           → System partially recovers
           → Adaptive threshold updates on k*'s infection
           → Adversary observes all ρₖ, picks new k*
"""
from __future__ import annotations
from collections import defaultdict
from typing import List, Dict
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core.simulation import generate_network, run_simulation
from core.paths import get_output_dir, is_interactive

RUNS     = 100
N        = 100
K        = 4
T_ROUNDS = 10

# Four named conditions (defence × adversary strategy)
CONDITIONS = [
    ('static_fixed',     False, False),
    ('adaptive_fixed',   True,  False),
    ('static_learning',  False, True),
    ('adaptive_learning',True,  True),
]
COND_COLORS = {
    'static_fixed':     '#888780',
    'adaptive_fixed':   '#378ADD',
    'static_learning':  '#E24B4A',
    'adaptive_learning':'#1D9E75',
}
COND_LABELS = {
    'static_fixed':     'Static def / Fixed adv',
    'adaptive_fixed':   'Adaptive def / Fixed adv',
    'static_learning':  'Static def / Learning adv',
    'adaptive_learning':'Adaptive def / Learning adv ★',
}
SQUAD_COLORS = ['#E24B4A', '#378ADD', '#1D9E75', '#9B59B6']


# ── Simulation helpers ────────────────────────────────────────────────────────

def _run_round(G, state0: dict, calib_set: set,
               threshold: float, alpha: float = 0.6,
               beta: float = 0.4, r: int = 3,
               p_c: float = 0.3) -> tuple[dict, dict]:
    """
    Run one simulation round.

    Uses run_simulation with record_states=True so we can extract the
    final state directly (avoids re-simulating the same round twice).

    Returns
    -------
    final_state  : dict {node: 0|1}
    group_rates  : dict {group_id: infected_fraction}
    """
    res = run_simulation(
        G, state0, calib_set,
        alpha=alpha, beta=beta, threshold=threshold,
        r=r, p_c=p_c, max_steps=100, record_states=True,
    )
    final_state = res['state_history'][-1] if res['state_history'] else state0.copy()

    by_group: Dict[int, List] = defaultdict(list)
    for node in G.nodes():
        by_group[G.nodes[node]['group']].append(node)

    group_rates = {
        g: sum(final_state[n] for n in nodes) / len(nodes)
        for g, nodes in by_group.items()
    }
    return final_state, group_rates


def _simulate_feedback(G, attack_count: int, T: int,
                       adaptive_defense: bool,
                       adaptive_adversary: bool) -> dict:
    """
    Run T rounds of the attack-propagate-recover feedback loop.

    adaptive_defense  : if True, threshold tightens each round based on
                        k*'s infection rate (learning defender).
    adaptive_adversary: if True, adversary re-targets
                        k*(t+1) = argmin_k ρₖ(t) after each round.
                        If False, adversary always attacks Squad 0.

    Returns
    -------
    group_rates : np.ndarray (T, K)  per-squad infection rate each round
    disparity   : np.ndarray (T,)    max_k ρₖ − min_k ρₖ each round
    targets     : np.ndarray (T,)    int  which squad was attacked each round
    thresholds  : np.ndarray (T,)    threshold value used each round
    """
    base_threshold = 0.3
    threshold      = base_threshold

    group_rates_history: List[List[float]] = []
    thresholds_history:  List[float]       = []
    targets_history:     List[int]         = []

    state:     dict = {n: 0 for n in G.nodes()}
    calib_set: set  = set()
    k_star:    int  = 0      # adversary's current target; start with Squad 0

    for t in range(T):
        targets_history.append(k_star)

        # ── Inject attack seeds into k* ───────────────────────────────────
        state_t = state.copy()
        attack_pool = [n for n in G.nodes() if G.nodes[n]['group'] == k_star]
        if attack_pool:
            seeds = np.random.choice(
                attack_pool,
                size=min(attack_count, len(attack_pool)),
                replace=False,
            )
            for n in seeds:
                state_t[n] = 1

        # ── Run one simulation round ──────────────────────────────────────
        final_state, group_rates = _run_round(
            G, state_t, calib_set, threshold=threshold)

        rates = [group_rates.get(g, 0.0) for g in range(K)]
        group_rates_history.append(rates)
        thresholds_history.append(threshold)

        # ── Adaptive defence: tighten threshold on k*'s infection ─────────
        if adaptive_defense:
            rho_target = group_rates.get(k_star, 0.0)
            threshold = min(0.8, base_threshold + 0.05 * (t + 1) * rho_target)

        # ── Adversary retrains: k*(t+1) = argmin_k ρₖ(t) ─────────────────
        if adaptive_adversary:
            k_star = int(np.argmin([group_rates.get(g, 0.0) for g in range(K)]))
        # else: k_star stays 0 (fixed adversary)

        state = final_state

    group_rates_arr = np.array(group_rates_history)            # (T, K)
    disparity = group_rates_arr.max(axis=1) - group_rates_arr.min(axis=1)  # (T,)

    return {
        'group_rates': group_rates_arr,
        'disparity':   disparity,
        'targets':     np.array(targets_history, dtype=int),
        'thresholds':  np.array(thresholds_history),
    }


# ── Main run function ─────────────────────────────────────────────────────────

def run(runs: int = RUNS, verbose: bool = True) -> dict:
    """
    Compare all four conditions across T feedback rounds,
    at two homophily levels: low (h=4) and high (h=20).
    """
    configs = [
        ('Low homophily  h=4',  0.6, 0.15),
        ('High homophily h=20', 0.6, 0.03),
    ]
    attack_count = 5
    T = T_ROUNDS

    all_data: dict = {}

    for label, p_in, p_out in configs:
        all_data[label] = {'h': p_in / p_out, 'p_out': p_out}

        for cond_name, adap_def, adap_adv in CONDITIONS:
            bufs: Dict[str, list] = {
                'group_rates': [], 'disparity': [],
                'targets': [], 'thresholds': [],
            }

            for run_i in range(runs):
                G   = generate_network(N, K, p_in, p_out)
                res = _simulate_feedback(G, attack_count, T, adap_def, adap_adv)
                for key in bufs:
                    bufs[key].append(res[key])

                if verbose and run_i == 0:
                    print(f"  {label} [{cond_name}]: "
                          f"final disparity={res['disparity'][-1]:.3f}  "
                          f"targets={res['targets'].tolist()}")

            gr  = np.array(bufs['group_rates'])     # (runs, T, K)
            tgt = np.array(bufs['targets'])         # (runs, T)

            # Target distribution: fraction of runs that attacked squad g at round t
            target_fracs = np.zeros((T, K))
            for g in range(K):
                target_fracs[:, g] = (tgt == g).mean(axis=0)

            all_data[label][cond_name] = {
                'group_rates':   gr.mean(axis=0),              # (T, K)
                'group_std':     gr.std(axis=0),               # (T, K)
                'disparity':     np.array(bufs['disparity']).mean(axis=0),  # (T,)
                'disp_std':      np.array(bufs['disparity']).std(axis=0),
                'target_fracs':  target_fracs,                  # (T, K)
                'thresholds':    np.array(bufs['thresholds']).mean(axis=0),
            }

            if verbose:
                d = all_data[label][cond_name]
                print(f"  {label} [{cond_name}]: "
                      f"final disparity={d['disparity'][-1]:.3f}")

    return dict(data=all_data, T=T, attack_count=attack_count)


# ── Plot function ─────────────────────────────────────────────────────────────

def plot(results: dict, save: bool = True) -> None:
    data         = results['data']
    T            = results['T']
    attack_count = results['attack_count']
    rounds       = np.arange(1, T + 1)
    configs      = list(data.keys())

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for row, label in enumerate(configs):
        d = data[label]

        # ── Panel 0: per-squad infection for adaptive_learning condition ──────
        ax = axes[row, 0]
        al = d['adaptive_learning']
        gr     = al['group_rates']    # (T, K)
        gr_std = al['group_std']
        tgt_f  = al['target_fracs']   # (T, K) fraction of runs targeting squad g

        for g in range(K):
            ax.plot(rounds, gr[:, g], 'o-', color=SQUAD_COLORS[g],
                    linewidth=2, label=f'Squad {g}')
            ax.fill_between(
                rounds,
                np.clip(gr[:, g] - gr_std[:, g], 0, 1),
                np.clip(gr[:, g] + gr_std[:, g], 0, 1),
                alpha=0.1, color=SQUAD_COLORS[g],
            )
        # Annotate most-frequently targeted squad each round
        for t_idx in range(T):
            dominant_k = int(np.argmax(tgt_f[t_idx]))
            frac = tgt_f[t_idx, dominant_k]
            if frac > 0.4:   # only label if one squad clearly dominates
                ax.annotate(f'k*={dominant_k}',
                            xy=(rounds[t_idx], gr[t_idx, dominant_k]),
                            xytext=(0, 8), textcoords='offset points',
                            ha='center', fontsize=6,
                            color=SQUAD_COLORS[dominant_k])

        ax.set_xlabel('Feedback round t')
        ax.set_ylabel('Infection rate ρₖ')
        ax.set_title(f'{label}\nPer-squad ρₖ — Adaptive def / Learning adv',
                     fontweight='bold', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(rounds)

        # ── Panel 1: system-wide disparity — all 4 conditions ─────────────────
        ax = axes[row, 1]
        for cond_name, _, _ in CONDITIONS:
            disp     = d[cond_name]['disparity']
            disp_std = d[cond_name]['disp_std']
            ls = '--' if 'fixed' in cond_name else '-'
            ax.plot(rounds, disp, 'o' + ls,
                    color=COND_COLORS[cond_name], linewidth=2,
                    label=COND_LABELS[cond_name])
            ax.fill_between(
                rounds,
                np.clip(disp - disp_std, 0, 1),
                np.clip(disp + disp_std, 0, 1),
                alpha=0.09, color=COND_COLORS[cond_name],
            )
        ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Feedback round t')
        ax.set_ylabel('Disparity = max_k(ρₖ) − min_k(ρₖ)')
        ax.set_title(f'{label}\nSystem-wide disparity (4 conditions)',
                     fontweight='bold', fontsize=10)
        ax.legend(fontsize=7.5)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(rounds)

        # ── Panel 2: adversary target distribution + threshold trajectory ─────
        ax = axes[row, 2]

        # Stacked area: fraction of runs targeting each squad, for learning adv
        tgt_f = d['adaptive_learning']['target_fracs']   # (T, K)
        bottom = np.zeros(T)
        for g in range(K):
            ax.bar(rounds, tgt_f[:, g], bottom=bottom,
                   color=SQUAD_COLORS[g], alpha=0.7,
                   label=f'Squad {g} targeted', width=0.6)
            bottom += tgt_f[:, g]
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Fraction of runs targeting squad g')
        ax.set_xlabel('Feedback round t')
        ax.set_title(f'{label}\nAdversary target distribution + threshold',
                     fontweight='bold', fontsize=10)
        ax.set_xticks(rounds)

        # Overlay threshold trajectories on twin axis
        ax2 = ax.twinx()
        ax2.plot(rounds, d['adaptive_learning']['thresholds'],
                 's-', color='#EF9F27', linewidth=2.5,
                 label='Adaptive threshold', zorder=5)
        ax2.plot(rounds, d['static_fixed']['thresholds'],
                 's--', color='#888780', linewidth=1.5,
                 label='Static threshold', alpha=0.7, zorder=4)
        ax2.set_ylabel('Infection threshold', color='#EF9F27')
        ax2.set_ylim(0, 1.0)
        ax2.tick_params(axis='y', labelcolor='#EF9F27')

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='upper left')

    plt.suptitle(
        f'Experiment 10 — Adversarial Retraining Loop  (T={T} rounds, attack={attack_count} seeds)\n'
        f'Adversary re-targets k* = argmin ρₖ  |  '
        f'New metric: system-wide disparity = max−min ρₖ',
        fontsize=13, fontweight='bold',
    )
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(get_output_dir(), 'exp10_feedback_loop.png'),
                    dpi=150, bbox_inches='tight')
    if is_interactive():
        plt.show()
    plt.close('all')

    # ── Summary printout ──────────────────────────────────────────────────────
    print(f"\n{'Condition':<25}  {'h=4 final disp':>16}  {'h=20 final disp':>16}")
    print('-' * 62)
    for cond_name, _, _ in CONDITIONS:
        row_str = f"{COND_LABELS[cond_name]:<25}"
        for label in configs:
            final_d = data[label][cond_name]['disparity'][-1]
            row_str += f"  {final_d:>16.3f}"
        print(row_str)


if __name__ == '__main__':
    print("Running Experiment 10 — Adversarial Retraining Loop...")
    data = run()
    plot(data, save=(RUNS >= 100))
    print("Experiment 10 complete")
