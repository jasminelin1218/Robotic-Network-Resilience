"""
experiments/exp10_feedback_loop.py
====================================
Experiment 10 — Feedback Loop: Repeated Attack Waves  (Topic 6 Phase 3)

Topic 6 mapping
---------------
Topic 6 Phase 3 requires a T-round feedback loop:
  Round t: Seed → Propagate → React → Retrain classifier → Repeat

In our robotics model there is no classifier to retrain, but we have an
exact structural equivalent:

  Round t: Attack seeds injected → Errors propagate (threshold contagion)
           → System partially recovers → ADAPTIVE response updates
           → New attack wave in round t+1

The "retraining" analog is the ADAPTIVE DEFENSE:
  After each round, the system observes which squads got infected and
  strengthens their defense (lowers their infection threshold locally).
  This mirrors how a classifier retrains on new data.

Two conditions compared:
  (A) Static defense: same threshold every round — errors accumulate
  (B) Adaptive defense: threshold tightens each round based on last round's
      infection rate — models an organization "learning" from attacks

Key metrics (from Topic 6 Phase 4):
  - Cumulative infection across T rounds for both conditions
  - Gap between Squad 0 and other squads across rounds
  - How quickly adaptive defense converges vs static

Research question:
  Does the feedback loop cause permanent divergence between groups
  (errors get permanently trapped), or does adaptive defense converge
  to a stable equilibrium?
"""
from __future__ import annotations
from typing import List, Dict
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import matplotlib; matplotlib.use('Agg')  # must be before pyplot
import matplotlib.pyplot as plt
from core.simulation import generate_network, initialize_agents, run_simulation
from core.paths import get_output_dir, is_interactive

RUNS = 100
N    = 100
K    = 4
T_ROUNDS = 10   # number of feedback rounds


def _run_round(G, state0: dict, calib_set: set,
               threshold: float, alpha: float = 0.6,
               beta: float = 0.4, r: int = 3,
               p_c: float = 0.3) -> tuple[dict, dict]:
    """
    Run one simulation round. Returns (final_state, per_group_infection).
    """
    from collections import defaultdict
    res = run_simulation(G, state0, calib_set,
                         alpha=alpha, beta=beta, threshold=threshold,
                         r=r, p_c=p_c, max_steps=100)

    # Reconstruct final state from the convergence
    cur = state0.copy()
    for _ in range(100):
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
                score = ((alpha+beta)*ir  if not out_nb else
                         (alpha+beta)*or_ if not in_nb  else
                         alpha*ir + beta*or_)
                if score >= threshold:
                    nxt[node] = 1; changed = True
            else:
                hi = sum(1 for n in in_nb if cur[n] == 0)
                if hi >= r:
                    if np.random.random() < p_c:
                        nxt[node] = 0; changed = True
        if not changed:
            break
        cur = nxt

    # Per-group rates
    by_group: Dict[int, List] = defaultdict(list)
    for node in G.nodes():
        by_group[G.nodes[node]['group']].append(node)

    group_rates = {}
    for g, nodes in by_group.items():
        group_rates[g] = sum(cur[n] for n in nodes) / len(nodes)

    return cur, group_rates


def _simulate_feedback(G, attack_count: int, T: int,
                        adaptive: bool) -> dict:
    """
    Run T rounds of attack-propagate-recover.

    adaptive=False: fixed threshold every round (static defense)
    adaptive=True:  threshold increases each round proportional to last
                    round's infection (the 'retrain' analog)
    """
    base_threshold = 0.3
    threshold = base_threshold

    # Track per-round metrics
    squad0_rates  = []
    others_rates  = []
    delta_rates   = []
    thresholds    = []

    # Initial state: all healthy
    state = {n: 0 for n in G.nodes()}
    calib_set: set = set()

    for t in range(T):
        # Inject new attack seeds into Squad 0 each round
        from core.simulation import initialize_agents
        state_t, calib_t = initialize_agents(
            G, attack_count, 'concentrated', 0, 'scattered', K)
        # Carry forward any infected nodes from last round
        for node in G.nodes():
            if state[node] == 1:
                state_t[node] = 1

        final_state, group_rates = _run_round(
            G, state_t, calib_set, threshold=threshold)

        s0_rate = group_rates.get(0, 0)
        ot_rate = np.mean([v for g, v in group_rates.items() if g != 0])

        squad0_rates.append(s0_rate)
        others_rates.append(ot_rate)
        delta_rates.append(s0_rate - ot_rate)
        thresholds.append(threshold)

        # Adaptive: tighten threshold based on last round's infection
        if adaptive:
            # Threshold increases proportionally to observed infection
            # Models the organization "learning" to be more cautious
            threshold = min(0.8, base_threshold + 0.05 * t * s0_rate)

        # Next round starts from this round's final state
        state = final_state

    return {
        'squad0_rates': np.array(squad0_rates),
        'others_rates': np.array(others_rates),
        'delta_rates':  np.array(delta_rates),
        'thresholds':   np.array(thresholds),
    }


def run(runs: int = RUNS, verbose: bool = True) -> dict:
    """
    Compare static vs adaptive defense across T feedback rounds,
    at two homophily levels: low (h=4) and high (h=20).
    """
    configs = [
        ('Low homophily  h=4',  0.6, 0.15),
        ('High homophily h=20', 0.6, 0.03),
    ]
    attack_count = 5
    T = T_ROUNDS

    all_data = {}

    for label, p_in, p_out in configs:
        h = p_in / p_out
        static_s0, static_ot, static_d = [], [], []
        adapt_s0,  adapt_ot,  adapt_d  = [], [], []
        static_thr, adapt_thr = [], []

        for run_i in range(runs):
            G = generate_network(N, K, p_in, p_out)

            # Static
            s_res = _simulate_feedback(G, attack_count, T, adaptive=False)
            static_s0.append(s_res['squad0_rates'])
            static_ot.append(s_res['others_rates'])
            static_d.append(s_res['delta_rates'])
            static_thr.append(s_res['thresholds'])

            # Adaptive
            a_res = _simulate_feedback(G, attack_count, T, adaptive=True)
            adapt_s0.append(a_res['squad0_rates'])
            adapt_ot.append(a_res['others_rates'])
            adapt_d.append(a_res['delta_rates'])
            adapt_thr.append(a_res['thresholds'])

            if verbose and run_i == 0:
                print(f"  {label}: first run Squad0_final={s_res['squad0_rates'][-1]:.3f} "
                      f"(static) vs {a_res['squad0_rates'][-1]:.3f} (adaptive)")

        all_data[label] = {
            'h': h, 'p_out': p_out,
            'static': {
                'squad0': np.mean(static_s0, axis=0),
                'others': np.mean(static_ot, axis=0),
                'delta':  np.mean(static_d,  axis=0),
                'thr':    np.mean(static_thr, axis=0),
                'std_s0': np.std(static_s0, axis=0),
                'std_d':  np.std(static_d,  axis=0),
            },
            'adaptive': {
                'squad0': np.mean(adapt_s0, axis=0),
                'others': np.mean(adapt_ot, axis=0),
                'delta':  np.mean(adapt_d,  axis=0),
                'thr':    np.mean(adapt_thr, axis=0),
                'std_s0': np.std(adapt_s0, axis=0),
                'std_d':  np.std(adapt_d,  axis=0),
            },
        }

        if verbose:
            s_final = all_data[label]['static']['squad0'][-1]
            a_final = all_data[label]['adaptive']['squad0'][-1]
            print(f"  {label}: Squad0 round {T} — static={s_final:.3f}, "
                  f"adaptive={a_final:.3f}")

    return dict(data=all_data, T=T, attack_count=attack_count)


def plot(results: dict, save: bool = True) -> None:
    data         = results['data']
    T            = results['T']
    attack_count = results['attack_count']
    rounds       = np.arange(1, T + 1)
    configs      = list(data.keys())

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    COLORS = {
        'static_s0':   '#E24B4A',
        'static_ot':   '#F0997B',
        'adaptive_s0': '#378ADD',
        'adaptive_ot': '#85B7EB',
        'delta_static':   '#E24B4A',
        'delta_adaptive': '#378ADD',
    }

    for row, label in enumerate(configs):
        d = data[label]
        h = d['h']

        # Panel 0: Squad 0 infection over rounds — static vs adaptive
        ax = axes[row, 0]
        s = d['static'];  a = d['adaptive']
        ax.plot(rounds, s['squad0'], 'o-', color=COLORS['static_s0'],
                linewidth=2, label='Static defense — Squad 0')
        ax.fill_between(rounds, np.clip(s['squad0']-s['std_s0'],0,1),
                         np.clip(s['squad0']+s['std_s0'],0,1),
                         alpha=0.12, color=COLORS['static_s0'])
        ax.plot(rounds, s['others'], 's--', color=COLORS['static_ot'],
                linewidth=1.5, label='Static defense — Other squads', alpha=0.8)
        ax.plot(rounds, a['squad0'], 'o-', color=COLORS['adaptive_s0'],
                linewidth=2, label='Adaptive defense — Squad 0')
        ax.fill_between(rounds, np.clip(a['squad0']-a['std_s0'],0,1),
                         np.clip(a['squad0']+a['std_s0'],0,1),
                         alpha=0.12, color=COLORS['adaptive_s0'])
        ax.plot(rounds, a['others'], 's--', color=COLORS['adaptive_ot'],
                linewidth=1.5, label='Adaptive defense — Other squads', alpha=0.8)
        ax.set_xlabel('Feedback round t')
        ax.set_ylabel('Infection rate')
        ax.set_title(f'{label}\nPer-group infection over T={T} rounds',
                      fontweight='bold')
        ax.legend(fontsize=7.5)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(rounds)

        # Panel 1: ΔInfection (disparity) over rounds
        ax = axes[row, 1]
        ax.plot(rounds, s['delta'], 'D-', color=COLORS['delta_static'],
                linewidth=2.5, label=f'Static defense Δ')
        ax.fill_between(rounds, s['delta']-s['std_d'], s['delta']+s['std_d'],
                         alpha=0.12, color=COLORS['delta_static'])
        ax.plot(rounds, a['delta'], 'D-', color=COLORS['delta_adaptive'],
                linewidth=2.5, label=f'Adaptive defense Δ')
        ax.fill_between(rounds, a['delta']-a['std_d'], a['delta']+a['std_d'],
                         alpha=0.12, color=COLORS['delta_adaptive'])
        ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Feedback round t')
        ax.set_ylabel('ΔInfection = Squad0 − Others')
        ax.set_title(f'{label}\nDisparity (Gaming Gap) over rounds',
                      fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(rounds)

        # Panel 2: Adaptive threshold trajectory
        ax = axes[row, 2]
        ax.plot(rounds, s['thr'], 'o--', color='#888780', linewidth=2,
                label='Static threshold (fixed)')
        ax.plot(rounds, a['thr'], 's-', color='#1D9E75', linewidth=2.5,
                label='Adaptive threshold (learning)')
        ax.set_xlabel('Feedback round t')
        ax.set_ylabel('Infection threshold')
        ax.set_title(f'{label}\nAdaptive defense: threshold evolution',
                      fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)
        ax.set_xticks(rounds)

    plt.suptitle(f'Experiment 10 — Feedback Loop: Repeated Attack Waves (T={T} rounds)\n'
                 f'Topic 6 Phase 3: Static vs Adaptive Defense — '
                 f'attack={attack_count} seeds/round in Squad 0',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(get_output_dir(), 'exp10_feedback_loop.png'),
                    dpi=150, bbox_inches='tight')
    if is_interactive(): plt.show()
    plt.close('all')

    # Print summary
    for label in configs:
        d = data[label]
        s_final_d = d['static']['delta'][-1]
        a_final_d = d['adaptive']['delta'][-1]
        print(f"  {label}:")
        print(f"    Static  final disparity: {s_final_d:+.3f}")
        print(f"    Adaptive final disparity: {a_final_d:+.3f}")
        improve = s_final_d - a_final_d
        print(f"    Adaptive reduces disparity by: {improve:+.3f}")


if __name__ == '__main__':
    print("Running Experiment 10 — Feedback Loop...")
    data = run()
    plot(data, save=(RUNS >= 100))
    print("Experiment 10 complete")
