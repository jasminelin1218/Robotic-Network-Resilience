"""
core/simulation.py
==================
Shared simulation engine for ECE227 project.
All experiments import from here — single source of truth.

Model
-----
- Network  : Stochastic Block Model (SBM) with K communities
- Infection : Threshold contagion weighted by in-group (α) and out-group (β) peer pressure
- Recovery  : Calibration-node proximity OR in-group healthy peer pressure
- Defense   : Optional dynamic edge severance when out-group infection rate > severance_h
"""

from __future__ import annotations
from typing import Optional, List, Dict, Tuple

import numpy as np
import networkx as nx
from collections import defaultdict
from joblib import Parallel, delayed
from typing import Optional


# ── Network generation ────────────────────────────────────────────────────────

def generate_network(N: int, K: int, p_in: float, p_out: float,
                     seed: Optional[int] = None) -> nx.Graph:
    """
    Generate a Stochastic Block Model communication network.

    Parameters
    ----------
    N     : total number of robots
    K     : number of squads / communities
    p_in  : intra-group edge probability
    p_out : inter-group edge probability  (homophily h = p_in / p_out)
    seed  : random seed; None = fresh random each call (used in Monte Carlo)

    Returns
    -------
    G : undirected NetworkX graph with node attribute 'group' ∈ [0, K-1]
    """
    sizes = [N // K] * K
    sizes[-1] += N - sum(sizes)          # absorb remainder into last group

    p_matrix = [[p_out] * K for _ in range(K)]
    for i in range(K):
        p_matrix[i][i] = p_in

    G = nx.stochastic_block_model(sizes, p_matrix, seed=seed)

    # Tag every node with its community index
    gid, count = 0, 0
    for node in G.nodes():
        G.nodes[node]['group'] = gid
        count += 1
        if count >= sizes[gid]:
            gid = min(gid + 1, K - 1)
            count = 0

    return G


# ── Agent initialisation ──────────────────────────────────────────────────────

def initialize_agents(
    G: nx.Graph,
    attack_count: int,
    attack_distribution: str,   # 'concentrated' | 'scattered'
    calib_count: int,
    calib_distribution: str,    # 'concentrated' | 'scattered' | 'attack_group' | 'other_group'
    K: int,
) -> tuple[dict, set]:
    """
    Set initial states and designate calibration (always-correct) nodes.

    Returns
    -------
    state     : dict {node_id: 0 (healthy) | 1 (infected)}
    calib_set : set of node IDs that are permanently anchored at state = 0
    """
    state = {n: 0 for n in G.nodes()}

    by_group: Dict[int, list] = defaultdict(list)
    for n in G.nodes():
        by_group[G.nodes[n]['group']].append(n)

    # ── Place calibration nodes ──────────────────────────────────────────────
    if calib_distribution == 'concentrated':
        calib_pool = by_group[0]
    elif calib_distribution == 'attack_group':
        calib_pool = by_group[0]
    elif calib_distribution == 'other_group':
        calib_pool = by_group[1 % K]
    else:   # 'scattered' — spread proportionally across all groups
        calib_pool = []
        per_group = max(1, calib_count // K)
        for g in range(K):
            calib_pool.extend(by_group[g][:per_group])

    calib_set: set = set()
    if calib_count > 0 and calib_pool:
        calib_set = set(np.random.choice(
            calib_pool,
            size=min(calib_count, len(calib_pool)),
            replace=False,
        ))

    # ── Place initial infections (cannot overlap with calibration nodes) ─────
    if attack_distribution == 'concentrated':
        attack_pool = [n for n in by_group[0] if n not in calib_set]
    else:   # 'scattered'
        attack_pool = [n for n in G.nodes() if n not in calib_set]

    if attack_count > 0 and attack_pool:
        seeds = np.random.choice(
            attack_pool,
            size=min(attack_count, len(attack_pool)),
            replace=False,
        )
        for n in seeds:
            state[n] = 1

    return state, calib_set


# ── Core simulation engine ────────────────────────────────────────────────────

def run_simulation(
    G: nx.Graph,
    state: dict,
    calib_set: set,
    alpha: float = 0.5,
    beta: float  = 0.5,
    threshold: float = 0.4,
    r: int   = 2,
    p_c: float = 0.8,
    use_dynamic_defense: bool = False,
    severance_h: float = 0.6,
    max_steps: int = 200,
    record_states: bool = False,
) -> dict:
    """
    Synchronous threshold-contagion simulation.

    Infection rule
    --------------
    score = α·r_in + β·r_out  ≥  threshold  →  robot becomes infected
    where r_in / r_out are infected-fraction among in-/out-group neighbours.
    Isolated nodes (no in-group OR no out-group neighbours) have the full
    weight (α+β) redirected to whichever neighbour set exists.

    Recovery rule
    -------------
    Infected robot recovers with probability p_c if:
      (a) directly connected to a calibration node, OR
      (b) ≥ r of its IN-GROUP neighbours are healthy
    (In-group peer pressure drives echo-chamber recovery.)

    Dynamic defense (optional)
    --------------------------
    If out-group infected fraction > severance_h, sever those edges
    for the remainder of the simulation run.

    Parameters
    ----------
    record_states : if True, return full list of per-node state dicts each step
                    (memory-intensive; used for animation only)

    Returns
    -------
    dict with keys:
        infection_over_time  : list[float]  — fraction infected at each step
        convergence_step     : int          — step where state stopped changing (-1 if not)
        peak_infection       : float
        final_infection      : float
        state_history        : list[dict]   — only present when record_states=True
    """
    G_sim     = G.copy()
    cur_state = state.copy()
    infection_over_time: List[float] = []
    state_history: List[dict] = []
    prev_state = None
    conv_step  = -1

    for step in range(max_steps):
        n_nodes       = G_sim.number_of_nodes()
        infected_frac = sum(cur_state.values()) / n_nodes
        infection_over_time.append(infected_frac)

        if record_states:
            state_history.append(cur_state.copy())

        # Convergence check
        if prev_state is not None and cur_state == prev_state:
            conv_step = step
            break

        prev_state = cur_state.copy()
        new_state  = cur_state.copy()

        for node in G_sim.nodes():
            if node in calib_set:
                new_state[node] = 0
                continue

            group     = G_sim.nodes[node]['group']
            neighbors = list(G_sim.neighbors(node))
            in_nb  = [n for n in neighbors if G_sim.nodes[n]['group'] == group]
            out_nb = [n for n in neighbors if G_sim.nodes[n]['group'] != group]

            # ── Infection ────────────────────────────────────────────────────
            if cur_state[node] == 0:
                in_rate  = (sum(cur_state[n] for n in in_nb)  / len(in_nb)
                            if in_nb  else 0.0)
                out_rate = (sum(cur_state[n] for n in out_nb) / len(out_nb)
                            if out_nb else 0.0)

                if not in_nb and not out_nb:
                    score = 0.0
                elif not in_nb:
                    score = (alpha + beta) * out_rate
                elif not out_nb:
                    score = (alpha + beta) * in_rate
                else:
                    score = alpha * in_rate + beta * out_rate

                if score >= threshold:
                    new_state[node] = 1

            # ── Recovery ─────────────────────────────────────────────────────
            else:
                near_calib  = any(n in calib_set for n in neighbors)
                healthy_in  = sum(1 for n in in_nb if cur_state[n] == 0)

                if near_calib or healthy_in >= r:
                    if np.random.random() < p_c:
                        new_state[node] = 0

            # ── Dynamic defense ───────────────────────────────────────────────
            if use_dynamic_defense and out_nb:
                out_inf_rate = sum(cur_state[n] for n in out_nb) / len(out_nb)
                if out_inf_rate > severance_h:
                    for n in list(out_nb):
                        if G_sim.has_edge(node, n):
                            G_sim.remove_edge(node, n)

        cur_state = new_state

    # Pad curve to max_steps for consistent array shapes across Monte Carlo runs
    while len(infection_over_time) < max_steps:
        infection_over_time.append(infection_over_time[-1])

    result = {
        'infection_over_time': infection_over_time,
        'convergence_step':    conv_step,
        'peak_infection':      max(infection_over_time),
        'final_infection':     infection_over_time[-1],
    }
    if record_states:
        result['state_history'] = state_history
    return result


# ── Monte Carlo wrapper ───────────────────────────────────────────────────────

def _single_run(i: int, N: int, K: int, p_in: float, p_out: float,
                attack_count: int, attack_distribution: str,
                calib_count: int, calib_distribution: str,
                alpha: float, beta: float, threshold: float,
                r: int, p_c: float,
                use_dynamic_defense: bool, severance_h: float,
                max_steps: int) -> dict:
    """One Monte Carlo trial executed in a worker process."""
    G = generate_network(N, K, p_in, p_out)
    state, calib_set = initialize_agents(
        G, attack_count, attack_distribution,
        calib_count, calib_distribution, K)
    return run_simulation(
        G, state, calib_set,
        alpha=alpha, beta=beta, threshold=threshold,
        r=r, p_c=p_c,
        use_dynamic_defense=use_dynamic_defense,
        severance_h=severance_h, max_steps=max_steps)


def monte_carlo(runs: int = 100, **kwargs) -> dict:
    """
    Run `runs` independent simulations in parallel and aggregate statistics.
    Uses joblib with all available CPU cores (n_jobs=-1).

    All generate_network / initialize_agents / run_simulation parameters
    are passed via **kwargs with sensible defaults.

    Returns
    -------
    dict with:
        mean_peak, std_peak    : peak infection rate statistics
        mean_final, std_final  : final infection rate statistics
        mean_conv, std_conv    : convergence step statistics
        mean_curve, std_curve  : per-step infection curve statistics
    """
    results = Parallel(n_jobs=-1, prefer='threads')(
        delayed(_single_run)(
            i,
            kwargs['N'],
            kwargs['K'],
            kwargs.get('p_in', 0.6),
            kwargs['p_out'],
            kwargs.get('attack_count', 2),
            kwargs.get('attack_distribution', 'concentrated'),
            kwargs.get('calib_count', 2),
            kwargs.get('calib_distribution', 'scattered'),
            kwargs.get('alpha', 0.5),
            kwargs.get('beta',  0.5),
            kwargs.get('threshold', 0.4),
            kwargs.get('r', 2),
            kwargs.get('p_c', 0.8),
            kwargs.get('use_dynamic_defense', False),
            kwargs.get('severance_h', 0.6),
            kwargs.get('max_steps', 200),
        )
        for i in range(runs)
    )

    peaks  = [r['peak_infection']      for r in results]
    finals = [r['final_infection']     for r in results]
    convs  = [r['convergence_step']    for r in results]
    curves = [r['infection_over_time'] for r in results]
    valid_convs = [c for c in convs if c != -1]

    return {
        'mean_peak':  np.mean(peaks),
        'std_peak':   np.std(peaks),
        'mean_final': np.mean(finals),
        'std_final':  np.std(finals),
        'mean_conv':  np.mean(valid_convs) if valid_convs else 200,
        'std_conv':   np.std(valid_convs)  if valid_convs else 0,
        'mean_curve': np.mean(curves, axis=0),
        'std_curve':  np.std(curves,  axis=0),
    }
