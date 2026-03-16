"""
core/metrics.py
===============
Derived metrics computed on top of simulation results.
Imported by experiments and the Streamlit app.
"""

from __future__ import annotations
from typing import List, Optional
import numpy as np
import networkx as nx
from core.simulation import generate_network, initialize_agents, run_simulation, monte_carlo


def algebraic_connectivity(G: nx.Graph) -> float:
    """
    Compute λ₂ — the second-smallest eigenvalue of the graph Laplacian.
    Also called the Fiedler value. Higher λ₂ = faster information diffusion.
    """
    L = nx.laplacian_matrix(G).toarray().astype(float)
    eigvals = np.sort(np.linalg.eigvalsh(L))
    return float(eigvals[1])


def recovery_rate(total_nodes: int, final_infected: int,
                  initial_seeds: int) -> float:
    """
    Fraction of non-seed nodes that successfully returned to healthy state.

    recovery_rate = (total - final_infected - 0) / (total - initial_seeds)
    = 1.0 means full recovery except the permanent seeds
    = 0.0 means no recovery at all
    """
    recoverable = total_nodes - initial_seeds
    if recoverable <= 0:
        return 0.0
    recovered = max(0, recoverable - final_infected)
    return recovered / recoverable


def find_cascade_threshold(
    infection_values: List[float],
    attack_intensities: List[int],
    cutoff: float = 0.5,
) -> Optional[float]:
    """
    Find the attack intensity at which peak infection first crosses `cutoff`.
    Returns None if it never crosses.
    """
    for i in range(len(infection_values) - 1):
        if infection_values[i] < cutoff <= infection_values[i + 1]:
            # Linear interpolation for a smoother estimate
            frac = (cutoff - infection_values[i]) / (
                infection_values[i + 1] - infection_values[i] + 1e-9)
            return attack_intensities[i] + frac * (
                attack_intensities[i + 1] - attack_intensities[i])
    return None


def find_h_crit(
    N: int = 100,
    K: int = 4,
    p_in: float = 0.6,
    attack_count: int = 3,
    runs: int = 100,
    recovery_cutoff: float = 0.10,
    p_out_range: np.ndarray | None = None,
) -> dict:
    """
    Sweep p_out (and thus h = p_in/p_out) with NO intervention (calib_count=0).
    Find h_crit: the homophily ratio above which the system can no longer
    self-recover (final infection > recovery_cutoff).

    Returns
    -------
    dict with keys: h_values, mean_final, std_final, mean_peak, h_crit
    """
    if p_out_range is None:
        p_out_range = np.linspace(0.02, 0.30, 15)

    h_values, mean_finals, std_finals, mean_peaks = [], [], [], []

    for p_out in p_out_range:
        res = monte_carlo(
            runs, N=N, K=K, p_in=p_in, p_out=p_out,
            attack_count=attack_count, attack_distribution='concentrated',
            calib_count=0, calib_distribution='scattered',
            r=3, p_c=0.6,
        )
        h = p_in / p_out
        h_values.append(h)
        mean_finals.append(res['mean_final'])
        std_finals.append(res['std_final'])
        mean_peaks.append(res['mean_peak'])

    # h_crit = first h where system fails to self-recover
    h_crit = None
    for i, (h, mf) in enumerate(zip(h_values, mean_finals)):
        if mf > recovery_cutoff:
            h_crit = h
            break

    return {
        'h_values':   np.array(h_values),
        'mean_final': np.array(mean_finals),
        'std_final':  np.array(std_finals),
        'mean_peak':  np.array(mean_peaks),
        'h_crit':     h_crit,
    }


def cheeger_constant(G: nx.Graph, K: int = 4) -> float:
    """
    Approximate the Cheeger constant (conductance) of the graph.
    For an SBM graph, use the natural community partition.
    conductance = cut_edges / min(vol(S), vol(V\S))
    where vol(S) = sum of degrees in S.

    Lower conductance = stronger bottleneck = more segregated.
    Directly requested by Topic 6 guidelines.
    """
    from collections import defaultdict
    by_group: dict = defaultdict(list)
    for n in G.nodes():
        grp = G.nodes[n].get('group', 0)
        by_group[grp].append(n)

    if len(by_group) < 2:
        return 0.0

    # Use group 0 vs rest as the partition
    S = set(by_group[0])
    V_minus_S = set(G.nodes()) - S

    cut = sum(1 for u, v in G.edges()
              if (u in S) != (v in S))
    vol_S = sum(dict(G.degree(S)).values())
    vol_rest = sum(dict(G.degree(V_minus_S)).values())

    if min(vol_S, vol_rest) == 0:
        return 0.0
    return cut / min(vol_S, vol_rest)
