"""
visualizations/network_plots.py
=================================
Publication-quality network visualizations.

v2: nodes show BOTH group (ring color) and infection state (fill color)
    so both dimensions are immediately readable at a glance.
    Positions are stable across snapshots (same layout seed).
"""
from __future__ import annotations
from typing import Optional, Dict, List
import numpy as np
import networkx as nx
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import matplotlib; matplotlib.use('Agg')  # must be before pyplot
import matplotlib.pyplot as plt
from core.simulation import generate_network, initialize_agents, run_simulation
from core.paths import get_output_dir, is_interactive

# ── Colour scheme ─────────────────────────────────────────────────────────────
# Fill = infection state
FILL_HEALTHY  = '#2ecc71'   # green
FILL_INFECTED = '#e74c3c'   # red
FILL_CALIB    = '#f39c12'   # gold

# Ring = group membership (up to 8 groups)
GROUP_RINGS = ['#3498db', '#9b59b6', '#1abc9c', '#e67e22',
               '#e91e63', '#00bcd4', '#ff5722', '#607d8b']

EDGE_INTRA = '#cccccc'
EDGE_INTER = '#4488cc'


def _group_layout(G: nx.Graph, seed: int = 42) -> dict:
    """Spring layout that pulls same-group nodes together into visible clusters."""
    G2 = G.copy()
    for u, v in G2.edges():
        G2[u][v]['weight'] = 4.0 if G2.nodes[u]['group'] == G2.nodes[v]['group'] else 0.2
    return nx.spring_layout(G2, weight='weight', seed=seed, k=1.5)


def _circular_group_layout(G: nx.Graph, seed: int = 42) -> dict:
    """
    Circular group layout: each squad occupies a fixed arc on a large ring.
    Nodes within each squad cluster around their squad centroid, giving
    immediate visual separation of group structure — cleaner than spring layout.
    """
    rng = np.random.default_rng(seed)
    groups = sorted(set(nx.get_node_attributes(G, 'group').values()))
    K = len(groups)
    by_group = {g: [n for n in G.nodes() if G.nodes[n]['group'] == g] for g in groups}

    R_outer = 1.0    # radius of the squad-center ring
    R_inner = 0.28   # spread of nodes within each squad

    pos = {}
    for i, g in enumerate(groups):
        theta = 2 * np.pi * i / K - np.pi / 2   # start from top, go clockwise
        cx, cy = R_outer * np.cos(theta), R_outer * np.sin(theta)
        members = by_group[g]
        n = len(members)
        for j, node in enumerate(members):
            phi = 2 * np.pi * j / n
            r = R_inner * (0.35 + 0.65 * rng.random())
            pos[node] = (cx + r * np.cos(phi), cy + r * np.sin(phi))
    return pos


def plot_network_snapshot(
    G: nx.Graph,
    state: dict,
    calib_set: set,
    pos: Optional[dict] = None,
    title: str = '',
    ax=None,
    show_group_labels: bool = True,
) -> dict:
    """
    Draw network with dual encoding:
      - Node FILL    = infection state (green/red/gold)
      - Node RING    = group membership (blue/purple/teal/orange...)
      - Edge color   = intra-group (gray) vs inter-group (blue)
    """
    show = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    if pos is None:
        pos = _circular_group_layout(G)

    # ── Edges ─────────────────────────────────────────────────────────────────
    intra = [(u, v) for u, v in G.edges()
             if G.nodes[u]['group'] == G.nodes[v]['group']]
    inter = [(u, v) for u, v in G.edges()
             if G.nodes[u]['group'] != G.nodes[v]['group']]
    nx.draw_networkx_edges(G, pos, edgelist=intra, ax=ax,
                           edge_color=EDGE_INTRA, alpha=0.25, width=0.6)
    nx.draw_networkx_edges(G, pos, edgelist=inter, ax=ax,
                           edge_color=EDGE_INTER, alpha=0.50, width=1.0)

    # ── Nodes: ring (group) + fill (state) ────────────────────────────────────
    groups = sorted(set(nx.get_node_attributes(G, 'group').values()))

    for node in G.nodes():
        x, y    = pos[node]
        grp     = G.nodes[node]['group']
        ring_c  = GROUP_RINGS[grp % len(GROUP_RINGS)]

        if node in calib_set:
            fill_c, marker, base_s, zorder = FILL_CALIB, '*', 420, 5
        elif state[node] == 1:
            fill_c, marker, base_s, zorder = FILL_INFECTED, 'o', 220, 4
        else:
            fill_c, marker, base_s, zorder = FILL_HEALTHY, 'o', 160, 3

        # Outer ring (group color) — drawn large
        ax.scatter(x, y, c=ring_c, marker=marker,
                   s=base_s * 3.0, zorder=zorder - 0.5,
                   linewidths=0)
        # White gap
        ax.scatter(x, y, c='white', marker=marker,
                   s=base_s * 1.6, zorder=zorder - 0.1,
                   linewidths=0)
        # Inner fill (infection state)
        ax.scatter(x, y, c=fill_c, marker=marker,
                   s=base_s, zorder=zorder,
                   edgecolors='none', linewidths=0)

    # ── Group centroid labels ─────────────────────────────────────────────────
    if show_group_labels:
        by_group: Dict[int, List] = {}
        for node in G.nodes():
            g = G.nodes[node]['group']
            by_group.setdefault(g, []).append(pos[node])
        for g, positions in by_group.items():
            cx = np.mean([p[0] for p in positions])
            cy = np.mean([p[1] for p in positions])
            ax.text(cx, cy, f'Squad {g}',
                    ha='center', va='center', fontsize=8, fontweight='bold',
                    color=GROUP_RINGS[g % len(GROUP_RINGS)],
                    path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    # ── Legend ────────────────────────────────────────────────────────────────
    n_inf   = sum(state.values())
    n_cal   = len(calib_set)
    n_hlth  = len(state) - n_inf - n_cal
    legend_elements = [
        mpatches.Patch(fc=FILL_HEALTHY,  label=f'Healthy ({n_hlth})'),
        mpatches.Patch(fc=FILL_INFECTED, label=f'Infected ({n_inf})'),
        mpatches.Patch(fc=FILL_CALIB,    label=f'Calibration ({n_cal})'),
        plt.Line2D([0],[0], color=EDGE_INTRA, lw=1.2, label='Intra-group edge'),
        plt.Line2D([0],[0], color=EDGE_INTER, lw=1.8, label='Inter-group edge'),
    ]
    # Add group color swatches
    for g in groups[:4]:
        legend_elements.append(
            mpatches.Patch(fc=GROUP_RINGS[g % len(GROUP_RINGS)],
                           label=f'Squad {g} (ring)'))
    ax.legend(handles=legend_elements, loc='upper left',
              fontsize=7.5, framealpha=0.9, ncol=1)
    ax.set_title(title, fontweight='bold', fontsize=10, pad=8)
    ax.axis('off')

    if show:
        plt.tight_layout()
        if is_interactive():
            plt.show()
    return pos


def _step_n(G, state0, calib_set, steps,
            alpha=0.6, beta=0.4, threshold=0.3, r=4, p_c=0.3):
    cur = state0.copy()
    for _ in range(steps):
        nxt = cur.copy()
        for node in G.nodes():
            if node in calib_set:
                nxt[node] = 0; continue
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
                    nxt[node] = 1
            else:
                nc = any(n in calib_set for n in nb)
                hi = sum(1 for n in in_nb if cur[n] == 0)
                if nc or hi >= r:
                    if np.random.random() < p_c:
                        nxt[node] = 0
        if nxt == cur:
            break
        cur = nxt
    return cur


def plot_snapshot_trio(N=100, K=4, p_in=0.6, p_out=0.02,
                       attack_count=10, calib_count=0,
                       seed=7, save=True):
    """Three panels: t=0, t=10, converged. Uses strong parameters to show spread."""
    np.random.seed(seed)
    G = generate_network(N, K, p_in, p_out, seed=seed)
    state0, calib = initialize_agents(
        G, attack_count, 'concentrated', calib_count, 'scattered', K)

    state10   = _step_n(G, state0, calib, 5)
    state_fin = _step_n(G, state0, calib, 200)

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    pos = _circular_group_layout(G, seed=seed)
    h = p_in / p_out

    plot_network_snapshot(G, state0, calib, pos=pos,
        title=f't = 0   Initial ({attack_count} seeds in Squad 0)',
        ax=axes[0])
    plot_network_snapshot(G, state10, calib, pos=pos,
        title=f't = 5   Spreading — {sum(state10.values())} infected',
        ax=axes[1])
    plot_network_snapshot(G, state_fin, calib, pos=pos,
        title=f'Converged — {sum(state_fin.values())} infected  '
              f'({N - sum(state_fin.values())} healthy)',
        ax=axes[2])

    plt.suptitle(f'Infection Spread Snapshots — K={K}, h={h:.0f}  '
                 f'(p_in={p_in}, p_out={p_out})\n'
                 'Ring color = squad membership   Fill color = infection state',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(get_output_dir(), 'network_snapshots.png'),
                    dpi=150, bbox_inches='tight')
    if is_interactive():
        plt.show()
    print("  Network snapshot trio saved ✓")


def plot_homophily_compare(N=80, K=4, attack_count=2, seed=42, save=True):
    """Low h vs high h — shows structural difference clearly."""
    np.random.seed(seed)
    configs = [
        ('Low homophily  h = 2  (well-connected)',  0.6, 0.30),
        ('High homophily h = 20  (segregated)',      0.6, 0.03),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, (label, p_in, p_out) in zip(axes, configs):
        G = generate_network(N, K, p_in, p_out, seed=seed)
        state, calib = initialize_agents(
            G, attack_count, 'concentrated', 0, 'scattered', K)
        pos = _circular_group_layout(G, seed=seed)
        plot_network_snapshot(G, state, calib, pos=pos, title=label, ax=ax)

    plt.suptitle('Network Structure: Low vs High Homophily\n'
                 'Blue edges = inter-group links — far fewer at h=20',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(get_output_dir(), 'network_homophily_compare.png'),
                    dpi=150, bbox_inches='tight')
    if is_interactive():
        plt.show()
    print("  Homophily comparison saved ✓")


def plot_infection_progression(N=100, K=4, p_out=0.02, attack_count=10,
                                n_steps=6, seed=7, save=True):
    """
    NEW: 2-row grid showing infection spreading step by step.
    Row 1: low homophily (h=2)   Row 2: high homophily (h=20)
    Directly shows how homophily traps errors.
    """
    np.random.seed(seed)
    steps_to_show = [0, 3, 6, 10, 20, 50]
    configs = [
        ('Low h = 2',  0.6, 0.30),
        ('High h = 20', 0.6, 0.03),
    ]

    fig, axes = plt.subplots(2, len(steps_to_show),
                              figsize=(4 * len(steps_to_show), 9))

    for row, (hlabel, p_in, p_out_val) in enumerate(configs):
        G = generate_network(N, K, p_in, p_out_val, seed=seed)
        state0, calib = initialize_agents(
            G, attack_count, 'concentrated', 0, 'scattered', K)
        pos = _circular_group_layout(G, seed=seed)

        for col, step in enumerate(steps_to_show):
            state_t = _step_n(G, state0, calib, step)
            inf_pct = sum(state_t.values()) / N * 100
            plot_network_snapshot(
                G, state_t, calib, pos=pos,
                title=f't={step}  ({inf_pct:.0f}% infected)',
                ax=axes[row, col],
                show_group_labels=(col == 0),
            )
            if col == 0:
                axes[row, col].set_ylabel(hlabel, fontsize=10,
                                           fontweight='bold', labelpad=10)

    plt.suptitle('Infection Progression: Low vs High Homophily\n'
                 'High homophily traps infection within the source squad',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(get_output_dir(), 'network_progression.png'),
                    dpi=150, bbox_inches='tight')
    if is_interactive():
        plt.show()
    print("  Infection progression grid saved ✓")


if __name__ == '__main__':
    print("Generating network visualizations...")
    plot_snapshot_trio()
    plot_homophily_compare()
    plot_infection_progression()
    print("Network visualizations complete ✓")
