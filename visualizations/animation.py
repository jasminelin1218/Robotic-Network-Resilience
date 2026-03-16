"""
visualizations/animation.py
=============================
Generate animated GIFs of infection spreading through the network.
Creates two contrasting animations: low homophily vs high homophily.

Output
------
outputs/figures/animation_low_homophily.gif
outputs/figures/animation_high_homophily.gif

Requirements
------------
pip install pillow  (for GIF saving via matplotlib)
"""

from __future__ import annotations
import numpy as np
import networkx as nx
import matplotlib.animation as animation
from matplotlib.patches import Patch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import matplotlib; matplotlib.use('Agg')  # must be before pyplot
import matplotlib.pyplot as plt
from core.paths import get_output_dir, is_interactive
from core.simulation import generate_network, initialize_agents, run_simulation


INFECTED_COL = '#E24B4A'
HEALTHY_COL  = '#639922'
CALIB_COL    = '#EF9F27'


def _group_layout(G, seed=42):
    G2 = G.copy()
    for u, v in G2.edges():
        G2[u][v]['weight'] = 3.0 if G2.nodes[u]['group'] == G2.nodes[v]['group'] else 0.3
    return nx.spring_layout(G2, weight='weight', seed=seed, k=1.2)


def make_animation(
    G: nx.Graph,
    state_history: List[dict],
    calib_set: set,
    pos: dict,
    title: str,
    output_path: str,
    fps: int = 4,
    max_frames: int = 40,
) -> None:
    """Render a GIF of the infection spreading timestep by timestep."""
    frames = state_history[:max_frames]

    fig, ax = plt.subplots(figsize=(7, 6))

    # Pre-compute edge lists
    intra = [(u, v) for u, v in G.edges()
             if G.nodes[u]['group'] == G.nodes[v]['group']]
    inter = [(u, v) for u, v in G.edges()
             if G.nodes[u]['group'] != G.nodes[v]['group']]

    xs_intra = [[pos[u][0], pos[v][0]] for u, v in intra]
    ys_intra = [[pos[u][1], pos[v][1]] for u, v in intra]
    xs_inter = [[pos[u][0], pos[v][0]] for u, v in inter]
    ys_inter = [[pos[u][1], pos[v][1]] for u, v in inter]

    def draw_frame(frame_idx):
        ax.clear()
        state = frames[frame_idx]

        # Edges
        for xs, ys in zip(xs_intra, ys_intra):
            ax.plot(xs, ys, color='#aaaaaa', alpha=0.2, linewidth=0.5, zorder=1)
        for xs, ys in zip(xs_inter, ys_inter):
            ax.plot(xs, ys, color='#4488cc', alpha=0.4, linewidth=0.8, zorder=1)

        # Nodes
        for node in G.nodes():
            x, y = pos[node]
            if node in calib_set:
                c, m, s = CALIB_COL, '*', 300
            elif state[node] == 1:
                c, m, s = INFECTED_COL, 'o', 160
            else:
                c, m, s = HEALTHY_COL, 'o', 100
            ax.scatter(x, y, c=c, marker=m, s=s,
                       zorder=2, edgecolors='white', linewidths=0.5)

        inf_count = sum(state.values())
        ax.set_title(f'{title}\nStep {frame_idx}  —  '
                     f'{inf_count} infected / {len(state)} total',
                     fontsize=10, fontweight='bold')

        legend_elements = [
            Patch(fc=HEALTHY_COL,  label='Healthy'),
            Patch(fc=INFECTED_COL, label='Infected'),
            Patch(fc=CALIB_COL,    label='Calibration'),
        ]
        ax.legend(handles=legend_elements, loc='upper left',
                  fontsize=8, framealpha=0.85)
        ax.axis('off')

    ani = animation.FuncAnimation(
        fig, draw_frame,
        frames=len(frames),
        interval=1000 // fps,
        repeat=True,
    )

    ani.save(output_path, writer='pillow', fps=fps)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def generate_animations(seed: int = 42) -> None:
    np.random.seed(seed)
    N, K = 80, 4

    configs = [
        ('Low homophily (h=2) — errors spread widely',
         0.6, 0.30, 'animation_low_homophily.gif'),
        ('High homophily (h=20) — errors trapped in squad',
         0.6, 0.03, 'animation_high_homophily.gif'),
    ]

    for title, p_in, p_out, fname in configs:
        print(f"  Generating: {fname}  (h={p_in/p_out:.1f})")
        G     = generate_network(N, K, p_in, p_out, seed=seed)
        state, calib = initialize_agents(G, 2, 'concentrated',
                                          2, 'scattered', K)
        pos   = _group_layout(G, seed=seed)

        res = run_simulation(G, state, calib,
                             alpha=0.6, beta=0.3, threshold=0.4,
                             r=2, p_c=0.7, max_steps=50,
                             record_states=True)

        out_path = os.path.join(get_output_dir(), fname)
        make_animation(G, res['state_history'], calib, pos,
                       title=title, output_path=out_path, fps=5)


if __name__ == '__main__':
    print("Generating infection spread animations...")
    try:
        generate_animations()
        print("Animations complete ✓")
        print("  → Add these GIFs to your GitHub README!")
    except Exception as e:
        print(f"  Animation failed: {e}")
        print("  Try: pip install pillow")
