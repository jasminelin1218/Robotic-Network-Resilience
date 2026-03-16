"""
app.py — Streamlit Interactive Demo
=====================================
ECE227 Project: Network Resilience in Multi-Robot Systems

Run locally:
    streamlit run app.py

Deploy to cloud:
    1. Push this repo to GitHub
    2. Go to share.streamlit.io
    3. Connect repo → set main file = app.py → Deploy
    4. Copy the public URL into your README badge
"""

import streamlit as st
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from core.simulation import generate_network, initialize_agents, run_simulation
from core.metrics import algebraic_connectivity

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Robot Network Resilience Simulator",
    page_icon="🤖",
    layout="wide",
)

# ── Colour constants ──────────────────────────────────────────────────────────
HEALTHY_COL  = '#2d9e5a'
INFECTED_COL = '#e84343'
CALIB_COL    = '#f5a623'
GROUP_PALETTES = ['#3a7bd5', '#2d9e5a', '#e84343', '#f5a623',
                  '#9b59b6', '#e67e22', '#1abc9c', '#e91e63']


# ── Sidebar: all parameters ───────────────────────────────────────────────────
with st.sidebar:
    st.title("🤖 Simulation Parameters")
    st.markdown("---")

    st.subheader("Network structure")
    N    = st.slider("Total robots (N)", 20, 200, 100, 10)
    K    = st.slider("Number of squads (K)", 2, 8, 4, 1)
    p_in = st.slider("Intra-group edge prob (p_in)", 0.1, 0.9, 0.6, 0.05)
    p_out= st.slider("Inter-group edge prob (p_out)", 0.01, 0.30, 0.05, 0.01)
    h    = p_in / p_out
    st.metric("Homophily ratio h = p_in / p_out", f"{h:.1f}")

    st.markdown("---")
    st.subheader("Attack parameters")
    attack_count = st.slider("Attack seeds", 1, max(2, N//5), 2, 1)
    attack_dist  = st.selectbox("Attack distribution",
                                ['concentrated', 'scattered'])

    st.markdown("---")
    st.subheader("Intervention")
    calib_count = st.slider("Calibration nodes", 0, 10, 2, 1)
    calib_dist  = st.selectbox("Calibration placement",
                               ['scattered', 'concentrated',
                                'attack_group', 'other_group'])
    use_dd      = st.checkbox("Dynamic edge severance", value=False)
    sev_h       = st.slider("Severance threshold", 0.3, 0.9, 0.6, 0.05,
                            disabled=not use_dd)

    st.markdown("---")
    st.subheader("Contagion model")
    alpha     = st.slider("α (in-group weight)",  0.0, 1.0, 0.5, 0.05)
    beta      = st.slider("β (out-group weight)", 0.0, 1.0, 0.5, 0.05)
    threshold = st.slider("Infection threshold",  0.1, 0.9, 0.4, 0.05)
    r         = st.slider("Recovery peer count (r)", 1, 5, 2, 1)
    p_c       = st.slider("Recovery probability (p_c)", 0.1, 1.0, 0.8, 0.05)
    max_steps = st.slider("Max simulation steps", 50, 500, 200, 50)

    st.markdown("---")
    seed = st.number_input("Random seed", min_value=0, max_value=9999,
                           value=42, step=1)
    run_btn = st.button("▶  Run Simulation", type="primary", use_container_width=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.title("Network Resilience in Multi-Robot Systems")
st.markdown(
    "Simulating how **sensor-error misinformation spreads** through a robot swarm "
    "modelled as a Stochastic Block Model (SBM) network. "
    "Adjust parameters in the sidebar and click **Run Simulation**."
)

# ── Helper: draw network ──────────────────────────────────────────────────────
def draw_network(G, state, calib_set, pos, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    intra = [(u, v) for u, v in G.edges()
             if G.nodes[u]['group'] == G.nodes[v]['group']]
    inter = [(u, v) for u, v in G.edges()
             if G.nodes[u]['group'] != G.nodes[v]['group']]
    nx.draw_networkx_edges(G, pos, edgelist=intra, ax=ax,
                           edge_color='#bbbbbb', alpha=0.3, width=0.5)
    nx.draw_networkx_edges(G, pos, edgelist=inter, ax=ax,
                           edge_color='#4488cc', alpha=0.5, width=0.9)

    for node in G.nodes():
        x, y = pos[node]
        if node in calib_set:
            c, m, s = CALIB_COL, '*', 280
        elif state[node] == 1:
            c, m, s = INFECTED_COL, 'o', 150
        else:
            c, m, s = HEALTHY_COL, 'o', 100
        ax.scatter(x, y, c=c, marker=m, s=s,
                   zorder=3, edgecolors='white', linewidths=0.5)

    legend_handles = [
        mpatches.Patch(fc=HEALTHY_COL,  label='Healthy'),
        mpatches.Patch(fc=INFECTED_COL, label='Infected'),
        mpatches.Patch(fc=CALIB_COL,    label='Calibration'),
    ]
    ax.legend(handles=legend_handles, loc='upper left', fontsize=8)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.axis('off')
    fig.tight_layout()
    return fig


# ── Run simulation ────────────────────────────────────────────────────────────
if run_btn:
    np.random.seed(int(seed))

    with st.spinner("Building network and running simulation..."):
        G = generate_network(N, K, p_in, p_out, seed=int(seed))
        state0, calib_set = initialize_agents(
            G, attack_count, attack_dist, calib_count, calib_dist, K)

        # Layout (group-aware spring)
        G2 = G.copy()
        for u, v in G2.edges():
            G2[u][v]['weight'] = (3.0 if G2.nodes[u]['group'] ==
                                  G2.nodes[v]['group'] else 0.3)
        pos = nx.spring_layout(G2, weight='weight', seed=int(seed), k=1.2)

        res = run_simulation(
            G, state0, calib_set,
            alpha=alpha, beta=beta, threshold=threshold,
            r=r, p_c=p_c,
            use_dynamic_defense=use_dd, severance_h=sev_h,
            max_steps=max_steps,
        )

        # Final state reconstruction from infection curve
        # (run a second quick pass to get per-node states)
        state_final = state0.copy()
        for _ in range(max_steps):
            nxt = state_final.copy()
            changed = False
            for node in G.nodes():
                if node in calib_set:
                    nxt[node] = 0
                    continue
                grp = G.nodes[node]['group']
                nb  = list(G.neighbors(node))
                in_nb  = [n for n in nb if G.nodes[n]['group'] == grp]
                out_nb = [n for n in nb if G.nodes[n]['group'] != grp]
                if state_final[node] == 0:
                    ir  = sum(state_final[n] for n in in_nb) / len(in_nb)  if in_nb  else 0
                    or_ = sum(state_final[n] for n in out_nb)/ len(out_nb) if out_nb else 0
                    score = (alpha + beta) * ir if not out_nb else (
                            (alpha + beta) * or_ if not in_nb else
                            alpha * ir + beta * or_)
                    if score >= threshold:
                        nxt[node] = 1; changed = True
                else:
                    nc = any(n in calib_set for n in nb)
                    hi = sum(1 for n in in_nb if state_final[n] == 0)
                    if nc or hi >= r:
                        if np.random.random() < p_c:
                            nxt[node] = 0; changed = True
            state_final = nxt
            if not changed:
                break

        lam2 = algebraic_connectivity(G)

    # ── Metric cards ──────────────────────────────────────────────────────────
    st.markdown("---")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Peak infection",
                f"{res['peak_infection']:.1%}")
    col2.metric("Final infection",
                f"{res['final_infection']:.1%}")
    col3.metric("Convergence step",
                str(res['convergence_step']) if res['convergence_step'] != -1
                else "> " + str(max_steps))
    col4.metric("Algebraic connectivity λ₂", f"{lam2:.4f}")
    col5.metric("Homophily h", f"{h:.1f}")

    # ── Two-column layout: network | infection curve ──────────────────────────
    st.markdown("---")
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Network state at t = 0  →  converged")
        tab1, tab2 = st.tabs(["Initial (t = 0)", "Final (converged)"])
        with tab1:
            st.pyplot(draw_network(G, state0, calib_set, pos,
                                   f"Initial — {sum(state0.values())} infected"))
        with tab2:
            st.pyplot(draw_network(G, state_final, calib_set, pos,
                                   f"Final — {sum(state_final.values())} infected"))

    with right:
        st.subheader("Infection curve over time")
        curve = res['infection_over_time']
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(curve, color='#e84343', linewidth=2.5, label='Infection rate')
        ax2.axhline(res['peak_infection'], color='gray',
                    linestyle='--', linewidth=1, alpha=0.6,
                    label=f"Peak = {res['peak_infection']:.2f}")
        ax2.axhline(0.10, color='green', linestyle=':', linewidth=1,
                    alpha=0.8, label='Recovery threshold (10%)')
        ax2.fill_between(range(len(curve)), curve, alpha=0.12, color='#e84343')
        ax2.set_xlabel('Simulation step')
        ax2.set_ylabel('Fraction infected')
        ax2.set_ylim(0, 1.05)
        ax2.set_title('Infection dynamics')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        st.pyplot(fig2)

        # Per-group breakdown bar chart
        st.subheader("Final infection by squad")
        by_group = {}
        for node in G.nodes():
            g = G.nodes[node]['group']
            by_group.setdefault(g, {'healthy': 0, 'infected': 0, 'calib': 0})
            if node in calib_set:
                by_group[g]['calib'] += 1
            elif state_final[node] == 1:
                by_group[g]['infected'] += 1
            else:
                by_group[g]['healthy'] += 1

        fig3, ax3 = plt.subplots(figsize=(6, 3))
        groups = sorted(by_group.keys())
        x = np.arange(len(groups))
        w = 0.25
        ax3.bar(x - w, [by_group[g]['healthy']  for g in groups],
                w, label='Healthy',  color=HEALTHY_COL,  alpha=0.85)
        ax3.bar(x,     [by_group[g]['infected'] for g in groups],
                w, label='Infected', color=INFECTED_COL, alpha=0.85)
        ax3.bar(x + w, [by_group[g]['calib']    for g in groups],
                w, label='Calibration', color=CALIB_COL, alpha=0.85)
        ax3.set_xticks(x)
        ax3.set_xticklabels([f'Squad {g}' for g in groups])
        ax3.set_ylabel('Number of robots')
        ax3.set_title('Per-squad breakdown at convergence')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3, axis='y')
        fig3.tight_layout()
        st.pyplot(fig3)

else:
    # Welcome state — shown before first run
    st.info(
        "👈  Set your parameters in the sidebar, then click **Run Simulation**.\n\n"
        "**Quick start:** Try raising **p_out** from 0.05 → 0.20 to see how "
        "more inter-group connections speed up both infection and recovery."
    )
    st.markdown("""
    ### What this simulates
    | Parameter | What it controls |
    |-----------|-----------------|
    | K | Number of robot squads |
    | p_in / p_out | Edge density within vs between squads |
    | h = p_in/p_out | **Homophily ratio** — higher = more segregated |
    | α, β | Trust weight on in-group vs out-group peers |
    | Threshold | Fraction of infected neighbours needed to spread error |
    | Calibration nodes | Always-correct "GPS anchor" robots |
    | Dynamic severance | Firewall — cuts links to heavily-infected neighbours |
    """)
