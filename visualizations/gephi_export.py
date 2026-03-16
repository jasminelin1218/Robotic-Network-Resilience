"""
visualizations/gephi_export.py
================================
Export NetworkX graphs as .gexf files for Gephi visualization.

Usage
-----
    python visualizations/gephi_export.py

Then in Gephi:
  1. File → Open → select a .gexf file
  2. Layout → ForceAtlas2 → Run for ~30s → Stop
  3. Appearance → Nodes → Color → Partition → 'group'
  4. Appearance → Nodes → Size → Ranking → 'degree'
  5. File → Export → Graph file → PNG (high resolution)
"""

import networkx as nx
from core.paths import get_output_dir
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.simulation import generate_network, initialize_agents


N = 100
K = 4


def _enrich(G: nx.Graph, state: dict, calib_set: set) -> nx.Graph:
    """Add node attributes Gephi can read for coloring."""
    for node in G.nodes():
        G.nodes[node]['infection_state'] = int(state[node])
        G.nodes[node]['is_calibration']  = int(node in calib_set)
        G.nodes[node]['degree']          = G.degree(node)
        # combined_state: 0=healthy, 1=infected, 2=calibration
        if node in calib_set:
            G.nodes[node]['node_type'] = 2
        elif state[node] == 1:
            G.nodes[node]['node_type'] = 1
        else:
            G.nodes[node]['node_type'] = 0
    for u, v in G.edges():
        G[u][v]['edge_type'] = (
            'intra' if G.nodes[u]['group'] == G.nodes[v]['group'] else 'inter'
        )
    return G


def export_all(seed: int = 42) -> None:
    np.random.seed(seed)
    configs = [
        ('low_homophily_h2',    0.6, 0.30, 2, 0),
        ('high_homophily_h10',  0.6, 0.06, 2, 0),
        ('high_homophily_h20',  0.6, 0.03, 2, 0),
        ('with_calibration',    0.6, 0.06, 3, 3),
    ]

    for name, p_in, p_out, atk, cal in configs:
        G = generate_network(N, K, p_in, p_out, seed=seed)
        state, calib = initialize_agents(G, atk, 'concentrated',
                                         cal, 'scattered', K)
        G = _enrich(G, state, calib)

        out_path = os.path.join(get_output_dir('gephi'), f'{name}.gexf')
        nx.write_gexf(G, out_path)
        h = p_in / p_out
        print(f"  Exported: {name}.gexf  (h={h:.1f}, "
              f"attack={atk}, calib={cal})")

    print(f"\n  All .gexf files saved to: {get_output_dir('gephi')}")
    print("\n  Gephi instructions:")
    print("  1. Open Gephi → File → Open → select a .gexf file")
    print("  2. Layout panel → ForceAtlas2 → Run 30s → Stop")
    print("  3. Appearance → Nodes → Color → Partition → 'group'")
    print("  4. Appearance → Nodes → Size → Ranking → 'degree'")
    print("  5. File → Export → Graph file → PNG at 2000px")
    print("  6. Save images to outputs/gephi/screenshots/")


if __name__ == '__main__':
    print("Exporting Gephi (.gexf) files...")
    export_all()
    print("Gephi export complete ✓")
