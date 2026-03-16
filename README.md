# Homophily and Information Segregation in Multi-Agent Robotic Networks
**ECE 227 — Network Science · UC San Diego · Final Project**

> How does network homophily influence the spread of sensor-error misinformation through a robot swarm — and what structural interventions enable recovery?

---

## Overview

We model a multi-robot swarm as a **Stochastic Block Model (SBM)** network where communities represent squads operating in different physical regions. A sensor-error "infection" spreads through the network via threshold contagion, and we study how the degree of homophily (within-squad vs. cross-squad communication) governs both the spread and recovery of misinformation.

**Three core research questions:**
1. How does network homophily influence the spread of erroneous sensor readings?
2. Under what structural conditions does the swarm recover from misinformation cascades?
3. What interventions — calibration nodes, dynamic defense, squad sizing — maximize resilience?

---

## Model

### Network
A **Stochastic Block Model** with K communities of equal size.
- `p_in` — intra-squad edge probability
- `p_out` — inter-squad edge probability
- `h = p_in / p_out` — homophily ratio

### Infection rule
```
score = α · r_in + β · r_out  ≥  threshold   →   robot becomes infected
```
`r_in` / `r_out` are the infected fractions among in- and out-group neighbours respectively.
`α` weights in-group peer pressure; `β` weights out-group peer pressure.

### Recovery rule
An infected robot recovers with probability `p_c` each step if:
- it is directly connected to a **calibration node** (always-correct GPS anchor), **or**
- at least `r` of its in-group neighbours are currently healthy

### Dynamic defense (optional)
When a robot's out-group infected fraction exceeds a severance threshold `h_sev`, all cross-group edges to infected neighbours are severed for the remainder of the run.

---

## Repository Structure

```
ece227_project/
├── core/
│   ├── simulation.py        # SBM generation, agent init, simulation engine, Monte Carlo
│   ├── metrics.py           # λ₂, Cheeger constant, cascade threshold, h_crit
│   └── paths.py             # Output directory helpers; sets Agg backend on import
├── experiments/
│   ├── exp1_topology.py     # K × p_out heatmap + optimal K* (with 95% CI bands)
│   ├── exp2_calibration.py  # Calibration count × placement strategy (Welch's t-test)
│   ├── exp3_attack.py       # Attack intensity sweep + cascade threshold detection
│   ├── exp4_dynamic_defense.py  # Dynamic severance vs static (Welch's t-test)
│   ├── exp5_alpha_beta.py   # α/β in-/out-group weight sensitivity surface
│   ├── exp6_phase_diagram.py    # 2D phase diagram: (h, attack) → recovery / collapse
│   ├── exp7_spectral.py     # Algebraic connectivity λ₂ + Cheeger constant vs infection
│   ├── exp8_scalability.py  # (a) Swarm size N=50→500; (b) SBM vs ER / BA / WS
│   ├── exp9_disparity.py    # Inter-group infection disparity ΔInfection(h)
│   ├── exp10_feedback_loop.py   # Repeated attack waves with adaptive threshold defense
│   ├── exp_centrality.py    # Degree / betweenness / eigenvector / closeness vs h
│   └── exp_degree_dist.py   # Degree distribution, clustering, path length vs h
├── visualizations/
│   ├── network_plots.py     # Publication-quality network snapshots (circular squad layout)
│   ├── gephi_export.py      # Export .gexf files for Gephi
│   └── animation.py         # Animated GIFs of infection spreading
├── outputs/                 # Auto-generated on run (figures/ sub-dir is gitignored)
│   ├── exp*.png             # One figure per experiment
│   ├── *.gif                # Infection spread animations
│   └── gephi/               # .gexf files for Gephi
├── app.py                   # Streamlit interactive demo
├── run_all.py               # Reproduce all results with one command
├── requirements.txt
└── demo.html                # Static demo preview
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/jasminelin1218/Robotic-Network-Resilience.git
cd Robotic-Network-Resilience

# 2. Install dependencies (Python 3.10+)
pip install -r requirements.txt

# 3. Run all 12 experiments  (~3–5 min, parallelised via joblib)
python run_all.py

# Fast test run (20 Monte Carlo trials instead of 100, still saves plots)
python run_all.py --fast

# Run specific experiments only
python run_all.py --exp 1 6 7

# 4. Launch interactive demo
streamlit run app.py

# 5. Export Gephi files
python visualizations/gephi_export.py
```

> **Conda users:** activate the environment before running:
> `conda activate course && python run_all.py`

All output figures are saved to `outputs/` automatically.

---

## Experiments

| # | Script | What it measures | Key finding |
|---|--------|-----------------|-------------|
| 1 | `exp1_topology.py` | K (squad count) × p_out heatmap | There is an optimal K* that minimises peak infection; too many or too few squads increases vulnerability |
| 2 | `exp2_calibration.py` | Calibration node count × placement strategy | Placing calibration nodes in the **attacked group** outperforms all other strategies; concentration near the source beats scattering |
| 3 | `exp3_attack.py` | Attack seed count vs infection cascade | Clear cascade threshold at 8–12 seeds — below which the swarm recovers, above which it collapses |
| 4 | `exp4_dynamic_defense.py` | Severance threshold h_sev sweep | Optimal h_sev exists; too aggressive a cutoff severs recovery pathways alongside infection pathways |
| 5 | `exp5_alpha_beta.py` | α/β weight sensitivity surface | High α (in-group bias) keeps errors local but makes them sticky; high β accelerates both spread and recovery |
| 6 | `exp6_phase_diagram.py` | 2D phase diagram: homophily × attack intensity | Defines the exact h_crit boundary separating recovery from collapse in parameter space |
| 7 | `exp7_spectral.py` | Algebraic connectivity λ₂ + Cheeger constant | λ₂ correlates positively with peak infection — well-connected graphs spread errors faster but also recover faster |
| 8 | `exp8_scalability.py` | (a) N=50→500; (b) SBM vs ER/BA/WS | Cascade behaviour is robust across swarm sizes; SBM produces qualitatively distinct dynamics from scale-free (BA) graphs |
| 9 | `exp9_disparity.py` | ΔInfection = squad-0 rate − other-squad rate | High homophily traps misinformation inside the attacked squad — disparity increases monotonically with h |
| 10 | `exp10_feedback_loop.py` | Repeated attack waves with adaptive defense | Adaptive threshold lowering confers measurable resilience improvement across successive attack rounds |
| C | `exp_centrality.py` | Degree / betweenness / eigenvector / closeness | At high h, betweenness concentrates on inter-group bridge nodes — the best candidates for calibration placement |
| D | `exp_degree_dist.py` | Degree distribution + clustering + path length | Low-h SBM ≈ Poisson (ER-like); high-h SBM shows a bimodal distribution reflecting intra/inter-group structure |

---

## Statistical Methods

All experiments run **100 independent Monte Carlo trials** (parallelised via `joblib`).

- **95% confidence intervals** — `± 1.96 · σ / √n` shown as shaded bands on line charts
- **Welch's t-tests** — used in Exp 2 and Exp 4 to compare strategies with unequal variance; annotated on plots as `t`, `p`, and significance level (`*`, `**`, `***`)

---

## Output Artifacts

After `python run_all.py`, the `outputs/` directory contains:

| File | Description |
|------|-------------|
| `exp1_heatmaps.png`, `exp1_optimal_K.png` | Topology heatmap + optimal K line chart |
| `exp2_calibration.png` | Calibration strategy comparison |
| `exp3_attack.png` | Attack cascade threshold |
| `exp4_dynamic_defense.png` | Dynamic vs static defense |
| `exp5_alpha_beta.png` | α/β sensitivity surface |
| `exp6_phase_diagram.png`, `exp6_hcrit_curve.png` | Phase boundary plots |
| `exp7_spectral.png`, `exp7_fiedler.png`, `exp7_cheeger.png` | Spectral analysis |
| `exp8a_scaling.png`, `exp8b_graph_comparison.png` | Scalability results |
| `exp9_disparity.png`, `exp9_disparity_spectral.png` | Inter-group disparity |
| `exp10_feedback_loop.png` | Adaptive defense over rounds |
| `exp_centrality.png` | Centrality distributions |
| `exp_degree_dist.png` | Degree distribution + statistics |
| `network_snapshots.png`, `network_homophily_compare.png`, `network_progression.png` | Network visualisations |
| `animation_low_homophily.gif`, `animation_high_homophily.gif` | Infection spread animations |
| `gephi/*.gexf` | Gephi-ready network files |

---

## Dependencies

| Package | Use |
|---------|-----|
| `networkx >= 3.0` | Graph generation and analysis |
| `numpy >= 1.24` | Numerical computation |
| `scipy >= 1.10` | Spectral analysis, statistical tests |
| `matplotlib >= 3.7` | Figures |
| `joblib >= 1.3` | Parallel Monte Carlo |
| `streamlit >= 1.32` | Interactive demo |
| `pillow >= 10.0` | GIF animation export |
| Gephi (external) | Network visualisation GUI |

---

