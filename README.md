# Homophily and Information Segregation in Multi-Agent Robotic Networks
**ECE 227 — Network Science · Final Project**

> How does network homophily influence the spread of sensor-error misinformation through a robot swarm — and what structural interventions enable recovery?

---

## Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

> _Deploy to [share.streamlit.io](https://share.streamlit.io) and replace the URL above._

---

## Overview

We model a multi-robot swarm as a **Stochastic Block Model (SBM)** network where:

- **Communities** → robot squads operating in different regions
- **Homophily** → robots preferentially communicate within the same squad (ratio h = p_in / p_out)
- **Infection** → sensor misclassification spreading via threshold contagion
- **Recovery** → calibration nodes (always-correct GPS anchors) or in-group peer pressure

**Three research questions:**

1. How does network homophily influence the spread of erroneous information?
2. Under what structural conditions does the system recover from misinformation?
3. Can increasing cross-community connectivity improve resilience?

---

## Repository Structure

```
ECE227-project/
├── core/
│   ├── simulation.py       # Network generation, agents, simulation engine, Monte Carlo
│   └── metrics.py          # λ₂, h_crit, recovery rate, cascade threshold
├── experiments/
│   ├── exp1_topology.py    # K × p_out heatmap + optimal K*
│   ├── exp2_calibration.py # Calibration node count × placement strategy
│   ├── exp3_attack.py      # Attack intensity + cascade threshold
│   ├── exp4_dynamic_defense.py  # Dynamic edge severance vs static
│   ├── exp5_alpha_beta.py  # α/β weight sensitivity
│   ├── exp6_phase_diagram.py    # 2D phase diagram: h × attack → recovery/collapse
│   └── exp7_spectral.py    # Algebraic connectivity λ₂ vs infection (Fiedler)
├── visualizations/
│   ├── network_plots.py    # Snapshot trios + homophily comparison figures
│   ├── gephi_export.py     # Export .gexf files for Gephi
│   └── animation.py        # Animated GIFs of infection spreading
├── outputs/                # Generated on run (gitignored)
│   ├── figures/            # All matplotlib figures (.png)
│   └── gephi/              # .gexf files for Gephi
├── app.py                  # Streamlit interactive demo
├── run_all.py              # Reproduce all results with one command
└── requirements.txt
```

---

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/ECE227-project.git
cd ECE227-project
pip install -r requirements.txt

# Run all experiments (≈ 3–5 min with parallelism)
python run_all.py

# Quick test run (20 Monte Carlo trials instead of 100)
python run_all.py --fast

# Run specific experiments only
python run_all.py --exp 1 6 7

# Launch interactive demo
streamlit run app.py

# Export Gephi files (then open in Gephi GUI)
python visualizations/gephi_export.py
```

---

## Experiments

| # | Name | Key Finding |
|---|------|-------------|
| 1 | Topological Resilience | Optimal squad count K* minimises infection across all p_out values |
| 2 | Calibration Deployment | Scattered nodes outperform concentrated; placement near attack source matters |
| 3 | Attack Robustness | Concentrated attacks reach cascade threshold at lower intensity |
| 4 | Dynamic Defense | Optimal severance threshold h exists; too aggressive cuts recovery pathways |
| 5 | α/β Sensitivity | High α (in-group bias) makes errors sticky but slows cross-group spread |
| 6 | **Phase Diagram** | 2D boundary between recovery and collapse in (h, attack) space |
| 7 | **Spectral Analysis** | λ₂ correlates positively with peak infection (Fiedler theorem) |

---

## Model Details

### Infection rule
```
score = α · r_in + β · r_out  ≥  threshold  →  robot becomes infected
```
where `r_in`, `r_out` are infected fractions among in-/out-group neighbours.

### Recovery rule
Infected robot recovers with probability `p_c` if:
- Connected to a calibration node, **or**
- ≥ `r` of its in-group neighbours are healthy

### Dynamic defense (optional)
If out-group infected fraction > `severance_h`, sever those cross-group edges.

---

## Reproducibility

All random seeds are controllable. Full reproduction:
```bash
python run_all.py        # generates all figures in outputs/figures/
```
Monte Carlo parallelism via `joblib` — runtime ≈ 3 min on a 4-core machine.

---

## Tools

- Python 3.10+ · NetworkX · NumPy · SciPy · Matplotlib · Streamlit · joblib
- Gephi (external GUI) for network visualization

---

## Authors

_ECE 227 · UC San Diego_
