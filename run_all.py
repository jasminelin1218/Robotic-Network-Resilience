"""
run_all.py
==========
One command to reproduce all results.

    python run_all.py              # full run, RUNS=100
    python run_all.py --fast       # quick test, RUNS=20
    python run_all.py --exp 1 2 6  # only specific experiments

Outputs
-------
All figures → outputs/
All .gexf   → outputs/gephi/
Summary table printed to console
"""

import argparse
import time
import numpy as np
import sys, os

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description='Run all ECE227 project experiments.')
parser.add_argument('--fast', action='store_true',
                    help='Quick run with RUNS=20 (for testing)')
parser.add_argument('--exp', nargs='+', type=int,
                    help='Run only specific experiment numbers, e.g. --exp 1 6')
parser.add_argument('--name', type=str, default=None,
                    help='Custom folder name (default: run_YYYYMMDD_HHMMSS)')
args = parser.parse_args()

RUNS = 20 if args.fast else 100
SAVE_FIGS = True   # always save figures

if args.fast:
    print(f"⚡ Fast mode: RUNS={RUNS}")

# ── Import all modules ────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

# Use non-interactive backend so plt.show() never blocks and
# figures save silently without opening windows
import matplotlib
matplotlib.use('Agg')

from core.paths import set_run_folder, get_run_folder
run_folder = set_run_folder()

import experiments.exp1_topology       as exp1
import experiments.exp2_calibration    as exp2
import experiments.exp3_attack         as exp3
import experiments.exp4_dynamic_defense as exp4
import experiments.exp5_alpha_beta     as exp5
import experiments.exp6_phase_diagram  as exp6
import experiments.exp7_spectral       as exp7
import experiments.exp8_scalability    as exp8
import experiments.exp9_disparity      as exp9
import experiments.exp10_feedback_loop  as exp10
import experiments.exp_centrality      as exp_cent
import experiments.exp_degree_dist     as exp_deg
import visualizations.network_plots    as net_plots
import visualizations.gephi_export     as gephi
import visualizations.animation        as anim

EXP_MAP = {
    1: ('Topological Resilience',          exp1),
    2: ('Calibration Node Strategy',       exp2),
    3: ('Attack Robustness',               exp3),
    4: ('Dynamic Defense',                 exp4),
    5: ('α/β Sensitivity',                 exp5),
    6: ('Phase Diagram',                   exp6),
    7: ('Spectral Analysis',               exp7),
    8: ('Scalability + Graph Comparison',  exp8),
    9: ('Inter-Group Disparity (Topic 6)',  exp9),
   10: ('Feedback Loop (Topic 6)',           exp10),
   11: ('Centrality Analysis',               exp_cent),
   12: ('Degree Distribution',               exp_deg),
}

to_run = args.exp if args.exp else list(EXP_MAP.keys())

print("\n" + "="*65)
print(f"  Output folder: outputs/")
print("  ECE227 — Robot Network Resilience: Full Experiment Suite")
print("="*65)
print(f"  Runs per config : {RUNS}")
print(f"  Experiments     : {to_run}")
print(f"  Save figures    : always → outputs/")
print("="*65 + "\n")

# ── Network visualisations (always generated) ─────────────────────────────────
print("Generating network visualizations...")
t0 = time.time()
net_plots.plot_snapshot_trio()
net_plots.plot_homophily_compare()
print(f"  ✓ done in {time.time()-t0:.1f}s\n")

# ── Gephi export ──────────────────────────────────────────────────────────────
print("Exporting Gephi files...")
t0 = time.time()
gephi.export_all()  # gephi .gexf files always export (they're small and fast)
print(f"  ✓ done in {time.time()-t0:.1f}s\n")

# ── Experiments ───────────────────────────────────────────────────────────────
all_results = {}
total_t0 = time.time()

for exp_num in to_run:
    if exp_num not in EXP_MAP:
        print(f"  WARNING: Experiment {exp_num} not recognised, skipping.")
        continue

    name, module = EXP_MAP[exp_num]
    print(f"Running Experiment {exp_num} — {name}  (RUNS={RUNS})")
    t0 = time.time()

    # exp7 uses 'trials' instead of 'runs' (per-p_out repetitions)
    if exp_num == 7:
        data = module.run(trials=max(RUNS // 3, 10), verbose=True)
    else:
        data = module.run(runs=RUNS, verbose=True)
    module.plot(data, save=SAVE_FIGS)
    all_results[exp_num] = data

    print(f"  ✓ Experiment {exp_num} complete in {time.time()-t0:.1f}s\n")

# ── Animations ────────────────────────────────────────────────────────────────
print("Generating animations (GIFs)...")
t0 = time.time()
try:
    anim.generate_animations()
    print(f"  ✓ done in {time.time()-t0:.1f}s\n")
except Exception as e:
    print(f"  ⚠ Animation skipped: {e}  (install pillow to enable)\n")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  RESULTS SUMMARY")
print("="*65)

if 1 in all_results:
    d = all_results[1]
    best = d['heat_peak'].min(axis=1)
    K_star = d['K_list'][int(np.argmin(best))]
    print(f"  Exp 1 — Optimal squad count K* = {K_star}")
    print(f"          Best peak infection    = {best.min():.3f}")

if 2 in all_results:
    d = all_results[2]
    for dist in exp2.DISTRIBUTIONS:
        best_idx = int(np.argmin(d['results'][dist]['final']))
        print(f"  Exp 2 — {dist:13s}: best at "
              f"{d['calib_counts'][best_idx]} calib nodes, "
              f"final = {min(d['results'][dist]['final']):.3f}")

if 3 in all_results:
    d = all_results[3]
    from core.metrics import find_cascade_threshold
    for dist in exp3.ATTACK_DISTRIBUTIONS:
        ct = find_cascade_threshold(d['results'][dist]['peak'],
                                    d['attack_intensities'])
        print(f"  Exp 3 — {dist:13s}: cascade threshold ≈ "
              f"{ct:.1f} seeds" if ct else
              f"  Exp 3 — {dist:13s}: threshold not reached")

if 4 in all_results:
    d = all_results[4]
    best_h  = d['severance_thresholds'][int(np.argmin(d['dyn']['peak']))]
    improve = d['static_res']['mean_peak'] - min(d['dyn']['peak'])
    print(f"  Exp 4 — Optimal severance h = {best_h:.2f}, "
          f"peak improvement = {improve:.3f}")

if 7 in all_results:
    d = all_results[7]
    print(f"  Exp 7 — Pearson r (λ₂ vs peak):  r = {d['r_peak']:+.4f}"
          f"  (p = {d['p_peak']:.4f})")
    print(f"          Pearson r (λ₂ vs final): r = {d['r_final']:+.4f}"
          f"  (p = {d['p_final']:.4f})")

total_time = time.time() - total_t0
print(f"\n  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
print(f"  All figures saved to: outputs/")
print(f"  All Gephi files:      outputs/gephi/")

if 9 in all_results:
    d = all_results[9]
    sig = ('***' if d['p_delta_h'] < 0.001 else '**' if d['p_delta_h'] < 0.01
           else '*' if d['p_delta_h'] < 0.05 else 'n.s.')
    print(f"  Exp 9 — Disparity r vs h:       r = {d['r_delta_h']:+.4f} {sig}")
    print(f"          Disparity r vs lambda2:  r = {d['r_delta_lam']:+.4f}")
    max_idx = int(np.argmax(d['delta_final']))
    print(f"          Max Delta at h={d['h'][max_idx]:.1f}: {d['delta_final'][max_idx]:+.3f}")

if 10 in all_results:
    d = all_results[10]
    for label in list(d['data'].keys())[:1]:
        s_d = d['data'][label]['static']['delta'][-1]
        a_d = d['data'][label]['adaptive']['delta'][-1]
        print(f"  Exp 10 — ({label}) Static disparity={s_d:+.3f}, Adaptive={a_d:+.3f}")
print("="*65)
print("\n  Next steps:")
print("  1. Open Gephi → load .gexf files → export screenshots")
print("  2. streamlit run app.py   (to launch the interactive demo)")
print("  3. Write your report using figures from outputs/figures/")
