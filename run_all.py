"""
run_all.py
==========
Master runner — reproduces every result in narrative order.

The experiments are organised as a THREE-ACT story:

  Act 1 — UNDERSTAND THE SYSTEM  (what determines vulnerability?)
    Exp 8b  → Why SBM? Validate model against ER/BA/WS
    Exp 7   → λ₂ (algebraic connectivity) predicts spread
    Exp 1   → K × h resilience surface — find the safe operating zone

  Act 2 — DESIGN DEFENSES  (what interventions work, and when?)
    Exp 2   → Where to place calibration nodes
    Exp 3   → How many attack seeds trigger a cascade?
    Exp 5   → How do trust weights α/β change the vulnerability?
    Exp 4   → Dynamic edge severance — does isolation help?
    Exp 6   → Phase diagram: map the h_crit boundary exactly

  Act 3 — STRESS-TEST & FAIRNESS  (does it generalise, at what cost?)
    Exp 8a  → Scalability: do results hold as N grows?
    Exp 9   → Disparity: who bears the cost of homophily? (gaming gap)
    Exp 10  → Feedback: repeated attack waves — static vs adaptive defense

  Supplementary
    exp_centrality  → Which robots are structurally most important?
    exp_degree_dist → Degree distribution confirms SBM block structure

Usage
-----
    python run_all.py              # full run, RUNS=100
    python run_all.py --fast       # quick smoke test, RUNS=20
    python run_all.py --exp 1 2 6  # only specific experiment numbers
    python run_all.py --act 1      # only Act 1 experiments (8b, 7, 1)

Outputs
-------
All figures → outputs/
All .gexf   → outputs/gephi/
"""

import argparse
import time
import numpy as np
import sys, os

# ── Argument parsing ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description='Run ECE227 experiments in narrative order.')
parser.add_argument('--fast', action='store_true',
                    help='Quick run: RUNS=20 (for testing)')
parser.add_argument('--runs', type=int, default=None,
                    help='Override number of Monte Carlo runs (e.g. --runs 200)')
parser.add_argument('--exp', nargs='+', type=int,
                    help='Run only specific experiment numbers, e.g. --exp 1 6')
parser.add_argument('--act', type=int, choices=[1, 2, 3],
                    help='Run only a specific act (1, 2, or 3)')
parser.add_argument('--name', type=str, default=None,
                    help='Custom output folder name')
args = parser.parse_args()

if args.runs is not None:
    RUNS = args.runs
elif args.fast:
    RUNS = 20
else:
    RUNS = 100
SAVE_FIGS = True

if args.fast:
    print("⚡  Fast mode: RUNS=20")

# ── Imports ────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

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


# ── Narrative structure ────────────────────────────────────────────────────────
# Each entry: (exp_number, label, module, act, hypothesis)
NARRATIVE = [
    # ── ACT 1 ─────────────────────────────────────────────────────────────────
    (None, 'ACT 1', None, 1,
     'Hypothesis: network topology — specifically λ₂ — is the primary driver\n'
     '  of error spread. Before asking how to defend, we must understand what\n'
     '  makes a network vulnerable in the first place.'),

    (8,  'Scalability + Graph Comparison (8b first: validate SBM)', exp8,  1,
     'Why SBM? Before studying homophily effects we must confirm that SBM\n'
     '  captures real robot-network structure better than ER, BA, or WS.'),

    (7,  'Spectral Analysis (λ₂ vs spread)',                         exp7,  1,
     'λ₂ (algebraic connectivity) should predict infection speed.\n'
     '  If confirmed, topology IS the vulnerability dial.'),

    (1,  'Topological Resilience (K × h surface)',                   exp1,  1,
     'Having established that λ₂ matters, we now map the full K × h\n'
     '  parameter space to find the safe operating zone.'),

    # ── ACT 2 ─────────────────────────────────────────────────────────────────
    (None, 'ACT 2', None, 2,
     'Hypothesis: calibration nodes and adaptive isolation can suppress\n'
     '  cascade; but locality of placement and timing of activation matter.'),

    (2,  'Calibration Node Strategy',                                exp2,  2,
     'Topology sets the risk; calibration nodes are our first lever.\n'
     '  Where they are placed determines how much they help.'),

    (3,  'Attack Robustness & Cascade Threshold',                    exp3,  2,
     'Calibration placement is effective — but only up to a point.\n'
     '  How many attacker seeds does it take to trigger a cascade?'),

    (5,  'α/β Trust-Weight Sensitivity',                             exp5,  2,
     'The cascade threshold depends on the trust model (α, β).\n'
     '  Do robots that weight in-group peers more heavily resist better?'),

    (4,  'Dynamic Defense (edge severance)',                         exp4,  2,
     'Trust weights alone are not enough. Can severing cross-squad\n'
     '  edges at runtime stop a cascade once it has started?'),

    (6,  'Phase Diagram (h_crit boundary)',                          exp6,  2,
     'Dynamic defense works — but only in certain regimes.\n'
     '  We now map the exact (h, attack) boundary between recovery\n'
     '  and collapse so a system designer can read off the safe zone.'),

    # ── ACT 3 ─────────────────────────────────────────────────────────────────
    (None, 'ACT 3', None, 3,
     'Hypothesis: structural inequity and scale effects will reveal\n'
     '  limits of the model — the defenses found in Act 2 may not\n'
     '  be fair or may break down at larger N.'),

    (8,  'Scalability (8a: varying N)',                              exp8,  3,
     'Do the Act 2 results hold as the swarm scales from 50 to 500?\n'
     '  If not, the defense strategy needs to be N-aware.'),

    (9,  'Inter-Group Disparity / Gaming Gap',                       exp9,  3,
     'Scalability confirmed — but who bears the cost?\n'
     '  Squad 0 (attacked) and other squads experience infection very\n'
     '  differently. We define Δ = Inf(Squad 0) − Inf(others) as the\n'
     '  "gaming gap" and measure how it grows with h.'),

    (10, 'Feedback Loop: Repeated Attack Waves',                     exp10, 3,
     'The gaming gap is real. In a repeated-attack setting does it\n'
     '  compound over time, or does adaptive defense contain it?\n'
     '  This closes the loop back to the original research objective.'),

    # ── Supplementary ──────────────────────────────────────────────────────────
    (None, 'SUPPLEMENTARY', None, 0,
     'Additional structural analyses that support the main findings.'),

    (11, 'Centrality Analysis',                                      exp_cent, 0,
     'Which robots are structurally most important? Identifies\n'
     '  high-betweenness nodes that make ideal calibration targets.'),

    (12, 'Degree Distribution',                                      exp_deg,  0,
     'Confirms the SBM block structure via degree distribution.\n'
     '  Validates Exp 8b findings from a different angle.'),
]

# exp number → module (for --exp flag)
EXP_MODULE = {row[0]: row[2] for row in NARRATIVE if row[0] is not None}
EXP_LABEL  = {row[0]: row[1] for row in NARRATIVE if row[0] is not None}
EXP_ACT    = {row[0]: row[3] for row in NARRATIVE if row[0] is not None}


# ── Determine which experiments to run ────────────────────────────────────────
if args.exp:
    to_run = [n for n in args.exp if n in EXP_MODULE]
elif args.act:
    to_run = [row[0] for row in NARRATIVE
              if row[0] is not None and row[3] == args.act]
else:
    to_run = [row[0] for row in NARRATIVE if row[0] is not None]

# deduplicate while preserving order (exp8 appears in both Act 1 and Act 3)
seen = set()
to_run_unique = []
for n in to_run:
    if n not in seen:
        seen.add(n)
        to_run_unique.append(n)
to_run = to_run_unique


# ── Header ─────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  ECE227 — Robot Network Resilience: Narrative Experiment Suite")
print("="*70)
print(f"  Runs per config : {RUNS}")
print(f"  Experiments     : {to_run}")
print(f"  Output folder   : outputs/")
print("="*70 + "\n")


# ── Network visualisations (always generated) ──────────────────────────────────
print("Generating network visualizations...")
t0 = time.time()
net_plots.plot_snapshot_trio()
net_plots.plot_homophily_compare()
print(f"  ✓ done in {time.time()-t0:.1f}s\n")

# ── Gephi export ───────────────────────────────────────────────────────────────
print("Exporting Gephi files...")
t0 = time.time()
gephi.export_all()
print(f"  ✓ done in {time.time()-t0:.1f}s\n")


# ── Main loop — narrative order ────────────────────────────────────────────────
all_results  = {}
total_t0     = time.time()
current_act  = None
_act_names   = {1: 'ACT 1 — Understand the System',
                2: 'ACT 2 — Design Defenses',
                3: 'ACT 3 — Stress-Test & Fairness',
                0: 'Supplementary'}

# Walk narrative rows in order, print banners and run matching experiments
for row in NARRATIVE:
    exp_num, label, module, act, hypothesis = row

    # ── Act banner ────────────────────────────────────────────────────────────
    if exp_num is None:
        if not args.exp:              # suppress banners when user picks exps
            print("\n" + "─"*70)
            print(f"  {_act_names[act].upper()}")
            print(f"  {hypothesis}")
            print("─"*70)
        continue

    # ── Skip if not in the run list ───────────────────────────────────────────
    if exp_num not in to_run:
        continue

    # Skip if already run (exp8 deduplication)
    if exp_num in all_results:
        continue

    # ── Run ───────────────────────────────────────────────────────────────────
    print(f"\n▶ Exp {exp_num:2d} — {label}")
    print(f"  Why now: {hypothesis.strip()}")
    t0 = time.time()

    try:
        if exp_num == 7:
            data = module.run(trials=max(RUNS // 3, 10), verbose=True)
        elif exp_num == 8:
            # run BOTH sub-experiments; results are merged in the module
            data = module.run(runs=RUNS, verbose=True)
        else:
            data = module.run(runs=RUNS, verbose=True)

        module.plot(data, save=SAVE_FIGS)
        all_results[exp_num] = data
        print(f"  ✓ complete in {time.time()-t0:.1f}s")

    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback; traceback.print_exc()


# ── Animations ─────────────────────────────────────────────────────────────────
print("\n" + "─"*70)
print("  ANIMATIONS")
print("─"*70)
print("Generating GIF animations...")
t0 = time.time()
try:
    anim.generate_animations()
    print(f"  ✓ done in {time.time()-t0:.1f}s")
except Exception as e:
    print(f"  ⚠ Animation skipped: {e}  (install pillow to enable)")


# ── Results summary ────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  RESULTS SUMMARY — closing the loop on research objectives")
print("="*70)

if 8 in all_results:
    d = all_results[8]
    if 'graph_models' in d:
        sbm_peak = d['graph_models']['SBM']['peak']
        print(f"  Exp 8b — SBM peak infection:  {sbm_peak:.3f}  "
              f"(SBM ≠ ER confirms community structure matters)")

if 7 in all_results:
    d = all_results[7]
    print(f"  Exp 7  — Pearson r (λ₂ vs peak):  r = {d['r_peak']:+.4f}"
          f"  (p = {d['p_peak']:.4f})")
    print(f"           Pearson r (λ₂ vs final): r = {d['r_final']:+.4f}"
          f"  (p = {d['p_final']:.4f})")

if 1 in all_results:
    d = all_results[1]
    best = d['heat_peak'].min(axis=1)
    K_star = d['K_list'][int(np.argmin(best))]
    print(f"  Exp 1  — Optimal squad count K* = {K_star}"
          f",  best peak = {best.min():.3f}")

print()

if 2 in all_results:
    d = all_results[2]
    for dist in exp2.DISTRIBUTIONS:
        best_idx = int(np.argmin(d['results'][dist]['final']))
        print(f"  Exp 2  — {dist:13s}: best at "
              f"{d['calib_counts'][best_idx]} calib nodes, "
              f"final = {min(d['results'][dist]['final']):.3f}")

if 3 in all_results:
    d = all_results[3]
    from core.metrics import find_cascade_threshold
    for dist in exp3.ATTACK_DISTRIBUTIONS:
        ct = find_cascade_threshold(d['results'][dist]['peak'],
                                    d['attack_intensities'])
        print(f"  Exp 3  — {dist:13s}: cascade threshold ≈ "
              f"{ct:.1f} seeds" if ct else
              f"  Exp 3  — {dist:13s}: threshold not reached")

if 4 in all_results:
    d = all_results[4]
    best_h  = d['severance_thresholds'][int(np.argmin(d['dyn']['peak']))]
    improve = d['static_res']['mean_peak'] - min(d['dyn']['peak'])
    print(f"  Exp 4  — Optimal severance threshold = {best_h:.2f}, "
          f"peak reduction = {improve:.3f}")

print()

if 8 in all_results and 'N' in all_results[8]:
    d = all_results[8]
    print(f"  Exp 8a — Peak infection at N=50: {d['peak'][0]:.3f}, "
          f"N=400: {d['peak'][-1]:.3f}  (scale-invariant: {'yes' if abs(d['peak'][-1]-d['peak'][0])<0.15 else 'no'})")

if 9 in all_results:
    d = all_results[9]
    sig = ('***' if d['p_delta_h'] < 0.001 else
           '**'  if d['p_delta_h'] < 0.01  else
           '*'   if d['p_delta_h'] < 0.05  else 'n.s.')
    print(f"  Exp 9  — Gaming gap r vs h:       r = {d['r_delta_h']:+.4f} {sig}")
    print(f"           Gaming gap r vs λ₂:      r = {d['r_delta_lam']:+.4f}")
    max_idx = int(np.argmax(d['delta_final']))
    print(f"           Max Δ at h={d['h'][max_idx]:.1f}: {d['delta_final'][max_idx]:+.3f}")

if 10 in all_results:
    d = all_results[10]
    for label in list(d['data'].keys())[:1]:
        sf_d = d['data'][label]['static_fixed']['disparity'][-1]
        al_d = d['data'][label]['adaptive_learning']['disparity'][-1]
        improve = sf_d - al_d
        print(f"  Exp 10 — ({label})")
        print(f"           Static/Fixed  disparity (round {d['T']}):   {sf_d:+.3f}")
        print(f"           Adaptive/Learning disparity (round {d['T']}): {al_d:+.3f}")
        print(f"           Adaptive+learning reduces disparity by: {improve:+.3f}")

total_time = time.time() - total_t0
print(f"\n  Total time : {total_time:.1f}s ({total_time/60:.1f} min)")
print(f"  Figures    → outputs/")
print(f"  Gephi      → outputs/gephi/")
print("="*70)
print("\n  Next steps:")
print("  1. Open Gephi → load .gexf files → apply ForceAtlas2")
print("  2. streamlit run app.py  (interactive demo)")
print("  3. Figures tell the story: Act 1 → Act 2 → Act 3")
