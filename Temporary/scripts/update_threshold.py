"""
scripts/update_threshold.py

Threshold comparison across three criteria:
  - argmax(F1)           — balanced precision/recall
  - argmax(2R+P)         — recall-weighted (roadmap spec)
  - Intuition (0.50)     — pragmatic middle ground

For each criterion, shows the top 5 thresholds ranked by that metric,
then applies all three chosen thresholds to pipeline_config.yaml
(you pick which one to keep at the end).

Usage:
    python scripts/update_threshold.py
"""

import numpy as np
import json
import yaml
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# ── Load OOF predictions ──────────────────────────────────────────────
y_true = np.load("results/oof_y_true_thresh2.npy")
y_prob  = np.load("results/oof_y_prob_thresh2.npy")

# ── Build full sweep table ────────────────────────────────────────────
thresholds = np.round(np.arange(0.20, 0.81, 0.05), 2)
rows = []
for t in thresholds:
    y_pred = (y_prob >= t).astype(int)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, pos_label=1, average='binary', zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    rows.append(dict(
        threshold=t,
        precision=round(p, 4),
        recall=round(r, 4),
        f1=round(f, 4),
        recall_weighted=round(2*r + p, 4),   # 2R+P score
        tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
    ))

# ── Helper: print ranked top-5 table ─────────────────────────────────
def print_top5(rows, sort_key, label):
    ranked = sorted(rows, key=lambda x: -x[sort_key])
    print(f"\n{'─'*70}")
    print(f"  TOP 5 by {label}")
    print(f"{'─'*70}")
    print(f"  {'Rank':>4}  {'Thresh':>7}  {'Precision':>10}  {'Recall':>8}  "
          f"{'F1':>8}  {'2R+P':>8}  {'TP':>5}  {'FP':>5}  {'FN':>5}")
    print(f"  {'─'*4}  {'─'*7}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*8}  "
          f"{'─'*5}  {'─'*5}  {'─'*5}")
    for i, r in enumerate(ranked[:5], 1):
        marker = " ◀" if i == 1 else ""
        print(f"  {i:>4}  {r['threshold']:>7.2f}  {r['precision']:>10.4f}  "
              f"{r['recall']:>8.4f}  {r['f1']:>8.4f}  {r['recall_weighted']:>8.4f}  "
              f"{r['tp']:>5}  {r['fp']:>5}  {r['fn']:>5}{marker}")
    return ranked[0]['threshold']

# ── Helper: print single threshold detail ────────────────────────────
def print_threshold_detail(rows, t, label):
    row = next(r for r in rows if r['threshold'] == t)
    fn_rate = row['fn'] / (row['fn'] + row['tp'])
    print(f"\n  [{label}]  threshold = {t}")
    print(f"    Precision : {row['precision']:.4f}")
    print(f"    Recall    : {row['recall']:.4f}")
    print(f"    F1        : {row['f1']:.4f}")
    print(f"    2R+P      : {row['recall_weighted']:.4f}")
    print(f"    TP={row['tp']}  FP={row['fp']}  FN={row['fn']}  TN={row['tn']}")
    print(f"    FN rate   : {fn_rate:.3f}  ({row['fn']} bad channels missed)")
    return row

# ── Section 1: Top 5 per criterion ───────────────────────────────────
print("\n" + "═"*70)
print("  BCR THRESHOLD COMPARISON — Day 15")
print("═"*70)

best_f1    = print_top5(rows, 'f1',              'F1 (balanced)')
best_rw    = print_top5(rows, 'recall_weighted', '2R+P (recall-weighted)')

# Intuition: rank by "recall ≥ 0.55 AND precision ≥ 0.40" — pragmatic band
intuition_scored = [(abs(r['threshold'] - 0.50), r['threshold']) for r in rows]
best_intuition = 0.50  # anchored; shown in context of its neighbours
print_top5(rows, 'f1', 'Intuition anchor (0.50 and neighbours — ranked by F1)')

# ── Section 2: Side-by-side comparison of the three chosen thresholds ──
print(f"\n{'═'*70}")
print("  CHOSEN THRESHOLD COMPARISON")
print(f"{'═'*70}")

ref_row  = next(r for r in rows if r['threshold'] == 0.60)   # Day 11 reference
row_f1   = print_threshold_detail(rows, best_f1,        'argmax(F1)')
row_rw   = print_threshold_detail(rows, best_rw,        'argmax(2R+P)')
row_int  = print_threshold_detail(rows, best_intuition, 'Intuition (0.50)')

print(f"\n  {'─'*68}")
print(f"  Day 11 reference  threshold=0.604  "
      f"Recall={ref_row['recall']:.4f}  F1={ref_row['f1']:.4f}  "
      f"TP={ref_row['tp']}  FP={ref_row['fp']}  FN={ref_row['fn']}")

# ── Section 3: Delta table vs Day 11 baseline ─────────────────────────
print(f"\n{'═'*70}")
print("  DELTA vs DAY 11 BASELINE (threshold=0.604)")
print(f"{'═'*70}")
print(f"  {'Criterion':<22}  {'Thresh':>7}  {'ΔRecall':>9}  {'ΔFN':>7}  "
      f"{'ΔFP':>7}  {'ΔF1':>8}")
print(f"  {'─'*22}  {'─'*7}  {'─'*9}  {'─'*7}  {'─'*7}  {'─'*8}")

for row, label in [(row_f1, 'argmax(F1)'), (row_rw, 'argmax(2R+P)'), (row_int, 'Intuition (0.50)')]:
    d_recall = row['recall']    - ref_row['recall']
    d_fn     = row['fn']        - ref_row['fn']
    d_fp     = row['fp']        - ref_row['fp']
    d_f1     = row['f1']        - ref_row['f1']
    print(f"  {label:<22}  {row['threshold']:>7.2f}  "
          f"{d_recall:>+9.4f}  {d_fn:>+7}  {d_fp:>+7}  {d_f1:>+8.4f}")

# ── Section 4: Write all three to pipeline_config.yaml ────────────────
print(f"\n{'═'*70}")
print("  CONFIG UPDATE")
print(f"{'═'*70}")

config_path = Path("configs/pipeline_config.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

config['bcr']['decision_threshold']          = float(best_intuition)  # default: intuition
config['bcr']['decision_threshold_argmax_f1']  = float(best_f1)
config['bcr']['decision_threshold_argmax_2rp'] = float(best_rw)

with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print(f"\n  Updated configs/pipeline_config.yaml:")
print(f"    bcr.decision_threshold            = {best_intuition}  ← active (intuition)")
print(f"    bcr.decision_threshold_argmax_f1  = {best_f1}")
print(f"    bcr.decision_threshold_argmax_2rp = {best_rw}")
print(f"\n  To switch the active threshold, edit bcr.decision_threshold manually.")

# ── Save enriched sweep JSON ───────────────────────────────────────────
with open("results/threshold_sweep.json", "w") as f:
    json.dump(rows, f, indent=2)
print(f"  Saved: results/threshold_sweep.json  ({len(rows)} rows)")
print()
