# Save as scripts/threshold_sweep.py and run:
# python scripts/threshold_sweep.py

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
import json

# Load OOF predictions (written by train.py on Day 10)
y_true = np.load("results/oof_y_true_thresh2.npy")
y_prob = np.load("results/oof_y_prob_thresh2.npy")

print(f"OOF AUPRC (reference): {average_precision_score(y_true, y_prob):.4f}")
print(f"Random baseline:        0.0390")
print()
print(f"{'Thresh':>8}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'TP':>6}  {'FP':>6}  {'FN':>6}")
print("-" * 68)

results = []
for t in np.arange(0.20, 0.81, 0.05):
    y_pred = (y_prob >= t).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, pos_label=1, average='binary', zero_division=0)
    tp = int(((y_pred==1)&(y_true==1)).sum())
    fp = int(((y_pred==1)&(y_true==0)).sum())
    fn = int(((y_pred==0)&(y_true==1)).sum())
    print(f"{t:8.2f}  {p:10.4f}  {r:8.4f}  {f:8.4f}  {tp:6d}  {fp:6d}  {fn:6d}")
    results.append(dict(threshold=round(t,2), precision=round(p,4), recall=round(r,4), f1=round(f,4), tp=tp, fp=fp, fn=fn))

# Identify argmax F1, argmax recall_weighted, and current threshold
best_f1_idx = max(range(len(results)), key=lambda i: results[i]['f1'])
best_rw_idx = max(range(len(results)), key=lambda i: 2*results[i]['recall'] + results[i]['precision'])
print()
print(f"argmax(F1):             threshold={results[best_f1_idx]['threshold']}  F1={results[best_f1_idx]['f1']}")
print(f"argmax(2R+P):           threshold={results[best_rw_idx]['threshold']}  Recall={results[best_rw_idx]['recall']}")
print(f"Current (Day 11):       threshold=0.604  F1=0.506")

# Save for use in Task 2
with open("results/threshold_sweep.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved: results/threshold_sweep.json")

