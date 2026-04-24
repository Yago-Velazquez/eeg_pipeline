"""
scripts/threshold_sweep.py

Threshold sweep on OOF predictions.

Usage:
    python scripts/threshold_sweep.py --label-strategy hard_threshold --model xgboost
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bad_channel_rejection import build_run_tag  # noqa: E402


def main(tag: str):
    y_true = np.load(f"results/oof_y_true_{tag}.npy")
    y_prob = np.load(f"results/oof_y_prob_{tag}.npy")

    print(f"tag: {tag}")
    print(f"OOF AUPRC (reference): {average_precision_score(y_true, y_prob):.4f}")
    print(f"Random baseline:        0.0390")
    print()
    print(
        f"{'Thresh':>8}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  "
        f"{'TP':>6}  {'FP':>6}  {'FN':>6}"
    )
    print("-" * 68)

    results = []
    for t in np.arange(0.20, 0.81, 0.05):
        y_pred = (y_prob >= t).astype(int)
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, pos_label=1, average="binary", zero_division=0
        )
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        print(
            f"{t:8.2f}  {p:10.4f}  {r:8.4f}  {f:8.4f}  "
            f"{tp:6d}  {fp:6d}  {fn:6d}"
        )
        results.append(dict(
            threshold=round(t, 2), precision=round(p, 4),
            recall=round(r, 4), f1=round(f, 4),
            tp=tp, fp=fp, fn=fn,
        ))

    best_f1 = max(results, key=lambda x: x["f1"])
    best_rw = max(results, key=lambda x: 2 * x["recall"] + x["precision"])
    print()
    print(f"argmax(F1):   threshold={best_f1['threshold']}  F1={best_f1['f1']}")
    print(
        f"argmax(2R+P): threshold={best_rw['threshold']}  "
        f"Recall={best_rw['recall']}"
    )

    out_path = f"results/threshold_sweep_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-strategy", default="hard_threshold")
    parser.add_argument("--model", default="xgboost")
    parser.add_argument(
        "--use-engineered-features",
        action="store_true",
        help="DEPRECATED — sweep thresholds on the _fe OOF file.",
    )
    parser.add_argument(
        "--use-impedance-interactions",
        action="store_true",
        help="Sweep thresholds on the _impedance_ix OOF file.",
    )
    args = parser.parse_args()
    main(tag=build_run_tag(
        args.label_strategy,
        args.model,
        use_engineered_features=args.use_engineered_features,
        use_impedance_interactions=args.use_impedance_interactions,
    ))
