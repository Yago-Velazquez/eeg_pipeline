"""
scripts/update_threshold.py

Threshold comparison across three criteria:
  - argmax(F1)           — balanced precision/recall
  - argmax(2R+P)         — recall-weighted
  - Intuition (0.50)     — pragmatic middle ground

Writes all three to pipeline_config.yaml. The active threshold is
bcr.decision_threshold; the other two are stored for reference.

Usage:
    python scripts/update_threshold.py --label-strategy hard_threshold --model xgboost
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bad_channel_rejection import build_run_tag  # noqa: E402


def sweep_thresholds(y_true, y_prob):
    thresholds = np.round(np.arange(0.20, 0.81, 0.05), 2)
    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, pos_label=1, average="binary", zero_division=0
        )
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        rows.append(dict(
            threshold=float(t),
            precision=round(p, 4), recall=round(r, 4), f1=round(f, 4),
            recall_weighted=round(2 * r + p, 4),
            tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
        ))
    return rows


def print_top5(rows, sort_key, label):
    ranked = sorted(rows, key=lambda x: -x[sort_key])
    print(f"\nTop 5 by {label}")
    print(
        f"  {'Rank':>4}  {'Thresh':>7}  {'Precision':>10}  {'Recall':>8}  "
        f"{'F1':>8}  {'2R+P':>8}  {'TP':>5}  {'FP':>5}  {'FN':>5}"
    )
    for i, r in enumerate(ranked[:5], 1):
        marker = " <--" if i == 1 else ""
        print(
            f"  {i:>4}  {r['threshold']:>7.2f}  {r['precision']:>10.4f}  "
            f"{r['recall']:>8.4f}  {r['f1']:>8.4f}  {r['recall_weighted']:>8.4f}  "
            f"{r['tp']:>5}  {r['fp']:>5}  {r['fn']:>5}{marker}"
        )
    return ranked[0]["threshold"]


def main(tag: str):
    y_true = np.load(f"results/oof_y_true_{tag}.npy")
    y_prob = np.load(f"results/oof_y_prob_{tag}.npy")
    rows = sweep_thresholds(y_true, y_prob)

    print(f"\nBCR THRESHOLD COMPARISON — tag={tag}")

    best_f1 = print_top5(rows, "f1", "F1 (balanced)")
    best_rw = print_top5(rows, "recall_weighted", "2R+P (recall-weighted)")
    best_intuition = 0.50
    print_top5(rows, "f1", "Intuition anchor (0.50 neighbours, ranked by F1)")

    config_path = Path("configs/pipeline_config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}
    config.setdefault("bcr", {})
    config["bcr"]["decision_threshold"] = float(best_intuition)
    config["bcr"]["decision_threshold_argmax_f1"] = float(best_f1)
    config["bcr"]["decision_threshold_argmax_2rp"] = float(best_rw)
    config["bcr"]["active_tag"] = tag

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\nUpdated {config_path}:")
    print(f"  bcr.decision_threshold            = {best_intuition}")
    print(f"  bcr.decision_threshold_argmax_f1  = {best_f1}")
    print(f"  bcr.decision_threshold_argmax_2rp = {best_rw}")

    sweep_path = f"results/threshold_sweep_{tag}.json"
    with open(sweep_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"  Saved: {sweep_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-strategy", default="hard_threshold")
    parser.add_argument("--model", default="xgboost")
    parser.add_argument(
        "--use-engineered-features",
        action="store_true",
        help="DEPRECATED — update thresholds from the _fe OOF file.",
    )
    parser.add_argument(
        "--use-impedance-interactions",
        action="store_true",
        help="Update thresholds from the _impedance_ix OOF file.",
    )
    args = parser.parse_args()
    main(tag=build_run_tag(
        args.label_strategy,
        args.model,
        use_engineered_features=args.use_engineered_features,
        use_impedance_interactions=args.use_impedance_interactions,
    ))
