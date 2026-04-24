"""
scripts/verify_threshold.py

Sanity-check the active threshold from pipeline_config.yaml.

Usage:
    python scripts/verify_threshold.py --label-strategy hard_threshold --model xgboost
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bad_channel_rejection import build_run_tag  # noqa: E402


def main(tag: str):
    y_true = np.load(f"results/oof_y_true_{tag}.npy")
    y_prob = np.load(f"results/oof_y_prob_{tag}.npy")

    with open("configs/pipeline_config.yaml") as f:
        config = yaml.safe_load(f)
    threshold = config["bcr"]["decision_threshold"]
    print(f"Loaded threshold: {threshold}  (tag={tag})")

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"Recall:    {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"FN rate:   {fn_rate:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-strategy", default="dawid_skene")
    parser.add_argument("--model", default="lightgbm")
    parser.add_argument(
        "--use-engineered-features",
        action="store_true",
        help="DEPRECATED — verify threshold against the _fe OOF file.",
    )
    parser.add_argument(
        "--use-impedance-interactions",
        action="store_true",
        help="Verify threshold against the _impedance_ix OOF file.",
    )
    args = parser.parse_args()
    main(tag=build_run_tag(
        args.label_strategy,
        args.model,
        use_engineered_features=args.use_engineered_features,
        use_impedance_interactions=args.use_impedance_interactions,
    ))
