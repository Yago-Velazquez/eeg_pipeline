"""
scripts/fn_analysis.py

False negative analysis on OOF predictions.

Usage:
    python scripts/fn_analysis.py --label-strategy hard_threshold --model xgboost
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bad_channel_rejection import build_run_tag  # noqa: E402


def main(tag: str):
    y_true = np.load(f"results/oof_y_true_{tag}.npy")
    y_prob = np.load(f"results/oof_y_prob_{tag}.npy")

    with open("configs/pipeline_config.yaml") as f:
        config = yaml.safe_load(f)
    threshold = config["bcr"]["decision_threshold"]

    y_pred = (y_prob >= threshold).astype(int)

    df = pd.read_csv("data/raw/Bad_channels_for_ML.csv")
    df["y_true"] = y_true
    df["y_prob"] = y_prob
    df["y_pred"] = y_pred

    fn_mask = (df["y_true"] == 1) & (df["y_pred"] == 0)
    tp_mask = (df["y_true"] == 1) & (df["y_pred"] == 1)

    n_bad = int((df["y_true"] == 1).sum())
    print(f"tag:                   {tag}")
    print(f"threshold:             {threshold}")
    print(f"Total bad channels:    {n_bad}")
    print(f"True positives:        {tp_mask.sum()} ({tp_mask.sum()/n_bad*100:.1f}%)")
    print(f"False negatives:       {fn_mask.sum()} ({fn_mask.sum()/n_bad*100:.1f}%)")
    print()

    print("Top 10 channels by FN count:")
    fn_by_channel = (
        df[fn_mask].groupby("Channel labels").size().sort_values(ascending=False).head(10)
    )
    total_by_channel = df[df["y_true"] == 1].groupby("Channel labels").size()
    print(
        pd.DataFrame({
            "FN_count": fn_by_channel,
            "total_bad": total_by_channel,
            "FN_rate": (fn_by_channel / total_by_channel).round(3),
        }).to_string()
    )
    print()

    df["visit"] = df["Session"].str.split("-").str[0].astype(int)
    print("FN rate by visit:")
    for v in sorted(df["visit"].unique()):
        mask_v = df["visit"] == v
        bad_v = (df.loc[mask_v, "y_true"] == 1).sum()
        fn_v = ((df["visit"] == v) & fn_mask).sum()
        print(f"  Visit {v}: FN={fn_v}/{bad_v}  rate={fn_v/max(bad_v,1):.3f}")
    print()

    fn_df = (
        df[fn_mask][["Channel labels", "Session", "y_prob"]]
        .sort_values("y_prob", ascending=False)
    )
    print("Top 10 hardest FNs:")
    print(fn_df.head(10).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-strategy", default="hard_threshold")
    parser.add_argument("--model", default="xgboost")
    parser.add_argument(
        "--use-engineered-features",
        action="store_true",
        help="DEPRECATED — analyse the _fe OOF file.",
    )
    parser.add_argument(
        "--use-impedance-interactions",
        action="store_true",
        help="Analyse the _impedance_ix OOF file.",
    )
    args = parser.parse_args()
    main(tag=build_run_tag(
        args.label_strategy,
        args.model,
        use_engineered_features=args.use_engineered_features,
        use_impedance_interactions=args.use_impedance_interactions,
    ))
