"""
bad_channel_rejection/site_generalisation.py

Leave-one-visit-out evaluation. Trains on 3 visits, tests on the 4th.
Rotates through all 4 visits.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from . import build_run_tag
from .dataset import (
    add_missingness_flags,
    impute_and_encode_channels,
    load_bcr_data,
)
from .features import FeaturePreprocessor, preprocess_fold
from .label_quality import build_label_artifacts
from .logging_config import setup_logging
from .models import create_model

warnings.filterwarnings("ignore")

logger = setup_logging(__name__)

BCR_CSV = "data/raw/Bad_channels_for_ML.csv"
BAD_THRESHOLD = 2
REFERENCE_AUPRC = 0.518


def run_leave_one_visit_out(
    label_strategy: str = "hard_threshold",
    model_name: str = "xgboost",
    use_engineered_features: bool = False,
    use_impedance_interactions: bool = False,
):
    if use_impedance_interactions:
        engineering_kwargs = {
            "use_channel_bad_rate": False,
            "use_spatial_pruner": False,
            "use_impedance_interactions": True,
        }
        per_fold_mode = True
    else:
        engineering_kwargs = None
        per_fold_mode = use_engineered_features

    tag = build_run_tag(
        label_strategy,
        model_name,
        use_engineered_features=use_engineered_features,
        use_impedance_interactions=use_impedance_interactions,
    )

    df = load_bcr_data(BCR_CSV)
    df = add_missingness_flags(df)

    with open("configs/feature_cols.json") as f:
        feature_cols = json.load(f)

    df, _ = impute_and_encode_channels(df, feature_cols)

    artifacts = build_label_artifacts(df, strategy=label_strategy)
    y = artifacts.y_hard
    channel_labels = df["Channel labels"].astype(str).to_numpy()
    weights = (
        artifacts.sample_weights
        if not np.allclose(artifacts.sample_weights, 1.0)
        else None
    )

    visits = sorted(df["visit"].unique())
    logger.info(f"Visits found: {visits}")
    for v in visits:
        mask = df["visit"].values == v
        n_subj = df.loc[mask, "subject_id"].nunique()
        bad_r = y[mask].mean()
        logger.info(
            f"  Visit {v}: {n_subj} subjects, {mask.sum()} rows, "
            f"bad_rate={bad_r:.3f}"
        )

    results = []
    for test_visit in visits:
        train_mask = (df["visit"] != test_visit).values
        test_mask = (df["visit"] == test_visit).values

        X_train_df = df.loc[train_mask, feature_cols]
        X_test_df = df.loc[test_mask, feature_cols]
        y_train = y[train_mask]
        y_test = y[test_mask]
        w_train = weights[train_mask] if weights is not None else None

        spw_train = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

        if per_fold_mode:
            X_train, X_test, _, _, _ = preprocess_fold(
                X_train_df, X_test_df, y_train,
                channel_labels_tr=channel_labels[train_mask],
                channel_labels_va=channel_labels[test_mask],
                use_engineered_features=True,
                engineering_kwargs=engineering_kwargs,
            )
        else:
            prep = FeaturePreprocessor()
            X_train = prep.fit_transform(X_train_df)
            X_test = prep.transform(X_test_df)

        model = create_model(model_name, scale_pos_weight=spw_train)
        model.fit(X_train, y_train, sample_weight=w_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        auprc = average_precision_score(y_test, y_prob)
        auroc = roc_auc_score(y_test, y_prob)

        results.append({
            "test_visit": int(test_visit),
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "bad_rate_test": round(float(y_test.mean()), 4),
            "auprc": round(auprc, 4),
            "auroc": round(auroc, 4),
        })
        logger.info(
            f"Visit {test_visit} held out: AUPRC={auprc:.4f}  "
            f"AUROC={auroc:.4f}  (bad_rate={y_test.mean():.3f})"
        )

    auprc_vals = [r["auprc"] for r in results]
    mean_auprc = float(np.mean(auprc_vals))
    std_auprc = float(np.std(auprc_vals))
    gap = REFERENCE_AUPRC - mean_auprc

    logger.info(
        f"\nMean AUPRC (LOVO): {mean_auprc:.4f} ± {std_auprc:.4f}"
    )
    logger.info(
        f"GroupKFold reference:  {REFERENCE_AUPRC:.4f}  "
        f"(subject-stratified)"
    )
    logger.info(f"Generalisation gap: {gap:+.4f}")

    out = {
        "label_strategy": label_strategy,
        "model_name": model_name,
        "use_engineered_features": use_engineered_features,
        "use_impedance_interactions": use_impedance_interactions,
        "tag": tag,
        "per_visit": results,
        "mean_auprc": round(mean_auprc, 4),
        "std_auprc": round(std_auprc, 4),
        "groupkfold_ref_auprc": REFERENCE_AUPRC,
        "generalisation_gap": round(gap, 4),
    }
    Path("results").mkdir(exist_ok=True)
    path = Path(f"results/site_generalisation_{tag}.json")
    path.write_text(json.dumps(out, indent=2))
    logger.info(f"Saved: {path}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-strategy", default="hard_threshold")
    parser.add_argument("--model", default="xgboost")
    parser.add_argument(
        "--use-engineered-features",
        action="store_true",
        help="DEPRECATED — run LOVO with all three engineered transforms.",
    )
    parser.add_argument(
        "--use-impedance-interactions",
        action="store_true",
        help="Run LOVO with ImpedanceInteractionFeatures only.",
    )
    args = parser.parse_args()
    run_leave_one_visit_out(
        args.label_strategy,
        args.model,
        use_engineered_features=args.use_engineered_features,
        use_impedance_interactions=args.use_impedance_interactions,
    )
