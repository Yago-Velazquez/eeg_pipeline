"""
bad_channel_rejection/feature_ablation.py

Feature-group ablation study for BCR.

Runs 6 conditions (A–F) via GroupKFold(5) CV:
    A  all features               (baseline)
    B  no impedance features
    C  no frequency features
    D  no spatial (correlation) features
    E  no decomposition features
    F  impedance_missing only

Uses the winning label strategy from the two-stage ablation (default:
hard_threshold with XGBoost). To run with the DS winner, pass
--label-strategy dawid_skene.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
from dotenv import load_dotenv
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from .dataset import build_feature_matrix
from .features import FeaturePreprocessor, preprocess_fold
from .logging_config import setup_logging
from .models import create_model

load_dotenv()
os.environ.setdefault("WANDB_MODE", "offline")

logger = setup_logging(__name__)

CSV_PATH = "data/raw/Bad_channels_for_ML.csv"
RESULTS_DIR = Path("results")
OUT_MD = RESULTS_DIR / "bcr_feature_ablation.md"
OUT_MD_ENGINEERED = RESULTS_DIR / "bcr_engineered_ablation.md"

GROUP_PATTERNS = {
    "impedance": ["impedance_missing"],
    "frequency": [
        " Median frequency",
        " First quartile frequency",
        " Third quartile frequency",
    ],
    "spatial": [
        " Correllation with neighbors",
        " Correllation with second-degree",
        " Global correlation",
    ],
    "decomposition": [
        " PCA", " ICA", " residuals", " reconstruction", " Kurtosis",
        " Low gamma", " independence", " unmixing",
    ],
}


def get_group_cols(feature_cols: list[str], patterns: list[str]) -> list[str]:
    return [c for c in feature_cols if any(p in c for p in patterns)]


def run_condition(
    X_df: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    scale_pos_weight: float,
    keep_cols: list[str] | None,
    condition_name: str,
    model_name: str,
    sample_weights: np.ndarray | None = None,
) -> dict:
    if keep_cols is not None:
        X_df = X_df[keep_cols]

    gkf = GroupKFold(n_splits=5)
    fold_auprc, fold_auroc = [], []

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_df, y, groups)):
        X_tr, X_val = X_df.iloc[tr_idx], X_df.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        prep = FeaturePreprocessor()
        X_tr_np = prep.fit_transform(X_tr)
        X_val_np = prep.transform(X_val)

        if X_tr_np.shape[1] == 0:
            logger.warning(f"  fold {fold}: 0 features after preprocessing")
            continue

        w_tr = sample_weights[tr_idx] if sample_weights is not None else None
        model = create_model(model_name, scale_pos_weight=scale_pos_weight)
        model.fit(X_tr_np, y_tr, sample_weight=w_tr, eval_set=(X_val_np, y_val))

        y_prob = model.predict_proba(X_val_np)[:, 1]
        fold_auprc.append(average_precision_score(y_val, y_prob))
        fold_auroc.append(roc_auc_score(y_val, y_prob))

    result = {
        "condition": condition_name,
        "n_features": len(keep_cols) if keep_cols is not None else X_df.shape[1],
        "auprc_mean": float(np.mean(fold_auprc)),
        "auprc_std": float(np.std(fold_auprc)),
        "auroc_mean": float(np.mean(fold_auroc)),
        "auroc_std": float(np.std(fold_auroc)),
    }
    logger.info(
        f"[{condition_name}] AUPRC={result['auprc_mean']:.4f} ± "
        f"{result['auprc_std']:.4f}  AUROC={result['auroc_mean']:.4f}"
    )
    return result


def main(label_strategy: str = "hard_threshold", model_name: str = "xgboost"):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading feature matrix...")
    out = build_feature_matrix(CSV_PATH, label_strategy=label_strategy)
    X_df = pd.DataFrame(out["X"], columns=out["feature_cols"])
    y = out["y_hard"]
    groups = out["groups"]
    spw = out["scale_pos_weight"]
    weights = (
        out["sample_weights"]
        if not np.allclose(out["sample_weights"], 1.0)
        else None
    )

    group_cols = {
        gname: get_group_cols(out["feature_cols"], patterns)
        for gname, patterns in GROUP_PATTERNS.items()
    }
    for gname, cols in group_cols.items():
        logger.info(f"  Group [{gname}]: {len(cols)} features")

    conditions = [
        ("A_all_features", None),
        ("B_no_impedance",
            [c for c in out["feature_cols"] if c not in group_cols["impedance"]]),
        ("C_no_frequency",
            [c for c in out["feature_cols"] if c not in group_cols["frequency"]]),
        ("D_no_spatial",
            [c for c in out["feature_cols"] if c not in group_cols["spatial"]]),
        ("E_no_decomposition",
            [c for c in out["feature_cols"] if c not in group_cols["decomposition"]]),
        ("F_impedance_only", ["impedance_missing"]),
    ]

    wandb.init(
        project="eeg-bcr",
        name=f"feature_ablation_{label_strategy}_{model_name}",
        config={
            "experiment": "feature_group_ablation",
            "label_strategy": label_strategy,
            "model_name": model_name,
        },
        settings=wandb.Settings(init_timeout=120),
    )

    results = []
    t0 = time.time()
    for cname, keep_cols in conditions:
        logger.info(f"\nRunning condition: {cname}")
        r = run_condition(
            X_df, y, groups, spw, keep_cols, cname, model_name, weights
        )
        results.append(r)
        wandb.log({
            f"{cname}/auprc": r["auprc_mean"],
            f"{cname}/auroc": r["auroc_mean"],
            f"{cname}/n_features": r["n_features"],
        })

    logger.info(f"\nTotal time: {(time.time()-t0)/60:.1f} min")
    wandb.finish()

    df_r = pd.DataFrame(results)
    baseline_auprc = df_r.loc[
        df_r["condition"] == "A_all_features", "auprc_mean"
    ].values[0]
    df_r["delta_auprc"] = df_r["auprc_mean"] - baseline_auprc
    df_r["pct_change"] = (df_r["delta_auprc"] / baseline_auprc * 100).round(1)

    _write_report(df_r, OUT_MD)
    logger.info(f"Report written -> {OUT_MD}")


ENGINEERED_CONDITIONS: list[tuple[str, dict]] = [
    ("E_baseline", {}),
    (
        "E_bad_rate_only",
        {
            "use_channel_bad_rate": True,
            "use_spatial_pruner": False,
            "use_impedance_interactions": False,
        },
    ),
    (
        "E_spatial_pruner_only",
        {
            "use_channel_bad_rate": False,
            "use_spatial_pruner": True,
            "use_impedance_interactions": False,
        },
    ),
    (
        "E_impedance_interactions_only",
        {
            "use_channel_bad_rate": False,
            "use_spatial_pruner": False,
            "use_impedance_interactions": True,
        },
    ),
    (
        "E_all_three",
        {
            "use_channel_bad_rate": True,
            "use_spatial_pruner": True,
            "use_impedance_interactions": True,
        },
    ),
]


def run_engineered_condition(
    X_df: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    channel_labels: np.ndarray,
    scale_pos_weight: float,
    condition_name: str,
    engineering_kwargs: dict,
    model_name: str,
    sample_weights: np.ndarray | None = None,
) -> dict:
    """Per-fold GroupKFold(5) with the engineered preprocessing chain."""
    use_eng = bool(engineering_kwargs)
    gkf = GroupKFold(n_splits=5)
    fold_auprc, fold_auroc = [], []
    last_names: list[str] = []

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_df, y, groups)):
        X_tr_df = X_df.iloc[tr_idx]
        X_va_df = X_df.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        cl_tr = channel_labels[tr_idx] if use_eng else None
        cl_va = channel_labels[val_idx] if use_eng else None

        X_tr_np, X_val_np, names, _, _ = preprocess_fold(
            X_tr_df, X_va_df, y_tr,
            channel_labels_tr=cl_tr,
            channel_labels_va=cl_va,
            use_engineered_features=use_eng,
            engineering_kwargs=engineering_kwargs if use_eng else None,
        )
        last_names = names

        w_tr = sample_weights[tr_idx] if sample_weights is not None else None
        model = create_model(model_name, scale_pos_weight=scale_pos_weight)
        model.fit(X_tr_np, y_tr, sample_weight=w_tr, eval_set=(X_val_np, y_val))

        y_prob = model.predict_proba(X_val_np)[:, 1]
        fold_auprc.append(average_precision_score(y_val, y_prob))
        fold_auroc.append(roc_auc_score(y_val, y_prob))

    result = {
        "condition": condition_name,
        "n_features": len(last_names),
        "auprc_mean": float(np.mean(fold_auprc)),
        "auprc_std": float(np.std(fold_auprc)),
        "auroc_mean": float(np.mean(fold_auroc)),
        "auroc_std": float(np.std(fold_auroc)),
    }
    logger.info(
        f"[{condition_name}] AUPRC={result['auprc_mean']:.4f} ± "
        f"{result['auprc_std']:.4f}  AUROC={result['auroc_mean']:.4f}  "
        f"n_features={result['n_features']}"
    )
    return result


def main_engineered(
    label_strategy: str = "dawid_skene", model_name: str = "lightgbm"
):
    """5-condition ablation of the engineered transform chain."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Loading feature matrix (engineered ablation)...")
    out = build_feature_matrix(CSV_PATH, label_strategy=label_strategy)
    X_df = pd.DataFrame(out["X"], columns=out["feature_cols"])
    y = out["y_hard"]
    groups = out["groups"]
    channel_labels = out["channel_labels"]
    spw = out["scale_pos_weight"]
    weights = (
        out["sample_weights"]
        if not np.allclose(out["sample_weights"], 1.0)
        else None
    )

    wandb.init(
        project="eeg-bcr",
        name=f"engineered_ablation_{label_strategy}_{model_name}",
        config={
            "experiment": "engineered_feature_ablation",
            "label_strategy": label_strategy,
            "model_name": model_name,
        },
        settings=wandb.Settings(init_timeout=120),
    )

    results = []
    t0 = time.time()
    for cname, eng_kwargs in ENGINEERED_CONDITIONS:
        logger.info(f"\nRunning condition: {cname}  kwargs={eng_kwargs}")
        r = run_engineered_condition(
            X_df, y, groups, channel_labels, spw,
            cname, eng_kwargs, model_name, weights,
        )
        results.append(r)
        wandb.log({
            f"{cname}/auprc": r["auprc_mean"],
            f"{cname}/auroc": r["auroc_mean"],
            f"{cname}/n_features": r["n_features"],
        })

    logger.info(f"\nTotal time: {(time.time()-t0)/60:.1f} min")
    wandb.finish()

    df_r = pd.DataFrame(results)
    baseline_auprc = df_r.loc[
        df_r["condition"] == "E_baseline", "auprc_mean"
    ].values[0]
    df_r["delta_auprc"] = df_r["auprc_mean"] - baseline_auprc
    df_r["pct_change"] = (df_r["delta_auprc"] / baseline_auprc * 100).round(1)

    _write_engineered_report(df_r, OUT_MD_ENGINEERED)
    logger.info(f"Report written -> {OUT_MD_ENGINEERED}")


def _write_engineered_report(df: pd.DataFrame, path: Path):
    baseline = df.loc[df["condition"] == "E_baseline"].iloc[0]
    lines = [
        "# BCR Engineered-Feature Ablation",
        "",
        "Five-condition study isolating the contribution of each "
        "label-aware transform. All transforms are fit per-fold "
        "(GroupKFold(5) by subject) on training data only.",
        "",
        "| Condition | # Features | AUPRC (mean±std) | AUROC | ΔAUPRC | % change |",
        "|---|---|---|---|---|---|",
    ]
    for _, row in df.iterrows():
        dstr = (
            f"{row['delta_auprc']:+.4f}"
            if row["condition"] != "E_baseline" else "—"
        )
        pstr = (
            f"{row['pct_change']:+.1f}%"
            if row["condition"] != "E_baseline" else "—"
        )
        lines.append(
            f"| {row['condition']} | {row['n_features']} "
            f"| {row['auprc_mean']:.4f} ± {row['auprc_std']:.4f} "
            f"| {row['auroc_mean']:.4f} | {dstr} | {pstr} |"
        )
    lines += [
        "",
        "## Baseline",
        f"- E_baseline (FeaturePreprocessor only, fit per-fold): "
        f"AUPRC = {baseline['auprc_mean']:.4f} ± "
        f"{baseline['auprc_std']:.4f}",
        f"- Random baseline: 0.039",
        "",
        "## Notes",
        "- Ceiling bounded by Krippendorff α = 0.211 on the labels.",
        "- All transforms fit on training-fold labels only — no leakage.",
    ]
    path.write_text("\n".join(lines))


def _write_report(df: pd.DataFrame, path: Path):
    baseline = df.loc[df["condition"] == "A_all_features"].iloc[0]
    lines = [
        "# BCR Feature Group Ablation", "",
        "| Condition | # Features | AUPRC (mean±std) | AUROC | ΔAUPRC | % change |",
        "|---|---|---|---|---|---|",
    ]
    for _, row in df.iterrows():
        dstr = (
            f"{row['delta_auprc']:+.4f}"
            if row["condition"] != "A_all_features" else "—"
        )
        pstr = (
            f"{row['pct_change']:+.1f}%"
            if row["condition"] != "A_all_features" else "—"
        )
        lines.append(
            f"| {row['condition']} | {row['n_features']} "
            f"| {row['auprc_mean']:.4f} ± {row['auprc_std']:.4f} "
            f"| {row['auroc_mean']:.4f} | {dstr} | {pstr} |"
        )
    lines += [
        "", "## Baseline",
        f"- All features: AUPRC = {baseline['auprc_mean']:.4f} ± "
        f"{baseline['auprc_std']:.4f}",
        f"- Random baseline: 0.039",
    ]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-strategy", default="hard_threshold")
    parser.add_argument("--model", default="xgboost")
    parser.add_argument(
        "--mode",
        default="group",
        choices=["group", "engineered"],
        help=(
            "'group': existing A-F feature-group ablation (default). "
            "'engineered': 5-condition engineered-transform ablation."
        ),
    )
    args = parser.parse_args()
    if args.mode == "engineered":
        main_engineered(
            label_strategy=args.label_strategy, model_name=args.model
        )
    else:
        main(label_strategy=args.label_strategy, model_name=args.model)
