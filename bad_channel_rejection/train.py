"""
bad_channel_rejection/train.py

Cross-validation training for the BCR pipeline. Supports:
  - Multiple labelling strategies (hard_threshold, entropy_weights,
    dawid_skene, dawid_skene_soft) via --label-strategy. All four emit
    hard binary targets; they differ only in their per-sample weight
    scheme (uniform / entropy / DS-hard-confidence / DS-full-confidence).
  - Multiple model backends (xgboost, lightgbm, catboost) via --model
  - Sample-weight-aware fitting for any strategy that emits non-uniform
    weights

Outputs
-------
results/oof_y_true_{strategy}_{model}.npy
results/oof_y_prob_{strategy}_{model}.npy
results/bcr_model_{strategy}_{model}.{ext}
W&B run with per-fold + summary metrics

Run
---
    python -m bad_channel_rejection.train --label-strategy hard_threshold --model xgboost
    python -m bad_channel_rejection.train --label-strategy dawid_skene    --model lightgbm
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
from dotenv import load_dotenv
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

from . import build_run_tag
from .dataset import build_feature_matrix
from .features import FeaturePreprocessor, preprocess_fold
from .logging_config import setup_logging
from .models import MODEL_EXT, SUPPORTED_MODELS, create_model

load_dotenv()
os.environ.setdefault("WANDB_MODE", "offline")
warnings.filterwarnings("ignore", category=UserWarning)

logger = setup_logging(__name__)

DATA_PATH = Path("data/raw/Bad_channels_for_ML.csv")
MODEL_DIR = Path("results")
N_SPLITS = 5
RANDOM_SEED = 42


def load_prepared_data(
    label_strategy: str = "hard_threshold",
    bad_threshold: int = 2,
    use_engineered_features: bool = False,
) -> dict:
    """Load CSV -> (optionally scaled) feature matrix, labels, groups.

    When ``use_engineered_features`` is False (default), the existing
    pipeline runs: FeaturePreprocessor is fit once on the full dataset
    and ``X`` is the already-scaled numpy matrix.

    When True, preprocessing is deferred to per-fold inside run_cv
    (zero leakage for scaler + label-aware engineered transforms). The
    raw DataFrame is returned under ``X_df`` along with channel_labels
    for the fold encoder.
    """
    out = build_feature_matrix(
        DATA_PATH,
        bad_threshold=bad_threshold,
        label_strategy=label_strategy,
    )
    raw_df = pd.DataFrame(out["X"], columns=out["feature_cols"])

    result = {
        "y_hard": out["y_hard"],
        "y_soft": out["y_soft"],
        "sample_weights": out["sample_weights"],
        "groups": out["groups"],
        "channel_labels": out["channel_labels"],
        "scale_pos_weight": out["scale_pos_weight"],
        "label_strategy": out["label_strategy"],
        "use_engineered_features": use_engineered_features,
    }

    if use_engineered_features:
        result["X_df"] = raw_df
        result["X"] = None
        result["preprocessor"] = None
        logger.info(
            f"Data loaded (engineered mode — per-fold preprocessing): "
            f"X_df={raw_df.shape}  bad_rate={out['y_hard'].mean():.4f}  "
            f"strategy={out['label_strategy']}  "
            f"spw={out['scale_pos_weight']:.2f}"
        )
    else:
        preprocessor = FeaturePreprocessor()
        X = preprocessor.fit_transform(raw_df)
        assert np.isnan(X).sum() == 0, "NaNs found in feature matrix"
        assert np.isinf(X).sum() == 0, "Infs found in feature matrix"
        result["X"] = X
        result["X_df"] = raw_df
        result["preprocessor"] = preprocessor
        logger.info(
            f"Data loaded: X={X.shape}  bad_rate={out['y_hard'].mean():.4f}  "
            f"strategy={out['label_strategy']}  "
            f"spw={out['scale_pos_weight']:.2f}"
        )

    return result


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """All BCR metrics at default threshold 0.5. y_true must be binary."""
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "auprc": float(average_precision_score(y_true, y_prob)),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def run_cv(
    X: np.ndarray | None,
    y: np.ndarray,
    groups: np.ndarray,
    spw: float,
    model_name: str,
    sample_weights: np.ndarray | None = None,
    tag: str = "default",
    X_df: pd.DataFrame | None = None,
    channel_labels: np.ndarray | None = None,
    use_engineered_features: bool = False,
    engineering_kwargs: dict | None = None,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """GroupKFold(5) CV. Sample weights are sliced per fold (train only).

    Two modes:
      - Pre-scaled (use_engineered_features=False): ``X`` is the
        already-transformed numpy matrix; folds slice it directly.
      - Per-fold (use_engineered_features=True): ``X_df`` is the raw
        DataFrame and ``channel_labels`` is the string array; both
        FeaturePreprocessor and FeatureEngineeringPipeline are fit
        per fold on the training split only.
    """
    if use_engineered_features:
        assert X_df is not None, "X_df required when use_engineered_features=True"
        assert channel_labels is not None, (
            "channel_labels required when use_engineered_features=True"
        )
        split_X = X_df
        n_rows = len(X_df)
    else:
        assert X is not None, "X required when use_engineered_features=False"
        split_X = X
        n_rows = len(X)

    cv = GroupKFold(n_splits=N_SPLITS)
    fold_metrics = []
    oof_y_true = np.empty(n_rows, dtype=int)
    oof_y_prob = np.empty(n_rows, dtype=float)

    for fold, (tr_idx, va_idx) in enumerate(
        cv.split(split_X, y, groups), start=1
    ):
        y_tr, y_va = y[tr_idx], y[va_idx]

        tr_subjects = set(groups[tr_idx])
        va_subjects = set(groups[va_idx])
        assert tr_subjects.isdisjoint(va_subjects), (
            f"Subject leakage in fold {fold}"
        )

        if use_engineered_features:
            X_tr_df = X_df.iloc[tr_idx]
            X_va_df = X_df.iloc[va_idx]
            cl_tr = channel_labels[tr_idx]
            cl_va = channel_labels[va_idx]
            X_tr, X_va, _, _, _ = preprocess_fold(
                X_tr_df, X_va_df, y_tr,
                channel_labels_tr=cl_tr,
                channel_labels_va=cl_va,
                use_engineered_features=True,
                engineering_kwargs=engineering_kwargs,
            )
        else:
            X_tr, X_va = X[tr_idx], X[va_idx]

        w_tr = sample_weights[tr_idx] if sample_weights is not None else None

        model = create_model(model_name, scale_pos_weight=spw)
        model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=(X_va, y_va))

        y_prob = model.predict_proba(X_va)[:, 1]
        oof_y_true[va_idx] = y_va
        oof_y_prob[va_idx] = y_prob

        m = compute_metrics(y_va, y_prob)
        m["fold"] = fold
        m["best_iter"] = model.best_iteration
        m["val_bad_rate"] = float(y_va.mean())
        m["val_subjects"] = len(va_subjects)
        fold_metrics.append(m)

        logger.info(
            f"[{tag}] Fold {fold}/5 — AUPRC={m['auprc']:.4f}  "
            f"AUROC={m['auroc']:.4f}  F1={m['f1']:.3f}  "
            f"best_iter={m['best_iter']}  bad_rate={m['val_bad_rate']:.4f}"
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    oof_true_path = MODEL_DIR / f"oof_y_true_{tag}.npy"
    oof_prob_path = MODEL_DIR / f"oof_y_prob_{tag}.npy"
    np.save(str(oof_true_path), oof_y_true)
    np.save(str(oof_prob_path), oof_y_prob)
    logger.info(f"OOF predictions saved -> {oof_true_path}, {oof_prob_path}")

    return fold_metrics, oof_y_true, oof_y_prob


def summarise_folds(fold_metrics: list[dict]) -> dict:
    df = pd.DataFrame(fold_metrics)
    summary = {}
    for col in ["auprc", "auroc", "f1", "precision", "recall"]:
        summary[f"mean_{col}"] = float(df[col].mean())
        summary[f"std_{col}"] = float(df[col].std())
    summary["mean_best_iter"] = float(df["best_iter"].mean())

    logger.info("=" * 60)
    logger.info("CV Summary (mean ± std across 5 folds)")
    logger.info("=" * 60)
    logger.info(
        f"AUPRC :  {summary['mean_auprc']:.4f} ± {summary['std_auprc']:.4f}"
    )
    logger.info(
        f"AUROC :  {summary['mean_auroc']:.4f} ± {summary['std_auroc']:.4f}"
    )
    logger.info(
        f"F1    :  {summary['mean_f1']:.4f} ± {summary['std_f1']:.4f}"
    )
    logger.info(f"Avg best_iter: {summary['mean_best_iter']:.0f}")
    logger.info(f"Random AUPRC baseline: 0.039")
    logger.info(f"Lift over random: {summary['mean_auprc'] / 0.039:.1f}x")
    logger.info("=" * 60)
    return summary


def train_final_model(
    X: np.ndarray | None,
    y: np.ndarray,
    spw: float,
    model_name: str,
    best_n_estimators: int,
    tag: str,
    sample_weights: np.ndarray | None = None,
    X_df: pd.DataFrame | None = None,
    channel_labels: np.ndarray | None = None,
    use_engineered_features: bool = False,
    engineering_kwargs: dict | None = None,
):
    """Train on full data using CV best_iter, save to disk.

    In engineered mode, FeaturePreprocessor + FeatureEngineeringPipeline
    are refit on the full training set before the model is trained.
    """
    if use_engineered_features:
        assert X_df is not None and channel_labels is not None
        X, _, _, _, _ = preprocess_fold(
            X_df, X_df, y,
            channel_labels_tr=channel_labels,
            channel_labels_va=channel_labels,
            use_engineered_features=True,
            engineering_kwargs=engineering_kwargs,
        )

    overrides = (
        {"n_estimators": best_n_estimators}
        if model_name != "catboost"
        else {"iterations": best_n_estimators}
    )
    model = create_model(model_name, scale_pos_weight=spw, **overrides)
    model.fit(X, y, sample_weight=sample_weights, eval_set=None)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"bcr_model_{tag}.{MODEL_EXT[model_name]}"
    model.save(model_path)
    logger.info(f"Final model saved -> {model_path}")

    loaded = create_model(model_name, scale_pos_weight=spw).load(model_path)
    sample_probs = loaded.predict_proba(X[:5])[:, 1]
    assert len(sample_probs) == 5
    logger.info(f"Load smoke test OK — sample probs: {sample_probs.round(3)}")
    return model, model_path


def train(
    label_strategy: str = "hard_threshold",
    model_name: str = "xgboost",
    bad_threshold: int = 2,
    use_engineered_features: bool = False,
    use_impedance_interactions: bool = False,
    engineering_kwargs: dict | None = None,
):
    """Full training run: load -> CV -> W&B log -> final model save.

    When ``use_impedance_interactions`` is True, the per-fold pipeline
    enables only ``ImpedanceInteractionFeatures``; ChannelBadRateEncoder
    and SpatialFeaturePruner are disabled. This takes precedence over
    ``use_engineered_features``.
    """
    if use_impedance_interactions:
        engineering_kwargs = {
            "use_channel_bad_rate": False,
            "use_spatial_pruner": False,
            "use_impedance_interactions": True,
        }
        per_fold_mode = True
    else:
        per_fold_mode = use_engineered_features

    tag = build_run_tag(
        label_strategy,
        model_name,
        use_engineered_features=use_engineered_features,
        use_impedance_interactions=use_impedance_interactions,
    )
    logger.info(f"Starting run: {tag}")

    wandb_tags = ["bcr", model_name, label_strategy]
    if use_impedance_interactions:
        wandb_tags.append("impedance_interactions")
    elif use_engineered_features:
        wandb_tags.append("engineered_features")

    run = wandb.init(
        project="eeg-bcr",
        name=f"bcr_{tag}",
        tags=wandb_tags,
        config={
            "label_strategy": label_strategy,
            "model_name": model_name,
            "bad_threshold": bad_threshold,
            "n_splits": N_SPLITS,
            "random_seed": RANDOM_SEED,
            "use_engineered_features": use_engineered_features,
            "use_impedance_interactions": use_impedance_interactions,
            "engineering_kwargs": engineering_kwargs or {},
        },
        settings=wandb.Settings(init_timeout=120),
    )

    data = load_prepared_data(
        label_strategy, bad_threshold,
        use_engineered_features=per_fold_mode,
    )
    n_rows = len(data["y_hard"])
    n_features = (
        data["X"].shape[1] if data["X"] is not None else data["X_df"].shape[1]
    )
    wandb.config.update({
        "scale_pos_weight": data["scale_pos_weight"],
        "n_features_pre_engineering": n_features,
        "n_rows": n_rows,
        "bad_rate": float(data["y_hard"].mean()),
    })

    weights = (
        data["sample_weights"]
        if not np.allclose(data["sample_weights"], 1.0)
        else None
    )
    if weights is not None:
        logger.info(
            f"Sample weights active — mean={weights.mean():.4f}  "
            f"nonzero={(weights > 0).sum()}/{len(weights)}"
        )

    fold_metrics, oof_y_true, oof_y_prob = run_cv(
        data["X"],
        data["y_hard"],
        data["groups"],
        data["scale_pos_weight"],
        model_name,
        sample_weights=weights,
        tag=tag,
        X_df=data["X_df"],
        channel_labels=data["channel_labels"],
        use_engineered_features=per_fold_mode,
        engineering_kwargs=engineering_kwargs,
    )

    for m in fold_metrics:
        wandb.log({f"fold_{m['fold']}/{k}": v for k, v in m.items() if k != "fold"})

    summary = summarise_folds(fold_metrics)
    wandb.log(summary)

    best_n = max(1, int(round(summary["mean_best_iter"])))
    train_final_model(
        data["X"],
        data["y_hard"],
        data["scale_pos_weight"],
        model_name,
        best_n,
        tag,
        sample_weights=weights,
        X_df=data["X_df"],
        channel_labels=data["channel_labels"],
        use_engineered_features=per_fold_mode,
        engineering_kwargs=engineering_kwargs,
    )

    wandb.log({"final_model_n_estimators": best_n})
    run.finish()
    logger.info(f"Run complete: {tag}")
    return fold_metrics, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BCR model")
    parser.add_argument(
        "--label-strategy",
        default="hard_threshold",
        choices=[
            "hard_threshold",
            "entropy_weights",
            "dawid_skene",
            "dawid_skene_soft",
        ],
    )
    parser.add_argument(
        "--model", default="xgboost", choices=list(SUPPORTED_MODELS)
    )
    parser.add_argument("--threshold", type=int, default=2)
    parser.add_argument(
        "--use-engineered-features",
        action="store_true",
        help=(
            "DEPRECATED for this project. Enables all three label-aware "
            "transforms (ChannelBadRateEncoder, SpatialFeaturePruner, "
            "ImpedanceInteractionFeatures) fit per CV fold. The combined "
            "condition underperformed baseline in ablation; prefer "
            "--use-impedance-interactions."
        ),
    )
    parser.add_argument(
        "--use-impedance-interactions",
        action="store_true",
        help=(
            "Enable ONLY ImpedanceInteractionFeatures (fit per CV fold). "
            "ChannelBadRateEncoder and SpatialFeaturePruner stay off. "
            "This is the only engineered condition that beats baseline. "
            "Mutually exclusive with --use-engineered-features."
        ),
    )
    args = parser.parse_args()
    train(
        label_strategy=args.label_strategy,
        model_name=args.model,
        bad_threshold=args.threshold,
        use_engineered_features=args.use_engineered_features,
        use_impedance_interactions=args.use_impedance_interactions,
    )
