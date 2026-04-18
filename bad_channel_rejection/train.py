"""bad_channel_rejection/train.py

XGBoost cross-validation training for BCR.

Run:
    python -m bad_channel_rejection.train --threshold 2
    python -m bad_channel_rejection.train --threshold 1
"""

import argparse
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import wandb
from dotenv import load_dotenv
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)

from bad_channel_rejection.dataset import build_feature_matrix
from bad_channel_rejection.features import FeaturePreprocessor

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

DATA_PATH   = Path("data/raw/Bad_channels_for_ML.csv")
MODEL_DIR   = Path("results")
N_SPLITS    = 5
RANDOM_SEED = 42

XGB_PARAMS = dict(
    n_estimators          = 500,
    max_depth             = 6,
    learning_rate         = 0.05,
    subsample             = 0.8,
    colsample_bytree      = 0.8,
    eval_metric           = "aucpr",
    early_stopping_rounds = 30,
    random_state          = RANDOM_SEED,
    tree_method           = "hist",   # fast on M2 CPU
    device                = "cpu",    # XGBoost doesn't support MPS
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_prepared_data(bad_threshold: int = 2):
    """Load BCR CSV → scaled feature matrix, labels, groups.

    Returns
    -------
    X            : np.ndarray  (18900, 138)
    y            : np.ndarray  (18900,)  binary int
    groups       : np.ndarray  (18900,)  subject_id for GroupKFold
    spw          : float       scale_pos_weight = n_good / n_bad
    preprocessor : FeaturePreprocessor  fitted on full dataset (for final model)
    """
    X_raw, y, groups, feature_cols, scale_pos_weight = build_feature_matrix(
        DATA_PATH,
        bad_threshold=bad_threshold
    )
    preprocessor = FeaturePreprocessor()
    X = preprocessor.fit_transform(pd.DataFrame(X_raw, columns=feature_cols))

    logger.info(
        f"Data loaded — X={X.shape}, bad_rate={y.mean():.4f} "
        f"({int((y==1).sum())} bad / {int((y==0).sum())} good), "
        f"scale_pos_weight={scale_pos_weight:.2f}"
    )
    assert np.isnan(X).sum() == 0, "NaNs found in feature matrix"
    assert np.isinf(X).sum() == 0, "Infs found in feature matrix"

    return X, y, groups, scale_pos_weight, preprocessor


# ── Per-fold metrics ───────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute all BCR metrics at default threshold 0.5."""
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "auprc":     float(average_precision_score(y_true, y_prob)),
        "auroc":     float(roc_auc_score(y_true, y_prob)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
    }


# ── Cross-validation loop ─────────────────────────────────────────────────────

def run_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
           spw: float) -> list[dict]:
    """5-fold GroupKFold CV. Returns list of per-fold metric dicts."""
    cv = GroupKFold(n_splits=N_SPLITS)
    fold_metrics = []

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y, groups), start=1):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        # Verify no subject leakage
        tr_subjects = set(groups[tr_idx])
        va_subjects = set(groups[va_idx])
        assert tr_subjects.isdisjoint(va_subjects), \
            f"Subject leakage detected in fold {fold}!"

        model = xgb.XGBClassifier(
            **XGB_PARAMS,
            scale_pos_weight=float(spw),
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )

        y_prob = model.predict_proba(X_va)[:, 1]
        m = compute_metrics(y_va, y_prob)
        m["fold"]         = fold
        m["best_iter"]    = int(model.best_iteration)
        m["val_bad_rate"] = float(y_va.mean())
        m["val_subjects"] = int(len(va_subjects))

        fold_metrics.append(m)
        logger.info(
            f"Fold {fold}/5 — AUPRC={m['auprc']:.4f} | AUROC={m['auroc']:.4f} | "
            f"F1={m['f1']:.3f} | best_iter={m['best_iter']} | "
            f"val_bad_rate={m['val_bad_rate']:.4f}"
        )

    return fold_metrics


# ── Summary ────────────────────────────────────────────────────────────────────

def summarise_folds(fold_metrics: list[dict]) -> dict:
    """Compute mean ± std across folds and log a summary table."""
    df = pd.DataFrame(fold_metrics)
    summary = {}
    for col in ["auprc", "auroc", "f1", "precision", "recall"]:
        summary[f"mean_{col}"] = float(df[col].mean())
        summary[f"std_{col}"]  = float(df[col].std())

    best_iters = df["best_iter"].tolist()
    summary["mean_best_iter"] = float(np.mean(best_iters))

    logger.info("\n" + "=" * 50)
    logger.info("CV Summary (mean ± std across 5 folds)")
    logger.info("=" * 50)
    logger.info(f"AUPRC :  {summary['mean_auprc']:.4f} ± {summary['std_auprc']:.4f}")
    logger.info(f"AUROC :  {summary['mean_auroc']:.4f} ± {summary['std_auroc']:.4f}")
    logger.info(f"F1    :  {summary['mean_f1']:.4f} ± {summary['std_f1']:.4f}")
    logger.info(f"Prec  :  {summary['mean_precision']:.4f} ± {summary['std_precision']:.4f}")
    logger.info(f"Recall:  {summary['mean_recall']:.4f} ± {summary['std_recall']:.4f}")
    logger.info(f"Avg best_iter: {summary['mean_best_iter']:.0f}")
    logger.info(f"Random AUPRC baseline: 0.039")
    logger.info(f"Lift over random: {summary['mean_auprc'] / 0.039:.1f}x")
    logger.info("=" * 50)

    return summary


# ── Final model ────────────────────────────────────────────────────────────────

def train_final_model(X: np.ndarray, y: np.ndarray, spw: float,
                      best_n_estimators: int, bad_threshold: int):
    """Train on full dataset using mean best_iter from CV. Save to disk."""
    # Remove early stopping — no eval_set for full-data training
    params = {k: v for k, v in XGB_PARAMS.items()
              if k not in ("early_stopping_rounds", "eval_metric")}
    params["n_estimators"] = best_n_estimators

    final_model = xgb.XGBClassifier(**params, scale_pos_weight=float(spw))
    final_model.fit(X, y, verbose=False)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"bcr_model_thresh{bad_threshold}.json"
    final_model.save_model(str(model_path))
    logger.info(f"Final model saved → {model_path}")

    # Smoke test
    loaded = xgb.XGBClassifier()
    loaded.load_model(str(model_path))
    proba = loaded.predict_proba(X[:5])[:, 1]
    assert len(proba) == 5
    logger.info(f"Load smoke test OK — sample probs: {proba.round(3)}")

    return final_model, model_path


# ── W&B training entry point ──────────────────────────────────────────────────

def train_with_wandb(bad_threshold: int = 2):
    """Full training run: load → CV → W&B log → final model save."""
    run_name = f"bcr_xgb_thresh{bad_threshold}"
    logger.info(f"Starting run: {run_name}")

    run = wandb.init(
        project = "eeg-pipeline",
        name    = run_name,
        tags    = ["bcr", "xgboost", f"thresh={bad_threshold}"],
        config  = {
            **XGB_PARAMS,
            "bad_threshold": bad_threshold,
            "n_splits":      N_SPLITS,
            "random_seed":   RANDOM_SEED,
        },
    )

    # Load data
    X, y, groups, spw, preprocessor = load_prepared_data(bad_threshold)
    wandb.config.update({
        "scale_pos_weight": spw,
        "n_features":       X.shape[1],
        "n_rows":           X.shape[0],
        "bad_rate":         float(y.mean()),
        "n_bad":            int((y == 1).sum()),
        "n_good":           int((y == 0).sum()),
    })

    # Cross-validation
    fold_metrics = run_cv(X, y, groups, spw)

    # Log per-fold metrics to W&B
    for m in fold_metrics:
        wandb.log({
            f"fold_{m['fold']}/auprc":        m["auprc"],
            f"fold_{m['fold']}/auroc":        m["auroc"],
            f"fold_{m['fold']}/f1":           m["f1"],
            f"fold_{m['fold']}/precision":    m["precision"],
            f"fold_{m['fold']}/recall":       m["recall"],
            f"fold_{m['fold']}/best_iter":    m["best_iter"],
            f"fold_{m['fold']}/val_bad_rate": m["val_bad_rate"],
        })

    # Summary metrics
    summary = summarise_folds(fold_metrics)
    wandb.log(summary)

    # Train final model on full data
    best_n = max(1, int(round(summary["mean_best_iter"])))
    final_model, model_path = train_final_model(
        X, y, spw, best_n, bad_threshold
    )
    wandb.log({"final_model_n_estimators": best_n})
    wandb.save(str(model_path))

    run.finish()
    logger.info(f"Run complete: {run_name}")
    return fold_metrics, summary


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost BCR model")
    parser.add_argument(
        "--threshold", type=int, default=2,
        help="Bad label threshold: 2 = score≥2 (3.9%% bad), 1 = score≥1 (12.8%% bad)"
    )
    args = parser.parse_args()
    train_with_wandb(bad_threshold=args.threshold)
