"""
bad_channel_rejection/evaluate.py

BCR evaluation: loads OOF predictions saved by train.py, finds optimal
decision threshold, produces plots + report.

Usage:
    python -m bad_channel_rejection.evaluate \\
        --label-strategy dawid_skene --model xgboost

The tag = "{label_strategy}_{model}" must match what train.py produced.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
import numpy as np
import wandb
from dotenv import load_dotenv
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import build_run_tag
from .logging_config import setup_logging

load_dotenv()
os.environ.setdefault("WANDB_MODE", "offline")

logger = setup_logging(__name__)

RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
WANDB_PROJECT = "eeg-bcr"
RANDOM_BASELINE = 0.039

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_oof_predictions(tag: str) -> tuple[np.ndarray, np.ndarray]:
    ytrue_path = RESULTS_DIR / f"oof_y_true_{tag}.npy"
    yprob_path = RESULTS_DIR / f"oof_y_prob_{tag}.npy"
    if not ytrue_path.exists() or not yprob_path.exists():
        raise FileNotFoundError(
            f"OOF files not found for tag={tag!r}. Expected:\n"
            f"  {ytrue_path}\n  {yprob_path}\n"
            f"Run: python -m bad_channel_rejection.train "
            f"--label-strategy <strategy> --model <model>"
        )
    y_true = np.load(ytrue_path)
    y_prob = np.load(yprob_path)
    logger.info(
        f"Loaded OOF — n={len(y_true)}  bad_rate={y_true.mean():.4f}  tag={tag}"
    )
    return y_true, y_prob


def find_optimal_threshold(
    y_true: np.ndarray, y_prob: np.ndarray
) -> float:
    prec, rec, threshs = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
    best_idx = int(np.argmax(f1_scores))
    threshold = float(threshs[best_idx])
    logger.info(f"Optimal threshold (argmax F1): {threshold:.4f}")
    return threshold


def plot_pr_curve(
    y_true: np.ndarray, y_prob: np.ndarray, auprc: float, tag: str
) -> str:
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.step(
        rec, prec, where="post", color="#58a6ff", lw=2,
        label=f"Model AUPRC = {auprc:.4f}",
    )
    ax.axhline(
        y=RANDOM_BASELINE, color="#f85149", ls="--", lw=1.2,
        label=f"Random baseline = {RANDOM_BASELINE:.3f}",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"PR Curve — BCR ({tag}, OOF CV)")
    ax.legend(loc="upper right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    fig.tight_layout()
    path = FIGURES_DIR / f"bcr_pr_curve_{tag}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"PR curve saved -> {path}")
    return str(path)


def plot_confusion_matrix(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float, tag: str
) -> str:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Good", "Bad"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix (threshold={threshold:.4f}, {tag})")
    fig.tight_layout()
    path = FIGURES_DIR / f"bcr_confusion_matrix_{tag}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Confusion matrix saved -> {path}")
    return str(path)


def write_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    auprc: float,
    auroc: float,
    tag: str,
) -> str:
    y_pred = (y_prob >= threshold).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    report = f"""# BCR Evaluation — {tag}

## Source
Out-of-fold predictions from GroupKFold(5) CV. Each row's prediction came
from a model that never saw that row's subject during training.

## Metrics (pooled OOF)
| Metric    | Value  |
|-----------|--------|
| AUPRC     | {auprc:.4f} |
| AUROC     | {auroc:.4f} |
| F1        | {f1:.4f} |
| Precision | {prec:.4f} |
| Recall    | {rec:.4f} |
| Threshold | {threshold:.4f} |

## Confusion matrix (at optimal threshold)
| | Predicted Good | Predicted Bad |
|---|---|---|
| **True Good** | {tn} (TN) | {fp} (FP) |
| **True Bad**  | {fn} (FN) | {tp} (TP) |

## Context
- Random AUPRC baseline: {RANDOM_BASELINE}
- Lift over random: {auprc / RANDOM_BASELINE:.1f}×
- Label noise context: Krippendorff α = 0.211 (fair agreement)
"""
    path = RESULTS_DIR / f"bcr_evaluation_{tag}.md"
    path.write_text(report)
    logger.info(f"Report saved -> {path}")
    return str(path)


def main(tag: str):
    wandb.init(
        project=WANDB_PROJECT,
        name=f"bcr_evaluation_{tag}",
        tags=["evaluation", tag],
        settings=wandb.Settings(init_timeout=120),
    )

    y_true, y_prob = load_oof_predictions(tag)
    auprc = average_precision_score(y_true, y_prob)
    auroc = roc_auc_score(y_true, y_prob)
    threshold = find_optimal_threshold(y_true, y_prob)

    y_pred = (y_prob >= threshold).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)

    logger.info(
        f"Results — AUPRC={auprc:.4f}  AUROC={auroc:.4f}  "
        f"F1={f1:.4f}  P={prec:.4f}  R={rec:.4f}"
    )
    logger.info(f"Lift over random: {auprc / RANDOM_BASELINE:.1f}×")

    pr_path = plot_pr_curve(y_true, y_prob, auprc, tag)
    cm_path = plot_confusion_matrix(y_true, y_prob, threshold, tag)
    write_report(y_true, y_prob, threshold, auprc, auroc, tag)

    wandb.summary.update({
        "auprc": auprc, "auroc": auroc, "f1": f1,
        "precision": prec, "recall": rec,
        "optimal_threshold": threshold,
        "lift_over_random": auprc / RANDOM_BASELINE,
    })
    wandb.log({
        "pr_curve": wandb.Image(pr_path),
        "confusion_matrix": wandb.Image(cm_path),
    })
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-strategy", default="hard_threshold")
    parser.add_argument("--model", default="xgboost")
    parser.add_argument(
        "--use-engineered-features",
        action="store_true",
        help="DEPRECATED — evaluate the _fe run (all three transforms).",
    )
    parser.add_argument(
        "--use-impedance-interactions",
        action="store_true",
        help="Evaluate the _impedance_ix run (impedance interactions only).",
    )
    args = parser.parse_args()
    main(tag=build_run_tag(
        args.label_strategy,
        args.model,
        use_engineered_features=args.use_engineered_features,
        use_impedance_interactions=args.use_impedance_interactions,
    ))
