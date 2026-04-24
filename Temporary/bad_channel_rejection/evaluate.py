"""bad_channel_rejection/evaluate.py

BCR evaluation: loads out-of-fold predictions saved by train.py,
finds the optimal decision threshold, and produces plots + report.

No retraining. No data leakage. Fast (~seconds).

Prerequisites: run train.py first so these files exist:
    results/oof_y_true_thresh2.npy
    results/oof_y_prob_thresh2.npy

Run: python -m bad_channel_rejection.evaluate
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb
from dotenv import load_dotenv
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_recall_curve, f1_score,
    precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay,
)

load_dotenv()

# Force W&B offline — avoids network timeout on restricted connections
os.environ["WANDB_MODE"] = "offline"

# ── Config ────────────────────────────────────────────────────────────────────
BAD_THRESHOLD = 2
RESULTS_DIR   = "results"
FIGURES_DIR   = "results/figures"
WANDB_PROJECT = "eeg-bcr"

OOF_TRUE_PATH = f"{RESULTS_DIR}/oof_y_true_thresh{BAD_THRESHOLD}.npy"
OOF_PROB_PATH = f"{RESULTS_DIR}/oof_y_prob_thresh{BAD_THRESHOLD}.npy"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ── Load OOF predictions ──────────────────────────────────────────────────────

def load_oof_predictions() -> tuple[np.ndarray, np.ndarray]:
    """Load out-of-fold predictions saved by train.py."""
    if not os.path.exists(OOF_TRUE_PATH) or not os.path.exists(OOF_PROB_PATH):
        raise FileNotFoundError(
            f"OOF files not found. Run train.py first:\n"
            f"  python -m bad_channel_rejection.train --threshold {BAD_THRESHOLD}\n"
            f"Expected files:\n  {OOF_TRUE_PATH}\n  {OOF_PROB_PATH}"
        )
    y_true = np.load(OOF_TRUE_PATH)
    y_prob = np.load(OOF_PROB_PATH)
    print(f"[evaluate] Loaded OOF predictions — n={len(y_true)}, "
          f"bad_rate={y_true.mean():.4f}")
    assert len(y_true) == len(y_prob), "OOF array length mismatch"
    assert not np.isnan(y_prob).any(), "NaNs in OOF probabilities"
    return y_true, y_prob


# ── Threshold ─────────────────────────────────────────────────────────────────

def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """argmax(F1) sweep over all PR-curve thresholds."""
    prec, rec, threshs = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
    best_idx = np.argmax(f1_scores)
    threshold = float(threshs[best_idx])
    print(f"[evaluate] Optimal threshold (argmax F1): {threshold:.4f}")
    return threshold


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, auprc: float) -> str:
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.step(rec, prec, where='post', color='#58a6ff', lw=2,
            label=f"XGBoost AUPRC = {auprc:.4f}")
    ax.axhline(y=0.039, color='#f85149', ls='--', lw=1.2,
               label="Random baseline AUPRC = 0.039")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — BCR (score≥2, OOF CV)")
    ax.legend(loc="upper right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    fig.tight_layout()
    path = f"{FIGURES_DIR}/bcr_pr_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] PR curve saved → {path}")
    return path


def plot_confusion_matrix(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> str:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Good", "Bad"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix (threshold={threshold:.4f})")
    fig.tight_layout()
    path = f"{FIGURES_DIR}/bcr_confusion_matrix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] Confusion matrix saved → {path}")
    return path


# ── Report ────────────────────────────────────────────────────────────────────

def write_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    auprc: float,
    auroc: float,
) -> str:
    y_pred = (y_prob >= threshold).astype(int)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    report = f"""# BCR Evaluation Report

## Source
Out-of-fold (OOF) predictions from GroupKFold(5) CV in train.py.
Each row's prediction came from a model that never saw that row's subject during training.
No retraining performed here — these are the unbiased CV scores.

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
- Random AUPRC baseline: 0.039
- Lift over random: {auprc / 0.039:.1f}×
- Label noise ceiling ≈ 0.40–0.50 (Site 3 inter-rater r ≈ 0.10)
- Optimal threshold written to configs/pipeline_config.yaml
"""
    path = f"{RESULTS_DIR}/bcr_evaluation.md"
    with open(path, "w") as f:
        f.write(report)
    print(f"[evaluate] Report saved → {path}")
    return path


# ── Config update ─────────────────────────────────────────────────────────────

def update_pipeline_config(threshold: float):
    config_path = "configs/pipeline_config.yaml"
    try:
        with open(config_path, "r") as f:
            content = f.read()
        if "bcr:" not in content:
            content += f"\n\nbcr:\n  decision_threshold: {threshold:.4f}\n"
        else:
            lines = content.split("\n")
            new_lines = []
            found = False
            for line in lines:
                if "decision_threshold" in line:
                    new_lines.append(f"  decision_threshold: {threshold:.4f}")
                    found = True
                else:
                    new_lines.append(line)
            content = "\n".join(new_lines)
            if not found:
                content += f"\n  decision_threshold: {threshold:.4f}\n"
        with open(config_path, "w") as f:
            f.write(content)
        print(f"[evaluate] pipeline_config.yaml updated: "
              f"bcr.decision_threshold={threshold:.4f}")
    except FileNotFoundError:
        print(f"[evaluate] WARNING: configs/pipeline_config.yaml not found.")
        print(f"  Create it and add:\n  bcr:\n    decision_threshold: {threshold:.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    wandb.init(
        project=WANDB_PROJECT,
        name="bcr_evaluation_thresh2",
        tags=["evaluation", "thresh2", "oof"],
        settings=wandb.Settings(init_timeout=120),
    )

    # Load OOF predictions produced by train.py — no retraining
    y_true, y_prob = load_oof_predictions()

    # Metrics on pooled OOF scores
    auprc = average_precision_score(y_true, y_prob)
    auroc = roc_auc_score(y_true, y_prob)
    threshold = find_optimal_threshold(y_true, y_prob)

    y_pred = (y_prob >= threshold).astype(int)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)

    print(f"\n[evaluate] === RESULTS (unbiased OOF) ===")
    print(f"  AUPRC     = {auprc:.4f}")
    print(f"  AUROC     = {auroc:.4f}")
    print(f"  F1        = {f1:.4f}  (at threshold={threshold:.4f})")
    print(f"  Precision = {prec:.4f}")
    print(f"  Recall    = {rec:.4f}")
    print(f"  Lift over random baseline (0.039): {auprc / 0.039:.1f}×")

    # Plots
    pr_path = plot_pr_curve(y_true, y_prob, auprc)
    cm_path = plot_confusion_matrix(y_true, y_prob, threshold)

    # Report + config
    write_report(y_true, y_prob, threshold, auprc, auroc)
    update_pipeline_config(threshold)

    # W&B
    wandb.summary.update({
        "auprc":             auprc,
        "auroc":             auroc,
        "f1":                f1,
        "precision":         prec,
        "recall":            rec,
        "optimal_threshold": threshold,
        "random_baseline":   0.039,
        "lift_over_random":  auprc / 0.039,
    })
    wandb.log({
        "pr_curve":        wandb.Image(pr_path),
        "confusion_matrix": wandb.Image(cm_path),
    })

    wandb.finish()
    print("\n[evaluate] Done.")


if __name__ == "__main__":
    main()
