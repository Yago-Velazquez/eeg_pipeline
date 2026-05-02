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
    """Load OOF predictions saved either by train.py (tagged) or Stage 3 (best)."""
    if tag == "best":
        ytrue_path = RESULTS_DIR / "oof_y_true_best.npy"
        yprob_path = RESULTS_DIR / "oof_y_prob_best.npy"
        hint = (
            "Run: python -m bad_channel_rejection.ablation_stage3_hpo "
            "--winning-strategy <X> --winning-model <Y> --count 50"
        )
    else:
        ytrue_path = RESULTS_DIR / f"oof_y_true_{tag}.npy"
        yprob_path = RESULTS_DIR / f"oof_y_prob_{tag}.npy"
        hint = (
            "Run: python -m bad_channel_rejection.train "
            "--label-strategy <strategy> --model <model>"
        )
    if not ytrue_path.exists() or not yprob_path.exists():
        raise FileNotFoundError(
            f"OOF files not found for tag={tag!r}. Expected:\n"
            f"  {ytrue_path}\n  {yprob_path}\n{hint}"
        )
    y_true = np.load(ytrue_path)
    y_prob = np.load(yprob_path)
    logger.info(
        f"Loaded OOF — n={len(y_true)}  bad_rate={y_true.mean():.4f}  tag={tag}"
    )
    return y_true, y_prob


def _metrics_at_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> dict:
    """Compute precision / recall / F1 / TP / FP / TN / FN at a fixed threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    return {
        "threshold": float(threshold),
        "precision": float(p),
        "recall":    float(r),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def find_f1_optimal_threshold(
    y_true: np.ndarray, y_prob: np.ndarray
) -> float:
    """Threshold that maximises F1 score across the PR curve."""
    prec, rec, threshs = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
    best_idx = int(np.argmax(f1_scores))
    threshold = float(threshs[best_idx])
    logger.info(f"F1-optimal threshold: {threshold:.4f}")
    return threshold


def find_threshold_for_recall(
    y_true: np.ndarray, y_prob: np.ndarray, target_recall: float
) -> float | None:
    """Highest threshold that still achieves recall >= target_recall.

    Returns None if no threshold can hit the target.
    """
    prec, rec, threshs = precision_recall_curve(y_true, y_prob)
    # rec[-1] is recall at the smallest threshold (~all positives); strip it.
    rec_t = rec[:-1]
    mask = rec_t >= target_recall
    if not mask.any():
        logger.warning(
            f"No threshold achieves recall >= {target_recall:.2f}. "
            f"Max OOF recall = {rec_t.max():.3f}"
        )
        return None
    # Among thresholds that meet the target, pick the highest one (most
    # precision while still hitting the recall floor).
    valid = np.where(mask)[0]
    best_idx = int(valid[np.argmax(threshs[valid])])
    threshold = float(threshs[best_idx])
    logger.info(
        f"Recall>={target_recall:.2f} threshold: {threshold:.4f} "
        f"(precision={prec[best_idx]:.3f}, recall={rec[best_idx]:.3f})"
    )
    return threshold


def find_threshold_for_precision(
    y_true: np.ndarray, y_prob: np.ndarray, target_precision: float
) -> float | None:
    """Lowest threshold that still achieves precision >= target_precision.

    Returns None if no threshold can hit the target.
    """
    prec, rec, threshs = precision_recall_curve(y_true, y_prob)
    prec_t = prec[:-1]
    mask = prec_t >= target_precision
    if not mask.any():
        logger.warning(
            f"No threshold achieves precision >= {target_precision:.2f}. "
            f"Max OOF precision = {prec_t.max():.3f}"
        )
        return None
    # Lowest threshold that meets target → maximises recall under the
    # precision floor.
    valid = np.where(mask)[0]
    best_idx = int(valid[np.argmin(threshs[valid])])
    threshold = float(threshs[best_idx])
    logger.info(
        f"Precision>={target_precision:.2f} threshold: {threshold:.4f} "
        f"(precision={prec[best_idx]:.3f}, recall={rec[best_idx]:.3f})"
    )
    return threshold


# Backwards-compatible alias — older callers import this name.
def find_optimal_threshold(y_true, y_prob):
    return find_f1_optimal_threshold(y_true, y_prob)


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
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    tag: str,
    op_label: str | None = None,
) -> str:
    """Plot confusion matrix at a single threshold.

    `op_label` is the operating-point label ('f1', 'recall85', 'prec80', ...).
    It's appended to the filename so multiple operating points can coexist.
    """
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Good", "Bad"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    title_suffix = f"@{op_label}" if op_label else ""
    ax.set_title(
        f"Confusion Matrix {title_suffix} "
        f"(threshold={threshold:.4f}, {tag})".strip()
    )
    fig.tight_layout()
    suffix = f"_{op_label}" if op_label else ""
    path = FIGURES_DIR / f"bcr_confusion_matrix_{tag}{suffix}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Confusion matrix saved -> {path}")
    return str(path)


def write_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    auprc: float,
    auroc: float,
    tag: str,
    operating_points: dict[str, dict],
) -> str:
    """Write a Markdown evaluation report covering N operating points.

    `operating_points` maps a short label ('f1', 'recall85', 'prec80', ...)
    to a metrics dict produced by `_metrics_at_threshold()`.
    """
    lines = [
        f"# BCR Evaluation — {tag}",
        "",
        "## Source",
        "Out-of-fold predictions from GroupKFold(5) CV. Each row's prediction came",
        "from a model that never saw that row's subject during training.",
        "",
        "## Threshold-independent metrics",
        "| Metric | Value |",
        "|--------|-------|",
        f"| AUPRC  | {auprc:.4f} |",
        f"| AUROC  | {auroc:.4f} |",
        f"| Lift over random | {auprc / RANDOM_BASELINE:.1f}× (baseline = {RANDOM_BASELINE}) |",
        "",
        "## Operating points",
        "| Mode | Threshold | Precision | Recall | F1 | TP | FP | FN | TN |",
        "|------|-----------|-----------|--------|----|----|----|----|----|",
    ]

    op_descriptions = {
        "f1":       "F1-optimal (balanced research benchmark)",
        "recall":   "Recall target (production BCR — catch most bad channels)",
        "precision": "Precision target (conservative auto-action)",
    }

    for label, m in operating_points.items():
        if m is None:
            lines.append(
                f"| **{op_descriptions.get(label.split('_')[0], label)}** "
                f"| — | — | — | — | — | — | — | — |  *(target unreachable)*"
            )
            continue
        pretty_label = op_descriptions.get(label.split("_")[0], label)
        lines.append(
            f"| **{pretty_label}** "
            f"| {m['threshold']:.4f} "
            f"| {m['precision']:.4f} "
            f"| {m['recall']:.4f} "
            f"| {m['f1']:.4f} "
            f"| {m['tp']} | {m['fp']} | {m['fn']} | {m['tn']} |"
        )

    lines += [
        "",
        "## Confusion matrices",
        "Plots in `results/figures/bcr_confusion_matrix_<tag>_<mode>.png`.",
        "",
        "## Context",
        f"- Random AUPRC baseline: {RANDOM_BASELINE}",
        "- Label noise context: Krippendorff α = 0.211 (fair agreement)",
        "- For BCR production use, prefer the **Recall target** row over F1-optimal:",
        "  missing a bad channel costs more than interpolating an extra good one.",
    ]

    path = RESULTS_DIR / f"bcr_evaluation_{tag}.md"
    path.write_text("\n".join(lines))
    logger.info(f"Report saved -> {path}")
    return str(path)


def main(
    tag: str,
    target_recall: float = 0.85,
    target_precision: float = 0.80,
):
    wandb.init(
        project=WANDB_PROJECT,
        name=f"bcr_evaluation_{tag}",
        tags=["evaluation", tag],
        settings=wandb.Settings(init_timeout=120),
    )

    y_true, y_prob = load_oof_predictions(tag)
    auprc = average_precision_score(y_true, y_prob)
    auroc = roc_auc_score(y_true, y_prob)

    # ── Three operating points ────────────────────────────────────────────────
    thr_f1   = find_f1_optimal_threshold(y_true, y_prob)
    thr_rec  = find_threshold_for_recall(y_true, y_prob, target_recall)
    thr_prec = find_threshold_for_precision(y_true, y_prob, target_precision)

    op_f1   = _metrics_at_threshold(y_true, y_prob, thr_f1)
    op_rec  = _metrics_at_threshold(y_true, y_prob, thr_rec)  if thr_rec  is not None else None
    op_prec = _metrics_at_threshold(y_true, y_prob, thr_prec) if thr_prec is not None else None

    operating_points = {
        "f1":                              op_f1,
        f"recall_ge_{int(target_recall * 100)}":     op_rec,
        f"precision_ge_{int(target_precision * 100)}": op_prec,
    }

    logger.info(
        f"AUPRC={auprc:.4f}  AUROC={auroc:.4f}  "
        f"lift={auprc / RANDOM_BASELINE:.1f}× over random"
    )
    for name, op in operating_points.items():
        if op is None:
            logger.info(f"  [{name}] target unreachable")
            continue
        logger.info(
            f"  [{name}] thr={op['threshold']:.4f}  "
            f"P={op['precision']:.3f}  R={op['recall']:.3f}  "
            f"F1={op['f1']:.3f}"
        )

    # ── Plots ─────────────────────────────────────────────────────────────────
    pr_path = plot_pr_curve(y_true, y_prob, auprc, tag)
    cm_paths: dict[str, str] = {}
    cm_paths["f1"] = plot_confusion_matrix(y_true, y_prob, thr_f1, tag, op_label="f1")
    if thr_rec is not None:
        cm_paths[f"recall{int(target_recall * 100)}"] = plot_confusion_matrix(
            y_true, y_prob, thr_rec, tag,
            op_label=f"recall{int(target_recall * 100)}",
        )
    if thr_prec is not None:
        cm_paths[f"prec{int(target_precision * 100)}"] = plot_confusion_matrix(
            y_true, y_prob, thr_prec, tag,
            op_label=f"prec{int(target_precision * 100)}",
        )

    # ── Report + W&B ──────────────────────────────────────────────────────────
    write_report(y_true, y_prob, auprc, auroc, tag, operating_points)

    summary = {
        "auprc": auprc, "auroc": auroc,
        "lift_over_random": auprc / RANDOM_BASELINE,
        "target_recall":    target_recall,
        "target_precision": target_precision,
    }
    for name, op in operating_points.items():
        if op is None:
            continue
        for key in ("threshold", "precision", "recall", "f1"):
            summary[f"{name}/{key}"] = op[key]
    wandb.summary.update(summary)

    log_payload = {"pr_curve": wandb.Image(pr_path)}
    for label, path in cm_paths.items():
        log_payload[f"confusion_matrix_{label}"] = wandb.Image(path)
    wandb.log(log_payload)
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-strategy", default="hard_threshold")
    parser.add_argument("--model", default="xgboost")
    parser.add_argument(
        "--from-stage3",
        action="store_true",
        help="Evaluate the Stage 3 HPO winner: read oof_y_*_best.npy and use "
             "tag='best'. Overrides --label-strategy / --model.",
    )
    parser.add_argument(
        "--target-recall",
        type=float,
        default=0.85,
        help=("Recall floor for the production-mode operating point. "
              "Returns the highest threshold that still meets it (default 0.85)."),
    )
    parser.add_argument(
        "--target-precision",
        type=float,
        default=0.80,
        help=("Precision floor for the conservative-mode operating point. "
              "Returns the lowest threshold that still meets it (default 0.80)."),
    )
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
    tag = "best" if args.from_stage3 else build_run_tag(
        args.label_strategy,
        args.model,
        use_engineered_features=args.use_engineered_features,
        use_impedance_interactions=args.use_impedance_interactions,
    )
    main(
        tag=tag,
        target_recall=args.target_recall,
        target_precision=args.target_precision,
    )
