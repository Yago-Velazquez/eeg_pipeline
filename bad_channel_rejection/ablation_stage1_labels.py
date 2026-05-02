"""
bad_channel_rejection/ablation_stage1_labels.py

Stage 1 of the two-stage BCR ablation: label strategy ablation.

All conditions train XGBoost on the same hard binary labels; they differ
only in the per-sample weight scheme passed to fit(sample_weight=...).

    A: hard_threshold     uniform weights (baseline)
    B: entropy_weights    w = 1 - H(votes) / H_max
    C: dawid_skene        w = |q_i - 0.5| * 2      (DS hard-confidence)
    D: dawid_skene_soft   w = P(chosen label | q_i) (DS full-confidence)

Run (all conditions):
    python -m bad_channel_rejection.ablation_stage1_labels

Run (single condition):
    python -m bad_channel_rejection.ablation_stage1_labels --label-strategy dawid_skene
"""

from __future__ import annotations

import argparse
import json
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
from .features import FeaturePreprocessor
from .logging_config import setup_logging
from .models import create_model

load_dotenv()
os.environ.setdefault("WANDB_MODE", "offline")

logger = setup_logging(__name__)

DATA_PATH = "data/raw/Bad_channels_for_ML.csv"
RESULTS_DIR = Path("results")
N_FOLDS = 5
RANDOM_STATE = 42
WANDB_PROJECT = "eeg-bcr"

# GLAD and GLAD_SOFT are excluded: GLAD's scalar skill parameter cannot
# separate sensitivity from specificity, causing near-total posterior
# collapse on this imbalanced dataset (n_bad=18 vs expected 732).
# DS and MACE both handle asymmetric raters correctly and are retained.
STAGE1_CONDITIONS = [
    ("A", "hard_threshold",   "score>=2, uniform weights (baseline)"),
    ("B", "entropy_weights",  "score>=2, entropy-derived weights"),
    ("C", "dawid_skene",      "DS posterior + |q-0.5|*2 weights"),
    ("D", "dawid_skene_soft", "DS posterior + full-confidence weights"),
    ("E", "mace",             "MACE posterior + |q-0.5|*2 weights"),
    ("F", "mace_soft",        "MACE posterior + full-confidence weights"),
]


def _run_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    spw: float,
    model_name: str,
    label: str,
    sample_weights: np.ndarray | None = None,
) -> dict:
    """Single condition: GroupKFold(5) CV, return mean metrics."""
    gkf = GroupKFold(n_splits=N_FOLDS)
    auprcs, aurocs = [], []

    for fold_idx, (tr_idx, va_idx) in enumerate(
        gkf.split(X, y, groups), start=1
    ):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]
        w_tr = (
            sample_weights[tr_idx] if sample_weights is not None else None
        )

        model = create_model(model_name, scale_pos_weight=spw)
        model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=(X_va, y_va))

        y_prob = model.predict_proba(X_va)[:, 1]
        auprc = average_precision_score(y_va, y_prob)
        auroc = roc_auc_score(y_va, y_prob)
        auprcs.append(auprc)
        aurocs.append(auroc)
        logger.info(
            f"    [{label}] fold {fold_idx}: "
            f"AUPRC={auprc:.4f}  AUROC={auroc:.4f}"
        )

    return {
        "label": label,
        "auprc_mean": float(np.mean(auprcs)),
        "auprc_std": float(np.std(auprcs)),
        "auroc_mean": float(np.mean(aurocs)),
        "auroc_std": float(np.std(aurocs)),
    }


def run_stage1_label_ablation(
    label_strategy: str | None = None,
) -> dict:
    """Stage 1: iterate over label strategies with XGBoost fixed.

    Args:
        label_strategy: If given, only that one condition is run (useful
            for re-running a single condition without repeating others).
            Must match one of the strategy names in STAGE1_CONDITIONS.
    """
    logger.info("=" * 60)
    logger.info("STAGE 1 — LABEL ABLATION (model = XGBoost)")
    logger.info("=" * 60)

    conditions_to_run = STAGE1_CONDITIONS
    if label_strategy is not None:
        conditions_to_run = [
            c for c in STAGE1_CONDITIONS if c[1] == label_strategy
        ]
        if not conditions_to_run:
            valid = [c[1] for c in STAGE1_CONDITIONS]
            raise ValueError(
                f"Unknown label strategy {label_strategy!r}. "
                f"Valid options: {valid}"
            )

    results = {}
    for cid, strategy, desc in conditions_to_run:
        logger.info(f"\n--- Condition {cid} ({strategy}): {desc} ---")
        out = build_feature_matrix(
            DATA_PATH, label_strategy=strategy, bad_threshold=2
        )
        prep = FeaturePreprocessor()
        X = prep.fit_transform(
            pd.DataFrame(out["X"], columns=out["feature_cols"])
        )

        weights = (
            out["sample_weights"]
            if not np.allclose(out["sample_weights"], 1.0)
            else None
        )

        r = _run_cv(
            X,
            out["y_hard"],
            out["groups"],
            out["scale_pos_weight"],
            model_name="xgboost",
            label=f"S1_{cid}_{strategy}",
            sample_weights=weights,
        )
        r["condition_id"] = cid
        r["strategy"] = strategy
        r["description"] = desc
        results[cid] = r

    winner_cid = max(results.keys(), key=lambda k: results[k]["auprc_mean"])
    baseline_cid = next(
        (cid for cid, _, _ in conditions_to_run if cid == "A"),
        winner_cid,
    )
    baseline = results[baseline_cid]["auprc_mean"]

    logger.info("\n" + "=" * 60)
    logger.info("STAGE 1 RESULTS")
    logger.info("=" * 60)
    for cid, r in results.items():
        delta = r["auprc_mean"] - baseline
        marker = " <-- winner" if cid == winner_cid else ""
        logger.info(
            f"  {cid} {r['strategy']:<22}: "
            f"AUPRC={r['auprc_mean']:.4f} ± {r['auprc_std']:.4f}  "
            f"Δ={delta:+.4f}{marker}"
        )

    return {
        "conditions": results,
        "winner_condition_id": winner_cid,
        "winner_strategy": results[winner_cid]["strategy"],
        "baseline_auprc": baseline,
    }


def _write_stage1_report(stage1: dict, out_path: Path) -> None:
    baseline = stage1["baseline_auprc"]
    lines = [
        "# BCR Stage 1 — Label Ablation Report",
        "",
        "Model held constant: **XGBoost**",
        "",
        "| ID | Strategy | AUPRC (mean ± std) | Δ vs A |",
        "|----|----------|--------------------|--------|",
    ]
    for cid, r in stage1["conditions"].items():
        d = r["auprc_mean"] - baseline
        dstr = f"{d:+.4f}" if cid != "A" else "—"
        lines.append(
            f"| {cid} | {r['strategy']} | "
            f"{r['auprc_mean']:.4f} ± {r['auprc_std']:.4f} | {dstr} |"
        )
    lines += [
        "",
        f"**Winner:** `{stage1['winner_strategy']}` "
        f"(condition {stage1['winner_condition_id']})",
    ]
    out_path.write_text("\n".join(lines))
    logger.info(f"Report saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="BCR Stage 1: label strategy ablation (model = XGBoost)"
    )
    parser.add_argument(
        "--label-strategy",
        default=None,
        help="Run a single label strategy instead of all conditions "
        "(e.g. --label-strategy dawid_skene)",
    )
    args = parser.parse_args()

    wandb.init(
        project=WANDB_PROJECT,
        name="bcr_ablation_stage1_labels",
        tags=["ablation", "stage1", "labels"],
        settings=wandb.Settings(init_timeout=120),
    )
    t0 = time.time()

    stage1 = run_stage1_label_ablation(args.label_strategy)

    wandb.log({
        f"stage1/{cid}/auprc": r["auprc_mean"]
        for cid, r in stage1["conditions"].items()
    })
    wandb.summary.update({
        "stage1_winner": stage1["winner_strategy"],
        "runtime_min": (time.time() - t0) / 60,
    })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "stage1_results.json"
    results_path.write_text(json.dumps(stage1, indent=2))
    logger.info(f"Raw results -> {results_path}")

    _write_stage1_report(stage1, RESULTS_DIR / "stage1_report.md")

    wandb.finish()

    winner_cid = stage1["winner_condition_id"]
    winner_strategy = stage1["winner_strategy"]
    print(
        f"\nStage 1 winner: {winner_cid} — {winner_strategy}  "
        f"(AUPRC={stage1['conditions'][winner_cid]['auprc_mean']:.4f})"
    )
    logger.info(f"Done. Runtime: {(time.time() - t0) / 60:.1f} min.")


if __name__ == "__main__":
    main()
