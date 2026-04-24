"""
bad_channel_rejection/rater_ablation.py

Two-stage ablation for BCR.

Stage 1 — LABEL ABLATION (model held constant = XGBoost)
    All four conditions train XGBoost on the SAME hard binary labels
    (y = 1[score >= 2] or y = 1[q_i >= 0.5]); they differ only in the
    per-sample weight scheme passed to `fit(sample_weight=...)`.

    A: hard_threshold     uniform weights (baseline)
    B: entropy_weights    w = 1 - H(votes) / H_max
    C: dawid_skene        w = |q_i - 0.5| * 2      (DS hard-confidence)
    D: dawid_skene_soft   w = P(chosen label|q_i)  (DS full-confidence)

    Winner = argmax(mean OOF AUPRC) across 5 folds.

Stage 2 — MODEL ABLATION (label strategy = Stage 1 winner)
    X: XGBoost   (baseline)
    L: LightGBM
    C: CatBoost

    Winner = argmax(mean OOF AUPRC).

Final recommendation: winning label × winning model.

Why two stages, not a 3x4 grid?
--------------------------------
A full grid of 12 conditions gives 12 numbers with overlapping confounds — if
LightGBM wins on entropy weights but XGBoost wins on DS posteriors, the
contribution of "label" vs "model" is ambiguous. Sequential ablation isolates
the effect of each axis. This is the standard controlled-variable design in
methods papers (e.g., Raschka 2018 on model evaluation).

Run
---
    python -m bad_channel_rejection.rater_ablation
"""

from __future__ import annotations

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

STAGE1_CONDITIONS = [
    ("A", "hard_threshold", "score>=2, uniform weights (baseline)"),
    ("B", "entropy_weights", "score>=2, entropy-derived weights"),
    ("C", "dawid_skene", "q>=0.5 hard + |q-0.5|*2 weights"),
    ("D", "dawid_skene_soft", "q>=0.5 hard + full-confidence weights"),
]

STAGE2_MODELS = ["xgboost", "lightgbm", "catboost"]

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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


def run_stage1_label_ablation() -> dict:
    """Stage 1: iterate over 4 labelling strategies with XGBoost."""
    logger.info("=" * 60)
    logger.info("STAGE 1 — LABEL ABLATION (model = XGBoost)")
    logger.info("=" * 60)

    results = {}
    for cid, strategy, desc in STAGE1_CONDITIONS:
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
    baseline = results["A"]["auprc_mean"]

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


def run_stage2_model_ablation(winning_strategy: str) -> dict:
    """Stage 2: iterate over 3 models using the winning label strategy."""
    logger.info("\n" + "=" * 60)
    logger.info(
        f"STAGE 2 — MODEL ABLATION (label strategy = {winning_strategy})"
    )
    logger.info("=" * 60)

    out = build_feature_matrix(
        DATA_PATH, label_strategy=winning_strategy, bad_threshold=2
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

    results = {}
    for model_name in STAGE2_MODELS:
        logger.info(f"\n--- Model: {model_name} ---")
        r = _run_cv(
            X,
            out["y_hard"],
            out["groups"],
            out["scale_pos_weight"],
            model_name=model_name,
            label=f"S2_{model_name}",
            sample_weights=weights,
        )
        r["model"] = model_name
        results[model_name] = r

    winner = max(results.keys(), key=lambda k: results[k]["auprc_mean"])
    xgb_baseline = results["xgboost"]["auprc_mean"]

    logger.info("\n" + "=" * 60)
    logger.info("STAGE 2 RESULTS")
    logger.info("=" * 60)
    for mname, r in results.items():
        delta = r["auprc_mean"] - xgb_baseline
        marker = " <-- winner" if mname == winner else ""
        logger.info(
            f"  {mname:<10}: AUPRC={r['auprc_mean']:.4f} ± "
            f"{r['auprc_std']:.4f}  Δvs_xgb={delta:+.4f}{marker}"
        )

    return {
        "conditions": results,
        "winner_model": winner,
        "xgboost_baseline_auprc": xgb_baseline,
    }


def write_report(stage1: dict, stage2: dict, out_path: Path):
    lines = [
        "# BCR Two-Stage Ablation Report",
        "",
        "## Background",
        "",
        "Inter-rater reliability analysis on this dataset shows:",
        "- Krippendorff α = 0.211 (fair agreement)",
        "- D_o = 6.8% observed pairwise disagreement",
        "- Site 3 sensitivity (DS) = 0.075 — nearly blind to bad channels",
        "- Site 4a sensitivity (DS) = 0.714 — most reliable positive detector",
        "- 493 channels (2.6%) at score=2 represent maximum ambiguity",
        "",
        "## Stage 1 — Label ablation (model = XGBoost)",
        "",
        "| ID | Strategy | AUPRC (mean ± std) | Δ vs A |",
        "|----|----------|--------------------|--------|",
    ]
    baseline = stage1["baseline_auprc"]
    for cid, r in stage1["conditions"].items():
        d = r["auprc_mean"] - baseline
        dstr = f"{d:+.4f}" if cid != "A" else "—"
        lines.append(
            f"| {cid} | {r['strategy']} | "
            f"{r['auprc_mean']:.4f} ± {r['auprc_std']:.4f} | {dstr} |"
        )
    lines.append("")
    lines.append(
        f"**Stage 1 winner:** `{stage1['winner_strategy']}` "
        f"(condition {stage1['winner_condition_id']})"
    )
    lines.append("")
    lines.append(
        f"## Stage 2 — Model ablation (label = {stage1['winner_strategy']})"
    )
    lines.append("")
    lines.append("| Model | AUPRC (mean ± std) | Δ vs XGBoost |")
    lines.append("|-------|--------------------|--------------|")
    xgb_b = stage2["xgboost_baseline_auprc"]
    for mname, r in stage2["conditions"].items():
        d = r["auprc_mean"] - xgb_b
        dstr = f"{d:+.4f}" if mname != "xgboost" else "—"
        lines.append(
            f"| {mname} | {r['auprc_mean']:.4f} ± {r['auprc_std']:.4f} | "
            f"{dstr} |"
        )
    lines.append("")
    lines.append(f"**Stage 2 winner:** `{stage2['winner_model']}`")
    lines.append("")
    lines.append("## Recommended production config")
    lines.append("")
    lines.append(f"- Label strategy: `{stage1['winner_strategy']}`")
    lines.append(f"- Model backend : `{stage2['winner_model']}`")
    lines.append("")
    lines.append(
        "Run "
        f"`python -m bad_channel_rejection.train "
        f"--label-strategy {stage1['winner_strategy']} "
        f"--model {stage2['winner_model']}` "
        "to train the final production model."
    )

    out_path.write_text("\n".join(lines))
    logger.info(f"Report saved -> {out_path}")


def main():
    wandb.init(
        project=WANDB_PROJECT,
        name="bcr_two_stage_ablation",
        tags=["ablation", "two_stage"],
        settings=wandb.Settings(init_timeout=120),
    )
    t0 = time.time()

    stage1 = run_stage1_label_ablation()
    wandb.log({
        f"stage1/{cid}/auprc": r["auprc_mean"]
        for cid, r in stage1["conditions"].items()
    })

    stage2 = run_stage2_model_ablation(stage1["winner_strategy"])
    wandb.log({
        f"stage2/{m}/auprc": r["auprc_mean"]
        for m, r in stage2["conditions"].items()
    })

    wandb.summary.update({
        "stage1_winner": stage1["winner_strategy"],
        "stage2_winner": stage2["winner_model"],
        "total_runtime_min": (time.time() - t0) / 60,
    })

    out = RESULTS_DIR / "ablation_results.json"
    out.write_text(json.dumps({"stage1": stage1, "stage2": stage2}, indent=2))
    logger.info(f"Raw results -> {out}")

    write_report(stage1, stage2, RESULTS_DIR / "ablation_report.md")

    wandb.finish()
    logger.info(
        f"\nDone. Total runtime: {(time.time()-t0)/60:.1f} min.\n"
        f"Winner: {stage1['winner_strategy']} × {stage2['winner_model']}"
    )


if __name__ == "__main__":
    main()
