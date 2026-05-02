"""
bad_channel_rejection/ablation_stage2_models.py

Stage 2 SHARED CORE — utilities for the two-step model selection.

Stage 2 is now split into two distinct ablations to disentangle the questions
"which ML *strategy* fits this data?" from "which *architecture* within that
strategy is best?":

    Stage 2a — STRATEGY ABLATION
        Runs ONE representative per broad ML family on the same fixed folds
        and picks the winning family.
        See `ablation_stage2a_strategies.py`.

    Stage 2b — ARCHITECTURE ABLATION
        Within the family selected by 2a, evaluates every implementation
        belonging to that family and picks the winning architecture.
        See `ablation_stage2b_architectures.py`.

This module exposes the building blocks both stages share:

    get_device()              auto-detect cuda / mps / cpu
    load_data_and_folds()     CSV → features → pre-computed GroupKFold splits
    run_cv()                  single-model GroupKFold CV
    run_model_ablation_core() generic ablation over a list of models

Fairness rules enforced inside `run_model_ablation_core()`
----------------------------------------------------------
1. Same folds     : pre-computed fold indices reused by every model.
2. Same metric    : mean GroupKFold AUPRC is the sole selection criterion.
3. Same budget    : every model gets the same n_estimators / n_epochs ceiling.
                    Early stopping may end training earlier; the cap is fixed.
4. Device logging : device + best_iteration recorded per model per fold.
5. Final CV       : the winner is re-evaluated on the same folds as a stability
                    check — large drift between ablation and final CV is flagged.
6. Margin guard   : a winner whose margin over second place is < 0.005 AUPRC
                    is flagged as within numerical-noise range.

Running this file directly now prints a redirect notice — use 2a / 2b instead.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
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

# ── Public constants (re-exported by 2a / 2b) ──────────────────────────────────
DATA_PATH = "data/raw/Bad_channels_for_ML.csv"
RESULTS_DIR = Path("results")
N_FOLDS = 5
RANDOM_STATE = 42
WANDB_PROJECT = "eeg-bcr"
MARGIN_CAUTION = 0.005   # AUPRC margin below which the winner is flagged


# ── Device detection ───────────────────────────────────────────────────────────


def get_device() -> str:
    """Return 'cuda', 'mps', or 'cpu' depending on what's available."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


# ── Data + folds (shared by 2a and 2b) ─────────────────────────────────────────


def load_data_and_folds(winning_strategy: str) -> dict:
    """Load data with the given label strategy and pre-compute fold indices.

    The returned bundle is reused across every model in 2a and 2b — guaranteeing
    that all models see *identical* train/validation splits (fairness rule 1).

    Returns
    -------
    dict with keys:
        X, y, groups, scale_pos_weight, sample_weights, fold_indices,
        feature_cols, label_strategy
    """
    out = build_feature_matrix(
        DATA_PATH, label_strategy=winning_strategy, bad_threshold=2
    )
    prep = FeaturePreprocessor()
    X = prep.fit_transform(
        pd.DataFrame(out["X"], columns=out["feature_cols"])
    )
    y = out["y_hard"]
    groups = out["groups"]
    spw = out["scale_pos_weight"]
    weights = (
        out["sample_weights"]
        if not np.allclose(out["sample_weights"], 1.0)
        else None
    )

    fold_indices = list(GroupKFold(n_splits=N_FOLDS).split(X, y, groups))
    logger.info(
        "Fold sizes: "
        + "  ".join(
            f"fold{i+1}: tr={len(tr)} va={len(va)}"
            for i, (tr, va) in enumerate(fold_indices)
        )
    )

    return {
        "X": X,
        "y": y,
        "groups": groups,
        "scale_pos_weight": spw,
        "sample_weights": weights,
        "fold_indices": fold_indices,
        "feature_cols": out["feature_cols"],
        "label_strategy": winning_strategy,
    }


# ── Single-model CV ────────────────────────────────────────────────────────────


def run_cv(
    X: np.ndarray,
    y: np.ndarray,
    fold_indices: list[tuple[np.ndarray, np.ndarray]],
    spw: float,
    model_name: str,
    label: str,
    sample_weights: np.ndarray | None = None,
    device: str = "cpu",
) -> dict:
    """GroupKFold CV over pre-computed fold indices.

    Returns per-fold AUPRC / AUROC / best_iteration plus aggregated means.
    """
    auprcs, aurocs, best_iters = [], [], []

    for fold_idx, (tr_idx, va_idx) in enumerate(fold_indices, start=1):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]
        w_tr = sample_weights[tr_idx] if sample_weights is not None else None

        model = create_model(model_name, scale_pos_weight=spw, device=device)
        model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=(X_va, y_va))

        y_prob = model.predict_proba(X_va)[:, 1]
        auprc = average_precision_score(y_va, y_prob)
        auroc = roc_auc_score(y_va, y_prob)
        auprcs.append(auprc)
        aurocs.append(auroc)
        best_iters.append(model.best_iteration)
        logger.info(
            f"    [{label}] fold {fold_idx}: "
            f"AUPRC={auprc:.4f}  AUROC={auroc:.4f}  "
            f"best_iter={model.best_iteration}"
        )

    return {
        "label": label,
        "auprc_mean": float(np.mean(auprcs)),
        "auprc_std": float(np.std(auprcs)),
        "auroc_mean": float(np.mean(aurocs)),
        "auroc_std": float(np.std(aurocs)),
        "auprc_per_fold": [float(v) for v in auprcs],
        "best_iterations": best_iters,
        "best_iter_mean": float(np.mean(best_iters)),
        "device": device,
    }


# ── Generic ablation core (shared by 2a and 2b) ────────────────────────────────


def run_model_ablation_core(
    data_bundle: dict,
    models_to_run: list[str],
    stage_label: str,
    device: str = "cpu",
    final_cv_for_winner: bool = True,
) -> dict:
    """Generic model ablation — fairness rules 1–6 enforced.

    Parameters
    ----------
    data_bundle : dict
        Output of `load_data_and_folds()`.
    models_to_run : list[str]
        Model names (must be entries of SUPPORTED_MODELS in models.py).
    stage_label : str
        Tag prefix for logging — typically "2a" or "2b".
    device : str
        'cuda', 'mps', or 'cpu'.
    final_cv_for_winner : bool
        If True (default), re-runs CV for the winner as a stability check.

    Returns
    -------
    dict with keys: conditions (per-model metrics), winner_model,
    winner_cautious, ranked, final_cv (if requested), n_folds, device.
    """
    X = data_bundle["X"]
    y = data_bundle["y"]
    fold_indices = data_bundle["fold_indices"]
    spw = data_bundle["scale_pos_weight"]
    weights = data_bundle["sample_weights"]

    results: dict[str, dict] = {}
    for model_name in models_to_run:
        logger.info(f"\n--- {stage_label} | model: {model_name} ---")
        r = run_cv(
            X, y, fold_indices, spw,
            model_name=model_name,
            label=f"S{stage_label}_{model_name}",
            sample_weights=weights,
            device=device,
        )
        r["model"] = model_name
        results[model_name] = r

    # Rule 2 — rank by mean AUPRC only.
    ranked = sorted(
        results.keys(), key=lambda k: results[k]["auprc_mean"], reverse=True
    )
    winner = ranked[0]

    # Rule 6 — margin guard.
    cautious = False
    if len(ranked) > 1:
        margin = (
            results[winner]["auprc_mean"] - results[ranked[1]]["auprc_mean"]
        )
        if margin < MARGIN_CAUTION:
            cautious = True
            logger.warning(
                f"CAUTION: margin between '{winner}' and '{ranked[1]}' is "
                f"{margin:.4f} AUPRC — within numerical-noise range "
                f"(<{MARGIN_CAUTION})."
            )

    # Rule 5 — final CV for the winner.
    final_cv = None
    if final_cv_for_winner and len(models_to_run) > 1:
        logger.info(f"\n--- Final CV: {winner} (winner confirmation) ---")
        final_cv = run_cv(
            X, y, fold_indices, spw,
            model_name=winner,
            label=f"S{stage_label}_{winner}_final",
            sample_weights=weights,
            device=device,
        )
        drift = abs(final_cv["auprc_mean"] - results[winner]["auprc_mean"])
        if drift > 0.005:
            logger.warning(
                f"Final CV AUPRC={final_cv['auprc_mean']:.4f} drifted "
                f"{drift:.4f} from ablation — winner may be unstable."
            )
        else:
            logger.info(
                f"Final CV AUPRC={final_cv['auprc_mean']:.4f}  "
                f"(drift={drift:.4f} — stable)"
            )

    logger.info("\n" + "=" * 60)
    logger.info(f"STAGE {stage_label} RESULTS  (ranked by AUPRC)")
    logger.info("=" * 60)
    for mname in ranked:
        r = results[mname]
        marker = " <-- winner" if mname == winner else ""
        caution_flag = (
            " [CAUTION: tiny margin]" if mname == winner and cautious else ""
        )
        logger.info(
            f"  {mname:<16}: "
            f"AUPRC={r['auprc_mean']:.4f} ± {r['auprc_std']:.4f}  "
            f"best_iter={r['best_iter_mean']:.0f}  "
            f"device={r['device']}{marker}{caution_flag}"
        )

    return {
        "conditions": results,
        "ranked": ranked,
        "winner_model": winner,
        "winner_cautious": cautious,
        "final_cv": final_cv,
        "n_folds": N_FOLDS,
        "device": device,
        "label_strategy": data_bundle["label_strategy"],
    }


# ── Redirect notice ────────────────────────────────────────────────────────────


def main():
    msg = (
        "This module is now a SHARED CORE for the two new stage-2 entry points.\n"
        "Run them in sequence:\n"
        "  python -m bad_channel_rejection.ablation_stage2a_strategies "
        "--winning-strategy <STRATEGY>\n"
        "  python -m bad_channel_rejection.ablation_stage2b_architectures "
        "--winning-strategy <STRATEGY> --winning-family <FAMILY>"
    )
    print(msg)


if __name__ == "__main__":
    main()
