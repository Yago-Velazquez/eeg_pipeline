"""
bad_channel_rejection/ablation_stage3_hpo.py

Stage 3 — HYPER-PARAMETER OPTIMISATION on the winning configuration.

The winning configuration is the triple
    (label_strategy   from Stage 1,
     ML family        from Stage 2a,
     architecture     from Stage 2b)
collapsed into the pair `(winning_strategy, winning_model)` that this stage
takes as input.

Optimisation strategy
---------------------
We use **Optuna** (TPE sampler — Bayesian) instead of W&B Sweeps.  This is a
deliberate change so that **all N trials live inside ONE W&B run** instead of
generating N separate runs:

    Run "bcr_hpo_<model>"
      step=0  → trial 0 metrics + hyperparameters
      step=1  → trial 1 metrics + hyperparameters
      ...
      step=N-1
      summary → best AUPRC, best config, model path

Each trial logs:
    trial/auprc_mean, trial/auprc_std, trial/auroc_mean, trial/auroc_std
    trial/best_so_far              (running best AUPRC up to this trial)
    trial/hp/<name>                (sampled hyper-parameter values)

The W&B run page therefore shows:
- A line plot of `trial/auprc_mean` vs step (the sampler's progress curve)
- Time-series for every hyperparameter, useful for visual exploration
- Best values in the run summary

After all trials, the best config is used to retrain on the full dataset and
the final model is saved to `results/best_model.<ext>`.

Run:
    python -m bad_channel_rejection.ablation_stage3_hpo \\
        --winning-strategy mace --winning-model lightgbm --count 50

Resume / extend a study (rare; mostly cosmetic since trials are cheap):
    python -m bad_channel_rejection.ablation_stage3_hpo \\
        --winning-strategy mace --winning-model lightgbm \\
        --count 20 --study-name <previous-study-name>
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import wandb
from dotenv import load_dotenv
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from .dataset import build_feature_matrix
from .features import FeaturePreprocessor
from .logging_config import setup_logging
from .models import MODEL_EXT, SUPPORTED_MODELS, create_model

load_dotenv()
os.environ.setdefault("WANDB_MODE", "offline")
# Silence Optuna's per-trial INFO log lines; we have our own logger below.
optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = setup_logging(__name__)

DATA_PATH = "data/raw/Bad_channels_for_ML.csv"
RESULTS_DIR = Path("results")
N_FOLDS = 5
RANDOM_STATE = 42
WANDB_PROJECT = "eeg-bcr"


# ── Per-model hyperparameter search spaces ─────────────────────────────────────
#
# Format is preserved from the previous W&B Sweeps version so that the
# specifications stay declarative. `_suggest_from_spec()` translates each
# entry into the matching Optuna `trial.suggest_*` call.
#
# Keys:
#   "values"       → categorical (any list of choices)
#   "distribution" → "int_uniform"        (int in [min, max])
#                  | "uniform"            (float in [min, max])
#                  | "log_uniform_values" (float, log-uniform on actual values)
#
# max_depth=0 is a sentinel for None in Random Forest (translated below).

SWEEP_SPACES: dict[str, dict[str, Any]] = {
    "elasticnet": {
        "C":        {"min": 1e-3, "max": 100.0, "distribution": "log_uniform_values"},
        "l1_ratio": {"min": 0.0,  "max": 1.0,   "distribution": "uniform"},
    },
    "random_forest": {
        "n_estimators":     {"values": [200, 300, 500, 750, 1000]},
        "max_depth":        {"values": [0, 5, 10, 15, 20]},   # 0 → None
        "max_features":     {"values": ["sqrt", "log2", 0.3, 0.5]},
        "min_samples_leaf": {"min": 1, "max": 10, "distribution": "int_uniform"},
    },
    "xgboost": {
        "max_depth":        {"min": 3,    "max": 10,   "distribution": "int_uniform"},
        "learning_rate":    {"min": 0.01, "max": 0.3,  "distribution": "log_uniform_values"},
        "subsample":        {"min": 0.5,  "max": 1.0,  "distribution": "uniform"},
        "colsample_bytree": {"min": 0.5,  "max": 1.0,  "distribution": "uniform"},
        "min_child_weight": {"min": 1,    "max": 10,   "distribution": "int_uniform"},
        "gamma":            {"min": 1e-4, "max": 1.0,  "distribution": "log_uniform_values"},
        "reg_alpha":        {"min": 1e-4, "max": 10.0, "distribution": "log_uniform_values"},
        "reg_lambda":       {"min": 1e-4, "max": 10.0, "distribution": "log_uniform_values"},
    },
    "lightgbm": {
        "num_leaves":        {"min": 15,   "max": 255,  "distribution": "int_uniform"},
        "learning_rate":     {"min": 0.01, "max": 0.3,  "distribution": "log_uniform_values"},
        "subsample":         {"min": 0.5,  "max": 1.0,  "distribution": "uniform"},
        "colsample_bytree":  {"min": 0.5,  "max": 1.0,  "distribution": "uniform"},
        "min_child_samples": {"min": 5,    "max": 100,  "distribution": "int_uniform"},
        "reg_alpha":         {"min": 1e-4, "max": 10.0, "distribution": "log_uniform_values"},
        "reg_lambda":        {"min": 1e-4, "max": 10.0, "distribution": "log_uniform_values"},
    },
    "catboost": {
        "depth":           {"min": 4,    "max": 10,   "distribution": "int_uniform"},
        "learning_rate":   {"min": 0.01, "max": 0.3,  "distribution": "log_uniform_values"},
        "subsample":       {"min": 0.5,  "max": 1.0,  "distribution": "uniform"},
        "l2_leaf_reg":     {"min": 1.0,  "max": 10.0, "distribution": "log_uniform_values"},
        "random_strength": {"min": 0.1,  "max": 10.0, "distribution": "log_uniform_values"},
    },
    "ft_transformer": {
        "d_token":       {"values": [32, 64, 128, 256]},
        "n_layers":      {"values": [2, 3, 4]},
        "n_heads":       {"values": [4, 8]},
        "ffn_dropout":   {"min": 0.0,  "max": 0.5,  "distribution": "uniform"},
        "attn_dropout":  {"min": 0.0,  "max": 0.5,  "distribution": "uniform"},
        "learning_rate": {"min": 1e-5, "max": 1e-3, "distribution": "log_uniform_values"},
        "weight_decay":  {"min": 1e-6, "max": 1e-3, "distribution": "log_uniform_values"},
        "batch_size":    {"values": [128, 256, 512]},
    },
}


# ── Helpers ────────────────────────────────────────────────────────────────────


def _suggest_from_spec(trial: optuna.Trial, name: str, spec: dict) -> Any:
    """Map a SWEEP_SPACES entry to the matching Optuna `trial.suggest_*` call."""
    if "values" in spec:
        return trial.suggest_categorical(name, spec["values"])
    dist = spec.get("distribution", "uniform")
    if dist == "int_uniform":
        return trial.suggest_int(name, spec["min"], spec["max"])
    if dist == "log_uniform_values":
        return trial.suggest_float(name, spec["min"], spec["max"], log=True)
    if dist == "uniform":
        return trial.suggest_float(name, spec["min"], spec["max"])
    raise ValueError(f"Unknown distribution {dist!r} for hyperparameter {name!r}")


def _overrides_from_config(model_name: str, cfg: dict) -> dict:
    """Translate raw sampled config into create_model() overrides.

    Handles model-specific sentinel values and constraints:
    - Random Forest: max_depth=0 is mapped to None.
    - FT-Transformer: d_token must be divisible by n_heads; if not, n_heads
      is clamped to the largest valid divisor (trial is not aborted).
    """
    overrides = dict(cfg)

    if model_name == "random_forest":
        if overrides.get("max_depth") == 0:
            overrides["max_depth"] = None

    if model_name == "ft_transformer":
        d = overrides.get("d_token", 64)
        h = overrides.get("n_heads", 8)
        if d % h != 0:
            valid = [k for k in [1, 2, 4, 8, 16] if d % k == 0]
            overrides["n_heads"] = max(v for v in valid if v <= h) if valid else 1
            logger.debug(
                f"FT-Transformer: d_token={d} not divisible by n_heads={h}; "
                f"clamped to n_heads={overrides['n_heads']}"
            )

    return overrides


def _run_cv(
    X: np.ndarray,
    y: np.ndarray,
    fold_indices: list[tuple[np.ndarray, np.ndarray]],
    spw: float,
    weights: np.ndarray | None,
    model_name: str,
    device: str,
    overrides: dict,
) -> tuple[float, float, float, float]:
    """Run GroupKFold CV for one HPO trial.

    Same fold layout as Stage 2 (deterministic GroupKFold) so HPO results are
    directly comparable to the Stage 2 ablation numbers.

    Returns
    -------
    auprc_mean, auprc_std, auroc_mean, auroc_std
    """
    auprcs, aurocs = [], []
    for tr_idx, va_idx in fold_indices:
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]
        w_tr = weights[tr_idx] if weights is not None else None

        model = create_model(model_name, scale_pos_weight=spw, device=device, **overrides)
        model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=(X_va, y_va))

        y_prob = model.predict_proba(X_va)[:, 1]
        auprcs.append(average_precision_score(y_va, y_prob))
        aurocs.append(roc_auc_score(y_va, y_prob))

    return (
        float(np.mean(auprcs)),
        float(np.std(auprcs)),
        float(np.mean(aurocs)),
        float(np.std(aurocs)),
    )


# ── Core orchestration ─────────────────────────────────────────────────────────


def run_stage3_hpo(
    winning_strategy: str,
    winning_model: str,
    count: int = 50,
    device: str = "cpu",
    study_name: str | None = None,
) -> dict:
    """Stage 3: Optuna-driven HPO inside a SINGLE W&B run.

    Args
    ----
    winning_strategy : label strategy from Stage 1 (e.g. 'mace').
    winning_model    : architecture from Stage 2b (e.g. 'lightgbm').
    count            : number of Optuna trials to run.
    device           : 'cuda', 'mps', or 'cpu'.
    study_name       : optional Optuna study name (mostly cosmetic).

    Returns
    -------
    dict with keys: study_name, best_config, best_auprc_mean, best_auprc_std,
                    best_auroc_mean, all_trials, model_path, n_trials, device.
    """
    if winning_model not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown model {winning_model!r}. Valid: {SUPPORTED_MODELS}"
        )
    if winning_model not in SWEEP_SPACES:
        raise ValueError(
            f"No HPO search space defined for {winning_model!r}. "
            f"Add an entry to SWEEP_SPACES."
        )

    logger.info("\n" + "=" * 60)
    logger.info(
        f"STAGE 3 — HPO  "
        f"(strategy={winning_strategy}  model={winning_model}  "
        f"trials={count}  device={device})"
    )
    logger.info("=" * 60)

    # ── Data ──────────────────────────────────────────────────────────────────
    out = build_feature_matrix(
        DATA_PATH, label_strategy=winning_strategy, bad_threshold=2
    )
    prep = FeaturePreprocessor()
    X = prep.fit_transform(
        pd.DataFrame(out["X"], columns=out["feature_cols"])
    )
    y      = out["y_hard"]
    groups = out["groups"]
    spw    = out["scale_pos_weight"]
    raw_w  = out["sample_weights"]
    weights = raw_w if not np.allclose(raw_w, 1.0) else None

    # Same fold layout as Stage 2 for direct comparability.
    fold_indices = list(GroupKFold(n_splits=N_FOLDS).split(X, y, groups))
    logger.info(
        "Fold sizes: "
        + "  ".join(
            f"fold{i+1}: tr={len(tr)} va={len(va)}"
            for i, (tr, va) in enumerate(fold_indices)
        )
    )

    # ── ONE W&B run for the entire HPO experiment ─────────────────────────────
    if wandb.run is not None:
        wandb.finish()
    wandb.init(
        project=WANDB_PROJECT,
        name=f"bcr_hpo_{winning_model}",
        tags=["hpo", "stage3", winning_model, winning_strategy],
        config={
            "winning_strategy": winning_strategy,
            "winning_model":    winning_model,
            "n_trials_planned": count,
            "device":           device,
            "n_folds":          N_FOLDS,
            "sampler":          "optuna_tpe",
        },
        settings=wandb.Settings(init_timeout=120),
    )

    trial_results: list[dict] = []

    def objective(trial: optuna.Trial) -> float:
        # 1) Sample a configuration from SWEEP_SPACES via Optuna
        spaces = SWEEP_SPACES[winning_model]
        cfg = {n: _suggest_from_spec(trial, n, s) for n, s in spaces.items()}
        overrides = _overrides_from_config(winning_model, cfg)

        # 2) Cross-validated evaluation
        try:
            auprc_mean, auprc_std, auroc_mean, auroc_std = _run_cv(
                X, y, fold_indices, spw, weights, winning_model, device, overrides
            )
        except Exception as exc:
            logger.warning(f"Trial {trial.number} failed: {exc}")
            wandb.log(
                {"trial/failed": 1, "trial/auprc_mean": 0.0},
                step=trial.number,
            )
            return 0.0

        # 3) Update running best (so the W&B chart shows the convergence curve)
        prev_best = max((r["auprc_mean"] for r in trial_results), default=0.0)
        running_best = max(prev_best, auprc_mean)

        # 4) Log this trial as ONE step inside the single W&B run
        log_payload = {
            "trial/auprc_mean":   auprc_mean,
            "trial/auprc_std":    auprc_std,
            "trial/auroc_mean":   auroc_mean,
            "trial/auroc_std":    auroc_std,
            "trial/best_so_far":  running_best,
        }
        for k, v in cfg.items():
            log_payload[f"trial/hp/{k}"] = v
        wandb.log(log_payload, step=trial.number)

        logger.info(
            f"Trial {trial.number:>3}: "
            f"AUPRC={auprc_mean:.4f} ± {auprc_std:.4f}  "
            f"(best so far: {running_best:.4f})"
        )

        trial_results.append({
            "trial_number": trial.number,
            "config":       overrides,
            "auprc_mean":   auprc_mean,
            "auprc_std":    auprc_std,
            "auroc_mean":   auroc_mean,
            "auroc_std":    auroc_std,
        })
        return auprc_mean

    # ── Optuna study (TPE Bayesian sampler) ───────────────────────────────────
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        study_name=study_name or f"bcr_hpo_{winning_model}_{int(time.time())}",
    )
    logger.info(f"Optuna study: {study.study_name}  ({count} trials, TPE sampler)")
    study.optimize(objective, n_trials=count, show_progress_bar=False)

    if not trial_results:
        wandb.finish()
        raise RuntimeError("No trials completed successfully.")

    best = max(trial_results, key=lambda r: r["auprc_mean"])
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 3 — BEST TRIAL")
    logger.info("=" * 60)
    logger.info(f"  Trial #: {best['trial_number']}")
    logger.info(f"  Config : {best['config']}")
    logger.info(
        f"  AUPRC  : {best['auprc_mean']:.4f} ± {best['auprc_std']:.4f}"
    )

    # ── OOF predictions for the best config ───────────────────────────────────
    # We re-run CV one final time with the winning hyperparameters and assemble
    # out-of-fold predictions for every row. These feed `evaluate.py` (threshold
    # selection, PR curves, calibration), `shap_analysis.py`, etc., and remain
    # an unbiased estimate of generalization for the chosen config — no row was
    # in its own training set when its prediction was made.
    logger.info("\nComputing OOF predictions for best config...")
    oof_y_true = np.zeros(len(y), dtype=np.int64)
    oof_y_prob = np.zeros(len(y), dtype=np.float64)
    for fi, (tr_idx, va_idx) in enumerate(fold_indices, start=1):
        m = create_model(
            winning_model, scale_pos_weight=spw, device=device, **best["config"]
        )
        m.fit(
            X[tr_idx], y[tr_idx],
            sample_weight=weights[tr_idx] if weights is not None else None,
            eval_set=(X[va_idx], y[va_idx]),
        )
        oof_y_true[va_idx] = y[va_idx]
        oof_y_prob[va_idx] = m.predict_proba(X[va_idx])[:, 1]
        logger.info(f"  OOF fold {fi}/{N_FOLDS} done")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    oof_true_path = RESULTS_DIR / "oof_y_true_best.npy"
    oof_prob_path = RESULTS_DIR / "oof_y_prob_best.npy"
    np.save(oof_true_path, oof_y_true)
    np.save(oof_prob_path, oof_y_prob)
    logger.info(f"OOF predictions saved -> {oof_true_path}, {oof_prob_path}")

    # `best_config.json` is consumed by site_generalisation.py and any
    # production wrapper that needs to reconstruct the winning model.
    best_config_path = RESULTS_DIR / "best_config.json"
    best_config_path.write_text(json.dumps({
        "winning_strategy": winning_strategy,
        "winning_model":    winning_model,
        "best_config":      best["config"],
        "best_auprc_mean":  best["auprc_mean"],
        "best_auprc_std":   best["auprc_std"],
        "device":           device,
        "n_folds":          N_FOLDS,
    }, indent=2))
    logger.info(f"Best config saved -> {best_config_path}")

    # ── Retrain on full dataset with best config ──────────────────────────────
    logger.info("\nRetraining on full dataset with best config...")
    final_model = create_model(
        winning_model, scale_pos_weight=spw, device=device, **best["config"]
    )
    final_model.fit(X, y, sample_weight=weights)   # no eval_set → full budget

    ext = MODEL_EXT[winning_model]
    model_path = RESULTS_DIR / f"best_model.{ext}"
    final_model.save(model_path)
    logger.info(f"Final model saved -> {model_path}")

    # ── Final summary on the SAME W&B run ─────────────────────────────────────
    wandb.summary.update({
        "best_trial":       best["trial_number"],
        "best_auprc_mean":  best["auprc_mean"],
        "best_auprc_std":   best["auprc_std"],
        "best_auroc_mean":  best["auroc_mean"],
        "best_config":      best["config"],
        "model_path":       str(model_path),
        "n_trials":         len(trial_results),
        "study_name":       study.study_name,
    })
    wandb.finish()

    return {
        "study_name":       study.study_name,
        "best_trial":       best["trial_number"],
        "best_config":      best["config"],
        "best_auprc_mean":  best["auprc_mean"],
        "best_auprc_std":   best["auprc_std"],
        "best_auroc_mean":  best["auroc_mean"],
        "all_trials":       trial_results,
        "model_path":       str(model_path),
        "winning_model":    winning_model,
        "winning_strategy": winning_strategy,
        "n_trials":         len(trial_results),
        "device":           device,
    }


# ── Report writer ──────────────────────────────────────────────────────────────


def _write_stage3_report(stage3: dict, out_path: Path) -> None:
    best_auprc = stage3["best_auprc_mean"]
    trials = sorted(
        stage3["all_trials"], key=lambda r: r["auprc_mean"], reverse=True
    )

    lines = [
        "# BCR Stage 3 — HPO Report",
        "",
        f"Model: **{stage3['winning_model']}** | "
        f"Label strategy: **{stage3['winning_strategy']}** | "
        f"Trials: **{stage3['n_trials']}** | "
        f"Device: **{stage3['device']}**",
        "",
        f"Optuna study: `{stage3['study_name']}`",
        "",
        "## Best Configuration",
        "",
        f"Best trial: **#{stage3['best_trial']}**",
        "",
        "```json",
        json.dumps(stage3["best_config"], indent=2),
        "```",
        "",
        f"**Best CV AUPRC:** {best_auprc:.4f} ± {stage3['best_auprc_std']:.4f}",
        f"**Best CV AUROC:** {stage3['best_auroc_mean']:.4f}",
        f"**Saved model:**   `{stage3['model_path']}`",
        "",
        "## All Trials (top 10)",
        "",
        "| Rank | Trial # | AUPRC mean | AUPRC std | AUROC mean |",
        "|------|---------|------------|-----------|------------|",
    ]
    for rank, t in enumerate(trials[:10], 1):
        marker = " ✓" if rank == 1 else ""
        lines.append(
            f"| {rank} | {t['trial_number']}{marker} | "
            f"{t['auprc_mean']:.4f} | {t['auprc_std']:.4f} | "
            f"{t['auroc_mean']:.4f} |"
        )

    out_path.write_text("\n".join(lines))
    logger.info(f"Report saved -> {out_path}")


# ── Entry point ────────────────────────────────────────────────────────────────


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


def main():
    parser = argparse.ArgumentParser(
        description="BCR Stage 3: HPO on winning (label_strategy, model) pair"
    )
    parser.add_argument(
        "--winning-strategy",
        required=True,
        help="Label strategy from Stage 1 (e.g. mace)",
    )
    parser.add_argument(
        "--winning-model",
        required=True,
        help=f"Model from Stage 2. Choices: {list(SUPPORTED_MODELS)}",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=50,
        help="Number of Optuna trials to run (default: 50)",
    )
    parser.add_argument(
        "--study-name",
        default=None,
        help="Optional Optuna study name (default: auto-generated)",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Override device detection (default: auto-detect).",
    )
    args = parser.parse_args()

    device = args.device if args.device is not None else get_device()
    logger.info(
        f"Device: {device} "
        f"({'auto-detected' if args.device is None else 'override'})"
    )

    t0 = time.time()
    stage3 = run_stage3_hpo(
        winning_strategy=args.winning_strategy,
        winning_model=args.winning_model,
        count=args.count,
        device=device,
        study_name=args.study_name,
    )

    # Local artefacts (separate from W&B; persisted regardless of W&B status).
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "stage3_results.json"
    results_path.write_text(json.dumps(stage3, indent=2))
    logger.info(f"Raw results -> {results_path}")
    _write_stage3_report(stage3, RESULTS_DIR / "stage3_report.md")

    print(
        f"\nStage 3 complete.\n"
        f"  Best AUPRC : {stage3['best_auprc_mean']:.4f} ± "
        f"{stage3['best_auprc_std']:.4f}  (trial #{stage3['best_trial']})\n"
        f"  Best config: {stage3['best_config']}\n"
        f"  Model saved: {stage3['model_path']}\n"
        f"  Optuna study: {stage3['study_name']}"
    )
    logger.info(f"Done. Runtime: {(time.time() - t0) / 60:.1f} min.")


if __name__ == "__main__":
    main()
