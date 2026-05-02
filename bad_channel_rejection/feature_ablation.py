"""
bad_channel_rejection/feature_ablation.py

Feature-group ablation study for BCR.

Three modes (chosen via --mode)
---------------
1. ``group``     : 6 hand-crafted A–F conditions (legacy, default).
2. ``engineered``: 5-condition ablation of the engineered transforms.
3. ``stage3``    : Stage-3 HPO-aware feature-category analysis.
                   - Uses the winning model + hyperparameters automatically.
                   - Covers every feature column via EXTENDED_GROUP_PATTERNS;
                     uncategorised features land in an `other` bucket.
                   - Runs leave-one-category-out across all categories.
                   - Auto-prunes categories whose removal didn't hurt AUPRC by
                     more than --prune-threshold (default 0.005).
                   - Retrains on the reduced feature set with the same HPO
                     config and saves results/best_model_reduced.<ext> +
                     results/best_config_reduced.json.

Run
---
    # Stage-3 aware analysis (recommended after Stage 3 HPO):
    python -m bad_channel_rejection.feature_ablation --mode stage3

    # Legacy 6-condition group ablation:
    python -m bad_channel_rejection.feature_ablation --mode group \\
        --label-strategy mace --model lightgbm
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

import json

from .dataset import build_feature_matrix
from .features import FeaturePreprocessor, preprocess_fold
from .logging_config import setup_logging
from .models import MODEL_EXT, create_model

load_dotenv()
os.environ.setdefault("WANDB_MODE", "offline")

logger = setup_logging(__name__)

CSV_PATH = "data/raw/Bad_channels_for_ML.csv"
RESULTS_DIR = Path("results")
OUT_MD = RESULTS_DIR / "bcr_feature_ablation.md"
OUT_MD_ENGINEERED = RESULTS_DIR / "bcr_engineered_ablation.md"
OUT_MD_STAGE3 = RESULTS_DIR / "bcr_feature_category_analysis.md"

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

# Stage-3-mode patterns: comprehensive coverage of the 156-column feature matrix,
# including amplitude statistics and the categorical channel-position encoding
# (channel_label_enc) that aren't covered by the legacy GROUP_PATTERNS.
EXTENDED_GROUP_PATTERNS = {
    "impedance":     ["impedance_missing"],
    "frequency":     [
        " Median frequency",
        " First quartile frequency",
        " Third quartile frequency",
    ],
    "amplitude":     [" Standard deviation", " Skewness"],
    "spatial":       [
        " Correllation with neighbors",
        " Correllation with second-degree",
        " Global correlation",
        "channel_label_enc",
    ],
    "band_power":    [" Low gamma", " high gamma"],
    "decomposition": [
        " PCA", " ICA", " residuals", " reconstruction",
        " Kurtosis", " independence", " unmixing",
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
    overrides: dict | None = None,
    device: str = "cpu",
) -> dict:
    """Run GroupKFold(5) for one feature subset.

    `overrides` is forwarded to create_model() — pass the Stage 3 HPO
    winning hyperparameters here to evaluate the reduced feature set under
    the same model configuration as production.
    """
    if keep_cols is not None:
        X_df = X_df[keep_cols]

    overrides = dict(overrides or {})

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
        model = create_model(
            model_name, scale_pos_weight=scale_pos_weight,
            device=device, **overrides,
        )
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


# ══════════════════════════════════════════════════════════════════════════════
# Stage-3 aware feature category analysis
# ══════════════════════════════════════════════════════════════════════════════


def _resolve_stage3_inputs() -> tuple[str, str, dict, float]:
    """Read results/best_config.json and return
    (label_strategy, model_name, overrides, ref_auprc)."""
    cfg_path = RESULTS_DIR / "best_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"{cfg_path} not found. Run Stage 3 HPO first:\n"
            "  python -m bad_channel_rejection.ablation_stage3_hpo "
            "--winning-strategy <X> --winning-model <Y> --count 50"
        )
    cfg = json.loads(cfg_path.read_text())
    return (
        cfg["winning_strategy"],
        cfg["winning_model"],
        dict(cfg.get("best_config", {})),
        float(cfg.get("best_auprc_mean", 0.0)),
    )


def _categorize_features(
    feature_cols: list[str], group_patterns: dict[str, list[str]]
) -> tuple[dict[str, list[str]], list[str]]:
    """Map every feature to a category. Features matching no pattern are
    returned as `uncategorized` (the caller may add them as 'other' if it
    wants leave-one-out coverage of them)."""
    matched: set[str] = set()
    categorised: dict[str, list[str]] = {}
    for cat, patterns in group_patterns.items():
        cols = [c for c in feature_cols if any(p in c for p in patterns)]
        categorised[cat] = cols
        matched.update(cols)
    uncategorised = [c for c in feature_cols if c not in matched]
    return categorised, uncategorised


def main_stage3(
    prune_threshold: float = 0.005,
    auto_prune: bool = True,
    device: str = "cpu",
):
    """Stage-3 aware feature-category analysis + reduced-set retraining.

    1. Reads results/best_config.json for the HPO-winning model + hyperparams.
    2. Categorises every feature column via EXTENDED_GROUP_PATTERNS; any
       uncategorised feature lands in an `other` bucket so it's still probed.
    3. Runs leave-one-category-out GroupKFold(5) CV with the HPO config.
    4. If `auto_prune`, removes every category whose absence didn't hurt AUPRC
       by more than `prune_threshold`. Logs which categories are being kept
       vs removed.
    5. Trains the final reduced model on the full dataset, saves it as
       results/best_model_reduced.<ext> and writes
       results/best_config_reduced.json describing the reduced configuration.
    6. Writes a Markdown report at results/bcr_feature_category_analysis.md.

    Returns the per-condition results dataframe.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Read Stage 3 winner ────────────────────────────────────────────────
    label_strategy, model_name, overrides, ref_auprc = _resolve_stage3_inputs()
    logger.info("\n" + "=" * 60)
    logger.info("STAGE-3 FEATURE CATEGORY ANALYSIS")
    logger.info("=" * 60)
    logger.info(
        f"Model: {model_name}  |  Label strategy: {label_strategy}  |  "
        f"Stage 3 reference AUPRC: {ref_auprc:.4f}"
    )
    logger.info(f"HPO overrides: {overrides}")

    # ── 2. Load features + labels ─────────────────────────────────────────────
    out = build_feature_matrix(CSV_PATH, label_strategy=label_strategy)
    X_df = pd.DataFrame(out["X"], columns=out["feature_cols"])
    y = out["y_hard"]
    groups = out["groups"]
    spw = out["scale_pos_weight"]
    raw_w = out["sample_weights"]
    weights = raw_w if not np.allclose(raw_w, 1.0) else None

    # ── 3. Categorise ─────────────────────────────────────────────────────────
    group_cols, uncategorized = _categorize_features(
        out["feature_cols"], EXTENDED_GROUP_PATTERNS
    )
    logger.info("\nFeature categorisation:")
    for cat, cols in group_cols.items():
        logger.info(f"  {cat:<14}: {len(cols):>3} features")
    if uncategorized:
        logger.warning(
            f"  {len(uncategorized)} features did not match any pattern — "
            "added as 'other' category. Sample: "
            f"{uncategorized[:3]}"
        )
        group_cols["other"] = uncategorized

    # ── 4. Leave-one-category-out ablation ────────────────────────────────────
    logger.info("\n--- Leave-one-category-out CV ---")
    conditions: list[tuple[str, list[str] | None]] = [("A_baseline_all", None)]
    for cat, cols in group_cols.items():
        if not cols:
            continue
        keep = [c for c in out["feature_cols"] if c not in cols]
        conditions.append((f"drop_{cat}", keep))

    results = []
    for cname, keep_cols in conditions:
        r = run_condition(
            X_df, y, groups, spw, keep_cols, cname, model_name, weights,
            overrides=overrides, device=device,
        )
        results.append(r)

    df_r = pd.DataFrame(results)
    baseline_auprc = float(
        df_r.loc[df_r["condition"] == "A_baseline_all", "auprc_mean"].values[0]
    )
    df_r["delta_auprc"] = df_r["auprc_mean"] - baseline_auprc
    df_r["pct_change"] = (df_r["delta_auprc"] / baseline_auprc * 100).round(1)

    # ── 5. Auto-prune ─────────────────────────────────────────────────────────
    removable: list[tuple[str, float]] = []
    if auto_prune:
        logger.info(f"\n--- Auto-prune (threshold = -{prune_threshold:.4f}) ---")
        for _, row in df_r.iterrows():
            cond = row["condition"]
            if not cond.startswith("drop_"):
                continue
            cat = cond.replace("drop_", "")
            delta = float(row["delta_auprc"])
            if delta >= -prune_threshold:
                removable.append((cat, delta))
                logger.info(f"  REMOVE {cat:<14}  ΔAUPRC={delta:+.4f}")
            else:
                logger.info(f"  KEEP   {cat:<14}  ΔAUPRC={delta:+.4f}  (above threshold)")
    else:
        logger.info("Auto-prune disabled — no categories removed.")

    # ── 6. Build + evaluate + retrain reduced set ─────────────────────────────
    reduced_cols = list(out["feature_cols"])
    for cat, _ in removable:
        for col in group_cols[cat]:
            if col in reduced_cols:
                reduced_cols.remove(col)

    n_removed = len(out["feature_cols"]) - len(reduced_cols)
    logger.info(
        f"\nReduced feature set: {len(reduced_cols)}/{len(out['feature_cols'])} "
        f"({n_removed} removed)"
    )

    if removable:
        logger.info("\n--- Reduced-set GroupKFold evaluation ---")
        X_reduced_df = X_df[reduced_cols]
        reduced_eval = run_condition(
            X_reduced_df, y, groups, spw, reduced_cols, "REDUCED_FINAL",
            model_name, weights, overrides=overrides, device=device,
        )

        logger.info("\n--- Training reduced production model on full data ---")
        prep = FeaturePreprocessor()
        X_reduced_full = prep.fit_transform(X_reduced_df)
        final_model = create_model(
            model_name, scale_pos_weight=spw, device=device, **overrides
        )
        final_model.fit(X_reduced_full, y, sample_weight=weights)

        ext = MODEL_EXT[model_name]
        reduced_model_path = RESULTS_DIR / f"best_model_reduced.{ext}"
        final_model.save(reduced_model_path)
        logger.info(f"Reduced model saved -> {reduced_model_path}")

        reduced_config = {
            "winning_strategy":           label_strategy,
            "winning_model":              model_name,
            "best_config":                overrides,
            "n_features_full":            len(out["feature_cols"]),
            "n_features_reduced":         len(reduced_cols),
            "removed_categories":         [cat for cat, _ in removable],
            "removed_n_features":         n_removed,
            "reduced_feature_cols":       reduced_cols,
            "auprc_full_groupkfold":      baseline_auprc,
            "auprc_reduced_groupkfold":   reduced_eval["auprc_mean"],
            "auprc_reduced_std":          reduced_eval["auprc_std"],
            "auprc_delta_reduced_minus_full": (
                reduced_eval["auprc_mean"] - baseline_auprc
            ),
            "prune_threshold":            prune_threshold,
        }
        reduced_config_path = RESULTS_DIR / "best_config_reduced.json"
        reduced_config_path.write_text(json.dumps(reduced_config, indent=2))
        logger.info(f"Reduced config saved -> {reduced_config_path}")
    else:
        reduced_eval = None
        logger.info(
            "No removable categories → no reduced model produced. "
            "Use the existing best_model.<ext> for production."
        )

    # ── 7. Report ─────────────────────────────────────────────────────────────
    _write_stage3_report(
        df_r, group_cols, removable, reduced_eval,
        baseline_auprc, ref_auprc, label_strategy, model_name, overrides,
        prune_threshold, OUT_MD_STAGE3,
    )
    logger.info(f"\nReport saved -> {OUT_MD_STAGE3}")

    return df_r


def _write_stage3_report(
    df_r: pd.DataFrame,
    group_cols: dict[str, list[str]],
    removable: list[tuple[str, float]],
    reduced_eval: dict | None,
    baseline_auprc: float,
    ref_auprc: float,
    label_strategy: str,
    model_name: str,
    overrides: dict,
    prune_threshold: float,
    path: Path,
) -> None:
    """Markdown report for the Stage-3 feature-category analysis."""
    lines = [
        "# BCR Feature Category Analysis (Stage-3 aware)",
        "",
        f"**Model**: `{model_name}`  |  **Label strategy**: `{label_strategy}`  |  "
        f"**Stage 3 reference AUPRC**: {ref_auprc:.4f}",
        "",
        "## Feature categorisation",
        "",
        "| Category | # Features | Sample |",
        "|----------|------------|--------|",
    ]
    for cat, cols in group_cols.items():
        sample = ", ".join(c.strip() for c in cols[:3])
        if len(cols) > 3:
            sample += ", …"
        lines.append(f"| `{cat}` | {len(cols)} | {sample if cols else '—'} |")

    lines += [
        "",
        "## Leave-one-category-out CV",
        "",
        f"Baseline (all features): AUPRC = {baseline_auprc:.4f}",
        "",
        "| Condition | # Features | AUPRC | AUROC | ΔAUPRC | % change | Verdict |",
        "|-----------|------------|-------|-------|--------|----------|---------|",
    ]
    for _, row in df_r.iterrows():
        cond = row["condition"]
        if cond == "A_baseline_all":
            verdict = "—"
            dstr, pstr = "—", "—"
        else:
            cat = cond.replace("drop_", "")
            removed = any(c == cat for c, _ in removable)
            verdict = "🗑️ removable" if removed else "🔒 keep"
            dstr = f"{row['delta_auprc']:+.4f}"
            pstr = f"{row['pct_change']:+.1f}%"
        lines.append(
            f"| `{cond}` | {row['n_features']} "
            f"| {row['auprc_mean']:.4f} ± {row['auprc_std']:.4f} "
            f"| {row['auroc_mean']:.4f} "
            f"| {dstr} | {pstr} | {verdict} |"
        )

    lines += [
        "",
        f"Auto-prune threshold: ΔAUPRC ≥ -{prune_threshold:.4f} → category is removable.",
        "",
        "## Reduced feature set",
        "",
    ]
    if removable:
        n_full = sum(len(v) for v in group_cols.values())
        n_red  = n_full - sum(len(group_cols[c]) for c, _ in removable)
        lines += [
            f"**Removed categories**: " +
            ", ".join(f"`{c}`" for c, _ in removable),
            "",
            f"**Reduced feature count**: {n_red} / {n_full}",
            "",
        ]
        if reduced_eval is not None:
            delta_red = reduced_eval["auprc_mean"] - baseline_auprc
            lines += [
                "### Reduced-set evaluation (GroupKFold(5))",
                "",
                "| Metric | Full model | Reduced model | Δ |",
                "|--------|-----------:|--------------:|:--|",
                (
                    f"| AUPRC  | {baseline_auprc:.4f} "
                    f"| {reduced_eval['auprc_mean']:.4f} ± "
                    f"{reduced_eval['auprc_std']:.4f} "
                    f"| {delta_red:+.4f} |"
                ),
                (
                    f"| AUROC  | (see Stage 3) "
                    f"| {reduced_eval['auroc_mean']:.4f} | — |"
                ),
                "",
                "**Production artefacts**:",
                "- `results/best_model_reduced.<ext>` — retrained on full data with the HPO config",
                "- `results/best_config_reduced.json` — feature list + comparison metrics",
                "",
                "### Recommendation",
                "",
            ]
            if abs(delta_red) <= prune_threshold:
                lines.append(
                    "✅ Reduced model performs within noise of the full model — "
                    "**ship the reduced model** for faster inference and a simpler feature pipeline."
                )
            elif delta_red > 0:
                lines.append(
                    "🎉 Reduced model **outperforms** the full model — likely the removed "
                    "categories were adding noise. Ship the reduced model."
                )
            else:
                lines.append(
                    "⚠️ Reduced model lost more AUPRC than the threshold suggested. "
                    "Re-run with a stricter `--prune-threshold` or inspect which "
                    "category is responsible (drop them one at a time)."
                )
    else:
        lines += [
            "No category was removable under the current threshold "
            f"({prune_threshold:.4f}). Every feature category contributes "
            "to the full model's AUPRC. **Use `results/best_model.<ext>` "
            "for production.**",
        ]

    lines += [
        "",
        "## HPO config used",
        "",
        "```json",
        json.dumps(overrides, indent=2),
        "```",
    ]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-strategy", default="hard_threshold")
    parser.add_argument("--model", default="xgboost")
    parser.add_argument(
        "--mode",
        default="group",
        choices=["group", "engineered", "stage3"],
        help=(
            "'group': legacy 6-condition A-F feature-group ablation (default). "
            "'engineered': 5-condition engineered-transform ablation. "
            "'stage3': Stage-3 HPO-aware feature-category analysis with "
            "auto-prune + reduced-model retrain."
        ),
    )
    parser.add_argument(
        "--prune-threshold",
        type=float,
        default=0.005,
        help="Stage-3 mode only. ΔAUPRC threshold: a category is removable if "
             "leaving it out costs less AUPRC than this (default 0.005).",
    )
    parser.add_argument(
        "--no-auto-prune",
        action="store_true",
        help="Stage-3 mode only. Disable auto-prune (just report; no retrain).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Stage-3 mode only. Device override for create_model().",
    )
    args = parser.parse_args()
    if args.mode == "stage3":
        main_stage3(
            prune_threshold=args.prune_threshold,
            auto_prune=not args.no_auto_prune,
            device=args.device,
        )
    elif args.mode == "engineered":
        main_engineered(
            label_strategy=args.label_strategy, model_name=args.model
        )
    else:
        main(label_strategy=args.label_strategy, model_name=args.model)
