"""
bad_channel_rejection/feature_ablation.py
──────────────────────────────────────────
Feature-group ablation study for BCR.

Runs 6 XGBoost CV conditions (A–F) and logs to W&B.
Writes results/bcr_feature_ablation.md.

Usage:
    python -m bad_channel_rejection.feature_ablation
"""

import json, os, time
import numpy as np
import pandas as pd
import wandb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

from .dataset import build_feature_matrix
from .features import FeaturePreprocessor

# ── Paths ──────────────────────────────────────────────────────────
CSV_PATH = "data/raw/Bad_channels_for_ML.csv"
RESULTS_DIR = "results"
OUT_MD = "results/bcr_feature_ablation.md"

# ── Feature group definitions (patterns match column substrings) ───
GROUP_PATTERNS = {
    "impedance":     ["impedance_missing"],
    "frequency":     [" Median frequency", " First quartile frequency", " Third quartile frequency"],
    "spatial":       [" Correllation with neighbors", " Correllation with second-degree", " Global correlation"],
    "decomposition": [" PCA", " ICA", " residuals", " reconstruction", " Kurtosis",
                      " Low gamma", " independence", " unmixing"],
}

def get_group_cols(feature_cols: list[str], patterns: list[str]) -> list[str]:
    """Return column names matching any pattern in the list."""
    return [c for c in feature_cols if any(p in c for p in patterns)]


def run_condition(
    X_df: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    scale_pos_weight: float,
    keep_cols: list[str] | None,   # None = use all
    condition_name: str,
) -> dict:
    """
    Run one GroupKFold(5) CV with the specified column subset.
    Returns dict with mean AUPRC, AUROC across folds.
    """
    if keep_cols is not None:
        X_df = X_df[keep_cols]

    gkf = GroupKFold(n_splits=5)
    fold_auprc, fold_auroc = [], []

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_df, y, groups)):
        X_tr, X_val = X_df.iloc[tr_idx], X_df.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        prep = FeaturePreprocessor()
        X_tr_np = prep.fit_transform(X_tr)
        X_val_np = prep.transform(X_val)

        # Guard: if only 1 feature remains after preprocessing, skip variance threshold
        if X_tr_np.shape[1] == 0:
            print(f"  [WARN] fold {fold}: 0 features after preprocessing — skipping")
            continue

        xgb = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            early_stopping_rounds=30,
            tree_method="hist",
            device="cpu",
            verbosity=0,
            random_state=42,
        )
        xgb.fit(X_tr_np, y_tr, eval_set=[(X_val_np, y_val)], verbose=False)

        y_prob = xgb.predict_proba(X_val_np)[:, 1]
        fold_auprc.append(average_precision_score(y_val, y_prob))
        fold_auroc.append(roc_auc_score(y_val, y_prob))

    result = {
        "condition":   condition_name,
        "n_features":  X_df.shape[1] if keep_cols is None else len(keep_cols),
        "auprc_mean":  float(np.mean(fold_auprc)),
        "auprc_std":   float(np.std(fold_auprc)),
        "auroc_mean":  float(np.mean(fold_auroc)),
        "auroc_std":   float(np.std(fold_auroc)),
    }
    print(f"  [{condition_name}] AUPRC={result['auprc_mean']:.4f} ± {result['auprc_std']:.4f}  "
          f"AUROC={result['auroc_mean']:.4f} ({X_df.shape[1]} features)")
    return result


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Load data ───────────────────────────────────────────────────
    print("Loading feature matrix …")
    X_raw, y, groups, feature_cols, scale_pos_weight = build_feature_matrix(
        CSV_PATH, save_cols_to=None, bad_threshold=2
    )
    X_df = pd.DataFrame(X_raw, columns=feature_cols)
    print(f"  Shape: {X_df.shape}  |  bad rate: {y.mean():.3%}  |  spw: {scale_pos_weight:.1f}")

    # ── Derive group column lists ────────────────────────────────────
    group_cols = {}
    for gname, patterns in GROUP_PATTERNS.items():
        group_cols[gname] = get_group_cols(feature_cols, patterns)
        print(f"  Group [{gname}]: {len(group_cols[gname])} features")

    all_grouped = set(c for cols in group_cols.values() for c in cols)
    other_cols = [c for c in feature_cols if c not in all_grouped]
    print(f"  Other (StdDev + channel_label_enc etc.): {len(other_cols)} features")

    # ── Build condition list ─────────────────────────────────────────
    conditions = [
        ("A_all_features",      None),
        ("B_no_impedance",      [c for c in feature_cols if c not in group_cols["impedance"]]),
        ("C_no_frequency",      [c for c in feature_cols if c not in group_cols["frequency"]]),
        ("D_no_spatial",        [c for c in feature_cols if c not in group_cols["spatial"]]),
        ("E_no_decomposition",  [c for c in feature_cols if c not in group_cols["decomposition"]]),
        ("F_impedance_only",    ["impedance_missing"]),
    ]

    # ── W&B init ─────────────────────────────────────────────────────
    wandb.init(project="eeg-bcr", name="day12_feature_ablation", config={
        "experiment": "feature_group_ablation",
        "n_conditions": len(conditions),
        "bad_threshold": 2,
        "cv_folds": 5,
    })

    # ── Run all conditions ───────────────────────────────────────────
    results = []
    t0 = time.time()
    for cname, keep_cols in conditions:
        print(f"\nRunning condition: {cname}")
        r = run_condition(X_df, y, groups, scale_pos_weight, keep_cols, cname)
        results.append(r)
        wandb.log({
            f"{cname}/auprc": r["auprc_mean"],
            f"{cname}/auroc": r["auroc_mean"],
            f"{cname}/n_features": r["n_features"],
        })

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/60:.1f} min")
    wandb.log({"total_time_min": elapsed/60})
    wandb.finish()

    # ── Save results ────────────────────────────────────────────────
    df_r = pd.DataFrame(results)
    baseline_auprc = df_r.loc[df_r["condition"] == "A_all_features", "auprc_mean"].values[0]
    df_r["delta_auprc"] = df_r["auprc_mean"] - baseline_auprc
    df_r["pct_change"] = (df_r["delta_auprc"] / baseline_auprc * 100).round(1)

    _write_report(df_r, OUT_MD)
    print(f"\nReport written → {OUT_MD}")

    # ── Print table to terminal ──────────────────────────────────────
    print("\n" + "="*70)
    print(df_r[["condition", "n_features", "auprc_mean", "auprc_std", "auroc_mean", "delta_auprc"]].to_string(index=False))
    print("="*70)


def _write_report(df: pd.DataFrame, path: str):
    baseline = df.loc[df["condition"] == "A_all_features"].iloc[0]
    lines = [
        "# BCR Feature Group Ablation", "",
        "Generated by `bad_channel_rejection/feature_ablation.py` — Day 12", "",
        "## Results Table", "",
        "| Condition | # Features | AUPRC (mean±std) | AUROC (mean±std) | ΔAUPRC | % change |",
        "|---|---|---|---|---|---|",
    ]
    for _, row in df.iterrows():
        delta_str = f"{row['delta_auprc']:+.4f}" if row["condition"] != "A_all_features" else "—"
        pct_str   = f"{row['pct_change']:+.1f}%"   if row["condition"] != "A_all_features" else "—"
        lines.append(
            f"| {row['condition']} | {row['n_features']} "
            f"| {row['auprc_mean']:.4f} ± {row['auprc_std']:.4f} "
            f"| {row['auroc_mean']:.4f} ± {row['auroc_std']:.4f} "
            f"| {delta_str} | {pct_str} |"
        )
    lines += [
        "", "## Interpretation", "",
        f"- **Baseline (all features):** AUPRC = {baseline['auprc_mean']:.4f} ± {baseline['auprc_std']:.4f}",
        f"- **Random baseline:** AUPRC ≈ 0.039 (3.9% bad rate)",
        "- *(Fill in interpretation once numbers are available.)*",
        "", "## Minimum Viable Feature Set", "",
        "*(Fill in after reviewing ΔAUPRC values.)*",
        "", "## Notes", "",
        "- Each condition re-fits `FeaturePreprocessor` independently per fold (no leakage).",
        "- `scale_pos_weight` is fixed at the dataset-level value (not recomputed per fold).",
        "- All runs use XGBoost with early stopping (rounds=30) to prevent overfitting in low-feature conditions.",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
