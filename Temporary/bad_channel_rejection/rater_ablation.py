"""bad_channel_rejection/rater_ablation.py

Rater noise ablation: compare AUPRC trained with all-rater consensus
vs. consensus excluding Site 3.

Site 3 has r ≈ 0.10 with all other rater sites — we test whether
its vote degrades label quality.

Run: python -m bad_channel_rejection.rater_ablation
"""
import os
import numpy as np
import pandas as pd
import wandb
from dotenv import load_dotenv
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

from bad_channel_rejection.dataset import build_feature_matrix, load_bcr_data
from bad_channel_rejection.features import FeaturePreprocessor

load_dotenv()

# Force W&B offline — avoids network timeout on restricted connections
os.environ["WANDB_MODE"] = "offline"

DATA_PATH     = "data/raw/Bad_channels_for_ML.csv"
RESULTS_DIR   = "results"
N_FOLDS       = 5
RANDOM_STATE  = 42
WANDB_PROJECT = "eeg-bcr"

RATER_COLS_NO_SITE3 = ["Bad (site 2)", "Bad (site 4a)", "Bad (site 4b)"]

XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "eval_metric": "aucpr",
    "early_stopping_rounds": 30,
    "device": "cpu",
    "random_state": RANDOM_STATE,
    "verbosity": 0,
}

os.makedirs(RESULTS_DIR, exist_ok=True)


def load_features():
    """Load and preprocess feature matrix. Returns X (np.ndarray), groups, feature_cols."""
    X_raw, y_all4, groups, feature_cols, scale_pos_weight, sample_weights = build_feature_matrix(
        DATA_PATH, bad_threshold=2
    )
    preprocessor = FeaturePreprocessor()
    X = preprocessor.fit_transform(pd.DataFrame(X_raw, columns=feature_cols))
    print(f"[ablation] X={X.shape}, bad_rate={y_all4.mean():.4f}")
    return X, y_all4, groups, scale_pos_weight, sample_weights


def build_targets_no_site3(df: pd.DataFrame) -> np.ndarray:
    """Recompute bad label using only sites 2, 4a, 4b (excluding site 3).
    Majority vote of 3 raters: >= 2 out of 3 = bad.
    """
    score_no3 = df[RATER_COLS_NO_SITE3].sum(axis=1)
    y_no3 = (score_no3 >= 2).astype(int)
    print(f"  No-site3 bad rate: {y_no3.mean():.4f} "
          f"({y_no3.sum()} bad / {len(y_no3)} total)")
    return y_no3.values


def run_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    scale_pos_weight: float,
    label: str,
    sample_weights: np.ndarray,
) -> dict:
    gkf = GroupKFold(n_splits=N_FOLDS)
    auprcs, aurocs = [], []

    for fold_idx, (train_idx, val_idx) in enumerate(
        gkf.split(X, y, groups), start=1
    ):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        sw_tr = sample_weights[train_idx]

        model = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            **XGB_PARAMS
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            sample_weight=sw_tr,
            verbose=False,
        )

        y_prob = model.predict_proba(X_val)[:, 1]
        auprc = average_precision_score(y_val, y_prob)
        auroc = roc_auc_score(y_val, y_prob)
        auprcs.append(auprc)
        aurocs.append(auroc)
        print(f"    [{label}] fold {fold_idx}: AUPRC={auprc:.4f}  AUROC={auroc:.4f}")

    return {
        "label": label,
        "auprc_mean": np.mean(auprcs),
        "auprc_std": np.std(auprcs),
        "auroc_mean": np.mean(aurocs),
        "auroc_std": np.std(aurocs),
    }


def _write_report(res_all4: dict, res_no3: dict, delta: float):
    finding = (
        "Site 3 **is** degrading label quality (ΔAUPRC > 0.05)."
        if delta > 0.05
        else "Site 3 does **not** substantially degrade label quality."
    )
    report = f"""# Rater Noise Ablation Report

## Setup
- Feature matrix: fixed (same X for both conditions)
- Condition A: bad label = Bad (score) >= 2 (all 4 rater sites)
- Condition B: bad label recomputed from sites 2, 4a, 4b only (site 3 excluded, majority vote >= 2/3)
- CV: GroupKFold(5) by subject_id

## Results

| Condition          | AUPRC mean | AUPRC std | AUROC mean |
|--------------------|------------|-----------|------------|
| All 4 raters       | {res_all4['auprc_mean']:.4f}     | {res_all4['auprc_std']:.4f}    | {res_all4['auroc_mean']:.4f}     |
| No site 3          | {res_no3['auprc_mean']:.4f}     | {res_no3['auprc_std']:.4f}    | {res_no3['auroc_mean']:.4f}     |
| **ΔAUPRC**         | **{delta:+.4f}**     |           |            |

## Finding
{finding}

Site 3 inter-rater correlation with all other sites: r ≈ 0.10 (confirmed in EDA).
For context, sites 4a<->4b show r = 0.42. Site 3 is a statistical outlier in rater agreement.

## Interpretation for report
This is a **methods note**, not a model failure. The label noise ceiling
limits maximum achievable AUPRC regardless of model quality.
All primary results use the full consensus score (Condition A) for reproducibility.
"""
    path = f"{RESULTS_DIR}/rater_noise_ablation.md"
    with open(path, "w") as f:
        f.write(report)
    print(f"[ablation] Report saved → {path}")


def main():
    run = wandb.init(
        project=WANDB_PROJECT,
        name="bcr_rater_ablation",
        tags=["ablation", "rater_noise"],
        settings=wandb.Settings(init_timeout=120),
    )

    # Load feature matrix (X) — same for both conditions
    X, y_all4, groups, spw_all4, sample_weights = load_features()

    # Build no-site-3 target from raw dataframe
    raw_df = load_bcr_data(DATA_PATH)
    y_no3 = build_targets_no_site3(raw_df)
    spw_no3 = float((y_no3 == 0).sum() / (y_no3 == 1).sum())

    print(f"\nscale_pos_weight all-4 raters: {spw_all4:.2f}")
    print(f"scale_pos_weight no-site3:     {spw_no3:.2f}")

    print("\n--- Condition A: all 4 raters ---")
    res_all4 = run_cv(X, y_all4, groups, spw_all4, "all_raters", sample_weights)

    print("\n--- Condition B: no site 3 ---")
    res_no3  = run_cv(X, y_no3,  groups, spw_no3,  "no_site3",  sample_weights)

    delta = res_no3["auprc_mean"] - res_all4["auprc_mean"]

    print(f"\n=== RATER ABLATION RESULT ===")
    print(f"  All 4 raters: AUPRC = {res_all4['auprc_mean']:.4f} ± {res_all4['auprc_std']:.4f}")
    print(f"  No site 3:    AUPRC = {res_no3['auprc_mean']:.4f} ± {res_no3['auprc_std']:.4f}")
    print(f"  ΔAUPRC = {delta:+.4f}")
    if delta > 0.05:
        print("  → Site 3 IS degrading labels (ΔAUPRC > 0.05). Document as label noise finding.")
    elif delta > 0:
        print("  → Small positive delta: site 3 adds mild noise but effect is minor.")
    else:
        print("  → No improvement: site 3 is not uniquely harmful. Dataset label quality is consistent.")

    # W&B
    wandb.summary.update({
        "all_raters_auprc": res_all4["auprc_mean"],
        "no_site3_auprc":   res_no3["auprc_mean"],
        "delta_auprc": delta,
        "site3_degrades_labels": bool(delta > 0.05),
    })
    wandb.finish()

    _write_report(res_all4, res_no3, delta)


if __name__ == "__main__":
    main()
