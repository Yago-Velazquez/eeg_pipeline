"""bad_channel_rejection/shap_analysis.py

SHAP-style analysis on trained BCR XGBoost model.
Uses XGBoost's native pred_contribs=True to avoid SHAP/XGBoost
base_score compatibility issues.

Produces beeswarm plot and bar chart of top-20 features.

Run: python -m bad_channel_rejection.shap_analysis
"""
import os
import json
import tempfile
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb
from dotenv import load_dotenv
from xgboost import Booster, DMatrix

from bad_channel_rejection.dataset import build_feature_matrix
from bad_channel_rejection.features import FeaturePreprocessor

load_dotenv()

# Force W&B offline — avoids network timeout on restricted connections
os.environ["WANDB_MODE"] = "offline"

DATA_PATH        = "data/raw/Bad_channels_for_ML.csv"
MODEL_PATH       = "results/bcr_model_thresh2.json"
FIGURES_DIR      = "results/figures"
RESULTS_DIR      = "results"
WANDB_PROJECT    = "eeg-bcr"
MAX_SHAP_SAMPLES = 2000  # subsample for speed on M2

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_patched_booster(model_path: str) -> Booster:
    """Load XGBoost model and patch base_score for SHAP compatibility.

    XGBoost 2.x stores base_score as '[5E-1]' in the model JSON.
    SHAP's XGBTreeModelLoader calls float() on that string and crashes.
    Fix: read the JSON file, strip brackets, write to temp file, reload.
    save_config/load_config does NOT work — SHAP reads from the model
    binary, not the runtime config.
    """
    with open(model_path) as f:
        model_json = json.load(f)

    lmp = model_json["learner"]["learner_model_param"]
    raw = lmp.get("base_score", "")
    if isinstance(raw, str) and raw.startswith("[") and raw.endswith("]"):
        lmp["base_score"] = raw.strip("[]")
        print(f"[shap] Patched base_score: {repr(raw)} → {repr(lmp['base_score'])}")

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        tmp_path = f.name
        json.dump(model_json, f)

    booster = Booster()
    booster.load_model(tmp_path)
    os.unlink(tmp_path)
    return booster


def load_data():
    """Load + preprocess feature matrix.

    Returns
    -------
    X              : np.ndarray  (18900, 138)
    y              : np.ndarray  (18900,)
    feature_names  : list[str]   138 post-preprocessing column names
                     (exact names from preprocessor.feature_names_out_,
                      so impedance_missing and all survivors are labelled correctly)
    """
    X_raw, y, groups, feature_cols, scale_pos_weight = build_feature_matrix(
        DATA_PATH,
        save_cols_to="results/feature_columns.json",
        bad_threshold=2,
    )
    preprocessor = FeaturePreprocessor()
    X = preprocessor.fit_transform(pd.DataFrame(X_raw, columns=feature_cols))

    # Use the preprocessor's own record of surviving column names —
    # this is the only correct source after drops + VarianceThreshold.
    feature_names = preprocessor.feature_names_out_

    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    print(f"[shap] X={X.shape}, bad_rate={y.mean():.4f}")
    print(f"[shap] feature_names count: {len(feature_names)} "
          f"({'matches X cols ✓' if len(feature_names) == X.shape[1] else 'MISMATCH ✗'})")

    # Confirm impedance_missing survived
    imp_in_names = any("impedance_missing" in n.lower() for n in feature_names)
    print(f"[shap] impedance_missing in feature names: {'YES ✓' if imp_in_names else 'NO ✗'}")

    return X, y, feature_names


def compute_shap_values_native(
    booster: Booster, X_sub: np.ndarray, feature_names: list
) -> tuple[np.ndarray, np.ndarray]:
    """Compute SHAP values using XGBoost native pred_contribs=True.

    Returns
    -------
    shap_vals   : (n_samples, n_features)
    base_values : (n_samples,)
    """
    dmatrix = DMatrix(X_sub, feature_names=feature_names)
    contribs = booster.predict(dmatrix, pred_contribs=True)
    # Last column is the bias / base value
    return contribs[:, :-1], contribs[:, -1]


def main():
    wandb.init(
        project=WANDB_PROJECT,
        name="bcr_shap",
        tags=["shap", "interpretability"],
        settings=wandb.Settings(init_timeout=120),
    )

    # ── Load data + correct feature names ─────────────────────────────────────
    X, y, feature_names = load_data()

    # ── Load model ────────────────────────────────────────────────────────────
    booster = load_patched_booster(MODEL_PATH)

    # ── Subsample ─────────────────────────────────────────────────────────────
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), size=min(MAX_SHAP_SAMPLES, len(X)), replace=False)
    X_sub = X[idx]

    # ── Compute SHAP values ───────────────────────────────────────────────────
    print(f"[shap] Computing SHAP values for {len(X_sub)} samples...")
    shap_vals, base_values = compute_shap_values_native(booster, X_sub, feature_names)
    print(f"[shap] SHAP values shape: {shap_vals.shape}")

    # ── Top-20 by mean |SHAP| ─────────────────────────────────────────────────
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    sorted_idx    = np.argsort(mean_abs_shap)[::-1]
    top20_idx     = sorted_idx[:20]
    top20_names   = [feature_names[i] for i in top20_idx]
    top20_values  = mean_abs_shap[top20_idx]

    print("\n[shap] Top-20 features by mean |SHAP|:")
    for rank, (name, val) in enumerate(zip(top20_names, top20_values), 1):
        flag = " <-- impedance_missing!" if "impedance_missing" in name.lower() else ""
        print(f"  {rank:2d}. {name:<55} {val:.5f}{flag}")

    # Global rank of impedance_missing
    imp_rank = None
    for rank, i in enumerate(sorted_idx, 1):
        if "impedance_missing" in feature_names[i].lower():
            imp_rank = rank
            break

    if imp_rank is not None:
        print(f"\n  ✓ impedance_missing global rank: #{imp_rank}  "
              f"(in top-10: {'YES' if imp_rank <= 10 else 'NO'})")
    else:
        print("\n  ✗ impedance_missing not found — check FeaturePreprocessor output")

    # ── Plot 1: Beeswarm ──────────────────────────────────────────────────────
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_vals[:, top20_idx],
        X_sub[:, top20_idx],
        feature_names=top20_names,
        show=False,
        max_display=20,
    )
    beeswarm_path = f"{FIGURES_DIR}/bcr_shap_beeswarm.png"
    plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[shap] Beeswarm saved → {beeswarm_path}")

    # ── Plot 2: Bar chart ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#ffa657" if "impedance" in n.lower() else "#58a6ff"
              for n in top20_names]
    ax.barh(range(len(top20_names)), top20_values, color=colors)
    ax.set_yticks(range(len(top20_names)))
    ax.set_yticklabels(top20_names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("BCR — Top-20 Features by SHAP Importance\n"
                 "(orange = impedance features)")
    fig.tight_layout()
    bar_path = f"{FIGURES_DIR}/bcr_shap_bar.png"
    fig.savefig(bar_path, dpi=150)
    plt.close(fig)
    print(f"[shap] Bar chart saved → {bar_path}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    df_top = pd.DataFrame({
        "rank":          range(1, len(top20_names) + 1),
        "feature":       top20_names,
        "mean_abs_shap": top20_values,
        "is_impedance":  ["impedance" in n.lower() for n in top20_names],
    })
    csv_path = f"{RESULTS_DIR}/bcr_shap_top20.csv"
    df_top.to_csv(csv_path, index=False)
    print(f"[shap] Top-20 CSV saved → {csv_path}")

    # ── W&B ───────────────────────────────────────────────────────────────────
    wandb.log({
        "shap_beeswarm": wandb.Image(beeswarm_path),
        "shap_bar":      wandb.Image(bar_path),
    })
    if imp_rank is not None:
        wandb.summary["impedance_missing_shap_rank"] = imp_rank
        wandb.summary["impedance_missing_in_top10"]  = imp_rank <= 10

    wandb.finish()
    print("\n[shap] Done.")


if __name__ == "__main__":
    main()
