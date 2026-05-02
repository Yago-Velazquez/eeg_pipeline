"""
bad_channel_rejection/shap_analysis.py

SHAP analysis on the trained BCR model.  Supports XGBoost, LightGBM, CatBoost
via SHAP's TreeExplainer.  XGBoost uses native pred_contribs to avoid base_score
compat issues; others use standard TreeExplainer.

Two modes
---------
1. Stage 3 winner (recommended after running ablation_stage3_hpo.py):

       python -m bad_channel_rejection.shap_analysis --from-stage3

   Reads `results/best_config.json` to discover the winning model name and
   label strategy, then explains `results/best_model.<ext>`.

2. Tagged train.py output (legacy):

       python -m bad_channel_rejection.shap_analysis \\
           --label-strategy mace --model lightgbm
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import wandb
from dotenv import load_dotenv

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import build_run_tag
from .dataset import build_feature_matrix
from .features import FeaturePreprocessor, preprocess_fold
from .logging_config import setup_logging
from .models import MODEL_EXT

load_dotenv()
os.environ.setdefault("WANDB_MODE", "offline")

logger = setup_logging(__name__)

DATA_PATH = "data/raw/Bad_channels_for_ML.csv"
FIGURES_DIR = Path("results/figures")
RESULTS_DIR = Path("results")
MAX_SHAP_SAMPLES = 2000

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _xgb_shap(model_path: str, X_sub: np.ndarray, feature_names: list[str]):
    """XGBoost: use native pred_contribs after patching base_score format."""
    import xgboost as xgb

    with open(model_path) as f:
        model_json = json.load(f)
    lmp = model_json["learner"]["learner_model_param"]
    raw = lmp.get("base_score", "")
    if isinstance(raw, str) and raw.startswith("[") and raw.endswith("]"):
        lmp["base_score"] = raw.strip("[]")
        logger.info(f"Patched base_score: {raw!r} -> {lmp['base_score']!r}")

    with tempfile.NamedTemporaryFile(
        suffix=".json", delete=False, mode="w"
    ) as f:
        tmp_path = f.name
        json.dump(model_json, f)

    booster = xgb.Booster()
    booster.load_model(tmp_path)
    os.unlink(tmp_path)

    dmatrix = xgb.DMatrix(X_sub, feature_names=feature_names)
    contribs = booster.predict(dmatrix, pred_contribs=True)
    return contribs[:, :-1], contribs[:, -1]


def _generic_shap(model, X_sub: np.ndarray, feature_names: list[str]):
    """LightGBM / CatBoost: standard TreeExplainer."""
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sub)
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]
    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)) and len(np.atleast_1d(base)) > 1:
        base = base[1]
    return shap_values, np.full(len(X_sub), float(base))


def _resolve_stage3_inputs() -> tuple[str, str, Path, str]:
    """Read `results/best_config.json` and return (label_strategy, model_name,
    model_path, tag) for the Stage 3 HPO winner."""
    cfg_path = RESULTS_DIR / "best_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"{cfg_path} not found. Run Stage 3 HPO first:\n"
            "  python -m bad_channel_rejection.ablation_stage3_hpo "
            "--winning-strategy <X> --winning-model <Y> --count 50"
        )
    cfg = json.loads(cfg_path.read_text())
    label_strategy = cfg["winning_strategy"]
    model_name     = cfg["winning_model"]
    model_path     = RESULTS_DIR / f"best_model.{MODEL_EXT[model_name]}"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Re-run Stage 3 HPO."
        )
    return label_strategy, model_name, model_path, "best"


def main(
    label_strategy: str | None = None,
    model_name: str | None = None,
    use_engineered_features: bool = False,
    use_impedance_interactions: bool = False,
    from_stage3: bool = False,
):
    import shap

    if from_stage3:
        # Stage 3 winner — discover everything from results/best_config.json.
        label_strategy, model_name, model_path, tag = _resolve_stage3_inputs()
        engineering_kwargs = None
        per_fold_mode = False   # Stage 3 used plain FeaturePreprocessor
        logger.info(
            f"Stage 3 mode: explaining {model_name} "
            f"(label={label_strategy})  model_path={model_path}"
        )
    else:
        if label_strategy is None or model_name is None:
            raise ValueError(
                "label_strategy and model_name are required when "
                "from_stage3=False"
            )
        if use_impedance_interactions:
            engineering_kwargs = {
                "use_channel_bad_rate": False,
                "use_spatial_pruner": False,
                "use_impedance_interactions": True,
            }
            per_fold_mode = True
        else:
            engineering_kwargs = None
            per_fold_mode = use_engineered_features

        tag = build_run_tag(
            label_strategy,
            model_name,
            use_engineered_features=use_engineered_features,
            use_impedance_interactions=use_impedance_interactions,
        )
        model_path = RESULTS_DIR / f"bcr_model_{tag}.{MODEL_EXT[model_name]}"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}. Run train.py first."
            )

    wandb.init(
        project="eeg-bcr",
        name=f"bcr_shap_{tag}",
        tags=["shap", tag],
        settings=wandb.Settings(init_timeout=120),
    )

    out = build_feature_matrix(DATA_PATH, label_strategy=label_strategy)
    raw_df = pd.DataFrame(out["X"], columns=out["feature_cols"])
    if per_fold_mode:
        X, _, feature_names, _, _ = preprocess_fold(
            raw_df, raw_df, out["y_hard"],
            channel_labels_tr=out["channel_labels"],
            channel_labels_va=out["channel_labels"],
            use_engineered_features=True,
            engineering_kwargs=engineering_kwargs,
        )
    else:
        prep = FeaturePreprocessor()
        X = prep.fit_transform(raw_df)
        feature_names = prep.feature_names_out_

    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), size=min(MAX_SHAP_SAMPLES, len(X)), replace=False)
    X_sub = X[idx]

    logger.info(f"Computing SHAP for {len(X_sub)} samples ({model_name})")
    if model_name == "xgboost":
        shap_vals, base_values = _xgb_shap(str(model_path), X_sub, feature_names)
    else:
        from .models import create_model
        m = create_model(model_name, scale_pos_weight=1.0).load(model_path)
        shap_vals, base_values = _generic_shap(m._model, X_sub, feature_names)

    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    sorted_idx = np.argsort(mean_abs_shap)[::-1]
    top20_idx = sorted_idx[:20]
    top20_names = [feature_names[i] for i in top20_idx]
    top20_values = mean_abs_shap[top20_idx]

    logger.info("Top-20 features by mean |SHAP|:")
    for rank, (name, val) in enumerate(zip(top20_names, top20_values), 1):
        flag = (
            " <-- impedance_missing"
            if "impedance_missing" in name.lower() else ""
        )
        logger.info(f"  {rank:2d}. {name:<55} {val:.5f}{flag}")

    imp_rank = None
    for rank, i in enumerate(sorted_idx, 1):
        if "impedance_missing" in feature_names[i].lower():
            imp_rank = rank
            break
    if imp_rank is not None:
        logger.info(
            f"impedance_missing rank: #{imp_rank}  "
            f"(top-10: {'YES' if imp_rank <= 10 else 'NO'})"
        )

    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_vals[:, top20_idx],
        X_sub[:, top20_idx],
        feature_names=top20_names,
        show=False,
        max_display=20,
    )
    beeswarm_path = FIGURES_DIR / f"bcr_shap_beeswarm_{tag}.png"
    plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = [
        "#ffa657" if "impedance" in n.lower() else "#58a6ff"
        for n in top20_names
    ]
    ax.barh(range(len(top20_names)), top20_values, color=colors)
    ax.set_yticks(range(len(top20_names)))
    ax.set_yticklabels(top20_names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"BCR — Top-20 Features ({tag})")
    fig.tight_layout()
    bar_path = FIGURES_DIR / f"bcr_shap_bar_{tag}.png"
    fig.savefig(bar_path, dpi=150)
    plt.close(fig)

    df_top = pd.DataFrame({
        "rank": range(1, len(top20_names) + 1),
        "feature": top20_names,
        "mean_abs_shap": top20_values,
        "is_impedance": [
            "impedance" in n.lower() for n in top20_names
        ],
    })
    csv_path = RESULTS_DIR / f"bcr_shap_top20_{tag}.csv"
    df_top.to_csv(csv_path, index=False)
    logger.info(f"Saved -> {beeswarm_path}, {bar_path}, {csv_path}")

    wandb.log({
        "shap_beeswarm": wandb.Image(str(beeswarm_path)),
        "shap_bar": wandb.Image(str(bar_path)),
    })
    if imp_rank is not None:
        wandb.summary["impedance_missing_rank"] = imp_rank
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from-stage3",
        action="store_true",
        help="Explain the Stage 3 HPO winner (results/best_model.<ext> "
             "+ results/best_config.json). Overrides --label-strategy / --model.",
    )
    parser.add_argument("--label-strategy", default="hard_threshold")
    parser.add_argument("--model", default="xgboost")
    parser.add_argument(
        "--use-engineered-features",
        action="store_true",
        help="DEPRECATED — explain the _fe model (all three transforms).",
    )
    parser.add_argument(
        "--use-impedance-interactions",
        action="store_true",
        help="Explain the _impedance_ix model (impedance interactions only).",
    )
    args = parser.parse_args()
    if args.from_stage3:
        main(from_stage3=True)
    else:
        main(
            args.label_strategy,
            args.model,
            use_engineered_features=args.use_engineered_features,
            use_impedance_interactions=args.use_impedance_interactions,
        )
