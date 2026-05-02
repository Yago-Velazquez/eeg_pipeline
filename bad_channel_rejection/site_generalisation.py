"""
bad_channel_rejection/site_generalisation.py

Leave-one-out generalisation across either visits or subjects.

Two modes
---------
- LOVO (leave-one-visit-out)    : 4 folds — train on 3 visits, test on the 4th.
                                  Tests temporal robustness.
- LOSO (leave-one-subject-out)  : 43 folds — each subject held out once.
                                  Gold-standard "will this work on a new subject?"
                                  number for the thesis. Higher statistical power
                                  than LOVO at the cost of compute.

Stage 3 integration
-------------------
With ``--from-stage3``, the script reads ``results/best_config.json`` and uses
the HPO-winning hyperparameters for every fold. Without it, default model
params are used (legacy mode for train.py outputs).

Run
---
    # LOVO with HPO winner
    python -m bad_channel_rejection.site_generalisation --from-stage3 --mode lovo

    # LOSO (slower; this is the unbiased number for the thesis)
    python -m bad_channel_rejection.site_generalisation --from-stage3 --mode loso

    # Both in one run
    python -m bad_channel_rejection.site_generalisation --from-stage3 --mode both

    # Legacy (no HPO config)
    python -m bad_channel_rejection.site_generalisation \\
        --label-strategy mace --model lightgbm --mode lovo
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from . import build_run_tag
from .dataset import (
    NON_FEATURE_COLS,
    add_missingness_flags,
    impute_and_encode_channels,
    load_bcr_data,
)
from .features import FeaturePreprocessor
from .label_quality import build_label_artifacts
from .logging_config import setup_logging
from .models import create_model

warnings.filterwarnings("ignore")
logger = setup_logging(__name__)

BCR_CSV = "data/raw/Bad_channels_for_ML.csv"
RESULTS_DIR = Path("results")
BAD_THRESHOLD = 2
DEFAULT_GROUPKFOLD_REFERENCE = 0.5399   # overridden when --from-stage3 is used


# ── Stage 3 helper ─────────────────────────────────────────────────────────────


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
        float(cfg.get("best_auprc_mean", DEFAULT_GROUPKFOLD_REFERENCE)),
    )


# ── Data setup ─────────────────────────────────────────────────────────────────


def _setup_data(label_strategy: str):
    """Load + impute + encode + label once. Reused across LOVO and LOSO."""
    df = load_bcr_data(BCR_CSV)
    df = add_missingness_flags(df)

    cfg_path = Path("configs/feature_cols.json")
    if cfg_path.exists():
        feature_cols = json.loads(cfg_path.read_text())
        # Some legacy feature_cols.json files include `channel_label_enc`,
        # which is generated inside impute_and_encode_channels — drop it
        # before passing to the imputer to avoid double-handling.
        feature_cols = [c for c in feature_cols if c != "channel_label_enc"]
    else:
        feature_cols = [
            c for c in df.columns
            if c not in NON_FEATURE_COLS
            and c != "site"
            and df[c].dtype != object
        ]

    df, _ = impute_and_encode_channels(df, feature_cols)
    feature_cols = feature_cols + ["channel_label_enc"]

    artifacts = build_label_artifacts(
        df, strategy=label_strategy, threshold=BAD_THRESHOLD
    )
    return df, feature_cols, artifacts


# ── Generic LOO loop ───────────────────────────────────────────────────────────


def run_loo_cv(
    label_strategy: str,
    model_name: str,
    mode: str = "visit",
    overrides: dict | None = None,
    device: str = "cpu",
    skip_empty: bool = True,
) -> dict:
    """Run leave-one-out CV across `mode` ∈ {'visit', 'subject'}.

    Parameters
    ----------
    label_strategy : passed to build_label_artifacts() — must match the
        strategy used to fit the model.
    model_name     : key from SUPPORTED_MODELS in models.py.
    mode           : 'visit' → LOVO (4 folds);
                     'subject' → LOSO (43 folds — leave-one-subject-out).
    overrides      : extra kwargs forwarded to create_model() (typically the
                     Stage 3 HPO winning hyperparameters).
    device         : 'cpu', 'cuda', or 'mps'. Routed through create_model().
    skip_empty     : skip held-out groups whose test set has no positives
                     (AUPRC is undefined). Logs a clear warning.

    Returns
    -------
    dict with summary stats (auprc_mean / auprc_std / min / max), per-fold
    rows, and the field `mode`.
    """
    if mode not in {"visit", "subject"}:
        raise ValueError(f"mode must be 'visit' or 'subject', got {mode!r}")
    overrides = dict(overrides or {})

    df, feature_cols, artifacts = _setup_data(label_strategy)
    y = artifacts.y_hard
    raw_w = artifacts.sample_weights
    weights = raw_w if not np.allclose(raw_w, 1.0) else None

    group_col = "visit" if mode == "visit" else "subject_id"
    groups = sorted(df[group_col].unique())

    logger.info(f"Mode: leave-one-{mode}-out  ({len(groups)} folds)")
    logger.info(f"Label strategy: {label_strategy}  Model: {model_name}")
    if overrides:
        logger.info(f"Overrides applied: {overrides}")

    results: list[dict] = []
    pooled_y_true: list[np.ndarray] = []
    pooled_y_prob: list[np.ndarray] = []

    for held_out in groups:
        train_mask = (df[group_col] != held_out).values
        test_mask  = (df[group_col] == held_out).values

        y_train = y[train_mask]
        y_test  = y[test_mask]

        if skip_empty and (y_test.sum() == 0 or y_train.sum() == 0):
            logger.warning(
                f"  {mode}={held_out}: no positives in "
                f"{'test' if y_test.sum() == 0 else 'train'} — skipping"
            )
            continue

        X_train_df = df.loc[train_mask, feature_cols]
        X_test_df  = df.loc[test_mask,  feature_cols]
        w_train = weights[train_mask] if weights is not None else None
        spw_train = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

        prep = FeaturePreprocessor()
        X_train = prep.fit_transform(X_train_df)
        X_test  = prep.transform(X_test_df)

        model = create_model(
            model_name, scale_pos_weight=spw_train, device=device, **overrides
        )
        model.fit(X_train, y_train, sample_weight=w_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        auprc = average_precision_score(y_test, y_prob)
        auroc = roc_auc_score(y_test, y_prob)

        # Stash predictions for the pooled metric
        pooled_y_true.append(np.asarray(y_test))
        pooled_y_prob.append(np.asarray(y_prob))

        results.append({
            f"held_out_{mode}": int(held_out),
            "n_train":          int(train_mask.sum()),
            "n_test":           int(test_mask.sum()),
            "n_bad_test":       int(y_test.sum()),
            "bad_rate_test":    round(float(y_test.mean()), 4),
            "auprc":            round(float(auprc), 4),
            "auroc":            round(float(auroc), 4),
        })
        logger.info(
            f"  {mode}={held_out:>3}: AUPRC={auprc:.4f}  AUROC={auroc:.4f}  "
            f"(n_test={test_mask.sum()}, bads={y_test.sum()})"
        )

    if not results:
        raise RuntimeError(f"No folds completed for mode={mode}")

    auprc_vals = np.array([r["auprc"] for r in results])
    auroc_vals = np.array([r["auroc"] for r in results])

    # Pooled metrics — concatenate every fold's predictions and compute ONE
    # AUPRC / AUROC over the full pool. Each prediction is weighted equally,
    # so subjects with few positives do not dominate the headline number
    # (which they would in the arithmetic mean of per-fold AUPRCs).
    pooled_y_true_arr = np.concatenate(pooled_y_true)
    pooled_y_prob_arr = np.concatenate(pooled_y_prob)
    pooled_auprc = float(average_precision_score(pooled_y_true_arr, pooled_y_prob_arr))
    pooled_auroc = float(roc_auc_score(pooled_y_true_arr, pooled_y_prob_arr))

    return {
        "mode":              mode,
        "label_strategy":    label_strategy,
        "model_name":        model_name,
        "overrides":         overrides,
        "n_folds_completed": len(results),
        "n_folds_total":     len(groups),
        # Per-fold averaging (each held-out group weighted equally)
        "auprc_mean":        round(float(auprc_vals.mean()), 4),
        "auprc_std":         round(float(auprc_vals.std()), 4),
        "auprc_median":      round(float(np.median(auprc_vals)), 4),
        "auprc_min":         round(float(auprc_vals.min()), 4),
        "auprc_max":         round(float(auprc_vals.max()), 4),
        "auroc_mean":        round(float(auroc_vals.mean()), 4),
        "auroc_std":         round(float(auroc_vals.std()), 4),
        # Pooled — every prediction weighted equally (the honest headline)
        "auprc_pooled":      round(pooled_auprc, 4),
        "auroc_pooled":      round(pooled_auroc, 4),
        "n_predictions":     int(len(pooled_y_true_arr)),
        "n_positives":       int(pooled_y_true_arr.sum()),
        "per_fold":          results,
    }


# ── Reporting ──────────────────────────────────────────────────────────────────


def _summarise_fold(result: dict, ref_auprc: float) -> None:
    gap_mean   = ref_auprc - result["auprc_mean"]
    gap_pooled = ref_auprc - result["auprc_pooled"]
    mode = result["mode"]
    logger.info(
        f"\n  AUPRC (mean across folds, {mode}): "
        f"{result['auprc_mean']:.4f} ± {result['auprc_std']:.4f}  "
        f"[median={result['auprc_median']:.4f}, "
        f"min={result['auprc_min']:.4f}, max={result['auprc_max']:.4f}]"
    )
    logger.info(
        f"  AUPRC (pooled — all predictions concatenated): "
        f"{result['auprc_pooled']:.4f}  "
        f"(n={result['n_predictions']}, positives={result['n_positives']})"
    )
    logger.info(
        f"  GroupKFold reference: {ref_auprc:.4f}  "
        f"(gap_mean={gap_mean:+.4f}, gap_pooled={gap_pooled:+.4f})"
    )
    # The pooled gap is the honest one — judge healthiness off it.
    if abs(gap_pooled) <= 0.02:
        logger.info("  → Pooled generalisation looks healthy (gap within ±0.02).")
    elif gap_pooled > 0.05:
        logger.warning(
            "  → Significant pooled gap: model degrades on held-out groups."
        )
    else:
        logger.info("  → Mild pooled generalisation gap.")


def _plot_auprc_vs_nbad(
    result: dict, out_path: Path, ref_auprc: float
) -> None:
    """Scatter plot of per-fold AUPRC vs number of positives in the test set.

    Surfaces the small-positive-count instability: subjects with very few
    bad channels (< ~10) tend to produce extreme AUPRCs that drag the
    arithmetic mean around. The pooled / median lines are the more
    trustworthy summary statistics.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_fold = result["per_fold"]
    nbads  = np.array([f["n_bad_test"] for f in per_fold])
    auprcs = np.array([f["auprc"] for f in per_fold])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(
        nbads, auprcs, alpha=0.75, color="#58a6ff",
        s=70, edgecolor="black", linewidth=0.6, zorder=3,
    )
    ax.axhline(
        result["auprc_mean"], color="#f85149", ls="--", lw=1.4,
        label=f"per-fold mean = {result['auprc_mean']:.3f}",
    )
    ax.axhline(
        result["auprc_median"], color="#3fb950", ls="--", lw=1.4,
        label=f"per-fold median = {result['auprc_median']:.3f}",
    )
    ax.axhline(
        result["auprc_pooled"], color="#a371f7", ls="-", lw=1.7,
        label=f"pooled (honest) = {result['auprc_pooled']:.3f}",
    )
    ax.axhline(
        ref_auprc, color="grey", ls=":", lw=1.2,
        label=f"GroupKFold ref = {ref_auprc:.3f}",
    )
    mode = result["mode"]
    ax.set_xlabel(f"# positives in held-out {mode}")
    ax.set_ylabel("AUPRC")
    ax.set_title(
        f"Leave-one-{mode}-out: AUPRC vs positive-class count "
        f"(n folds = {len(per_fold)})"
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim([0, 1.05])
    ax.set_xlim(left=0)
    ax.grid(alpha=0.3, zorder=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Scatter plot saved -> {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="LOVO / LOSO generalisation evaluation"
    )
    parser.add_argument(
        "--from-stage3",
        action="store_true",
        help="Use Stage 3 HPO winner (results/best_config.json).",
    )
    parser.add_argument("--label-strategy", default="hard_threshold")
    parser.add_argument("--model", default="xgboost")
    parser.add_argument(
        "--mode",
        choices=["lovo", "loso", "both"],
        default="lovo",
        help="LOVO=leave-one-visit-out (4 folds). "
             "LOSO=leave-one-subject-out (43 folds, ~3-5 min). "
             "both = run LOVO followed by LOSO.",
    )
    parser.add_argument(
        "--device", choices=["cpu", "cuda", "mps"], default="cpu",
    )
    parser.add_argument(
        "--use-engineered-features", action="store_true",
        help="DEPRECATED — legacy train.py compat only.",
    )
    parser.add_argument(
        "--use-impedance-interactions", action="store_true",
        help="DEPRECATED — legacy train.py compat only.",
    )
    args = parser.parse_args()

    # ── Resolve config ────────────────────────────────────────────────────────
    if args.from_stage3:
        label_strategy, model_name, overrides, ref_auprc = _resolve_stage3_inputs()
        tag = "best"
        logger.info(
            f"Stage 3 mode: model={model_name}, strategy={label_strategy}"
        )
        logger.info(f"GroupKFold reference AUPRC: {ref_auprc:.4f}")
    else:
        label_strategy = args.label_strategy
        model_name     = args.model
        overrides      = {}
        ref_auprc      = DEFAULT_GROUPKFOLD_REFERENCE
        tag = build_run_tag(
            label_strategy, model_name,
            use_engineered_features=args.use_engineered_features,
            use_impedance_interactions=args.use_impedance_interactions,
        )

    # ── Run requested modes ───────────────────────────────────────────────────
    modes_to_run: list[str] = []
    if args.mode in {"lovo", "both"}:
        modes_to_run.append("visit")
    if args.mode in {"loso", "both"}:
        modes_to_run.append("subject")

    figures_dir = RESULTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict] = {}
    for mode in modes_to_run:
        logger.info("\n" + "=" * 60)
        logger.info(f"LEAVE-ONE-{mode.upper()}-OUT  ({tag})")
        logger.info("=" * 60)
        r = run_loo_cv(
            label_strategy=label_strategy,
            model_name=model_name,
            mode=mode,
            overrides=overrides,
            device=args.device,
        )
        r["groupkfold_ref_auprc"]    = ref_auprc
        r["generalisation_gap_mean"] = round(ref_auprc - r["auprc_mean"], 4)
        r["generalisation_gap_pooled"] = round(ref_auprc - r["auprc_pooled"], 4)
        # Backwards-compatible alias for older readers
        r["generalisation_gap"]      = r["generalisation_gap_pooled"]
        _summarise_fold(r, ref_auprc)

        # Scatter plot — most informative for LOSO (43 points), still
        # useful for LOVO (4 points) as a quick visual sanity check.
        plot_path = figures_dir / f"loo_{mode}_auprc_vs_nbad_{tag}.png"
        _plot_auprc_vs_nbad(r, plot_path, ref_auprc)

        all_results[mode] = r

    # ── Persist ───────────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"site_generalisation_{tag}.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    logger.info(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
