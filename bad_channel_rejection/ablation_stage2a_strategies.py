"""
bad_channel_rejection/ablation_stage2a_strategies.py

Stage 2a — ML STRATEGY ABLATION (broad family selection).

Runs ONE representative from each broad ML family on the same pre-computed
GroupKFold(5) folds and picks the winning family.  Stage 2b then exhaustively
evaluates every architecture inside the winning family.

Strategy representatives
------------------------
    linear      → ElasticNet         (regularised logistic regression)
    bagging     → Random Forest      (bootstrap-aggregated decision trees)
    boosting    → XGBoost            (canonical gradient boosting baseline)
    transformer → FT-Transformer     (feature-tokeniser transformer, rtdl)
    mlp         → ResNet-Tabular     (residual MLP for tabular, rtdl)
    foundation  → TabPFN             (in-context tabular foundation model)

Why these picks?
----------------
- ElasticNet captures the linear family with one knob (l1_ratio) spanning
  Lasso ↔ Ridge.  The 2b stage tests pure Ridge / Lasso variants.
- Random Forest is the de-facto bagging baseline against ExtraTrees in 2b.
- XGBoost is the longest-established boosting library.  LightGBM, CatBoost
  and HistGradientBoosting are evaluated against it in 2b only if boosting
  wins this stage.
- FT-Transformer is the strongest published tabular transformer
  (Gorishniy 2021); SAINT / TabTransformer / TabPFN are evaluated in 2b.
- ResNet-Tabular is the rtdl baseline for residual MLPs vs NODE / TabNet.
- TabPFN is the SOTA in-context foundation model — a different paradigm
  (no per-task gradient training) so it gets its own family for fair
  comparison against TabICL / Mitra in 2b.

GPU note
--------
Most non-classical members (transformers, MLPs, foundation models) require a
CUDA GPU.  Pass `--device cuda` (or rely on auto-detection on Colab) to route
each backend to the correct accelerator.  Tree-based models stay on CPU
unless they have native GPU support (XGBoost / LightGBM / CatBoost).

The fairness rules from the shared core (same folds, same metric, same budget,
device logging, final CV, margin guard) all apply — see
`ablation_stage2_models.py`.

Run (all strategies):
    python -m bad_channel_rejection.ablation_stage2a_strategies \\
        --winning-strategy dawid_skene --device cuda

Run (single strategy):
    python -m bad_channel_rejection.ablation_stage2a_strategies \\
        --winning-strategy dawid_skene --strategy boosting --device cuda
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import wandb

from .ablation_stage2_models import (
    MARGIN_CAUTION,
    N_FOLDS,
    RESULTS_DIR,
    WANDB_PROJECT,
    get_device,
    load_data_and_folds,
    run_model_ablation_core,
)
from .logging_config import setup_logging

logger = setup_logging(__name__)

# (family, representative_model, description)
#
# One representative per family — chosen to be the most established / canonical
# member so that 2a's family ranking is not biased by any single library's
# implementation quirks.  Stage 2b then exhaustively compares all architectures
# inside the winning family.
STRATEGY_REPRESENTATIVES: list[tuple[str, str, str]] = [
    ("linear",      "elasticnet",     "L1/L2 regularised logistic regression"),
    ("bagging",     "random_forest",  "Bootstrap-aggregated decision trees"),
    ("boosting",    "xgboost",        "Gradient boosting (XGBoost as canonical baseline)"),
    ("transformer", "ft_transformer", "Feature-tokeniser transformer (rtdl)"),
    ("mlp",         "resnet_tabular", "Residual MLP for tabular data (rtdl)"),
    ("foundation",  "tabpfn",         "In-context tabular foundation model (Prior-Labs)"),
]


def run_stage2a_strategy_ablation(
    winning_strategy: str,
    strategy_filter: str | None = None,
    device: str = "cpu",
) -> dict:
    """Stage 2a: pick the winning ML family.

    Args:
        winning_strategy : label strategy from Stage 1 (e.g. 'dawid_skene').
        strategy_filter  : if given, only that family is evaluated.
        device           : 'cuda', 'mps', or 'cpu'.

    Returns
    -------
    dict with keys: conditions, winner_family, winner_model_representative,
    winner_cautious, families (full table), final_cv, label_strategy, device.
    """
    families_to_run = STRATEGY_REPRESENTATIVES
    if strategy_filter is not None:
        families_to_run = [
            t for t in STRATEGY_REPRESENTATIVES if t[0] == strategy_filter
        ]
        if not families_to_run:
            valid = [t[0] for t in STRATEGY_REPRESENTATIVES]
            raise ValueError(
                f"Unknown strategy {strategy_filter!r}. Valid: {valid}"
            )

    logger.info("\n" + "=" * 60)
    logger.info(
        f"STAGE 2a — STRATEGY ABLATION  "
        f"(label={winning_strategy}  device={device})"
    )
    logger.info("=" * 60)
    for fam, model_name, desc in families_to_run:
        logger.info(f"  {fam:<12} → {model_name:<16} ({desc})")

    data_bundle = load_data_and_folds(winning_strategy)
    models_to_run = [m for _, m, _ in families_to_run]

    core = run_model_ablation_core(
        data_bundle=data_bundle,
        models_to_run=models_to_run,
        stage_label="2a",
        device=device,
        final_cv_for_winner=True,
    )

    # Map model→family for downstream stages.
    model_to_family = {m: fam for fam, m, _ in families_to_run}
    family_to_model = {fam: m for fam, m, _ in families_to_run}
    winner_family = model_to_family[core["winner_model"]]

    # Build per-family table for the report.
    families: dict[str, dict] = {}
    for fam, model_name, desc in families_to_run:
        r = core["conditions"][model_name]
        families[fam] = {
            "family": fam,
            "representative_model": model_name,
            "description": desc,
            "auprc_mean": r["auprc_mean"],
            "auprc_std": r["auprc_std"],
            "auroc_mean": r["auroc_mean"],
            "best_iter_mean": r["best_iter_mean"],
            "device": r["device"],
        }

    return {
        "conditions": core["conditions"],
        "families": families,
        "winner_family": winner_family,
        "winner_model_representative": family_to_model[winner_family],
        "winner_cautious": core["winner_cautious"],
        "final_cv": core["final_cv"],
        "label_strategy": winning_strategy,
        "device": device,
        "n_folds": N_FOLDS,
    }


def _write_stage2a_report(stage2a: dict, out_path: Path) -> None:
    label = stage2a["label_strategy"]
    families = stage2a["families"]
    winner_family = stage2a["winner_family"]
    cautious = stage2a["winner_cautious"]
    final_cv = stage2a.get("final_cv") or {}

    ranked = sorted(
        families.keys(), key=lambda f: families[f]["auprc_mean"], reverse=True
    )

    lines = [
        "# BCR Stage 2a — Strategy Ablation Report",
        "",
        f"Label strategy: **{label}** | "
        f"Device: **{stage2a['device']}** | "
        f"Folds: **{stage2a['n_folds']}**",
        "",
        "| Rank | Family | Representative | AUPRC (mean ± std) | Best iter |",
        "|------|--------|----------------|--------------------|-----------|",
    ]
    for rank, fam in enumerate(ranked, 1):
        r = families[fam]
        marker = " ✓" if fam == winner_family else ""
        lines.append(
            f"| {rank} | {fam}{marker} | {r['representative_model']} | "
            f"{r['auprc_mean']:.4f} ± {r['auprc_std']:.4f} | "
            f"{r['best_iter_mean']:.0f} |"
        )

    caution_note = (
        f"\n> ⚠️ **Caution:** winning margin < {MARGIN_CAUTION} AUPRC — "
        "result may be within numerical-noise range.\n"
        if cautious else ""
    )
    lines += [
        "",
        f"**Winning family:** `{winner_family}` "
        f"(representative: `{stage2a['winner_model_representative']}`)"
        f"{caution_note}",
    ]

    if final_cv:
        winner_model = stage2a["winner_model_representative"]
        drift = abs(
            final_cv["auprc_mean"]
            - stage2a["conditions"][winner_model]["auprc_mean"]
        )
        drift_note = "stable ✓" if drift <= 0.005 else f"⚠️ drift={drift:.4f}"
        lines += [
            "",
            "## Final CV (winner re-evaluation on same folds)",
            "",
            "| AUPRC mean | AUPRC std | Drift vs ablation |",
            "|------------|-----------|-------------------|",
            f"| {final_cv['auprc_mean']:.4f} | {final_cv['auprc_std']:.4f} | "
            f"{drift_note} |",
        ]

    lines += [
        "",
        "## Next step",
        "",
        f"Run Stage 2b within the winning family:",
        "",
        "```",
        "python -m bad_channel_rejection.ablation_stage2b_architectures \\",
        f"    --winning-strategy {label} \\",
        f"    --winning-family {winner_family}",
        "```",
    ]

    out_path.write_text("\n".join(lines))
    logger.info(f"Report saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "BCR Stage 2a: ML strategy ablation "
            "(one representative per family)"
        )
    )
    parser.add_argument(
        "--winning-strategy",
        required=True,
        help="Label strategy from Stage 1 (e.g. dawid_skene)",
    )
    parser.add_argument(
        "--strategy",
        default=None,
        choices=[t[0] for t in STRATEGY_REPRESENTATIVES],
        help="Run a single ML family instead of all four.",
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

    wandb.init(
        project=WANDB_PROJECT,
        name="bcr_ablation_stage2a_strategies",
        tags=["ablation", "stage2a", "strategies"],
        config={
            "winning_strategy": args.winning_strategy,
            "device": device,
            "n_folds": N_FOLDS,
            "margin_caution_threshold": MARGIN_CAUTION,
        },
        settings=wandb.Settings(init_timeout=120),
    )
    t0 = time.time()

    stage2a = run_stage2a_strategy_ablation(
        winning_strategy=args.winning_strategy,
        strategy_filter=args.strategy,
        device=device,
    )

    for fam, r in stage2a["families"].items():
        wandb.log({
            f"stage2a/{fam}/auprc": r["auprc_mean"],
            f"stage2a/{fam}/auprc_std": r["auprc_std"],
            f"stage2a/{fam}/best_iter_mean": r["best_iter_mean"],
            f"stage2a/{fam}/device": r["device"],
        })
    if stage2a.get("final_cv"):
        wandb.log({
            "stage2a/final_cv/auprc": stage2a["final_cv"]["auprc_mean"],
            "stage2a/final_cv/auprc_std": stage2a["final_cv"]["auprc_std"],
        })
    wandb.summary.update({
        "stage2a_winner_family": stage2a["winner_family"],
        "stage2a_winner_representative": stage2a["winner_model_representative"],
        "stage2a_winner_cautious": stage2a["winner_cautious"],
        "device": device,
        "runtime_min": (time.time() - t0) / 60,
    })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "stage2a_results.json"
    results_path.write_text(json.dumps(stage2a, indent=2))
    logger.info(f"Raw results -> {results_path}")
    _write_stage2a_report(stage2a, RESULTS_DIR / "stage2a_report.md")

    wandb.finish()

    print(
        f"\nStage 2a winner family: {stage2a['winner_family']} "
        f"(rep={stage2a['winner_model_representative']})"
        + ("\n⚠️  Tiny margin — treat winner cautiously."
           if stage2a["winner_cautious"] else "")
    )
    logger.info(f"Done. Runtime: {(time.time() - t0) / 60:.1f} min.")


if __name__ == "__main__":
    main()
