"""
bad_channel_rejection/ablation_stage2b_architectures.py

Stage 2b — ARCHITECTURE ABLATION within the winning ML family.

Given the family chosen by Stage 2a, evaluate every backend implementation
that belongs to that family on the same fixed folds and pick the best
architecture.  This isolates within-family implementation differences once
the broad strategy choice has already been made.

Family → architecture members
-----------------------------
    linear      : ElasticNet, Ridge, Lasso                    (3 members)
    bagging     : Random Forest, ExtraTrees                   (2 members)
    boosting    : XGBoost, LightGBM, CatBoost,                (4 members)
                  HistGradientBoosting
    transformer : FT-Transformer, SAINT, TabTransformer,      (4 members)
                  TabPFN                                       (rtdl + HuggingFace)
    mlp         : ResNet-Tabular, NODE, TabNet                (3 members)
    foundation  : TabPFN, TabICL, Mitra                       (3 members, HF Hub)

Single-member families are short-circuited: no extra CV is run, and the sole
member is declared winner with metrics inherited from Stage 2a.  This avoids
redundant compute when the architecture choice is forced.

GPU note
--------
Most members from the transformer / mlp / foundation families need a CUDA GPU.
Pass `--device cuda` (or rely on auto-detection on Colab) so each backend is
routed to the correct accelerator.  Tree-based members support GPU only for
XGBoost / LightGBM / CatBoost — the rest stay on CPU.

The fairness rules from the shared core (same folds, same metric, same budget,
device logging, final CV, margin guard) all apply — see
`ablation_stage2_models.py`.

Run (default = all members of the winning family):
    python -m bad_channel_rejection.ablation_stage2b_architectures \\
        --winning-strategy dawid_skene \\
        --winning-family boosting \\
        --device cuda

Run a single architecture within the family (re-run / debug):
    python -m bad_channel_rejection.ablation_stage2b_architectures \\
        --winning-strategy dawid_skene \\
        --winning-family boosting \\
        --model lightgbm \\
        --device cuda
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

STRATEGY_MEMBERS: dict[str, list[str]] = {
    # Classical linear models
    "linear": [
        "elasticnet",
        "ridge",
        "lasso",
    ],

    # Ensemble bagging methods
    "bagging": [
        "random_forest",
        "extra_trees",
    ],

    # Gradient boosting methods — still dominant on tabular benchmarks
    "boosting": [
        "xgboost",
        "lightgbm",
        "catboost",
        "histgradientboosting",       # sklearn's built-in, fast native support
    ],

    # Transformer / attention-based deep tabular models
    "transformer": [
        "ft_transformer",             # Feature Tokenizer + Transformer (rtdl)
        "saint",                      # Self-Attention and Intersample Attention Transformer
        "tab_transformer",            # TabTransformer: categorical embeddings via attention
        "tabpfn",                     # TabPFN v2/2.5 — SOTA foundation model (Prior-Labs/tabpfn_2_5)
    ],

    # MLP / residual deep tabular models
    "mlp": [
        "resnet_tabular",             # ResNet for tabular (rtdl / pytorch_tabular)
        "node",                       # Neural Oblivious Decision Ensembles
        "tabnet",                     # TabNet: sequential attention for feature selection
    ],

    # Tabular foundation models (in-context / zero-shot)
    "foundation": [
        "tabpfn",                     # Prior-Labs/tabpfn_2_5 on HF Hub
        "tabicl",                     # TabICL: in-context learning on large tabular data
        "mitra",                      # Mitra: mixed synthetic priors foundation model
    ],
}


def run_stage2b_architecture_ablation(
    winning_strategy: str,
    winning_family: str,
    model_filter: str | None = None,
    device: str = "cpu",
) -> dict:
    """Stage 2b: pick the winning architecture within the chosen family.

    Args:
        winning_strategy : label strategy from Stage 1.
        winning_family   : ML family from Stage 2a (linear / bagging /
                           boosting / transformer).
        model_filter     : if given, run that single architecture only.
        device           : 'cuda', 'mps', or 'cpu'.

    Returns
    -------
    dict with keys: conditions, winner_model, winner_family, winner_cautious,
    final_cv, label_strategy, n_members, skipped, device.
    """
    if winning_family not in STRATEGY_MEMBERS:
        raise ValueError(
            f"Unknown family {winning_family!r}. "
            f"Valid: {list(STRATEGY_MEMBERS)}"
        )

    members = STRATEGY_MEMBERS[winning_family]
    if model_filter is not None:
        if model_filter not in members:
            raise ValueError(
                f"Model {model_filter!r} is not a member of family "
                f"{winning_family!r}. Members: {members}"
            )
        members = [model_filter]

    logger.info("\n" + "=" * 60)
    logger.info(
        f"STAGE 2b — ARCHITECTURE ABLATION  "
        f"(family={winning_family}  label={winning_strategy}  device={device})"
    )
    logger.info(f"Members: {members}")
    logger.info("=" * 60)

    data_bundle = load_data_and_folds(winning_strategy)

    # Short-circuit: only one architecture in this family → declare winner
    # without running a redundant ablation.
    if len(members) == 1:
        sole = members[0]
        logger.info(
            f"Family '{winning_family}' has a single member: {sole!r}. "
            "Running CV once for completeness (no comparison)."
        )
        core = run_model_ablation_core(
            data_bundle=data_bundle,
            models_to_run=[sole],
            stage_label="2b",
            device=device,
            final_cv_for_winner=False,   # nothing to confirm against
        )
        return {
            "conditions": core["conditions"],
            "ranked": core["ranked"],
            "winner_model": sole,
            "winner_family": winning_family,
            "winner_cautious": False,
            "final_cv": None,
            "label_strategy": winning_strategy,
            "n_members": 1,
            "skipped_comparison": True,
            "device": device,
            "n_folds": N_FOLDS,
        }

    # Multi-member family — run a real ablation.
    core = run_model_ablation_core(
        data_bundle=data_bundle,
        models_to_run=members,
        stage_label="2b",
        device=device,
        final_cv_for_winner=True,
    )

    return {
        "conditions": core["conditions"],
        "ranked": core["ranked"],
        "winner_model": core["winner_model"],
        "winner_family": winning_family,
        "winner_cautious": core["winner_cautious"],
        "final_cv": core["final_cv"],
        "label_strategy": winning_strategy,
        "n_members": len(members),
        "skipped_comparison": False,
        "device": device,
        "n_folds": N_FOLDS,
    }


def _write_stage2b_report(stage2b: dict, out_path: Path) -> None:
    label = stage2b["label_strategy"]
    family = stage2b["winner_family"]
    winner = stage2b["winner_model"]
    cautious = stage2b["winner_cautious"]
    conditions = stage2b["conditions"]
    final_cv = stage2b.get("final_cv") or {}

    ranked = sorted(
        conditions.keys(),
        key=lambda k: conditions[k]["auprc_mean"],
        reverse=True,
    )

    lines = [
        "# BCR Stage 2b — Architecture Ablation Report",
        "",
        f"Family: **{family}** | Label strategy: **{label}** | "
        f"Device: **{stage2b['device']}** | "
        f"Folds: **{stage2b['n_folds']}**",
        "",
    ]

    if stage2b["skipped_comparison"]:
        lines += [
            f"> ℹ️ Family `{family}` has a single member — no architecture "
            f"comparison performed.",
            "",
        ]

    lines += [
        "| Rank | Architecture | AUPRC (mean ± std) | Best iter |",
        "|------|--------------|--------------------|-----------|",
    ]
    for rank, mname in enumerate(ranked, 1):
        r = conditions[mname]
        marker = " ✓" if mname == winner else ""
        lines.append(
            f"| {rank} | {mname}{marker} | "
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
        f"**Winning architecture:** `{winner}`  "
        f"(family: `{family}`){caution_note}",
    ]

    if final_cv:
        drift = abs(
            final_cv["auprc_mean"] - conditions[winner]["auprc_mean"]
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
        "Run Stage 3 hyper-parameter optimisation:",
        "",
        "```",
        "python -m bad_channel_rejection.ablation_stage3_hpo \\",
        f"    --winning-strategy {label} \\",
        f"    --winning-model {winner} \\",
        "    --count 50",
        "```",
    ]

    out_path.write_text("\n".join(lines))
    logger.info(f"Report saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "BCR Stage 2b: architecture ablation within the winning ML family"
        )
    )
    parser.add_argument(
        "--winning-strategy",
        required=True,
        help="Label strategy from Stage 1 (e.g. dawid_skene)",
    )
    parser.add_argument(
        "--winning-family",
        required=True,
        choices=list(STRATEGY_MEMBERS.keys()),
        help="ML family chosen by Stage 2a.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Run a single architecture within the family (debug / re-run)",
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
        name=f"bcr_ablation_stage2b_{args.winning_family}",
        tags=["ablation", "stage2b", "architectures", args.winning_family],
        config={
            "winning_strategy": args.winning_strategy,
            "winning_family": args.winning_family,
            "device": device,
            "n_folds": N_FOLDS,
            "margin_caution_threshold": MARGIN_CAUTION,
        },
        settings=wandb.Settings(init_timeout=120),
    )
    t0 = time.time()

    stage2b = run_stage2b_architecture_ablation(
        winning_strategy=args.winning_strategy,
        winning_family=args.winning_family,
        model_filter=args.model,
        device=device,
    )

    for m, r in stage2b["conditions"].items():
        wandb.log({
            f"stage2b/{m}/auprc": r["auprc_mean"],
            f"stage2b/{m}/auprc_std": r["auprc_std"],
            f"stage2b/{m}/best_iter_mean": r["best_iter_mean"],
            f"stage2b/{m}/device": r["device"],
        })
    if stage2b.get("final_cv"):
        wandb.log({
            "stage2b/final_cv/auprc": stage2b["final_cv"]["auprc_mean"],
            "stage2b/final_cv/auprc_std": stage2b["final_cv"]["auprc_std"],
        })
    wandb.summary.update({
        "stage2b_winner_model": stage2b["winner_model"],
        "stage2b_winner_family": stage2b["winner_family"],
        "stage2b_winner_cautious": stage2b["winner_cautious"],
        "stage2b_skipped_comparison": stage2b["skipped_comparison"],
        "device": device,
        "runtime_min": (time.time() - t0) / 60,
    })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "stage2b_results.json"
    results_path.write_text(json.dumps(stage2b, indent=2))
    logger.info(f"Raw results -> {results_path}")
    _write_stage2b_report(stage2b, RESULTS_DIR / "stage2b_report.md")

    wandb.finish()

    print(
        f"\nStage 2b winner: {stage2b['winner_model']} "
        f"(family: {stage2b['winner_family']})"
        + ("\n⚠️  Tiny margin — treat winner cautiously."
           if stage2b["winner_cautious"] else "")
        + ("\nℹ️  Family had only one member — winner inherited."
           if stage2b["skipped_comparison"] else "")
    )
    logger.info(f"Done. Runtime: {(time.time() - t0) / 60:.1f} min.")


if __name__ == "__main__":
    main()
