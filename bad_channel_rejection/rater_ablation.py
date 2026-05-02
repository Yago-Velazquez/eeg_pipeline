"""
bad_channel_rejection/rater_ablation.py

Full four-stage ablation + HPO pipeline for BCR.

Stage 1  — LABEL ABLATION (model held constant = XGBoost)
    Conditions A–F differ only in label / weight strategy.
    Winner = argmax(mean GroupKFold AUPRC).

Stage 2a — STRATEGY ABLATION (label held constant = Stage 1 winner)
    One representative per ML family (linear / bagging / boosting / transformer)
    on the same fixed folds.
    Winner = winning ML *family*.

Stage 2b — ARCHITECTURE ABLATION (family held constant = Stage 2a winner)
    Every architecture inside the winning family on the same fixed folds.
    Winner = winning *architecture*.

Stage 3  — HYPER-PARAMETER OPTIMISATION (architecture held constant = Stage 2b winner)
    W&B Bayesian sweep across the architecture's hyper-parameter space.
    Final model retrained on full data and saved.

Why split Stage 2 in two?
-------------------------
Lumping ElasticNet, Random Forest, three boosting libraries and a transformer
into one ranking confounds two distinct questions:

    Q1: which ML *strategy* fits this data?
    Q2: which *implementation* of that strategy is best?

Stage 2a answers Q1 with one rep per family; Stage 2b answers Q2 within the
winning family.  This avoids penalising / rewarding a family for having more
or fewer entries in the bake-off.

Run
---
    python -m bad_channel_rejection.rater_ablation [--hpo] [--hpo-count N]

For individual stages see:
    python -m bad_channel_rejection.ablation_stage1_labels         --help
    python -m bad_channel_rejection.ablation_stage2a_strategies    --help
    python -m bad_channel_rejection.ablation_stage2b_architectures --help
    python -m bad_channel_rejection.ablation_stage3_hpo            --help
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import wandb
from dotenv import load_dotenv

from .ablation_stage1_labels import run_stage1_label_ablation
from .ablation_stage2_models import get_device
from .ablation_stage2a_strategies import run_stage2a_strategy_ablation
from .ablation_stage2b_architectures import run_stage2b_architecture_ablation
from .ablation_stage3_hpo import run_stage3_hpo
from .logging_config import setup_logging

load_dotenv()
os.environ.setdefault("WANDB_MODE", "offline")

logger = setup_logging(__name__)

RESULTS_DIR = Path("results")
WANDB_PROJECT = "eeg-bcr"


def write_report(
    stage1: dict,
    stage2a: dict,
    stage2b: dict,
    out_path: Path,
    stage3: dict | None = None,
) -> None:
    lines = [
        "# BCR Four-Stage Ablation Report",
        "",
        "## Background",
        "",
        "Inter-rater reliability analysis on this dataset shows:",
        "- Krippendorff α = 0.211 (fair agreement)",
        "- D_o = 6.8% observed pairwise disagreement",
        "- Site 3 sensitivity (DS) = 0.075 — nearly blind to bad channels",
        "- Site 4a sensitivity (DS) = 0.714 — most reliable positive detector",
        "- 493 channels (2.6%) at score=2 represent maximum ambiguity",
        "",
        "## Stage 1 — Label ablation (model = XGBoost)",
        "",
        "| ID | Strategy | AUPRC (mean ± std) | Δ vs A |",
        "|----|----------|--------------------|--------|",
    ]
    baseline = stage1["baseline_auprc"]
    for cid, r in stage1["conditions"].items():
        d = r["auprc_mean"] - baseline
        dstr = f"{d:+.4f}" if cid != "A" else "—"
        lines.append(
            f"| {cid} | {r['strategy']} | "
            f"{r['auprc_mean']:.4f} ± {r['auprc_std']:.4f} | {dstr} |"
        )
    lines += [
        "",
        f"**Stage 1 winner:** `{stage1['winner_strategy']}` "
        f"(condition {stage1['winner_condition_id']})",
        "",
        f"## Stage 2a — Strategy ablation "
        f"(label = {stage1['winner_strategy']})",
        "",
        "| Family | Representative | AUPRC (mean ± std) |",
        "|--------|----------------|--------------------|",
    ]
    ranked2a = sorted(
        stage2a["families"].keys(),
        key=lambda f: stage2a["families"][f]["auprc_mean"],
        reverse=True,
    )
    for fam in ranked2a:
        r = stage2a["families"][fam]
        marker = " ✓" if fam == stage2a["winner_family"] else ""
        lines.append(
            f"| {fam}{marker} | {r['representative_model']} | "
            f"{r['auprc_mean']:.4f} ± {r['auprc_std']:.4f} |"
        )
    lines += [
        "",
        f"**Stage 2a winning family:** `{stage2a['winner_family']}` "
        f"(rep: `{stage2a['winner_model_representative']}`)",
        "",
        f"## Stage 2b — Architecture ablation "
        f"(family = {stage2a['winner_family']})",
        "",
    ]
    if stage2b["skipped_comparison"]:
        lines.append(
            f"> ℹ️ Family `{stage2a['winner_family']}` has a single member "
            "— no architecture comparison performed.\n"
        )
    lines += [
        "| Architecture | AUPRC (mean ± std) | Best iter |",
        "|--------------|--------------------|-----------|",
    ]
    ranked2b = sorted(
        stage2b["conditions"].keys(),
        key=lambda k: stage2b["conditions"][k]["auprc_mean"],
        reverse=True,
    )
    for mname in ranked2b:
        r = stage2b["conditions"][mname]
        marker = " ✓" if mname == stage2b["winner_model"] else ""
        lines.append(
            f"| {mname}{marker} | "
            f"{r['auprc_mean']:.4f} ± {r['auprc_std']:.4f} | "
            f"{r['best_iter_mean']:.0f} |"
        )

    lines += [
        "",
        f"**Stage 2b winning architecture:** `{stage2b['winner_model']}`",
        "",
        "## Recommended production configuration",
        "",
        f"- Label strategy : `{stage1['winner_strategy']}`",
        f"- ML family      : `{stage2a['winner_family']}`",
        f"- Architecture   : `{stage2b['winner_model']}`",
        "",
        "Run Stage 3 HPO to optimise the winning configuration:",
        "",
        "```",
        "python -m bad_channel_rejection.ablation_stage3_hpo \\",
        f"    --winning-strategy {stage1['winner_strategy']} \\",
        f"    --winning-model {stage2b['winner_model']} \\",
        "    --count 50",
        "```",
    ]

    if stage3 is not None:
        lines += [
            "",
            f"## Stage 3 — HPO ({stage2b['winner_model']})",
            "",
            f"Trials: **{stage3['n_trials']}** | "
            f"W&B sweep: `{stage3['sweep_id']}`",
            "",
            "**Best CV AUPRC:** "
            f"{stage3['best_auprc_mean']:.4f} ± "
            f"{stage3['best_auprc_std']:.4f}",
            "",
            "**Best config:**",
            "```json",
            json.dumps(stage3["best_config"], indent=2),
            "```",
            "",
            f"**Final model:** `{stage3['model_path']}`",
        ]

    out_path.write_text("\n".join(lines))
    logger.info(f"Report saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="BCR full pipeline: stages 1 → 2a → 2b → (optional) 3"
    )
    parser.add_argument(
        "--hpo",
        action="store_true",
        help="Run Stage 3 HPO after stages 1, 2a, 2b complete.",
    )
    parser.add_argument(
        "--hpo-count",
        type=int,
        default=50,
        help="Number of W&B sweep trials in Stage 3 (default: 50).",
    )
    args = parser.parse_args()

    device = get_device()
    logger.info(f"Detected device: {device}")

    wandb.init(
        project=WANDB_PROJECT,
        name="bcr_full_pipeline",
        tags=["ablation", "full_pipeline"],
        config={"device": device, "hpo": args.hpo},
        settings=wandb.Settings(init_timeout=120),
    )
    t0 = time.time()

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    stage1 = run_stage1_label_ablation()
    wandb.log({
        f"stage1/{cid}/auprc": r["auprc_mean"]
        for cid, r in stage1["conditions"].items()
    })

    # ── Stage 2a ──────────────────────────────────────────────────────────────
    stage2a = run_stage2a_strategy_ablation(
        winning_strategy=stage1["winner_strategy"], device=device
    )
    wandb.log({
        f"stage2a/{fam}/auprc": r["auprc_mean"]
        for fam, r in stage2a["families"].items()
    })

    # ── Stage 2b ──────────────────────────────────────────────────────────────
    stage2b = run_stage2b_architecture_ablation(
        winning_strategy=stage1["winner_strategy"],
        winning_family=stage2a["winner_family"],
        device=device,
    )
    wandb.log({
        f"stage2b/{m}/auprc": r["auprc_mean"]
        for m, r in stage2b["conditions"].items()
    })

    # ── Stage 3 (optional) ────────────────────────────────────────────────────
    stage3: dict | None = None
    if args.hpo:
        stage3 = run_stage3_hpo(
            winning_strategy=stage1["winner_strategy"],
            winning_model=stage2b["winner_model"],
            count=args.hpo_count,
            device=device,
        )
        wandb.log({
            "stage3/best_auprc_mean": stage3["best_auprc_mean"],
            "stage3/best_auprc_std": stage3["best_auprc_std"],
            "stage3/n_trials": stage3["n_trials"],
        })

    wandb.summary.update({
        "stage1_winner": stage1["winner_strategy"],
        "stage2a_winner_family": stage2a["winner_family"],
        "stage2b_winner_model": stage2b["winner_model"],
        "stage3_best_auprc": (stage3 or {}).get("best_auprc_mean"),
        "device": device,
        "total_runtime_min": (time.time() - t0) / 60,
    })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "ablation_results.json"
    out.write_text(json.dumps({
        "stage1":  stage1,
        "stage2a": stage2a,
        "stage2b": stage2b,
        "stage3":  stage3,
    }, indent=2))
    logger.info(f"Raw results -> {out}")

    write_report(
        stage1, stage2a, stage2b,
        RESULTS_DIR / "ablation_report.md",
        stage3=stage3,
    )

    wandb.finish()
    logger.info(
        f"\nDone. Total runtime: {(time.time()-t0)/60:.1f} min.\n"
        f"Winner: label={stage1['winner_strategy']} | "
        f"family={stage2a['winner_family']} | "
        f"arch={stage2b['winner_model']}"
        + (f" | hpo_auprc={stage3['best_auprc_mean']:.4f}" if stage3 else "")
    )


if __name__ == "__main__":
    main()
