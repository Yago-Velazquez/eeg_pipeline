"""
bad_channel_rejection/triage.py

Two-tier human-in-the-loop pipeline for production BCR.

Single-threshold operating points always sacrifice either precision or recall
on this dataset (Krippendorff α=0.211 — even raters disagree on the marginal
cases).  Rather than lock in one threshold, this module declares **two**
thresholds that partition the probability axis into three actionable bands:

    P(bad) >= thr_high      → AUTO-INTERPOLATE   (high-precision band)
    thr_low < P < thr_high  → FLAG FOR REVIEW    (ambiguous middle)
    P(bad) <= thr_low       → AUTO-ACCEPT        (high-recall band — bads here are rare)

Defaults
--------
- ``auto_bad_precision = 0.80``  → highest probability band where we expect
  ≥80% of flagged channels to truly be bad.
- ``auto_good_recall   = 0.95``  → lowest probability band where we expect
  to miss ≤5% of bad channels among the auto-accepted ones.

These map to two operating points on the PR curve, computed once from the
out-of-fold predictions saved by Stage 3 HPO.

Workflow
--------
1. Calibration (one-time, off OOF predictions)::

       python -m bad_channel_rejection.triage calibrate --from-stage3

   Produces:
       results/triage_thresholds.json     (thr_low + thr_high + tier stats)
       results/triage_report.md           (human-readable summary)
       results/figures/triage_histogram.png

2. Production inference (per new EEG session)::

       from bad_channel_rejection.triage import predict_and_triage
       probs, decisions = predict_and_triage(model, X, thr_low, thr_high)
       # decisions ∈ {"auto_good", "review", "auto_bad"}

   The "auto_bad" channels can then be passed straight to
   ``interpolation.interpolate_bad_channels()``; the "review" channels go to
   a queue for an expert; the "auto_good" channels need no action.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .logging_config import setup_logging

logger = setup_logging(__name__)

RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
THRESHOLDS_PATH = RESULTS_DIR / "triage_thresholds.json"

DEFAULT_AUTO_BAD_PRECISION = 0.80
DEFAULT_AUTO_GOOD_RECALL   = 0.95

# Tier labels used everywhere
TIER_AUTO_GOOD = "auto_good"
TIER_REVIEW    = "review"
TIER_AUTO_BAD  = "auto_bad"


# ── Threshold calibration ──────────────────────────────────────────────────────


def compute_triage_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    auto_bad_precision: float = DEFAULT_AUTO_BAD_PRECISION,
    auto_good_recall:   float = DEFAULT_AUTO_GOOD_RECALL,
) -> tuple[float, float]:
    """Calibrate (thr_low, thr_high) from OOF predictions.

    thr_high : lowest threshold whose precision over [thr_high, 1] is
               >= auto_bad_precision  (auto-interpolate band).
    thr_low  : highest threshold whose recall over [thr_low, 1] is
               >= auto_good_recall    (everything below is auto-accepted).

    Raises
    ------
    ValueError if either target is unreachable, or if the resulting bands
    collapse into each other (thr_low >= thr_high).
    """
    # We re-implement the threshold search here (instead of importing from
    # evaluate.py) to keep the module self-contained for production use —
    # `evaluate.py` pulls in matplotlib + W&B which we don't need at inference.
    from sklearn.metrics import precision_recall_curve

    prec, rec, threshs = precision_recall_curve(y_true, y_prob)
    prec_t, rec_t = prec[:-1], rec[:-1]   # align with `threshs`

    # thr_high — auto-bad band
    mask_p = prec_t >= auto_bad_precision
    if not mask_p.any():
        raise ValueError(
            f"No threshold achieves precision >= {auto_bad_precision:.2f}. "
            f"Max OOF precision = {prec_t.max():.3f}. "
            "Lower auto_bad_precision or improve the model."
        )
    valid_p = np.where(mask_p)[0]
    thr_high = float(threshs[valid_p[np.argmin(threshs[valid_p])]])

    # thr_low — auto-good band
    mask_r = rec_t >= auto_good_recall
    if not mask_r.any():
        raise ValueError(
            f"No threshold achieves recall >= {auto_good_recall:.2f}. "
            f"Max OOF recall = {rec_t.max():.3f}. "
            "Lower auto_good_recall."
        )
    valid_r = np.where(mask_r)[0]
    thr_low = float(threshs[valid_r[np.argmax(threshs[valid_r])]])

    if thr_low >= thr_high:
        raise ValueError(
            f"Triage bands collapsed: thr_low ({thr_low:.4f}) >= "
            f"thr_high ({thr_high:.4f}). The model lacks the dynamic range to "
            "support a two-tier policy with these targets — relax either "
            "auto_bad_precision or auto_good_recall."
        )

    logger.info(
        f"Calibrated: thr_low={thr_low:.4f}  thr_high={thr_high:.4f}"
    )
    return thr_low, thr_high


# ── Classification ─────────────────────────────────────────────────────────────


def classify_channels(
    y_prob: np.ndarray, thr_low: float, thr_high: float
) -> np.ndarray:
    """Return a numpy array of tier labels for each prediction.

    Output dtype is `<U10` (unicode strings ≤10 chars) for readability.
    Use `tiers == TIER_AUTO_BAD` etc. for masks.
    """
    if thr_low >= thr_high:
        raise ValueError(
            f"thr_low ({thr_low}) must be < thr_high ({thr_high})"
        )
    tiers = np.full(len(y_prob), TIER_REVIEW, dtype="<U10")
    tiers[y_prob <= thr_low]  = TIER_AUTO_GOOD
    tiers[y_prob >= thr_high] = TIER_AUTO_BAD
    return tiers


def triage_statistics(
    y_true: np.ndarray, y_prob: np.ndarray, thr_low: float, thr_high: float
) -> dict[str, dict[str, Any]]:
    """Compute per-tier counts + observed bad-rate.

    Returns a dict::
        {
          "auto_good": {"n": ..., "n_bad": ..., "bad_rate": ...,
                        "pct_of_all": ..., "pct_of_bads": ...},
          "review":    {...},
          "auto_bad":  {...},
        }
    """
    tiers = classify_channels(y_prob, thr_low, thr_high)
    n_total = len(y_true)
    n_total_bads = int(y_true.sum())

    out: dict[str, dict[str, Any]] = {}
    for tier in (TIER_AUTO_GOOD, TIER_REVIEW, TIER_AUTO_BAD):
        mask = tiers == tier
        n = int(mask.sum())
        if n == 0:
            out[tier] = {
                "n": 0, "n_bad": 0, "bad_rate": 0.0,
                "pct_of_all": 0.0, "pct_of_bads": 0.0,
            }
            continue
        n_bad = int(y_true[mask].sum())
        out[tier] = {
            "n":           n,
            "n_bad":       n_bad,
            "bad_rate":    round(n_bad / n, 4),
            "pct_of_all":  round(n / n_total, 4),
            "pct_of_bads": round(n_bad / max(n_total_bads, 1), 4),
        }
    return out


# ── Review-queue ordering ──────────────────────────────────────────────────────


def review_queue(
    y_prob: np.ndarray,
    *,
    thr_low: float,
    thr_high: float,
    channel_indices: np.ndarray | None = None,
    channel_labels: np.ndarray | None = None,
    extra_cols: dict[str, np.ndarray] | None = None,
):
    """Return the review-tier channels sorted by P(bad) descending.

    Highest-probability ambiguous channels surface first so a human reviewer
    can stop early once they've handled the most-likely bads.

    Parameters
    ----------
    y_prob          : per-channel P(bad).
    thr_low, thr_high : calibrated triage thresholds.
    channel_indices : optional original indices (default 0..n-1).
    channel_labels  : optional channel name strings (e.g. 'T7', 'O1', ...).
    extra_cols      : optional additional columns to attach (e.g. true labels
                      for audit when calibrating off OOF predictions).

    Returns
    -------
    pandas.DataFrame sorted by probability descending, with columns:
        rank, channel_index, probability, [channel_label,] [extras...]
    """
    import pandas as pd

    tiers = classify_channels(y_prob, thr_low, thr_high)
    review_mask = tiers == TIER_REVIEW

    if channel_indices is None:
        channel_indices = np.arange(len(y_prob))

    queue_probs = y_prob[review_mask]
    queue_idx   = channel_indices[review_mask]
    order = np.argsort(queue_probs)[::-1]

    data: dict[str, Any] = {
        "rank":          np.arange(1, len(queue_probs) + 1),
        "channel_index": queue_idx[order],
        "probability":   queue_probs[order],
    }
    if channel_labels is not None:
        data["channel_label"] = np.asarray(channel_labels)[review_mask][order]
    if extra_cols:
        for k, v in extra_cols.items():
            data[k] = np.asarray(v)[review_mask][order]

    return pd.DataFrame(data)


# ── Production hook ────────────────────────────────────────────────────────────


def predict_and_triage(
    model,
    X: np.ndarray,
    thr_low: float,
    thr_high: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Production-time triage: probs + tier per channel.

    Parameters
    ----------
    model : object exposing `predict_proba(X) -> (n, 2)` (any BaseBCRModel).
    X     : preprocessed feature matrix for the new EEG session.
    thr_low, thr_high : calibrated thresholds (see `load_thresholds`).

    Returns
    -------
    probs    : np.ndarray of shape (n,) — P(bad) per channel.
    tiers    : np.ndarray of shape (n,) — strings in {auto_good, review, auto_bad}.
    """
    probs = model.predict_proba(X)[:, 1]
    tiers = classify_channels(probs, thr_low, thr_high)
    return probs, tiers


# ── Persistence ────────────────────────────────────────────────────────────────


def save_thresholds(
    thr_low: float, thr_high: float, stats: dict, path: Path = THRESHOLDS_PATH,
    auto_bad_precision: float = DEFAULT_AUTO_BAD_PRECISION,
    auto_good_recall:   float = DEFAULT_AUTO_GOOD_RECALL,
) -> None:
    """Persist the calibrated thresholds + tier statistics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "thr_low":             thr_low,
        "thr_high":            thr_high,
        "auto_bad_precision":  auto_bad_precision,
        "auto_good_recall":    auto_good_recall,
        "tier_statistics":     stats,
    }
    path.write_text(json.dumps(payload, indent=2))
    logger.info(f"Thresholds saved -> {path}")


def load_thresholds(path: Path = THRESHOLDS_PATH) -> tuple[float, float]:
    """Load (thr_low, thr_high) from disk."""
    if not path.exists():
        raise FileNotFoundError(
            f"Triage thresholds not found at {path}. "
            "Run: python -m bad_channel_rejection.triage calibrate --from-stage3"
        )
    payload = json.loads(path.read_text())
    return float(payload["thr_low"]), float(payload["thr_high"])


# ── Plotting ───────────────────────────────────────────────────────────────────


def _plot_triage_histogram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thr_low: float,
    thr_high: float,
    out_path: Path,
) -> None:
    """Histogram of P(bad) split by true label, with the two thresholds marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 1, 60)
    ax.hist(
        y_prob[y_true == 0], bins=bins, alpha=0.55, color="#58a6ff",
        label="True good", density=True,
    )
    ax.hist(
        y_prob[y_true == 1], bins=bins, alpha=0.55, color="#f85149",
        label="True bad", density=True,
    )
    ax.axvline(thr_low,  color="black", ls="--", lw=1.4,
               label=f"thr_low = {thr_low:.4f}  (auto-good cutoff)")
    ax.axvline(thr_high, color="black", ls="-",  lw=1.4,
               label=f"thr_high = {thr_high:.4f}  (auto-bad cutoff)")

    # Tier shading
    ax.axvspan(0,        thr_low,  alpha=0.06, color="green")
    ax.axvspan(thr_low,  thr_high, alpha=0.06, color="orange")
    ax.axvspan(thr_high, 1,        alpha=0.06, color="red")

    ax.set_xlabel("P(bad)")
    ax.set_ylabel("density")
    ax.set_title("Triage bands over OOF probabilities")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim([0, 1])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Histogram saved -> {out_path}")


def _write_triage_report(
    thr_low: float, thr_high: float, stats: dict, out_path: Path,
    auto_bad_precision: float, auto_good_recall: float,
) -> None:
    n_total      = sum(s["n"] for s in stats.values())
    n_total_bads = sum(s["n_bad"] for s in stats.values())

    lines = [
        "# BCR Triage — Two-Tier Human Review Pipeline",
        "",
        "## Thresholds (calibrated on OOF predictions)",
        "",
        f"| Threshold | Value | Meaning |",
        f"|-----------|-------|---------|",
        f"| `thr_low`  | **{thr_low:.4f}** | "
        f"P(bad) ≤ this → **auto-accept** (recall≥{auto_good_recall:.0%} guarantee) |",
        f"| `thr_high` | **{thr_high:.4f}** | "
        f"P(bad) ≥ this → **auto-interpolate** (precision≥{auto_bad_precision:.0%}) |",
        "",
        "## Tier statistics (over OOF)",
        "",
        "| Tier | n channels | % of all | n bads in tier | bad-rate in tier | % of all bads |",
        "|------|-----------:|---------:|---------------:|-----------------:|--------------:|",
    ]
    pretty = {
        TIER_AUTO_GOOD: "🟢 Auto-accept",
        TIER_REVIEW:    "🟡 Review",
        TIER_AUTO_BAD:  "🔴 Auto-interpolate",
    }
    for tier in (TIER_AUTO_GOOD, TIER_REVIEW, TIER_AUTO_BAD):
        s = stats[tier]
        lines.append(
            f"| {pretty[tier]} | {s['n']:,} | {s['pct_of_all']:.1%} | "
            f"{s['n_bad']:,} | {s['bad_rate']:.1%} | {s['pct_of_bads']:.1%} |"
        )
    lines += [
        "",
        f"**Totals:** {n_total:,} channels — {n_total_bads:,} bads "
        f"({n_total_bads / max(n_total, 1):.1%} bad rate).",
        "",
        "## Operational interpretation",
        "",
        f"- The **auto-accept** tier ({stats[TIER_AUTO_GOOD]['pct_of_all']:.0%} of all "
        f"channels) misses {stats[TIER_AUTO_GOOD]['pct_of_bads']:.1%} of all bad "
        f"channels — within the {1-auto_good_recall:.0%} miss budget.",
        f"- The **auto-interpolate** tier "
        f"({stats[TIER_AUTO_BAD]['pct_of_all']:.1%} of all channels) is "
        f"{stats[TIER_AUTO_BAD]['bad_rate']:.0%}-pure — i.e. of every 100 channels "
        "auto-flagged, ~{:.0f} are real bads.".format(stats[TIER_AUTO_BAD]['bad_rate']*100),
        f"- The **review** tier ({stats[TIER_REVIEW]['pct_of_all']:.1%} of all "
        f"channels) is what an expert needs to look at. It contains "
        f"{stats[TIER_REVIEW]['pct_of_bads']:.0%} of all bad channels at a "
        f"{stats[TIER_REVIEW]['bad_rate']:.1%} bad-rate.",
        "",
        "## Production usage",
        "",
        "```python",
        "from bad_channel_rejection.triage import (",
        "    load_thresholds, predict_and_triage,",
        ")",
        "import joblib",
        "",
        'thr_low, thr_high = load_thresholds()                      # results/triage_thresholds.json',
        'model = joblib.load("results/best_model.pkl")              # Stage 3 winner',
        "probs, tiers = predict_and_triage(model, X_session, thr_low, thr_high)",
        "",
        "# auto-interpolate the auto_bad ones",
        "from bad_channel_rejection.interpolation import interpolate_bad_channels",
        'bad_idx = np.where(tiers == "auto_bad")[0].tolist()',
        "eeg_repaired = interpolate_bad_channels(eeg, bad_idx, ch_names)",
        "",
        "# queue the review tier for an expert",
        'review_idx = np.where(tiers == "review")[0].tolist()',
        "```",
    ]
    out_path.write_text("\n".join(lines))
    logger.info(f"Report saved -> {out_path}")


# ── Calibration command ───────────────────────────────────────────────────────


def _load_oof_predictions(from_stage3: bool, tag: str | None) -> tuple[np.ndarray, np.ndarray]:
    if from_stage3:
        ytrue_path = RESULTS_DIR / "oof_y_true_best.npy"
        yprob_path = RESULTS_DIR / "oof_y_prob_best.npy"
    else:
        if not tag:
            raise ValueError("Either --from-stage3 or --tag must be provided.")
        ytrue_path = RESULTS_DIR / f"oof_y_true_{tag}.npy"
        yprob_path = RESULTS_DIR / f"oof_y_prob_{tag}.npy"
    if not ytrue_path.exists() or not yprob_path.exists():
        raise FileNotFoundError(
            f"OOF files not found:\n  {ytrue_path}\n  {yprob_path}"
        )
    return np.load(ytrue_path), np.load(yprob_path)


def calibrate(
    from_stage3: bool = True,
    tag: str | None = None,
    auto_bad_precision: float = DEFAULT_AUTO_BAD_PRECISION,
    auto_good_recall:   float = DEFAULT_AUTO_GOOD_RECALL,
) -> dict:
    """End-to-end calibration: load OOF, compute thresholds, save artefacts."""
    y_true, y_prob = _load_oof_predictions(from_stage3, tag)
    logger.info(
        f"Loaded OOF — n={len(y_true)}  bad_rate={y_true.mean():.4f}"
    )

    thr_low, thr_high = compute_triage_thresholds(
        y_true, y_prob,
        auto_bad_precision=auto_bad_precision,
        auto_good_recall=auto_good_recall,
    )
    stats = triage_statistics(y_true, y_prob, thr_low, thr_high)

    logger.info("Triage tier breakdown:")
    for tier in (TIER_AUTO_GOOD, TIER_REVIEW, TIER_AUTO_BAD):
        s = stats[tier]
        logger.info(
            f"  {tier:<10}: n={s['n']:>5}  "
            f"({s['pct_of_all']:.1%} of all, {s['pct_of_bads']:.1%} of bads, "
            f"bad_rate={s['bad_rate']:.1%})"
        )

    save_thresholds(
        thr_low, thr_high, stats,
        auto_bad_precision=auto_bad_precision,
        auto_good_recall=auto_good_recall,
    )
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _plot_triage_histogram(
        y_true, y_prob, thr_low, thr_high,
        FIGURES_DIR / "triage_histogram.png",
    )
    _write_triage_report(
        thr_low, thr_high, stats,
        RESULTS_DIR / "triage_report.md",
        auto_bad_precision=auto_bad_precision,
        auto_good_recall=auto_good_recall,
    )

    # OOF review queue — sorted by descending P(bad). Useful as an audit log:
    # the top of this CSV is what an expert would tackle first if they were
    # working through the dataset that produced these OOF predictions.
    queue_df = review_queue(
        y_prob,
        thr_low=thr_low,
        thr_high=thr_high,
        extra_cols={"true_label": y_true},
    )
    queue_path = RESULTS_DIR / "review_queue_oof.csv"
    queue_df.to_csv(queue_path, index=False)
    logger.info(
        f"Review queue saved -> {queue_path}  "
        f"(n={len(queue_df)}; sorted by P(bad) desc)"
    )

    return {
        "thr_low":  thr_low,
        "thr_high": thr_high,
        "tier_statistics": stats,
        "review_queue_size": len(queue_df),
    }


# ── CLI ────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="BCR triage — two-tier human-in-the-loop pipeline"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("calibrate", help="Calibrate thresholds from OOF")
    pc.add_argument(
        "--from-stage3",
        action="store_true",
        help="Read oof_y_*_best.npy from Stage 3 HPO (recommended).",
    )
    pc.add_argument(
        "--tag",
        default=None,
        help="Alternative: read oof_y_*_<tag>.npy from train.py outputs.",
    )
    pc.add_argument(
        "--auto-bad-precision",
        type=float,
        default=DEFAULT_AUTO_BAD_PRECISION,
        help=f"Precision floor for auto-interpolate band "
             f"(default {DEFAULT_AUTO_BAD_PRECISION}).",
    )
    pc.add_argument(
        "--auto-good-recall",
        type=float,
        default=DEFAULT_AUTO_GOOD_RECALL,
        help=f"Recall floor for the auto-good cutoff: at most "
             f"(1 - auto_good_recall) of bads land in the auto-accept tier "
             f"(default {DEFAULT_AUTO_GOOD_RECALL}).",
    )

    args = parser.parse_args()
    if args.cmd == "calibrate":
        if not args.from_stage3 and not args.tag:
            parser.error("Either --from-stage3 or --tag is required.")
        calibrate(
            from_stage3=args.from_stage3,
            tag=args.tag,
            auto_bad_precision=args.auto_bad_precision,
            auto_good_recall=args.auto_good_recall,
        )


if __name__ == "__main__":
    main()
