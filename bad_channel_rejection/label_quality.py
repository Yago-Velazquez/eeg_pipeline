"""
bad_channel_rejection/label_quality.py

Principled label quality estimation for BCR.

This module centralises everything related to deriving labels and per-sample
weights from the 4-rater vote matrix. It replaces the earlier heuristic
DISAGREEMENT_WEIGHTS map with methods grounded in inter-rater agreement theory.

Four label/weight strategies
----------------------------
All four produce the SAME binary targets up to the hard-threshold vs
Dawid-Skene-posterior choice; they differ only in how per-sample weights
are computed. This keeps the classifier interface identical across
conditions (XGBoost cannot consume soft targets).

  1. hard_threshold       : y = 1[score >= 2], w = 1 (uniform)
  2. entropy_weights      : y = 1[score >= 2],
                            w = 1 - H(vote_distribution) / H_max
                            — score=2 (2v2 split) receives weight 0
  3. dawid_skene          : y = 1[q_i >= 0.5],
                            w = |q_i - 0.5| * 2  (hard-confidence;
                            weight 0 at q=0.5, weight 1 at q in {0, 1})
  4. dawid_skene_soft     : y = 1[q_i >= 0.5],
                            w = q_i       if y == 1
                                1 - q_i   if y == 0
                            i.e. the posterior probability assigned to the
                            chosen label. Minimum weight is 0.5 at q=0.5;
                            samples never drop out entirely.

Reference
---------
Dawid, A.P. & Skene, A.M. (1979). Maximum likelihood estimation of observer
error-rates using the EM algorithm. Applied Statistics 28(1), 20-28.

The `dawid-skene` package provides a clean implementation; we wrap it so the
rest of the pipeline does not depend on its exact API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .logging_config import setup_logging

logger = setup_logging(__name__)

RATER_COLS = ["Bad (site 2)", "Bad (site 3)", "Bad (site 4a)", "Bad (site 4b)"]


@dataclass
class LabelArtifacts:
    """Container for everything a labelling strategy produces.

    Attributes
    ----------
    y_hard : np.ndarray (n,) int
        Binary labels {0, 1} for classification training.
    y_soft : np.ndarray (n,) float, optional
        Posterior probabilities P(truly bad) in [0, 1]. Only set for
        Dawid-Skene; None for hard threshold and entropy strategies.
    sample_weights : np.ndarray (n,) float32
        Per-sample training weights. Always non-negative; not necessarily
        summing to anything in particular.
    strategy : str
        Name of the strategy that produced these artifacts.
    metadata : dict
        Strategy-specific diagnostics (e.g., fitted rater confusion matrices).
    """

    y_hard: np.ndarray
    sample_weights: np.ndarray
    strategy: str
    y_soft: np.ndarray | None = None
    metadata: dict | None = None


def _validate_votes(votes: np.ndarray) -> None:
    if votes.ndim != 2:
        raise ValueError(f"votes must be 2-D, got shape {votes.shape}")
    if not np.all((votes == 0) | (votes == 1)):
        raise ValueError("votes must contain only 0 or 1 values")
    if votes.shape[1] != len(RATER_COLS):
        raise ValueError(
            f"expected {len(RATER_COLS)} raters, got {votes.shape[1]}"
        )


def compute_hard_threshold_labels(
    score: pd.Series, threshold: int = 2
) -> LabelArtifacts:
    """Strategy 1: binary label via score >= threshold, uniform weights.

    This is the original BCR label. Included here for parity in ablations.
    """
    y = (score >= threshold).astype(int).values
    w = np.ones(len(y), dtype=np.float32)
    logger.info(
        f"hard_threshold: threshold={threshold}, bad_rate={y.mean():.4f}, "
        f"n_bad={int(y.sum())}/{len(y)}"
    )
    return LabelArtifacts(
        y_hard=y,
        sample_weights=w,
        strategy=f"hard_threshold_score_ge_{threshold}",
    )


def compute_entropy_weights(
    votes: np.ndarray, threshold: int = 2
) -> LabelArtifacts:
    """Strategy 3: weight = 1 - H(vote_distribution) / H_max.

    Weight map for 4 raters:
        score 0 -> weight 1.00 (all good)
        score 1 -> weight 0.19 (1 dissenter)
        score 2 -> weight 0.00 (2 vs 2 split)
        score 3 -> weight 0.19 (1 dissenter)
        score 4 -> weight 1.00 (all bad)

    The score=2 rows get weight 0, meaning they are excluded from the
    gradient entirely. This is a stronger statement than the earlier
    {0.5, 0.75, 1.0} heuristic and reflects that those labels are
    genuinely undecidable.
    """
    _validate_votes(votes)
    n_raters = votes.shape[1]

    k = votes.sum(axis=1).astype(int)
    p_bad = k / n_raters

    safe = (p_bad > 0) & (p_bad < 1)
    H = np.zeros(len(p_bad), dtype=np.float64)
    H[safe] = -(
        p_bad[safe] * np.log2(p_bad[safe])
        + (1 - p_bad[safe]) * np.log2(1 - p_bad[safe])
    )
    H_max = 1.0
    w = (1.0 - H / H_max).astype(np.float32)

    y = (k >= threshold).astype(int)

    uniq, counts = np.unique(k, return_counts=True)
    logger.info("entropy_weights distribution:")
    for score_val, cnt in zip(uniq, counts):
        mask = k == score_val
        example_w = float(w[mask][0]) if cnt > 0 else float("nan")
        logger.info(
            f"  score={score_val}  weight={example_w:.3f}  "
            f"count={cnt}  ({cnt/len(k)*100:.2f}%)"
        )
    logger.info(
        f"entropy_weights mean={w.mean():.4f}, "
        f"effective sample reduction={(1 - w.mean())*100:.1f}%"
    )

    return LabelArtifacts(
        y_hard=y,
        sample_weights=w,
        strategy="entropy_weights",
        metadata={"mean_weight": float(w.mean())},
    )


def fit_dawid_skene(
    votes: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-4,
) -> tuple[np.ndarray, dict]:
    """Fit Dawid-Skene EM model to rater vote matrix.

    Parameters
    ----------
    votes : np.ndarray (n_samples, n_raters)
        Binary vote matrix. Values must be 0 or 1.
    max_iter : int
        Maximum EM iterations.
    tol : float
        Convergence tolerance on log-likelihood change.

    Returns
    -------
    posteriors : np.ndarray (n_samples, 2)
        Columns are P(truly good), P(truly bad).
    fit_info : dict
        Diagnostic info: per-rater 2x2 confusion matrices, prevalence,
        iterations to convergence, final log-likelihood.

    Notes
    -----
    Implemented from scratch rather than via the `dawid-skene` package to
    avoid external API drift. The algorithm is a standard two-class,
    per-rater-confusion-matrix EM:

      E-step: q_i = P(z_i=1 | votes) under current pi_r, prevalence p
      M-step: re-estimate pi_r and p from q_i

    Reference: Dawid & Skene (1979).
    """
    _validate_votes(votes)
    n, R = votes.shape

    prevalence = np.array([0.95, 0.05])
    pi = np.zeros((R, 2, 2))
    for r in range(R):
        pi[r] = np.array([[0.9, 0.1], [0.1, 0.9]])

    prev_loglik = -np.inf
    for iteration in range(max_iter):
        log_q = np.log(prevalence + 1e-12)[None, :].repeat(n, axis=0)
        for r in range(R):
            vr = votes[:, r]
            log_pi_r = np.log(pi[r] + 1e-12)
            log_q += log_pi_r[:, vr].T

        log_q_max = log_q.max(axis=1, keepdims=True)
        log_norm = log_q_max + np.log(
            np.exp(log_q - log_q_max).sum(axis=1, keepdims=True)
        )
        log_posterior = log_q - log_norm
        q = np.exp(log_posterior)

        loglik = float(log_norm.sum())

        new_prevalence = q.mean(axis=0)
        new_pi = np.zeros_like(pi)
        for r in range(R):
            vr = votes[:, r]
            for true_cls in range(2):
                denom = q[:, true_cls].sum() + 1e-12
                for obs_cls in range(2):
                    mask = vr == obs_cls
                    new_pi[r, true_cls, obs_cls] = (
                        q[mask, true_cls].sum() / denom
                    )

        prevalence = new_prevalence
        pi = new_pi

        if abs(loglik - prev_loglik) < tol:
            logger.info(
                f"Dawid-Skene converged after {iteration+1} iterations "
                f"(loglik={loglik:.2f})"
            )
            break
        prev_loglik = loglik
    else:
        logger.warning(
            f"Dawid-Skene did not converge in {max_iter} iterations; "
            f"final loglik={loglik:.2f}"
        )

    fit_info = {
        "prevalence": prevalence.tolist(),
        "confusion_matrices": {
            RATER_COLS[r]: pi[r].tolist() for r in range(R)
        },
        "n_iterations": iteration + 1,
        "loglik": loglik,
    }

    for r in range(R):
        sensitivity = pi[r, 1, 1]
        specificity = pi[r, 0, 0]
        logger.info(
            f"  {RATER_COLS[r]}: sensitivity={sensitivity:.3f}  "
            f"specificity={specificity:.3f}"
        )

    return q, fit_info


def compute_dawid_skene_labels(
    votes: np.ndarray,
    use_soft_target: bool = False,
) -> LabelArtifacts:
    """Strategy 2 / 4: Dawid-Skene posteriors.

    Both variants return the SAME hard labels ``y_hard = 1[q_i >= 0.5]``;
    they differ only in the per-sample weight scheme. XGBoost (and other
    classifier backends) cannot consume soft targets, so the posterior is
    instead injected through ``sample_weight``.

    Parameters
    ----------
    votes : np.ndarray (n_samples, 4) in {0, 1}
        Rater vote matrix.
    use_soft_target : bool
        If False (strategy ``dawid_skene``):
            w = |q_i - 0.5| * 2
            — hard-confidence. Samples at q=0.5 get weight 0 (excluded);
              samples at q in {0, 1} get weight 1.
        If True (strategy ``dawid_skene_soft``):
            w = q_i        if y_hard == 1
                1 - q_i    if y_hard == 0
            — full-confidence: the posterior probability assigned to the
              chosen label. Weight is 0.5 at q=0.5, rises to ~1.0 as q
              approaches 0 or 1. Samples never drop out entirely.

    Returns
    -------
    LabelArtifacts
    """
    _validate_votes(votes)
    q, fit_info = fit_dawid_skene(votes)
    q_bad = q[:, 1].astype(np.float32)

    y_hard = (q_bad >= 0.5).astype(int)

    if use_soft_target:
        w = np.where(y_hard == 1, q_bad, 1.0 - q_bad).astype(np.float32)
        strategy = "dawid_skene_full_confidence"
    else:
        w = (np.abs(q_bad - 0.5) * 2.0).astype(np.float32)
        strategy = "dawid_skene_hard_plus_confidence"

    logger.info(
        f"dawid_skene: bad_rate={y_hard.mean():.4f}  "
        f"q_mean={q_bad.mean():.4f}  q_median={float(np.median(q_bad)):.4f}"
    )
    logger.info(
        f"  confidence weights: mean={w.mean():.4f}  "
        f"min={w.min():.4f}  max={w.max():.4f}"
    )

    return LabelArtifacts(
        y_hard=y_hard,
        y_soft=q_bad,
        sample_weights=w,
        strategy=strategy,
        metadata=fit_info,
    )


def compute_per_rater_reliability(
    df: pd.DataFrame, threshold: int = 2
) -> dict:
    """Compute each rater's accuracy vs the 4-rater consensus.

    Useful for reporting and sanity-checking the Dawid-Skene confusion
    matrices against a simpler benchmark.
    """
    consensus = (df[RATER_COLS].sum(axis=1) >= threshold).astype(int).values
    reliability = {}
    for col in RATER_COLS:
        agree = (df[col].values == consensus).mean()
        reliability[col] = float(agree)
    return reliability


def build_label_artifacts(
    df: pd.DataFrame,
    strategy: Literal[
        "hard_threshold",
        "entropy_weights",
        "dawid_skene",
        "dawid_skene_soft",
    ] = "hard_threshold",
    threshold: int = 2,
) -> LabelArtifacts:
    """Dispatch to the correct strategy. Primary entry point for dataset.py."""
    if strategy == "hard_threshold":
        return compute_hard_threshold_labels(df["Bad (score)"], threshold)
    if strategy == "entropy_weights":
        votes = df[RATER_COLS].values.astype(int)
        return compute_entropy_weights(votes, threshold)
    if strategy == "dawid_skene":
        votes = df[RATER_COLS].values.astype(int)
        return compute_dawid_skene_labels(votes, use_soft_target=False)
    if strategy == "dawid_skene_soft":
        votes = df[RATER_COLS].values.astype(int)
        return compute_dawid_skene_labels(votes, use_soft_target=True)
    raise ValueError(f"Unknown labelling strategy: {strategy!r}")
