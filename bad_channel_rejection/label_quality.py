"""
bad_channel_rejection/label_quality.py

Principled label quality estimation for BCR.

This module centralises everything related to deriving labels and per-sample
weights from the 4-rater vote matrix. It replaces the earlier heuristic
DISAGREEMENT_WEIGHTS map with methods grounded in inter-rater agreement theory.

Eight label/weight strategies
------------------------------
All eight produce the SAME binary targets (hard labels) + per-sample weights.
They differ in how the latent "true label" posterior is estimated and how
confidence is converted into a weight. The classifier interface is identical
across conditions.

  1. hard_threshold    : y = 1[score >= 2], w = 1 (uniform)
  2. entropy_weights   : y = 1[score >= 2],
                         w = 1 - H(vote_distribution) / H_max
                         — score=2 (2v2 split) receives weight 0
  3. dawid_skene       : y = 1[q_i >= 0.5]  (q from DS EM),
                         w = |q_i - 0.5| * 2
  4. dawid_skene_soft  : y = 1[q_i >= 0.5],
                         w = q_i if y=1, else 1-q_i  (never drops out)
  5. glad              : y = 1[q_i >= 0.5]  (q from GLAD EM),
                         w = |q_i - 0.5| * 2
  6. glad_soft         : same q, w = q_i if y=1, else 1-q_i
  7. mace              : y = 1[q_i >= 0.5]  (q from MACE EM),
                         w = |q_i - 0.5| * 2
  8. mace_soft         : same q, w = q_i if y=1, else 1-q_i

References
----------
Dawid & Skene (1979). Maximum likelihood estimation of observer error-rates
  using the EM algorithm. Applied Statistics 28(1), 20-28.
Whitehill et al. (2009). Whose vote should count more: Optimal integration of
  labels from annotators of unknown expertise. NeurIPS 22.
Hovy et al. (2013). Learning whom to trust with MACE. NAACL-HLT.
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


def fit_glad(
    votes: np.ndarray,
    max_iter: int = 300,
    tol: float = 1e-4,
    lr: float = 0.5,
    beta_clip: float = 5.0,
) -> tuple[np.ndarray, dict]:
    """Fit the GLAD model (Whitehill et al. 2009) to a binary vote matrix.

    Each annotator r has a scalar skill β_r; each item i has a scalar
    easiness α_i.  The probability that annotator r is *correct* on item i
    is s_ir = σ(β_r · exp(α_i)).  Parameters are estimated via EM.

    Numerical safeguards
    --------------------
    - Gradients are normalised by n (for β) and R (for α) so that the
      learning rate is scale-independent of dataset size.
    - β is hard-clipped to [-beta_clip, beta_clip] after each update.
    - α is hard-clipped to [-5, 5] to prevent exp overflow.
    - Prevalence p is floored at half the empirical bad rate to break the
      EM feedback loop where p→0 collapses all posteriors to zero. This is
      equivalent to a mild Dirichlet prior and is necessary for highly
      imbalanced datasets with near-constant annotators.

    Returns
    -------
    posteriors : np.ndarray (n, 2) — columns [P(good), P(bad)]
    fit_info   : dict with rater skills, convergence info, log-likelihood
    """
    _validate_votes(votes)
    n, R = votes.shape

    empirical_bad_rate = float(votes.mean())
    p_floor = empirical_bad_rate * 0.5   # prevents p from collapsing to 0

    beta = np.ones(R, dtype=np.float64)    # rater skills
    alpha = np.zeros(n, dtype=np.float64)  # item easiness
    p = empirical_bad_rate                 # P(z=1)

    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))

    prev_loglik = -np.inf
    loglik = -np.inf

    for iteration in range(max_iter):
        ea = np.exp(np.clip(alpha, -5.0, 5.0))                    # (n,)
        s = _sigmoid(beta[None, :] * ea[:, None])                  # (n, R)

        log_s = np.log(np.clip(s, 1e-12, 1.0))
        log_1ms = np.log(np.clip(1.0 - s, 1e-12, 1.0))
        log_p1 = np.log(p + 1e-12) + (votes * log_s + (1 - votes) * log_1ms).sum(axis=1)
        log_p0 = np.log(1.0 - p + 1e-12) + ((1 - votes) * log_s + votes * log_1ms).sum(axis=1)

        log_max = np.maximum(log_p1, log_p0)
        log_Z = log_max + np.log(np.exp(log_p1 - log_max) + np.exp(log_p0 - log_max))
        q = np.exp(log_p1 - log_Z)
        loglik = float(log_Z.sum())

        # Floor prevents the EM collapse: p→0 ⟹ q→0 ⟹ p→0 cycle.
        p = max(float(q.mean()), p_floor)

        agreement = q[:, None] * votes + (1.0 - q[:, None]) * (1.0 - votes)  # (n, R)
        residual = agreement - s                                               # (n, R)

        # Normalise by n so lr is dataset-size-independent.
        grad_beta = (ea[:, None] * residual).sum(axis=0) / n
        grad_alpha = (beta[None, :] * ea[:, None] * residual).sum(axis=1) / R

        beta += lr * grad_beta
        beta = np.clip(beta, -beta_clip, beta_clip)
        alpha += lr * grad_alpha
        alpha = np.clip(alpha, -5.0, 5.0)

        if abs(loglik - prev_loglik) < tol:
            logger.info(
                f"GLAD converged after {iteration + 1} iterations "
                f"(loglik={loglik:.2f})"
            )
            break
        prev_loglik = loglik
    else:
        logger.warning(
            f"GLAD did not converge in {max_iter} iterations; "
            f"final loglik={loglik:.2f}"
        )

    posteriors = np.column_stack([1.0 - q, q])

    fit_info = {
        "rater_skills": {RATER_COLS[r]: float(beta[r]) for r in range(R)},
        "alpha_mean": float(alpha.mean()),
        "alpha_std": float(alpha.std()),
        "n_iterations": iteration + 1,
        "loglik": loglik,
    }
    for r in range(R):
        logger.info(f"  GLAD {RATER_COLS[r]}: skill β={beta[r]:.3f}")

    return posteriors, fit_info


def fit_mace(
    votes: np.ndarray,
    max_iter: int = 500,
    tol: float = 1e-4,
) -> tuple[np.ndarray, dict]:
    """Fit the MACE model (Hovy et al. 2013) to a binary vote matrix.

    Each annotator r either "knows" the true label (probability 1-θ_r) or
    "spams" (probability θ_r), drawing from their personal Bernoulli(ξ_r).
    The M-step for θ and ξ is analytic; only the E-step requires iteration.

    A tiny annotation-noise term ε = 1e-6 is added so that P(x|T,S=0) is
    never exactly 0 or 1, keeping the E-step numerically stable.

    Returns
    -------
    posteriors : np.ndarray (n, 2) — columns [P(good), P(bad)]
    fit_info   : dict with per-rater spam rates, strategies, log-likelihood
    """
    _validate_votes(votes)
    n, R = votes.shape

    theta = np.full(R, 0.5)               # P(spamming)
    xi = votes.mean(axis=0).astype(float)  # P(spam label = 1)
    p = float(votes.mean())               # P(T=1)
    eps = 1e-6                             # annotation noise for stability

    prev_loglik = -np.inf
    loglik = -np.inf

    for iteration in range(max_iter):
        # P(x_ir | T=t) for t ∈ {0, 1}:
        #   P(x | T=1) = (1-θ)*[(1-ε)*I(x=1) + ε*I(x=0)] + θ*Bern(x; ξ)
        #   P(x | T=0) = (1-θ)*[(1-ε)*I(x=0) + ε*I(x=1)] + θ*Bern(x; ξ)
        spam = xi[None, :] * votes + (1.0 - xi[None, :]) * (1.0 - votes)   # (n, R)
        know_T1 = votes * (1.0 - eps) + (1.0 - votes) * eps               # (n, R)
        know_T0 = (1.0 - votes) * (1.0 - eps) + votes * eps               # (n, R)
        px_T1 = (1.0 - theta[None, :]) * know_T1 + theta[None, :] * spam  # (n, R)
        px_T0 = (1.0 - theta[None, :]) * know_T0 + theta[None, :] * spam  # (n, R)

        # E-step
        log_p1 = np.log(p + 1e-12) + np.log(np.clip(px_T1, 1e-12, 1.0)).sum(axis=1)
        log_p0 = np.log(1.0 - p + 1e-12) + np.log(np.clip(px_T0, 1e-12, 1.0)).sum(axis=1)

        log_max = np.maximum(log_p1, log_p0)
        log_Z = log_max + np.log(np.exp(log_p1 - log_max) + np.exp(log_p0 - log_max))
        q = np.exp(log_p1 - log_Z)         # P(T_i = 1 | data), shape (n,)
        loglik = float(log_Z.sum())

        # M-step
        p = float(q.mean())

        # E[S_ir | x_ir, T_i] = P(spam | x, T=t) marginalised over T
        # P(S=1 | x, T=t) = θ * P(x|spam) / P(x|T=t)
        e_spam_T1 = theta[None, :] * spam / np.clip(px_T1, 1e-12, 1.0)
        e_spam_T0 = theta[None, :] * spam / np.clip(px_T0, 1e-12, 1.0)
        e_spam = np.clip(
            q[:, None] * e_spam_T1 + (1.0 - q[:, None]) * e_spam_T0, 0.0, 1.0
        )  # (n, R)

        theta = np.clip(e_spam.mean(axis=0), 1e-3, 1.0 - 1e-3)
        xi = np.clip(
            (e_spam * votes).sum(axis=0) / (e_spam.sum(axis=0) + 1e-12),
            1e-6, 1.0 - 1e-6,
        )

        if abs(loglik - prev_loglik) < tol:
            logger.info(
                f"MACE converged after {iteration + 1} iterations "
                f"(loglik={loglik:.2f})"
            )
            break
        prev_loglik = loglik
    else:
        logger.warning(
            f"MACE did not converge in {max_iter} iterations; "
            f"final loglik={loglik:.2f}"
        )

    posteriors = np.column_stack([1.0 - q, q])

    fit_info = {
        "spam_rates": {RATER_COLS[r]: float(theta[r]) for r in range(R)},
        "spam_strategies": {RATER_COLS[r]: float(xi[r]) for r in range(R)},
        "n_iterations": iteration + 1,
        "loglik": loglik,
    }
    for r in range(R):
        logger.info(
            f"  MACE {RATER_COLS[r]}: θ={theta[r]:.3f}  ξ={xi[r]:.3f}"
        )

    return posteriors, fit_info


def _posterior_to_artifacts(
    q_bad: np.ndarray,
    strategy_hard: str,
    strategy_soft: str,
    use_soft_target: bool,
    fit_info: dict,
) -> LabelArtifacts:
    """Convert a posterior P(bad) vector into a LabelArtifacts using the
    same hard/soft weight scheme as Dawid-Skene.

    Hard  (use_soft_target=False): w = |q - 0.5| * 2
    Soft  (use_soft_target=True) : w = q if y=1, else 1-q
    """
    y_hard = (q_bad >= 0.5).astype(int)

    if use_soft_target:
        w = np.where(y_hard == 1, q_bad, 1.0 - q_bad).astype(np.float32)
        strategy = strategy_soft
    else:
        w = (np.abs(q_bad - 0.5) * 2.0).astype(np.float32)
        strategy = strategy_hard

    logger.info(
        f"{strategy}: bad_rate={y_hard.mean():.4f}  "
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


def compute_glad_labels(
    votes: np.ndarray,
    use_soft_target: bool = False,
) -> LabelArtifacts:
    """Strategy 5/6: GLAD posteriors → hard labels + confidence weights."""
    _validate_votes(votes)
    posteriors, fit_info = fit_glad(votes)
    q_bad = posteriors[:, 1].astype(np.float32)
    return _posterior_to_artifacts(
        q_bad,
        strategy_hard="glad_hard_confidence",
        strategy_soft="glad_full_confidence",
        use_soft_target=use_soft_target,
        fit_info=fit_info,
    )


def compute_mace_labels(
    votes: np.ndarray,
    use_soft_target: bool = False,
) -> LabelArtifacts:
    """Strategy 7/8: MACE posteriors → hard labels + confidence weights."""
    _validate_votes(votes)
    posteriors, fit_info = fit_mace(votes)
    q_bad = posteriors[:, 1].astype(np.float32)
    return _posterior_to_artifacts(
        q_bad,
        strategy_hard="mace_hard_confidence",
        strategy_soft="mace_full_confidence",
        use_soft_target=use_soft_target,
        fit_info=fit_info,
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


LABEL_STRATEGIES = (
    "hard_threshold",
    "entropy_weights",
    "dawid_skene",
    "dawid_skene_soft",
    "glad",
    "glad_soft",
    "mace",
    "mace_soft",
)


def build_label_artifacts(
    df: pd.DataFrame,
    strategy: str = "hard_threshold",
    threshold: int = 2,
) -> LabelArtifacts:
    """Dispatch to the correct strategy. Primary entry point for dataset.py."""
    if strategy == "hard_threshold":
        return compute_hard_threshold_labels(df["Bad (score)"], threshold)
    votes = df[RATER_COLS].values.astype(int)
    if strategy == "entropy_weights":
        return compute_entropy_weights(votes, threshold)
    if strategy == "dawid_skene":
        return compute_dawid_skene_labels(votes, use_soft_target=False)
    if strategy == "dawid_skene_soft":
        return compute_dawid_skene_labels(votes, use_soft_target=True)
    if strategy == "glad":
        return compute_glad_labels(votes, use_soft_target=False)
    if strategy == "glad_soft":
        return compute_glad_labels(votes, use_soft_target=True)
    if strategy == "mace":
        return compute_mace_labels(votes, use_soft_target=False)
    if strategy == "mace_soft":
        return compute_mace_labels(votes, use_soft_target=True)
    raise ValueError(
        f"Unknown labelling strategy: {strategy!r}. "
        f"Valid options: {LABEL_STRATEGIES}"
    )
