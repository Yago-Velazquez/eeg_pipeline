"""
tests/test_label_quality.py

Tests for label_quality.py — Dawid-Skene, entropy, hard threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bad_channel_rejection.label_quality import (
    RATER_COLS,
    build_label_artifacts,
    compute_dawid_skene_labels,
    compute_entropy_weights,
    compute_hard_threshold_labels,
    compute_per_rater_reliability,
    fit_dawid_skene,
)


def test_hard_threshold(raw_df):
    artifacts = compute_hard_threshold_labels(raw_df["Bad (score)"])
    assert artifacts.y_hard.shape == (len(raw_df),)
    assert set(artifacts.y_hard.tolist()) <= {0, 1}
    assert np.allclose(artifacts.sample_weights, 1.0)
    assert artifacts.y_soft is None


def test_entropy_weights_mapping(raw_df):
    votes = raw_df[RATER_COLS].values.astype(int)
    artifacts = compute_entropy_weights(votes)

    for score_val, expected_weight in [
        (0, 1.0),
        (1, 0.189),
        (2, 0.0),
        (3, 0.189),
        (4, 1.0),
    ]:
        mask = raw_df["Bad (score)"].values == score_val
        if mask.sum() > 0:
            actual = artifacts.sample_weights[mask][0]
            assert abs(actual - expected_weight) < 0.01, (
                f"Score {score_val}: expected {expected_weight}, got {actual}"
            )


def test_dawid_skene_convergence(raw_df):
    votes = raw_df[RATER_COLS].values.astype(int)
    q, info = fit_dawid_skene(votes, max_iter=100, tol=1e-4)

    assert q.shape == (len(raw_df), 2)
    assert np.allclose(q.sum(axis=1), 1.0)
    assert info["n_iterations"] < 100, "Should converge before max_iter"
    assert "confusion_matrices" in info


def test_dawid_skene_rater_ordering(raw_df):
    """Site 4a must have higher DS sensitivity than site 3 on this dataset."""
    votes = raw_df[RATER_COLS].values.astype(int)
    _, info = fit_dawid_skene(votes)
    cms = info["confusion_matrices"]
    sens = {r: cms[r][1][1] for r in RATER_COLS}
    assert sens["Bad (site 4a)"] > sens["Bad (site 3)"], (
        "Site 4a should be more sensitive than site 3"
    )
    assert sens["Bad (site 3)"] < 0.2, (
        "Site 3 should have very low sensitivity (near-blind to bad channels)"
    )


def test_dawid_skene_weights_range(raw_df):
    votes = raw_df[RATER_COLS].values.astype(int)
    artifacts = compute_dawid_skene_labels(votes, use_soft_target=False)
    assert (artifacts.sample_weights >= 0).all()
    assert (artifacts.sample_weights <= 1.0).all()
    assert artifacts.y_soft is not None
    assert (artifacts.y_soft >= 0).all() and (artifacts.y_soft <= 1).all()


def test_dawid_skene_soft_full_confidence_weights(raw_df):
    """use_soft_target=True: w = P(chosen label | q_i).

    Weight equals q_bad when y_hard == 1, and 1 - q_bad when y_hard == 0.
    Minimum weight is 0.5 (at q=0.5); maximum is ~1.0 (at q in {0, 1}).
    Hard labels match the non-soft variant (same threshold q >= 0.5).
    """
    votes = raw_df[RATER_COLS].values.astype(int)
    soft = compute_dawid_skene_labels(votes, use_soft_target=True)
    hard = compute_dawid_skene_labels(votes, use_soft_target=False)

    assert np.array_equal(soft.y_hard, hard.y_hard), (
        "Both variants must produce identical hard labels"
    )

    q = soft.y_soft
    expected_w = np.where(soft.y_hard == 1, q, 1.0 - q)
    assert np.allclose(soft.sample_weights, expected_w, atol=1e-6)

    assert (soft.sample_weights >= 0.5 - 1e-6).all(), (
        "Full-confidence weights should never drop below 0.5"
    )
    assert (soft.sample_weights <= 1.0 + 1e-6).all()


@pytest.mark.parametrize(
    "strategy",
    ["hard_threshold", "entropy_weights", "dawid_skene", "dawid_skene_soft"],
)
def test_build_label_artifacts_dispatch(raw_df, strategy):
    artifacts = build_label_artifacts(raw_df, strategy=strategy)
    assert artifacts.y_hard.shape == (len(raw_df),)
    assert artifacts.sample_weights.shape == (len(raw_df),)
    assert artifacts.strategy is not None


def test_per_rater_reliability(raw_df):
    rel = compute_per_rater_reliability(raw_df)
    assert set(rel.keys()) == set(RATER_COLS)
    for r, v in rel.items():
        assert 0.0 <= v <= 1.0
    assert rel["Bad (site 4b)"] > rel["Bad (site 2)"], (
        "Site 4b should agree with consensus more than site 2"
    )


def test_validate_votes_rejects_bad_input():
    from bad_channel_rejection.label_quality import _validate_votes
    with pytest.raises(ValueError):
        _validate_votes(np.array([[0, 1, 2, 0]]))
    with pytest.raises(ValueError):
        _validate_votes(np.array([0, 1, 1, 0]))
    with pytest.raises(ValueError):
        _validate_votes(np.zeros((10, 5), dtype=int))
