"""
tests/test_dataset.py

Integration tests for dataset.py against the real BCR CSV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bad_channel_rejection.dataset import (
    add_missingness_flags,
    build_feature_matrix,
    impute_and_encode_channels,
    load_bcr_data,
)


def test_load_parses_session(raw_df, csv_path):
    df = load_bcr_data(csv_path)
    assert "subject_id" in df.columns
    assert "visit" in df.columns
    assert "site" in df.columns
    assert df.shape[0] == 18900, f"Expected 18900 rows, got {df.shape[0]}"
    assert df["subject_id"].nunique() == 43
    assert set(df["visit"].unique()) == {1, 2, 3, 4}


def test_missingness_flag_visit3(csv_path):
    df = load_bcr_data(csv_path)
    df = add_missingness_flags(df)
    assert "impedance_missing" in df.columns
    assert set(df["impedance_missing"].unique()) <= {0, 1}

    visit_miss = df.groupby("visit")["impedance_missing"].mean()
    assert visit_miss[3] > 0.5, (
        "Visit 3 should have majority impedance_missing"
    )
    for v in [1, 2, 4]:
        assert visit_miss[v] < 0.1, (
            f"Visit {v} should have near-zero missingness"
        )


def test_imputation_removes_nans_and_infs(csv_path):
    df = load_bcr_data(csv_path)
    df = add_missingness_flags(df)
    NON_FEATURE = {
        "Session", "Task", "Channel labels",
        "Bad (site 2)", "Bad (site 3)", "Bad (site 4a)", "Bad (site 4b)",
        "Bad (score)", "Impedance (start)", "Impedance (end)",
        "subject_id", "visit", "site",
    }
    feature_cols = [
        c for c in df.columns
        if c not in NON_FEATURE and df[c].dtype != object
    ]
    df_imp, _ = impute_and_encode_channels(df, feature_cols)

    for c in feature_cols + ["channel_label_enc"]:
        assert not df_imp[c].isnull().any(), f"NaN found in {c}"
        if df_imp[c].dtype != object:
            assert not np.isinf(df_imp[c].values).any(), f"Inf found in {c}"


@pytest.mark.parametrize(
    "strategy",
    ["hard_threshold", "entropy_weights", "dawid_skene", "dawid_skene_soft"],
)
def test_build_feature_matrix_all_strategies(csv_path, strategy):
    out = build_feature_matrix(csv_path, label_strategy=strategy)

    assert out["X"].shape[0] == 18900
    assert out["X"].shape[1] > 100
    assert not np.isnan(out["X"]).any()
    assert not np.isinf(out["X"]).any()

    assert out["y_hard"].shape == (18900,)
    assert set(out["y_hard"].tolist()) <= {0, 1}

    assert out["sample_weights"].shape == (18900,)
    assert (out["sample_weights"] >= 0).all()

    assert out["groups"].shape == (18900,)
    assert len(np.unique(out["groups"])) == 43

    assert out["scale_pos_weight"] > 1.0

    if strategy in ("dawid_skene", "dawid_skene_soft"):
        assert out["y_soft"] is not None
        assert out["y_soft"].shape == (18900,)
        assert (out["y_soft"] >= 0).all() and (out["y_soft"] <= 1).all()
    else:
        assert out["y_soft"] is None


def test_hard_threshold_bad_rate(csv_path):
    out = build_feature_matrix(csv_path, label_strategy="hard_threshold")
    bad_rate = out["y_hard"].mean()
    assert 0.03 < bad_rate < 0.05, (
        f"Expected bad rate ~3.9%, got {bad_rate:.4f}"
    )


def test_entropy_weights_score2_zero(csv_path, raw_df):
    """Score=2 samples must receive weight 0 under entropy strategy."""
    out = build_feature_matrix(csv_path, label_strategy="entropy_weights")
    score2_mask = (raw_df["Bad (score)"] == 2).values
    assert score2_mask.sum() > 0
    assert np.allclose(out["sample_weights"][score2_mask], 0.0), (
        "All score=2 samples must have weight 0"
    )


def test_entropy_weights_unanimous_full(csv_path, raw_df):
    """Score=0 and score=4 must receive weight 1.0."""
    out = build_feature_matrix(csv_path, label_strategy="entropy_weights")
    unanimous_mask = raw_df["Bad (score)"].isin([0, 4]).values
    assert np.allclose(out["sample_weights"][unanimous_mask], 1.0)
