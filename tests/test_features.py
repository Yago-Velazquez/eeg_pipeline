"""
tests/test_features.py

Integration tests for features.py on the real BCR CSV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bad_channel_rejection.features import (
    REDUNDANT_DROP_COLS,
    FeaturePreprocessor,
)


def test_preprocessor_fit_transform(feature_matrix):
    X_df = pd.DataFrame(
        feature_matrix["X"], columns=feature_matrix["feature_cols"]
    )
    prep = FeaturePreprocessor()
    X_out = prep.fit_transform(X_df)

    assert X_out.shape[0] == X_df.shape[0]
    assert X_out.shape[1] < X_df.shape[1], (
        "Preprocessor should drop some columns"
    )
    assert X_out.shape[1] > 100
    assert not np.isnan(X_out).any()
    assert not np.isinf(X_out).any()


def test_preprocessor_transform_idempotent(feature_matrix):
    X_df = pd.DataFrame(
        feature_matrix["X"], columns=feature_matrix["feature_cols"]
    )
    prep = FeaturePreprocessor()
    X_fit = prep.fit_transform(X_df)
    X_again = prep.transform(X_df)
    assert np.allclose(X_fit, X_again)


def test_preprocessor_drops_redundant(feature_matrix):
    X_df = pd.DataFrame(
        feature_matrix["X"], columns=feature_matrix["feature_cols"]
    )
    prep = FeaturePreprocessor()
    prep.fit_transform(X_df)

    for col in REDUNDANT_DROP_COLS:
        if col in X_df.columns:
            assert col not in prep.feature_names_out_, (
                f"Redundant column {col!r} should be dropped"
            )


def test_preprocessor_keeps_impedance_missing(feature_matrix):
    X_df = pd.DataFrame(
        feature_matrix["X"], columns=feature_matrix["feature_cols"]
    )
    prep = FeaturePreprocessor()
    prep.fit_transform(X_df)
    assert "impedance_missing" in prep.feature_names_out_, (
        "impedance_missing must survive preprocessing"
    )


def test_preprocessor_no_leakage(feature_matrix):
    """Preprocessor fitted on train subset must not peek at val."""
    n = feature_matrix["X"].shape[0]
    split = int(0.8 * n)
    X_df = pd.DataFrame(
        feature_matrix["X"], columns=feature_matrix["feature_cols"]
    )

    prep = FeaturePreprocessor()
    X_tr = prep.fit_transform(X_df.iloc[:split])
    X_va = prep.transform(X_df.iloc[split:])

    assert X_tr.shape[1] == X_va.shape[1]
    assert X_tr.shape[0] == split
    assert X_va.shape[0] == n - split
    assert not np.isnan(X_va).any()
    assert not np.isinf(X_va).any()
