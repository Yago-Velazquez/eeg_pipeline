"""
tests/test_feature_engineering.py

Unit + integration tests for the label-aware feature-engineering chain.

Covers
------
- Per-channel bad-rate encoder: fit correctness, unseen-channel fallback,
  replaces channel_label_enc.
- SpatialFeaturePruner: top-K selection, non-spatial passthrough, MI is
  fit on train labels only (no val leakage).
- ImpedanceInteractionFeatures: correct column count, exact product
  values, raw decomposition features retained.
- FeatureEngineeringPipeline end-to-end on the real CSV via the
  preprocess_fold helper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bad_channel_rejection.feature_engineering import (
    DECOMPOSITION_PATTERNS,
    SPATIAL_PATTERNS,
    ChannelBadRateEncoder,
    FeatureEngineeringPipeline,
    ImpedanceInteractionFeatures,
    SpatialFeaturePruner,
)
from bad_channel_rejection.features import FeaturePreprocessor, preprocess_fold


# ---------------------------------------------------------------------------
# ChannelBadRateEncoder
# ---------------------------------------------------------------------------


def test_channel_bad_rate_encoder_basic_fit():
    rng = np.random.default_rng(0)
    n = 200
    channels = rng.choice(["A", "B", "C"], size=n)
    y = np.where(channels == "A", 1, 0)  # A always bad, B/C always good
    X = rng.normal(size=(n, 4))
    names = ["f0", "f1", "f2", "channel_label_enc"]

    enc = ChannelBadRateEncoder()
    X_out, names_out = enc.fit_transform(X, names, y, channels)

    assert enc.rates_["A"] == pytest.approx(1.0)
    assert enc.rates_["B"] == pytest.approx(0.0)
    assert enc.rates_["C"] == pytest.approx(0.0)
    assert enc.global_rate_ == pytest.approx(y.mean())
    assert "channel_bad_rate" in names_out
    assert "channel_label_enc" not in names_out
    assert X_out.shape == (n, len(names_out))


def test_channel_bad_rate_encoder_unseen_channel_fallback():
    """A channel that appears only in val must fall back to train global rate."""
    rng = np.random.default_rng(1)
    n_tr = 100
    tr_channels = rng.choice(["A", "B"], size=n_tr)
    y_tr = np.where(tr_channels == "A", 1, 0)
    X_tr = rng.normal(size=(n_tr, 3))
    names = ["f0", "f1", "channel_label_enc"]

    enc = ChannelBadRateEncoder()
    enc.fit(X_tr, names, y_tr, tr_channels)
    expected_global = float(y_tr.mean())
    assert enc.global_rate_ == pytest.approx(expected_global)
    assert set(enc.rates_) == {"A", "B"}

    # Val contains an unseen channel "C"
    val_channels = np.array(["A", "C", "C", "B", "C"])
    X_va = rng.normal(size=(len(val_channels), 3))
    X_out, names_out = enc.transform(X_va, names, val_channels)
    # Last col is channel_bad_rate
    bad_rate_col = X_out[:, names_out.index("channel_bad_rate")]
    assert bad_rate_col[0] == pytest.approx(enc.rates_["A"])
    assert bad_rate_col[1] == pytest.approx(expected_global)  # C -> fallback
    assert bad_rate_col[2] == pytest.approx(expected_global)
    assert bad_rate_col[3] == pytest.approx(enc.rates_["B"])
    assert bad_rate_col[4] == pytest.approx(expected_global)


def test_channel_bad_rate_encoder_replaces_ordinal():
    rng = np.random.default_rng(2)
    n = 50
    channels = rng.choice(["A", "B"], size=n)
    y = rng.integers(0, 2, size=n)
    X = rng.normal(size=(n, 3))
    names = ["f0", "channel_label_enc", "f1"]

    enc = ChannelBadRateEncoder()
    X_out, names_out = enc.fit_transform(X, names, y, channels)

    assert names_out == ["f0", "f1", "channel_bad_rate"]
    # The original f0 column must be preserved in the same position
    assert np.allclose(X_out[:, 0], X[:, 0])
    assert np.allclose(X_out[:, 1], X[:, 2])


# ---------------------------------------------------------------------------
# SpatialFeaturePruner
# ---------------------------------------------------------------------------


def _make_spatial_names(n_spatial: int) -> list[str]:
    return [f" Correllation with neighbors stat{i}" for i in range(n_spatial)]


def test_spatial_pruner_keeps_top_k_and_passthrough():
    rng = np.random.default_rng(3)
    n = 600
    y = rng.integers(0, 2, size=n)

    spatial_names = _make_spatial_names(6)
    nonspatial_names = ["f_nonspatial_0", "f_nonspatial_1", "impedance_missing"]
    names = spatial_names + nonspatial_names

    # Make spatial feature #2 strongly correlated with y, the rest noise
    X = rng.normal(size=(n, len(names)))
    X[:, 2] = y + rng.normal(scale=0.1, size=n)

    pruner = SpatialFeaturePruner(top_k=3)
    X_out, names_out = pruner.fit_transform(X, names, y)

    assert len(pruner.kept_spatial_) == 3
    assert spatial_names[2] in pruner.kept_spatial_
    # All non-spatial columns survive
    for nn in nonspatial_names:
        assert nn in names_out
    # Exactly 3 of the 6 spatial columns remain
    kept_spatial_out = [n for n in names_out if n in spatial_names]
    assert len(kept_spatial_out) == 3


def test_spatial_pruner_fit_on_train_only():
    """MI-based selection must be determined by train labels, not val."""
    rng = np.random.default_rng(4)
    n = 400

    spatial_names = _make_spatial_names(5)
    names = spatial_names + ["f_other"]

    # Spatial #0 strongly predicts y on train half
    X = rng.normal(size=(n, len(names)))
    y = rng.integers(0, 2, size=n)

    X_tr = X[: n // 2].copy()
    X_va = X[n // 2:].copy()
    y_tr = y[: n // 2]
    # Inject strong signal ONLY into train copy
    X_tr[:, 0] = y_tr + rng.normal(scale=0.01, size=len(y_tr))

    pruner = SpatialFeaturePruner(top_k=1)
    pruner.fit(X_tr, names, y_tr)
    assert pruner.kept_spatial_ == [spatial_names[0]]

    # Transform must use only the fit-time selection — val labels unused
    X_va_out, names_va = pruner.transform(X_va, names)
    kept = [n for n in names_va if n in spatial_names]
    assert kept == [spatial_names[0]]
    # The surviving spatial column in X_va_out equals X_va[:, 0] (pass-through)
    assert np.allclose(
        X_va_out[:, names_va.index(spatial_names[0])], X_va[:, 0]
    )


# ---------------------------------------------------------------------------
# ImpedanceInteractionFeatures
# ---------------------------------------------------------------------------


def test_impedance_interactions_adds_k_cols_and_keeps_originals():
    rng = np.random.default_rng(5)
    n = 300
    y = rng.integers(0, 2, size=n)

    decomp_names = [f" PCA stat{i}" for i in range(8)]
    names = ["impedance_missing"] + decomp_names + ["f_other"]

    X = rng.normal(size=(n, len(names)))
    X[:, 0] = rng.integers(0, 2, size=n)  # impedance_missing binary
    # Make decomposition feature #3 correlated with y
    X[:, 1 + 3] = y + rng.normal(scale=0.1, size=n)

    top_k = 5
    inter = ImpedanceInteractionFeatures(top_k=top_k)
    X_out, names_out = inter.fit_transform(X, names, y)

    assert len(names_out) == len(names) + top_k
    # Original columns preserved
    for nn in names:
        assert nn in names_out
    # Exactly 5 new interaction columns named impedance_missing_x_*
    new_cols = [n for n in names_out if n.startswith("impedance_missing_x_")]
    assert len(new_cols) == top_k


def test_impedance_interaction_values_are_exact_products():
    rng = np.random.default_rng(6)
    n = 100
    y = rng.integers(0, 2, size=n)

    decomp_names = [" PCA stat0", " PCA stat1", " ICA stat0"]
    names = ["impedance_missing"] + decomp_names
    X = rng.normal(size=(n, len(names)))
    imp = rng.integers(0, 2, size=n).astype(float)
    X[:, 0] = imp

    inter = ImpedanceInteractionFeatures(top_k=2)
    X_out, names_out = inter.fit_transform(X, names, y)

    for dname in inter.top_decomp_names_:
        interaction_col = f"impedance_missing_x_{dname.strip()}"
        idx_new = names_out.index(interaction_col)
        idx_src = names.index(dname)
        expected = X[:, idx_src] * imp
        assert np.allclose(X_out[:, idx_new], expected)


# ---------------------------------------------------------------------------
# FeatureEngineeringPipeline — end-to-end on real data
# ---------------------------------------------------------------------------


def test_pipeline_end_to_end_on_real_data(feature_matrix):
    """Full chain: FeaturePreprocessor -> FeatureEngineeringPipeline."""
    X_df = pd.DataFrame(
        feature_matrix["X"], columns=feature_matrix["feature_cols"]
    )
    y = feature_matrix["y_hard"]
    channel_labels = feature_matrix["channel_labels"]

    n = len(y)
    split = int(0.8 * n)

    X_tr_out, X_va_out, names_out, _, fe = preprocess_fold(
        X_df.iloc[:split],
        X_df.iloc[split:],
        y[:split],
        channel_labels_tr=channel_labels[:split],
        channel_labels_va=channel_labels[split:],
        use_engineered_features=True,
    )

    assert fe is not None
    assert X_tr_out.shape[0] == split
    assert X_va_out.shape[0] == n - split
    assert X_tr_out.shape[1] == X_va_out.shape[1] == len(names_out)
    assert not np.isnan(X_tr_out).any()
    assert not np.isnan(X_va_out).any()

    # channel_label_enc replaced with channel_bad_rate
    assert "channel_label_enc" not in names_out
    assert "channel_bad_rate" in names_out

    # Exactly 3 spatial columns kept
    spatial_cols = [n for n in names_out
                    if any(p in n for p in SPATIAL_PATTERNS)]
    assert len(spatial_cols) == 3

    # 5 interaction columns added
    interaction_cols = [n for n in names_out
                        if n.startswith("impedance_missing_x_")]
    assert len(interaction_cols) == 5


def test_pipeline_respects_fold_split_for_bad_rate(feature_matrix):
    """Bad-rate encoder must use only the training fold's labels."""
    X_df = pd.DataFrame(
        feature_matrix["X"], columns=feature_matrix["feature_cols"]
    )
    y = feature_matrix["y_hard"]
    channel_labels = feature_matrix["channel_labels"]

    n = len(y)
    # Two different splits — bad-rate values should differ
    split_a = int(0.6 * n)
    split_b = int(0.75 * n)

    prep = FeaturePreprocessor()
    X_pre_a = prep.fit_transform(X_df.iloc[:split_a])
    names_pre_a = list(prep.feature_names_out_)

    prep_b = FeaturePreprocessor()
    X_pre_b = prep_b.fit_transform(X_df.iloc[:split_b])
    names_pre_b = list(prep_b.feature_names_out_)

    enc_a = ChannelBadRateEncoder()
    enc_a.fit(X_pre_a, names_pre_a, y[:split_a], channel_labels[:split_a])

    enc_b = ChannelBadRateEncoder()
    enc_b.fit(X_pre_b, names_pre_b, y[:split_b], channel_labels[:split_b])

    # Same channels learned in both (every channel appears across subjects),
    # but the per-channel rates will differ between training subsets.
    shared = set(enc_a.rates_) & set(enc_b.rates_)
    diffs = [
        abs(enc_a.rates_[c] - enc_b.rates_[c]) for c in shared
    ]
    assert max(diffs) > 0, "Bad-rate encoder not sensitive to the fold split"


def test_pipeline_transform_idempotent(feature_matrix):
    X_df = pd.DataFrame(
        feature_matrix["X"], columns=feature_matrix["feature_cols"]
    )
    y = feature_matrix["y_hard"]
    channel_labels = feature_matrix["channel_labels"]

    prep = FeaturePreprocessor()
    X_pre = prep.fit_transform(X_df)
    fe = FeatureEngineeringPipeline(verbose=False)
    X_a, names_a = fe.fit_transform(
        X_pre, prep.feature_names_out_, y, channel_labels
    )
    X_b, names_b = fe.transform(X_pre, prep.feature_names_out_, channel_labels)
    assert names_a == names_b
    assert np.allclose(X_a, X_b)


def test_pipeline_disable_channel_encoder_does_not_need_channel_labels():
    """If channel encoder is off, the pipeline must not require channel_labels."""
    rng = np.random.default_rng(7)
    n = 100
    X = rng.normal(size=(n, 4))
    names = ["impedance_missing", " PCA stat0", " PCA stat1", "f_other"]
    X[:, 0] = rng.integers(0, 2, size=n)
    y = rng.integers(0, 2, size=n)

    fe = FeatureEngineeringPipeline(
        use_channel_bad_rate=False,
        use_spatial_pruner=False,
        use_impedance_interactions=True,
        verbose=False,
    )
    X_out, names_out = fe.fit_transform(X, names, y, channel_labels=None)
    assert len(names_out) >= len(names)  # raw kept, interactions added


# ---------------------------------------------------------------------------
# Leakage smoke-test via preprocess_fold
# ---------------------------------------------------------------------------


def test_preprocess_fold_no_leakage(feature_matrix):
    """Fitting on one subset must not peek at the other."""
    X_df = pd.DataFrame(
        feature_matrix["X"], columns=feature_matrix["feature_cols"]
    )
    y = feature_matrix["y_hard"]
    cl = feature_matrix["channel_labels"]
    n = len(y)
    split = int(0.7 * n)

    X_tr, X_va, names, _, _ = preprocess_fold(
        X_df.iloc[:split], X_df.iloc[split:], y[:split],
        channel_labels_tr=cl[:split],
        channel_labels_va=cl[split:],
        use_engineered_features=True,
    )
    assert X_tr.shape[1] == X_va.shape[1] == len(names)
    assert not np.isnan(X_va).any()
    assert not np.isinf(X_va).any()
