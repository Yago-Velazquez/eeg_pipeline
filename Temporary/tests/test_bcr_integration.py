"""
tests/test_bcr_integration.py

Integration smoke tests for BadChannelDetector.

Prerequisites
-------------
- results/bcr_model_thresh2.json       (saved by train.py)
- results/bcr_model_thresh2_meta.json  (saved by migrate_bcr_meta.py or
                                        any save() call after Day 13)
- data/raw/Bad_channels_for_ML.csv

Run
---
    pytest tests/test_bcr_integration.py -v
"""

import time
import pytest
import numpy as np
import pandas as pd

from bad_channel_rejection.model    import BadChannelDetector
from bad_channel_rejection.dataset  import build_feature_matrix
from bad_channel_rejection.features import FeaturePreprocessor

# ── Paths ─────────────────────────────────────────────────────────────────────

MODEL_PATH = "results/bcr_model_thresh2.json"
DATA_PATH  = "data/raw/Bad_channels_for_ML.csv"
N_SAMPLE   = 20   # rows to use for inference tests — includes both classes


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def detector() -> BadChannelDetector:
    """Load the detector once for the entire test module."""
    return BadChannelDetector(model_path=MODEL_PATH)


@pytest.fixture(scope="module")
def sample_data() -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """
    Return (X_df, y, channel_labels) for N_SAMPLE rows.

    Uses build_feature_matrix + FeaturePreprocessor to replicate
    exactly the preprocessing that train.py applied — no shortcuts.
    """
    X_raw, y, groups, feature_cols, _, _sw = build_feature_matrix(
        DATA_PATH, bad_threshold=2
    )
    X_df_raw = pd.DataFrame(X_raw, columns=feature_cols)

    # Fit preprocessor on all data (fine for integration testing —
    # we just need valid preprocessed rows, not unbiased CV scores)
    prep   = FeaturePreprocessor()
    X_proc = prep.fit_transform(X_df_raw)
    X_df   = pd.DataFrame(X_proc, columns=prep.feature_names_out_)

    # Grab N_SAMPLE rows: first 10 good + first 10 bad (guarantees both classes)
    good_idx = np.where(y == 0)[0][:10]
    bad_idx  = np.where(y == 1)[0][:10]
    idx      = np.concatenate([good_idx, bad_idx])

    labels = [f"CH_{i}" for i in range(len(idx))]
    return X_df.iloc[idx].reset_index(drop=True), y[idx], labels


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestLoading:

    def test_loads_without_error(self, detector: BadChannelDetector) -> None:
        """Model loads, reports fitted, threshold matches Day 11 value."""
        assert detector.is_fitted
        assert detector.threshold == pytest.approx(0.604, abs=1e-3)

    def test_feature_names_populated(self, detector: BadChannelDetector) -> None:
        """Sidecar was found and feature_names loaded (138 features expected)."""
        assert detector._feature_names is not None, (
            "feature_names is None — sidecar missing. "
            "Run scripts/migrate_bcr_meta.py to generate it."
        )
        assert len(detector._feature_names) == 138, (
            f"Expected 138 features, got {len(detector._feature_names)}"
        )

    def test_repr_informative(self, detector: BadChannelDetector) -> None:
        r = repr(detector)
        assert "fitted" in r
        assert "138" in r


class TestPredictProba:

    def test_shape(
        self,
        detector: BadChannelDetector,
        sample_data: tuple,
    ) -> None:
        X_df, y, _ = sample_data
        proba = detector.predict_proba(X_df)
        assert proba.shape == (len(X_df),), (
            f"Expected ({len(X_df)},), got {proba.shape}"
        )

    def test_range(
        self,
        detector: BadChannelDetector,
        sample_data: tuple,
    ) -> None:
        X_df, _, _ = sample_data
        proba = detector.predict_proba(X_df)
        assert proba.min() >= 0.0
        assert proba.max() <= 1.0

    def test_no_nan(
        self,
        detector: BadChannelDetector,
        sample_data: tuple,
    ) -> None:
        X_df, _, _ = sample_data
        proba = detector.predict_proba(X_df)
        assert not np.isnan(proba).any()


class TestPredict:

    def test_returns_bool_array(
        self,
        detector: BadChannelDetector,
        sample_data: tuple,
    ) -> None:
        X_df, _, _ = sample_data
        preds = detector.predict(X_df)
        assert preds.dtype == bool, f"Expected bool, got {preds.dtype}"
        assert preds.shape == (len(X_df),)

    def test_detects_some_bad(
        self,
        detector: BadChannelDetector,
        sample_data: tuple,
    ) -> None:
        """With 10 known-bad rows in sample, at least some should be flagged."""
        X_df, _, _ = sample_data
        preds = detector.predict(X_df)
        # Last 10 rows are bad — model should catch at least some
        bad_preds = preds[10:]
        assert bad_preds.sum() >= 1, (
            "Model flagged zero bad channels from 10 known-bad rows — "
            "check threshold or feature alignment."
        )


class TestPredictChannels:

    def test_returns_dict(
        self,
        detector: BadChannelDetector,
        sample_data: tuple,
    ) -> None:
        X_df, _, labels = sample_data
        result = detector.predict_channels(X_df, labels)
        assert isinstance(result, dict)
        assert len(result) == len(X_df)

    def test_values_are_bool(
        self,
        detector: BadChannelDetector,
        sample_data: tuple,
    ) -> None:
        X_df, _, labels = sample_data
        result = detector.predict_channels(X_df, labels)
        assert all(isinstance(v, bool) for v in result.values()), (
            "predict_channels must return bool values, not numpy bools"
        )

    def test_keys_match_labels(
        self,
        detector: BadChannelDetector,
        sample_data: tuple,
    ) -> None:
        X_df, _, labels = sample_data
        result = detector.predict_channels(X_df, labels)
        assert list(result.keys()) == labels

    def test_length_mismatch_raises(
        self,
        detector: BadChannelDetector,
        sample_data: tuple,
    ) -> None:
        X_df, _, _ = sample_data
        with pytest.raises(AssertionError, match="Mismatch"):
            detector.predict_channels(X_df, ["only_one_label"])


class TestLatency:

    def test_inference_under_10ms(
        self,
        detector: BadChannelDetector,
        sample_data: tuple,
    ) -> None:
        """Inference on N_SAMPLE rows must complete in < 10 ms."""
        X_df, _, _ = sample_data
        _, elapsed_ms = detector.predict_timed(X_df)
        print(f"\n  Inference latency ({N_SAMPLE} rows): {elapsed_ms:.2f} ms")
        assert elapsed_ms < 10.0, (
            f"Inference too slow: {elapsed_ms:.1f} ms. "
            "Ensure the model is loaded once at init, not inside predict()."
        )


class TestUnfittedGuard:

    def test_predict_proba_raises_if_unfitted(self) -> None:
        det = BadChannelDetector()   # no model_path
        dummy = pd.DataFrame(np.zeros((5, 3)))
        with pytest.raises(RuntimeError, match="not fitted"):
            det.predict_proba(dummy)

    def test_save_raises_if_unfitted(self, tmp_path) -> None:
        det = BadChannelDetector()
        with pytest.raises(RuntimeError, match="not fitted"):
            det.save(tmp_path / "should_not_exist.json")
