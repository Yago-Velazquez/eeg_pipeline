"""
bad_channel_rejection/model.py

BadChannelDetector — production-facing API wrapping a trained BCR model.

Unlike models.py (which wraps raw estimators for training), this class
adds channel-level predict APIs, threshold management, and metadata.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .logging_config import setup_logging
from .models import MODEL_EXT, create_model

logger = setup_logging(__name__)


class BadChannelDetector:
    """Production wrapper for a trained BCR model.

    Parameters
    ----------
    threshold : float
        Decision threshold on predicted probability.
    model_name : str
        Backend: 'xgboost', 'lightgbm', or 'catboost'.
    model_path : str or Path, optional
        If provided, load the model immediately on init.
    """

    DEFAULT_THRESHOLD = 0.5

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        model_name: str = "xgboost",
        model_path: str | Path | None = None,
    ):
        self.threshold = threshold
        self.model_name = model_name
        self._model = None
        self._feature_names: list[str] | None = None
        self._meta: dict[str, Any] = {}

        if model_path is not None:
            self.load(model_path)

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        scale_pos_weight: float = 24.8,
    ) -> "BadChannelDetector":
        self._feature_names = list(X.columns)
        self._model = create_model(self.model_name, scale_pos_weight)
        self._model.fit(X.values, y, sample_weight=sample_weight)
        self._meta = {
            "n_train": int(len(y)),
            "n_bad": int(y.sum()),
            "threshold": self.threshold,
            "feature_names": self._feature_names,
            "model_name": self.model_name,
        }
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self._model.predict_proba(X.values)[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(X) >= self.threshold

    def predict_channels(
        self, X: pd.DataFrame, channel_labels: list[str]
    ) -> dict[str, bool]:
        assert len(channel_labels) == len(X), (
            f"Mismatch: {len(channel_labels)} labels vs {len(X)} rows"
        )
        preds = self.predict(X)
        return {ch: bool(bad) for ch, bad in zip(channel_labels, preds)}

    def save(self, path: str | Path) -> None:
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save(path)
        meta_path = self._meta_path(path)
        meta_path.write_text(json.dumps(self._meta, indent=2))
        logger.info(f"Model saved -> {path}")
        logger.info(f"Meta saved -> {meta_path}")

    def load(self, path: str | Path) -> "BadChannelDetector":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        meta_path = self._meta_path(path)
        if meta_path.exists():
            self._meta = json.loads(meta_path.read_text())
            self._feature_names = self._meta.get("feature_names")
            self.threshold = self._meta.get("threshold", self.threshold)
            self.model_name = self._meta.get("model_name", self.model_name)

        self._model = create_model(self.model_name, scale_pos_weight=1.0)
        self._model.load(path)
        return self

    def predict_timed(
        self, X: pd.DataFrame
    ) -> tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        preds = self.predict(X)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return preds, elapsed_ms

    @staticmethod
    def _meta_path(model_path: Path) -> Path:
        return model_path.with_name(model_path.stem + "_meta.json")

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                "BadChannelDetector not fitted. Call fit() or load()."
            )

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "unfitted"
        n_feat = len(self._feature_names) if self._feature_names else 0
        return (
            f"BadChannelDetector(model={self.model_name!r}, "
            f"threshold={self.threshold}, status={status}, "
            f"n_features={n_feat})"
        )
