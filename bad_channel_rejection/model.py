"""
bad_channel_rejection/model.py

BadChannelDetector — clean API wrapping the trained XGBoost BCR model.

The sidecar *_meta.json is written by save() and read by load().
If it is missing (e.g. model was saved directly via train.py before
this class existed), run the one-time migration in scripts/migrate_bcr_meta.py.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb


class BadChannelDetector:
    """
    Wraps a trained XGBoost model to predict bad channels.

    Parameters
    ----------
    threshold : float
        Decision threshold on predicted probability (default: 0.604,
        tuned to argmax F1 on OOF folds — Day 11).
    model_path : str | Path | None
        If provided, load the model immediately on init.
    """

    DEFAULT_THRESHOLD = 0.604   # from Day 11 evaluate.py

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        model_path: Optional[str | Path] = None,
    ) -> None:
        self.threshold      = threshold
        self._model: Optional[xgb.XGBClassifier] = None
        self._feature_names: Optional[List[str]]  = None
        self._meta: Dict                           = {}

        if model_path is not None:
            self.load(model_path)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        groups: np.ndarray,
    ) -> "BadChannelDetector":
        """Train XGBoost on the full dataset.

        Call ONCE after cross-validation confirms hyperparameters.
        groups is accepted for API symmetry but not used in the final fit.
        """
        self._feature_names = list(X.columns)

        self._model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=24.8,   # 18168 good / 732 bad
            eval_metric="aucpr",
            random_state=42,
            tree_method="hist",      # CPU-safe; works on M2 + Colab
            device="cpu",
            n_jobs=-1,
        )
        self._model.fit(X, y)

        self._meta = {
            "n_train":       int(len(y)),
            "n_bad":         int(y.sum()),
            "threshold":     self.threshold,
            "feature_names": self._feature_names,
        }
        return self

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(bad) for each row. Shape: (n,)."""
        self._check_fitted()
        return self._model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return bool array. True = predicted bad. Shape: (n,)."""
        return self.predict_proba(X) >= self.threshold

    def predict_channels(
        self,
        X: pd.DataFrame,
        channel_labels: List[str],
    ) -> Dict[str, bool]:
        """Return {channel_name: is_bad} dict for a single session."""
        assert len(channel_labels) == len(X), (
            f"Mismatch: {len(channel_labels)} labels vs {len(X)} rows"
        )
        preds = self.predict(X)
        return {ch: bool(bad) for ch, bad in zip(channel_labels, preds)}

    # ── Persist ───────────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save model (XGBoost native JSON) + metadata sidecar."""
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self._model.save_model(str(path))

        meta_path = self._meta_path(path)
        with open(meta_path, "w") as f:
            json.dump(self._meta, f, indent=2)

        print(f"✅  Model saved → {path}")
        print(f"✅  Meta  saved → {meta_path}")

    def load(self, path: str | Path) -> "BadChannelDetector":
        """Load model + metadata sidecar from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found: {path}\n"
                "Run train.py first, or check the path."
            )

        self._model = xgb.XGBClassifier()
        self._model.load_model(str(path))

        meta_path = self._meta_path(path)
        if meta_path.exists():
            with open(meta_path) as f:
                self._meta = json.load(f)
            self._feature_names = self._meta.get("feature_names")
            self.threshold      = self._meta.get("threshold", self.threshold)
        else:
            # Sidecar missing — model was saved directly via train.py.
            # Run scripts/migrate_bcr_meta.py to generate it.
            import warnings
            warnings.warn(
                f"Sidecar not found: {meta_path}\n"
                "feature_names will be None. Run scripts/migrate_bcr_meta.py "
                "to generate the sidecar.",
                UserWarning,
                stacklevel=2,
            )
        return self

    # ── Latency helper ────────────────────────────────────────────────────────

    def predict_timed(self, X: pd.DataFrame) -> tuple[np.ndarray, float]:
        """Return (predictions, elapsed_ms). Used in integration tests."""
        t0 = time.perf_counter()
        preds = self.predict(X)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return preds, elapsed_ms

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _meta_path(model_path: Path) -> Path:
        """Sidecar lives next to the model: bcr_model_thresh2_meta.json."""
        return model_path.with_name(model_path.stem + "_meta.json")

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                "BadChannelDetector is not fitted. Call fit() or load()."
            )

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "unfitted"
        n_feat = len(self._feature_names) if self._feature_names else 0
        return (
            f"BadChannelDetector(threshold={self.threshold}, "
            f"status={status}, n_features={n_feat})"
        )
