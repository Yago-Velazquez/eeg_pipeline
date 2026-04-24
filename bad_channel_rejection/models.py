"""
bad_channel_rejection/models.py

Model factory for the BCR pipeline. Provides a uniform interface over
XGBoost, LightGBM, and CatBoost so train.py and evaluate.py can swap models
without caring about backend-specific APIs.

All models expose:
    - fit(X, y, sample_weight=None, eval_set=None)
    - predict_proba(X) -> (n, 2)
    - best_iteration : int
    - save(path), load(path)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from .logging_config import setup_logging

logger = setup_logging(__name__)

SUPPORTED_MODELS = ("xgboost", "lightgbm", "catboost")
MODEL_EXT = {"xgboost": "json", "lightgbm": "pkl", "catboost": "cbm"}

DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "xgboost": {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "aucpr",
        "early_stopping_rounds": 30,
        "tree_method": "hist",
        "device": "cpu",
        "random_state": 42,
        "verbosity": 0,
    },
    "lightgbm": {
        "n_estimators": 500,
        "max_depth": -1,
        "num_leaves": 63,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "metric": "average_precision",
        "random_state": 42,
        "verbosity": -1,
        "force_col_wise": True,
    },
    "catboost": {
        "iterations": 500,
        "depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "rsm": 0.8,
        "eval_metric": "PRAUC",
        "early_stopping_rounds": 30,
        "random_state": 42,
        "verbose": False,
        "bootstrap_type": "Bernoulli",
    },
}


class BaseBCRModel(ABC):
    """Uniform model interface for BCR training."""

    def __init__(self, scale_pos_weight: float, **overrides):
        self.scale_pos_weight = float(scale_pos_weight)
        self._overrides = overrides
        self._model = None
        self._best_iteration: int | None = None

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> "BaseBCRModel":
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        ...

    @property
    def best_iteration(self) -> int:
        if self._best_iteration is None:
            return DEFAULT_PARAMS[self.name].get("n_estimators", 500)
        return self._best_iteration

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def save(self, path: str | Path) -> None:
        ...

    @abstractmethod
    def load(self, path: str | Path) -> "BaseBCRModel":
        ...


class XGBoostBCR(BaseBCRModel):
    name = "xgboost"

    def fit(self, X, y, sample_weight=None, eval_set=None):
        from xgboost import XGBClassifier

        params = {**DEFAULT_PARAMS["xgboost"], **self._overrides}
        if eval_set is None:
            params.pop("early_stopping_rounds", None)
            params.pop("eval_metric", None)

        self._model = XGBClassifier(
            scale_pos_weight=self.scale_pos_weight, **params
        )
        fit_kwargs = {"verbose": False}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        if eval_set is not None:
            fit_kwargs["eval_set"] = [eval_set]
        self._model.fit(X, y, **fit_kwargs)

        self._best_iteration = int(
            getattr(self._model, "best_iteration", 0)
            or params["n_estimators"]
        )
        return self

    def predict_proba(self, X):
        return self._model.predict_proba(X)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(path))

    def load(self, path):
        from xgboost import XGBClassifier
        self._model = XGBClassifier()
        self._model.load_model(str(path))
        return self


class LightGBMBCR(BaseBCRModel):
    name = "lightgbm"

    def fit(self, X, y, sample_weight=None, eval_set=None):
        from lightgbm import LGBMClassifier, early_stopping, log_evaluation

        params = {**DEFAULT_PARAMS["lightgbm"], **self._overrides}

        self._model = LGBMClassifier(
            scale_pos_weight=self.scale_pos_weight, **params
        )
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        if eval_set is not None:
            fit_kwargs["eval_set"] = [eval_set]
            fit_kwargs["callbacks"] = [
                early_stopping(30, verbose=False),
                log_evaluation(0),
            ]
        self._model.fit(X, y, **fit_kwargs)

        self._best_iteration = int(
            getattr(self._model, "best_iteration_", 0)
            or params["n_estimators"]
        )
        return self

    def predict_proba(self, X):
        return self._model.predict_proba(X)

    def save(self, path):
        import joblib
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, str(path))

    def load(self, path):
        import joblib
        self._model = joblib.load(str(path))
        return self


class CatBoostBCR(BaseBCRModel):
    name = "catboost"

    def fit(self, X, y, sample_weight=None, eval_set=None):
        from catboost import CatBoostClassifier

        params = {**DEFAULT_PARAMS["catboost"], **self._overrides}
        if eval_set is None:
            params.pop("early_stopping_rounds", None)

        self._model = CatBoostClassifier(
            scale_pos_weight=self.scale_pos_weight, **params
        )
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        if eval_set is not None:
            fit_kwargs["eval_set"] = eval_set
        self._model.fit(X, y, **fit_kwargs)

        self._best_iteration = int(
            getattr(self._model, "best_iteration_", 0)
            or params["iterations"]
        )
        return self

    def predict_proba(self, X):
        return self._model.predict_proba(X)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(path))

    def load(self, path):
        from catboost import CatBoostClassifier
        self._model = CatBoostClassifier()
        self._model.load_model(str(path))
        return self


def create_model(
    name: str, scale_pos_weight: float, **overrides
) -> BaseBCRModel:
    """Factory: create a BaseBCRModel by name."""
    name = name.lower()
    if name == "xgboost":
        return XGBoostBCR(scale_pos_weight, **overrides)
    if name == "lightgbm":
        return LightGBMBCR(scale_pos_weight, **overrides)
    if name == "catboost":
        return CatBoostBCR(scale_pos_weight, **overrides)
    raise ValueError(
        f"Unknown model: {name!r}. Supported: {SUPPORTED_MODELS}"
    )
