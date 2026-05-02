"""
bad_channel_rejection/models.py

Model factory for the BCR pipeline.  Provides a uniform interface over 17
backends across 6 ML families so train.py and the ablation scripts can swap
models without caring about backend-specific APIs.

Family → backend map
--------------------
    linear      : ElasticNet, Ridge, Lasso
    bagging     : RandomForest, ExtraTrees
    boosting    : XGBoost, LightGBM, CatBoost, HistGradientBoosting
    transformer : FT-Transformer, SAINT, TabTransformer, TabPFN
    mlp         : ResNet-Tabular, NODE, TabNet
    foundation  : TabPFN, TabICL, Mitra

All models expose:
    - fit(X, y, sample_weight=None, eval_set=None)
    - predict_proba(X) -> (n, 2)
    - best_iteration : int
    - save(path), load(path)

External package dependencies (only loaded lazily when the model is requested)
- xgboost, lightgbm, catboost                    — boosting
- torch                                          — all transformer / mlp
- pytorch-tabnet                                 — TabNet
- tabpfn                                         — TabPFN
- tabicl                                         — TabICL    (Prior-Labs)
- transformers + huggingface_hub                 — Mitra (HF Hub)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from .logging_config import setup_logging

logger = setup_logging(__name__)

SUPPORTED_MODELS = (
    # linear
    "elasticnet", "ridge", "lasso",
    # bagging
    "random_forest", "extra_trees",
    # boosting
    "xgboost", "lightgbm", "catboost", "histgradientboosting",
    # transformer
    "ft_transformer", "saint", "tab_transformer", "tabpfn",
    # mlp
    "resnet_tabular", "node", "tabnet",
    # foundation
    "tabicl", "mitra",
)
MODEL_EXT = {
    # sklearn
    "elasticnet": "pkl", "ridge": "pkl", "lasso": "pkl",
    "random_forest": "pkl", "extra_trees": "pkl",
    "histgradientboosting": "pkl",
    # boosting
    "xgboost": "json", "lightgbm": "pkl", "catboost": "cbm",
    # torch
    "ft_transformer": "pt", "saint": "pt", "tab_transformer": "pt",
    "resnet_tabular": "pt", "node": "pt",
    # external libs
    "tabnet": "zip", "tabpfn": "pkl",
    "tabicl": "pkl", "mitra": "pkl",
}

DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    # ── linear ────────────────────────────────────────────────────────────────
    "elasticnet": {
        "penalty": "elasticnet", "solver": "saga",
        "l1_ratio": 0.5, "C": 1.0,
        "max_iter": 5000, "tol": 1e-3, "random_state": 42,
    },
    "ridge": {
        "penalty": "l2", "solver": "lbfgs",
        "C": 1.0, "max_iter": 5000, "tol": 1e-3, "random_state": 42,
    },
    "lasso": {
        "penalty": "l1", "solver": "saga",
        "C": 1.0, "max_iter": 5000, "tol": 1e-3, "random_state": 42,
    },
    # ── bagging ───────────────────────────────────────────────────────────────
    "random_forest": {
        "n_estimators": 500, "max_depth": None,
        "max_features": "sqrt", "min_samples_leaf": 2,
        "n_jobs": -1, "random_state": 42,
    },
    "extra_trees": {
        "n_estimators": 500, "max_depth": None,
        "max_features": "sqrt", "min_samples_leaf": 2,
        "n_jobs": -1, "random_state": 42,
    },
    # ── boosting ──────────────────────────────────────────────────────────────
    "xgboost": {
        "n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "eval_metric": "aucpr", "early_stopping_rounds": 30,
        "tree_method": "hist", "device": "cpu",
        "random_state": 42, "verbosity": 0,
    },
    "lightgbm": {
        "n_estimators": 500, "max_depth": -1, "num_leaves": 63,
        "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
        "metric": "average_precision", "random_state": 42,
        "verbosity": -1, "force_col_wise": True,
    },
    "catboost": {
        "iterations": 500, "depth": 6, "learning_rate": 0.05,
        "eval_metric": "PRAUC", "early_stopping_rounds": 30,
        "random_state": 42, "verbose": False,
        # bootstrap / subsampling injected at fit-time (device-dependent).
    },
    "histgradientboosting": {
        "max_iter": 500, "max_leaf_nodes": 31, "learning_rate": 0.05,
        "l2_regularization": 0.0, "random_state": 42,
        "early_stopping": True, "n_iter_no_change": 30,
        "validation_fraction": 0.1,
    },
    # ── transformer / mlp / NODE (PyTorch shared base) ────────────────────────
    #
    # Defaults tuned for imbalanced tabular data (~3.9% positive rate):
    #   - n_epochs=150 (up from 100): give models enough wall-clock budget.
    #   - patience=25  (up from 10):  validation BCE on the small minority
    #                                 class is noisy — short patience triggers
    #                                 spurious early stops in the first ~10
    #                                 epochs. The shared base also enforces a
    #                                 warm_up_epochs phase during which early
    #                                 stopping cannot fire at all.
    #   - learning_rate per-model:    transformers stay at 1e-4 (stable);
    #                                 ResNet drops to 5e-4 (was 1e-3 — too
    #                                 aggressive, killed training in 3 epochs);
    #                                 NODE stays at 1e-3 (its soft trees need
    #                                 a higher LR to push thresholds).
    "ft_transformer": {
        "d_token": 64, "n_layers": 3, "n_heads": 8,
        "ffn_dropout": 0.1, "attn_dropout": 0.1,
        "learning_rate": 1e-4, "weight_decay": 1e-5,
        "batch_size": 256, "n_epochs": 150, "patience": 25,
        "warm_up_epochs": 10, "device": "cpu",
    },
    "saint": {
        "d_token": 32, "n_layers": 3, "n_heads": 4,
        "ffn_dropout": 0.1, "attn_dropout": 0.1,
        "learning_rate": 1e-4, "weight_decay": 1e-5,
        "batch_size": 256, "n_epochs": 150, "patience": 25,
        "warm_up_epochs": 10, "device": "cpu",
    },
    "tab_transformer": {
        "d_token": 32, "n_layers": 3, "n_heads": 4,
        "ffn_dropout": 0.1, "attn_dropout": 0.1,
        "learning_rate": 1e-4, "weight_decay": 1e-5,
        "batch_size": 256, "n_epochs": 150, "patience": 25,
        "warm_up_epochs": 10, "device": "cpu",
    },
    "resnet_tabular": {
        "d_main": 256, "d_hidden": 512, "n_blocks": 3,
        "hidden_dropout": 0.2, "residual_dropout": 0.0,
        "learning_rate": 5e-4, "weight_decay": 1e-5,
        "batch_size": 256, "n_epochs": 150, "patience": 25,
        "warm_up_epochs": 10, "device": "cpu",
    },
    "node": {
        "n_trees": 64, "depth": 4,
        "learning_rate": 1e-3, "weight_decay": 1e-5,
        "batch_size": 256, "n_epochs": 150, "patience": 25,
        "warm_up_epochs": 10, "device": "cpu",
    },
    # ── external libs ─────────────────────────────────────────────────────────
    "tabnet": {
        "n_d": 16, "n_a": 16, "n_steps": 3, "gamma": 1.5,
        "learning_rate": 2e-2, "weight_decay": 1e-5,
        "batch_size": 1024, "max_epochs": 100, "patience": 10, "device": "cpu",
    },
    "tabpfn": {"device": "cpu", "n_estimators": 4, "ignore_pretraining_limits": True},
    "tabicl": {"device": "cpu"},
    "mitra": {"device": "cpu"},
}


# ── Base ───────────────────────────────────────────────────────────────────────


class BaseBCRModel(ABC):
    """Uniform model interface for BCR training."""

    def __init__(self, scale_pos_weight: float, **overrides):
        self.scale_pos_weight = float(scale_pos_weight)
        self._overrides = overrides
        self._model = None
        self._best_iteration: int | None = None

    @abstractmethod
    def fit(self, X, y, sample_weight=None, eval_set=None) -> "BaseBCRModel": ...

    @abstractmethod
    def predict_proba(self, X) -> np.ndarray: ...

    @property
    def best_iteration(self) -> int:
        if self._best_iteration is None:
            return DEFAULT_PARAMS[self.name].get(
                "n_estimators",
                DEFAULT_PARAMS[self.name].get("max_iter", 500),
            )
        return self._best_iteration

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def save(self, path) -> None: ...

    @abstractmethod
    def load(self, path) -> "BaseBCRModel": ...


# ══════════════════════════════════════════════════════════════════════════════
# 1. Sklearn linear (ElasticNet / Ridge / Lasso)
# ══════════════════════════════════════════════════════════════════════════════


class _SklearnLinearBase(BaseBCRModel):
    """Shared logic: standardise → LogisticRegression with the right penalty."""

    def fit(self, X, y, sample_weight=None, eval_set=None):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        params = {**DEFAULT_PARAMS[self.name], **self._overrides}

        self._scaler = StandardScaler()
        X_sc = self._scaler.fit_transform(X)

        self._model = LogisticRegression(
            class_weight={0: 1.0, 1: self.scale_pos_weight}, **params
        )
        fit_kwargs: dict[str, Any] = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        self._model.fit(X_sc, y, **fit_kwargs)
        self._best_iteration = 1
        return self

    def predict_proba(self, X):
        return self._model.predict_proba(self._scaler.transform(X))

    def save(self, path):
        import joblib
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump((self._scaler, self._model), str(path))

    def load(self, path):
        import joblib
        self._scaler, self._model = joblib.load(str(path))
        return self


class ElasticNetBCR(_SklearnLinearBase):
    """LogisticRegression with ElasticNet (L1+L2) penalty."""
    name = "elasticnet"


class RidgeBCR(_SklearnLinearBase):
    """LogisticRegression with pure L2 penalty."""
    name = "ridge"


class LassoBCR(_SklearnLinearBase):
    """LogisticRegression with pure L1 penalty."""
    name = "lasso"


# ══════════════════════════════════════════════════════════════════════════════
# 2. Sklearn tree ensembles (RandomForest / ExtraTrees)
# ══════════════════════════════════════════════════════════════════════════════


class _SklearnTreeEnsembleBase(BaseBCRModel):
    """Shared logic for forest ensembles. No early stopping."""

    @property
    def _classifier_class(self):
        raise NotImplementedError

    def fit(self, X, y, sample_weight=None, eval_set=None):
        params = {**DEFAULT_PARAMS[self.name], **self._overrides}
        self._model = self._classifier_class(
            class_weight={0: 1.0, 1: self.scale_pos_weight}, **params
        )
        fit_kwargs: dict[str, Any] = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        self._model.fit(X, y, **fit_kwargs)
        self._best_iteration = params["n_estimators"]
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


class RandomForestBCR(_SklearnTreeEnsembleBase):
    name = "random_forest"

    @property
    def _classifier_class(self):
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier


class ExtraTreesBCR(_SklearnTreeEnsembleBase):
    name = "extra_trees"

    @property
    def _classifier_class(self):
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier


# ══════════════════════════════════════════════════════════════════════════════
# 3. HistGradientBoosting (sklearn)
# ══════════════════════════════════════════════════════════════════════════════


class HistGradientBoostingBCR(BaseBCRModel):
    """sklearn's native histogram-based gradient boosting.

    Class imbalance is handled via per-sample weights (multiplying positive
    rows by scale_pos_weight). HistGradientBoosting does not accept
    class_weight directly across all sklearn versions, so this is the safe path.
    """

    name = "histgradientboosting"

    def fit(self, X, y, sample_weight=None, eval_set=None):
        from sklearn.ensemble import HistGradientBoostingClassifier

        params = {**DEFAULT_PARAMS["histgradientboosting"], **self._overrides}
        self._model = HistGradientBoostingClassifier(**params)

        # Inject class imbalance via sample weights.
        eff_sw = (
            np.asarray(sample_weight, dtype=np.float64).copy()
            if sample_weight is not None
            else np.ones(len(y), dtype=np.float64)
        )
        eff_sw[y == 1] *= self.scale_pos_weight

        self._model.fit(X, y, sample_weight=eff_sw)
        self._best_iteration = int(getattr(self._model, "n_iter_", params["max_iter"]))
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


# ══════════════════════════════════════════════════════════════════════════════
# 4. XGBoost / LightGBM / CatBoost
# ══════════════════════════════════════════════════════════════════════════════


class XGBoostBCR(BaseBCRModel):
    name = "xgboost"

    def fit(self, X, y, sample_weight=None, eval_set=None):
        from xgboost import XGBClassifier

        params = {**DEFAULT_PARAMS["xgboost"], **self._overrides}
        if eval_set is None:
            params.pop("early_stopping_rounds", None)
            params.pop("eval_metric", None)

        self._model = XGBClassifier(scale_pos_weight=self.scale_pos_weight, **params)
        fit_kwargs = {"verbose": False}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        if eval_set is not None:
            fit_kwargs["eval_set"] = [eval_set]
        self._model.fit(X, y, **fit_kwargs)

        self._best_iteration = int(
            getattr(self._model, "best_iteration", 0) or params["n_estimators"]
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
        self._model = LGBMClassifier(scale_pos_weight=self.scale_pos_weight, **params)
        fit_kwargs: dict[str, Any] = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        if eval_set is not None:
            fit_kwargs["eval_set"] = [eval_set]
            fit_kwargs["callbacks"] = [
                early_stopping(30, verbose=False), log_evaluation(0),
            ]
        self._model.fit(X, y, **fit_kwargs)
        self._best_iteration = int(
            getattr(self._model, "best_iteration_", 0) or params["n_estimators"]
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

        if params.get("task_type") == "GPU":
            # GPU restrictions:
            #   - rsm only for pairwise modes
            #   - bootstrap_type must be Bernoulli / Poisson
            #   - bagging_temperature is CPU-only
            #   - PRAUC has no GPU kernel: CatBoost falls back to CPU eval
            #     every 5 iters, which discretises the early-stopping signal
            #     and triggered premature stops (fold 1 bailing at iter 28
            #     while other folds ran 117–313). Use AUC instead — GPU-native
            #     and a near-monotonic surrogate for PRAUC on imbalanced data.
            params.pop("rsm", None)
            params.pop("bagging_temperature", None)
            params.setdefault("bootstrap_type", "Bernoulli")
            params.setdefault("subsample", 0.8)
            if params.get("eval_metric") == "PRAUC":
                params["eval_metric"] = "AUC"
        else:
            params.pop("subsample", None)
            params.setdefault("bootstrap_type", "Bayesian")
            params.setdefault("bagging_temperature", 1.0)
            params.setdefault("rsm", 0.8)

        if eval_set is None:
            params.pop("early_stopping_rounds", None)

        self._model = CatBoostClassifier(scale_pos_weight=self.scale_pos_weight, **params)
        fit_kwargs: dict[str, Any] = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        if eval_set is not None:
            fit_kwargs["eval_set"] = eval_set
        self._model.fit(X, y, **fit_kwargs)
        self._best_iteration = int(
            getattr(self._model, "best_iteration_", 0) or params["iterations"]
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


# ══════════════════════════════════════════════════════════════════════════════
# 5. PyTorch tabular base + network builders
# ══════════════════════════════════════════════════════════════════════════════


class _TorchTabularBase(BaseBCRModel):
    """Shared train / eval / save / load loop for PyTorch tabular models.

    Subclasses override `_make_net_config(n_features, params)` and
    `_build_network(net_config)` to specify their architecture.
    """

    # ---- subclass hooks ------------------------------------------------------

    def _make_net_config(self, n_features: int, params: dict) -> dict: ...
    def _build_network(self, net_config: dict): ...

    # ---- shared API ----------------------------------------------------------

    def fit(self, X, y, sample_weight=None, eval_set=None):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        params = {**DEFAULT_PARAMS[self.name], **self._overrides}
        device_str: str = params.pop("device", "cpu")
        self._device = torch.device(device_str)

        n_features = X.shape[1]
        self._net_config = self._make_net_config(n_features, params)
        self._net = self._build_network(self._net_config).to(self._device)

        pos_weight = torch.tensor([self.scale_pos_weight], device=self._device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
        optimizer = torch.optim.Adam(
            self._net.parameters(),
            lr=params["learning_rate"],
            weight_decay=params["weight_decay"],
        )

        tensors = [
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        ]
        if sample_weight is not None:
            tensors.append(torch.tensor(sample_weight, dtype=torch.float32))
        loader = DataLoader(
            TensorDataset(*tensors),
            batch_size=params["batch_size"], shuffle=True, drop_last=False,
        )

        # Warm-up phase — early stopping cannot fire while epoch < warm_up.
        # Validation loss on imbalanced minority classes is noisy in the first
        # few epochs; without this, models like ResNet were stopping at
        # epoch 3 of 150 before learning anything.
        warm_up_epochs = int(params.get("warm_up_epochs", 0))
        patience = int(params["patience"])

        best_val = float("inf"); best_state = None; best_epoch = params["n_epochs"]
        no_improve = 0

        for epoch in range(params["n_epochs"]):
            self._net.train()
            for batch in loader:
                xb = batch[0].to(self._device); yb = batch[1].to(self._device)
                wb = batch[2].to(self._device) if sample_weight is not None else None
                logits = self._net(xb).squeeze(-1)
                per = loss_fn(logits, yb)
                loss = (per * wb).mean() if wb is not None else per.mean()
                optimizer.zero_grad(); loss.backward(); optimizer.step()

            if eval_set is not None:
                X_va, y_va = eval_set
                self._net.eval()
                with torch.no_grad():
                    xv = torch.tensor(X_va, dtype=torch.float32).to(self._device)
                    yv = torch.tensor(y_va, dtype=torch.float32).to(self._device)
                    val_loss = loss_fn(self._net(xv).squeeze(-1), yv).mean().item()
                if val_loss < best_val - 1e-5:
                    best_val = val_loss; best_epoch = epoch + 1; no_improve = 0
                    best_state = {k: v.cpu().clone() for k, v in self._net.state_dict().items()}
                else:
                    # Block early-stop signal during warm-up — keep counter at 0.
                    if epoch + 1 < warm_up_epochs:
                        no_improve = 0
                    else:
                        no_improve += 1
                        if no_improve >= patience:
                            logger.debug(
                                f"{self.name} early stop @ epoch {epoch + 1} "
                                f"(warm_up={warm_up_epochs}, patience={patience})"
                            )
                            break

        if best_state is not None:
            self._net.load_state_dict({k: v.to(self._device) for k, v in best_state.items()})
        self._best_iteration = best_epoch
        return self

    def predict_proba(self, X):
        import torch
        self._net.eval()
        with torch.no_grad():
            xt = torch.tensor(X, dtype=torch.float32).to(self._device)
            probs = torch.sigmoid(self._net(xt).squeeze(-1)).cpu().numpy()
        return np.column_stack([1.0 - probs, probs])

    def save(self, path):
        import torch
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": self._net_config, "state_dict": self._net.state_dict()}, str(path))

    def load(self, path):
        import torch
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        self._net_config = ckpt["config"]
        self._net = self._build_network(self._net_config)
        self._net.load_state_dict(ckpt["state_dict"])
        self._device = torch.device("cpu")
        return self


# ── Network builders (called lazily inside subclasses) ─────────────────────────


def _build_ft_transformer(n_features, d_token, n_layers, n_heads, ffn_dropout, attn_dropout):
    """FT-Transformer (Gorishniy 2021): per-feature tokenizer + CLS + transformer."""
    import torch
    import torch.nn as nn

    class _FTNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.W = nn.Parameter(torch.empty(n_features, d_token))
            self.b = nn.Parameter(torch.zeros(n_features, d_token))
            nn.init.normal_(self.W, std=0.01)
            self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
            nn.init.normal_(self.cls, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=d_token, nhead=n_heads, dim_feedforward=d_token * 4,
                dropout=ffn_dropout, activation="gelu",
                batch_first=True, norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.head = nn.Sequential(nn.LayerNorm(d_token), nn.Linear(d_token, 1))

        def forward(self, x):
            tokens = x.unsqueeze(-1) * self.W + self.b
            cls = self.cls.expand(x.size(0), -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
            return self.head(self.transformer(tokens)[:, 0])

    return _FTNet()


def _build_saint(n_features, d_token, n_layers, n_heads, ffn_dropout, attn_dropout):
    """SAINT-lite (Somepalli 2021): FT-style tokenizer + self-attention over
    features, with intersample attention block stacked alternately.

    For tractability on a small dataset we implement intersample attention as
    a single self-attention pass over the batch after each transformer block.
    """
    import torch
    import torch.nn as nn

    class _SaintBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.feat_attn = nn.TransformerEncoderLayer(
                d_model=d_token, nhead=n_heads, dim_feedforward=d_token * 4,
                dropout=ffn_dropout, activation="gelu",
                batch_first=True, norm_first=True,
            )
            self.row_attn = nn.MultiheadAttention(
                embed_dim=d_token * (n_features + 1), num_heads=n_heads,
                dropout=attn_dropout, batch_first=True,
            )
            self.row_norm = nn.LayerNorm(d_token * (n_features + 1))

        def forward(self, x):
            # x: (B, F+1, d_token)
            x = self.feat_attn(x)
            B, T, D = x.shape
            flat = x.reshape(B, 1, T * D)              # treat each row as one token
            attn_out, _ = self.row_attn(flat, flat, flat)
            x = self.row_norm(flat + attn_out).reshape(B, T, D)
            return x

    class _SaintNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.W = nn.Parameter(torch.empty(n_features, d_token))
            self.b = nn.Parameter(torch.zeros(n_features, d_token))
            nn.init.normal_(self.W, std=0.01)
            self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
            nn.init.normal_(self.cls, std=0.02)
            self.blocks = nn.ModuleList([_SaintBlock() for _ in range(n_layers)])
            self.head = nn.Sequential(nn.LayerNorm(d_token), nn.Linear(d_token, 1))

        def forward(self, x):
            tokens = x.unsqueeze(-1) * self.W + self.b
            cls = self.cls.expand(x.size(0), -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
            for blk in self.blocks:
                tokens = blk(tokens)
            return self.head(tokens[:, 0])

    return _SaintNet()


def _build_tab_transformer(n_features, d_token, n_layers, n_heads, ffn_dropout, attn_dropout):
    """TabTransformer (Huang 2020): per-feature tokenizer + transformer, no CLS,
    mean-pool over feature embeddings before head.

    Original paper targets categorical features with embedding tables; for
    all-numeric inputs we use a linear tokenizer (same as FT-Transformer) but
    keep the no-CLS, mean-pool readout from the original TabTransformer head.
    """
    import torch
    import torch.nn as nn

    class _TabTNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.W = nn.Parameter(torch.empty(n_features, d_token))
            self.b = nn.Parameter(torch.zeros(n_features, d_token))
            nn.init.normal_(self.W, std=0.01)
            layer = nn.TransformerEncoderLayer(
                d_model=d_token, nhead=n_heads, dim_feedforward=d_token * 4,
                dropout=ffn_dropout, activation="gelu",
                batch_first=True, norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.head = nn.Sequential(
                nn.LayerNorm(d_token),
                nn.Linear(d_token, d_token), nn.GELU(),
                nn.Linear(d_token, 1),
            )

        def forward(self, x):
            tokens = x.unsqueeze(-1) * self.W + self.b
            tokens = self.transformer(tokens)
            pooled = tokens.mean(dim=1)
            return self.head(pooled)

    return _TabTNet()


def _build_resnet_tabular(n_features, d_main, d_hidden, n_blocks, hidden_dropout, residual_dropout):
    """ResNet-Tabular (Gorishniy 2021): residual MLP with bottleneck blocks.

    Each block: LayerNorm → Linear(d_main → d_hidden) → ReLU → Dropout →
    Linear(d_hidden → d_main) → Dropout → add residual.
    """
    import torch
    import torch.nn as nn

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(d_main)
            self.linear1 = nn.Linear(d_main, d_hidden)
            self.linear2 = nn.Linear(d_hidden, d_main)
            self.hidden_drop = nn.Dropout(hidden_dropout)
            self.res_drop = nn.Dropout(residual_dropout)

        def forward(self, x):
            z = self.norm(x)
            z = torch.relu(self.linear1(z))
            z = self.hidden_drop(z)
            z = self.linear2(z)
            z = self.res_drop(z)
            return x + z

    class _ResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = nn.Linear(n_features, d_main)
            self.blocks = nn.ModuleList([_Block() for _ in range(n_blocks)])
            self.head = nn.Sequential(nn.LayerNorm(d_main), nn.ReLU(), nn.Linear(d_main, 1))

        def forward(self, x):
            h = self.input(x)
            for blk in self.blocks:
                h = blk(h)
            return self.head(h)

    return _ResNet()


def _build_node(n_features, n_trees, depth):
    """NODE-lite (Popov 2019): ensemble of differentiable oblivious decision trees.

    Each tree softly selects a feature per depth-level (softmax over features),
    applies a learned threshold (sigmoid → soft binary), then the per-leaf
    weights are combined by the implied path-probability product. Lighter-weight
    than the entmax15 NODE in the paper, but captures the same architectural
    spirit (tree ensemble of soft oblivious decisions).
    """
    import torch
    import torch.nn as nn

    class _Tree(nn.Module):
        def __init__(self):
            super().__init__()
            self.feat_logits = nn.Parameter(torch.randn(depth, n_features) * 0.01)
            self.thresholds = nn.Parameter(torch.zeros(depth))
            self.leaves = nn.Parameter(torch.randn(2 ** depth) * 0.1)

        def forward(self, x):
            B = x.size(0)
            feat_attn = torch.softmax(self.feat_logits, dim=-1)   # (D, F)
            chosen = x @ feat_attn.T                               # (B, D)
            decisions = torch.sigmoid(chosen - self.thresholds)    # (B, D)
            # Build (B, 2**depth) leaf-path probabilities via outer-products
            leaf_probs = torch.ones(B, 1, device=x.device)
            for d in range(depth):
                go_right = decisions[:, d:d + 1]
                go_left = 1.0 - go_right
                leaf_probs = torch.cat(
                    [leaf_probs * go_left, leaf_probs * go_right], dim=1
                )
            return (leaf_probs * self.leaves).sum(dim=1, keepdim=True)

    class _NODE(nn.Module):
        def __init__(self):
            super().__init__()
            self.trees = nn.ModuleList([_Tree() for _ in range(n_trees)])
            self.bias = nn.Parameter(torch.zeros(1))

        def forward(self, x):
            outs = torch.stack([t(x) for t in self.trees], dim=1).sum(dim=1)
            return outs + self.bias

    return _NODE()


# ── PyTorch concrete classes ───────────────────────────────────────────────────


class FTTransformerBCR(_TorchTabularBase):
    name = "ft_transformer"

    def _make_net_config(self, n_features, params):
        return {
            "n_features": n_features,
            "d_token":     params["d_token"],
            "n_layers":    params["n_layers"],
            "n_heads":     params["n_heads"],
            "ffn_dropout": params["ffn_dropout"],
            "attn_dropout": params["attn_dropout"],
        }

    def _build_network(self, cfg):
        return _build_ft_transformer(**cfg)


class SAINTBCR(_TorchTabularBase):
    name = "saint"

    def _make_net_config(self, n_features, params):
        return {
            "n_features": n_features,
            "d_token":     params["d_token"],
            "n_layers":    params["n_layers"],
            "n_heads":     params["n_heads"],
            "ffn_dropout": params["ffn_dropout"],
            "attn_dropout": params["attn_dropout"],
        }

    def _build_network(self, cfg):
        return _build_saint(**cfg)


class TabTransformerBCR(_TorchTabularBase):
    name = "tab_transformer"

    def _make_net_config(self, n_features, params):
        return {
            "n_features": n_features,
            "d_token":     params["d_token"],
            "n_layers":    params["n_layers"],
            "n_heads":     params["n_heads"],
            "ffn_dropout": params["ffn_dropout"],
            "attn_dropout": params["attn_dropout"],
        }

    def _build_network(self, cfg):
        return _build_tab_transformer(**cfg)


class ResNetTabularBCR(_TorchTabularBase):
    name = "resnet_tabular"

    def _make_net_config(self, n_features, params):
        return {
            "n_features":       n_features,
            "d_main":           params["d_main"],
            "d_hidden":         params["d_hidden"],
            "n_blocks":         params["n_blocks"],
            "hidden_dropout":   params["hidden_dropout"],
            "residual_dropout": params["residual_dropout"],
        }

    def _build_network(self, cfg):
        return _build_resnet_tabular(**cfg)


class NODEBCR(_TorchTabularBase):
    name = "node"

    def _make_net_config(self, n_features, params):
        return {
            "n_features": n_features,
            "n_trees":    params["n_trees"],
            "depth":      params["depth"],
        }

    def _build_network(self, cfg):
        return _build_node(**cfg)


# ══════════════════════════════════════════════════════════════════════════════
# 6. TabNet (pytorch-tabnet)
# ══════════════════════════════════════════════════════════════════════════════


class TabNetBCR(BaseBCRModel):
    """Wraps pytorch_tabnet.tab_model.TabNetClassifier.

    Requires:  pip install pytorch-tabnet
    """
    name = "tabnet"

    def fit(self, X, y, sample_weight=None, eval_set=None):
        from pytorch_tabnet.tab_model import TabNetClassifier
        import torch

        params = {**DEFAULT_PARAMS["tabnet"], **self._overrides}
        device_str = params.pop("device", "cpu")
        max_epochs = params.pop("max_epochs", 100)
        patience = params.pop("patience", 10)
        batch_size = params.pop("batch_size", 1024)
        lr = params.pop("learning_rate", 2e-2)
        wd = params.pop("weight_decay", 1e-5)

        self._device = device_str
        self._model = TabNetClassifier(
            n_d=params.get("n_d", 16), n_a=params.get("n_a", 16),
            n_steps=params.get("n_steps", 3), gamma=params.get("gamma", 1.5),
            optimizer_fn=torch.optim.Adam,
            optimizer_params={"lr": lr, "weight_decay": wd},
            device_name=device_str, verbose=0,
        )
        # TabNet uses class weights via loss_fn; here we fold scale_pos_weight
        # into per-sample weights.
        eff_sw = (
            np.asarray(sample_weight, dtype=np.float32).copy()
            if sample_weight is not None
            else np.ones(len(y), dtype=np.float32)
        )
        eff_sw[y == 1] *= self.scale_pos_weight

        eval_sets = [(np.asarray(eval_set[0]), np.asarray(eval_set[1]))] if eval_set else None
        self._model.fit(
            X_train=np.asarray(X), y_train=np.asarray(y),
            eval_set=eval_sets, eval_metric=["auc"],
            max_epochs=max_epochs, patience=patience,
            batch_size=batch_size, virtual_batch_size=min(128, batch_size),
            weights=eff_sw, drop_last=False,
        )
        self._best_iteration = int(getattr(self._model, "best_epoch", max_epochs) or max_epochs)
        return self

    def predict_proba(self, X):
        return self._model.predict_proba(np.asarray(X))

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # pytorch-tabnet saves a .zip — strip extension to match its convention
        p = str(path)
        self._model.save_model(p[:-4] if p.endswith(".zip") else p)

    def load(self, path):
        from pytorch_tabnet.tab_model import TabNetClassifier
        self._model = TabNetClassifier()
        p = str(path)
        self._model.load_model(p if p.endswith(".zip") else p + ".zip")
        return self


# ══════════════════════════════════════════════════════════════════════════════
# 7. Foundation models — TabPFN / TabICL / Mitra
# ══════════════════════════════════════════════════════════════════════════════


class TabPFNBCR(BaseBCRModel):
    """Wraps tabpfn.TabPFNClassifier (Prior-Labs/tabpfn_2_5).

    TabPFN is in-context: there is no per-task gradient training. Calling fit()
    just stores the training rows; predict_proba() runs the foundation model
    with those rows as the in-context demonstrations.

    Requires: pip install tabpfn   (pulls weights from HuggingFace on first use)
    """
    name = "tabpfn"

    def fit(self, X, y, sample_weight=None, eval_set=None):
        from tabpfn import TabPFNClassifier

        params = {**DEFAULT_PARAMS["tabpfn"], **self._overrides}
        self._model = TabPFNClassifier(
            device=params.get("device", "cpu"),
            n_estimators=params.get("n_estimators", 4),
            ignore_pretraining_limits=params.get("ignore_pretraining_limits", True),
        )
        # TabPFN does not support sample weights — we fold scale_pos_weight in
        # by oversampling positives (cheap given the small positive count).
        if sample_weight is None:
            self._model.fit(np.asarray(X), np.asarray(y))
        else:
            # oversample using the weights as relative draw probabilities
            rng = np.random.default_rng(42)
            p = np.asarray(sample_weight, dtype=np.float64)
            p = p / p.sum()
            idx = rng.choice(len(y), size=len(y), replace=True, p=p)
            self._model.fit(np.asarray(X)[idx], np.asarray(y)[idx])
        self._best_iteration = 1
        return self

    def predict_proba(self, X):
        return self._model.predict_proba(np.asarray(X))

    def save(self, path):
        import joblib
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, str(path))

    def load(self, path):
        import joblib
        self._model = joblib.load(str(path))
        return self


class TabICLBCR(BaseBCRModel):
    """Wraps tabicl.TabICLClassifier (Prior-Labs).

    TabICL is the large-data sibling of TabPFN — same in-context paradigm but
    handles much larger demonstration sets.

    Requires: pip install tabicl   (pulls weights from HuggingFace on first use)
    """
    name = "tabicl"

    def fit(self, X, y, sample_weight=None, eval_set=None):
        from tabicl import TabICLClassifier
        params = {**DEFAULT_PARAMS["tabicl"], **self._overrides}
        self._model = TabICLClassifier(device=params.get("device", "cpu"))

        if sample_weight is None:
            self._model.fit(np.asarray(X), np.asarray(y))
        else:
            rng = np.random.default_rng(42)
            p = np.asarray(sample_weight, dtype=np.float64); p = p / p.sum()
            idx = rng.choice(len(y), size=len(y), replace=True, p=p)
            self._model.fit(np.asarray(X)[idx], np.asarray(y)[idx])
        self._best_iteration = 1
        return self

    def predict_proba(self, X):
        return self._model.predict_proba(np.asarray(X))

    def save(self, path):
        import joblib
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, str(path))

    def load(self, path):
        import joblib
        self._model = joblib.load(str(path))
        return self


class MitraBCR(BaseBCRModel):
    """Wraps Mitra (AWS) tabular foundation model via AutoGluon.

    Mitra is loaded through autogluon.tabular's TabularPredictor with the
    'MITRA' model, which downloads weights from HuggingFace on first use.

    Requires: pip install autogluon.tabular[mitra]
    """
    name = "mitra"

    def fit(self, X, y, sample_weight=None, eval_set=None):
        try:
            from autogluon.tabular import TabularPredictor
        except ImportError as e:
            raise ImportError(
                "Mitra requires AutoGluon: pip install 'autogluon.tabular[mitra]'"
            ) from e

        import pandas as pd
        df = pd.DataFrame(np.asarray(X))
        df["__y__"] = np.asarray(y)

        self._model = TabularPredictor(label="__y__", verbosity=0).fit(
            df, hyperparameters={"MITRA": {}},
            sample_weight=(
                pd.Series(sample_weight, name="__w__")
                if sample_weight is not None else None
            ),
            time_limit=None,
        )
        self._best_iteration = 1
        return self

    def predict_proba(self, X):
        import pandas as pd
        df = pd.DataFrame(np.asarray(X))
        out = self._model.predict_proba(df)
        # AutoGluon returns a DataFrame indexed by class; ensure (n, 2) order.
        if hasattr(out, "values"):
            cols = list(out.columns)
            order = [cols.index(0), cols.index(1)] if 0 in cols else [0, 1]
            return out.values[:, order]
        return np.asarray(out)

    def save(self, path):
        import joblib
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, str(path))

    def load(self, path):
        import joblib
        self._model = joblib.load(str(path))
        return self


# ══════════════════════════════════════════════════════════════════════════════
# 8. Factory
# ══════════════════════════════════════════════════════════════════════════════


_MODEL_CLASSES = {
    "elasticnet":            ElasticNetBCR,
    "ridge":                 RidgeBCR,
    "lasso":                 LassoBCR,
    "random_forest":         RandomForestBCR,
    "extra_trees":           ExtraTreesBCR,
    "histgradientboosting":  HistGradientBoostingBCR,
    "xgboost":               XGBoostBCR,
    "lightgbm":              LightGBMBCR,
    "catboost":              CatBoostBCR,
    "ft_transformer":        FTTransformerBCR,
    "saint":                 SAINTBCR,
    "tab_transformer":       TabTransformerBCR,
    "resnet_tabular":        ResNetTabularBCR,
    "node":                  NODEBCR,
    "tabnet":                TabNetBCR,
    "tabpfn":                TabPFNBCR,
    "tabicl":                TabICLBCR,
    "mitra":                 MitraBCR,
}

# Models whose `device` kwarg is consumed natively by their PyTorch-based fit().
_TORCH_NATIVE_DEVICE = {
    "ft_transformer", "saint", "tab_transformer", "resnet_tabular", "node",
    "tabnet", "tabpfn", "tabicl", "mitra",
}


def create_model(
    name: str, scale_pos_weight: float, device: str = "cpu", **overrides
) -> BaseBCRModel:
    """Factory: create a BaseBCRModel by name with proper GPU routing.

    GPU routing
    -----------
    - device='cuda':
        XGBoost          → device="cuda"
        LightGBM         → device="gpu"
        CatBoost         → task_type="GPU"
        all torch models → device="cuda"
        TabPFN/TabICL    → device="cuda"
        Mitra            → device="cuda" (consumed by AutoGluon)
    - device='mps':
        Tree backends    → CPU (no MPS support)
        Torch tabular    → device="mps"  (PyTorch native MPS)
        Foundation       → CPU fallback (foundation models often lack MPS)
    - device='cpu':
        no GPU params for anyone.
    """
    name = name.lower()

    if device == "cuda":
        if name == "xgboost":
            overrides.setdefault("device", "cuda")
        elif name == "lightgbm":
            overrides.setdefault("device", "gpu")
        elif name == "catboost":
            overrides.setdefault("task_type", "GPU")
        elif name in _TORCH_NATIVE_DEVICE:
            overrides.setdefault("device", "cuda")
    elif device == "mps":
        if name in {"ft_transformer", "saint", "tab_transformer",
                    "resnet_tabular", "node"}:
            overrides.setdefault("device", "mps")
        # tree backends, foundation models: stay on CPU (default DEFAULT_PARAMS)

    if name not in _MODEL_CLASSES:
        raise ValueError(
            f"Unknown model: {name!r}. Supported: {SUPPORTED_MODELS}"
        )
    return _MODEL_CLASSES[name](scale_pos_weight, **overrides)
