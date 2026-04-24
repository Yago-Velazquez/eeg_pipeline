"""
bad_channel_rejection/features.py

Feature preprocessing pipeline for BCR.

Transforms the raw ~156-column feature matrix from dataset.py into a clean,
scaled, deduplicated ~138-feature input ready for boosting models.

Pipeline (stateful, fit on train / apply to val — no leakage):
    1. Drop 18 near-redundant columns (r > 0.99 with their 'mean' counterpart)
    2. RobustScaler on continuous columns (fit on train only)
    3. 'impedance_missing' passed through unscaled (binary)
    4. VarianceThreshold to drop any near-zero variance survivors
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import RobustScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .logging_config import setup_logging

logger = setup_logging(__name__)

BINARY_COLS: list[str] = ["impedance_missing"]

REDUNDANT_DROP_COLS: list[str] = [
    " Standard deviation (median)",
    " Standard deviation (1st quartile)",
    " Standard deviation (3rd quartile)",
    " Standard deviation (maximum)",
    " Global correlation (median)",
    " Global correlation (1st quartile)",
    " Global correlation (3rd quartile)",
    " Signal-wide relative residuals (3rd quartile)",
    " Signal-wide relative residuals (minimun)",
    " Window-specific relative residuals (median)",
    " Window-specific PCA (median)",
    " Window-specific PCA (3rd quartile)",
    " Window-specific independence (median)",
    " Window-specific reconstruction correlation (median)",
    " Correllation with neighbors (median)",
    " Correllation with second-degree neighbors (median)",
    " Low gamma/high gamma ratio (median)",
    " Signal-wide PCA (mean)",
]


class FeaturePreprocessor:
    """Fit-on-train, apply-to-val feature preprocessing for BCR.

    Attributes (after fit_transform)
    --------------------------------
    feature_names_in_  : list[str]
    feature_names_out_ : list[str]
    n_features_out     : int
    """

    def __init__(self, variance_threshold: float = 0.0):
        self.variance_threshold = variance_threshold
        self.scaler = RobustScaler()
        self.selector = VarianceThreshold(threshold=variance_threshold)
        self.feature_names_in_: list[str] = []
        self.feature_names_out_: list[str] = []
        self._fitted = False

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self._preprocess(X, fit=True)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        assert self._fitted, "Call fit_transform() before transform()"
        return self._preprocess(X, fit=False)

    @property
    def n_features_out(self) -> int:
        return len(self.feature_names_out_)

    def _preprocess(self, X: pd.DataFrame, fit: bool) -> np.ndarray:
        X_df = X.copy()
        cols_to_drop = [c for c in REDUNDANT_DROP_COLS if c in X_df.columns]
        X_df = X_df.drop(columns=cols_to_drop)

        binary_present = [c for c in BINARY_COLS if c in X_df.columns]
        cont_cols = [c for c in X_df.columns if c not in binary_present]

        X_cont = X_df[cont_cols]
        X_binary = X_df[binary_present]

        if fit:
            X_cont_scaled = (
                self.scaler.fit_transform(X_cont)
                if X_cont.shape[1] > 0
                else X_cont
            )
            self.feature_names_in_ = cont_cols + binary_present
        else:
            X_cont_scaled = (
                self.scaler.transform(X_cont)
                if X_cont.shape[1] > 0
                else X_cont
            )

        X_scaled = np.hstack([X_cont_scaled, X_binary.to_numpy()])
        scaled_cols = cont_cols + binary_present

        if fit:
            X_out = self.selector.fit_transform(X_scaled)
            kept_mask = self.selector.get_support()
            self.feature_names_out_ = [
                scaled_cols[i] for i, keep in enumerate(kept_mask) if keep
            ]
            self._fitted = True
        else:
            X_out = self.selector.transform(X_scaled)

        assert not np.isnan(X_out).any(), "NaN found after preprocessing"
        assert not np.isinf(X_out).any(), "Inf found after preprocessing"
        return X_out

    def plot_correlation_heatmap(
        self,
        X: pd.DataFrame,
        title: str = "Feature Correlation Heatmap",
        save_path: str | None = None,
    ) -> None:
        corr = X.corr().abs()
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(
            corr, ax=ax, cmap="YlOrRd", vmin=0, vmax=1, square=True,
            linewidths=0, cbar_kws={"shrink": 0.6},
            xticklabels=False, yticklabels=False,
        )
        ax.set_title(title, fontsize=13, pad=12)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=120, bbox_inches="tight")
            logger.info(f"Saved: {save_path}")
        plt.close(fig)

    def print_summary(self) -> None:
        assert self._fitted, "Call fit_transform() first"
        n_raw = len(self.feature_names_in_)
        n_dropped = len(REDUNDANT_DROP_COLS)
        n_var_drop = n_raw - self.n_features_out
        logger.info("FeaturePreprocessor summary:")
        logger.info(f"  Raw features in   : {n_raw}")
        logger.info(f"  Redundant dropped : {n_dropped}")
        logger.info(f"  Variance-dropped  : {n_var_drop}")
        logger.info(f"  Features out      : {self.n_features_out}")

    def save_feature_list(
        self, path: str = "configs/bcr_selected_features.json"
    ) -> None:
        assert self._fitted, "Call fit_transform() first"
        out = {
            "n_features_raw": len(self.feature_names_in_),
            "n_redundant_dropped": len(REDUNDANT_DROP_COLS),
            "n_features_selected": self.n_features_out,
            "feature_names_out": self.feature_names_out_,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(out, indent=2))
        logger.info(f"Saved {self.n_features_out} features -> {path}")


def preprocess_fold(
    X_tr_df: pd.DataFrame,
    X_va_df: pd.DataFrame,
    y_tr: np.ndarray,
    channel_labels_tr: np.ndarray | None = None,
    channel_labels_va: np.ndarray | None = None,
    use_engineered_features: bool = False,
    engineering_kwargs: dict | None = None,
    variance_threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, list[str], FeaturePreprocessor, object | None]:
    """Per-fold preprocessing: fit on train, transform val.

    Runs FeaturePreprocessor always. If ``use_engineered_features`` is
    True, also runs FeatureEngineeringPipeline afterwards (fit on train
    only). Returns (X_tr_out, X_va_out, feature_names_out, preprocessor,
    engineering_or_None).
    """
    prep = FeaturePreprocessor(variance_threshold=variance_threshold)
    X_tr = prep.fit_transform(X_tr_df)
    X_va = prep.transform(X_va_df)
    names = list(prep.feature_names_out_)

    fe = None
    if use_engineered_features:
        from .feature_engineering import FeatureEngineeringPipeline

        fe_kwargs = dict(engineering_kwargs or {})
        fe = FeatureEngineeringPipeline(**fe_kwargs)
        needs_channels = fe.channel_encoder is not None
        if needs_channels:
            assert channel_labels_tr is not None and channel_labels_va is not None, (
                "channel_labels_tr and channel_labels_va required when "
                "ChannelBadRateEncoder is enabled"
            )
        X_tr, names_tr = fe.fit_transform(
            X_tr, names, y_tr, channel_labels_tr
        )
        X_va, names_va = fe.transform(X_va, names, channel_labels_va)
        assert names_tr == names_va, (
            "Train/val feature names diverged after engineering"
        )
        names = names_tr

    assert not np.isnan(X_tr).any(), "NaN in X_tr after preprocess_fold"
    assert not np.isnan(X_va).any(), "NaN in X_va after preprocess_fold"
    assert not np.isinf(X_tr).any(), "Inf in X_tr after preprocess_fold"
    assert not np.isinf(X_va).any(), "Inf in X_va after preprocess_fold"
    return X_tr, X_va, names, prep, fe


def run_correlation_audit(
    feature_cols: list[str],
    X: np.ndarray,
    threshold: float = 0.99,
    print_pairs: bool = True,
) -> pd.DataFrame:
    """Identify near-redundant feature pairs (r > threshold)."""
    X_df = pd.DataFrame(X, columns=feature_cols)
    corr = X_df.corr().abs()
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    upper = corr.where(mask)
    pairs = (
        upper.stack()
        .reset_index()
        .rename(columns={"level_0": "feat_a", "level_1": "feat_b", 0: "r"})
        .query(f"r > {threshold}")
        .sort_values("r", ascending=False)
        .reset_index(drop=True)
    )
    if print_pairs:
        logger.info(f"Pairs with r > {threshold}: {len(pairs)}")
        unique_drops = sorted(set(pairs["feat_b"].tolist()))
        logger.info(f"Unique columns to drop: {len(unique_drops)}")
    return pairs
