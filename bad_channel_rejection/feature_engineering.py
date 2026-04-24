"""
bad_channel_rejection/feature_engineering.py

Label-aware feature transforms for BCR. All three are fit-on-train,
transform-on-val only — they MUST be instantiated fresh per CV fold.

Transforms
----------
ChannelBadRateEncoder
    Replaces the globally-computed ordinal ``channel_label_enc`` with a
    continuous per-channel bad rate computed from the training fold's
    labels. Channels unseen at fit time fall back to the training-fold
    global bad rate.

SpatialFeaturePruner
    Keeps only the top-K spatial (neighbour-correlation / global-
    correlation) features ranked by mutual information with the
    training-fold label. All non-spatial columns pass through untouched.

ImpedanceInteractionFeatures
    Appends ``impedance_missing × <feature>`` interaction columns for
    the top-K decomposition features (ranked by MI on the training
    fold). Raw decomposition features are kept alongside the new
    interactions so the tree can split on both branches.

The three are composed by :class:`FeatureEngineeringPipeline`, which
owns ordering, column-name bookkeeping, and logging. Chain is applied
AFTER ``FeaturePreprocessor`` — inputs are a dense numpy array plus the
matching list of column names.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_selection import mutual_info_classif

from .logging_config import setup_logging

logger = setup_logging(__name__)


SPATIAL_PATTERNS: list[str] = [
    " Correllation with neighbors",
    " Correllation with second-degree",
    " Global correlation",
]

DECOMPOSITION_PATTERNS: list[str] = [
    " PCA",
    " ICA",
    " residuals",
    " reconstruction",
    " Kurtosis",
    " Low gamma",
    " independence",
    " unmixing",
]


def _is_spatial(name: str) -> bool:
    return any(p in name for p in SPATIAL_PATTERNS)


def _is_decomposition(name: str) -> bool:
    return any(p in name for p in DECOMPOSITION_PATTERNS)


class ChannelBadRateEncoder:
    """Per-channel bad-rate encoding fit on training labels only.

    Parameters
    ----------
    new_col_name : str
        Column name appended to the output matrix.
    drop_original_ordinal : bool
        If True (default), drop ``channel_label_enc`` from the output.
    """

    def __init__(
        self,
        new_col_name: str = "channel_bad_rate",
        drop_original_ordinal: bool = True,
    ):
        self.new_col_name = new_col_name
        self.drop_original_ordinal = drop_original_ordinal
        self.rates_: dict[str, float] = {}
        self.global_rate_: float = 0.0
        self._fitted = False

    def fit(
        self,
        X: np.ndarray,
        feature_names: list[str],
        y: np.ndarray,
        channel_labels: np.ndarray,
    ) -> "ChannelBadRateEncoder":
        y = np.asarray(y).astype(float)
        channel_labels = np.asarray(channel_labels).astype(str)
        assert len(channel_labels) == len(y), (
            f"channel_labels length {len(channel_labels)} != y length {len(y)}"
        )

        self.global_rate_ = float(y.mean())
        unique = np.unique(channel_labels)
        self.rates_ = {
            str(ch): float(y[channel_labels == ch].mean()) for ch in unique
        }
        self._fitted = True
        return self

    def transform(
        self,
        X: np.ndarray,
        feature_names: list[str],
        channel_labels: np.ndarray,
    ) -> tuple[np.ndarray, list[str]]:
        assert self._fitted, "Call fit() before transform()"
        assert len(channel_labels) == X.shape[0], (
            f"channel_labels length {len(channel_labels)} != X rows {X.shape[0]}"
        )
        channel_labels = np.asarray(channel_labels).astype(str)

        mapped = np.fromiter(
            (self.rates_.get(str(ch), self.global_rate_) for ch in channel_labels),
            dtype=float,
            count=len(channel_labels),
        ).reshape(-1, 1)

        names = list(feature_names)
        X_out = X
        if self.drop_original_ordinal and "channel_label_enc" in names:
            drop_idx = names.index("channel_label_enc")
            X_out = np.delete(X_out, drop_idx, axis=1)
            names = [n for n in names if n != "channel_label_enc"]

        X_out = np.hstack([X_out, mapped])
        names = names + [self.new_col_name]
        return X_out, names

    def fit_transform(
        self,
        X: np.ndarray,
        feature_names: list[str],
        y: np.ndarray,
        channel_labels: np.ndarray,
    ) -> tuple[np.ndarray, list[str]]:
        self.fit(X, feature_names, y, channel_labels)
        return self.transform(X, feature_names, channel_labels)


class SpatialFeaturePruner:
    """Keep only top-K spatial features by MI with the training label."""

    def __init__(self, top_k: int = 3, random_state: int = 42):
        self.top_k = top_k
        self.random_state = random_state
        self.kept_spatial_: list[str] = []
        self.dropped_spatial_: list[str] = []
        self.mi_scores_: dict[str, float] = {}
        self._fitted = False

    def fit(
        self,
        X: np.ndarray,
        feature_names: list[str],
        y: np.ndarray,
    ) -> "SpatialFeaturePruner":
        y = np.asarray(y).astype(int)
        spatial_idx = [i for i, n in enumerate(feature_names) if _is_spatial(n)]

        if not spatial_idx:
            self.kept_spatial_ = []
            self.dropped_spatial_ = []
            self.mi_scores_ = {}
            self._fitted = True
            return self

        spatial_names = [feature_names[i] for i in spatial_idx]
        X_spatial = X[:, spatial_idx]
        mi = mutual_info_classif(
            X_spatial, y, random_state=self.random_state
        )
        self.mi_scores_ = {
            name: float(score) for name, score in zip(spatial_names, mi)
        }

        k = min(self.top_k, len(spatial_idx))
        order = np.argsort(mi)[::-1]
        top_local = order[:k]
        self.kept_spatial_ = [spatial_names[i] for i in top_local]
        self.dropped_spatial_ = [
            n for n in spatial_names if n not in self.kept_spatial_
        ]
        self._fitted = True
        return self

    def transform(
        self, X: np.ndarray, feature_names: list[str]
    ) -> tuple[np.ndarray, list[str]]:
        assert self._fitted, "Call fit() before transform()"
        kept_set = set(self.kept_spatial_)
        keep_mask = np.array(
            [(not _is_spatial(n)) or (n in kept_set) for n in feature_names],
            dtype=bool,
        )
        X_out = X[:, keep_mask]
        names_out = [n for n, k in zip(feature_names, keep_mask) if k]
        return X_out, names_out

    def fit_transform(
        self, X: np.ndarray, feature_names: list[str], y: np.ndarray
    ) -> tuple[np.ndarray, list[str]]:
        self.fit(X, feature_names, y)
        return self.transform(X, feature_names)


class ImpedanceInteractionFeatures:
    """Append ``impedance_missing × top-K decomposition feature`` columns.

    Raw decomposition features are NOT removed — the tree gets both the
    original feature (splittable on its own) and the interaction
    (splittable conditional on ``impedance_missing == 1``).
    """

    def __init__(
        self,
        top_k: int = 5,
        random_state: int = 42,
        impedance_col: str = "impedance_missing",
    ):
        self.top_k = top_k
        self.random_state = random_state
        self.impedance_col = impedance_col
        self.top_decomp_names_: list[str] = []
        self.mi_scores_: dict[str, float] = {}
        self._fitted = False

    def fit(
        self,
        X: np.ndarray,
        feature_names: list[str],
        y: np.ndarray,
    ) -> "ImpedanceInteractionFeatures":
        y = np.asarray(y).astype(int)
        if self.impedance_col not in feature_names:
            logger.warning(
                f"{self.impedance_col!r} not in feature names — "
                "ImpedanceInteractionFeatures will no-op"
            )
            self.top_decomp_names_ = []
            self._fitted = True
            return self

        decomp_idx = [
            i for i, n in enumerate(feature_names) if _is_decomposition(n)
        ]
        if not decomp_idx:
            self.top_decomp_names_ = []
            self._fitted = True
            return self

        decomp_names = [feature_names[i] for i in decomp_idx]
        X_decomp = X[:, decomp_idx]
        mi = mutual_info_classif(
            X_decomp, y, random_state=self.random_state
        )
        self.mi_scores_ = {
            name: float(score) for name, score in zip(decomp_names, mi)
        }

        k = min(self.top_k, len(decomp_idx))
        order = np.argsort(mi)[::-1]
        self.top_decomp_names_ = [decomp_names[i] for i in order[:k]]
        self._fitted = True
        return self

    def transform(
        self, X: np.ndarray, feature_names: list[str]
    ) -> tuple[np.ndarray, list[str]]:
        assert self._fitted, "Call fit() before transform()"
        if not self.top_decomp_names_ or self.impedance_col not in feature_names:
            return X, list(feature_names)

        imp_idx = feature_names.index(self.impedance_col)
        imp_col = X[:, imp_idx].reshape(-1, 1)

        new_cols = []
        new_names = []
        for dname in self.top_decomp_names_:
            di = feature_names.index(dname)
            new_cols.append(X[:, di].reshape(-1, 1) * imp_col)
            new_names.append(f"impedance_missing_x_{dname.strip()}")

        X_out = np.hstack([X] + new_cols)
        names_out = list(feature_names) + new_names
        return X_out, names_out

    def fit_transform(
        self, X: np.ndarray, feature_names: list[str], y: np.ndarray
    ) -> tuple[np.ndarray, list[str]]:
        self.fit(X, feature_names, y)
        return self.transform(X, feature_names)


class FeatureEngineeringPipeline:
    """Chain of the three label-aware transforms.

    All three are independently toggleable so individual contributions
    can be ablated. Fit once on a training fold, then call ``transform``
    on the matching val fold.
    """

    def __init__(
        self,
        use_channel_bad_rate: bool = True,
        use_spatial_pruner: bool = True,
        use_impedance_interactions: bool = True,
        spatial_top_k: int = 3,
        interaction_top_k: int = 5,
        random_state: int = 42,
        verbose: bool = True,
    ):
        self.use_channel_bad_rate = use_channel_bad_rate
        self.use_spatial_pruner = use_spatial_pruner
        self.use_impedance_interactions = use_impedance_interactions
        self.spatial_top_k = spatial_top_k
        self.interaction_top_k = interaction_top_k
        self.random_state = random_state
        self.verbose = verbose

        self.channel_encoder: ChannelBadRateEncoder | None = (
            ChannelBadRateEncoder() if use_channel_bad_rate else None
        )
        self.spatial_pruner: SpatialFeaturePruner | None = (
            SpatialFeaturePruner(top_k=spatial_top_k, random_state=random_state)
            if use_spatial_pruner
            else None
        )
        self.interactions: ImpedanceInteractionFeatures | None = (
            ImpedanceInteractionFeatures(
                top_k=interaction_top_k, random_state=random_state
            )
            if use_impedance_interactions
            else None
        )

        self.feature_names_in_: list[str] = []
        self.feature_names_out_: list[str] = []
        self._fitted = False

    def _log(self, msg: str) -> None:
        if self.verbose:
            logger.info(msg)

    def fit_transform(
        self,
        X: np.ndarray,
        feature_names: list[str],
        y: np.ndarray,
        channel_labels: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        self.feature_names_in_ = list(feature_names)
        names = list(feature_names)
        self._log(
            f"FeatureEngineeringPipeline.fit_transform — input: {len(names)} "
            f"features"
        )

        if self.channel_encoder is not None:
            assert channel_labels is not None, (
                "channel_labels is required when use_channel_bad_rate=True"
            )
            self.channel_encoder.fit(X, names, y, channel_labels)
            X, names = self.channel_encoder.transform(X, names, channel_labels)
            self._log(
                f"  ChannelBadRateEncoder: {len(names)} features "
                f"(learned {len(self.channel_encoder.rates_)} channels, "
                f"fallback global rate={self.channel_encoder.global_rate_:.4f}, "
                f"swapped channel_label_enc -> channel_bad_rate)"
            )

        if self.spatial_pruner is not None:
            self.spatial_pruner.fit(X, names, y)
            X, names = self.spatial_pruner.transform(X, names)
            self._log(
                f"  SpatialFeaturePruner: {len(names)} features "
                f"(kept {len(self.spatial_pruner.kept_spatial_)}/"
                f"{len(self.spatial_pruner.kept_spatial_) + len(self.spatial_pruner.dropped_spatial_)} "
                f"spatial; kept={self.spatial_pruner.kept_spatial_})"
            )

        if self.interactions is not None:
            self.interactions.fit(X, names, y)
            X, names = self.interactions.transform(X, names)
            self._log(
                f"  ImpedanceInteractionFeatures: {len(names)} features "
                f"(added {len(self.interactions.top_decomp_names_)} interactions "
                f"with top decomposition features: "
                f"{self.interactions.top_decomp_names_})"
            )

        self.feature_names_out_ = names
        self._fitted = True
        self._log(
            f"FeatureEngineeringPipeline.fit_transform — output: {len(names)} "
            f"features"
        )
        return X, names

    def transform(
        self,
        X: np.ndarray,
        feature_names: list[str],
        channel_labels: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        assert self._fitted, "Call fit_transform() before transform()"
        names = list(feature_names)
        if self.channel_encoder is not None:
            assert channel_labels is not None, (
                "channel_labels is required for transform when "
                "use_channel_bad_rate=True"
            )
            X, names = self.channel_encoder.transform(X, names, channel_labels)
        if self.spatial_pruner is not None:
            X, names = self.spatial_pruner.transform(X, names)
        if self.interactions is not None:
            X, names = self.interactions.transform(X, names)
        return X, names
