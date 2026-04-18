# bad_channel_rejection/features.py
"""
Feature preprocessing pipeline for BCR.

Transforms the raw 156-column feature matrix from dataset.py into a clean,
scaled, deduplicated ~139-feature input ready for XGBoost.

Design
------
- Stateful: fit on train fold, apply to val fold (no leakage)
- Mirrors sklearn API: fit_transform(X_train_df) / transform(X_val_df)
- Input: pd.DataFrame with feature_cols from build_feature_matrix()
- Output: np.ndarray, shape (n_samples, ~139)

Key transforms (in order)
--------------------------
1. Drop 17 near-redundant columns (r > 0.99 with their 'mean' counterpart)
   Identified via run_correlation_audit() on Day 9 — 24 pairs, 17 unique drops
2. RobustScaler on all continuous columns (fit on train only)
3. 'impedance_missing' passed through unscaled (binary column)
4. VarianceThreshold to drop any near-zero variance survivors (fit on train only)

Column naming notes
-------------------
- All 155 signal feature columns have a leading space, e.g. ' Standard deviation (mean)'
- The engineered binary column is 'impedance_missing' (no leading space)
- Impedance (start) / Impedance (end) are NOT in the feature matrix (excluded as metadata)
  -> no log-transform step needed

Confirmed from audit on Day 9:
  Raw features        : 156
  Redundant dropped   : 17 (from 24 correlated pairs -> 17 unique cols)
  Features after dedup: 139
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# ── Binary columns — passed through unscaled ────────────────────────────────
# 'impedance_missing' has no leading space (engineered column from dataset.py)
BINARY_COLS: list[str] = ['impedance_missing']

# ── Near-redundant columns to DROP (r > 0.99 audit, Day 9) ──────────────────
# 24 correlated pairs found -> 17 unique columns to drop.
# Strategy: keep the (mean) statistic per family; drop (median), (1st/3rd quartile),
# (maximum), (minimun) variants when r > 0.99 with the mean.
#
# To reproduce: run_correlation_audit(feature_cols, X) defined at bottom of file.
REDUNDANT_DROP_COLS: list[str] = [
    # Standard deviation family
    ' Standard deviation (median)',
    ' Standard deviation (1st quartile)',
    ' Standard deviation (3rd quartile)',
    ' Standard deviation (maximum)',

    # Global correlation family
    ' Global correlation (median)',
    ' Global correlation (1st quartile)',
    ' Global correlation (3rd quartile)',

    # Signal-wide relative residuals family
    ' Signal-wide relative residuals (3rd quartile)',
    ' Signal-wide relative residuals (minimun)',

    # Window-specific relative residuals family
    ' Window-specific relative residuals (median)',

    # Window-specific PCA family
    ' Window-specific PCA (median)',
    ' Window-specific PCA (3rd quartile)',

    # Window-specific independence family
    ' Window-specific independence (median)',

    # Window-specific reconstruction correlation family
    ' Window-specific reconstruction correlation (median)',

    # Correlation with neighbors families
    ' Correllation with neighbors (median)',
    ' Correllation with second-degree neighbors (median)',

    # Low gamma/high gamma ratio family
    ' Low gamma/high gamma ratio (median)',
    
    # Constant after inf-imputation - zero information
    ' Signal-wide PCA (mean)'
]


class FeaturePreprocessor:
    """
    Fit-on-train, apply-to-val feature preprocessing for BCR.

    Usage
    -----
    prep = FeaturePreprocessor()
    X_train = prep.fit_transform(X_train_df)   # fits scaler + selector
    X_val   = prep.transform(X_val_df)         # applies fitted transforms only

    Attributes (after fit_transform)
    ----------------------------------
    feature_names_in_  : list[str] -- column names entering the scaler
    feature_names_out_ : list[str] -- column names surviving VarianceThreshold
    n_features_out     : int
    """

    def __init__(self, variance_threshold: float = 0.0):
        self.variance_threshold = variance_threshold
        self.scaler   = RobustScaler()
        self.selector = VarianceThreshold(threshold=variance_threshold)
        self.feature_names_in_:  list[str] = []
        self.feature_names_out_: list[str] = []
        self._fitted = False

    # ── Public API ────────────────────────────────────────────────────────────

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit on X (train fold) and return transformed array."""
        return self._preprocess(X, fit=True)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Apply fitted transforms to X (val/test fold). No re-fitting."""
        assert self._fitted, "Call fit_transform() before transform()"
        return self._preprocess(X, fit=False)

    @property
    def n_features_out(self) -> int:
        return len(self.feature_names_out_)

    # ── Core pipeline ─────────────────────────────────────────────────────────

    def _preprocess(self, X: pd.DataFrame, fit: bool) -> np.ndarray:
        """
        Full preprocessing pipeline:
          1. Drop near-redundant columns
          2. Separate binary from continuous
          3. RobustScaler on continuous (fit only on train)
          4. Recombine continuous + binary
          5. VarianceThreshold (fit only on train)
          6. Assert no NaN / Inf
        """
        X_df = X.copy()

        # 1. Drop near-redundant columns (only those actually present)
        cols_to_drop = [c for c in REDUNDANT_DROP_COLS if c in X_df.columns]
        X_df = X_df.drop(columns=cols_to_drop)

        # 2. Separate binary (unscaled) from continuous
        binary_present = [c for c in BINARY_COLS if c in X_df.columns]
        cont_cols      = [c for c in X_df.columns if c not in binary_present]

        X_cont   = X_df[cont_cols]
        X_binary = X_df[binary_present]

        # 3. RobustScaler — fit on train only
        if fit:
            X_cont_scaled = self.scaler.fit_transform(X_cont) if X_cont.shape[1] > 0 else X_cont
            self.feature_names_in_ = cont_cols + binary_present
        else:
            X_cont_scaled = self.scaler.transform(X_cont) if X_cont.shape[1] > 0 else X_cont


        # 4. Recombine scaled continuous + unscaled binary
        X_scaled    = np.hstack([X_cont_scaled, X_binary.to_numpy()])
        scaled_cols = cont_cols + binary_present

        # 5. VarianceThreshold — fit on train only
        if fit:
            X_out     = self.selector.fit_transform(X_scaled)
            kept_mask = self.selector.get_support()
            self.feature_names_out_ = [
                scaled_cols[i] for i, keep in enumerate(kept_mask) if keep
            ]
            self._fitted = True
        else:
            X_out = self.selector.transform(X_scaled)

        # 6. Sanity guards
        assert not np.isnan(X_out).any(), \
            f"NaN found after preprocessing! Shape: {X_out.shape}"
        assert not np.isinf(X_out).any(), \
            f"Inf found after preprocessing! Shape: {X_out.shape}"

        return X_out

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def plot_correlation_heatmap(
        self,
        X: pd.DataFrame,
        title: str = "Feature Correlation Heatmap",
        save_path: str | None = None,
    ) -> None:
        """
        Plot absolute correlation heatmap. Call before and after preprocessing
        to visually verify deduplication removed the high-correlation blocks.
        """
        corr = X.corr().abs()
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(
            corr,
            ax=ax,
            cmap='YlOrRd',
            vmin=0, vmax=1,
            square=True,
            linewidths=0,
            cbar_kws={'shrink': 0.6},
            xticklabels=False,
            yticklabels=False,
        )
        ax.set_title(title, fontsize=13, pad=12)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=120, bbox_inches='tight')
            print(f"Saved: {save_path}")
        plt.close(fig)

    def print_summary(self) -> None:
        """Print a human-readable summary of what the preprocessor did."""
        assert self._fitted, "Call fit_transform() first"
        n_raw      = 156
        n_dropped  = len(REDUNDANT_DROP_COLS)
        n_after_dd = n_raw - n_dropped
        n_var_drop = n_after_dd - self.n_features_out
        print("FeaturePreprocessor summary")
        print(f"  Raw features in    : {n_raw}")
        print(f"  Redundant dropped  : {n_dropped}")
        print(f"  After dedup        : {n_after_dd}")
        print(f"  Variance-dropped   : {n_var_drop}")
        print(f"  Features out       : {self.n_features_out}")
        print(f"  Binary cols kept   : {[c for c in BINARY_COLS if c in self.feature_names_out_]}")

    def save_feature_list(
        self, path: str = "configs/bcr_selected_features.json"
    ) -> None:
        """Persist final feature names to configs/ for reproducibility."""
        assert self._fitted, "Call fit_transform() first"
        out = {
            "n_features_raw":      156,
            "n_redundant_dropped": len(REDUNDANT_DROP_COLS),
            "n_features_selected": self.n_features_out,
            "feature_names_out":   self.feature_names_out_,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(out, indent=2))
        print(f"Saved {self.n_features_out} features -> {path}")


# ── Convenience audit function ────────────────────────────────────────────────

def run_correlation_audit(
    feature_cols: list[str],
    X: np.ndarray,
    threshold: float = 0.99,
    print_pairs: bool = True,
) -> pd.DataFrame:
    """
    One-time audit to identify near-redundant feature pairs.
    Run once to verify / update REDUNDANT_DROP_COLS.

    Usage
    -----
    from bad_channel_rejection.dataset import build_feature_matrix
    from bad_channel_rejection.features import run_correlation_audit
    X, y, groups, feature_cols, spw = build_feature_matrix('data/raw/Bad_channels_for_ML.csv')
    pairs = run_correlation_audit(feature_cols, X)
    """
    X_df  = pd.DataFrame(X, columns=feature_cols)
    corr  = X_df.corr().abs()
    mask  = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    upper = corr.where(mask)
    pairs = (
        upper.stack()
             .reset_index()
             .rename(columns={'level_0': 'feat_a', 'level_1': 'feat_b', 0: 'r'})
             .query(f'r > {threshold}')
             .sort_values('r', ascending=False)
             .reset_index(drop=True)
    )
    if print_pairs:
        print(f"\nPairs with r > {threshold}: {len(pairs)}")
        print(pairs.to_string(index=False))
        unique_drops = sorted(set(pairs['feat_b'].tolist()))
        print(f"\nUnique columns to drop: {len(unique_drops)}")
        for c in unique_drops:
            print(f"  {repr(c)}")
    return pairs
