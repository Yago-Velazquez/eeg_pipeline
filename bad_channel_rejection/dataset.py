"""
bad_channel_rejection/dataset.py

Load and feature-engineer the BCR CSV. Delegates labelling to label_quality.py.

Pipeline
--------
  load_bcr_data        -> parse Session into subject_id / visit / site
  add_missingness_flag -> engineer impedance_missing BEFORE imputation
  impute_and_encode    -> median-impute floats, ordinal-encode channels
  build_label_artifacts -> (from label_quality) y_hard, y_soft?, sample_weights
  build_feature_matrix -> full pipeline producing X, labels, weights, groups

Labelling strategies (selected via `label_strategy` argument). All eight
emit hard binary labels + per-sample weights; they differ only in how the
latent-true-label posterior is estimated and how it is converted to a weight:
  - "hard_threshold"   : y = 1[score >= 2],  w = 1 (uniform)
  - "entropy_weights"  : y = 1[score >= 2],  w = 1 - H / H_max
  - "dawid_skene"      : y = 1[q_DS >= 0.5], w = |q - 0.5| * 2
  - "dawid_skene_soft" : y = 1[q_DS >= 0.5], w = P(chosen label | q)
  - "glad"             : y = 1[q_GL >= 0.5], w = |q - 0.5| * 2
  - "glad_soft"        : y = 1[q_GL >= 0.5], w = P(chosen label | q)
  - "mace"             : y = 1[q_MA >= 0.5], w = |q - 0.5| * 2
  - "mace_soft"        : y = 1[q_MA >= 0.5], w = P(chosen label | q)

`y_soft` (the raw Dawid-Skene posterior q_i) is still exposed on the
returned dict for both DS strategies — useful for SHAP and inspecting
raw posteriors — but is never routed to `model.fit()` as a target.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from .label_quality import LabelArtifacts, build_label_artifacts
from .logging_config import setup_logging

logger = setup_logging(__name__)

NON_FEATURE_COLS = [
    "Session", "Task", "Channel labels",
    "Bad (site 2)", "Bad (site 3)", "Bad (site 4a)", "Bad (site 4b)",
    "Bad (score)",
    "Impedance (start)", "Impedance (end)",
    "subject_id", "visit", "site",
]


def load_bcr_data(path: str) -> pd.DataFrame:
    """Load BCR CSV and parse Session column into subject_id, visit, site."""
    df = pd.read_csv(path)

    def parse_session(s: str):
        parts = str(s).split("-")
        return int(parts[1]), int(parts[2]), int(parts[0])

    parsed = df["Session"].apply(
        lambda x: pd.Series(
            parse_session(x), index=["subject_id", "site", "visit"]
        )
    )
    df = pd.concat([parsed, df], axis=1)

    logger.info(f"Loaded data: shape={df.shape}")
    logger.info(f"Unique subjects: {df['subject_id'].nunique()}")
    logger.info(f"Unique visits: {sorted(df['visit'].unique())}")
    logger.info(
        f"Unique sites: {df['site'].nunique()} (constant — will be dropped)"
    )
    return df


def add_missingness_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add impedance_missing flag BEFORE imputation.

    Visit 3 has 78.3% missing impedance vs 0% for visits 1, 2, 4. This is
    structured missingness that carries predictive signal and must be
    preserved as an explicit binary feature.
    """
    df = df.copy()
    df["impedance_missing"] = df["Impedance (start)"].isnull().astype(int)
    visit_miss = df.groupby("visit")["impedance_missing"].mean().round(3)
    logger.info(f"Impedance missing by visit:\n{visit_miss}")
    return df


def impute_and_encode_channels(
    df: pd.DataFrame, feature_cols: list
) -> tuple[pd.DataFrame, SimpleImputer]:
    """Median-impute float features, ordinal-encode channel labels by bad rate.

    impedance_missing stays binary — never imputed or log-transformed.
    inf/-inf values are replaced with NaN before imputation.
    """
    df = df.copy()

    channel_order = (
        df.groupby("Channel labels")["Bad (score)"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    channel_map = {ch: i for i, ch in enumerate(channel_order)}
    df["channel_label_enc"] = df["Channel labels"].map(channel_map)
    logger.info(f"Top-5 channels by bad rate: {channel_order[:5]}")

    impute_cols = [
        c for c in feature_cols
        if c != "impedance_missing" and df[c].dtype != object
    ]

    inf_cols = [c for c in impute_cols if np.isinf(df[c]).any()]
    if inf_cols:
        logger.info(
            f"Columns with inf values: {len(inf_cols)} — replacing with NaN"
        )
    df[impute_cols] = df[impute_cols].replace([np.inf, -np.inf], np.nan)

    imp = SimpleImputer(strategy="median")
    df[impute_cols] = imp.fit_transform(df[impute_cols])

    all_feature_cols = feature_cols + ["channel_label_enc"]
    remaining_nans = df[all_feature_cols].isnull().sum().sum()
    assert remaining_nans == 0, f"Still {remaining_nans} NaNs after imputation"
    logger.info("Zero NaN/inf values confirmed after imputation")

    return df, imp


def build_targets_and_groups(
    df: pd.DataFrame,
    label_strategy: str = "hard_threshold",
    bad_threshold: int = 2,
) -> tuple[LabelArtifacts, np.ndarray, float]:
    """Build labels + weights via the chosen strategy, plus groups and spw.

    Returns
    -------
    artifacts : LabelArtifacts
        y_hard, optional y_soft, sample_weights, strategy name, metadata.
    groups : np.ndarray (n,)
        subject_id for GroupKFold splitting.
    scale_pos_weight : float
        n_good / n_bad computed from y_hard. Used by boosting classifiers.
    """
    artifacts = build_label_artifacts(
        df, strategy=label_strategy, threshold=bad_threshold
    )
    y = artifacts.y_hard
    n_neg = int((y == 0).sum())
    n_pos = int((y == 1).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)

    logger.info(f"Label strategy: {artifacts.strategy}")
    logger.info(
        f"Class distribution — good: {n_neg}, bad: {n_pos}  "
        f"(bad_rate={n_pos/(n_neg+n_pos):.4f})"
    )
    logger.info(f"scale_pos_weight = {scale_pos_weight:.2f}")

    groups = df["subject_id"].values
    assert len(np.unique(groups)) == 43, (
        f"Expected 43 subjects, got {len(np.unique(groups))}"
    )

    return artifacts, groups, scale_pos_weight


def build_feature_matrix(
    csv_path: str,
    save_cols_to: str | None = None,
    bad_threshold: int = 2,
    label_strategy: str = "hard_threshold",
) -> dict:
    """Full BCR pipeline.

    Parameters
    ----------
    csv_path : str
        Path to Bad_channels_for_ML.csv.
    save_cols_to : str or None
        Optional path to save feature column names as JSON.
    bad_threshold : int
        Score threshold for the hard_threshold / entropy_weights strategies.
    label_strategy : str
        One of {"hard_threshold", "entropy_weights", "dawid_skene",
        "dawid_skene_soft", "glad", "glad_soft", "mace", "mace_soft"}.

    Returns
    -------
    dict with keys:
        X : np.ndarray (n, n_features)
        y_hard : np.ndarray (n,) binary
        y_soft : np.ndarray (n,) or None
        sample_weights : np.ndarray (n,) float32
        groups : np.ndarray (n,)
        channel_labels : np.ndarray (n,) dtype=object  (raw strings)
        feature_cols : list[str]
        scale_pos_weight : float
        label_strategy : str
        label_metadata : dict
    """
    df = load_bcr_data(csv_path)
    df = add_missingness_flags(df)

    feature_cols = [
        c for c in df.columns
        if c not in NON_FEATURE_COLS
        and c != "site"
        and df[c].dtype != object
    ]

    df, _ = impute_and_encode_channels(df, feature_cols)
    feature_cols = feature_cols + ["channel_label_enc"]

    artifacts, groups, spw = build_targets_and_groups(
        df, label_strategy=label_strategy, bad_threshold=bad_threshold
    )

    X = df[feature_cols].values
    channel_labels = df["Channel labels"].astype(str).to_numpy()
    assert X.shape[0] == 18900, f"Expected 18900 rows, got {X.shape[0]}"
    assert len(channel_labels) == X.shape[0]
    assert not np.isnan(X).any(), "NaN found in feature matrix"
    assert not np.isinf(X).any(), "Inf found in feature matrix"

    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Unique subjects: {len(np.unique(groups))}")
    logger.info(f"Unique channel labels: {len(np.unique(channel_labels))}")

    if save_cols_to:
        pathlib.Path(save_cols_to).parent.mkdir(parents=True, exist_ok=True)
        with open(save_cols_to, "w") as f:
            json.dump(feature_cols, f, indent=2)
        logger.info(f"Feature cols saved -> {save_cols_to}")

    return {
        "X": X,
        "y_hard": artifacts.y_hard,
        "y_soft": artifacts.y_soft,
        "sample_weights": artifacts.sample_weights,
        "groups": groups,
        "channel_labels": channel_labels,
        "feature_cols": feature_cols,
        "scale_pos_weight": spw,
        "label_strategy": artifacts.strategy,
        "label_metadata": artifacts.metadata or {},
    }


if __name__ == "__main__":
    out = build_feature_matrix(
        "data/raw/Bad_channels_for_ML.csv",
        save_cols_to="configs/feature_cols.json",
        label_strategy="hard_threshold",
    )
    logger.info(
        f"Done. X={out['X'].shape}  "
        f"strategy={out['label_strategy']}  "
        f"bad_rate={out['y_hard'].mean():.4f}"
    )
