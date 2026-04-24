"""
BCR Dataset — load and feature-engineer the bad channel rejection CSV.

Responsibilities:
  - load_bcr_data()              : load CSV, parse Session into subject_id/visit
  - add_missingness_flags()      : engineer impedance_missing BEFORE imputation
                                   (visit 3 has 78.3% missing — structured, not random)
  - impute_and_encode_channels() : median imputation + ordinal channel encoding
  - build_feature_matrix()       : full pipeline → X, y, groups, feature_cols

Inter-rater disagreement handling (added):
  - build_targets_and_groups() now returns sample_weights alongside y and groups.
  - Weights encode rater consensus strength derived from Bad (score):
      score 0 or 4  → weight 1.0   (unanimous — fully trust the label)
      score 1 or 3  → weight 0.75  (one dissenter — slight uncertainty)
      score 2       → weight 0.5   (2 vs 2 split — maximum ambiguity; half-weight)
  - These weights are passed to XGBClassifier.fit(sample_weight=...) in train.py,
    feature_ablation.py, and rater_ablation.py. Callers that ignore the new return
    value are unaffected — the signature change is backward-compatible via the
    existing positional unpacking (spw is still the 5th element).
"""

import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


# Columns to exclude from the feature matrix
NON_FEATURE_COLS = [
    'Session', 'Task', 'Channel labels',
    'Bad (site 2)', 'Bad (site 3)', 'Bad (site 4a)', 'Bad (site 4b)',
    'Bad (score)',
    'Impedance (start)', 'Impedance (end)',
    'subject_id', 'visit', 'site',
]

# ── Disagreement weight map ───────────────────────────────────────────────────
# Derived from Krippendorff α analysis: α=0.211, D_o=0.068, 493 channels at
# score=2 (2 vs 2 split). Weights down-weight boundary cases proportionally
# to the number of dissenting raters.
#
# Rationale:
#   score=0 or 4 → all 4 raters agree    → full weight (1.0)
#   score=1 or 3 → 1 rater dissents      → 75% weight
#   score=2      → 2 vs 2 split          → 50% weight (maximum ambiguity)
#
# These are intentionally conservative. The goal is not to discard boundary
# cases but to reduce their gradient contribution during training, letting
# unanimous examples dominate parameter updates.
DISAGREEMENT_WEIGHTS = {0: 1.0, 1: 0.75, 2: 0.5, 3: 0.75, 4: 1.0}


def load_bcr_data(path: str) -> pd.DataFrame:
    """Load BCR CSV and parse Session column into subject_id, visit, site."""
    df = pd.read_csv(path)

    def parse_session(s: str):
        parts = str(s).split('-')
        visit      = int(parts[0])
        subject_id = int(parts[1])
        site       = int(parts[2])
        return subject_id, site, visit

    parsed = df['Session'].apply(
        lambda x: pd.Series(parse_session(x),
                             index=['subject_id', 'site', 'visit'])
    )
    df = pd.concat([parsed, df], axis=1)

    print(f"Loaded:           {df.shape}")
    print(f"Unique subjects:  {df['subject_id'].nunique()}")    # expect 43
    print(f"Unique visits:    {sorted(df['visit'].unique())}")  # expect [1,2,3,4]
    print(f"Unique sites:     {df['site'].nunique()} (constant — will be dropped)")
    return df


def add_missingness_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add impedance_missing flag BEFORE any imputation.

    Visit 3 has 78.3% missing impedance vs 0% for visits 1, 2, 4.
    This is structured missingness (different measurement protocol),
    not random — it carries predictive signal and must be preserved
    as an explicit binary feature.
    """
    df = df.copy()
    df['impedance_missing'] = df['Impedance (start)'].isnull().astype(int)

    visit_miss = df.groupby('visit')['impedance_missing'].mean().round(3)
    print("\nImpedance missing by visit:\n", visit_miss)

    return df


def impute_and_encode_channels(df: pd.DataFrame,
                                feature_cols: list):
    """
    Median-impute float features, ordinal-encode channel labels by bad rate.
    impedance_missing stays binary — never impute or log-transform it.
    inf/-inf values are replaced with NaN before imputation.
    """
    df = df.copy()

    # Ordinal-encode channel labels: rank by descending bad rate
    # T7, T8 expected near the top from EDA
    channel_order = (
        df.groupby('Channel labels')['Bad (score)']
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    channel_map = {ch: i for i, ch in enumerate(channel_order)}
    df['channel_label_enc'] = df['Channel labels'].map(channel_map)
    print(f"\nTop-5 channels by bad rate: {channel_order[:5]}")

    # Impute float features — exclude impedance_missing and non-numeric
    impute_cols = [
        c for c in feature_cols
        if c != 'impedance_missing'
        and df[c].dtype != object
    ]

    # Diagnose inf values before replacing
    inf_cols = [c for c in impute_cols if np.isinf(df[c]).any()]
    if inf_cols:
        print(f"\nColumns with inf values: {len(inf_cols)} — replacing with NaN")
        print(f"  First 5: {inf_cols[:5]}")

    # Replace inf/-inf with NaN so median imputer can handle them
    df[impute_cols] = df[impute_cols].replace([np.inf, -np.inf], np.nan)

    imp = SimpleImputer(strategy='median')
    df[impute_cols] = imp.fit_transform(df[impute_cols])

    all_feature_cols = feature_cols + ['channel_label_enc']
    remaining_nans = df[all_feature_cols].isnull().sum().sum()
    assert remaining_nans == 0, f"Still {remaining_nans} NaNs after imputation!"
    print("✓ Zero NaN/inf values confirmed after imputation")

    return df, imp


def compute_disagreement_weights(score: pd.Series) -> np.ndarray:
    """
    Map Bad (score) → per-sample weight encoding rater consensus strength.

    Parameters
    ----------
    score : pd.Series
        The raw Bad (score) column (integer 0–4).

    Returns
    -------
    np.ndarray, shape (n,), dtype float32
        Weight per sample. Unanimous cases weight 1.0; 2 vs 2 splits
        weight 0.5. See DISAGREEMENT_WEIGHTS for the full map.

    Notes
    -----
    These weights are multiplied with scale_pos_weight inside XGBoost via
    the sample_weight argument to fit(). XGBoost combines both by scaling
    the gradient contribution: effective_weight = sample_weight * (spw if y==1 else 1).
    """
    weights = score.map(DISAGREEMENT_WEIGHTS).values.astype(np.float32)

    # Sanity: no NaN or zero weights
    assert not np.isnan(weights).any(), "NaN in disagreement weights — check score range"
    assert (weights > 0).all(), "Zero weight found — all samples must have positive weight"

    # Print distribution for the training log
    unique_scores, counts = np.unique(score.values, return_counts=True)
    print("\nDisagreement weight distribution:")
    print(f"  {'Score':>6}  {'Weight':>6}  {'Count':>7}  {'% total':>8}")
    for s, c in zip(unique_scores, counts):
        w = DISAGREEMENT_WEIGHTS[int(s)]
        print(f"  {s:>6}  {w:>6.2f}  {c:>7}  {c/len(score)*100:>7.2f}%")
    mean_w = weights.mean()
    print(f"  Mean weight: {mean_w:.4f}  "
          f"(effective sample reduction: {(1 - mean_w)*100:.1f}%)")

    return weights


def build_targets_and_groups(
    df: pd.DataFrame,
    bad_threshold: int = 2,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Return y (binary), groups (subject_id), scale_pos_weight, sample_weights.

    Parameters
    ----------
    df            : DataFrame with Bad (score) and subject_id columns.
    bad_threshold : Minimum score to label a channel as bad. Default 2.

    Returns
    -------
    y                : np.ndarray (n,)   binary labels
    groups           : np.ndarray (n,)   subject_id for GroupKFold
    scale_pos_weight : float             n_good / n_bad for XGBoost
    sample_weights   : np.ndarray (n,)   per-sample disagreement weight (float32)

    Notes
    -----
    The sample_weights are derived from the raw Bad (score) using
    DISAGREEMENT_WEIGHTS. They are independent of bad_threshold —
    a channel with score=2 is always maximally ambiguous regardless of
    whether the threshold is 1 or 2.
    """
    y = (df['Bad (score)'] >= bad_threshold).astype(int)

    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    scale_pos_weight = n_neg / n_pos

    print(f"\nLabel threshold: Bad (score) >= {bad_threshold}")
    print(f"Class distribution — good: {n_neg}, bad: {n_pos}")
    print(f"Bad rate: {n_pos / (n_neg + n_pos):.4f}")
    print(f"scale_pos_weight = {scale_pos_weight:.2f}")

    groups = df['subject_id'].values
    assert len(np.unique(groups)) == 43, \
        f"Expected 43 subjects, got {len(np.unique(groups))}"

    # ── NEW: disagreement weights ─────────────────────────────────────────────
    sample_weights = compute_disagreement_weights(df['Bad (score)'])

    return y.values, groups, scale_pos_weight, sample_weights


def build_feature_matrix(csv_path: str,
                          save_cols_to: str = None,
                          bad_threshold: int = 2):
    """
    Full BCR feature matrix pipeline.

    Parameters
    ----------
    csv_path      : path to Bad_channels_for_ML.csv
    save_cols_to  : optional path to save feature column names as JSON
    bad_threshold : minimum Bad (score) to label a channel as bad.
                    2 = score>=2 (3.9% bad, scale_pos_weight~24.8)  ← default
                    1 = score>=1 (12.8% bad, scale_pos_weight~6.8)  ← sensitivity experiment

    Returns
    -------
    X                — np.ndarray (18900, n_features)
    y                — np.ndarray (18900,) binary labels
    groups           — np.ndarray (18900,) subject_id for GroupKFold
    feature_cols     — list of feature column names
    scale_pos_weight — float
    sample_weights   — np.ndarray (18900,) disagreement-based sample weights  ← NEW
    """
    df = load_bcr_data(csv_path)

    # Step 1: missingness flag BEFORE imputation
    df = add_missingness_flags(df)

    # Step 2: identify numeric feature columns
    # site is excluded — constant (only 1 site), zero information
    feature_cols = [
        c for c in df.columns
        if c not in NON_FEATURE_COLS
        and c != 'site'
        and df[c].dtype != object
    ]

    # Step 3: impute + encode channels
    df, imputer = impute_and_encode_channels(df, feature_cols)
    feature_cols = feature_cols + ['channel_label_enc']

    # Step 4: targets + groups + sample weights
    y, groups, scale_pos_weight, sample_weights = build_targets_and_groups(
        df, bad_threshold
    )

    X = df[feature_cols].values
    assert X.shape[0] == 18900, f"Expected 18900 rows, got {X.shape[0]}"
    assert not np.isnan(X).any(), "NaN found in feature matrix!"
    assert not np.isinf(X).any(), "Inf found in feature matrix!"

    print(f"\n✓ Feature matrix: {X.shape}")
    print(f"✓ Unique subjects: {len(np.unique(groups))}")
    print(f"✓ scale_pos_weight: {scale_pos_weight:.2f}")
    print(f"✓ First 5 features: {feature_cols[:5]}")

    if save_cols_to:
        pathlib.Path(save_cols_to).parent.mkdir(parents=True, exist_ok=True)
        with open(save_cols_to, 'w') as f:
            json.dump(feature_cols, f, indent=2)
        print(f"✓ Feature cols saved → {save_cols_to}")

    return X, y, groups, feature_cols, scale_pos_weight, sample_weights


if __name__ == '__main__':
    X, y, groups, cols, spw, sw = build_feature_matrix(
        'data/raw/Bad_channels_for_ML.csv',
        save_cols_to='configs/feature_cols.json'
    )
