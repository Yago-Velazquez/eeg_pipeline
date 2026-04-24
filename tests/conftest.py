"""
tests/conftest.py

Shared fixtures for the BCR test suite. All fixtures are session-scoped
so the real 18,900-row CSV is only loaded once across all tests.

CSV location
------------
The real CSV must be reachable at one of these paths (checked in order):
    1. $BCR_TEST_CSV            (env var override)
    2. data/raw/Bad_channels_for_ML.csv  (project convention)
    3. /mnt/user-data/uploads/Bad_channels_for_ML.csv  (sandbox fallback)

If none exist, real-data tests are skipped with a clear message.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest


CANDIDATE_PATHS = [
    os.environ.get("BCR_TEST_CSV"),
    "data/raw/Bad_channels_for_ML.csv",
    "/mnt/user-data/uploads/Bad_channels_for_ML.csv",
]


def _find_csv() -> str | None:
    for p in CANDIDATE_PATHS:
        if p and Path(p).exists():
            return p
    return None


@pytest.fixture(scope="session")
def csv_path() -> str:
    p = _find_csv()
    if p is None:
        pytest.skip(
            "Real BCR CSV not found. Set BCR_TEST_CSV env var or place the "
            "file at data/raw/Bad_channels_for_ML.csv"
        )
    return p


@pytest.fixture(scope="session")
def raw_df(csv_path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


@pytest.fixture(scope="session")
def feature_matrix(csv_path):
    """Built once, reused across tests that only need the feature matrix.

    Uses hard_threshold strategy because it is the cheapest (no EM) and
    the matrix itself is independent of labelling.
    """
    from bad_channel_rejection.dataset import build_feature_matrix
    return build_feature_matrix(csv_path, label_strategy="hard_threshold")


@pytest.fixture(scope="session")
def preprocessed_X(feature_matrix):
    from bad_channel_rejection.features import FeaturePreprocessor
    prep = FeaturePreprocessor()
    X = prep.fit_transform(
        pd.DataFrame(
            feature_matrix["X"], columns=feature_matrix["feature_cols"]
        )
    )
    return {"X": X, "preprocessor": prep}
