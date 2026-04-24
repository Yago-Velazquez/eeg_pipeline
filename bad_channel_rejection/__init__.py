"""
bad_channel_rejection — BCR pipeline package.

Public API
----------
  build_feature_matrix, build_label_artifacts — data & label construction
  FeaturePreprocessor                         — feature preprocessing
  create_model, BadChannelDetector            — model API
  setup_logging                               — logging config
  build_run_tag                               — OOF/model filename tag
"""

from .dataset import build_feature_matrix
from .features import FeaturePreprocessor
from .label_quality import (
    LabelArtifacts,
    build_label_artifacts,
    compute_dawid_skene_labels,
    compute_entropy_weights,
    compute_hard_threshold_labels,
    fit_dawid_skene,
)
from .logging_config import setup_logging
from .model import BadChannelDetector
from .models import SUPPORTED_MODELS, create_model


def build_run_tag(
    label_strategy: str,
    model_name: str,
    use_engineered_features: bool = False,
    use_impedance_interactions: bool = False,
) -> str:
    """Build the filename tag used for OOF arrays and saved models.

    Exactly one of ``use_engineered_features`` and
    ``use_impedance_interactions`` may be True.
    """
    if use_engineered_features and use_impedance_interactions:
        raise ValueError(
            "--use-engineered-features and --use-impedance-interactions "
            "are mutually exclusive"
        )
    if use_impedance_interactions:
        suffix = "_impedance_ix"
    elif use_engineered_features:
        suffix = "_fe"
    else:
        suffix = ""
    return f"{label_strategy}_{model_name}{suffix}"


__all__ = [
    "build_feature_matrix",
    "build_label_artifacts",
    "build_run_tag",
    "LabelArtifacts",
    "FeaturePreprocessor",
    "fit_dawid_skene",
    "compute_dawid_skene_labels",
    "compute_entropy_weights",
    "compute_hard_threshold_labels",
    "BadChannelDetector",
    "create_model",
    "SUPPORTED_MODELS",
    "setup_logging",
]
