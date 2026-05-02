"""
bad_channel_rejection/detect_session.py

Production wrapper — single entrypoint for running the BCR pipeline on a new
EEG session.

Given a session DataFrame (one row per channel, same schema as the training
``Bad_channels_for_ML.csv``) the wrapper:

    1. Loads the Stage 3 winning model (`results/best_model.<ext>`).
    2. Loads the calibrated triage thresholds (`results/triage_thresholds.json`).
    3. Recreates the training-time feature pipeline so probabilities are
       consistent with what the model saw at fit time. The training CSV is
       loaded once and fitted FeaturePreprocessor stats are reused for the
       session rows (transductive — no leakage, just consistent scaling).
    4. Predicts P(bad) per channel + triage tier (auto_good / review / auto_bad).
    5. Builds a review-queue DataFrame sorted by descending P(bad).
    6. (optional) Auto-interpolates the auto-bad channels via MNE spherical
       splines if a raw EEG array + channel names + sample-rate are provided.
    7. Returns a structured result dict and (if `output_dir` is given) writes
       the review queue CSV.

Public API
----------
    detect_and_repair_session(session_df, ...)   # in-memory DataFrame
    detect_and_repair_session_from_csv(path, ...) # convenience wrapper

CLI smoke test
--------------
Pretend one subject from the training CSV is a "new session" — the wrapper
should reproduce the OOF probabilities for that subject's rows::

    python -m bad_channel_rejection.detect_session smoke-test --subject 102
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .dataset import (
    add_missingness_flags,
    impute_and_encode_channels,
    load_bcr_data,
)
from .features import FeaturePreprocessor
from .interpolation import interpolate_bad_channels
from .logging_config import setup_logging
from .models import MODEL_EXT, create_model
from .triage import (
    TIER_AUTO_BAD,
    TIER_AUTO_GOOD,
    TIER_REVIEW,
    load_thresholds,
    predict_and_triage,
    review_queue,
)

logger = setup_logging(__name__)

DEFAULT_TRAINING_CSV   = Path("data/raw/Bad_channels_for_ML.csv")
DEFAULT_CONFIG_PATH    = Path("results/best_config.json")
DEFAULT_THRESHOLDS     = Path("results/triage_thresholds.json")
DEFAULT_RESULTS_DIR    = Path("results")


# ── Internal helpers ──────────────────────────────────────────────────────────


def _load_winning_model(config_path: Path, model_path: Path | None):
    """Load the Stage 3 winning model based on `best_config.json`.

    Returns (model, label_strategy, model_name, overrides).
    """
    if not config_path.exists():
        raise FileNotFoundError(
            f"{config_path} not found. Run Stage 3 HPO first:\n"
            "  python -m bad_channel_rejection.ablation_stage3_hpo "
            "--winning-strategy <X> --winning-model <Y> --count 50"
        )
    cfg = json.loads(config_path.read_text())
    label_strategy = cfg["winning_strategy"]
    model_name     = cfg["winning_model"]
    overrides      = dict(cfg.get("best_config", {}))

    if model_path is None:
        ext = MODEL_EXT[model_name]
        model_path = DEFAULT_RESULTS_DIR / f"best_model.{ext}"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Re-run Stage 3 HPO."
        )

    # `scale_pos_weight` is a fit-time hyperparameter; for inference-only
    # loading we pass any positive value — the loaded weights override it.
    model = create_model(model_name, scale_pos_weight=1.0, **overrides)
    model.load(model_path)
    logger.info(
        f"Loaded {model_name} model from {model_path} "
        f"(label_strategy={label_strategy})"
    )
    return model, label_strategy, model_name, overrides


def _prepare_session_features(
    session_df: pd.DataFrame,
    training_csv: Path,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Run the training-time dataset preprocessing on session_df.

    Combines training rows + session rows → load_bcr_data → add_missingness_flags
    → impute_and_encode_channels (so channel encoding matches training) →
    FeaturePreprocessor.fit on training rows only → transform session rows.

    Returns (X_session, feature_cols, channel_labels_session).
    """
    if "Session" not in session_df.columns or "Channel labels" not in session_df.columns:
        raise ValueError(
            "session_df must include the columns 'Session' and 'Channel labels' "
            "(same schema as Bad_channels_for_ML.csv)."
        )

    # Load + parse the training CSV through the same pipeline used in dataset.py
    train_df = load_bcr_data(str(training_csv))

    # Reset the session index BEFORE doing axis=1 concat: if session_df was
    # filtered from a larger DataFrame (e.g. the smoke test), it keeps the
    # original row positions and the concat below would produce a misaligned
    # union (parsed has indices 1000–1377, session_df.reset has 0–377 →
    # 756 rows of NaN-padded garbage).
    session_df = session_df.reset_index(drop=True)

    # Parse the session's "Session" column the same way load_bcr_data parses it.
    def _parse(s: str):
        parts = str(s).split("-")
        return int(parts[1]), int(parts[2]), int(parts[0])

    parsed = session_df["Session"].apply(
        lambda x: pd.Series(_parse(x), index=["subject_id", "site", "visit"])
    )
    session_parsed = pd.concat([parsed, session_df], axis=1)
    logger.info(
        f"Session: n_rows={len(session_parsed)}  "
        f"unique_channels={session_parsed['Channel labels'].nunique()}"
    )

    # Concatenate so add_missingness_flags + impute_and_encode_channels see the
    # combined distribution. Channel encoding is fit on combined data — for a
    # full session with all expected channels this matches the training mapping.
    n_train = len(train_df)
    combined = pd.concat([train_df, session_parsed], ignore_index=True)

    combined = add_missingness_flags(combined)

    from .dataset import NON_FEATURE_COLS
    feature_cols = [
        c for c in combined.columns
        if c not in NON_FEATURE_COLS
        and c != "site"
        and combined[c].dtype != object
    ]
    combined, _ = impute_and_encode_channels(combined, feature_cols)
    feature_cols = feature_cols + ["channel_label_enc"]

    # Split back into training / session rows
    train_X_df   = combined.iloc[:n_train][feature_cols]
    session_X_df = combined.iloc[n_train:][feature_cols]

    # Fit preprocessor on training rows only — preserves training-time scales
    prep = FeaturePreprocessor()
    prep.fit_transform(train_X_df)
    X_session = prep.transform(session_X_df)

    channel_labels_session = session_parsed["Channel labels"].astype(str).to_numpy()

    return X_session, feature_cols, channel_labels_session


# ── Public API ────────────────────────────────────────────────────────────────


def detect_and_repair_session(
    session_df: pd.DataFrame,
    *,
    eeg: np.ndarray | None = None,
    ch_names: list[str] | None = None,
    sfreq: float = 256.0,
    training_csv: Path = DEFAULT_TRAINING_CSV,
    config_path:  Path = DEFAULT_CONFIG_PATH,
    thresholds_path: Path = DEFAULT_THRESHOLDS,
    model_path:   Path | None = None,
    output_dir:   Path | None = None,
    session_id:   str | None = None,
    interpolation_method: str = "spherical",
    montage_name: str = "standard_1005",
) -> dict[str, Any]:
    """Run the production BCR pipeline on a new EEG session.

    Parameters
    ----------
    session_df : pd.DataFrame
        One row per channel for ONE session. Must contain at least the columns
        "Session", "Channel labels", "Impedance (start)", "Impedance (end)",
        plus all the spectral / decomposition / spatial features in the same
        format as Bad_channels_for_ML.csv.
    eeg : np.ndarray, optional
        Raw EEG (n_channels, n_samples). If given, auto-bad channels are
        interpolated and the repaired array is returned.
    ch_names : list[str], optional
        Required if `eeg` is given. Standard 10-05 channel names.
    sfreq : float
        Sampling rate, only used by the interpolator. Default 256 Hz.
    training_csv : Path
        Path to Bad_channels_for_ML.csv, used for FeaturePreprocessor stats.
    config_path : Path
        Path to results/best_config.json from Stage 3.
    thresholds_path : Path
        Path to results/triage_thresholds.json from triage calibration.
    model_path : Path, optional
        Override the default best_model.<ext> path.
    output_dir : Path, optional
        If given, the review-queue CSV is written here.
    session_id : str, optional
        Used to name the CSV; defaults to "session".
    interpolation_method : str
        "spherical" (default, MNE) or "zero".
    montage_name : str
        Standard montage for spherical interpolation.

    Returns
    -------
    dict with keys::

        probs           : np.ndarray (n_channels,) — P(bad) per channel
        tiers           : np.ndarray[str] — auto_good / review / auto_bad
        auto_bad_idx    : list[int]
        review_idx      : list[int]
        auto_good_idx   : list[int]
        review_queue    : pd.DataFrame sorted desc by P(bad)
        eeg_repaired    : np.ndarray | None  (only if `eeg` was provided)
        queue_csv_path  : Path | None  (only if `output_dir` was provided)
        summary         : counts per tier + total
        config          : (label_strategy, model_name) used
    """
    if eeg is not None and ch_names is None:
        raise ValueError("ch_names is required when eeg is provided.")
    if ch_names is not None and eeg is not None and len(ch_names) != eeg.shape[0]:
        raise ValueError(
            f"ch_names length ({len(ch_names)}) must equal eeg rows "
            f"({eeg.shape[0]})."
        )

    # ── 1. Load model + thresholds ───────────────────────────────────────────
    model, label_strategy, model_name, _ = _load_winning_model(
        config_path, model_path
    )
    thr_low, thr_high = load_thresholds(thresholds_path)

    # ── 2. Build session features through the training-time pipeline ──────────
    X_session, _, channel_labels_session = _prepare_session_features(
        session_df, training_csv
    )

    # ── 3. Predict + triage ───────────────────────────────────────────────────
    probs, tiers = predict_and_triage(model, X_session, thr_low, thr_high)

    # ── 4. Build review queue (sorted desc) ───────────────────────────────────
    queue_df = review_queue(
        probs,
        thr_low=thr_low,
        thr_high=thr_high,
        channel_labels=channel_labels_session,
    )

    # ── 5. Optionally write queue CSV ─────────────────────────────────────────
    queue_csv_path: Path | None = None
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        sid = session_id or "session"
        queue_csv_path = output_dir / f"{sid}_review_queue.csv"
        queue_df.to_csv(queue_csv_path, index=False)
        logger.info(f"Review queue CSV -> {queue_csv_path}  (n={len(queue_df)})")

    # ── 6. Optionally interpolate auto-bad channels ───────────────────────────
    auto_bad_idx  = np.where(tiers == TIER_AUTO_BAD)[0].tolist()
    review_idx    = np.where(tiers == TIER_REVIEW)[0].tolist()
    auto_good_idx = np.where(tiers == TIER_AUTO_GOOD)[0].tolist()

    eeg_repaired = None
    if eeg is not None and ch_names is not None and auto_bad_idx:
        eeg_repaired = interpolate_bad_channels(
            eeg, auto_bad_idx, ch_names,
            sfreq=sfreq,
            montage_name=montage_name,
            method=interpolation_method,
        )
        logger.info(
            f"Auto-interpolated {len(auto_bad_idx)} channels: "
            f"{[ch_names[i] for i in auto_bad_idx]}"
        )
    elif eeg is not None and not auto_bad_idx:
        # No auto-bad channels — return the input untouched
        eeg_repaired = eeg.copy()

    summary = {
        "n_total":     int(len(probs)),
        "n_auto_bad":  int(len(auto_bad_idx)),
        "n_review":    int(len(review_idx)),
        "n_auto_good": int(len(auto_good_idx)),
        "thr_low":     float(thr_low),
        "thr_high":    float(thr_high),
    }
    logger.info(
        f"Triage summary: auto_good={summary['n_auto_good']} | "
        f"review={summary['n_review']} | auto_bad={summary['n_auto_bad']}"
    )

    return {
        "probs":          probs,
        "tiers":          tiers,
        "auto_bad_idx":   auto_bad_idx,
        "review_idx":     review_idx,
        "auto_good_idx":  auto_good_idx,
        "review_queue":   queue_df,
        "eeg_repaired":   eeg_repaired,
        "queue_csv_path": queue_csv_path,
        "summary":        summary,
        "config":         {
            "label_strategy": label_strategy,
            "model_name":     model_name,
        },
    }


def detect_and_repair_session_from_csv(
    session_csv: Path,
    **kwargs,
) -> dict[str, Any]:
    """Convenience wrapper: read session DataFrame from a CSV file."""
    df = pd.read_csv(session_csv)
    sid = kwargs.pop("session_id", None) or Path(session_csv).stem
    return detect_and_repair_session(df, session_id=sid, **kwargs)


# ── CLI smoke test ────────────────────────────────────────────────────────────


def _smoke_test(subject_id: int, training_csv: Path) -> None:
    """Pretend one subject is a 'new session', verify the wrapper runs.

    The pipeline should produce sensible probabilities for the held-out
    subject. We can't validate exact equality with OOF predictions because
    the production model was retrained on ALL data (including this subject),
    so probabilities will differ from the OOF run — but they should still
    be reasonable (high AUROC on the subject's labels).
    """
    full = pd.read_csv(training_csv)
    full["__subject"] = full["Session"].apply(lambda s: int(str(s).split("-")[1]))
    session_df = full[full["__subject"] == subject_id].drop(columns="__subject")
    if len(session_df) == 0:
        raise SystemExit(
            f"Subject {subject_id} not found. Available IDs: "
            f"{sorted(full['__subject'].unique())[:10]}…"
        )

    logger.info(f"Smoke test: treating subject {subject_id} as a new session "
                f"(n_rows={len(session_df)})")

    # Channel names from the session df
    ch_names = session_df["Channel labels"].astype(str).tolist()

    result = detect_and_repair_session(
        session_df,
        ch_names=ch_names,
        output_dir=DEFAULT_RESULTS_DIR / "sessions",
        session_id=f"smoke_subject_{subject_id}",
    )

    # Compare against the labels that already exist in the CSV
    y_true = (session_df["Bad (score)"].values >= 2).astype(int)
    if y_true.sum() > 0:
        from sklearn.metrics import average_precision_score, roc_auc_score
        auprc = average_precision_score(y_true, result["probs"])
        auroc = roc_auc_score(y_true, result["probs"])
        logger.info(
            f"Sanity check on labels found in CSV: "
            f"AUPRC={auprc:.4f}  AUROC={auroc:.4f}  "
            f"(may be optimistic — production model was trained on this subject)"
        )

    print(json.dumps(result["summary"], indent=2))
    print(f"\nReview queue head (top 5 by P(bad)):")
    print(result["review_queue"].head().to_string(index=False))
    print(f"\nReview queue CSV: {result['queue_csv_path']}")


def main():
    parser = argparse.ArgumentParser(
        description="BCR production wrapper — run on a new EEG session."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser(
        "smoke-test",
        help="Run the wrapper on one subject from the training CSV.",
    )
    s.add_argument("--subject", type=int, required=True,
                   help="Subject ID to use as the 'new' session.")
    s.add_argument("--training-csv", type=Path, default=DEFAULT_TRAINING_CSV)

    r = sub.add_parser(
        "run",
        help="Run the wrapper on a session CSV.",
    )
    r.add_argument("--session-csv", type=Path, required=True,
                   help="Path to the session CSV (rows = channels).")
    r.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR / "sessions")
    r.add_argument("--training-csv", type=Path, default=DEFAULT_TRAINING_CSV)

    args = parser.parse_args()
    if args.cmd == "smoke-test":
        _smoke_test(args.subject, args.training_csv)
    elif args.cmd == "run":
        result = detect_and_repair_session_from_csv(
            args.session_csv,
            training_csv=args.training_csv,
            output_dir=args.output_dir,
        )
        print(json.dumps(result["summary"], indent=2))
        print(f"\nReview queue: {result['queue_csv_path']}")


if __name__ == "__main__":
    main()
