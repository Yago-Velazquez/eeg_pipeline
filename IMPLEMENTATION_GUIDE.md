# BCR Pipeline — Implementation & Testing Guide

This guide walks you through installing, testing, and running the BCR
pipeline end-to-end. The pipeline has been rewritten around three principles:
principled label quality estimation (Dawid-Skene), multi-model support
(XGBoost / LightGBM / CatBoost), and a staged ablation methodology.

---

## 1. Directory layout

```
your-repo/
├── bad_channel_rejection/         # main package (11 files)
│   ├── __init__.py
│   ├── logging_config.py          # centralised logging
│   ├── label_quality.py           # Dawid-Skene, entropy weights, hard threshold
│   ├── dataset.py                 # CSV loading + label dispatch
│   ├── features.py                # FeaturePreprocessor (scaling, dedup)
│   ├── models.py                  # XGBoost/LightGBM/CatBoost factory
│   ├── model.py                   # BadChannelDetector (production API)
│   ├── train.py                   # CV training entry point
│   ├── evaluate.py                # OOF evaluation
│   ├── rater_ablation.py          # two-stage ablation
│   ├── feature_ablation.py        # feature-group ablation
│   ├── site_generalisation.py     # leave-one-visit-out
│   ├── shap_analysis.py           # SHAP interpretability
│   └── interpolation.py           # spherical spline / zero-out
├── scripts/                       # CLI helpers (6 files)
│   ├── fn_analysis.py
│   ├── threshold_sweep.py
│   ├── update_threshold.py
│   ├── verify_threshold.py
│   ├── visit_badrate.py
│   └── interpolation_comparison.py
├── tests/                         # pytest suite
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_dataset.py
│   ├── test_features.py
│   ├── test_label_quality.py
│   └── test_models.py
├── configs/
│   └── pipeline_config.yaml
└── data/
    └── raw/
        └── Bad_channels_for_ML.csv
```

---

## 2. Installation

```bash
pip install xgboost lightgbm catboost scikit-learn pandas numpy \
            mne shap wandb python-dotenv pyyaml pytest matplotlib seaborn
```

No `dawid-skene` package needed — Dawid-Skene is implemented directly in
`label_quality.py`.

---

## 3. Run the test suite first

Before any training, verify everything compiles and the real-data paths work:

```bash
# from repo root
export BCR_TEST_CSV=data/raw/Bad_channels_for_ML.csv
pytest tests/ -v
```

Expected: ~35 tests pass, total runtime ~2–3 minutes. The Dawid-Skene
fit is the expensive part (converges in ~60 iterations).

If you see `ImportError`, check that `bad_channel_rejection` is on your
PYTHONPATH (running from repo root is usually enough).

### Common test failures and what they mean

| Failure | Meaning |
|---------|---------|
| `site 4a sensitivity not > site 3` | DS is producing suspicious confusion matrices — inspect `info["confusion_matrices"]` |
| `bad rate outside 3-5%` | Dataset may have changed; verify CSV is the same file |
| `score=2 weight != 0` (entropy) | `compute_entropy_weights` regression — check the weight map |

---

## 4. Pipeline execution order

The pipeline has three phases. Run them in order. Each phase produces
artifacts in `results/` that later phases consume.

### Phase 1 — Ablation (decides label strategy and model)

```bash
python -m bad_channel_rejection.rater_ablation
```

**Runtime:** ~20–40 minutes on CPU.

**What it does:**
- Stage 1: runs 4 label strategies (hard_threshold, entropy_weights,
  dawid_skene, dawid_skene_soft) with XGBoost, picks the winner by
  mean OOF AUPRC.
- Stage 2: runs 3 models (XGBoost, LightGBM, CatBoost) with the winning
  label strategy, picks the winning model.

**Outputs:**
- `results/ablation_results.json` — raw per-fold numbers for all 7 runs
- `results/ablation_report.md` — human-readable summary with winner

**What to check:** The report's final lines tell you the winning combo:

```
Recommended production config
- Label strategy: dawid_skene
- Model backend : lightgbm
```

### Phase 2 — Production training

Using the winners from Phase 1:

```bash
python -m bad_channel_rejection.train \
    --label-strategy dawid_skene \
    --model lightgbm
```

**Runtime:** ~5–10 minutes.

**Outputs:**
- `results/oof_y_true_dawid_skene_lightgbm.npy`
- `results/oof_y_prob_dawid_skene_lightgbm.npy`
- `results/bcr_model_dawid_skene_lightgbm.pkl` — final model
- `results/bcr_model_dawid_skene_lightgbm_meta.json` — feature names, threshold, etc.
- W&B run (offline by default)

**Critical check:** look for the subject leakage assertion in the log. If
it fires, stop — something is wrong with your `groups` column.

### Phase 3 — Evaluation and threshold selection

```bash
# 1. Compute pooled OOF metrics and initial threshold
python -m bad_channel_rejection.evaluate \
    --label-strategy dawid_skene --model lightgbm

# 2. Full threshold sweep 0.20-0.80
python scripts/threshold_sweep.py \
    --label-strategy dawid_skene --model lightgbm

# 3. Three-criteria comparison + write to pipeline_config.yaml
python scripts/update_threshold.py \
    --label-strategy dawid_skene --model lightgbm

# 4. Verify the threshold written to config is the one you want
python scripts/verify_threshold.py \
    --label-strategy dawid_skene --model lightgbm
```

After Phase 3, `configs/pipeline_config.yaml` has `bcr.decision_threshold`
set. Edit manually if you want a different criterion (argmax_f1 vs
argmax_2rp vs intuition).

### Phase 4 — Diagnostics (any order, all optional)

These analyses do not affect the model but are essential for a methods
section and for understanding where the model struggles:

```bash
# False negative breakdown by channel and visit
python scripts/fn_analysis.py --label-strategy dawid_skene --model lightgbm

# Visit-level bad rate (standalone EDA)
python scripts/visit_badrate.py

# Leave-one-visit-out generalisation
python -m bad_channel_rejection.site_generalisation \
    --label-strategy dawid_skene --model lightgbm

# Feature-group ablation
python -m bad_channel_rejection.feature_ablation \
    --label-strategy dawid_skene --model lightgbm

# SHAP importance on the trained model
python -m bad_channel_rejection.shap_analysis \
    --label-strategy dawid_skene --model lightgbm

# Interpolation strategy comparison (independent of BCR model)
python scripts/interpolation_comparison.py
```

---

## 5. What to check at each phase

### After Phase 1 (ablation)

- **Stage 1 delta signs.** If dawid_skene does not beat hard_threshold
  by more than ~0.005 AUPRC, the Dawid-Skene complexity may not be
  justified. Report it as a methods finding either way.
- **Stage 2 delta signs.** LightGBM typically edges out XGBoost by
  1–3% AUPRC on this type of tabular data. CatBoost varies more.
- **Per-rater sensitivity from DS.** The log prints this:
  ```
  Bad (site 3): sensitivity=0.075  specificity=0.998
  ```
  Site 3 sensitivity near 0.08 is the expected pathology. If it is
  above 0.3, the data has changed.

### After Phase 2 (training)

- Mean OOF AUPRC should match what Phase 1 reported for the same
  combo (within noise).
- `best_iteration` averaged across folds should be 150–400. If it
  is 500 (no early stopping triggered), consider raising `n_estimators`.

### After Phase 3 (threshold)

- Recall should be > 0.55 for a useful BCR model. If it is lower,
  your threshold is too aggressive.
- Check the F1 is reasonable (> 0.4 is what 0.50 AUPRC typically
  produces at the optimal threshold).

### After Phase 4 (diagnostics)

- `fn_analysis.py`: channels with FN rate > 0.5 (T7, T8 are the
  usual suspects) need feature-engineering attention.
- `site_generalisation.py`: "generalisation gap" > 0.1 means the
  model leaks visit-specific signal. Likely culprit: `impedance_missing`.
- `shap_analysis.py`: `impedance_missing` should rank in the top 10.
  If it does not, something broke.

---

## 6. Quick answers to likely questions

**Why a dict return from `build_feature_matrix` instead of a tuple?**
The old 5-tuple became a 6-tuple when weights were added, then would have
become a 7-tuple with soft targets. A dict avoids API creep — new fields
can be added without breaking callers. All scripts updated accordingly.

**Why not use scikit-learn's `GridSearchCV` for the ablation?**
Because a grid treats label strategy and model choice as equivalent
axes. Sequential ablation isolates the effect of each, which is
what a methods section requires. See `rater_ablation.py` docstring.

**Why implement Dawid-Skene from scratch?**
The `dawid-skene` PyPI package has not been updated since 2020 and has
no test suite. Sixty lines in `label_quality.py` give us full control
and verified convergence on this dataset.

**How do I swap in a new model backend?**
Add a subclass of `BaseBCRModel` in `models.py`, register it in the
`create_model` factory, and add it to `SUPPORTED_MODELS` and
`MODEL_EXT`. Every training/evaluation script picks it up automatically.

**How do I run without Weights & Biases?**
W&B is in offline mode by default (`WANDB_MODE=offline` is set in each
script). Runs are written to `./wandb/` but nothing is uploaded. To
fully disable, `export WANDB_MODE=disabled` before running.

**What breaks if the CSV changes?**
The assertions in `dataset.py` enforce 18,900 rows and 43 subjects.
Change the CSV → these assertions fire with a clear message. Update
the assertions or the CSV to match.

---

## 7. Production deployment

To use the trained model in another pipeline:

```python
from bad_channel_rejection import BadChannelDetector
import pandas as pd

detector = BadChannelDetector(
    model_name="lightgbm",
    model_path="results/bcr_model_dawid_skene_lightgbm.pkl",
)

# X must be preprocessed the same way training did
# (use FeaturePreprocessor fitted on training data)
predictions = detector.predict(X)
probabilities = detector.predict_proba(X)

# For a single session with named channels:
bad_map = detector.predict_channels(X_session, channel_names_list)
# {"Fp1": False, "T7": True, ...}
```

The detector auto-loads the sidecar `_meta.json`, which contains the
threshold, feature names, and model name — so you do not have to pass
them at inference time.
