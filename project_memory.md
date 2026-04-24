# Project Memory

## 1. Project Overview

**Goal:** Demonstrate the feasibility of two complementary AI-based EEG signal cleaning components — a Bad Channel Rejector (BCR) and an EEG Denoiser — each independently developed, validated, and designed for future integration.

**Intended pipeline:** Raw EEG → BCR → Denoiser → Clean EEG

**Scope:** Independent validation only. Joint end-to-end pipeline evaluation is explicitly out of scope and deferred to future work.

**Timeline:** 8 weeks · 4–6 h/day · MacBook Air M2 (16 GB) + Google Colab

**Roadmap version:** v3 (HTML) — final, do not revise.

---

## 2. Current State

**Day 16 of 56 — complete. BCR component fully implemented. Week 3 (BCR Hardening) complete.**

- All Week 1 work complete and tagged `v0.1-week1`
- All Week 2 work complete and tagged `v0.2-week2-bcr-trained`
- **BCR component is now end-to-end complete:** classifier → threshold → site validation → interpolation
- `bad_channel_rejection/interpolation.py` implemented and tested (Day 16)
- Full test suite: **49/49 passing** (16 interpolation + 6 BCR metrics + 12 denoiser metrics + 15 BCR integration)
- Day 16 committed and pushed
- **Next:** Day 17 — Denoiser noise synthesis + EEGDenoiseNet dataset (Phase 3 begins)

---

## 3. Key Components

### Bad Channel Rejection (BCR)
- **Model:** XGBoost classifier
- **Input:** Pre-computed feature table — one row per `(session, task, channel)` triplet
- **Output:** `bad_mask`, `bad_proba`, `interpolated_eeg`, `original_eeg`
- **Target:** Binary label — `Bad (score) ≥ 2` (at least 2 of 4 raters flagged the channel)
- **Primary metric:** AUPRC (random baseline = 0.039)
- **Cross-validation:** `GroupKFold(n_splits=5)` by `subject_id` (43 unique subjects)
- **Dataset:** `data/raw/Bad_channels_for_ML.csv` — 18,900 rows, 50 sessions, 43 subjects, 4 visits, 3.9% bad rate
- **Class imbalance:** `scale_pos_weight = 24.82`
- **Feature matrix:** (18,900, 156) raw → (18,900, 138) after preprocessing
- **CV result (thresh=2):** AUPRC = 0.518 ± 0.085 · AUROC = 0.921 ± 0.013 · avg best_iter = 179
- **OOF evaluation:** AUPRC = 0.513 · AUROC = 0.913 · F1 = 0.506 · lift = 13.2×
- **Active threshold: 0.50** — confusion matrix: TP=413, FP=526, FN=319, TN=17642
- **LOVO generalisation:** mean AUPRC = 0.486 ± 0.073; gap = −0.032 vs GroupKFold → PASS

### BCR Dataset module (`bad_channel_rejection/dataset.py`)
- `load_bcr_data(path)` — reads CSV, splits `Session` string (`{visit}-{subject_id}-{site}`) into fields; site is constant (1) and dropped
- `add_missingness_flags(df)` — adds `impedance_missing = Impedance (start).isnull()` BEFORE imputation; visit 3 = 78.3% missing
- `impute_and_encode_channels(df, feature_cols)` — replaces inf/-inf with NaN, then `SimpleImputer(median)`; ordinal-encodes `Channel labels` by descending bad rate → `channel_label_enc`; **returns tuple `(df, imputer)` — always unpack as `df, _ = ...`**
- `build_targets_and_groups(df, bad_threshold=2)` — `y = (Bad (score) >= bad_threshold).astype(int)`; asserts 43 unique subjects
- `build_feature_matrix(csv_path, save_cols_to, bad_threshold=2)` — full pipeline; returns 5 values `(X_raw, y, groups, feature_cols, scale_pos_weight)`; saves `configs/feature_cols.json`

### BCR Feature Preprocessing module (`bad_channel_rejection/features.py`)
- `FeaturePreprocessor` class — stateful, mirrors sklearn API
- `fit_transform(X_df)` — fits RobustScaler + VarianceThreshold on train fold; populates `feature_names_out_`; returns np.ndarray
- `transform(X_df)` — applies fitted transforms to val/test fold; asserts no NaN/Inf
- `feature_names_out_` — list of 138 post-preprocessing column names in correct order
- `REDUNDANT_DROP_COLS` — 18 hardcoded columns to drop
- `BINARY_COLS = ['impedance_missing']` — passed through unscaled
- RobustScaler guarded with `if X_cont.shape[1] > 0`

**Confirmed column naming:** All 155 signal feature columns have a leading space (e.g. `' Standard deviation (mean)'`). `impedance_missing` has no leading space.

**Final feature count:** 138 (156 raw − 18 redundant/constant drops)

**BCR feature values are stored in volts (not µV).** `Standard deviation (mean)` mean=0.000111 V (~0.111 µV), max=0.000772 V (~0.772 µV). Relevant for any downstream synthesis using BCR features as amplitude proxies.

### BCR Training module (`bad_channel_rejection/train.py`)
- `run_cv` saves OOF predictions to `results/oof_y_true_thresh{N}.npy` and `results/oof_y_prob_thresh{N}.npy`
- XGB params: n_estimators=500, max_depth=6, lr=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric='aucpr', early_stopping=30, tree_method='hist', device='cpu'
- CLI: `python -m bad_channel_rejection.train --threshold 2`

### BCR Evaluation module (`bad_channel_rejection/evaluate.py`)
- Loads OOF predictions — no retraining; finds argmax(F1) threshold; produces PR curve + confusion matrix
- **Note:** final saved model is NOT used for scoring — avoids leakage

### BCR Site Generalisation (`bad_channel_rejection/site_generalisation.py`)
- Leave-one-visit-out (LOVO) evaluation: trains on 3 visits, tests on 4th; rotates all 4
- `FeaturePreprocessor` fitted on train fold only — no leakage
- Saves `results/site_generalisation.json`
- **Results:** V1=0.423, V2=0.565, V3=0.405, V4=0.551 · mean=0.486±0.073 · gap=−0.032 → PASS
- **Key finding:** Visit 3 degradation fully explained by 78.3% impedance missingness

### BCR Rater Ablation (`bad_channel_rejection/rater_ablation.py`)
- **Result:** ΔAUPRC = −0.0042 — site 3 does NOT degrade label quality (null finding)

### BCR SHAP Analysis (`bad_channel_rejection/shap_analysis.py`)
- **Top features:** spatial correlation (#1), channel_label_enc (#2), PCA/ICA decomposition (#4–11)
- **`impedance_missing` rank: #74/138**

### BCR Feature Ablation (`bad_channel_rejection/feature_ablation.py`)
- **Key finding:** decomposition backbone (ΔAUPRC = −0.092); impedance dispensable (+0.005)

### BCR Model API (`bad_channel_rejection/model.py`)
- `BadChannelDetector` class — fit/predict/predict_proba/save/load; latency < 10 ms
- Sidecar: `results/bcr_model_thresh2_meta.json` — threshold + 138 feature names

### BCR Interpolation module (`bad_channel_rejection/interpolation.py`) — NEW Day 16
- `spherical_spline_interpolation(eeg, bad_indices, ch_names, sfreq, montage_name)` — MNE 1.12.0 `raw.interpolate_bads()`, default montage `standard_1005`, mode=`'accurate'`; converts µV→V for MNE then back
- `zero_out_channel(eeg, bad_indices)` — fallback when no montage available; sets bad channels to 0
- `interpolate_bad_channels(...)` — router; `method='spherical'` or `method='zero'`
- **Unit tests:** 16 tests, all passing; `test_other_channels_unchanged` uses `assert_allclose(rtol=1e-10)` — float64 unit-conversion rounding through MNE pipeline requires tolerance (not exact equality)
- **Comparison script:** `scripts/interpolation_comparison.py` — both methods reduce MSE >99.99% vs. 200 µV noise injection on 10 visit-4 sessions; spline vs. zero-out comparison degenerate due to BCR features being in volts (~0.1 µV proxy amplitude, 1800× smaller than noise)
- **Results documented in:** `results/interpolation_analysis.md`, `results/interpolation_comparison.json`

### EEG Denoiser
- **Model:** 1D U-Net (primary); EEG-Conformer (optional, go/no-go on Day 26)
- **Input:** `(batch, 1, 512)` float32 tensor — single-channel windowed EEG at 256 Hz
- **Output:** `(batch, 1, 512)` float32 — denoised segment
- **Target artifacts:** EOG (ocular) and EMG (muscle)
- **Primary metrics:** ΔSNR (≥ 8 dB target), Pearson r (≥ 0.85 target)
- **Dataset:** EEGDenoiseNet — 4,514 clean + 3,400 EOG + 5,598 EMG segments

### EEGDenoiseNet — confirmed array details

| Array                | Shape        | Rate   | Duration | Role                              |
|----------------------|--------------|--------|----------|-----------------------------------|
| EEG_all_epochs       | (4514, 512)  | 256 Hz | 2s       | Clean ground truth (EOG pipeline) |
| EOG_all_epochs       | (3400, 512)  | 256 Hz | 2s       | Ocular artifact                   |
| EMG_all_epochs       | (5598, 512)  | 256 Hz | 1s       | DO NOT USE for synthesis          |
| EEG_all_epochs_512hz | (4514, 1024) | 512 Hz | 2s       | Clean EEG for EMG mixing          |
| EMG_all_epochs_512hz | (5598, 1024) | 512 Hz | 2s       | Muscle artifact for synthesis     |

### Shared Preprocessing Module (`data/preprocessing.py`)
- `bandpass_filter`, `notch_filter`, `zscore_normalize`, `segment_signal`
- Roundtrip Pearson r = 0.9849 on EEG_all_epochs_512hz.npy (verified)

### Shared Metrics Module (`evaluate/metrics.py`)
- BCR: `compute_auprc`, `compute_auroc`, `compute_f1_at_threshold`, `best_f1_threshold`, `bcr_full_report`
- Denoiser: `snr_improvement`, `pearson_r`, `rrmse`, `rrmse_spectral`, `ssim_1d`, `denoiser_full_report`

### Colab Infrastructure (`notebooks/00_colab_setup.ipynb`)
- 3-cell bootstrap: mounts Drive, clones repo, symlinks both datasets, verifies GPU
- Integration smoke test passing on T4: 18,900 rows, 43 subjects, bad_rate=0.039

---

## 4. Key Decisions

| Decision | Rationale |
|---|---|
| AUPRC as primary BCR metric | Random baseline = 0.039; AUROC misleading (naive model scores 0.95) |
| GroupKFold by subject_id | 6 subjects span multiple sessions; fold by subject prevents leakage |
| Bad threshold = score ≥ 2 | 3.9% bad rate; score ≥ 1 (12.8%) tested on Day 10 — threshold 2 chosen |
| Active threshold = 0.50 | Recall-prioritised: −35 FN, +138 FP vs Day 11 baseline (0.604); ΔF1=−0.010 |
| argmax(2R+P) threshold (0.20) rejected | 3.6× FP increase (388→1380) disproportionate to recall gain |
| Site 3 retained in training | Rater ablation ΔAUPRC = −0.0042 — null finding; site 3 does not degrade labels |
| Impedance excluded from minimum viable set | ΔAUPRC = +0.005 when removed; dispensable for BCR performance |
| `save()` writes model + sidecar atomically | Ensures threshold and feature names always travel with model binary |
| Model sidecar committed to git | Binary gitignored; sidecar (JSON, small) tracked for reproducibility |
| OOF evaluation (no retraining) | Final model trained on all data — using it for scoring would cause leakage |
| SHAP via `pred_contribs` | `shap.TreeExplainer` incompatible with XGBoost 2.x; native API used instead |
| LOVO uses visit splits (not site) | Site column is constant (=1, dropped); visits 1–4 are the actual stratification |
| TabNet/TabSTAR deferred to Day 34 | Label noise ceiling (~AUPRC 0.70 max) limits gains; XGBoost well-suited to statistical feature aggregates |
| Interpolation: spherical spline primary, zero-out fallback | Spline uses spatial context (requires montage); zero-out is unconditional and safe when no montage available |
| `assert_allclose(rtol=1e-10)` for channel preservation test | MNE pipeline (µV→V→µV unit conversion) accumulates float64 rounding at machine epsilon; exact equality is wrong standard |
| Interpolation comparison script not fixed for degenerate result | BCR has no raw time-series; proxy amplitude from feature values is in volts (~0.1 µV), making spline vs. zero-out comparison uninformative — not worth chasing; unit tests provide real validation |

---

## 5. BCR Results Summary

### OOF Evaluation
| Metric | Value |
|---|---|
| AUPRC | 0.513 (13.2× above random baseline 0.039) |
| AUROC | 0.913 |
| F1 | 0.506 |
| Active threshold | 0.50 |
| TP / FP / FN / TN | 413 / 526 / 319 / 17642 |
| FN rate | 43.6% |

### Threshold Comparison
| Criterion | Threshold | Recall | F1 | FN |
|---|---|---|---|---|
| argmax(F1) | 0.60 | 0.516 | 0.505 | 354 |
| argmax(2R+P) | 0.20 | 0.700 | 0.390 | 220 |
| **Chosen (0.50)** | **0.50** | **0.564** | **0.494** | **319** |
| Day 11 reference | 0.604 | 0.516 | 0.505 | 354 |

### LOVO Generalisation
| Test visit | AUPRC | AUROC | Bad rate |
|---|---|---|---|
| 1 | 0.4228 | 0.8984 | 3.31% |
| 2 | 0.5647 | 0.9128 | 4.89% |
| 3 | 0.4051 | 0.8835 | 3.69% |
| 4 | 0.5514 | 0.9229 | 3.78% |
| Mean ± SD | 0.486 ± 0.073 | 0.904 ± 0.017 | — |

Generalisation gap: −0.032 (6.2% relative). Verdict: **PASS**. Visit 3 marginal — fully explained by 78.3% impedance missingness.

### Feature Ablation
| Condition | ΔAUPRC | Interpretation |
|---|---|---|
| No decomposition (E) | −0.092 | Dominant group |
| No spatial (D) | −0.015 | Moderate contributor |
| No frequency (C) | −0.019 | Moderate contributor |
| No impedance (B) | +0.005 | Dispensable |
| Impedance only (F) | ≈ random (0.041) | No signal alone |

### Rater Ablation
| Condition | AUPRC | ΔAUPRC |
|---|---|---|
| All 4 raters | 0.513 | — |
| Exclude site 3 | 0.509 | −0.004 (null finding) |

### SHAP Top Features
Rank 1: `corr_neighbors_1stQ` (spatial) · Rank 2: `channel_label_enc` · Ranks 4–11: PCA/ICA decomposition · Rank 74: `impedance_missing`

### Interpolation Comparison (Day 16)
| Method | Mean MSE (µV²) | % of corrupted | Beats corrupt (n/10) |
|---|---|---|---|
| Corrupted (no repair) | 41194.6 | 100% | — |
| Spherical spline (MNE) | ~0.0 | ~0% | 10/10 |
| Zero-out (fallback) | ~0.0 | ~0% | 10/10 |

Note: comparison degenerate — BCR feature amplitudes are in volts (~0.1 µV proxy) vs. 200 µV injected noise. Unit tests provide true validation.

### False Negative Analysis (threshold = 0.50)
- **Failure mode 1 — sparsity:** PPO10h (FN rate 0.875, 8 total bad), CCP1h (0.722), CCP2h (0.700), CPP1h/CPP2h (0.667) — high-density parieto-occipital channels with too few positive training examples
- **Failure mode 2 — threshold proximity:** All top-10 hardest FNs between 0.467–0.499; 4 of 10 from Visit 3
- **Best-learned channels:** T7 (FN rate 0.130, 46 bad examples), T8 (0.178, 45 bad examples)
- **Worst visit:** Visit 3 — FN rate 0.505, 141/319 total FNs (44.2%)

---

## 6. Open Issues / Unknowns

| Issue | Notes |
|---|---|
| BCR hyperparameters not tuned | Optuna sweep is a candidate; deferred — AUPRC already strong |
| `impute_and_encode_channels` returns tuple | **Must always unpack as `df, _ = ...`** — raw assignment to `df` causes TypeError on downstream indexing |
| Label noise ceiling | Max achievable AUPRC ≈ 0.70 (best rater pair r=0.42); no closed-form formula from r to AUPRC ceiling |
| High-density parieto-occipital FN rate | PPO10h/CCP1h/CCP2h/CPP1h/CPP2h have FN rates 0.667–0.875 due to sparsity; not addressable without more data |

---

## 7. Dataset Audit Findings (BCR)

| Finding | Value |
|---|---|
| Total rows | 18,900 |
| Sessions | 50 |
| Unique subjects | 43 (verified) |
| Visits | 4 (values 1, 2, 3, 4) |
| Sites | 1 (constant — column dropped) |
| Rater columns | `Bad (site 2)`, `Bad (site 3)`, `Bad (site 4a)`, `Bad (site 4b)` |
| Channel labels | 126 unique channels |
| Tasks | 5 (3-EO, 1-EO, 4-EC, 2-EC, 0-AR) |
| Bad rate (`score ≥ 2`) | 3.9% → 732 bad / 18,168 good |
| Borderline (score=1) | 1,692 rows (8.9%) |
| Class imbalance ratio | 24.82 : 1 |
| `scale_pos_weight` | 24.82 |
| AUPRC random baseline | 0.039 |
| Session ID format | `{visit}-{subject_id}-{site}` e.g. `1-103-1` |
| Subjects per visit | Visit 1: 10 · Visit 2: 10 · Visit 3: 20 · Visit 4: 10 |
| Bad rate per visit | V1: 3.31% · V2: 4.89% (highest) · V3: 3.69% · V4: 3.78% |
| Bad / subject per visit | V1: 12.5 · V2: 18.5 · V3: 13.95 · V4: 14.3 |
| Impedance missingness | 78.3% in visit 3, 0% in visits 1/2/4 — structured, not random |
| Top 5 worst channels | T7 (30.7%), T8 (30.0%), M1, CCP2h, AF7 |
| Rater correlation — site 3 vs others | r = 0.078–0.133 (outlier) |
| Rater correlation — sites 4a↔4b | r = 0.420 (best pair) |
| Signal columns with inf values | 5 — all in `Signal-wide relative residuals` group |
| Near-redundant pairs (r > 0.99) | 24 pairs → 17 unique columns dropped |
| Constant column post-imputation | `Signal-wide PCA (mean)` — dropped explicitly |
| Feature matrix shape (post Day 9) | (18,900, 138) after preprocessing |
| GroupKFold overlap | 0 on all 5 folds (verified) |
| T7 bad rate by visit | V1: 40.0% · V2: 30.0% · V3: 23.3% · V4: 36.7% |
| T8 bad rate by visit | V1: 26.7% · V2: 43.3% · V3: 18.3% · V4: 43.3% |
| Feature amplitude units | Volts (not µV) — `Standard deviation (mean)` mean=0.000111 V, max=0.000772 V |

## 8. Dataset Audit Findings (EEGDenoiseNet)

| Finding | Value |
|---|---|
| EEG clean epochs | 4,514 × 512 samples @ 256 Hz = 2s |
| EOG epochs | 3,400 × 512 samples @ 256 Hz = 2s |
| EMG epochs (256 Hz) | 5,598 × 512 samples @ 256 Hz = 1s — DO NOT USE |
| EEG 512 Hz epochs | 4,514 × 1024 samples @ 512 Hz = 2s |
| EMG 512 Hz epochs | 5,598 × 1024 samples @ 512 Hz = 2s |
| NaNs | 0 across all arrays |
| Normalisation | NOT pre-normalised (contrary to paper claim) |
| EEG clean mean epoch std | 218.83 µV, range 63.23–502.78 µV |
| EOG synthesis status | Confirmed correct — mix_snr() at 256 Hz |
| EMG synthesis status | Confirmed correct — mix_snr() at 512 Hz + resample_poly downsample |
| Alpha peak | ~10 Hz visible in clean EEG PSD — genuine resting-state confirmed |
| EEG/EOG hard cutoff | 80 Hz — pre-filtered in source data (paper Section 2.1) |
| Single-channel constraint | No montage — BCR spatial features impossible; independent validation required |

---

## 9. Environment & Infrastructure

| Item | Detail |
|---|---|
| Conda env | `eeg_pipeline` · Python 3.10 · arm64 |
| PyTorch | 2.x · MPS available ✓ |
| Key libraries | MNE 1.12.0, XGBoost, LightGBM, SHAP 0.49.1, W&B, PyWavelets (`pywt`), Einops, Braindecode |
| System dependency | `libomp` via Homebrew (required for XGBoost) |
| GitHub repo | `https://github.com/Yago-Velazquez/eeg_pipeline` (public) |
| numpy | 2.2.6 pinned in requirements.txt |
| Config | `configs/pipeline_config.yaml` — `bcr.decision_threshold = 0.50`; also stores `decision_threshold_argmax_f1 = 0.60` and `decision_threshold_argmax_2rp = 0.20` |
| Secrets | `.env` with W&B API key — gitignored |
| BCR data | `data/raw/Bad_channels_for_ML.csv` (local + Drive) |
| EEGDenoiseNet data | `data/raw/eegdenoisenet/data/` (local + Drive) |
| Google Drive path | `MyDrive/eeg_pipeline/data/raw/` — both datasets mirrored here |
| Colab setup | `notebooks/00_colab_setup.ipynb` — mounts Drive, clones repo, symlinks data, verifies GPU |
| Git tags | `v0.1-week1` · `v0.11-day11-bcr-evaluated` · `v0.2-week2-bcr-trained` |
| W&B projects | `eeg-pipeline` (train runs) · `eeg-bcr` (eval, rater ablation, SHAP, feature ablation) — all synced |
| OOF files | `results/oof_y_true_thresh2.npy`, `results/oof_y_prob_thresh2.npy` (gitignored — regenerate via train.py) |
| Model sidecar | `results/bcr_model_thresh2_meta.json` — committed; threshold + 138 feature names |
| .gitignore | `results/*.json` gitignored; `!results/bcr_model_thresh2_meta.json` tracked; `results/figures/` gitignored; `results/*.md` tracked |
| SHAP compatibility | XGBoost 2.x `base_score` patch at file level in `load_patched_booster()` |
| bcr_selected_features.json | Feature list nested under `data["feature_names_out"]` — not top-level |
| Key external repos | NCClab/EEGdenoiseNet (data) · ncclabsustech/Single-Channel-EEG-Denoise (Day 37 baselines) · ncclabsustech/DeepSeparator (DL baseline) |
| Test suite | 49 tests passing (16 interpolation + 6 BCR metrics + 12 denoiser metrics + 15 BCR integration) · 4.3s runtime |

---

## 10. Recent Updates

### Day 16
- Implemented `bad_channel_rejection/interpolation.py`: `spherical_spline_interpolation()` (MNE 1.12.0), `zero_out_channel()` fallback, `interpolate_bad_channels()` router
- 16 unit tests written and passing; `test_other_channels_unchanged` uses `assert_allclose(rtol=1e-10)` — MNE's µV→V→µV pipeline accumulates float64 rounding at machine epsilon, making exact equality the wrong standard
- Comparison script on 10 visit-4 sessions: both methods reduce MSE >99.99% vs. 200 µV noise; spline vs. zero-out comparison degenerate (BCR features in volts → ~0.1 µV proxy amplitude); accepted as-is, documented in `results/interpolation_analysis.md`
- Confirmed BCR feature values are in **volts** (`Standard deviation (mean)` mean=0.000111 V); relevant for any future amplitude-based synthesis from BCR features
- Full test suite: 49/49 passing — no regressions
- **BCR component now complete end-to-end**
- Committed and pushed

### Day 15
- Implemented `bad_channel_rejection/site_generalisation.py` — LOVO evaluation across all 4 visits; fixed `impute_and_encode_channels` tuple-unpack bug (`df, _ = ...`)
- Threshold sweep (0.20→0.80): chosen threshold = 0.50 (recall-weighted, −35 FN vs baseline at cost of +138 FP)
- Updated `configs/pipeline_config.yaml`: `bcr.decision_threshold=0.50`; also stores argmax_f1=0.60 and argmax_2rp=0.20
- LOVO results: V1=0.423, V2=0.565, V3=0.405, V4=0.551 · mean=0.486±0.073 · gap=−0.032 → **PASS**
- False negative analysis: FN rate=43.6%; two failure modes — sparsity (PPO10h FN rate 0.875) and threshold proximity (top-10 FNs all within 0.033 of threshold, 4/10 from Visit 3)
- Wrote `results/bcr_error_analysis.md` and `results/bcr_site_generalisation.md`

### Day 14
- Wrote `report/week2_log.md` and `report/methods_bcr.md`
- Generated and committed `results/bcr_model_thresh2_meta.json` sidecar
- Updated `.gitignore`; moved tag `v0.2-week2-bcr-trained` to current commit
- **Week 2 Go/No-Go: PASS**

### Day 13
- Implemented `bad_channel_rejection/model.py` — `BadChannelDetector` class
- Implemented `tests/test_bcr_integration.py` — 15 tests, all passing

### Days 8–12
- `dataset.py`, `features.py`, `train.py`, `evaluate.py`, `shap_analysis.py`, `rater_ablation.py`, `feature_ablation.py` implemented
- OOF: AUPRC=0.513, AUROC=0.913, F1=0.506, lift=13.2×; decomposition dominant (ΔAUPRC −0.092)

### Days 1–7
- Project scope, environment, data download, preprocessing/metrics modules, Colab setup, Week 1 tagged `v0.1-week1`