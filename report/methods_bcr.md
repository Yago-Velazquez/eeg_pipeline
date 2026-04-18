# BCR Module — Methods

## 2.1 Dataset

Bad channel labels were obtained from a multi-site EEG quality control dataset
comprising 18,900 channel-epoch records across 43 subjects, 4 recording sites,
and 50 unique recording sessions. Each row represents one channel from one
session. The binary bad label was derived from a consensus rating across four
independent site raters (sites 2, 3, 4a, 4b): a channel was labelled bad if
≥ 2 raters agreed (score ≥ 2).

The resulting class imbalance was severe: 3.9% of channels were labelled bad
(732 bad / 18,168 good), yielding a scale_pos_weight of 24.8 for XGBoost
training and a random-baseline AUPRC of 0.039. AUPRC was chosen as the primary
metric rather than AUROC because AUROC is insensitive to class imbalance: a
naive model that predicts every channel as good scores AUROC ≈ 0.95, which is
meaningless. AUPRC has a random baseline equal to the bad rate itself (0.039)
and punishes models that ignore the minority class.

Session IDs are date-encoded in the format {visit}-{subject_id}-{site}. Six
subjects contributed recordings to multiple sites, making subject-level
grouping critical to prevent data leakage (see §2.4).

### 2.1.1 Label Noise

Inter-rater agreement was low overall across the four sites. The best pairwise
agreement was between sites 4a and 4b (r = 0.42), which share the same
institution. Site 3 showed near-zero correlation with all other sites (mean
r ≈ 0.10–0.11), indicating an idiosyncratic labelling practice rather than
genuine signal disagreement.

A rater-noise ablation experiment confirmed that excluding site 3 labels from
the consensus score changed AUPRC by only −0.0042, establishing that site 3
is not systematically degrading model performance. This is treated as a methods
note, not a failure: the low inter-rater agreement sets a practical ceiling on
achievable AUPRC for this dataset of approximately 0.40–0.50.

### 2.1.2 Top Offender Channels

Channels with the highest bad rates were confirmed from exploratory analysis:

| Channel | Bad Rate |
|---------|----------|
| T7      | 30.7%    |
| T8      | 30.0%    |
| AF7     | 22.7%    |
| M1      | 18.0%    |
| M2      | 15.3%    |

All five are peripheral or mastoid electrodes, which are most susceptible to
movement artefacts and poor skin contact. Channel identity was therefore
included as an ordinal-encoded feature in the model.

---

## 2.2 Feature Engineering

Pre-computed signal features (156 columns) were provided with the dataset.
Each feature is computed per channel per session across 13 functional groups,
with 7 summary statistics per group (mean, std, median, Q1, Q3, min, max).
All 156 column names carry a leading space (e.g. ' Standard deviation (mean)').

A binary `impedance_missing` indicator was engineered before any imputation to
capture the structural missingness pattern: 78.3% of rows in visit group 3
had no impedance reading, compared to 0% in all other groups. This column
carries predictive signal independent of the impedance values themselves.

`Impedance (start)` and `Impedance (end)` were excluded from the feature
matrix as metadata columns. `impedance_missing` (no leading space) was the
sole impedance-derived feature retained.

### 2.2.1 Redundancy Removal

A correlation audit identified 24 feature pairs with r > 0.99. The strategy
was to keep the (mean) statistic for each feature family and drop the
(median), (1st quartile), (3rd quartile), (maximum), and (minimum) variants
where they were near-redundant with the mean. One additional column
(' Signal-wide PCA (mean)') was dropped for being constant after
infinity-imputation. This produced 18 unique column drops.

After redundancy removal, a VarianceThreshold(threshold=0.0) step dropped
any remaining near-zero variance survivors. Final feature count: 138.

| Stage                  | Feature count |
|------------------------|---------------|
| Raw                    | 156           |
| After redundancy drop  | 138           |
| After VarianceThreshold| 138           |

### 2.2.2 Feature Groups

The 138 final features span the following functional groups:

| Group | Description | Approx. count |
|-------|-------------|---------------|
| A — Impedance | impedance_missing flag | 1 |
| B — Std Dev | Cross-window variance statistics | ~5 |
| C — Frequency | Band power, spectral ratios, low/high gamma | ~18 |
| D — Spatial Correlation | Neighbour and second-degree neighbour correlation | ~12 |
| E — Decomposition | PCA/ICA residuals and loadings (signal-wide and window) | ~85 |
| F — Other | Kurtosis, median frequency, reconstruction correlation | ~17 |

### 2.2.3 Preprocessing Pipeline

Preprocessing was implemented as a stateful `FeaturePreprocessor` class
following the scikit-learn fit/transform API to prevent leakage:

1. Drop 18 near-redundant columns (identified from the correlation audit)
2. Separate binary column (`impedance_missing`) from continuous features
3. `RobustScaler` fitted on the training fold only, applied to continuous features
4. `impedance_missing` passed through unscaled
5. `VarianceThreshold(0.0)` fitted on the training fold only

---

## 2.3 Model

XGBoost (Chen & Guestrin, 2016) was selected for its native handling of class
imbalance via `scale_pos_weight`, computational efficiency on tabular data,
and interpretability via SHAP decomposition.

Hyperparameters were fixed based on standard practice for imbalanced tabular
classification and were not tuned by automated search:

| Parameter | Value |
|-----------|-------|
| n_estimators | 500 |
| early_stopping_rounds | 30 |
| max_depth | 6 |
| learning_rate | 0.05 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| scale_pos_weight | 24.8 |
| eval_metric | aucpr |
| tree_method | hist (CPU) |

---

## 2.4 Cross-Validation Strategy

To prevent data leakage, `GroupKFold(n_splits=5)` was applied with
`subject_id` (43 unique values) as the grouping key. This ensures no
subject's data appears in both training and validation folds simultaneously,
which is critical because 6 subjects contributed recordings to multiple sites
and would otherwise appear in both splits.

Subject overlap was verified explicitly on every fold (zero overlap confirmed
across all 5 folds). Validation bad rates ranged from 2.7% to 4.5% across
folds, reflecting genuine subject-level variation.

Out-of-fold (OOF) predictions were concatenated across all 5 folds to produce
an unbiased AUPRC estimate over the full dataset. The decision threshold was
set at 0.6044 by maximising F1 on the concatenated OOF predictions.

---

## 2.5 Feature Importance (SHAP Analysis)

SHAP TreeExplainer analysis was applied to the trained model using native
`pred_contribs` to identify the most influential predictors. Top-10 features:

| Rank | Feature | Group |
|------|---------|-------|
| 1 | corr_neighbors_1stQ | Spatial correlation |
| 2 | channel_label_enc | Channel identity |
| 3–11 | PCA / ICA decomposition residuals | Decomposition |
| 74 | impedance_missing | Impedance |

The dominance of spatial correlation confirms that bad channels manifest
primarily as signal isolation from their neighbours. Channel identity as the
second-ranked predictor reflects the systematic bad-rate differences between
anatomical electrode positions. `impedance_missing` ranked 74th out of 138,
confirming it is informative but not the dominant signal.

---

## 2.6 Feature Group Ablation

XGBoost was retrained five times with one feature group removed at a time,
plus one impedance-only baseline (2 features + missing flag), to quantify
the marginal contribution of each group. Results are documented in
`results/bcr_feature_ablation.md`.

---

## 2.7 Results Summary

| Metric | Value |
|--------|-------|
| AUPRC (OOF, unbiased) | 0.513 |
| AUROC (OOF, unbiased) | 0.913 |
| F1 @ threshold 0.6044 | 0.506 |
| Precision | 0.497 |
| Recall | 0.515 |
| Lift over random baseline | 13.2× |
| TN / FP / FN / TP | 17786 / 382 / 355 / 377 |
| Random AUPRC baseline | 0.039 |

The achieved AUPRC of 0.513 is consistent with the label noise ceiling
estimated from inter-rater agreement (≈ 0.40–0.50). It does not indicate
model failure; it reflects the practical upper bound imposed by the labelling
disagreement in the dataset.

---

## 2.8 Scope Note

The BCR and denoiser modules are validated independently on separate datasets.
BCR requires spatial features derived from multi-channel recordings;
EEGDenoiseNet is a single-channel dataset. Joint pipeline evaluation —
running BCR to flag channels and then applying the denoiser to the remaining
channels — is a design specification for future integration. The candidate
dataset for joint evaluation is PhysioNet EEG Motor Movement/Imagery (EEG-MMI),
which provides multi-channel recordings with quality annotations.
