# BCR Dataset — Audit Findings

## Basic statistics
- Total rows: 18,900 (channels × sessions)
- Sessions: 50 (encoded as session_group-subject_id-visit_id in Session column)
- Unique subjects: 43 (GroupKFold groups)
- Session groups: 4 (values 1, 2, 3, 4 — NOT the same as rater sites)
- Rater sites: 4 (sites 2, 3, 4a, 4b — encoded in Bad (site X) columns)
- Channel labels: 126 unique channels
- Features (pre-computed): 156 signal features + impedance columns
- Tasks: 5 (3-EO, 1-EO, 4-EC, 2-EC, 0-AR)

## Class imbalance
- Bad rate (score≥2): 3.9% → 732 bad / 18,168 good
- scale_pos_weight for XGBoost: 24.8
- Random AUPRC baseline: 0.039 ← USE THIS, not AUROC
- AUROC is misleading here (naive model scores 0.95 by predicting all GOOD)
- Borderline zone (score=1): 1,692 rows (8.9%) — treated as GOOD under
  threshold≥2, subject to threshold sensitivity experiment on Day 10

## Session group structure
- Session ID format: [session_group]-[subject_id]-[visit_id]
- 6 subjects appear in multiple session groups (cross-group subjects):
  subject 103 (3 groups), subjects 107, 123, 190, 247, 265 (2 groups each)
- GroupKFold MUST group by subject_id (43 values), NOT Session
- Bad rate per session group:
  - Session group 1: 3.3%
  - Session group 2: 4.9% (highest)
  - Session group 3: 3.7%
  - Session group 4: 3.3%

## Task analysis
- Bad rate varies by task — task type is a predictive feature:
  - 3-EO (Eyes Open, group 3): 5.3% — highest
  - 1-EO (Eyes Open, group 1): 4.8%
  - 4-EC (Eyes Closed, group 4): 4.1%
  - 2-EC (Eyes Closed, group 2): 3.5%
  - 0-AR (Artifact Rejection):  2.7% — lowest
- Eyes Open consistently worse than Eyes Closed — more movement,
  blinks, and muscle activity degrade electrode contact

## CRITICAL: Rater agreement (inter-site)
- Rater sites are 2, 3, 4a, 4b — independent bad channel judgements
- Full correlation matrix:
  - Site 2  ↔ Site 3:  r = 0.133
  - Site 2  ↔ Site 4a: r = 0.292
  - Site 2  ↔ Site 4b: r = 0.250
  - Site 3  ↔ Site 4a: r = 0.130
  - Site 3  ↔ Site 4b: r = 0.078
  - Site 4a ↔ Site 4b: r = 0.420 (best agreement — same institution)
- Site 3 mean correlation with all other sites: ~0.11
- This is label noise, not a model failure.
- Interpretation: AUPRC < 0.70 does not indicate model failure on this dataset.
- Overall agreement is low even between best pair (4a↔4b = 0.420) —
  this is a hard ceiling on achievable AUPRC for the entire dataset
- Planned ablation: Day 11 — exclude site 3 labels from training
  and measure AUPRC delta

## Impedance missingness
- 31.3% of rows have no impedance reading (17 sessions)
- Feature engineered: `impedance_missing` binary column (added BEFORE imputation)
- Missingness is entirely site-structured — not random:
  - Session group 1: 0.0% missing
  - Session group 2: 0.0% missing
  - Session group 3: 78.3% missing ← site 3 had no impedance protocol
  - Session group 4: 0.0% missing
- impedance_missing is a meaningful feature, not just a gap to fill

## Feature groups
- Total: 156 signal features across 13 groups × 7 statistics each
  (mean, std, median, Q1, Q3, min, max)
- Groups: Standard deviation, Low gamma/high gamma ratio,
  Global correlation, Signal-wide residuals, Window-specific residuals,
  Window-specific independence, Signal-wide PCA, Window-specific PCA,
  Signal-wide ICA, Signal-wide ICA50, Kurtosis, Median frequency,
  Correlation with neighbors, + spatial/frequency variants
- Plus: Impedance (start), Impedance (end), impedance_missing (3 features)
- Note: 27 feature pairs with r > 0.99 expected — to be deduplicated in Week 2

## Top offender channels
- Highest bad rate (confirmed from EDA):
  - T7:  30.7%
  - T8:  30.0%
  - AF7: 22.7%
  - M1:  18.0%
  - M2:  15.3%
- All are peripheral/mastoid electrodes — most susceptible to movement
  artifacts and poor skin contact
- Channel identity is a strong predictive feature for Week 2

## GroupKFold validation
- GroupKFold(5) by subject_id confirmed clean — overlap=0 on all 5 folds
- Val bad rate varies 2.7%–4.5% across folds (subject-level differences)
- Report mean ± std AUPRC across all 5 folds in Week 2, not single fold

## Pipeline note
- BCR and denoiser are validated independently on separate datasets
- BCR requires spatial features (multi-channel); EEGDenoiseNet is single-channel
- Joint pipeline evaluation is future work (candidate dataset: PhysioNet EEG-MMI)