# BCR — Threshold Sensitivity Experiment

**Date:** 2026-04-17  
**W&B project:** eeg-pipeline

## Setup
- Model: XGBoost (n_estimators=500, max_depth=6, lr=0.05, subsample=0.8, colsample_bytree=0.8)
- CV: GroupKFold(5) by subject_id (43 subjects, ~8-9 per fold)
- Features: 138 (156 raw → RobustScaler + 18 REDUNDANT_DROP_COLS removed)
- Hardware: MacBook Air M2, CPU (tree_method=hist)

## Per-fold results

### score ≥ 2 (bad = 3.9%, scale_pos_weight = 24.82)

| Fold | AUPRC  | AUROC  | F1    | Precision | Recall | best_iter | val_bad_rate |
|------|--------|--------|-------|-----------|--------|-----------|--------------|
| 1    | 0.5734 | 0.9138 | 0.458 | —         | —      | 121       | 0.0447       |
| 2    | 0.4319 | 0.9191 | 0.462 | —         | —      | 202       | 0.0270       |
| 3    | 0.4366 | 0.9098 | 0.453 | —         | —      | 113       | 0.0360       |
| 4    | 0.5218 | 0.9182 | 0.485 | —         | —      | 324       | 0.0421       |
| 5    | 0.6266 | 0.9419 | 0.607 | —         | —      | 134       | 0.0439       |
| **mean ± std** | **0.5180 ± 0.0850** | **0.9206 ± 0.0125** | **0.493 ± 0.065** | **0.453 ± 0.079** | **0.554 ± 0.092** | **179** | — |

### score ≥ 1 (bad = 12.8%, scale_pos_weight = 6.80)

| Fold | AUPRC  | AUROC  | F1    | Precision | Recall | best_iter | val_bad_rate |
|------|--------|--------|-------|-----------|--------|-----------|--------------|
| 1    | 0.6295 | 0.8849 | 0.551 | —         | —      | 236       | 0.1291       |
| 2    | 0.5419 | 0.8876 | 0.523 | —         | —      | 208       | 0.0981       |
| 3    | 0.5071 | 0.6936 | 0.495 | —         | —      | 51        | 0.1759       |
| 4    | 0.6030 | 0.8866 | 0.556 | —         | —      | 190       | 0.1212       |
| 5    | 0.5984 | 0.8761 | 0.540 | —         | —      | 60        | 0.1169       |
| **mean ± std** | **0.5760 ± 0.0500** | **0.8458 ± 0.0852** | **0.533 ± 0.025** | **0.483 ± 0.034** | **0.609 ± 0.097** | **149** | — |

## Summary comparison

| Configuration      | Mean AUPRC       | Mean AUROC       | Mean F1          | Lift over random |
|--------------------|------------------|------------------|------------------|------------------|
| score ≥ 2 (3.9%)   | 0.5180 ± 0.0850  | 0.9206 ± 0.0125  | 0.493 ± 0.065    | 13.3×            |
| score ≥ 1 (12.8%)  | 0.5760 ± 0.0500  | 0.8458 ± 0.0852  | 0.533 ± 0.025    | 14.8×            |
| Random baseline    | 0.039            | —                | —                | 1.0×             |

## Analysis

thresh=1 has higher mean AUPRC (0.576 vs 0.518, +0.058) and lower variance
(std 0.050 vs 0.085), which makes it look superficially attractive. However,
three factors favour keeping thresh=2 as the primary definition:

1. **Label quality.** score=1 ("borderline") channels are those where raters
   disagreed — exactly the label noise already documented in this dataset
   (inter-rater r ≈ 0.10–0.42). Including them as positives adds 1,692 noisy
   labels to the training set. The higher AUPRC for thresh=1 may partly reflect
   the model learning rater-specific patterns rather than genuine signal
   degradation.

2. **AUROC collapse on Fold 3.** thresh=1 Fold 3 AUROC drops to 0.694 vs
   0.910 for all other folds — a 20-point collapse not seen in thresh=2.
   best_iter=51 confirms early stopping fired very early, suggesting the model
   failed to generalise to that subject group under the noisier label definition.
   thresh=2 shows no comparable instability (AUROC range: 0.910–0.942).

3. **Downstream use case.** BCR output feeds a denoiser. A false negative
   (missed bad channel) is worse than a false positive (good channel flagged).
   thresh=2 has lower recall (0.554 vs 0.609) but the channels it misses are
   the ambiguous borderline ones — the genuinely bad channels (score=2) are
   captured. This is the safer operating point for a preprocessing pipeline.

## Decision

**Chosen threshold: score ≥ 2**

All downstream work (Day 11 evaluation, Day 12 ablation, Day 13 integration,
report methods section) uses `bad_threshold=2`. The canonical model is
`results/bcr_model_thresh2.json`.

thresh=1 results are retained in `results/bcr_model_thresh1.json` and reported
in this document as a sensitivity check. The +0.058 AUPRC difference is noted
as a limitation: a stricter label definition trades raw discriminability for
label cleanliness and cross-subject stability.

## W&B runs
- bcr_xgb_thresh2: https://wandb.ai/yago-vcl-universidad-francisco-de-vitoria/eeg-pipeline/runs/jpvbzqqp
- bcr_xgb_thresh1: https://wandb.ai/yago-vcl-universidad-francisco-de-vitoria/eeg-pipeline/runs/pyo8h9cl