# BCR Two-Stage Ablation Report

## Background

Inter-rater reliability analysis on this dataset shows:
- Krippendorff α = 0.211 (fair agreement)
- D_o = 6.8% observed pairwise disagreement
- Site 3 sensitivity (DS) = 0.075 — nearly blind to bad channels
- Site 4a sensitivity (DS) = 0.714 — most reliable positive detector
- 493 channels (2.6%) at score=2 represent maximum ambiguity

## Stage 1 — Label ablation (model = XGBoost)

| ID | Strategy | AUPRC (mean ± std) | Δ vs A |
|----|----------|--------------------|--------|
| A | hard_threshold | 0.5180 ± 0.0760 | — |
| B | entropy_weights | 0.5172 ± 0.1011 | -0.0008 |
| C | dawid_skene | 0.5218 ± 0.0855 | +0.0037 |
| D | dawid_skene_soft | 0.5178 ± 0.0806 | -0.0002 |

**Stage 1 winner:** `dawid_skene` (condition C)

## Stage 2 — Model ablation (label = dawid_skene)

| Model | AUPRC (mean ± std) | Δ vs XGBoost |
|-------|--------------------|--------------|
| xgboost | 0.5218 ± 0.0855 | — |
| lightgbm | 0.5261 ± 0.0879 | +0.0044 |
| catboost | 0.5184 ± 0.0844 | -0.0033 |

**Stage 2 winner:** `lightgbm`

## Recommended production config

- Label strategy: `dawid_skene`
- Model backend : `lightgbm`

Run `python -m bad_channel_rejection.train --label-strategy dawid_skene --model lightgbm` to train the final production model.