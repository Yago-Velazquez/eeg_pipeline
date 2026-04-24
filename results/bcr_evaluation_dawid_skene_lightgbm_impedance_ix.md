# BCR Evaluation — dawid_skene_lightgbm_impedance_ix

## Source
Out-of-fold predictions from GroupKFold(5) CV. Each row's prediction came
from a model that never saw that row's subject during training.

## Metrics (pooled OOF)
| Metric    | Value  |
|-----------|--------|
| AUPRC     | 0.5386 |
| AUROC     | 0.9188 |
| F1        | 0.5168 |
| Precision | 0.5093 |
| Recall    | 0.5246 |
| Threshold | 0.4093 |

## Confusion matrix (at optimal threshold)
| | Predicted Good | Predicted Bad |
|---|---|---|
| **True Good** | 17798 (TN) | 370 (FP) |
| **True Bad**  | 348 (FN) | 384 (TP) |

## Context
- Random AUPRC baseline: 0.039
- Lift over random: 13.8×
- Label noise context: Krippendorff α = 0.211 (fair agreement)
