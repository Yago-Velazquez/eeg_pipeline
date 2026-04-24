# BCR Evaluation — dawid_skene_lightgbm

## Source
Out-of-fold predictions from GroupKFold(5) CV. Each row's prediction came
from a model that never saw that row's subject during training.

## Metrics (pooled OOF)
| Metric    | Value  |
|-----------|--------|
| AUPRC     | 0.5297 |
| AUROC     | 0.9202 |
| F1        | 0.5206 |
| Precision | 0.6184 |
| Recall    | 0.4495 |
| Threshold | 0.7017 |

## Confusion matrix (at optimal threshold)
| | Predicted Good | Predicted Bad |
|---|---|---|
| **True Good** | 17965 (TN) | 203 (FP) |
| **True Bad**  | 403 (FN) | 329 (TP) |

## Context
- Random AUPRC baseline: 0.039
- Lift over random: 13.6×
- Label noise context: Krippendorff α = 0.211 (fair agreement)
