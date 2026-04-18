# BCR Evaluation Report

## Source
Out-of-fold (OOF) predictions from GroupKFold(5) CV in train.py.
Each row's prediction came from a model that never saw that row's subject during training.
No retraining performed here — these are the unbiased CV scores.

## Metrics (pooled OOF)
| Metric    | Value  |
|-----------|--------|
| AUPRC     | 0.5130 |
| AUROC     | 0.9131 |
| F1        | 0.5057 |
| Precision | 0.4967 |
| Recall    | 0.5150 |
| Threshold | 0.6044 |

## Confusion matrix (at optimal threshold)
| | Predicted Good | Predicted Bad |
|---|---|---|
| **True Good** | 17786 (TN) | 382 (FP) |
| **True Bad**  | 355 (FN) | 377 (TP) |

## Context
- Random AUPRC baseline: 0.039
- Lift over random: 13.2×
- Label noise ceiling ≈ 0.40–0.50 (Site 3 inter-rater r ≈ 0.10)
- Optimal threshold written to configs/pipeline_config.yaml
