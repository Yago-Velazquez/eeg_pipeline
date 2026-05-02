# BCR Evaluation — best

## Source
Out-of-fold predictions from GroupKFold(5) CV. Each row's prediction came
from a model that never saw that row's subject during training.

## Threshold-independent metrics
| Metric | Value |
|--------|-------|
| AUPRC  | 0.5451 |
| AUROC  | 0.9287 |
| Lift over random | 14.0× (baseline = 0.039) |

## Operating points
| Mode | Threshold | Precision | Recall | F1 | TP | FP | FN | TN |
|------|-----------|-----------|--------|----|----|----|----|----|
| **F1-optimal (balanced research benchmark)** | 0.7632 | 0.5782 | 0.5000 | 0.5363 | 366 | 267 | 366 | 17901 |
| **Recall target (production BCR — catch most bad channels)** | 0.0658 | 0.1782 | 0.8511 | 0.2946 | 623 | 2874 | 109 | 15294 |
| **Precision target (conservative auto-action)** | 0.9336 | 0.8000 | 0.2842 | 0.4194 | 208 | 52 | 524 | 18116 |

## Confusion matrices
Plots in `results/figures/bcr_confusion_matrix_<tag>_<mode>.png`.

## Context
- Random AUPRC baseline: 0.039
- Label noise context: Krippendorff α = 0.211 (fair agreement)
- For BCR production use, prefer the **Recall target** row over F1-optimal:
  missing a bad channel costs more than interpolating an extra good one.