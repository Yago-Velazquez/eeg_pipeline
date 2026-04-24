# BCR Engineered-Feature Ablation

Five-condition study isolating the contribution of each label-aware transform. All transforms are fit per-fold (GroupKFold(5) by subject) on training data only.

| Condition | # Features | AUPRC (mean±std) | AUROC | ΔAUPRC | % change |
|---|---|---|---|---|---|
| E_baseline | 138 | 0.5294 ± 0.0905 | 0.9242 | — | — |
| E_bad_rate_only | 138 | 0.5229 ± 0.0876 | 0.9111 | -0.0065 | -1.2% |
| E_spatial_pruner_only | 125 | 0.5323 ± 0.0862 | 0.9236 | +0.0029 | +0.6% |
| E_impedance_interactions_only | 143 | 0.5376 ± 0.0858 | 0.9270 | +0.0082 | +1.6% |
| E_all_three | 130 | 0.5209 ± 0.0918 | 0.9105 | -0.0085 | -1.6% |

## Baseline
- E_baseline (FeaturePreprocessor only, fit per-fold): AUPRC = 0.5294 ± 0.0905
- Random baseline: 0.039

## Notes
- Ceiling bounded by Krippendorff α = 0.211 on the labels.
- All transforms fit on training-fold labels only — no leakage.