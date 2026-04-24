# BCR Feature Group Ablation

| Condition | # Features | AUPRC (mean±std) | AUROC | ΔAUPRC | % change |
|---|---|---|---|---|---|
| A_all_features | 156 | 0.5294 ± 0.0905 | 0.9242 | — | — |
| B_no_impedance | 155 | 0.5321 ± 0.0864 | 0.9255 | +0.0027 | +0.5% |
| C_no_frequency | 135 | 0.5162 ± 0.0766 | 0.9262 | -0.0132 | -2.5% |
| D_no_spatial | 135 | 0.5399 ± 0.0901 | 0.9204 | +0.0105 | +2.0% |
| E_no_decomposition | 51 | 0.4115 ± 0.1080 | 0.8844 | -0.1179 | -22.3% |
| F_impedance_only | 1 | 0.0414 ± 0.0084 | 0.5280 | -0.4880 | -92.2% |

## Baseline
- All features: AUPRC = 0.5294 ± 0.0905
- Random baseline: 0.039