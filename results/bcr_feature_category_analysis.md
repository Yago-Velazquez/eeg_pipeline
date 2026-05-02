# BCR Feature Category Analysis (Stage-3 aware)

**Model**: `lightgbm`  |  **Label strategy**: `mace`  |  **Stage 3 reference AUPRC**: 0.5406

## Feature categorisation

| Category | # Features | Sample |
|----------|------------|--------|
| `impedance` | 1 | impedance_missing |
| `frequency` | 21 | Median frequency (mean), Median frequency (std), Median frequency (median), … |
| `amplitude` | 7 | Standard deviation (mean), Standard deviation (std), Standard deviation (median), … |
| `spatial` | 22 | Global correlation (mean), Global correlation (std), Global correlation (median), … |
| `band_power` | 7 | Low gamma/high gamma ratio (mean), Low gamma/high gamma ratio (std), Low gamma/high gamma ratio (median), … |
| `decomposition` | 98 | Signal-wide residuals (mean), Signal-wide residuals (std), Signal-wide residuals (median), … |

## Leave-one-category-out CV

Baseline (all features): AUPRC = 0.5331

| Condition | # Features | AUPRC | AUROC | ΔAUPRC | % change | Verdict |
|-----------|------------|-------|-------|--------|----------|---------|
| `A_baseline_all` | 156 | 0.5331 ± 0.0848 | 0.9283 | — | — | — |
| `drop_impedance` | 155 | 0.5164 ± 0.1115 | 0.9225 | -0.0166 | -3.1% | 🔒 keep |
| `drop_frequency` | 135 | 0.5073 ± 0.0570 | 0.9266 | -0.0258 | -4.8% | 🔒 keep |
| `drop_amplitude` | 149 | 0.5216 ± 0.1129 | 0.9183 | -0.0115 | -2.2% | 🔒 keep |
| `drop_spatial` | 134 | 0.5187 ± 0.0911 | 0.9151 | -0.0144 | -2.7% | 🔒 keep |
| `drop_band_power` | 149 | 0.5270 ± 0.0812 | 0.9255 | -0.0060 | -1.1% | 🔒 keep |
| `drop_decomposition` | 58 | 0.4431 ± 0.1063 | 0.9034 | -0.0899 | -16.9% | 🔒 keep |

Auto-prune threshold: ΔAUPRC ≥ -0.0050 → category is removable.

## Reduced feature set

No category was removable under the current threshold (0.0050). Every feature category contributes to the full model's AUPRC. **Use `results/best_model.<ext>` for production.**

## HPO config used

```json
{
  "num_leaves": 31,
  "learning_rate": 0.03718038736102388,
  "subsample": 0.5298233959146713,
  "colsample_bytree": 0.7210702069316316,
  "min_child_samples": 15,
  "reg_alpha": 0.4571499741016649,
  "reg_lambda": 0.00037975040647670835
}
```