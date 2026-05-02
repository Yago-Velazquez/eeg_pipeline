# BCR Stage 3 — HPO Report

Model: **lightgbm** | Label strategy: **mace** | Trials: **50** | Device: **cpu**

Optuna study: `bcr_hpo_lightgbm_1777667610`

## Best Configuration

Best trial: **#44**

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

**Best CV AUPRC:** 0.5406 ± 0.0852
**Best CV AUROC:** 0.9318
**Saved model:**   `results/best_model.pkl`

## All Trials (top 10)

| Rank | Trial # | AUPRC mean | AUPRC std | AUROC mean |
|------|---------|------------|-----------|------------|
| 1 | 44 ✓ | 0.5406 | 0.0852 | 0.9318 |
| 2 | 43 | 0.5405 | 0.0930 | 0.9280 |
| 3 | 46 | 0.5384 | 0.0824 | 0.9279 |
| 4 | 33 | 0.5368 | 0.0888 | 0.9282 |
| 5 | 9 | 0.5360 | 0.0868 | 0.9296 |
| 6 | 8 | 0.5351 | 0.0819 | 0.9293 |
| 7 | 36 | 0.5346 | 0.0823 | 0.9296 |
| 8 | 35 | 0.5344 | 0.0914 | 0.9273 |
| 9 | 32 | 0.5338 | 0.0822 | 0.9267 |
| 10 | 41 | 0.5305 | 0.0987 | 0.9291 |