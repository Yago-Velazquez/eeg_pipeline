# BCR Stage 2b — Architecture Ablation Report

Family: **boosting** | Label strategy: **mace** | Device: **cpu** | Folds: **5**

| Rank | Architecture | AUPRC (mean ± std) | Best iter |
|------|--------------|--------------------|-----------|
| 1 | lightgbm ✓ | 0.5250 ± 0.0926 | 129 |
| 2 | xgboost | 0.5203 ± 0.0870 | 152 |
| 3 | histgradientboosting | 0.5108 ± 0.0991 | 97 |
| 4 | catboost | 0.4822 ± 0.1047 | 119 |

**Winning architecture:** `lightgbm`  (family: `boosting`)
> ⚠️ **Caution:** winning margin < 0.005 AUPRC — result may be within numerical-noise range.


## Final CV (winner re-evaluation on same folds)

| AUPRC mean | AUPRC std | Drift vs ablation |
|------------|-----------|-------------------|
| 0.5250 | 0.0926 | stable ✓ |

## Next step

Run Stage 3 hyper-parameter optimisation:

```
python -m bad_channel_rejection.ablation_stage3_hpo \
    --winning-strategy mace \
    --winning-model lightgbm \
    --count 50
```