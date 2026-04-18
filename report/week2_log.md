# Week 2 Log — BCR Training & Evaluation

## Model
- Algorithm: XGBoost (GroupKFold n=5, grouped by subject_id)
- Features: 138 (156 raw → 18 dropped: 17 redundant r>0.99 + Signal-wide PCA constant)
- scale_pos_weight: 24.8 (18168 good / 732 bad)
- Decision threshold: 0.6044 (argmax F1 on OOF predictions)

## Results (unbiased OOF)
- AUPRC:  0.513  (random baseline: 0.039 — lift 13.2×)
- AUROC:  0.913
- F1:     0.506
- TN/FP/FN/TP: 17786 / 382 / 355 / 377

## Key findings
- Top feature: corr_neighbors_1stQ (spatial isolation is the strongest bad-channel signal)
- Channel identity (#2): T7=30.7%, T8=30.0%, AF7=22.7% bad rate — peripheral electrodes
- impedance_missing: SHAP rank #74/138 — informative but not dominant
- Rater noise ablation (site 3 excluded): ΔAUPRC = −0.0042 — null finding,
  site 3 does not degrade labels meaningfully
- Feature group ablation: [paste your top finding from bcr_feature_ablation.md]

## Go/No-Go check
- AUPRC = 0.513 >> 0.30 threshold → PROCEED to Week 3
