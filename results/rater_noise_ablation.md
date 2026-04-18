# Rater Noise Ablation Report

## Setup
- Feature matrix: fixed (same X for both conditions)
- Condition A: bad label = Bad (score) >= 2 (all 4 rater sites)
- Condition B: bad label recomputed from sites 2, 4a, 4b only (site 3 excluded, majority vote >= 2/3)
- CV: GroupKFold(5) by subject_id

## Results

| Condition          | AUPRC mean | AUPRC std | AUROC mean |
|--------------------|------------|-----------|------------|
| All 4 raters       | 0.5097     | 0.0872    | 0.9163     |
| No site 3          | 0.5055     | 0.0950    | 0.9173     |
| **ΔAUPRC**         | **-0.0042**     |           |            |

## Finding
Site 3 does **not** substantially degrade label quality.

Site 3 inter-rater correlation with all other sites: r ≈ 0.10 (confirmed in EDA).
For context, sites 4a<->4b show r = 0.42. Site 3 is a statistical outlier in rater agreement.

## Interpretation for report
This is a **methods note**, not a model failure. The label noise ceiling
limits maximum achievable AUPRC regardless of model quality.
All primary results use the full consensus score (Condition A) for reproducibility.
