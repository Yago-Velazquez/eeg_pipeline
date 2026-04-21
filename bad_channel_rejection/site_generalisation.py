"""
bad_channel_rejection/site_generalisation.py

Leave-one-visit-out evaluation to quantify cross-visit generalisation.
Trains on 3 visits, tests on the 4th. Rotates through all 4 visits.

Usage:
    python -m bad_channel_rejection.site_generalisation
"""

import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

from bad_channel_rejection.dataset import (
    load_bcr_data,
    add_missingness_flags,
    impute_and_encode_channels,
)
from bad_channel_rejection.features import FeaturePreprocessor

warnings.filterwarnings('ignore')

BCR_CSV       = "data/raw/Bad_channels_for_ML.csv"
BAD_THRESHOLD = 2

XGB_PARAMS = dict(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric='aucpr', early_stopping_rounds=30,
    tree_method='hist', device='cpu', verbosity=0,
)


def run_leave_one_visit_out():
    # 1. Load raw data
    df = load_bcr_data(BCR_CSV)
    df = add_missingness_flags(df)

    # 2. Load feature column list (saved by dataset.py on Day 8)
    with open("configs/feature_cols.json") as f:
        feature_cols = json.load(f)

    # 3. Unpack tuple — impute_and_encode_channels returns (df, imputer)
    df, _ = impute_and_encode_channels(df, feature_cols)

    # 4. Build labels now that df is a clean DataFrame
    y = (df['Bad (score)'] >= BAD_THRESHOLD).astype(int).values

    visits = sorted(df['visit'].unique())
    print(f"\nVisits found: {visits}")
    print(f"\nSubjects per visit:")
    for v in visits:
        n_subj = df.loc[df['visit'] == v, 'subject_id'].nunique()
        n_rows  = (df['visit'] == v).sum()
        bad_r   = y[df['visit'].values == v].mean()
        print(f"  Visit {v}: {n_subj} subjects, {n_rows} rows, "
              f"bad_rate={bad_r:.3f}")
    print()

    results = []
    for test_visit in visits:
        train_mask = (df['visit'] != test_visit).values
        test_mask  = (df['visit'] == test_visit).values

        X_train_df = df.loc[train_mask, feature_cols]
        X_test_df  = df.loc[test_mask,  feature_cols]
        y_train    = y[train_mask]
        y_test     = y[test_mask]

        spw_train = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

        # Fit preprocessor on train only — no leakage
        prep    = FeaturePreprocessor()
        X_train = prep.fit_transform(X_train_df)
        X_test  = prep.transform(X_test_df)

        model = XGBClassifier(scale_pos_weight=spw_train, **XGB_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train)],
            verbose=False,
        )

        y_prob = model.predict_proba(X_test)[:, 1]
        auprc  = average_precision_score(y_test, y_prob)
        auroc  = roc_auc_score(y_test, y_prob)

        results.append({
            'test_visit':    int(test_visit),
            'n_train':       int(train_mask.sum()),
            'n_test':        int(test_mask.sum()),
            'bad_rate_test': round(float(y_test.mean()), 4),
            'auprc':         round(auprc, 4),
            'auroc':         round(auroc, 4),
        })
        print(f"Visit {test_visit} held out → AUPRC={auprc:.4f}  "
              f"AUROC={auroc:.4f}  (n_test={test_mask.sum()}, "
              f"bad_rate={y_test.mean():.3f})")

    # Summary
    auprc_vals = [r['auprc'] for r in results]
    print()
    print(f"Mean AUPRC (LOVO):  {np.mean(auprc_vals):.4f} ± "
          f"{np.std(auprc_vals):.4f}")
    print(f"GroupKFold AUPRC:   0.518 ± 0.085  (subject-stratified, reference)")
    print(f"Generalisation gap: {0.518 - np.mean(auprc_vals):+.4f}  "
          f"(GroupKFold − LOVO)")

    # Save JSON for report
    out = {
        'per_visit':             results,
        'mean_auprc':            round(np.mean(auprc_vals), 4),
        'std_auprc':             round(np.std(auprc_vals), 4),
        'groupkfold_ref_auprc':  0.518,
        'generalisation_gap':    round(0.518 - np.mean(auprc_vals), 4),
    }
    Path("results").mkdir(exist_ok=True)
    with open("results/site_generalisation.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved: results/site_generalisation.json")
    return out


if __name__ == "__main__":
    run_leave_one_visit_out()
