import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# Load OOF predictions and raw data (in original row order)
y_true = np.load("results/oof_y_true_thresh2.npy")
y_prob = np.load("results/oof_y_prob_thresh2.npy")

# Use your chosen threshold from Task 2 (update CHOSEN_THRESHOLD)
with open("configs/pipeline_config.yaml") as f:
    config = yaml.safe_load(f)
CHOSEN_THRESHOLD = config['bcr']['decision_threshold']

y_pred = (y_prob >= CHOSEN_THRESHOLD).astype(int)

# Load the CSV to get channel labels and session info
df = pd.read_csv("data/raw/Bad_channels_for_ML.csv")
df['y_true'] = y_true
df['y_prob'] = y_prob
df['y_pred'] = y_pred

# False negatives: y_true=1 but y_pred=0 (bad channel, model says clean)
fn_mask = (df['y_true']==1) & (df['y_pred']==0)
tp_mask = (df['y_true']==1) & (df['y_pred']==1)

print(f"Total bad channels:    {(df['y_true']==1).sum()}")
print(f"True positives:        {tp_mask.sum()} ({tp_mask.sum()/(df['y_true']==1).sum()*100:.1f}%)")
print(f"False negatives:       {fn_mask.sum()} ({fn_mask.sum()/(df['y_true']==1).sum()*100:.1f}%)")
print()

# Which channels are most missed?
print("Top 10 channels by FN count:")
fn_by_channel = df[fn_mask].groupby('Channel labels').size().sort_values(ascending=False).head(10)
total_by_channel = df[df['y_true']==1].groupby('Channel labels').size()
print(pd.DataFrame({'FN_count': fn_by_channel,
                    'total_bad': total_by_channel,
                    'FN_rate': (fn_by_channel / total_by_channel).round(3)}).to_string())
print()

# Visit-level breakdown
df_visits = df.copy()
# Parse visit from Session column: format is {visit}-{subject}-{site}
df_visits['visit'] = df_visits['Session'].str.split('-').str[0].astype(int)
print("FN rate by visit:")
for v in sorted(df_visits['visit'].unique()):
    mask_v = df_visits['visit']==v
    bad_v  = (df_visits.loc[mask_v, 'y_true']==1).sum()
    fn_v   = ((df_visits['visit']==v) & fn_mask).sum()
    print(f"  Visit {v}: FN={fn_v}/{bad_v}  rate={fn_v/max(bad_v,1):.3f}")
print()

# Low-confidence predictions among FN: model was ambiguous
fn_df = df[fn_mask][['Channel labels', 'Session', 'y_prob']].sort_values('y_prob', ascending=False)
print("Top 10 hardest FNs (highest model confidence among missed bads):")
print(fn_df.head(10).to_string(index=False))

