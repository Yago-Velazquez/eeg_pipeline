import pandas as pd

df = pd.read_csv("data/raw/Bad_channels_for_ML.csv")
df['visit'] = df['Session'].str.split('-').str[0].astype(int)
df['is_bad'] = (df['Bad (score)'] >= 2).astype(int)

print("Visit-level summary:")
summary = df.groupby('visit').agg(
    n_rows=('is_bad', 'count'),
    n_bad=('is_bad', 'sum'),
    bad_rate=('is_bad', 'mean'),
    n_subjects=('Session', lambda x: x.str.split('-').str[1].nunique()),
).reset_index()
summary['bad_rate'] = summary['bad_rate'].round(4)
print(summary.to_string(index=False))

print()
print(f"Global bad rate: {df['is_bad'].mean():.4f}")
print(f"Highest visit:   visit {summary.loc[summary['bad_rate'].idxmax(), 'visit']}"
      f"  bad_rate={summary['bad_rate'].max():.4f}")
print(f"Lowest visit:    visit {summary.loc[summary['bad_rate'].idxmin(), 'visit']}"
      f"  bad_rate={summary['bad_rate'].min():.4f}")

# Channel bad rate by visit (for T7, T8 — the worst offenders)
print()
print("T7 and T8 bad rate by visit:")
for ch in ['T7', 'T8']:
    ch_df = df[df['Channel labels'] == ch]
    for v in sorted(ch_df['visit'].unique()):
        r = ch_df.loc[ch_df['visit']==v, 'is_bad'].mean()
        print(f"  {ch} visit {v}: {r:.3f}")

