# BCR Triage — Two-Tier Human Review Pipeline

## Thresholds (calibrated on OOF predictions)

| Threshold | Value | Meaning |
|-----------|-------|---------|
| `thr_low`  | **0.0149** | P(bad) ≤ this → **auto-accept** (recall≥95% guarantee) |
| `thr_high` | **0.9336** | P(bad) ≥ this → **auto-interpolate** (precision≥80%) |

## Tier statistics (over OOF)

| Tier | n channels | % of all | n bads in tier | bad-rate in tier | % of all bads |
|------|-----------:|---------:|---------------:|-----------------:|--------------:|
| 🟢 Auto-accept | 12,681 | 67.1% | 37 | 0.3% | 5.1% |
| 🟡 Review | 5,959 | 31.5% | 487 | 8.2% | 66.5% |
| 🔴 Auto-interpolate | 260 | 1.4% | 208 | 80.0% | 28.4% |

**Totals:** 18,900 channels — 732 bads (3.9% bad rate).

## Operational interpretation

- The **auto-accept** tier (67% of all channels) misses 5.1% of all bad channels — within the 5% miss budget.
- The **auto-interpolate** tier (1.4% of all channels) is 80%-pure — i.e. of every 100 channels auto-flagged, ~80 are real bads.
- The **review** tier (31.5% of all channels) is what an expert needs to look at. It contains 67% of all bad channels at a 8.2% bad-rate.

## Production usage

```python
from bad_channel_rejection.triage import (
    load_thresholds, predict_and_triage,
)
import joblib

thr_low, thr_high = load_thresholds()                      # results/triage_thresholds.json
model = joblib.load("results/best_model.pkl")              # Stage 3 winner
probs, tiers = predict_and_triage(model, X_session, thr_low, thr_high)

# auto-interpolate the auto_bad ones
from bad_channel_rejection.interpolation import interpolate_bad_channels
bad_idx = np.where(tiers == "auto_bad")[0].tolist()
eeg_repaired = interpolate_bad_channels(eeg, bad_idx, ch_names)

# queue the review tier for an expert
review_idx = np.where(tiers == "review")[0].tolist()
```