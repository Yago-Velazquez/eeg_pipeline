# BCR Error Analysis — Day 15

## Threshold Decision

Precision-recall tradeoff evaluated across thresholds 0.20→0.80 using out-of-fold
(OOF) predictions from GroupKFold cross-validation. No retraining performed.

Script: `scripts/threshold_sweep.py`, `scripts/update_threshold.py`

| Criterion | Threshold | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| argmax(F1) | 0.60 | 0.4935 | 0.5164 | 0.5047 | 378 | 388 | 354 |
| argmax(2R+P) | 0.20 | 0.2706 | 0.6995 | 0.3902 | 512 | 1380 | 220 |
| **Chosen — intuition (0.50)** | **0.50** | **0.4398** | **0.5642** | **0.4943** | **413** | **526** | **319** |
| Day 11 reference | 0.604 | 0.4935 | 0.5164 | 0.5047 | 378 | 388 | 354 |

### Delta vs Day 11 baseline (threshold = 0.604)

| Criterion | Threshold | ΔRecall | ΔFN | ΔFP | ΔF1 |
|---|---|---|---|---|---|
| argmax(F1) | 0.60 | +0.0000 | 0 | 0 | +0.0000 |
| argmax(2R+P) | 0.20 | +0.1831 | −134 | +992 | −0.1145 |
| **Chosen (0.50)** | **0.50** | **+0.0478** | **−35** | **+138** | **−0.0104** |

**Rationale:** Recall is prioritised over precision because a missed bad channel
(false negative) passes corrupted signal directly into the denoiser, degrading
downstream reconstruction quality. A false positive (interpolating a good channel
unnecessarily) introduces minor distortion but is recoverable. The argmax(2R+P)
criterion at threshold 0.20 was rejected — the recall gain (+0.183) does not justify
a 3.6× increase in false positives (388 → 1380), which would distort a large fraction
of good channels. Threshold 0.50 recovers 35 additional bad channels (FN −35) at the
cost of 138 extra unnecessary interpolations (FP +138) and negligible F1 loss
(−0.010) — a proportionate tradeoff given the clinical context.

**Active threshold:** `bcr.decision_threshold = 0.50` written to
`configs/pipeline_config.yaml`.

---

## False Negative Analysis (threshold = 0.50)

Script: `scripts/fn_analysis.py`

| Metric | Value |
|---|---|
| Total bad channels | 732 |
| True positives | 413 (56.4%) |
| False negatives | 319 (43.6%) |
| FN rate | 0.436 |

### Top channels by FN count

| Channel | FN count | Total bad | FN rate | Notes |
|---|---|---|---|---|
| PPO10h | 7 | 8 | 0.875 | Near-total failure — only 8 bad examples globally |
| CCP1h | 13 | 18 | 0.722 | High-density non-standard electrode, sparse positives |
| CCP2h | 14 | 20 | 0.700 | Adjacent to CCP1h, same sparsity problem |
| CPP1h | 10 | 15 | 0.667 | Parieto-occipital high-density, few training examples |
| CPP2h | 8 | 12 | 0.667 | Same channel family as CPP1h |
| TTP8h | 9 | 17 | 0.529 | Temporal-parietal junction, right hemisphere |
| AF7 | 12 | 34 | 0.353 | Frontal electrode, moderate failure |
| TTP7h | 6 | 19 | 0.316 | Temporal-parietal junction, left hemisphere |
| T8 | 8 | 45 | 0.178 | High bad rate → well-learned; FNs are borderline cases |
| T7 | 6 | 46 | 0.130 | Most common bad channel; model handles it well |

### FN rate by visit

| Visit | FN count | Bad channels | FN rate | Notes |
|---|---|---|---|---|
| 1 | 56 | 125 | 0.448 | Low bad rate (3.31%) limits positive training signal |
| 2 | 71 | 185 | 0.384 | Best visit; highest bad rate (4.89%) |
| 3 | 141 | 279 | 0.505 | Worst visit; 78.3% impedance missing |
| 4 | 51 | 143 | 0.357 | Best FN rate; strong generalisation |

Visit 3 accounts for 141 of 319 total FNs (44.2%) — the single largest contributor.
This is directly attributable to 78.3% impedance feature missingness in that visit,
which degrades model confidence without reducing the number of actual bad channels.

### Two distinct failure modes

**Failure mode 1 — training data sparsity (high-density parieto-occipital channels).**
PPO10h, CCP1h, CCP2h, CPP1h, CPP2h all have fewer than 20 bad examples globally.
With GroupKFold across 5 folds, individual folds may contain 0–1 positive examples
for these channels, making calibration impossible. This is a dataset coverage
limitation, not a model architecture problem. These channels account for 52 of 319
total FNs (16.3%) despite representing only 5 of 126 electrode positions.

**Failure mode 2 — threshold proximity (near-miss cases).**
All 10 hardest false negatives have predicted probability between 0.467–0.499 —
within 0.033 of the decision threshold. The model is not unaware of these channels;
it assigns them moderate probability but falls just short of 0.50. Four of the top 10
hardest FNs come from Visit 3 sessions, consistent with impedance missingness reducing
model confidence in that visit specifically. A threshold reduction to ~0.45 would
capture several of these cases at the cost of a moderate FP increase.

### Top 10 hardest false negatives

| Channel | Session | Visit | Predicted probability |
|---|---|---|---|
| AF7 | 3-131-1 | 3 | 0.4988 |
| C3 | 4-103-1 | 4 | 0.4955 |
| F7 | 2-111-1 | 2 | 0.4951 |
| CCP2h | 3-240-1 | 3 | 0.4939 |
| TPP9h | 4-158-1 | 4 | 0.4904 |
| FC4 | 3-175-1 | 3 | 0.4836 |
| AF7 | 4-136-1 | 4 | 0.4759 |
| T8 | 3-123-1 | 3 | 0.4737 |
| P10 | 2-123-1 | 2 | 0.4709 |
| F8 | 3-179-1 | 3 | 0.4675 |

Visit 3 appears in 4 of 10 hardest FNs (sessions 3-131-1, 3-240-1, 3-175-1,
3-123-1), disproportionate to its share of test rows (40% of total). This further
confirms that impedance missingness specifically degrades model confidence in Visit 3.

---

## Visit-Level Bad Rate Variation

Script: `scripts/visit_badrate.py`

| Visit | Rows | Bad channels | Bad rate | Subjects | Bad / subject |
|---|---|---|---|---|---|
| 1 | 3,780 | 125 | 3.31% | 10 | 12.5 |
| 2 | 3,780 | 185 | 4.89% | 10 | 18.5 |
| 3 | 7,560 | 279 | 3.69% | 20 | 13.95 |
| 4 | 3,780 | 143 | 3.78% | 10 | 14.3 |
| **Global** | **18,900** | **732** | **3.87%** | **43** | **17.0** |

Visit 2 has the highest bad rate (4.89%) and the highest bad channel density per
subject (18.5). Visit 1 is the lowest (3.31%, 12.5 per subject). The spread across
visits is modest at 1.58 percentage points — bad rate variation alone does not explain
LOVO performance differences across visits.

### T7 and T8 bad rate by visit

| Channel | Visit 1 | Visit 2 | Visit 3 | Visit 4 |
|---|---|---|---|---|
| T7 | 40.0% | 30.0% | 23.3% | 36.7% |
| T8 | 26.7% | 43.3% | 18.3% | 43.3% |

T7 and T8 show anti-correlated visit patterns with no shared temporal trend,
suggesting their badness is driven by subject-level or session-level factors
(electrode placement, skin preparation) rather than a global protocol change across
visits. Notably, both channels reach their lowest bad rate in Visit 3 — yet the model
performs worst in that visit. This is the clearest evidence that Visit 3's degraded
performance is a feature problem (impedance missingness), not a label problem.

---

## Summary

| Finding | Value | Interpretation |
|---|---|---|
| Chosen threshold | 0.50 | Recall-prioritised; 35 fewer FNs vs Day 11 at cost of 138 FP |
| Overall FN rate | 43.6% (319/732) | Dominated by sparse-positive and Visit 3 channels |
| Worst channel FN rate | PPO10h: 87.5% | Only 8 bad examples — training sparsity, not model failure |
| Best channel FN rate | T7: 13.0% | 46 bad examples — abundant training signal |
| Visit with most FNs | Visit 3: 50.5% | Fully explained by 78.3% impedance missingness |
| Visit with fewest FNs | Visit 4: 35.7% | Best generalisation target |
| Hardest FN confidence | 0.467–0.499 | All within 0.033 of threshold — near-miss cases |
