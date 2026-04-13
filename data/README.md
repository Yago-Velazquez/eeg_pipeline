## EEGDenoiseNet

- **Path**: `data/raw/eegdenoisenet/data/`
- **Source**: https://github.com/ncclabsustech/EEGdenoiseNet
- **Paper**: Zhang et al. 2021, arXiv:2009.11662v4
- **NaNs**: 0 across all arrays

### Array details

| Array                  | Shape         | Rate   | Duration | Role                        |
|------------------------|---------------|--------|----------|-----------------------------|
| EEG_all_epochs         | (4514, 512)   | 256 Hz | 2s       | Clean ground truth          |
| EOG_all_epochs         | (3400, 512)   | 256 Hz | 2s       | Ocular artifact             |
| EMG_all_epochs         | (5598, 512)   | 512 Hz | 1s       | Muscle artifact (DO NOT USE for synthesis) |
| EEG_all_epochs_512hz   | (4514, 1024)  | 512 Hz | 2s       | Clean EEG upsampled for EMG mixing |
| EMG_all_epochs_512hz   | (5598, 1024)  | 512 Hz | 2s       | Muscle artifact for correct synthesis |

### Normalisation status
Data is NOT pre-normalised despite paper claim of standardisation.
Confirmed by cell 3 output (02_denoiser_eda.ipynb):
- EEG_all_epochs mean epoch std    : 218.83 µV, range 63.23 – 502.78 µV
- EEG_all_epochs_512hz mean std    : ~228.55 µV (same data, upsampled)
- EMG_all_epochs_512hz mean std    : ~40,618 µV (~178x larger than EEG)
- Data appears normalised          : False

z-score must be applied per-epoch in dataset.py before feeding to model.
Per paper Equation 4: divide both noisy input AND clean target by std(noisy).
Save std value per epoch to recover real amplitude after inference.

### EOG synthesis — confirmed correct
mix_snr() with per-epoch RMS power matching at SNR=0 dB is correct for EOG.
Both EEG and EOG are at 256 Hz, same duration — no preprocessing needed.

Confirmed visually (denoiser_eog_pairs.png):
- Epoch 3839: moderate slow-wave contamination, amplitude ~±500 µV
- Epoch 2307: strong EOG burst, amplitude reaches ±1000 µV
- Epoch 1389: subtle contamination, noisy signal slightly larger than clean

Confirmed in PSD (denoiser_psd_comparison.png):
- EOG elevated ~1 order of magnitude below 5 Hz vs clean EEG
- EOG and clean overlap above 10 Hz — low-frequency contamination only
- Hard cutoff at 80 Hz preserved (pre-filtered in source data, paper Section 2.1)
- Alpha peak ~10 Hz visible in clean EEG — genuine resting-state data confirmed

### EMG synthesis — confirmed correct using 512 Hz files
The 256 Hz EMG_all_epochs.npy must NOT be used for synthesis — epoch length
mismatch (1s EMG vs 2s EEG) makes all synthesis approaches produce incorrect
spectral profiles. This was exhaustively verified in Day 4 EDA.

Solution: authors published pre-upsampled 512 Hz arrays which resolve the
mismatch entirely. Both EEG_all_epochs_512hz and EMG_all_epochs_512hz are
1024 samples at 512 Hz = 2 seconds. Direct mixing is valid.

Correct synthesis procedure (confirmed in Day 4, to be implemented Day 17):
1. Load EEG_all_epochs_512hz (4514, 1024) and EMG_all_epochs_512hz (5598, 1024)
2. Mix directly with mix_snr() — same shape, same rate, same duration
3. Downsample noisy result 512 Hz → 256 Hz via resample_poly(up=1, down=2)
4. Use EEG_all_epochs (256 Hz) as clean training target
5. Training pair: (512-sample noisy, 512-sample clean) = 2s at 256 Hz
Full 2-second epochs preserved — identical duration to EOG training pairs.

Confirmed visually (denoiser_emg_pairs.png):
- No boundary spikes or discontinuities
- Noisy signal consistently larger amplitude than clean across all epochs
- High-frequency jitter clearly visible — denser, more jagged texture than clean

Confirmed in PSD (denoiser_psd_comparison.png):
- EMG curve elevated above clean across full 0–128 Hz range
- EMG does NOT drop off at 80 Hz — stays elevated to 128 Hz
- Separation between EMG and clean/EOG larger above 40 Hz than below
- Confirms genuine broadband high-frequency muscle artifact character
- Flat profile above 80 Hz is a Nyquist constraint at 256 Hz, not a bug

### Single-channel constraint
EEGDenoiseNet has no electrode montage. BCR requires spatial features across
multiple channels (spatial correlation, impedance across montage, multi-channel
variance comparisons). These cannot be computed from single-channel data.
BCR and denoiser are therefore validated independently on separate datasets.

Joint evaluation candidate for future work: PhysioNet EEG-MMI (109 subjects,
64 ch, 10-20 montage). Documented in report/future_work_draft.md.# BCR Dataset — Audit Findings

## Basic statistics
- Total rows: 18,900 (channels × sessions)
- Sessions: 50 (encoded as session_group-subject_id-visit_id in Session column)
- Unique subjects: 43 (GroupKFold groups)
- Session groups: 4 (values 1, 2, 3, 4 — NOT the same as rater sites)
- Rater sites: 4 (sites 2, 3, 4a, 4b — encoded in Bad (site X) columns)
- Channel labels: 126 unique channels
- Features (pre-computed): 156 signal features + impedance columns
- Tasks: 5 (3-EO, 1-EO, 4-EC, 2-EC, 0-AR)

## Class imbalance
- Bad rate (score≥2): 3.9% → 732 bad / 18,168 good
- scale_pos_weight for XGBoost: 24.8
- Random AUPRC baseline: 0.039 ← USE THIS, not AUROC
- AUROC is misleading here (naive model scores 0.95 by predicting all GOOD)
- Borderline zone (score=1): 1,692 rows (8.9%) — treated as GOOD under
  threshold≥2, subject to threshold sensitivity experiment on Day 10

## Session group structure
- Session ID format: [session_group]-[subject_id]-[visit_id]
- 6 subjects appear in multiple session groups (cross-group subjects):
  subject 103 (3 groups), subjects 107, 123, 190, 247, 265 (2 groups each)
- GroupKFold MUST group by subject_id (43 values), NOT Session
- Bad rate per session group:
  - Session group 1: 3.3%
  - Session group 2: 4.9% (highest)
  - Session group 3: 3.7%
  - Session group 4: 3.3%

## Task analysis
- Bad rate varies by task — task type is a predictive feature:
  - 3-EO (Eyes Open, group 3): 5.3% — highest
  - 1-EO (Eyes Open, group 1): 4.8%
  - 4-EC (Eyes Closed, group 4): 4.1%
  - 2-EC (Eyes Closed, group 2): 3.5%
  - 0-AR (Artifact Rejection):  2.7% — lowest
- Eyes Open consistently worse than Eyes Closed — more movement,
  blinks, and muscle activity degrade electrode contact

## CRITICAL: Rater agreement (inter-site)
- Rater sites are 2, 3, 4a, 4b — independent bad channel judgements
- Full correlation matrix:
  - Site 2  ↔ Site 3:  r = 0.133
  - Site 2  ↔ Site 4a: r = 0.292
  - Site 2  ↔ Site 4b: r = 0.250
  - Site 3  ↔ Site 4a: r = 0.130
  - Site 3  ↔ Site 4b: r = 0.078
  - Site 4a ↔ Site 4b: r = 0.420 (best agreement — same institution)
- Site 3 mean correlation with all other sites: ~0.11
- This is label noise, not a model failure.
- Interpretation: AUPRC < 0.70 does not indicate model failure on this dataset.
- Overall agreement is low even between best pair (4a↔4b = 0.420) —
  this is a hard ceiling on achievable AUPRC for the entire dataset
- Planned ablation: Day 11 — exclude site 3 labels from training
  and measure AUPRC delta

## Impedance missingness
- 31.3% of rows have no impedance reading (17 sessions)
- Feature engineered: `impedance_missing` binary column (added BEFORE imputation)
- Missingness is entirely site-structured — not random:
  - Session group 1: 0.0% missing
  - Session group 2: 0.0% missing
  - Session group 3: 78.3% missing ← site 3 had no impedance protocol
  - Session group 4: 0.0% missing
- impedance_missing is a meaningful feature, not just a gap to fill

## Feature groups
- Total: 156 signal features across 13 groups × 7 statistics each
  (mean, std, median, Q1, Q3, min, max)
- Groups: Standard deviation, Low gamma/high gamma ratio,
  Global correlation, Signal-wide residuals, Window-specific residuals,
  Window-specific independence, Signal-wide PCA, Window-specific PCA,
  Signal-wide ICA, Signal-wide ICA50, Kurtosis, Median frequency,
  Correlation with neighbors, + spatial/frequency variants
- Plus: Impedance (start), Impedance (end), impedance_missing (3 features)
- Note: 27 feature pairs with r > 0.99 expected — to be deduplicated in Week 2

## Top offender channels
- Highest bad rate (confirmed from EDA):
  - T7:  30.7%
  - T8:  30.0%
  - AF7: 22.7%
  - M1:  18.0%
  - M2:  15.3%
- All are peripheral/mastoid electrodes — most susceptible to movement
  artifacts and poor skin contact
- Channel identity is a strong predictive feature for Week 2

## GroupKFold validation
- GroupKFold(5) by subject_id confirmed clean — overlap=0 on all 5 folds
- Val bad rate varies 2.7%–4.5% across folds (subject-level differences)
- Report mean ± std AUPRC across all 5 folds in Week 2, not single fold

## Pipeline note
- BCR and denoiser are validated independently on separate datasets
- BCR requires spatial features (multi-channel); EEGDenoiseNet is single-channel
- Joint pipeline evaluation is future work (candidate dataset: PhysioNet EEG-MMI)

## EEGDenoiseNet

- **Path**: `data/raw/eegdenoisenet/data/`
- **Source**: https://github.com/ncclabsustech/EEGdenoiseNet
- **Paper**: Zhang et al. 2021, arXiv:2009.11662v4
- **NaNs**: 0 across all arrays

### Array details

| Array                  | Shape         | Rate   | Duration | Role                        |
|------------------------|---------------|--------|----------|-----------------------------|
| EEG_all_epochs         | (4514, 512)   | 256 Hz | 2s       | Clean ground truth          |
| EOG_all_epochs         | (3400, 512)   | 256 Hz | 2s       | Ocular artifact             |
| EMG_all_epochs         | (5598, 512)   | 512 Hz | 1s       | Muscle artifact (DO NOT USE for synthesis) |
| EEG_all_epochs_512hz   | (4514, 1024)  | 512 Hz | 2s       | Clean EEG upsampled for EMG mixing |
| EMG_all_epochs_512hz   | (5598, 1024)  | 512 Hz | 2s       | Muscle artifact for correct synthesis |

### Normalisation status
Data is NOT pre-normalised despite paper claim of standardisation.
Confirmed by cell 3 output (02_denoiser_eda.ipynb):
- EEG_all_epochs mean epoch std    : 218.83 µV, range 63.23 – 502.78 µV
- EEG_all_epochs_512hz mean std    : ~228.55 µV (same data, upsampled)
- EMG_all_epochs_512hz mean std    : ~40,618 µV (~178x larger than EEG)
- Data appears normalised          : False

z-score must be applied per-epoch in dataset.py before feeding to model.
Per paper Equation 4: divide both noisy input AND clean target by std(noisy).
Save std value per epoch to recover real amplitude after inference.

### EOG synthesis — confirmed correct
mix_snr() with per-epoch RMS power matching at SNR=0 dB is correct for EOG.
Both EEG and EOG are at 256 Hz, same duration — no preprocessing needed.

Confirmed visually (denoiser_eog_pairs.png):
- Epoch 3839: moderate slow-wave contamination, amplitude ~±500 µV
- Epoch 2307: strong EOG burst, amplitude reaches ±1000 µV
- Epoch 1389: subtle contamination, noisy signal slightly larger than clean

Confirmed in PSD (denoiser_psd_comparison.png):
- EOG elevated ~1 order of magnitude below 5 Hz vs clean EEG
- EOG and clean overlap above 10 Hz — low-frequency contamination only
- Hard cutoff at 80 Hz preserved (pre-filtered in source data, paper Section 2.1)
- Alpha peak ~10 Hz visible in clean EEG — genuine resting-state data confirmed

### EMG synthesis — confirmed correct using 512 Hz files
The 256 Hz EMG_all_epochs.npy must NOT be used for synthesis — epoch length
mismatch (1s EMG vs 2s EEG) makes all synthesis approaches produce incorrect
spectral profiles. This was exhaustively verified in Day 4 EDA.

Solution: authors published pre-upsampled 512 Hz arrays which resolve the
mismatch entirely. Both EEG_all_epochs_512hz and EMG_all_epochs_512hz are
1024 samples at 512 Hz = 2 seconds. Direct mixing is valid.

Correct synthesis procedure (confirmed in Day 4, to be implemented Day 17):
1. Load EEG_all_epochs_512hz (4514, 1024) and EMG_all_epochs_512hz (5598, 1024)
2. Mix directly with mix_snr() — same shape, same rate, same duration
3. Downsample noisy result 512 Hz → 256 Hz via resample_poly(up=1, down=2)
4. Use EEG_all_epochs (256 Hz) as clean training target
5. Training pair: (512-sample noisy, 512-sample clean) = 2s at 256 Hz
Full 2-second epochs preserved — identical duration to EOG training pairs.

Confirmed visually (denoiser_emg_pairs.png):
- No boundary spikes or discontinuities
- Noisy signal consistently larger amplitude than clean across all epochs
- High-frequency jitter clearly visible — denser, more jagged texture than clean

Confirmed in PSD (denoiser_psd_comparison.png):
- EMG curve elevated above clean across full 0–128 Hz range
- EMG does NOT drop off at 80 Hz — stays elevated to 128 Hz
- Separation between EMG and clean/EOG larger above 40 Hz than below
- Confirms genuine broadband high-frequency muscle artifact character
- Flat profile above 80 Hz is a Nyquist constraint at 256 Hz, not a bug

### Single-channel constraint
EEGDenoiseNet has no electrode montage. BCR requires spatial features across
multiple channels (spatial correlation, impedance across montage, multi-channel
variance comparisons). These cannot be computed from single-channel data.
BCR and denoiser are therefore validated independently on separate datasets.

Joint evaluation candidate for future work: PhysioNet EEG-MMI (109 subjects,
64 ch, 10-20 montage). Documented in report/future_work_draft.md.