# Week 1 Log — Foundation

## Dataset Stats

### BCR Dataset
- Total rows: 18,900
- Subjects: 43 (decoded from Session format: site-subject-visit e.g. '1-103-1')
- Sites: 4 (session_group column)
- Bad epochs (Bad score ≥ 2): ~732 / 18,900 = 3.9%
- Class imbalance ratio: ~24.8:1 (good:bad)
- Tasks: 5 unique tasks per session

### EEGDenoiseNet
- Clean EEG: (4514, 512) — 4514 epochs × 512 samples @ 256 Hz
- EOG noise: (3400, 512) @ 256 Hz
- EEG 512 Hz: (4514, 1024) @ 512 Hz
- EMG 512 Hz: (5598, 1024) @ 512 Hz
- Single-channel only — no montage, no spatial structure
- BCR spatial features cannot be extracted from this dataset
- This is why joint pipeline evaluation is out of scope

## Scope Decision — v3
Two independent components, validated separately:
1. BCR (XGBoost) — trained on multi-channel BCR dataset
2. Denoiser (1D U-Net) — trained on EEGDenoiseNet (single-channel)
Joint pipeline evaluation: out of scope (requires multi-channel dataset
with montage + quality annotations, e.g. PhysioNet EEG-MMI — future work)

## Primary BCR Metric Rationale
AUPRC chosen over AUROC because:
- Random baseline AUPRC = bad_rate = 0.039
- With 24.8:1 class imbalance, AUROC inflates apparent performance
- AUPRC directly measures precision-recall tradeoff for the minority class

## Modules Built
- data/preprocessing.py — bandpass, notch, zscore, segment
- data/reconstruction.py — overlap-add Bartlett blend (roundtrip r = 0.9849)
- evaluate/metrics.py — auprc, auroc, f1, rrmse, snr_improvement, pearson_r, ssim_1d
- notebooks/00_colab_setup.ipynb — Colab bootstrap + integration smoke test

## Week 1 Status
- [x] Days 1–7 complete
- [x] Integration smoke test passing on Colab (T4 GPU)
- [x] v0.1-week1 tag pushed
