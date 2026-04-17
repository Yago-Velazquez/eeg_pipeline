"""
evaluate/metrics.py
Shared metrics for both pipeline components.
  BCR:      AUPRC (primary), AUROC, F1, precision, recall
  Denoiser: ΔSNR, Pearson r, RRMSE, SSIM-1D
"""
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    precision_recall_curve,
)
from scipy.stats import pearsonr
from scipy.signal import correlate


# ─────────────────────────────────────────────────────────────────
# BCR METRICS
# Random AUPRC baseline  = bad_rate = 0.039  (24.8:1 imbalance)
# AUROC is a secondary metric only — do NOT use as primary
# ─────────────────────────────────────────────────────────────────

def compute_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Area under the Precision-Recall curve (Average Precision).
    Primary BCR metric. Random baseline = 0.039 for this dataset.

    Args:
        y_true: binary labels, shape (N,)
        y_prob: predicted probabilities for positive class, shape (N,)
    Returns:
        float — AUPRC in [0, 1]
    """
    return float(average_precision_score(y_true, y_prob))


def compute_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """ROC-AUC. Secondary metric — misleading as primary given 24.8:1 imbalance."""
    return float(roc_auc_score(y_true, y_prob))


def compute_f1_at_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> float:
    """F1 score at a fixed decision threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    return float(f1_score(y_true, y_pred, zero_division=0))


def compute_precision_recall(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict:
    """Precision and recall at a fixed threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
    }


def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """
    Find the threshold that maximises F1.
    Returns (best_threshold, best_f1).
    Use this on the validation fold; apply the threshold to the test fold.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns one fewer threshold than pr points
    f1_scores = (2 * precisions[:-1] * recalls[:-1]) / (
        precisions[:-1] + recalls[:-1] + 1e-8
    )
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def bcr_full_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Convenience wrapper — returns all BCR metrics as a flat dict.
    Use this when logging to W&B: wandb.log(bcr_full_report(...))
    """
    pr = compute_precision_recall(y_true, y_prob, threshold)
    cm = confusion_matrix(y_true, (y_prob >= threshold).astype(int))
    tn, fp, fn_count, tp = cm.ravel()
    return {
        "auprc":     compute_auprc(y_true, y_prob),   # PRIMARY
        "auroc":     compute_auroc(y_true, y_prob),   # secondary
        "f1":        compute_f1_at_threshold(y_true, y_prob, threshold),
        "precision": pr["precision"],
        "recall":    pr["recall"],
        "tp": int(tp), "fp": int(fp),
        "tn": int(tn), "fn": int(fn_count),
        "threshold": threshold,
    }


# ─────────────────────────────────────────────────────────────────
# DENOISER METRICS
# Primary: snr_improvement (ΔSNR ≥ 8 dB), pearson_r (≥ 0.85)
# Standard benchmark set (matches EEGDnet, CLEnet, WGAN papers):
#   ΔSNR, CC (pearson_r), RRMSE_temporal, RRMSE_spectral
# ─────────────────────────────────────────────────────────────────

def snr_db(signal: np.ndarray, noise: np.ndarray) -> float:
    """
    SNR in dB: 10 * log10(||signal||^2 / ||noise||^2).
    signal: clean reference, shape (..., T)
    noise:  difference between noisy/denoised and clean, same shape
    """
    signal_power = np.mean(signal ** 2) + 1e-10
    noise_power  = np.mean(noise ** 2)  + 1e-10
    return float(10 * np.log10(signal_power / noise_power))


def snr_improvement(
    noisy: np.ndarray, denoised: np.ndarray, clean: np.ndarray
) -> float:
    """
    ΔSNR = SNR(denoised, clean) - SNR(noisy, clean)  [dB]
    Positive value = model improved SNR.
    MVS target: ΔSNR ≥ 8 dB
    """
    snr_before = snr_db(clean, noisy - clean)
    snr_after  = snr_db(clean, denoised - clean)
    return snr_after - snr_before


def pearson_r(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Pearson correlation between denoised and clean waveform.
    Inputs are flattened to 1D before computing.
    MVS target: r ≥ 0.85
    """
    r, _ = pearsonr(pred.ravel(), target.ravel())
    return float(r)


def rrmse(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Relative RMSE in the temporal domain = RMSE / ||target||_rms
    Scale-invariant; comparable across amplitude regimes.
    Matches RRMSEt in EEGDnet / CLEnet papers.
    """
    rmse = np.sqrt(np.mean((pred - target) ** 2))
    norm = np.sqrt(np.mean(target ** 2)) + 1e-10
    return float(rmse / norm)


def rrmse_spectral(
    pred: np.ndarray,
    target: np.ndarray,
    fs: float = 256.0,
    nperseg: int = 256,
) -> float:
    """
    Relative RMSE in the frequency domain (RRMSEf).
    RRMSEf = RMS(PSD(denoised) - PSD(clean)) / RMS(PSD(clean))

    Computed via Welch's method. Lower is better; 0 = perfect spectral
    preservation. Directly comparable to RRMSEf in EEGDnet, CLEnet,
    WGAN, and DuoCL papers.

    Args:
        pred:     denoised signal, shape (T,)
        target:   clean reference, shape (T,)
        fs:       sampling rate in Hz (default 256 — denoiser rate)
        nperseg:  Welch segment length in samples (default 256 = 1 s)
    """
    from scipy.signal import welch
    _, psd_pred   = welch(pred.ravel(),   fs=fs, nperseg=nperseg)
    _, psd_target = welch(target.ravel(), fs=fs, nperseg=nperseg)
    diff = psd_pred - psd_target
    norm = np.sqrt(np.mean(psd_target ** 2)) + 1e-10
    return float(np.sqrt(np.mean(diff ** 2)) / norm)


def ssim_1d(
    pred: np.ndarray,
    target: np.ndarray,
    window_size: int = 64,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """
    Structural Similarity Index (SSIM) adapted for 1D signals.
    Computed over non-overlapping windows; returns the mean SSIM.
    Both inputs should be normalised (z-scored) before calling.

    Args:
        pred:        denoised signal, shape (T,)
        target:      clean reference, shape (T,)
        window_size: samples per SSIM window
        k1, k2:      stability constants
    """
    pred   = pred.ravel()
    target = target.ravel()
    n_windows = len(target) // window_size
    if n_windows == 0:
        return float(np.nan)
    ssim_vals = []
    for i in range(n_windows):
        s = i * window_size
        e = s + window_size
        x, y = pred[s:e], target[s:e]
        mu_x, mu_y   = x.mean(), y.mean()
        sig_x, sig_y = x.std(), y.std()
        sig_xy        = np.mean((x - mu_x) * (y - mu_y))
        data_range    = max(target.max() - target.min(), 1e-10)
        c1 = (k1 * data_range) ** 2
        c2 = (k2 * data_range) ** 2
        num = (2 * mu_x * mu_y + c1) * (2 * sig_xy + c2)
        den = (mu_x**2 + mu_y**2 + c1) * (sig_x**2 + sig_y**2 + c2)
        ssim_vals.append(num / (den + 1e-10))
    return float(np.mean(ssim_vals))


def denoiser_full_report(
    noisy: np.ndarray,
    denoised: np.ndarray,
    clean: np.ndarray,
    fs: float = 256.0,
) -> dict:
    """
    Convenience wrapper — all denoiser metrics as a flat dict.
    Use: wandb.log(denoiser_full_report(noisy, denoised, clean))

    Metric set matches the field standard (EEGDnet, CLEnet, WGAN):
      ΔSNR, pearson_r (CC), rrmse (RRMSEt), rrmse_spectral (RRMSEf)
    """
    return {
        "delta_snr_db":    snr_improvement(noisy, denoised, clean),  # PRIMARY
        "pearson_r":       pearson_r(denoised, clean),                # PRIMARY
        "rrmse_temporal":  rrmse(denoised, clean),                    # RRMSEt
        "rrmse_spectral":  rrmse_spectral(denoised, clean, fs=fs),    # RRMSEf
        "ssim":            ssim_1d(denoised.ravel(), clean.ravel()),
        "snr_before_db":   snr_db(clean, noisy - clean),
        "snr_after_db":    snr_db(clean, denoised - clean),
    }