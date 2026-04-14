import numpy as np
from scipy import signal
from typing import Optional


def bandpass_filter(
    sig: np.ndarray,
    fs: int = 256,
    low: float = 1.0,
    high: float = 40.0,
    order: int = 5,
) -> np.ndarray:
    """
    Apply a zero-phase FIR bandpass filter (1–40 Hz by default).

    Parameters
    ----------
    sig   : 1-D or 2-D array.  Shape (T,) or (n_channels, T).
    fs    : Sampling frequency in Hz.
    low   : Low-cut frequency in Hz.
    high  : High-cut frequency in Hz.
    order : Filter order.

    Returns
    -------
    Filtered array, same shape as input.
    """
    nyq = fs / 2.0
    b, a = signal.butter(order, [low / nyq, high / nyq], btype="band")
    if sig.ndim == 1:
        return signal.filtfilt(b, a, sig).astype(np.float32)
    # 2-D: filter each channel independently
    return np.stack(
        [signal.filtfilt(b, a, row) for row in sig], axis=0
    ).astype(np.float32)


def notch_filter(
    sig: np.ndarray,
    fs: int = 256,
    freq: float = 50.0,
    quality: float = 30.0,
) -> np.ndarray:
    """
    Remove a narrow power-line frequency (50 Hz default) using a notch filter.

    Parameters
    ----------
    sig     : 1-D or 2-D array.
    fs      : Sampling frequency in Hz.
    freq    : Frequency to notch out in Hz.
    quality : Q-factor; higher = narrower notch.

    Returns
    -------
    Filtered array, same shape as input.
    """
    b, a = signal.iirnotch(freq, quality, fs)
    if sig.ndim == 1:
        return signal.filtfilt(b, a, sig).astype(np.float32)
    return np.stack(
        [signal.filtfilt(b, a, row) for row in sig], axis=0
    ).astype(np.float32)


def zscore_normalize(
    sig: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Z-score normalise a segment per channel.

    For 1-D input  (T,)          : single-channel normalisation.
    For 2-D input  (n_ch, T)     : per-channel normalisation.
    For 3-D input  (N, n_ch, T)  : per-segment per-channel normalisation.

    Parameters
    ----------
    sig : Array of shape (T,), (n_ch, T), or (N, n_ch, T).
    eps : Small constant to avoid division by zero.

    Returns
    -------
    Normalised array, same shape, float32.
    """
    sig = sig.astype(np.float32)
    if sig.ndim == 1:
        mu, std = sig.mean(), sig.std()
        return (sig - mu) / (std + eps)
    elif sig.ndim == 2:
        mu  = sig.mean(axis=-1, keepdims=True)
        std = sig.std(axis=-1,  keepdims=True)
        return (sig - mu) / (std + eps)
    elif sig.ndim == 3:
        mu  = sig.mean(axis=-1, keepdims=True)
        std = sig.std(axis=-1,  keepdims=True)
        return (sig - mu) / (std + eps)
    raise ValueError(f"zscore_normalize: unsupported ndim={sig.ndim}")


def segment_signal(
    sig: np.ndarray,
    window: int = 512,
    overlap: float = 0.5,
    normalize: bool = True,
) -> np.ndarray:
    """
    Slice a 1-D signal into overlapping windows.

    Parameters
    ----------
    sig       : 1-D array of shape (T,).
    window    : Window length in samples (default 512 = 2 s at 256 Hz).
    overlap   : Fraction overlap between consecutive windows [0, 1).
    normalize : If True, z-score each window before returning.

    Returns
    -------
    Array of shape (N_windows, window), float32.
    """
    step = int(window * (1.0 - overlap))
    starts = range(0, len(sig) - window + 1, step)
    segments = np.stack([sig[s : s + window] for s in starts], axis=0).astype(np.float32)
    if normalize:
        segments = zscore_normalize(segments[..., np.newaxis, :]).squeeze(1)
    return segments
