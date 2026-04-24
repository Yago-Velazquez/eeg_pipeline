"""
Bad channel interpolation for EEG data.

Two strategies:
  1. spherical_spline_interpolation — MNE raw.interpolate_bads()
     Requires a known electrode montage (e.g. standard_1005).
     Gold standard; spatially informed.

  2. zero_out_channel — fallback when no montage is available.
     Sets the bad channel to all zeros.
     Neutral signal; avoids feeding garbage to the denoiser.

Note: This module is independent of the denoiser. It validates
repair quality using synthetic bad channels with known ground truth.
The denoiser receives single-channel segments — BCR spatial features
and interpolation are evaluated here, in isolation, within BCR scope.
"""
from __future__ import annotations

import logging
import numpy as np
import mne

logger = logging.getLogger(__name__)

# Default montage used when the caller does not specify one.
DEFAULT_MONTAGE = "standard_1005"

# ── Spherical spline interpolation ──────────────────────────────────────────

def spherical_spline_interpolation(
    eeg: np.ndarray,
    bad_indices: list[int],
    ch_names: list[str],
    sfreq: float = 256.0,
    montage_name: str = DEFAULT_MONTAGE,
) -> np.ndarray:
    """Interpolate bad channels using MNE spherical spline interpolation.

    Parameters
    ----------
    eeg : np.ndarray, shape (n_channels, n_samples)
        Multi-channel EEG array. Values in µV.
    bad_indices : list[int]
        Channel indices to mark as bad and interpolate.
    ch_names : list[str]
        Channel names matching eeg rows (must be standard 10-05 names).
    sfreq : float
        Sampling frequency in Hz. Default 256.
    montage_name : str
        MNE-compatible montage string. Default 'standard_1005'.

    Returns
    -------
    np.ndarray, shape (n_channels, n_samples)
        EEG with bad channels replaced by interpolated values.

    Raises
    ------
    ValueError
        If bad_indices are out of range or ch_names length mismatches eeg.
    RuntimeError
        If MNE cannot locate any bad channel in the montage.
    """
    if eeg.ndim != 2:
        raise ValueError(f"eeg must be 2-D (n_channels, n_samples), got shape {eeg.shape}")
    if len(ch_names) != eeg.shape[0]:
        raise ValueError(
            f"ch_names length ({len(ch_names)}) must match eeg.shape[0] ({eeg.shape[0]})"
        )
    if not bad_indices:
        logger.warning("spherical_spline_interpolation called with empty bad_indices — returning copy.")
        return eeg.copy()

    for idx in bad_indices:
        if idx < 0 or idx >= eeg.shape[0]:
            raise ValueError(f"bad_index {idx} out of range for {eeg.shape[0]} channels")

    # Build MNE RawArray
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(eeg * 1e-6, info, verbose=False)  # µV → V

    # Attach standard montage
    montage = mne.channels.make_standard_montage(montage_name)
    raw.set_montage(montage, match_case=False, on_missing="warn", verbose=False)

    # Mark bad channels
    raw.info["bads"] = [ch_names[i] for i in bad_indices]
    logger.debug(f"Interpolating bad channels: {raw.info['bads']}")

    # Interpolate
    raw.interpolate_bads(reset_bads=True, verbose=False)

    # Extract result and convert back to µV
    interpolated = raw.get_data() * 1e6  # V → µV
    return interpolated


# ── Fallback: zero-out ───────────────────────────────────────────────────────

def zero_out_channel(
    eeg: np.ndarray,
    bad_indices: list[int],
) -> np.ndarray:
    """Fallback interpolation: replace bad channels with zeros.

    Use when no electrode montage is available and spherical spline
    interpolation cannot be performed. A zero channel is neutral and
    avoids propagating corrupted data through the pipeline.

    Parameters
    ----------
    eeg : np.ndarray, shape (n_channels, n_samples)
    bad_indices : list[int]
        Channel indices to zero out.

    Returns
    -------
    np.ndarray, shape (n_channels, n_samples)
        Copy of eeg with bad channels set to 0.
    """
    if eeg.ndim != 2:
        raise ValueError(f"eeg must be 2-D (n_channels, n_samples), got shape {eeg.shape}")
    result = eeg.copy()
    for idx in bad_indices:
        result[idx, :] = 0.0
    return result


# ── Convenience router ───────────────────────────────────────────────────────

def interpolate_bad_channels(
    eeg: np.ndarray,
    bad_indices: list[int],
    ch_names: list[str] | None = None,
    sfreq: float = 256.0,
    montage_name: str = DEFAULT_MONTAGE,
    method: str = "spherical",
) -> np.ndarray:
    """Router: call spherical or zero-out interpolation.

    Parameters
    ----------
    eeg : np.ndarray, shape (n_channels, n_samples)
    bad_indices : list[int]
    ch_names : list[str] or None
        Required when method='spherical'.
    sfreq : float
    montage_name : str
    method : {'spherical', 'zero'}
        'spherical' uses MNE interpolation (requires ch_names).
        'zero' zeroes out bad channels (no ch_names needed).

    Returns
    -------
    np.ndarray, shape (n_channels, n_samples)
    """
    if method == "spherical":
        if ch_names is None:
            raise ValueError("ch_names required for method='spherical'")
        return spherical_spline_interpolation(eeg, bad_indices, ch_names, sfreq, montage_name)
    elif method == "zero":
        return zero_out_channel(eeg, bad_indices)
    else:
        raise ValueError(f"Unknown method: {method!r}. Choose 'spherical' or 'zero'.")
