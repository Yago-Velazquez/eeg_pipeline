"""
bad_channel_rejection/interpolation.py

Bad channel interpolation for EEG data.

Two strategies:
  1. spherical_spline_interpolation — MNE raw.interpolate_bads()
  2. zero_out_channel                — fallback when no montage available

This module is independent of any denoiser. It validates repair quality
using synthetic bad channels with known ground truth.
"""

from __future__ import annotations

import numpy as np
import mne

from .logging_config import setup_logging

logger = setup_logging(__name__)

DEFAULT_MONTAGE = "standard_1005"


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
    eeg : np.ndarray (n_channels, n_samples) in µV
    bad_indices : list[int]
    ch_names : list[str]
        Channel names matching eeg rows (standard 10-05 names).
    sfreq : float
    montage_name : str

    Returns
    -------
    np.ndarray with bad channels replaced by interpolated values.
    """
    if eeg.ndim != 2:
        raise ValueError(
            f"eeg must be 2-D (n_channels, n_samples), got {eeg.shape}"
        )
    if len(ch_names) != eeg.shape[0]:
        raise ValueError(
            f"ch_names length ({len(ch_names)}) != eeg rows ({eeg.shape[0]})"
        )
    if not bad_indices:
        logger.warning(
            "spherical_spline_interpolation called with empty bad_indices"
        )
        return eeg.copy()

    for idx in bad_indices:
        if idx < 0 or idx >= eeg.shape[0]:
            raise ValueError(
                f"bad_index {idx} out of range for {eeg.shape[0]} channels"
            )

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(eeg * 1e-6, info, verbose=False)
    montage = mne.channels.make_standard_montage(montage_name)
    raw.set_montage(
        montage, match_case=False, on_missing="warn", verbose=False
    )
    raw.info["bads"] = [ch_names[i] for i in bad_indices]
    logger.debug(f"Interpolating: {raw.info['bads']}")
    raw.interpolate_bads(reset_bads=True, verbose=False)
    return raw.get_data() * 1e6


def zero_out_channel(
    eeg: np.ndarray, bad_indices: list[int]
) -> np.ndarray:
    """Fallback: replace bad channels with zeros."""
    if eeg.ndim != 2:
        raise ValueError(
            f"eeg must be 2-D (n_channels, n_samples), got {eeg.shape}"
        )
    result = eeg.copy()
    for idx in bad_indices:
        result[idx, :] = 0.0
    return result


def interpolate_bad_channels(
    eeg: np.ndarray,
    bad_indices: list[int],
    ch_names: list[str] | None = None,
    sfreq: float = 256.0,
    montage_name: str = DEFAULT_MONTAGE,
    method: str = "spherical",
) -> np.ndarray:
    """Router for the two interpolation methods."""
    if method == "spherical":
        if ch_names is None:
            raise ValueError("ch_names required for method='spherical'")
        return spherical_spline_interpolation(
            eeg, bad_indices, ch_names, sfreq, montage_name
        )
    if method == "zero":
        return zero_out_channel(eeg, bad_indices)
    raise ValueError(
        f"Unknown method: {method!r}. Choose 'spherical' or 'zero'."
    )
