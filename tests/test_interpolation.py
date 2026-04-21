"""
Unit tests for bad_channel_rejection/interpolation.py

Strategy
--------
1. Build a synthetic multi-channel EEG using the standard 10-05 subset.
2. Store the clean copy.
3. Corrupt one channel with large-amplitude noise (synthetic 'bad').
4. Interpolate with both methods.
5. Assert spherical spline MSE < corrupted MSE  (it reconstructed something better).
6. Assert zero-out MSE < corrupted MSE          (zeroing beats corruption).
7. Assert spherical spline MSE < zero-out MSE  (spherical should be better).
8. Test edge cases: empty bad list, out-of-range index, wrong dimensions.
"""

import pytest
import numpy as np
from bad_channel_rejection.interpolation import (
    spherical_spline_interpolation,
    zero_out_channel,
    interpolate_bad_channels,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

# Use a small but geographically distributed set of 10-05 channels.
# Having channels from multiple scalp regions makes spline interpolation meaningful.
CHANNELS = [
    "Fp1", "Fp2",
    "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "Oz", "O2",
]
N_CH = len(CHANNELS)   # 20
N_SAMPLES = 512
SFREQ = 256.0
BAD_IDX = 8  # C3 — centrally located, well surrounded by neighbours
RNG = np.random.default_rng(seed=42)


def make_synthetic_eeg() -> tuple[np.ndarray, np.ndarray]:
    """Return (clean_eeg, corrupted_eeg) shape (N_CH, N_SAMPLES), amplitude µV."""
    # Realistic alpha-band sinusoid base + low-level noise
    t = np.linspace(0, N_SAMPLES / SFREQ, N_SAMPLES)
    base = np.sin(2 * np.pi * 10 * t) * 20  # 10 Hz, 20 µV amplitude
    clean = np.tile(base, (N_CH, 1)) + RNG.normal(0, 3, (N_CH, N_SAMPLES))

    # Corrupt BAD_IDX channel with large noise burst
    corrupted = clean.copy()
    corrupted[BAD_IDX] = RNG.normal(0, 200, N_SAMPLES)  # 200 µV std — clearly bad
    return clean, corrupted


# ── Core MSE tests ────────────────────────────────────────────────────────────

class TestSphericaSplineInterpolation:

    def test_reduces_mse_vs_corrupted(self):
        """Interpolated channel MSE must be lower than corrupted channel MSE."""
        clean, corrupted = make_synthetic_eeg()
        interpolated = spherical_spline_interpolation(
            corrupted, [BAD_IDX], CHANNELS, sfreq=SFREQ
        )
        mse_corrupted = np.mean((corrupted[BAD_IDX] - clean[BAD_IDX]) ** 2)
        mse_interp = np.mean((interpolated[BAD_IDX] - clean[BAD_IDX]) ** 2)
        print(f"\n  MSE corrupted : {mse_corrupted:.2f} µV²")
        print(f"  MSE interp    : {mse_interp:.2f} µV²")
        print(f"  Reduction     : {(1 - mse_interp/mse_corrupted)*100:.1f}%")
        assert mse_interp < mse_corrupted, (
            f"Interpolation did not reduce MSE: {mse_interp:.2f} >= {mse_corrupted:.2f}"
        )

    def test_other_channels_unchanged(self):
        """Channels NOT in bad_indices must be bit-for-bit identical after interpolation."""
        clean, corrupted = make_synthetic_eeg()
        interpolated = spherical_spline_interpolation(
            corrupted, [BAD_IDX], CHANNELS, sfreq=SFREQ
        )
        for i in range(N_CH):
            if i == BAD_IDX:
                continue
            np.testing.assert_allclose(
                interpolated[i], corrupted[i],
                rtol=1e-10, atol=1e-10,
                err_msg=f"Channel {i} ({CHANNELS[i]}) was modified beyond float64 precision"
            )

    def test_output_shape_unchanged(self):
        """Output shape must match input shape."""
        _, corrupted = make_synthetic_eeg()
        out = spherical_spline_interpolation(corrupted, [BAD_IDX], CHANNELS, sfreq=SFREQ)
        assert out.shape == corrupted.shape

    def test_empty_bad_indices_returns_copy(self):
        """Empty bad_indices must return an unchanged copy without raising."""
        _, corrupted = make_synthetic_eeg()
        out = spherical_spline_interpolation(corrupted, [], CHANNELS, sfreq=SFREQ)
        np.testing.assert_array_equal(out, corrupted)

    def test_raises_on_wrong_ch_names_length(self):
        """Must raise ValueError when ch_names length mismatches eeg rows."""
        _, corrupted = make_synthetic_eeg()
        with pytest.raises(ValueError, match="ch_names length"):
            spherical_spline_interpolation(corrupted, [BAD_IDX], CHANNELS[:5], sfreq=SFREQ)

    def test_raises_on_out_of_range_index(self):
        """Out-of-range bad index must raise ValueError."""
        _, corrupted = make_synthetic_eeg()
        with pytest.raises(ValueError, match="out of range"):
            spherical_spline_interpolation(corrupted, [N_CH + 5], CHANNELS, sfreq=SFREQ)

    def test_raises_on_1d_input(self):
        """1-D input must raise ValueError."""
        _, corrupted = make_synthetic_eeg()
        with pytest.raises(ValueError, match="2-D"):
            spherical_spline_interpolation(corrupted[0], [0], [CHANNELS[0]], sfreq=SFREQ)


class TestZeroOutChannel:

    def test_bad_channel_is_zero(self):
        """After zero_out, the bad channel must be all zeros."""
        _, corrupted = make_synthetic_eeg()
        out = zero_out_channel(corrupted, [BAD_IDX])
        np.testing.assert_array_equal(out[BAD_IDX], np.zeros(N_SAMPLES))

    def test_reduces_mse_vs_corrupted(self):
        """Zero-out MSE must be lower than corrupted MSE when noise >> signal."""
        clean, corrupted = make_synthetic_eeg()
        out = zero_out_channel(corrupted, [BAD_IDX])
        mse_corrupted = np.mean((corrupted[BAD_IDX] - clean[BAD_IDX]) ** 2)
        mse_zero = np.mean((out[BAD_IDX] - clean[BAD_IDX]) ** 2)
        # With 200 µV noise std, zero is almost certainly better
        print(f"\n  MSE corrupted : {mse_corrupted:.2f} µV²")
        print(f"  MSE zero-out  : {mse_zero:.2f} µV²")
        assert mse_zero < mse_corrupted

    def test_other_channels_unchanged(self):
        """Other channels must be unchanged."""
        _, corrupted = make_synthetic_eeg()
        out = zero_out_channel(corrupted, [BAD_IDX])
        for i in range(N_CH):
            if i == BAD_IDX:
                continue
            np.testing.assert_array_equal(out[i], corrupted[i])

    def test_does_not_modify_in_place(self):
        """zero_out must return a copy, not modify the input array."""
        _, corrupted = make_synthetic_eeg()
        orig_val = corrupted[BAD_IDX].copy()
        zero_out_channel(corrupted, [BAD_IDX])
        np.testing.assert_array_equal(corrupted[BAD_IDX], orig_val)

    def test_raises_on_1d_input(self):
        _, corrupted = make_synthetic_eeg()
        with pytest.raises(ValueError, match="2-D"):
            zero_out_channel(corrupted[0], [0])


class TestInterpolateRouter:

    def test_spherical_route(self):
        _, corrupted = make_synthetic_eeg()
        out = interpolate_bad_channels(
            corrupted, [BAD_IDX], ch_names=CHANNELS, sfreq=SFREQ, method="spherical"
        )
        assert out.shape == corrupted.shape

    def test_zero_route(self):
        _, corrupted = make_synthetic_eeg()
        out = interpolate_bad_channels(corrupted, [BAD_IDX], method="zero")
        np.testing.assert_array_equal(out[BAD_IDX], np.zeros(N_SAMPLES))

    def test_spherical_requires_ch_names(self):
        _, corrupted = make_synthetic_eeg()
        with pytest.raises(ValueError, match="ch_names required"):
            interpolate_bad_channels(corrupted, [BAD_IDX], ch_names=None, method="spherical")

    def test_unknown_method_raises(self):
        _, corrupted = make_synthetic_eeg()
        with pytest.raises(ValueError, match="Unknown method"):
            interpolate_bad_channels(corrupted, [BAD_IDX], method="magic")
