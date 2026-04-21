"""
Compare interpolation methods on BCR test set data.

Strategy
--------
1. Load BCR CSV.
2. For each session in the test set (visit 4 subjects, n=10):
   - Reconstruct multi-channel EEG array from the signal features
     (using 'Standard deviation (mean)' as a proxy amplitude measure
     since we don't have raw time-series — we synthesize a plausible signal).
   - Choose a channel predicted bad by the BCR model (or random if none).
   - Corrupt it with known noise.
   - Interpolate with spherical and zero-out.
   - Record MSE for each method.
3. Print summary table.

Note: The BCR dataset does not contain raw time-series — it contains
pre-computed features. We therefore SIMULATE the test using synthetic
EEG shaped by the feature-derived amplitude statistics. This is the
correct approach: we are testing the interpolation logic, not the
feature extractor.
"""

import sys, json
import numpy as np
import pandas as pd
from pathlib import Path

from bad_channel_rejection.interpolation import interpolate_bad_channels

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH = Path("data/raw/Bad_channels_for_ML.csv")
SFREQ = 256.0
N_SAMPLES = 512
N_SESSIONS = 10  # limit to keep runtime short
RNG = np.random.default_rng(seed=42)

# Standard channels available in MNE's standard_1005
STANDARD_CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "Oz", "O2",
]
N_CH = len(STANDARD_CHANNELS)
BAD_CH_IDX = 8  # C3

def synthesize_session_eeg(session_rows: pd.DataFrame) -> np.ndarray:
    """Build synthetic multi-channel EEG using per-channel amplitude from BCR features.

    We use ' Standard deviation (mean)' as the channel amplitude proxy.
    Each channel gets a sinusoidal signal scaled to its measured std dev.
    This gives us realistic amplitude variation across channels/sessions.
    """
    std_col = [c for c in session_rows.columns if 'Standard deviation (mean)' in c]
    t = np.linspace(0, N_SAMPLES / SFREQ, N_SAMPLES)
    eeg = np.zeros((N_CH, N_SAMPLES))

    for i in range(N_CH):
        # Try to get amplitude from the corresponding row
        if i < len(session_rows) and std_col:
            amp = float(session_rows.iloc[i][std_col[0]]) * 1e6  # V → µV
            if np.isnan(amp) or amp <= 0:
                amp = 20.0
        else:
            amp = 20.0
        eeg[i] = np.sin(2 * np.pi * 10 * t) * amp + RNG.normal(0, amp * 0.15, N_SAMPLES)
    return eeg


def corrupt_channel(eeg: np.ndarray, ch_idx: int, noise_std: float = 200.0) -> np.ndarray:
    corrupted = eeg.copy()
    corrupted[ch_idx] = RNG.normal(0, noise_std, eeg.shape[1])
    return corrupted


def main():
    print("Loading BCR data...")
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    # Parse session
    df[['visit', 'subject_id', 'site']] = df['Session'].str.split('-', expand=True)
    df['visit'] = df['visit'].astype(int)

    # Use visit 4 as pseudo-test set (mirrors LOVO analysis)
    test_df = df[df['visit'] == 4].copy()
    sessions = test_df['Session'].unique()[:N_SESSIONS]

    results = []
    print(f"\nRunning comparison on {len(sessions)} test sessions (visit 4)...")
    print(f"  Bad channel injected: {STANDARD_CHANNELS[BAD_CH_IDX]} (idx={BAD_CH_IDX})")
    print(f"  Noise std: 200 µV\n")
    print(f"{'Session':<20} {'MSE corrupt':>12} {'MSE spline':>12} {'MSE zero':>12} {'spline<corrupt':>14} {'spline<zero':>12}")
    print("-" * 82)

    for sess in sessions:
        rows = test_df[test_df['Session'] == sess]
        clean_eeg = synthesize_session_eeg(rows)
        corrupted = corrupt_channel(clean_eeg, BAD_CH_IDX)

        # Spherical spline
        try:
            interp_sph = interpolate_bad_channels(
                corrupted, [BAD_CH_IDX], ch_names=STANDARD_CHANNELS,
                sfreq=SFREQ, method="spherical"
            )
            mse_sph = float(np.mean((interp_sph[BAD_CH_IDX] - clean_eeg[BAD_CH_IDX]) ** 2))
            sph_ok = True
        except Exception as e:
            mse_sph = float('nan')
            sph_ok = False
            print(f"  WARNING: spherical failed for {sess}: {e}")

        # Zero-out
        interp_zero = interpolate_bad_channels(corrupted, [BAD_CH_IDX], method="zero")
        mse_zero = float(np.mean((interp_zero[BAD_CH_IDX] - clean_eeg[BAD_CH_IDX]) ** 2))

        mse_corrupt = float(np.mean((corrupted[BAD_CH_IDX] - clean_eeg[BAD_CH_IDX]) ** 2))

        sph_beats_corrupt = "✓" if sph_ok and mse_sph < mse_corrupt else "✗"
        sph_beats_zero = "✓" if sph_ok and mse_sph < mse_zero else "✗"

        results.append({
            "session": sess, "mse_corrupt": mse_corrupt,
            "mse_sph": mse_sph, "mse_zero": mse_zero
        })
        print(f"{sess:<20} {mse_corrupt:>12.1f} {mse_sph:>12.1f} {mse_zero:>12.1f} {sph_beats_corrupt:>14} {sph_beats_zero:>12}")

    # Summary
    valid = [r for r in results if not np.isnan(r["mse_sph"])]
    avg_corrupt = np.mean([r["mse_corrupt"] for r in valid])
    avg_sph = np.mean([r["mse_sph"] for r in valid])
    avg_zero = np.mean([r["mse_zero"] for r in valid])
    sph_win_corrupt = sum(1 for r in valid if r["mse_sph"] < r["mse_corrupt"])
    sph_win_zero = sum(1 for r in valid if r["mse_sph"] < r["mse_zero"])

    print("-" * 82)
    print(f"\n{'SUMMARY':}")
    print(f"  Sessions evaluated : {len(valid)}")
    print(f"  Mean MSE corrupted : {avg_corrupt:.1f} µV²")
    print(f"  Mean MSE spline    : {avg_sph:.1f} µV²  ({avg_sph/avg_corrupt*100:.1f}% of corrupted)")
    print(f"  Mean MSE zero-out  : {avg_zero:.1f} µV²  ({avg_zero/avg_corrupt*100:.1f}% of corrupted)")
    print(f"  Spline beats corrupt: {sph_win_corrupt}/{len(valid)} sessions")
    print(f"  Spline beats zero   : {sph_win_zero}/{len(valid)} sessions")

    # Save
    out = {
        "n_sessions": len(valid), "avg_mse_corrupt": avg_corrupt,
        "avg_mse_spherical": avg_sph, "avg_mse_zero": avg_zero,
        "spline_beats_corrupt_rate": sph_win_corrupt / len(valid),
        "spline_beats_zero_rate": sph_win_zero / len(valid),
        "bad_channel": STANDARD_CHANNELS[BAD_CH_IDX],
    }
    Path("results/interpolation_comparison.json").write_text(json.dumps(out, indent=2))
    print(f"\n✓ Saved results/interpolation_comparison.json")


if __name__ == "__main__":
    main()
