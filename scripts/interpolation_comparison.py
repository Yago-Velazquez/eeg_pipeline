"""
scripts/interpolation_comparison.py

Compare interpolation methods on BCR test set data.

Strategy
--------
1. Load BCR CSV.
2. For each session in the test set (visit 4 subjects, n=10):
   - Synthesize multi-channel EEG from per-channel amplitude stats
   - Corrupt a known bad channel with noise
   - Interpolate with spherical and zero-out
   - Record MSE for each method

Note: BCR dataset contains pre-computed features, not raw time-series. We
synthesize plausible signal using 'Standard deviation (mean)' as amplitude.
We are testing interpolation logic, not the feature extractor.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bad_channel_rejection.interpolation import interpolate_bad_channels  # noqa: E402

DATA_PATH = Path("data/raw/Bad_channels_for_ML.csv")
SFREQ = 256.0
N_SAMPLES = 512
N_SESSIONS = 10
RNG = np.random.default_rng(seed=42)

STANDARD_CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "Oz", "O2",
]
N_CH = len(STANDARD_CHANNELS)
BAD_CH_IDX = 8


def synthesize_session_eeg(session_rows: pd.DataFrame) -> np.ndarray:
    std_col = [c for c in session_rows.columns if "Standard deviation (mean)" in c]
    t = np.linspace(0, N_SAMPLES / SFREQ, N_SAMPLES)
    eeg = np.zeros((N_CH, N_SAMPLES))
    for i in range(N_CH):
        if i < len(session_rows) and std_col:
            amp = float(session_rows.iloc[i][std_col[0]]) * 1e6
            if np.isnan(amp) or amp <= 0:
                amp = 20.0
        else:
            amp = 20.0
        eeg[i] = np.sin(2 * np.pi * 10 * t) * amp + RNG.normal(0, amp * 0.15, N_SAMPLES)
    return eeg


def corrupt_channel(eeg: np.ndarray, ch_idx: int, noise_std: float = 200.0):
    corrupted = eeg.copy()
    corrupted[ch_idx] = RNG.normal(0, noise_std, eeg.shape[1])
    return corrupted


def main():
    print("Loading BCR data...")
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    df[["visit", "subject_id", "site"]] = df["Session"].str.split("-", expand=True)
    df["visit"] = df["visit"].astype(int)

    test_df = df[df["visit"] == 4].copy()
    sessions = test_df["Session"].unique()[:N_SESSIONS]

    results = []
    print(f"\nRunning comparison on {len(sessions)} sessions (visit 4)...")
    print(f"  Bad channel: {STANDARD_CHANNELS[BAD_CH_IDX]}  noise_std=200 µV\n")
    print(
        f"{'Session':<20} {'MSE corrupt':>12} {'MSE spline':>12} "
        f"{'MSE zero':>12}"
    )
    print("-" * 60)

    for sess in sessions:
        rows = test_df[test_df["Session"] == sess]
        clean = synthesize_session_eeg(rows)
        corrupted = corrupt_channel(clean, BAD_CH_IDX)

        try:
            interp_sph = interpolate_bad_channels(
                corrupted, [BAD_CH_IDX], ch_names=STANDARD_CHANNELS,
                sfreq=SFREQ, method="spherical",
            )
            mse_sph = float(np.mean((interp_sph[BAD_CH_IDX] - clean[BAD_CH_IDX]) ** 2))
        except Exception as e:
            mse_sph = float("nan")
            print(f"  WARNING: spherical failed for {sess}: {e}")

        interp_zero = interpolate_bad_channels(
            corrupted, [BAD_CH_IDX], method="zero"
        )
        mse_zero = float(np.mean((interp_zero[BAD_CH_IDX] - clean[BAD_CH_IDX]) ** 2))
        mse_corrupt = float(np.mean((corrupted[BAD_CH_IDX] - clean[BAD_CH_IDX]) ** 2))

        results.append({
            "session": sess, "mse_corrupt": mse_corrupt,
            "mse_sph": mse_sph, "mse_zero": mse_zero,
        })
        print(f"{sess:<20} {mse_corrupt:>12.1f} {mse_sph:>12.1f} {mse_zero:>12.1f}")

    valid = [r for r in results if not np.isnan(r["mse_sph"])]
    avg_corrupt = np.mean([r["mse_corrupt"] for r in valid])
    avg_sph = np.mean([r["mse_sph"] for r in valid])
    avg_zero = np.mean([r["mse_zero"] for r in valid])

    print("-" * 60)
    print("\nSUMMARY")
    print(f"  Sessions evaluated : {len(valid)}")
    print(f"  Mean MSE corrupted : {avg_corrupt:.1f} µV²")
    print(f"  Mean MSE spline    : {avg_sph:.1f} µV²")
    print(f"  Mean MSE zero-out  : {avg_zero:.1f} µV²")

    out = {
        "n_sessions": len(valid),
        "avg_mse_corrupt": avg_corrupt,
        "avg_mse_spherical": avg_sph,
        "avg_mse_zero": avg_zero,
        "bad_channel": STANDARD_CHANNELS[BAD_CH_IDX],
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/interpolation_comparison.json").write_text(json.dumps(out, indent=2))
    print("\nSaved results/interpolation_comparison.json")


if __name__ == "__main__":
    main()
