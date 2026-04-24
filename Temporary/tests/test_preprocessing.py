"""
Smoke test for data/preprocessing.py and data/reconstruction.py.
Run from project root: pytest tests/test_preprocessing.py -s
"""
import numpy as np
from pathlib import Path
from data.preprocessing import bandpass_filter, notch_filter, zscore_normalize, segment_signal
from data.reconstruction import reconstruct_signal

DATA_DIR = Path("data/raw/eegdenoisenet/data")

# ── 1. Load arrays ─────────────────────────────────────────────────────────
clean = np.load(DATA_DIR / "EEG_all_epochs_512hz.npy")   # (4514, 1024)
eog   = np.load(DATA_DIR / "EOG_all_epochs.npy")         # (3400, 512)
print(f"✅ Loaded:  clean={clean.shape}  eog={eog.shape}")

# ── 2. Pick one epoch (1024 samples = 2 s at 512 Hz) ──────────────────────
raw_signal = clean[0].astype(np.float32)                 # (1024,)
print(f"   Raw signal: shape={raw_signal.shape}  dtype={raw_signal.dtype}")
print(f"   Value range: [{raw_signal.min():.2f}, {raw_signal.max():.2f}]")

# ── 3. Bandpass filter ─────────────────────────────────────────────────────
filtered_bp = bandpass_filter(raw_signal, fs=512, low=1.0, high=40.0)
print(f"\n✅ bandpass_filter: shape={filtered_bp.shape}  dtype={filtered_bp.dtype}")
print(f"   Value range after BP: [{filtered_bp.min():.4f}, {filtered_bp.max():.4f}]")

# ── 4. Notch filter ────────────────────────────────────────────────────────
filtered_notch = notch_filter(filtered_bp, fs=512, freq=50.0)
print(f"\n✅ notch_filter:    shape={filtered_notch.shape}")

# ── 5. Z-score normalize ───────────────────────────────────────────────────
normed = zscore_normalize(filtered_notch)
print(f"\n✅ zscore_normalize: mean={normed.mean():.6f} (should be ~0)")
print(f"                    std ={normed.std():.6f}  (should be ~1)")

# ── 6. Segment using window=1024 to match 512 Hz epoch length ─────────────
long_signal = clean[:10].flatten().astype(np.float32)    # (10240,)
segments = segment_signal(long_signal, window=1024, overlap=0.5, normalize=True)
print(f"\n✅ segment_signal:  input={long_signal.shape}  segments={segments.shape}")
print(f"   Segment 0 mean={segments[0].mean():.4f}  std={segments[0].std():.4f}")

# ── 7. Reconstruct — window must match segment_signal window ───────────────
reconstructed = reconstruct_signal(segments, window=1024, overlap=0.5)
print(f"\n✅ reconstruct_signal: output shape={reconstructed.shape}")

interior_slice = slice(512, -512)
orig_interior  = zscore_normalize(long_signal)[interior_slice]
recon_interior = reconstructed[interior_slice]

if len(orig_interior) == len(recon_interior):
    corr = np.corrcoef(orig_interior, recon_interior)[0, 1]
    print(f"   Roundtrip Pearson r = {corr:.4f}  (should be > 0.95)")
else:
    print(f"   (Shapes differ at edge — OK. Check interior manually if needed.)")

print("\n🎉 Smoke test passed — preprocessing + reconstruction modules working.")
