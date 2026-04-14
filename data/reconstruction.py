import numpy as np


def reconstruct_signal(
    segments: np.ndarray,
    window: int = 512,
    overlap: float = 0.5,
) -> np.ndarray:
    """
    Reconstruct a 1-D signal from overlapping segments using overlap-add.

    Parameters
    ----------
    segments : Array of shape (N_windows, window).
    window   : Window length (must match segment_signal window).
    overlap  : Overlap fraction (must match segment_signal overlap).

    Returns
    -------
    Reconstructed 1-D signal of shape (T_reconstructed,), float32.

    Notes
    -----
    Uses a triangular (Bartlett) window for smooth blending.
    Divides by accumulated weights to normalise overlapping regions.
    """
    step = int(window * (1.0 - overlap))
    n_windows = len(segments)
    total_len = step * (n_windows - 1) + window

    output  = np.zeros(total_len, dtype=np.float32)
    weights = np.zeros(total_len, dtype=np.float32)
    # Triangular blend window (smooth borders)
    blend_w = np.bartlett(window).astype(np.float32)

    for i, seg in enumerate(segments):
        start = i * step
        output[start : start + window]  += seg.astype(np.float32) * blend_w
        weights[start : start + window] += blend_w

    # Avoid division by zero at edges with very small weights
    weights = np.where(weights < 1e-8, 1.0, weights)
    return (output / weights).astype(np.float32)
