"""
BCR Interpolation — post-prediction bad channel interpolation.

Responsibilities:
  - Given BCR model predictions, interpolate flagged bad channels
    using spherical spline interpolation (MNE-based)
  - Intended as the final stage of the BCR pipeline before denoising
"""
