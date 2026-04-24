"""
Unit tests for evaluate/metrics.py
Run: pytest tests/test_metrics.py -v
"""
import numpy as np
import pytest
from evaluate.metrics import (
    compute_auprc, compute_auroc,
    compute_f1_at_threshold, compute_precision_recall,
    best_f1_threshold, bcr_full_report,
    snr_db, snr_improvement, pearson_r, rrmse, rrmse_spectral, ssim_1d,
    denoiser_full_report,
)

RNG = np.random.default_rng(42)


# ─── BCR ───────────────────────────────────────────────────────

class TestBCRMetrics:

    def test_auprc_perfect(self):
        # Perfect predictor → AUPRC = 1.0
        y_true = np.array([0, 0, 0, 1, 1])
        y_prob = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
        assert compute_auprc(y_true, y_prob) == pytest.approx(1.0, abs=1e-6)

    def test_auprc_random_approx_base_rate(self):
        # Random predictor → AUPRC ≈ bad_rate (0.039 for our dataset)
        # For this toy: 10% bad rate → random AUPRC ≈ 0.10
        n = 5000
        bad_rate = 0.10
        y_true = (RNG.random(n) < bad_rate).astype(int)
        y_prob = RNG.random(n)
        ap = compute_auprc(y_true, y_prob)
        assert 0.05 < ap < 0.20, f"Expected ~{bad_rate}, got {ap:.3f}"

    def test_auroc_random_near_half(self):
        n = 2000
        y_true = (RNG.random(n) < 0.5).astype(int)
        y_prob = RNG.random(n)
        auroc = compute_auroc(y_true, y_prob)
        assert 0.40 < auroc < 0.60

    def test_f1_at_threshold_all_pos(self):
        # Predict all positive → recall=1, precision=bad_rate → F1 low
        y_true = np.array([0]*9 + [1]*1)
        y_prob = np.ones(10)
        f1 = compute_f1_at_threshold(y_true, y_prob, threshold=0.5)
        assert 0.10 < f1 < 0.30

    def test_best_f1_threshold_returns_valid_range(self):
        y_true = (RNG.random(200) < 0.3).astype(int)
        y_prob = y_true * 0.7 + RNG.random(200) * 0.3
        t, f1 = best_f1_threshold(y_true, y_prob)
        assert 0.0 <= t <= 1.0
        assert 0.0 <= f1 <= 1.0

    def test_bcr_full_report_keys(self):
        y_true = np.array([0, 0, 1, 1, 0])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.3])
        report = bcr_full_report(y_true, y_prob, threshold=0.5)
        for key in ["auprc", "auroc", "f1", "precision", "recall", "tp", "tn"]:
            assert key in report, f"Missing key: {key}"


# ─── DENOISER ──────────────────────────────────────────────────

class TestDenoiserMetrics:

    def test_snr_db_known_value(self):
        # signal power = 1, noise power = 0.01 → SNR = 20 dB
        signal = np.ones(100)
        noise  = np.ones(100) * 0.1  # power = 0.01, so 10*log10(1/0.01) = 20
        assert snr_db(signal, noise) == pytest.approx(20.0, abs=0.1)

    def test_snr_improvement_positive_when_denoised_cleaner(self):
        T = 512
        clean    = np.sin(2 * np.pi * 10 * np.linspace(0, 2, T))
        noise    = RNG.normal(0, 2.0, T)
        noisy    = clean + noise
        # "denoised" = 80% clean + 20% noise — better than raw noisy
        denoised = 0.8 * clean + 0.2 * noise
        delta = snr_improvement(noisy, denoised, clean)
        assert delta > 0, f"Expected positive ΔSNR, got {delta:.2f} dB"

    def test_snr_improvement_zero_when_identical(self):
        # If denoised == noisy, ΔSNR should be ~0
        T = 512
        clean  = np.sin(2 * np.pi * 10 * np.linspace(0, 2, T))
        noisy  = clean + RNG.normal(0, 1.0, T)
        delta  = snr_improvement(noisy, noisy, clean)
        assert abs(delta) < 0.01

    def test_pearson_r_perfect(self):
        sig = np.sin(np.linspace(0, 4 * np.pi, 256))
        assert pearson_r(sig, sig) == pytest.approx(1.0, abs=1e-6)

    def test_pearson_r_orthogonal(self):
        x = np.sin(np.linspace(0, 4 * np.pi, 256))
        y = np.cos(np.linspace(0, 4 * np.pi, 256))
        r = pearson_r(x, y)
        assert abs(r) < 0.05, f"Expected ~0 for orthogonal signals, got {r:.4f}"

    def test_rrmse_zero_for_perfect(self):
        sig = np.random.randn(512)
        assert rrmse(sig, sig) == pytest.approx(0.0, abs=1e-9)

    def test_rrmse_large_for_noise(self):
        target = np.sin(np.linspace(0, 4 * np.pi, 512))
        pred   = target + RNG.normal(0, 10.0, 512)
        assert rrmse(pred, target) > 1.0

    def test_rrmse_spectral_zero_for_perfect(self):
        sig = np.sin(np.linspace(0, 4 * np.pi, 512))
        assert rrmse_spectral(sig, sig) == pytest.approx(0.0, abs=1e-6)

    def test_rrmse_spectral_large_for_noise(self):
        target = np.sin(np.linspace(0, 4 * np.pi, 512))
        pred   = target + RNG.normal(0, 5.0, 512)
        assert rrmse_spectral(pred, target) > 0.5

    def test_ssim_1d_perfect(self):
        sig = np.sin(np.linspace(0, 4 * np.pi, 512))
        # Normalise first (as will happen in practice)
        sig_n = (sig - sig.mean()) / (sig.std() + 1e-8)
        s = ssim_1d(sig_n, sig_n)
        assert s > 0.99

    def test_ssim_1d_low_for_noise(self):
        target = np.sin(np.linspace(0, 4 * np.pi, 512))
        noise  = RNG.normal(0, 3.0, 512)
        # Both normalised
        t_n = (target - target.mean()) / (target.std() + 1e-8)
        n_n = (noise  - noise.mean())  / (noise.std()  + 1e-8)
        s = ssim_1d(n_n, t_n)
        assert s < 0.5

    def test_denoiser_full_report_keys(self):
        T = 512
        clean    = np.sin(np.linspace(0, 4 * np.pi, T))
        noisy    = clean + RNG.normal(0, 1.0, T)
        denoised = clean + RNG.normal(0, 0.1, T)
        report = denoiser_full_report(noisy, denoised, clean)
        for key in ["delta_snr_db", "pearson_r", "rrmse_temporal", "rrmse_spectral", "ssim"]:
            assert key in report, f"Missing key: {key}"
        assert report["delta_snr_db"] > 0, "Clean-ish denoised should show positive ΔSNR"
