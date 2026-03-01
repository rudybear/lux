"""Tests for AI reference image matching (Phase 16)."""

import numpy as np
import pytest

from luxc.ai.reference_match import ImageComparison, MatchResult, compare_images


class TestCompareImagesIdentical:
    def test_compare_images_identical(self):
        """Same images produce PSNR ~100 and SSIM ~1.0, converged=True."""
        white = np.full((64, 64, 3), 255, dtype=np.uint8)

        result = compare_images(white, white)

        assert isinstance(result, ImageComparison)
        assert result.psnr >= 90.0  # Essentially identical -> 100.0
        assert result.ssim > 0.99
        assert result.mean_abs_error < 0.01
        assert result.converged is True


class TestCompareImagesDifferent:
    def test_compare_images_different(self):
        """Very different images produce low PSNR, converged=False."""
        white = np.full((64, 64, 3), 255, dtype=np.uint8)
        black = np.zeros((64, 64, 3), dtype=np.uint8)

        result = compare_images(white, black)

        assert isinstance(result, ImageComparison)
        # White vs black: MSE = 255^2, PSNR = 10*log10(255^2 / 255^2) = 0
        assert result.psnr < 5.0
        assert result.ssim < 0.1
        assert result.mean_abs_error > 200.0
        assert result.converged is False


class TestCompareImagesSimilar:
    def test_compare_images_similar(self):
        """Slightly different images produce moderate PSNR."""
        gray = np.full((64, 64, 3), 128, dtype=np.uint8)
        # Slightly lighter gray
        lighter = np.full((64, 64, 3), 140, dtype=np.uint8)

        result = compare_images(gray, lighter)

        assert isinstance(result, ImageComparison)
        # Small difference -> moderate-to-high PSNR
        assert 20.0 < result.psnr < 50.0
        assert result.mean_abs_error == pytest.approx(12.0, abs=0.5)


class TestCompareImagesSizeMismatch:
    def test_compare_images_size_mismatch(self):
        """Different sizes get cropped to the common region, no crash."""
        small = np.full((32, 32, 3), 200, dtype=np.uint8)
        large = np.full((64, 64, 3), 200, dtype=np.uint8)

        # Should not raise, crops to min(32, 64) = 32x32
        result = compare_images(small, large)

        assert isinstance(result, ImageComparison)
        # Same color values in the overlapping region -> high PSNR
        assert result.psnr >= 90.0
        assert result.converged is True


class TestImageComparisonThresholds:
    def test_image_comparison_thresholds(self):
        """Custom thresholds affect convergence determination."""
        gray = np.full((64, 64, 3), 128, dtype=np.uint8)
        lighter = np.full((64, 64, 3), 140, dtype=np.uint8)

        # With very strict thresholds, similar images do NOT converge
        strict = compare_images(gray, lighter, psnr_threshold=50.0, ssim_threshold=0.99)
        assert strict.converged is False

        # With relaxed thresholds, the same pair DOES converge
        relaxed = compare_images(gray, lighter, psnr_threshold=15.0, ssim_threshold=0.5)
        assert relaxed.converged is True

        # PSNR and SSIM values themselves do not change with thresholds
        assert strict.psnr == pytest.approx(relaxed.psnr)
        assert strict.ssim == pytest.approx(relaxed.ssim)
