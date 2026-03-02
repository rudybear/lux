"""Tests for image quality comparison metrics.

Tests cover:
- Identical images (PSNR=inf, SSIM=1.0, max_delta=0)
- Known pixel difference with expected PSNR range
- All-black vs all-white (very low PSNR)
- Size mismatch raises ValueError
- quality_check pass/fail logic
- Diff heatmap output file creation
"""

import math
import os
import tempfile

import numpy as np
import pytest
from PIL import Image

from tools.quality_metrics import compare_images, quality_check


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image(width, height, color=(128, 128, 128)):
    """Create an RGB uint8 numpy array filled with a single color."""
    arr = np.full((height, width, 3), color, dtype=np.uint8)
    return arr


def _save_temp_image(arr):
    """Save a numpy array as a temporary PNG and return the path."""
    f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    Image.fromarray(arr).save(f.name)
    f.close()
    return f.name


# ---------------------------------------------------------------------------
# Tests: compare_images
# ---------------------------------------------------------------------------

class TestCompareImagesIdentical:
    """Comparing an image to itself should yield perfect metrics."""

    def test_identical_images(self, tmp_path):
        arr = _make_image(64, 64, color=(100, 150, 200))
        path_a = str(tmp_path / "a.png")
        path_b = str(tmp_path / "b.png")
        Image.fromarray(arr).save(path_a)
        Image.fromarray(arr).save(path_b)

        result = compare_images(path_a, path_b)

        assert result["psnr"] == float("inf")
        assert result["ssim"] == pytest.approx(1.0, abs=1e-6)
        assert result["max_delta"] == 0
        assert result["width"] == 64
        assert result["height"] == 64


class TestCompareImagesKnownDifference:
    """A small known pixel difference should produce a finite PSNR."""

    def test_small_difference(self, tmp_path):
        arr_a = _make_image(32, 32, color=(128, 128, 128))
        arr_b = arr_a.copy()
        # Introduce a small difference: shift one pixel by 10
        arr_b[0, 0, 0] = 138

        path_a = str(tmp_path / "a.png")
        path_b = str(tmp_path / "b.png")
        Image.fromarray(arr_a).save(path_a)
        Image.fromarray(arr_b).save(path_b)

        result = compare_images(path_a, path_b)

        assert math.isfinite(result["psnr"])
        # PSNR should be high (small difference) but not infinite
        assert result["psnr"] > 30.0
        assert result["max_delta"] == 10
        assert result["ssim"] < 1.0


class TestCompareImagesBlackWhite:
    """All-black vs all-white should produce very low PSNR."""

    def test_black_vs_white(self, tmp_path):
        black = _make_image(16, 16, color=(0, 0, 0))
        white = _make_image(16, 16, color=(255, 255, 255))

        path_a = str(tmp_path / "black.png")
        path_b = str(tmp_path / "white.png")
        Image.fromarray(black).save(path_a)
        Image.fromarray(white).save(path_b)

        result = compare_images(path_a, path_b)

        assert math.isfinite(result["psnr"])
        # PSNR for black vs white is very low (close to 0 dB)
        assert result["psnr"] < 10.0
        assert result["max_delta"] == 255
        assert result["ssim"] < 0.1


class TestCompareImagesSizeMismatch:
    """Different-sized images should raise ValueError."""

    def test_size_mismatch_raises(self, tmp_path):
        small = _make_image(16, 16)
        large = _make_image(32, 32)

        path_a = str(tmp_path / "small.png")
        path_b = str(tmp_path / "large.png")
        Image.fromarray(small).save(path_a)
        Image.fromarray(large).save(path_b)

        with pytest.raises(ValueError, match="Image size mismatch"):
            compare_images(path_a, path_b)


class TestDiffOutput:
    """Diff heatmap should be written when diff_output is provided."""

    def test_diff_file_created(self, tmp_path):
        arr_a = _make_image(16, 16, color=(100, 100, 100))
        arr_b = _make_image(16, 16, color=(110, 120, 130))

        path_a = str(tmp_path / "a.png")
        path_b = str(tmp_path / "b.png")
        diff_path = str(tmp_path / "diff.png")
        Image.fromarray(arr_a).save(path_a)
        Image.fromarray(arr_b).save(path_b)

        compare_images(path_a, path_b, diff_output=diff_path)

        assert os.path.isfile(diff_path)
        # Verify it is a valid image
        diff_img = Image.open(diff_path)
        assert diff_img.size == (16, 16)


# ---------------------------------------------------------------------------
# Tests: quality_check
# ---------------------------------------------------------------------------

class TestQualityCheckPass:
    """quality_check should pass when metrics exceed thresholds."""

    def test_pass_both_above(self):
        result = quality_check(psnr=50.0, ssim=0.995)
        assert result["passed"] is True
        assert result["psnr_ok"] is True
        assert result["ssim_ok"] is True
        assert "PASS" in result["message"]

    def test_pass_infinite_psnr(self):
        result = quality_check(psnr=float("inf"), ssim=1.0)
        assert result["passed"] is True


class TestQualityCheckFailPsnr:
    """quality_check should fail when PSNR is below threshold."""

    def test_fail_low_psnr(self):
        result = quality_check(psnr=25.0, ssim=0.995)
        assert result["passed"] is False
        assert result["psnr_ok"] is False
        assert result["ssim_ok"] is True
        assert "FAIL" in result["message"]


class TestQualityCheckFailSsim:
    """quality_check should fail when SSIM is below threshold."""

    def test_fail_low_ssim(self):
        result = quality_check(psnr=50.0, ssim=0.90)
        assert result["passed"] is False
        assert result["psnr_ok"] is True
        assert result["ssim_ok"] is False
        assert "FAIL" in result["message"]
