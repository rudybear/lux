#!/usr/bin/env python3
"""Image quality comparison tool for Lux shader output validation.

Compares baseline and candidate images using PSNR and SSIM metrics.
Requires only PIL (Pillow) and numpy -- no OpenCV or scikit-image needed.

Usage:
    python tools/quality_metrics.py baseline.png candidate.png
    python tools/quality_metrics.py baseline.png candidate.png --diff diff_heatmap.png
    python tools/quality_metrics.py baseline.png candidate.png --psnr-threshold 35 --ssim-threshold 0.97
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def compare_images(
    baseline_path: str,
    candidate_path: str,
    diff_output: str | None = None,
) -> dict:
    """Compare two images for quality degradation.

    Loads both images, converts to numpy arrays, and computes pixel-level
    quality metrics.  Optionally writes a per-pixel difference heatmap PNG.

    Args:
        baseline_path: Path to the reference / ground-truth image.
        candidate_path: Path to the image under test.
        diff_output: If provided, path where the difference heatmap PNG is saved.

    Returns:
        dict with keys:
            psnr      -- Peak Signal-to-Noise Ratio in dB.
                         >40 dB  = imperceptible difference
                         >30 dB  = acceptable difference
                         float('inf') when images are identical.
            ssim      -- Structural Similarity Index in [0, 1].
                         >0.99 = visually lossless
                         >0.95 = acceptable
            max_delta -- Maximum absolute per-pixel difference (0-255 scale).
            width     -- Image width in pixels.
            height    -- Image height in pixels.

    Raises:
        FileNotFoundError: If either image path does not exist.
        ValueError: If the two images have different dimensions.
    """
    baseline_img = Image.open(baseline_path).convert("RGB")
    candidate_img = Image.open(candidate_path).convert("RGB")

    if baseline_img.size != candidate_img.size:
        raise ValueError(
            f"Image size mismatch: baseline is {baseline_img.size}, "
            f"candidate is {candidate_img.size}"
        )

    width, height = baseline_img.size

    # Convert to float64 arrays for precision (H x W x 3, range 0-255)
    a = np.asarray(baseline_img, dtype=np.float64)
    b = np.asarray(candidate_img, dtype=np.float64)

    diff = a - b

    # --- PSNR ---
    mse = float(np.mean(diff ** 2))
    if mse == 0.0:
        psnr = float("inf")
    else:
        psnr = 10.0 * math.log10(255.0 ** 2 / mse)

    # --- SSIM (full-image statistics, per-channel then averaged) ---
    ssim = _compute_ssim(a, b)

    # --- Max delta ---
    max_delta = int(np.max(np.abs(diff)))

    # --- Diff heatmap ---
    if diff_output is not None:
        _save_diff_heatmap(diff, diff_output)

    return {
        "psnr": psnr,
        "ssim": ssim,
        "max_delta": max_delta,
        "width": width,
        "height": height,
    }


def _compute_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Structural Similarity Index between two H x W x C float64 arrays.

    Uses per-channel global-statistics approach (Wang et al., 2004 simplified)
    and averages across channels.

    Constants follow the original paper:
        C1 = (K1 * L)^2,  K1 = 0.01, L = 255
        C2 = (K2 * L)^2,  K2 = 0.03, L = 255
    """
    c1 = (0.01 * 255.0) ** 2  # 6.5025
    c2 = (0.03 * 255.0) ** 2  # 58.5225

    num_channels = a.shape[2] if a.ndim == 3 else 1
    ssim_channels: list[float] = []

    for ch in range(num_channels):
        x = a[:, :, ch] if a.ndim == 3 else a
        y = b[:, :, ch] if b.ndim == 3 else b

        mu_x = float(np.mean(x))
        mu_y = float(np.mean(y))
        sigma_x_sq = float(np.var(x))
        sigma_y_sq = float(np.var(y))
        sigma_xy = float(np.mean((x - mu_x) * (y - mu_y)))

        # Luminance * Contrast-structure combined
        numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
        denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x_sq + sigma_y_sq + c2)

        ssim_channels.append(numerator / denominator)

    return float(np.mean(ssim_channels))


def _save_diff_heatmap(diff: np.ndarray, output_path: str) -> None:
    """Save a heatmap PNG visualising per-pixel absolute differences.

    Maps the per-pixel L2 magnitude across channels to a blue-red colour ramp:
    blue (0 difference) -> red (max difference).
    """
    # Per-pixel Euclidean distance across channels, then normalise to 0-1
    magnitude = np.sqrt(np.sum(diff ** 2, axis=2))
    max_mag = float(np.max(magnitude))
    if max_mag == 0.0:
        # Identical images -- produce an all-black heatmap
        h, w = magnitude.shape
        Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8)).save(output_path)
        return

    norm = magnitude / max_mag  # 0-1

    # Build an RGB heatmap: blue (cold) -> red (hot)
    # R channel ramps up, B channel ramps down, G peaks in the middle
    r = np.clip(norm * 2.0, 0.0, 1.0)
    g = np.clip(1.0 - np.abs(norm - 0.5) * 2.0, 0.0, 1.0)
    b = np.clip(1.0 - norm * 2.0, 0.0, 1.0)

    heatmap = np.stack([r, g, b], axis=2)
    heatmap_u8 = (heatmap * 255.0).astype(np.uint8)

    Image.fromarray(heatmap_u8, mode="RGB").save(output_path)


def quality_check(
    psnr: float,
    ssim: float,
    psnr_threshold: float = 40.0,
    ssim_threshold: float = 0.99,
) -> dict:
    """Check whether quality metrics pass the given thresholds.

    Args:
        psnr: Peak Signal-to-Noise Ratio value (dB).
        ssim: Structural Similarity Index value (0-1).
        psnr_threshold: Minimum acceptable PSNR (default 40.0 dB).
        ssim_threshold: Minimum acceptable SSIM (default 0.99).

    Returns:
        dict with keys:
            passed  -- True if both metrics pass.
            psnr_ok -- True if PSNR >= threshold.
            ssim_ok -- True if SSIM >= threshold.
            message -- Human-readable summary string.
    """
    psnr_ok = psnr >= psnr_threshold
    ssim_ok = ssim >= ssim_threshold
    passed = psnr_ok and ssim_ok

    parts: list[str] = []
    if passed:
        parts.append("PASS")
    else:
        parts.append("FAIL")

    psnr_str = f"{psnr:.2f}" if math.isfinite(psnr) else "inf"
    parts.append(f"PSNR={psnr_str} dB ({'ok' if psnr_ok else 'BELOW ' + str(psnr_threshold)})")
    parts.append(f"SSIM={ssim:.4f} ({'ok' if ssim_ok else 'BELOW ' + str(ssim_threshold)})")

    return {
        "passed": passed,
        "psnr_ok": psnr_ok,
        "ssim_ok": ssim_ok,
        "message": " | ".join(parts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two images for quality degradation (PSNR / SSIM)"
    )
    parser.add_argument("baseline", type=str, help="Path to baseline (reference) image")
    parser.add_argument("candidate", type=str, help="Path to candidate image")
    parser.add_argument(
        "--diff", type=str, default=None,
        help="Output path for difference heatmap PNG",
    )
    parser.add_argument(
        "--psnr-threshold", type=float, default=40.0,
        help="Minimum acceptable PSNR in dB (default: 40.0)",
    )
    parser.add_argument(
        "--ssim-threshold", type=float, default=0.99,
        help="Minimum acceptable SSIM (default: 0.99)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output results as JSON",
    )
    args = parser.parse_args()

    try:
        result = compare_images(args.baseline, args.candidate, diff_output=args.diff)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    check = quality_check(
        result["psnr"], result["ssim"],
        psnr_threshold=args.psnr_threshold,
        ssim_threshold=args.ssim_threshold,
    )

    if args.json:
        output = {**result, **check}
        print(json.dumps(output, indent=2, default=str))
    else:
        psnr_str = f"{result['psnr']:.2f}" if math.isfinite(result["psnr"]) else "inf"
        print(f"Image size : {result['width']}x{result['height']}")
        print(f"PSNR       : {psnr_str} dB")
        print(f"SSIM       : {result['ssim']:.6f}")
        print(f"Max delta  : {result['max_delta']}")
        print(f"Result     : {check['message']}")
        if args.diff:
            print(f"Diff heatmap: {args.diff}")

    sys.exit(0 if check["passed"] else 1)


if __name__ == "__main__":
    main()
