"""Reference image matching via render-compare loop.

Iteratively generates and refines a Lux shader to match a reference
photograph by comparing rendered output against the target image.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ImageComparison:
    """Result of comparing two images."""
    psnr: float
    ssim: float
    mean_abs_error: float
    converged: bool


@dataclass
class MatchResult:
    """Result of the reference matching loop."""
    final_result: object  # GenerateResult (avoid circular import)
    iterations: list[dict] = field(default_factory=list)
    comparison: ImageComparison | None = None
    converged: bool = False


def compare_images(
    reference,  # np.ndarray
    rendered,   # np.ndarray
    psnr_threshold: float = 25.0,
    ssim_threshold: float = 0.80,
) -> ImageComparison:
    """Compare two images using PSNR and simplified SSIM.

    Only requires numpy (no scikit-image dependency).

    Parameters
    ----------
    reference : np.ndarray
        Reference image as HxWxC uint8 array.
    rendered : np.ndarray
        Rendered image as HxWxC uint8 array (same shape as reference).
    psnr_threshold : float
        PSNR value above which images are considered converged.
    ssim_threshold : float
        SSIM value above which images are considered converged.

    Returns
    -------
    ImageComparison
        Comparison metrics and convergence flag.
    """
    import numpy as np

    # Ensure float for computation
    ref = reference.astype(np.float64)
    ren = rendered.astype(np.float64)

    # Resize if shapes don't match
    if ref.shape != ren.shape:
        # Crop or pad to match
        min_h = min(ref.shape[0], ren.shape[0])
        min_w = min(ref.shape[1], ren.shape[1])
        ref = ref[:min_h, :min_w]
        ren = ren[:min_h, :min_w]

    # Mean Absolute Error
    mae = float(np.mean(np.abs(ref - ren)))

    # PSNR
    mse = float(np.mean((ref - ren) ** 2))
    if mse < 1e-10:
        psnr_val = 100.0  # Essentially identical
    else:
        psnr_val = 10.0 * math.log10(255.0 ** 2 / mse)

    # Simplified SSIM (per-channel, then average)
    # Using the standard SSIM formula with default constants
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu_ref = np.mean(ref)
    mu_ren = np.mean(ren)
    sigma_ref_sq = np.var(ref)
    sigma_ren_sq = np.var(ren)
    sigma_cross = np.mean((ref - mu_ref) * (ren - mu_ren))

    numerator = (2 * mu_ref * mu_ren + C1) * (2 * sigma_cross + C2)
    denominator = (mu_ref ** 2 + mu_ren ** 2 + C1) * (sigma_ref_sq + sigma_ren_sq + C2)
    ssim_val = float(numerator / denominator) if denominator > 0 else 0.0

    converged = psnr_val >= psnr_threshold and ssim_val >= ssim_threshold

    return ImageComparison(
        psnr=psnr_val,
        ssim=ssim_val,
        mean_abs_error=mae,
        converged=converged,
    )


def render_lux_to_image(
    lux_source: str,
    scene: str = "sphere",
    width: int = 256,
    height: int = 256,
):
    """Compile Lux source and render to a numpy RGBA array.

    This is a best-effort function that requires optional rendering
    dependencies (wgpu, PIL). Returns None if rendering is not available.

    Parameters
    ----------
    lux_source : str
        Lux shader source code.
    scene : str
        Scene preset ("sphere", "plane", "cube").
    width, height : int
        Render resolution.

    Returns
    -------
    np.ndarray or None
        HxWx4 uint8 RGBA array, or None if rendering is unavailable.
    """
    # Rendering requires the engine module which may not be available
    # in all installations. This is a best-effort feature.
    try:
        from luxc.ai.generate import verify_lux_source_structured
        success, errors = verify_lux_source_structured(lux_source)
        if not success:
            return None
    except Exception:
        return None

    # Actual rendering would use the engine module
    # For now, return None to indicate rendering is not available
    # The match loop will skip the compare step and rely on AI judgment
    return None


def match_reference_image(
    reference_image_path: Path,
    description: str = "",
    max_iterations: int = 5,
    verify: bool = True,
    model: str | None = None,
    provider: str | None = None,
    base_url: str | None = None,
    scene: str = "sphere",
    width: int = 256,
    height: int = 256,
) -> MatchResult:
    """Iteratively generate and refine a shader to match a reference image.

    Pipeline:
    1. Generate initial material from the reference image
    2. Optionally compile + render the result
    3. Compare rendered output against reference
    4. If not converged, send reference + render + source + metrics to AI
    5. Repeat until converged or max_iterations reached

    Parameters
    ----------
    reference_image_path : Path
        Path to the reference image to match.
    description : str
        Optional text description to guide generation.
    max_iterations : int
        Maximum refinement iterations (default: 5).
    verify : bool
        Whether to verify compilation at each step.
    model, provider, base_url : str | None
        AI provider overrides.
    scene : str
        Scene preset for rendering.
    width, height : int
        Render resolution for comparison.

    Returns
    -------
    MatchResult
        Final result with iteration history and comparison metrics.
    """
    import numpy as np
    from luxc.ai.generate import (
        generate_material_from_image,
        _encode_image,
        _resolve_provider,
        extract_code,
        verify_lux_source_structured,
    )

    iterations: list[dict] = []

    # Step 1: Initial generation from reference image
    result = generate_material_from_image(
        reference_image_path,
        description=description,
        verify=verify,
        model=model,
        max_retries=2,
        provider=provider,
        base_url=base_url,
    )

    iterations.append({
        "iteration": 0,
        "source": result.lux_source,
        "compilation_success": result.compilation_success,
        "comparison": None,
    })

    if not result.compilation_success:
        return MatchResult(
            final_result=result,
            iterations=iterations,
            comparison=None,
            converged=False,
        )

    # Load reference image for comparison
    try:
        from PIL import Image
        ref_img = Image.open(reference_image_path).convert("RGB")
        ref_array = np.array(ref_img)
    except ImportError:
        ref_array = None

    comparison = None
    converged = False

    # Step 2-5: Iterative refinement loop
    ai_provider, config = _resolve_provider(provider, model, base_url)

    for i in range(1, max_iterations):
        # Try to render current shader
        rendered = render_lux_to_image(result.lux_source, scene, width, height)

        if rendered is not None and ref_array is not None:
            comparison = compare_images(ref_array, rendered[:, :, :3])
            if comparison.converged:
                converged = True
                iterations[-1]["comparison"] = {
                    "psnr": comparison.psnr,
                    "ssim": comparison.ssim,
                    "mae": comparison.mean_abs_error,
                }
                break

        # Build refinement prompt
        ref_b64, ref_media = _encode_image(reference_image_path)

        refinement_content: list[dict] = [
            {"type": "image_base64", "data": ref_b64, "media_type": ref_media},
            {"type": "text", "text": (
                f"Iteration {i}/{max_iterations}. The current Lux shader:\n\n"
                f"```lux\n{result.lux_source}\n```\n\n"
            )},
        ]

        if comparison is not None:
            refinement_content.append({
                "type": "text",
                "text": (
                    f"Comparison metrics: PSNR={comparison.psnr:.1f}, "
                    f"SSIM={comparison.ssim:.3f}, MAE={comparison.mean_abs_error:.1f}\n"
                    "Please refine the material to better match the reference image. "
                    "Adjust albedo, roughness, metallic, and layer parameters."
                ),
            })
        else:
            refinement_content.append({
                "type": "text",
                "text": (
                    "Rendering comparison is not available. "
                    "Please visually compare and refine the material parameters "
                    "to better match the reference image. "
                    "Output the complete, corrected Lux program."
                ),
            })

        from luxc.ai.system_prompt import build_material_extraction_prompt
        system_prompt = build_material_extraction_prompt()

        messages = [{"role": "user", "content": refinement_content}]

        if ai_provider.supports_vision:
            raw = ai_provider.complete_multimodal(
                system_prompt, messages, config.max_tokens
            )
        else:
            # Fallback to text-only refinement
            text_msg = refinement_content[-1]["text"]
            messages = [{"role": "user", "content": text_msg}]
            raw = ai_provider.complete(system_prompt, messages, config.max_tokens)

        lux_source = extract_code(raw)

        if verify:
            from luxc.builtins.types import clear_type_aliases
            clear_type_aliases()
            success, errors = verify_lux_source_structured(lux_source)
        else:
            success = True
            errors = []

        from luxc.ai.generate import GenerateResult
        result = GenerateResult(
            lux_source=lux_source,
            compilation_success=success,
            errors=errors,
            attempts=1,
        )

        iter_info: dict = {
            "iteration": i,
            "source": lux_source,
            "compilation_success": success,
            "comparison": None,
        }
        if comparison is not None:
            iter_info["comparison"] = {
                "psnr": comparison.psnr,
                "ssim": comparison.ssim,
                "mae": comparison.mean_abs_error,
            }
        iterations.append(iter_info)

    return MatchResult(
        final_result=result,
        iterations=iterations,
        comparison=comparison,
        converged=converged,
    )
