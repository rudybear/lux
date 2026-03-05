"""Multi-pixel heatmap runner for shader debugging.

Runs the shader interpreter across a grid of pixels, collecting output
colors, NaN events, statement counts, and variable values. Outputs PPM
images for visual analysis without any external dependencies.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

from luxc.parser.tree_builder import parse_lux
from luxc.parser.ast_nodes import Module, StageBlock
from luxc.debug.values import (
    LuxValue, LuxScalar, LuxVec, LuxInt, LuxBool, LuxImage,
    default_value, is_nan, make_solid_image,
)
from luxc.debug.interpreter import Interpreter
from luxc.debug.io import build_default_inputs, load_inputs_from_json


@dataclass
class PixelResult:
    """Result from running the shader at a single pixel."""
    x: int
    y: int
    u: float
    v: float
    output: LuxVec | None = None
    nan_count: int = 0
    statement_count: int = 0
    var_value: float = 0.0  # magnitude of tracked variable


@dataclass
class HeatmapResult:
    """Full grid of pixel results."""
    width: int
    height: int
    pixels: list[PixelResult] = field(default_factory=list)
    mode: str = "output"


def _find_stage(module: Module, stage_name: str) -> StageBlock | None:
    for stage in module.stages:
        if stage.stage_type == stage_name:
            return stage
    return None


def run_pixel_grid(
    source: str,
    stage_name: str,
    resolution: tuple[int, int] = (512, 512),
    grid_size: tuple[int, int] = (16, 16),
    mode: str = "output",
    var_name: str | None = None,
    check_nan: bool = False,
    input_path: Path | None = None,
) -> HeatmapResult:
    """Run the shader interpreter across a pixel grid.

    Args:
        source: Lux shader source text.
        stage_name: Stage to run (usually "fragment").
        resolution: Virtual screen resolution (W, H).
        grid_size: Number of sample points (cols, rows).
        mode: "output" | "nan" | "perf" | "var:<name>".
        var_name: Variable name to track (for mode="var:...").
        check_nan: Whether to track NaN events.
        input_path: Optional JSON input overrides.

    Returns:
        HeatmapResult with collected pixel data.
    """
    from luxc.analysis.type_checker import type_check
    from luxc.optimization.const_fold import constant_fold

    module = parse_lux(source)
    if module.surfaces or module.pipelines or module.environments or module.procedurals:
        from luxc.expansion.surface_expander import expand_surfaces
        expand_surfaces(module)
    type_check(module)
    constant_fold(module)

    stage = _find_stage(module, stage_name)
    if stage is None:
        raise ValueError(f"Stage '{stage_name}' not found")

    base_inputs = build_default_inputs(stage)
    if input_path:
        user_inputs = load_inputs_from_json(input_path)
        base_inputs.update(user_inputs)

    gw, gh = grid_size
    res_w, res_h = resolution
    result = HeatmapResult(width=gw, height=gh, mode=mode)

    for gy in range(gh):
        for gx in range(gw):
            # Compute UV for this grid cell
            u = (gx + 0.5) / gw
            v = (gy + 0.5) / gh

            # Build pixel-specific inputs
            pixel_inputs = copy.deepcopy(base_inputs)
            pixel_inputs["uv"] = LuxVec([u, v])
            pixel_inputs["texcoord"] = LuxVec([u, v])

            # World position from UV
            pixel_inputs["world_position"] = LuxVec([
                (u - 0.5) * 2.0, (v - 0.5) * 2.0, 0.0
            ])
            pixel_inputs["position"] = LuxVec([
                (u - 0.5) * 2.0, (v - 0.5) * 2.0, 0.0
            ])

            # Pixel coordinates
            px = int(u * res_w)
            py = int(v * res_h)

            lines = source.splitlines()
            interp = Interpreter(module, stage, source_lines=lines)

            try:
                interp_result = interp.run(inputs=pixel_inputs, trace_all=False)

                pr = PixelResult(x=gx, y=gy, u=u, v=v)
                pr.statement_count = interp_result.statements_executed
                pr.nan_count = len(interp_result.nan_events)

                if interp_result.output and isinstance(interp_result.output, LuxVec):
                    pr.output = interp_result.output

                # Track variable magnitude
                if var_name:
                    val = interp.env.get(var_name)
                    if val is not None:
                        if isinstance(val, LuxScalar):
                            pr.var_value = abs(val.value)
                        elif isinstance(val, LuxVec):
                            pr.var_value = sum(c * c for c in val.components) ** 0.5

                result.pixels.append(pr)
            except Exception:
                result.pixels.append(PixelResult(x=gx, y=gy, u=u, v=v))

    return result


def write_ppm(path: str, result: HeatmapResult) -> None:
    """Write a HeatmapResult as a P6 binary PPM image."""
    w, h = result.width, result.height

    # Map pixels to RGB based on mode
    rgb_data = bytearray()
    for pr in result.pixels:
        r, g, b = _pixel_to_rgb(pr, result.mode, result)
        rgb_data.extend([r, g, b])

    with open(path, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        f.write(rgb_data)


def _pixel_to_rgb(
    pr: PixelResult, mode: str, result: HeatmapResult
) -> tuple[int, int, int]:
    """Convert a pixel result to an RGB tuple based on visualization mode."""
    if mode == "output":
        if pr.output and isinstance(pr.output, LuxVec):
            c = pr.output.components
            r = _clamp_byte(c[0])
            g = _clamp_byte(c[1]) if len(c) > 1 else 0
            b = _clamp_byte(c[2]) if len(c) > 2 else 0
            return (r, g, b)
        return (0, 0, 0)

    elif mode == "nan":
        if pr.nan_count > 0:
            return (255, 0, 0)  # red = NaN present
        return (0, 200, 0)  # green = clean

    elif mode == "perf":
        # Heatmap: blue (few) → red (many statements)
        if not result.pixels:
            return (0, 0, 0)
        max_stmts = max(p.statement_count for p in result.pixels) or 1
        t = pr.statement_count / max_stmts
        return _heat_color(t)

    elif mode.startswith("var:"):
        if not result.pixels:
            return (0, 0, 0)
        max_val = max(p.var_value for p in result.pixels) or 1.0
        t = pr.var_value / max_val
        return _heat_color(t)

    return (128, 128, 128)


def _clamp_byte(f: float) -> int:
    """Clamp a float [0,1] to a byte [0,255]."""
    return max(0, min(255, int(f * 255.0 + 0.5)))


def _heat_color(t: float) -> tuple[int, int, int]:
    """Map a [0,1] value to a blue→cyan→green→yellow→red heatmap."""
    t = max(0.0, min(1.0, t))
    if t < 0.25:
        # Blue → Cyan
        s = t / 0.25
        return (0, int(255 * s), 255)
    elif t < 0.5:
        # Cyan → Green
        s = (t - 0.25) / 0.25
        return (0, 255, int(255 * (1 - s)))
    elif t < 0.75:
        # Green → Yellow
        s = (t - 0.5) / 0.25
        return (int(255 * s), 255, 0)
    else:
        # Yellow → Red
        s = (t - 0.75) / 0.25
        return (255, int(255 * (1 - s)), 0)
