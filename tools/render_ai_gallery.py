#!/usr/bin/env python3
"""Render AI material gallery — sphere preview screenshots for documentation.

Generates a grid of PBR material sphere renders for the AI.md documentation.
Each material demonstrates a different AI feature or material type.

Usage:
    python tools/render_ai_gallery.py
    python tools/render_ai_gallery.py --size 384
    python tools/render_ai_gallery.py --output screenshots/ai_gallery.png
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHADERCACHE = PROJECT_ROOT / "shadercache"
SCREENSHOTS = PROJECT_ROOT / "screenshots"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "playground"))

# ---------------------------------------------------------------------------
# Boilerplate: geometry + lighting + pipeline wrapping each surface
# ---------------------------------------------------------------------------

_GEOMETRY = """\
import brdf;

geometry StandardMesh {
    position: vec3,
    normal: vec3,
    uv: vec2,
    transform: MVP {
        model: mat4,
        view: mat4,
        projection: mat4,
    }
    outputs {
        world_pos: (model * vec4(position, 1.0)).xyz,
        world_normal: normalize((model * vec4(normal, 0.0)).xyz),
        frag_uv: uv,
        clip_pos: projection * view * model * vec4(position, 1.0),
    }
}
"""

_PIPELINE_TEMPLATE = """
pipeline Forward {{
    geometry: StandardMesh,
    surface: {surface_name},
}}
"""


def _make_brdf_source(
    name: str,
    albedo: tuple[float, float, float],
    roughness: float,
    metallic: float,
) -> str:
    """Build a complete compilable Lux program for a PBR material.

    Uses the layers syntax with an emission layer as ambient fill so metallic
    materials are visible without an environment map (metals have zero diffuse
    in PBR, so without ambient they appear black except at the specular highlight).
    """
    alb = f"vec3({albedo[0]:.3f}, {albedo[1]:.3f}, {albedo[2]:.3f})"
    base_color = f"{alb} * sample(albedo_tex, frag_uv).xyz"
    # Ambient via emission — fills dark regions (essential for metals with no diffuse)
    ambient_strength = 0.12 if metallic > 0.5 else 0.03
    ambient_color = f"{alb} * {ambient_strength:.2f}"
    surface = (
        f"surface {name} {{\n"
        f"    sampler2d albedo_tex,\n"
        f"    layers [\n"
        f"        base(albedo: {base_color}, roughness: {roughness:.3f}, metallic: {metallic:.1f}),\n"
        f"        emission(color: {ambient_color}),\n"
        f"    ]\n"
        f"}}\n"
    )
    pipeline = _PIPELINE_TEMPLATE.format(surface_name=name)
    return _GEOMETRY + "\n" + surface + pipeline


# ---------------------------------------------------------------------------
# Material definitions — each represents a feature/material type
# ---------------------------------------------------------------------------

MATERIALS: dict[str, dict] = {
    # --- Metals (roughness boosted for single-light visibility) ---
    "polished_gold":   {"label": "Polished Gold",   "category": "Text-to-Shader",    "albedo": (1.00, 0.77, 0.34), "roughness": 0.20, "metallic": 1.0},
    "brushed_copper":  {"label": "Brushed Copper",  "category": "Text-to-Shader",    "albedo": (0.93, 0.62, 0.52), "roughness": 0.35, "metallic": 1.0},
    "mirror_silver":   {"label": "Mirror Silver",   "category": "Text-to-Shader",    "albedo": (0.99, 0.99, 0.97), "roughness": 0.15, "metallic": 1.0},
    "rough_iron":      {"label": "Rough Iron",      "category": "AI Critique",       "albedo": (0.53, 0.51, 0.49), "roughness": 0.60, "metallic": 1.0},
    # --- Dielectrics ---
    "red_plastic":     {"label": "Red Plastic",     "category": "Text-to-Shader",    "albedo": (0.75, 0.05, 0.05), "roughness": 0.15, "metallic": 0.0},
    "rough_concrete":  {"label": "Rough Concrete",  "category": "Batch Generation",  "albedo": (0.51, 0.49, 0.46), "roughness": 0.85, "metallic": 0.0},
    "glazed_ceramic":  {"label": "Glazed Ceramic",  "category": "Image-to-Material", "albedo": (0.85, 0.85, 0.82), "roughness": 0.08, "metallic": 0.0},
    "dark_wood":       {"label": "Dark Wood",       "category": "Style Transfer",    "albedo": (0.43, 0.24, 0.11), "roughness": 0.65, "metallic": 0.0},
    # --- Special ---
    "brass_satin":     {"label": "Satin Brass",     "category": "Batch Generation",  "albedo": (0.91, 0.78, 0.42), "roughness": 0.30, "metallic": 1.0},
    "blue_tint":       {"label": "Frosted Glass",   "category": "Text-to-Shader",    "albedo": (0.85, 0.92, 1.00), "roughness": 0.40, "metallic": 0.0},
    "marble":          {"label": "Polished Marble",  "category": "Reference Match",  "albedo": (0.83, 0.79, 0.75), "roughness": 0.10, "metallic": 0.0},
    "charcoal":        {"label": "Charcoal",        "category": "AI Critique",       "albedo": (0.05, 0.05, 0.05), "roughness": 0.90, "metallic": 0.0},
}


def compile_material(name: str, mat: dict) -> bool:
    """Compile a Lux surface material to SPIR-V."""
    surface_name = "Mat_" + name.replace(" ", "")
    full_source = _make_brdf_source(
        surface_name,
        mat["albedo"],
        mat["roughness"],
        mat["metallic"],
    )
    lux_path = SHADERCACHE / f"ai_{name}.lux"
    lux_path.parent.mkdir(parents=True, exist_ok=True)
    lux_path.write_text(full_source, encoding="utf-8")

    cmd = [
        sys.executable, "-m", "luxc", str(lux_path),
        "-o", str(SHADERCACHE),
        "--pipeline", "Forward",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))

    if result.returncode != 0:
        print(f"  FAIL: {name} — {result.stderr.strip()}")
        return False

    vert = SHADERCACHE / f"ai_{name}.vert.spv"
    frag = SHADERCACHE / f"ai_{name}.frag.spv"
    if not vert.exists() or not frag.exists():
        print(f"  FAIL: {name} — missing .spv files")
        return False

    return True


def generate_neutral_texture(size: int = 512) -> np.ndarray:
    """Generate a neutral, mostly-white texture with subtle variation.

    A very subtle soft gradient so materials show their true color
    while still demonstrating texture sampling works.
    """
    img = np.full((size, size, 4), 255, dtype=np.uint8)  # start white
    for y in range(size):
        for x in range(size):
            u = x / size
            v = y / size
            # Very subtle warm-to-cool shift (stays near white)
            r = int(250 + 5 * np.sin(u * 3.14))
            g = int(248 + 5 * np.sin(v * 3.14))
            b = int(245 + 8 * np.cos(u * 2.0 + v * 2.0))
            img[y, x] = [min(255, max(220, r)), min(255, max(220, g)), min(255, max(220, b)), 255]
    return img


# Cache the texture so we only generate it once
_NEUTRAL_TEX = None


def get_neutral_texture() -> np.ndarray:
    global _NEUTRAL_TEX
    if _NEUTRAL_TEX is None:
        print("  Generating neutral texture...")
        _NEUTRAL_TEX = generate_neutral_texture(512)
    return _NEUTRAL_TEX


def render_sphere(name: str, size: int = 384) -> "np.ndarray | None":
    """Render a material on a sphere and return the pixel array."""
    from render_pbr import render_pbr
    vert = SHADERCACHE / f"ai_{name}.vert.spv"
    frag = SHADERCACHE / f"ai_{name}.frag.spv"
    try:
        tex = get_neutral_texture()
        # Frontal-ish light — specular highlights visible for metals
        return render_pbr(
            vert, frag, width=size, height=size,
            texture_data=tex, light_dir=(0.4, 0.5, 1.0),
        )
    except Exception as exc:
        print(f"  RENDER FAIL: {name} — {exc}")
        return None


def add_label(img, label: str, category: str = ""):
    """Add text label to bottom of image using PIL."""
    from PIL import Image, ImageDraw, ImageFont

    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    # Try to load a better font, fall back to default
    try:
        font = ImageFont.truetype("arial.ttf", 16)
        font_small = ImageFont.truetype("arial.ttf", 12)
    except (IOError, OSError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except (IOError, OSError):
            font = ImageFont.load_default()
            font_small = font

    h = pil_img.height
    w = pil_img.width

    # Semi-transparent dark background for label area (fast numpy path)
    arr = np.array(pil_img)
    arr[h - 44:h, :, :3] = arr[h - 44:h, :, :3] // 3
    arr[h - 44:h, :, 3] = 255
    pil_img = Image.fromarray(arr)
    draw = ImageDraw.Draw(pil_img)

    # Draw text
    draw.text((8, h - 40), label, fill=(255, 255, 255, 255), font=font)
    if category:
        draw.text((8, h - 20), category, fill=(180, 180, 180, 255), font=font_small)

    return np.array(pil_img)


def compose_grid(images: list, labels: list, categories: list, cols: int = 4) -> "np.ndarray":
    """Compose images into a grid with labels."""
    from PIL import Image

    rows_count = (len(images) + cols - 1) // cols

    # Add labels
    labeled = []
    for img, label, cat in zip(images, labels, categories):
        labeled.append(add_label(img, label, cat))

    cell_h, cell_w = labeled[0].shape[:2]
    gap = 4

    grid_w = cols * cell_w + (cols - 1) * gap
    grid_h = rows_count * cell_h + (rows_count - 1) * gap

    grid = np.full((grid_h, grid_w, 4), 30, dtype=np.uint8)
    grid[:, :, 3] = 255

    for i, img in enumerate(labeled):
        row = i // cols
        col = i % cols
        y = row * (cell_h + gap)
        x = col * (cell_w + gap)
        grid[y:y + cell_h, x:x + cell_w] = img

    return grid


def main():
    from render_harness import save_png

    parser = argparse.ArgumentParser(description="Render AI material gallery")
    parser.add_argument("--size", type=int, default=384, help="Sphere render size")
    parser.add_argument("--output", type=Path, default=SCREENSHOTS / "ai_gallery.png")
    parser.add_argument("--cols", type=int, default=4, help="Grid columns")
    args = parser.parse_args()

    SCREENSHOTS.mkdir(parents=True, exist_ok=True)
    SHADERCACHE.mkdir(parents=True, exist_ok=True)

    # Phase 1: Compile all materials
    print("Compiling materials...")
    compiled = []
    for name, mat in MATERIALS.items():
        ok = compile_material(name, mat)
        status = "ok" if ok else "FAIL"
        print(f"  [{status}] {name}")
        if ok:
            compiled.append(name)

    if not compiled:
        print("No materials compiled successfully.")
        sys.exit(1)

    # Phase 2: Render spheres
    print(f"\nRendering {len(compiled)} spheres at {args.size}x{args.size}...")
    images = []
    labels = []
    categories = []
    individual_dir = SCREENSHOTS / "ai_materials"
    individual_dir.mkdir(parents=True, exist_ok=True)

    for name in compiled:
        mat = MATERIALS[name]
        print(f"  Rendering {name}...")
        pixels = render_sphere(name, size=args.size)
        if pixels is not None:
            images.append(pixels)
            labels.append(mat["label"])
            categories.append(mat["category"])
            # Save individual screenshot too
            save_png(pixels, individual_dir / f"{name}.png")

    if not images:
        print("No renders succeeded.")
        sys.exit(1)

    # Phase 3: Compose gallery
    print(f"\nComposing {len(images)}-cell gallery...")
    grid = compose_grid(images, labels, categories, cols=args.cols)
    save_png(grid, args.output)
    print(f"Saved gallery: {args.output} ({args.output.stat().st_size // 1024} KB)")
    print(f"Saved {len(images)} individual renders to {individual_dir}/")


if __name__ == "__main__":
    main()
