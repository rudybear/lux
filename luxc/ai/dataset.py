"""Synthetic dataset generation for AI training.

Generates parametric and random physically-plausible material variants
from the PBR material database.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ShaderVariant:
    """A single material variant with metadata."""
    name: str
    description: str
    lux_source: str
    parameters: dict[str, float] = field(default_factory=dict)


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _format_vec3(r: float, g: float, b: float) -> str:
    return f"vec3({r:.3f}, {g:.3f}, {b:.3f})"


def _make_surface_source(
    name: str,
    albedo: tuple[float, float, float],
    roughness: float,
    metallic: float,
    ior: float | None = None,
    transmission: bool = False,
    coat_factor: float = 0.0,
    coat_roughness: float = 0.0,
) -> str:
    """Generate a complete Lux surface declaration."""
    lines = [f"surface {name} {{"]
    lines.append(f"    properties MatParams {{")
    lines.append(f"        base_color: vec3 = {_format_vec3(*albedo)},")
    lines.append(f"        roughness: scalar = {roughness:.3f},")
    if coat_factor > 0:
        lines.append(f"        coat_factor: scalar = {coat_factor:.3f},")
        lines.append(f"        coat_roughness: scalar = {coat_roughness:.3f},")
    if transmission and ior is not None:
        lines.append(f"        ior_value: scalar = {ior:.3f},")
    lines.append(f"    }},")
    lines.append(f"    layers [")
    lines.append(f"        base(")
    lines.append(f"            albedo: MatParams.base_color,")
    lines.append(f"            roughness: MatParams.roughness,")
    lines.append(f"            metallic: {metallic:.1f}")
    lines.append(f"        ),")
    if transmission and ior is not None:
        lines.append(f"        transmission(factor: 1.0, ior: MatParams.ior_value),")
    if coat_factor > 0:
        lines.append(f"        coat(factor: MatParams.coat_factor, roughness: MatParams.coat_roughness),")
    lines.append(f"    ]")
    lines.append(f"}}")
    return "\n".join(lines)


def expand_material_variants(
    base_material: str,
    roughness_steps: int = 5,
    albedo_variations: int = 3,
    layer_combinations: list[str] | None = None,
) -> list[ShaderVariant]:
    """Generate parametric variations of a PBR material entry.

    Parameters
    ----------
    base_material : str
        Name of a material in PBR_MATERIALS (e.g. "Copper").
    roughness_steps : int
        Number of roughness levels to sweep (default: 5).
    albedo_variations : int
        Number of albedo brightness variations (default: 3).
    layer_combinations : list[str] | None
        Optional layer combos to add (e.g. ["coat", "sheen"]).

    Returns
    -------
    list[ShaderVariant]
        Generated material variants.
    """
    from luxc.ai.materials import PBR_MATERIALS

    if base_material not in PBR_MATERIALS:
        raise ValueError(f"Unknown material: {base_material}")

    props = PBR_MATERIALS[base_material]
    base_albedo = props["albedo"]
    base_metallic = float(props["metallic"])
    base_ior = props.get("ior")
    base_transmission = bool(props.get("transmission", 0))

    if layer_combinations is None:
        layer_combinations = []

    variants: list[ShaderVariant] = []

    # Roughness sweep
    for ri in range(roughness_steps):
        roughness = ri / max(roughness_steps - 1, 1)

        # Albedo variations (brightness scaling)
        for ai in range(albedo_variations):
            if albedo_variations <= 1:
                scale = 1.0
            else:
                scale = 0.7 + 0.6 * (ai / (albedo_variations - 1))

            albedo = tuple(_clamp(c * scale) for c in base_albedo)
            safe_name = base_material.replace(" ", "").replace("(", "").replace(")", "")
            name = f"{safe_name}_R{ri}_A{ai}"

            # Base variant
            source = _make_surface_source(
                name=name,
                albedo=albedo,
                roughness=roughness,
                metallic=base_metallic,
                ior=base_ior,
                transmission=base_transmission,
            )
            variants.append(ShaderVariant(
                name=name,
                description=f"{base_material} roughness={roughness:.2f} brightness={scale:.2f}",
                lux_source=source,
                parameters={
                    "roughness": roughness,
                    "metallic": base_metallic,
                    "albedo_r": albedo[0],
                    "albedo_g": albedo[1],
                    "albedo_b": albedo[2],
                },
            ))

            # Layer combination variants
            for layer in layer_combinations:
                layer_name = f"{name}_{layer}"
                coat_f = 0.0
                coat_r = 0.0
                if layer == "coat":
                    coat_f = 0.8
                    coat_r = 0.05

                source = _make_surface_source(
                    name=layer_name,
                    albedo=albedo,
                    roughness=roughness,
                    metallic=base_metallic,
                    ior=base_ior,
                    transmission=base_transmission,
                    coat_factor=coat_f,
                    coat_roughness=coat_r,
                )
                variants.append(ShaderVariant(
                    name=layer_name,
                    description=f"{base_material} roughness={roughness:.2f} +{layer}",
                    lux_source=source,
                    parameters={
                        "roughness": roughness,
                        "metallic": base_metallic,
                        "albedo_r": albedo[0],
                        "albedo_g": albedo[1],
                        "albedo_b": albedo[2],
                        f"{layer}_factor": coat_f,
                    },
                ))

    return variants


def generate_random_materials(
    count: int = 100,
    seed: int = 42,
) -> list[ShaderVariant]:
    """Generate constrained random physically-plausible materials.

    Parameters
    ----------
    count : int
        Number of materials to generate (default: 100).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[ShaderVariant]
        Generated random materials with physically-plausible parameters.
    """
    rng = random.Random(seed)
    variants: list[ShaderVariant] = []

    for i in range(count):
        # Decide metal vs dielectric (30% chance of metal)
        is_metal = rng.random() < 0.3
        metallic = 1.0 if is_metal else 0.0

        if is_metal:
            # Metal: reflectance in [0.5, 1.0] range, any hue
            r = rng.uniform(0.4, 1.0)
            g = rng.uniform(0.3, 1.0)
            b = rng.uniform(0.3, 1.0)
        else:
            # Dielectric: albedo in [0.02, 0.95] range
            r = rng.uniform(0.02, 0.95)
            g = rng.uniform(0.02, 0.95)
            b = rng.uniform(0.02, 0.95)

        roughness = rng.uniform(0.0, 1.0)

        # 15% chance of transmission (dielectrics only)
        transmission = not is_metal and rng.random() < 0.15
        ior = rng.uniform(1.3, 2.5) if transmission else None

        # 20% chance of coat (any material)
        has_coat = rng.random() < 0.2
        coat_f = rng.uniform(0.3, 1.0) if has_coat else 0.0
        coat_r = rng.uniform(0.01, 0.15) if has_coat else 0.0

        name = f"RandomMat_{i:04d}"
        source = _make_surface_source(
            name=name,
            albedo=(r, g, b),
            roughness=roughness,
            metallic=metallic,
            ior=ior,
            transmission=transmission,
            coat_factor=coat_f,
            coat_roughness=coat_r,
        )

        params = {
            "roughness": roughness,
            "metallic": metallic,
            "albedo_r": r,
            "albedo_g": g,
            "albedo_b": b,
        }
        if transmission:
            params["ior"] = ior
        if has_coat:
            params["coat_factor"] = coat_f
            params["coat_roughness"] = coat_r

        desc_parts = ["metal" if is_metal else "dielectric"]
        if transmission:
            desc_parts.append(f"transmissive (IOR={ior:.2f})")
        if has_coat:
            desc_parts.append(f"coated ({coat_f:.2f})")

        variants.append(ShaderVariant(
            name=name,
            description=f"Random {', '.join(desc_parts)}, roughness={roughness:.2f}",
            lux_source=source,
            parameters=params,
        ))

    return variants
