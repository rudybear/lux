"""PBR material reference data for AI shader generation.

Contains physically measured material properties (albedo, roughness, metallic,
IOR, transmission) sourced from the Physically Based database (CC0 license,
physicallybased.info) and supplemented with well-known Unreal Engine reference
values.

All albedo values are in **linear sRGB** colour space.  Roughness and metallic
follow the standard glTF metallic-roughness parameterisation.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Material database
# ---------------------------------------------------------------------------
# Each entry maps a human-readable name to a dict with:
#   albedo     - (R, G, B) tuple in linear space, 0-1 per channel
#   roughness  - 0.0 (mirror) .. 1.0 (fully diffuse)
#   metallic   - 0 or 1
#   ior        - index of refraction (optional, omitted for metals)
#   transmission - 0 or 1 (optional, 1 = transmissive)
#   category   - grouping label
# ---------------------------------------------------------------------------

PBR_MATERIALS: dict[str, dict] = {
    # ------------------------------------------------------------------
    # Metals  (metallic = 1, roughness = 0 means "polished / ideal")
    # ------------------------------------------------------------------
    "Aluminum": {
        "albedo": (0.91, 0.92, 0.92),
        "roughness": 0.0,
        "metallic": 1,
        "category": "Metal",
    },
    "Brass": {
        "albedo": (0.91, 0.78, 0.42),
        "roughness": 0.0,
        "metallic": 1,
        "category": "Metal",
    },
    "Chromium": {
        "albedo": (0.65, 0.69, 0.70),
        "roughness": 0.0,
        "metallic": 1,
        "category": "Metal",
    },
    "Cobalt": {
        "albedo": (0.70, 0.70, 0.67),
        "roughness": 0.0,
        "metallic": 1,
        "category": "Metal",
    },
    "Copper": {
        "albedo": (0.93, 0.62, 0.52),
        "roughness": 0.0,
        "metallic": 1,
        "category": "Metal",
    },
    "Gold": {
        "albedo": (1.00, 0.77, 0.34),
        "roughness": 0.0,
        "metallic": 1,
        "category": "Metal",
    },
    "Iron": {
        "albedo": (0.53, 0.51, 0.49),
        "roughness": 0.0,
        "metallic": 1,
        "category": "Metal",
    },
    "Lead": {
        "albedo": (0.63, 0.64, 0.69),
        "roughness": 0.0,
        "metallic": 1,
        "category": "Metal",
    },
    "Mercury": {
        "albedo": (0.78, 0.78, 0.78),
        "roughness": 0.0,
        "metallic": 1,
        "category": "Metal",
    },
    "Nickel": {
        "albedo": (0.70, 0.64, 0.56),
        "roughness": 0.0,
        "metallic": 1,
        "category": "Metal",
    },
    "Palladium": {
        "albedo": (0.73, 0.70, 0.66),
        "roughness": 0.0,
        "metallic": 1,
        "category": "Metal",
    },
    "Platinum": {
        "albedo": (0.77, 0.73, 0.68),
        "roughness": 0.0,
        "metallic": 1,
        "category": "Metal",
    },
    "Silver": {
        "albedo": (0.99, 0.99, 0.97),
        "roughness": 0.0,
        "metallic": 1,
        "category": "Metal",
    },
    "Stainless Steel": {
        "albedo": (0.67, 0.64, 0.60),
        "roughness": 0.0,
        "metallic": 1,
        "category": "Metal",
    },
    "Titanium": {
        "albedo": (0.44, 0.40, 0.36),
        "roughness": 0.0,
        "metallic": 1,
        "category": "Metal",
    },
    "Tungsten": {
        "albedo": (0.54, 0.54, 0.52),
        "roughness": 0.0,
        "metallic": 1,
        "category": "Metal",
    },
    "Zinc": {
        "albedo": (0.81, 0.84, 0.87),
        "roughness": 0.0,
        "metallic": 1,
        "category": "Metal",
    },

    # ------------------------------------------------------------------
    # Crystals & Gems
    # ------------------------------------------------------------------
    "Diamond": {
        "albedo": (1.00, 1.00, 1.00),
        "roughness": 0.0,
        "metallic": 0,
        "ior": 2.42,
        "transmission": 1,
        "category": "Crystal",
    },
    "Glass": {
        "albedo": (1.00, 1.00, 1.00),
        "roughness": 0.0,
        "metallic": 0,
        "ior": 1.52,
        "transmission": 1,
        "category": "Crystal",
    },
    "Ice": {
        "albedo": (1.00, 1.00, 1.00),
        "roughness": 0.5,
        "metallic": 0,
        "ior": 1.31,
        "transmission": 1,
        "category": "Crystal",
    },
    "Marble": {
        "albedo": (0.83, 0.79, 0.75),
        "roughness": 0.0,
        "metallic": 0,
        "ior": 1.50,
        "category": "Crystal",
    },
    "Quartz": {
        "albedo": (1.00, 1.00, 1.00),
        "roughness": 0.0,
        "metallic": 0,
        "ior": 1.54,
        "transmission": 1,
        "category": "Crystal",
    },
    "Salt": {
        "albedo": (1.00, 1.00, 1.00),
        "roughness": 0.2,
        "metallic": 0,
        "ior": 1.54,
        "transmission": 1,
        "category": "Crystal",
    },
    "Sand": {
        "albedo": (0.44, 0.39, 0.23),
        "roughness": 0.9,
        "metallic": 0,
        "category": "Crystal",
    },
    "Sapphire": {
        "albedo": (0.67, 0.76, 0.86),
        "roughness": 0.0,
        "metallic": 0,
        "ior": 1.77,
        "transmission": 1,
        "category": "Crystal",
    },
    "Snow": {
        "albedo": (0.85, 0.85, 0.85),
        "roughness": 0.5,
        "metallic": 0,
        "ior": 1.31,
        "category": "Crystal",
    },

    # ------------------------------------------------------------------
    # Liquids
    # ------------------------------------------------------------------
    "Water": {
        "albedo": (1.00, 1.00, 1.00),
        "roughness": 0.0,
        "metallic": 0,
        "ior": 1.33,
        "transmission": 1,
        "category": "Liquid",
    },
    "Blood": {
        "albedo": (0.64, 0.003, 0.005),
        "roughness": 0.0,
        "metallic": 0,
        "ior": 1.35,
        "transmission": 1,
        "category": "Liquid",
    },
    "Coffee": {
        "albedo": (0.45, 0.13, 0.03),
        "roughness": 0.0,
        "metallic": 0,
        "ior": 1.34,
        "transmission": 1,
        "category": "Liquid",
    },
    "Honey": {
        "albedo": (0.83, 0.57, 0.04),
        "roughness": 0.0,
        "metallic": 0,
        "ior": 1.50,
        "transmission": 1,
        "category": "Liquid",
    },
    "Milk": {
        "albedo": (0.82, 0.81, 0.68),
        "roughness": 0.0,
        "metallic": 0,
        "ior": 1.35,
        "category": "Liquid",
    },

    # ------------------------------------------------------------------
    # Plastics
    # ------------------------------------------------------------------
    "Plastic (Acrylic)": {
        "albedo": (1.00, 1.00, 1.00),
        "roughness": 0.0,
        "metallic": 0,
        "ior": 1.49,
        "transmission": 1,
        "category": "Plastic",
    },
    "Plastic (PVC)": {
        "albedo": (1.00, 1.00, 1.00),
        "roughness": 0.0,
        "metallic": 0,
        "ior": 1.54,
        "transmission": 1,
        "category": "Plastic",
    },

    # ------------------------------------------------------------------
    # Organic
    # ------------------------------------------------------------------
    "Bone": {
        "albedo": (0.79, 0.79, 0.66),
        "roughness": 0.9,
        "metallic": 0,
        "category": "Organic",
    },
    "Charcoal": {
        "albedo": (0.02, 0.02, 0.02),
        "roughness": 0.9,
        "metallic": 0,
        "category": "Organic",
    },
    "Chocolate": {
        "albedo": (0.16, 0.09, 0.06),
        "roughness": 0.5,
        "metallic": 0,
        "category": "Organic",
    },
    "Egg Shell": {
        "albedo": (0.61, 0.62, 0.63),
        "roughness": 0.9,
        "metallic": 0,
        "category": "Organic",
    },
    "Pearl": {
        "albedo": (0.80, 0.75, 0.70),
        "roughness": 0.35,
        "metallic": 0,
        "ior": 1.68,
        "category": "Organic",
    },

    # ------------------------------------------------------------------
    # Human / Skin
    # ------------------------------------------------------------------
    "Skin (Light)": {
        "albedo": (0.85, 0.64, 0.55),
        "roughness": 0.5,
        "metallic": 0,
        "ior": 1.40,
        "category": "Human",
    },
    "Skin (Medium)": {
        "albedo": (0.62, 0.43, 0.34),
        "roughness": 0.5,
        "metallic": 0,
        "ior": 1.40,
        "category": "Human",
    },
    "Skin (Dark)": {
        "albedo": (0.28, 0.15, 0.08),
        "roughness": 0.5,
        "metallic": 0,
        "ior": 1.40,
        "category": "Human",
    },

    # ------------------------------------------------------------------
    # Man-made / Architectural
    # ------------------------------------------------------------------
    "Blackboard": {
        "albedo": (0.04, 0.04, 0.04),
        "roughness": 0.9,
        "metallic": 0,
        "category": "Manmade",
    },
    "Brick": {
        "albedo": (0.26, 0.10, 0.06),
        "roughness": 0.9,
        "metallic": 0,
        "category": "Manmade",
    },
    "Car Paint": {
        "albedo": (0.10, 0.10, 0.10),
        "roughness": 0.0,
        "metallic": 0,
        "category": "Manmade",
    },
    "Concrete": {
        "albedo": (0.51, 0.51, 0.51),
        "roughness": 0.5,
        "metallic": 0,
        "category": "Manmade",
    },
    "Office Paper": {
        "albedo": (0.79, 0.83, 0.88),
        "roughness": 0.8,
        "metallic": 0,
        "category": "Manmade",
    },
    "Porcelain": {
        "albedo": (0.75, 0.75, 0.72),
        "roughness": 0.0,
        "metallic": 0,
        "category": "Manmade",
    },
    "Tire": {
        "albedo": (0.02, 0.02, 0.02),
        "roughness": 0.7,
        "metallic": 0,
        "category": "Manmade",
    },
    "Whiteboard": {
        "albedo": (0.87, 0.87, 0.77),
        "roughness": 0.0,
        "metallic": 0,
        "category": "Manmade",
    },
    "Gray Card (18%)": {
        "albedo": (0.18, 0.18, 0.18),
        "roughness": 0.9,
        "metallic": 0,
        "category": "Manmade",
    },

    # ------------------------------------------------------------------
    # Supplementary: Unreal Engine non-metal base-colour intensities
    # (monochrome albedo, typical roughness estimated)
    # ------------------------------------------------------------------
    "Fresh Asphalt": {
        "albedo": (0.02, 0.02, 0.02),
        "roughness": 0.9,
        "metallic": 0,
        "category": "Ground",
    },
    "Worn Asphalt": {
        "albedo": (0.08, 0.08, 0.08),
        "roughness": 0.85,
        "metallic": 0,
        "category": "Ground",
    },
    "Bare Soil": {
        "albedo": (0.13, 0.13, 0.13),
        "roughness": 0.9,
        "metallic": 0,
        "category": "Ground",
    },
    "Green Grass": {
        "albedo": (0.21, 0.21, 0.21),
        "roughness": 0.8,
        "metallic": 0,
        "category": "Ground",
    },
    "Desert Sand": {
        "albedo": (0.36, 0.36, 0.36),
        "roughness": 0.9,
        "metallic": 0,
        "category": "Ground",
    },
    "Fresh Concrete": {
        "albedo": (0.51, 0.51, 0.51),
        "roughness": 0.5,
        "metallic": 0,
        "category": "Ground",
    },
    "Ocean Ice": {
        "albedo": (0.56, 0.56, 0.56),
        "roughness": 0.3,
        "metallic": 0,
        "category": "Ground",
    },
    "Fresh Snow": {
        "albedo": (0.81, 0.81, 0.81),
        "roughness": 0.5,
        "metallic": 0,
        "category": "Ground",
    },
}


# ---------------------------------------------------------------------------
# Roughness reference points (for the prompt)
# ---------------------------------------------------------------------------
_ROUGHNESS_REFERENCE = {
    "Mirror": 0.0,
    "Polished": 0.1,
    "Brushed": 0.3,
    "Rough": 0.6,
    "Matte": 0.9,
}


# ---------------------------------------------------------------------------
# Table builder
# ---------------------------------------------------------------------------

def _fmt_rgb(rgb: tuple[float, ...]) -> str:
    """Format an RGB tuple as a compact string."""
    return f"({rgb[0]:.2f}, {rgb[1]:.2f}, {rgb[2]:.2f})"


def build_material_reference() -> str:
    """Return a compact reference table suitable for embedding in an AI prompt.

    The table groups materials by category and includes all relevant PBR
    parameters.  Format is Markdown so it renders well in both raw text and
    rich displays.
    """
    lines: list[str] = []
    lines.append("## PBR Material Reference (Measured Values)")
    lines.append("")

    # -- Roughness cheat-sheet ---------------------------------------------
    lines.append("### Roughness Guide")
    for label, val in _ROUGHNESS_REFERENCE.items():
        lines.append(f"- {label}: {val:.1f}")
    lines.append("")

    # -- Collect materials by category -------------------------------------
    by_cat: dict[str, list[tuple[str, dict]]] = {}
    for name, props in PBR_MATERIALS.items():
        cat = props["category"]
        by_cat.setdefault(cat, []).append((name, props))

    # Desired category order
    cat_order = [
        "Metal", "Crystal", "Liquid", "Plastic",
        "Organic", "Human", "Manmade", "Ground",
    ]
    # Include any unexpected categories at the end
    for cat in by_cat:
        if cat not in cat_order:
            cat_order.append(cat)

    for cat in cat_order:
        entries = by_cat.get(cat)
        if not entries:
            continue

        is_metal = cat == "Metal"
        if is_metal:
            lines.append("### Metals (metallic = 1.0)")
            lines.append("| Material | Albedo (linear RGB) | Roughness |")
            lines.append("|---|---|---|")
        else:
            lines.append(f"### {cat} (metallic = 0.0)")
            lines.append(
                "| Material | Albedo (linear RGB) | Roughness | IOR |"
            )
            lines.append("|---|---|---|---|")

        for name, props in sorted(entries, key=lambda x: x[0]):
            alb = _fmt_rgb(props["albedo"])
            rough = f"{props['roughness']:.1f}"
            if is_metal:
                lines.append(f"| {name} | {alb} | {rough} |")
            else:
                ior_str = f"{props['ior']:.2f}" if "ior" in props else "-"
                trans = " T" if props.get("transmission") else ""
                lines.append(
                    f"| {name}{trans} | {alb} | {rough} | {ior_str} |"
                )

        lines.append("")

    lines.append(
        "T = transmissive (use `transmission` layer). "
        "Metals: roughness 0.0 = polished; adjust for brushed/worn finish."
    )
    return "\n".join(lines)
