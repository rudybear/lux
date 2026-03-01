#!/usr/bin/env python3
"""Generate BRDF parameter estimation dataset.

For each material x lighting condition, produces ground-truth parameter
records suitable for training inverse-rendering models.

Usage:
    python tools/generate_brdf_dataset.py --output brdf_dataset.jsonl
    python tools/generate_brdf_dataset.py --output brdf_dataset.jsonl --materials Copper Gold
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from luxc.ai.dataset import expand_material_variants, ShaderVariant
from luxc.ai.materials import PBR_MATERIALS


# Lighting conditions for dataset diversity
LIGHTING_CONDITIONS = [
    {
        "name": "studio",
        "description": "Studio three-point lighting",
        "key_direction": (0.5, 0.8, 0.3),
        "key_intensity": 1.0,
        "fill_intensity": 0.3,
        "ambient": 0.05,
    },
    {
        "name": "outdoor",
        "description": "Outdoor daylight from above",
        "key_direction": (0.0, 1.0, 0.2),
        "key_intensity": 1.2,
        "fill_intensity": 0.4,
        "ambient": 0.1,
    },
    {
        "name": "dramatic",
        "description": "Dramatic single-source side lighting",
        "key_direction": (1.0, 0.3, 0.0),
        "key_intensity": 2.0,
        "fill_intensity": 0.05,
        "ambient": 0.02,
    },
    {
        "name": "backlit",
        "description": "Strong backlighting with rim",
        "key_direction": (0.0, 0.3, -1.0),
        "key_intensity": 1.5,
        "fill_intensity": 0.2,
        "ambient": 0.03,
    },
    {
        "name": "overcast",
        "description": "Soft overcast sky lighting",
        "key_direction": (0.0, 1.0, 0.0),
        "key_intensity": 0.6,
        "fill_intensity": 0.5,
        "ambient": 0.2,
    },
]


def main():
    parser = argparse.ArgumentParser(
        description="Generate BRDF parameter estimation dataset"
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=Path("brdf_dataset.jsonl"),
        help="Output JSONL file",
    )
    parser.add_argument(
        "--materials", nargs="*", default=None,
        help="Specific materials (default: all)",
    )
    parser.add_argument(
        "--roughness-steps", type=int, default=5,
        help="Roughness steps (default: 5)",
    )
    args = parser.parse_args()

    materials = args.materials or list(PBR_MATERIALS.keys())
    records: list[dict] = []

    print(f"Generating BRDF dataset for {len(materials)} materials "
          f"x {len(LIGHTING_CONDITIONS)} lighting conditions...")

    for mat_name in materials:
        if mat_name not in PBR_MATERIALS:
            print(f"  Skipping unknown material: {mat_name}")
            continue

        props = PBR_MATERIALS[mat_name]
        variants = expand_material_variants(
            mat_name, roughness_steps=args.roughness_steps, albedo_variations=1
        )

        for variant in variants:
            for lighting in LIGHTING_CONDITIONS:
                record = {
                    "material": mat_name,
                    "variant": variant.name,
                    "lighting": lighting["name"],
                    "lighting_description": lighting["description"],
                    "lux_source": variant.lux_source,
                    "ground_truth": {
                        "albedo": [
                            variant.parameters.get("albedo_r", 0.5),
                            variant.parameters.get("albedo_g", 0.5),
                            variant.parameters.get("albedo_b", 0.5),
                        ],
                        "roughness": variant.parameters.get("roughness", 0.5),
                        "metallic": variant.parameters.get("metallic", 0.0),
                        "category": props["category"],
                    },
                    "lighting_params": {
                        "key_direction": lighting["key_direction"],
                        "key_intensity": lighting["key_intensity"],
                        "fill_intensity": lighting["fill_intensity"],
                        "ambient": lighting["ambient"],
                    },
                }
                if "ior" in variant.parameters:
                    record["ground_truth"]["ior"] = variant.parameters["ior"]
                records.append(record)

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
