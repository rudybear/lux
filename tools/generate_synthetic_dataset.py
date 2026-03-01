#!/usr/bin/env python3
"""Generate synthetic training data for AI shader generation.

Produces parametric and random material variants with metadata,
optionally rendering each to images for visual training data.

Usage:
    python tools/generate_synthetic_dataset.py --output dataset.jsonl
    python tools/generate_synthetic_dataset.py --output dataset.jsonl --count 500
    python tools/generate_synthetic_dataset.py --output dataset.jsonl --materials Copper Gold Glass
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from luxc.ai.dataset import (
    expand_material_variants,
    generate_random_materials,
    ShaderVariant,
)
from luxc.ai.materials import PBR_MATERIALS


def write_jsonl(variants: list[ShaderVariant], output: Path) -> None:
    """Write variants to a JSONL file."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for v in variants:
            record = {
                "name": v.name,
                "description": v.description,
                "lux_source": v.lux_source,
                "parameters": v.parameters,
            }
            f.write(json.dumps(record) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic PBR material dataset"
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=Path("dataset.jsonl"),
        help="Output JSONL file (default: dataset.jsonl)",
    )
    parser.add_argument(
        "--materials", nargs="*", default=None,
        help="Specific materials to expand (default: all)",
    )
    parser.add_argument(
        "--count", type=int, default=100,
        help="Number of random materials to generate (default: 100)",
    )
    parser.add_argument(
        "--roughness-steps", type=int, default=5,
        help="Roughness sweep steps per material (default: 5)",
    )
    parser.add_argument(
        "--albedo-variations", type=int, default=3,
        help="Albedo brightness variations (default: 3)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--skip-parametric", action="store_true",
        help="Skip parametric expansion, only generate random",
    )
    parser.add_argument(
        "--skip-random", action="store_true",
        help="Skip random generation, only do parametric expansion",
    )
    args = parser.parse_args()

    all_variants: list[ShaderVariant] = []

    # Parametric expansion
    if not args.skip_parametric:
        materials = args.materials or list(PBR_MATERIALS.keys())
        print(f"Expanding {len(materials)} materials...")
        for mat_name in materials:
            if mat_name not in PBR_MATERIALS:
                print(f"  Warning: unknown material '{mat_name}', skipping")
                continue
            variants = expand_material_variants(
                mat_name,
                roughness_steps=args.roughness_steps,
                albedo_variations=args.albedo_variations,
            )
            all_variants.extend(variants)
            print(f"  {mat_name}: {len(variants)} variants")

    # Random generation
    if not args.skip_random:
        print(f"Generating {args.count} random materials...")
        random_variants = generate_random_materials(
            count=args.count, seed=args.seed
        )
        all_variants.extend(random_variants)

    # Write output
    write_jsonl(all_variants, args.output)
    print(f"\nWrote {len(all_variants)} variants to {args.output}")


if __name__ == "__main__":
    main()
