"""Tests for AI synthetic dataset generation (Phase 16)."""

import pytest

from luxc.ai.dataset import ShaderVariant, expand_material_variants, generate_random_materials


class TestExpandMaterialVariants:
    def test_expand_material_variants_basic(self):
        """Expand 'Copper' with default params, verify count = roughness_steps * albedo_variations."""
        variants = expand_material_variants("Copper")

        # Default: roughness_steps=5, albedo_variations=3, no extra layers
        expected_count = 5 * 3
        assert len(variants) == expected_count

        for v in variants:
            assert isinstance(v, ShaderVariant)
            assert len(v.name) > 0
            assert len(v.lux_source) > 0
            assert "Copper" in v.description

    def test_expand_material_variants_with_layers(self):
        """Add coat layer, verify extra variants are generated."""
        without_layers = expand_material_variants("Copper", roughness_steps=3, albedo_variations=2)
        with_layers = expand_material_variants(
            "Copper", roughness_steps=3, albedo_variations=2, layer_combinations=["coat"]
        )

        base_count = 3 * 2
        assert len(without_layers) == base_count
        # Each base variant gets an additional coat variant
        assert len(with_layers) == base_count * 2

        # Verify some coat variants exist
        coat_variants = [v for v in with_layers if "coat" in v.name.lower()]
        assert len(coat_variants) == base_count

        # Coat variants should have coat_factor in parameters
        for v in coat_variants:
            assert "coat_factor" in v.parameters

    def test_expand_material_variants_unknown(self):
        """Unknown material raises ValueError."""
        with pytest.raises(ValueError, match="Unknown material"):
            expand_material_variants("TotallyFakeMaterial999")


class TestGenerateRandomMaterials:
    def test_generate_random_materials_count(self):
        """Verify count matches requested."""
        for count in [1, 10, 50]:
            variants = generate_random_materials(count=count)
            assert len(variants) == count

    def test_generate_random_materials_reproducible(self):
        """Same seed produces the same results."""
        a = generate_random_materials(count=20, seed=12345)
        b = generate_random_materials(count=20, seed=12345)

        assert len(a) == len(b)
        for va, vb in zip(a, b):
            assert va.name == vb.name
            assert va.lux_source == vb.lux_source
            assert va.parameters == vb.parameters

    def test_generate_random_materials_bounds(self):
        """All parameters are within physical bounds."""
        variants = generate_random_materials(count=200, seed=42)

        for v in variants:
            params = v.parameters
            # Roughness in [0, 1]
            assert 0.0 <= params["roughness"] <= 1.0, (
                f"{v.name}: roughness {params['roughness']} out of bounds"
            )

            # Metallic is 0 or 1 (binary)
            assert params["metallic"] in (0.0, 1.0), (
                f"{v.name}: metallic {params['metallic']} not 0 or 1"
            )

            # Albedo components in [0, 1]
            for ch in ("albedo_r", "albedo_g", "albedo_b"):
                assert 0.0 <= params[ch] <= 1.0, (
                    f"{v.name}: {ch} = {params[ch]} out of bounds"
                )

            # IOR if present: reasonable glass/crystal range
            if "ior" in params:
                assert 1.0 <= params["ior"] <= 3.0, (
                    f"{v.name}: ior {params['ior']} out of range"
                )

            # Coat factor if present: 0 < factor <= 1
            if "coat_factor" in params:
                assert 0.0 < params["coat_factor"] <= 1.0


class TestShaderVariantOutput:
    def test_shader_variant_source_valid(self):
        """Generated source contains a 'surface' declaration."""
        variants = generate_random_materials(count=5, seed=99)

        for v in variants:
            assert "surface" in v.lux_source, (
                f"{v.name}: missing 'surface' keyword in source"
            )

    def test_shader_variant_parameters(self):
        """Parameters dict has required keys (roughness, metallic, albedo)."""
        variants = generate_random_materials(count=10, seed=77)

        required_keys = {"roughness", "metallic", "albedo_r", "albedo_g", "albedo_b"}
        for v in variants:
            assert required_keys.issubset(v.parameters.keys()), (
                f"{v.name}: missing required parameter keys. "
                f"Has: {set(v.parameters.keys())}, needs: {required_keys}"
            )
