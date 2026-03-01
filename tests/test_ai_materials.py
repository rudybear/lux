"""Tests for PBR material reference data and system-prompt integration."""

import pytest

from luxc.ai.materials import PBR_MATERIALS, build_material_reference


class TestPBRMaterialsStructure:
    """Validate the PBR_MATERIALS dictionary has expected shape."""

    def test_materials_non_empty(self):
        assert len(PBR_MATERIALS) > 0

    def test_minimum_material_count(self):
        # We expect roughly 50+ curated materials.
        assert len(PBR_MATERIALS) >= 50

    def test_each_entry_has_required_keys(self):
        required = {"albedo", "roughness", "metallic", "category"}
        for name, props in PBR_MATERIALS.items():
            missing = required - set(props.keys())
            assert not missing, f"{name} is missing keys: {missing}"

    def test_albedo_is_three_tuple(self):
        for name, props in PBR_MATERIALS.items():
            alb = props["albedo"]
            assert isinstance(alb, tuple), f"{name}: albedo should be tuple"
            assert len(alb) == 3, f"{name}: albedo should have 3 components"

    def test_albedo_values_in_range(self):
        for name, props in PBR_MATERIALS.items():
            for i, ch in enumerate(props["albedo"]):
                assert 0.0 <= ch <= 1.0, (
                    f"{name}: albedo channel {i} = {ch} out of [0, 1]"
                )

    def test_metallic_is_zero_or_one(self):
        for name, props in PBR_MATERIALS.items():
            assert props["metallic"] in (0, 1), (
                f"{name}: metallic should be 0 or 1, got {props['metallic']}"
            )

    def test_roughness_in_range(self):
        for name, props in PBR_MATERIALS.items():
            r = props["roughness"]
            assert 0.0 <= r <= 1.0, (
                f"{name}: roughness = {r} out of [0, 1]"
            )

    def test_ior_positive_when_present(self):
        for name, props in PBR_MATERIALS.items():
            if "ior" in props:
                assert props["ior"] > 0, (
                    f"{name}: ior should be positive, got {props['ior']}"
                )

    def test_transmission_zero_or_one_when_present(self):
        for name, props in PBR_MATERIALS.items():
            if "transmission" in props:
                assert props["transmission"] in (0, 1), (
                    f"{name}: transmission should be 0 or 1"
                )


class TestKeyMaterialsExist:
    """Verify that commonly-requested materials are present."""

    @pytest.mark.parametrize("name", [
        "Copper", "Gold", "Silver", "Aluminum", "Iron",
    ])
    def test_metal_present(self, name):
        assert name in PBR_MATERIALS
        assert PBR_MATERIALS[name]["metallic"] == 1

    @pytest.mark.parametrize("name", [
        "Glass", "Diamond", "Water", "Marble",
    ])
    def test_dielectric_present(self, name):
        assert name in PBR_MATERIALS
        assert PBR_MATERIALS[name]["metallic"] == 0

    def test_skin_variants(self):
        skin_names = [n for n in PBR_MATERIALS if "Skin" in n]
        assert len(skin_names) >= 2

    def test_supplementary_ground_materials(self):
        assert "Fresh Asphalt" in PBR_MATERIALS
        assert "Desert Sand" in PBR_MATERIALS
        assert "Fresh Snow" in PBR_MATERIALS


class TestBuildMaterialReference:
    """Validate the formatted string returned by build_material_reference."""

    def test_returns_string(self):
        ref = build_material_reference()
        assert isinstance(ref, str)

    def test_contains_header(self):
        ref = build_material_reference()
        assert "PBR Material Reference" in ref

    def test_contains_metals_section(self):
        ref = build_material_reference()
        assert "Metals" in ref

    def test_contains_key_materials(self):
        ref = build_material_reference()
        for name in ("Copper", "Gold", "Glass", "Marble"):
            assert name in ref, f"{name} not found in reference table"

    def test_contains_roughness_guide(self):
        ref = build_material_reference()
        assert "Roughness Guide" in ref
        assert "Mirror" in ref
        assert "Matte" in ref

    def test_table_has_albedo_tuples(self):
        ref = build_material_reference()
        # Should contain formatted albedo like "(0.93, 0.62, 0.52)"
        assert "(0.93, 0.62, 0.52)" in ref  # Copper

    def test_transmission_marker(self):
        ref = build_material_reference()
        # Transmissive materials get a " T" suffix
        assert " T" in ref


class TestSystemPromptIntegration:
    """Ensure the material reference is embedded in system prompts."""

    def test_general_prompt_includes_materials(self):
        from luxc.ai.system_prompt import build_system_prompt
        prompt = build_system_prompt()
        assert "PBR Material Reference" in prompt
        assert "Gold" in prompt
        assert "Copper" in prompt

    def test_material_extraction_prompt_includes_materials(self):
        from luxc.ai.system_prompt import build_material_extraction_prompt
        prompt = build_material_extraction_prompt()
        assert "PBR Material Reference" in prompt
        assert "Glass" in prompt

    def test_materials_placed_before_constraints(self):
        from luxc.ai.system_prompt import build_system_prompt
        prompt = build_system_prompt()
        mat_pos = prompt.index("PBR Material Reference")
        con_pos = prompt.index("Critical Constraints")
        assert mat_pos < con_pos, (
            "Material reference should appear before constraints"
        )

    def test_materials_placed_after_stdlib(self):
        from luxc.ai.system_prompt import build_system_prompt
        prompt = build_system_prompt()
        std_pos = prompt.index("Standard Library Modules")
        mat_pos = prompt.index("PBR Material Reference")
        assert std_pos < mat_pos, (
            "Material reference should appear after stdlib"
        )
