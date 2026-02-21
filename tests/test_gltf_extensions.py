"""Tests for glTF PBR material extensions: clearcoat, sheen, transmission.

Tests shader compilation of all extension permutations and screenshot
rendering of Khronos test assets (ClearCoatTest, SheenChair, TransmissionTest).
"""

import json
import subprocess
import pytest
import numpy as np
from pathlib import Path
from luxc.compiler import compile_source
from luxc.builtins.types import clear_type_aliases

EXAMPLES = Path(__file__).parent.parent / "examples"
ASSETS = Path(__file__).parent.parent / "assets"
SCREENSHOTS = Path(__file__).parent.parent / "screenshots"
SHADERCACHE = Path(__file__).parent.parent / "shadercache"


def _has_spirv_tools() -> bool:
    try:
        subprocess.run(["spirv-as", "--version"], capture_output=True)
        return True
    except FileNotFoundError:
        return False


requires_spirv_tools = pytest.mark.skipif(
    not _has_spirv_tools(), reason="spirv-as/spirv-val not found on PATH"
)


def _compile_layered(tmp_path, pipeline, features, validate=True):
    """Compile gltf_pbr_layered.lux with a given pipeline and feature set."""
    clear_type_aliases()
    src = (EXAMPLES / "gltf_pbr_layered.lux").read_text()
    compile_source(
        src, "gltf_pbr_layered", tmp_path,
        source_dir=EXAMPLES,
        pipeline=pipeline,
        features=features,
        validate=validate,
    )


# ---------------------------------------------------------------------------
# Phase A+B: Shader compilation tests for all extension permutations
# ---------------------------------------------------------------------------

@requires_spirv_tools
class TestLayeredExtensionCompilation:
    """Verify that all extension feature permutations compile and validate."""

    def test_clearcoat_forward(self, tmp_path):
        _compile_layered(
            tmp_path, "GltfForward",
            {"has_normal_map", "has_emission", "has_clearcoat"},
        )
        assert (tmp_path / "gltf_pbr_layered+clearcoat+emission+normal_map.vert.spv").exists()
        assert (tmp_path / "gltf_pbr_layered+clearcoat+emission+normal_map.frag.spv").exists()

    def test_sheen_forward(self, tmp_path):
        _compile_layered(
            tmp_path, "GltfForward",
            {"has_normal_map", "has_emission", "has_sheen"},
        )
        assert (tmp_path / "gltf_pbr_layered+emission+normal_map+sheen.vert.spv").exists()
        assert (tmp_path / "gltf_pbr_layered+emission+normal_map+sheen.frag.spv").exists()

    def test_transmission_forward(self, tmp_path):
        _compile_layered(
            tmp_path, "GltfForward",
            {"has_normal_map", "has_emission", "has_transmission"},
        )
        assert (tmp_path / "gltf_pbr_layered+emission+normal_map+transmission.vert.spv").exists()
        assert (tmp_path / "gltf_pbr_layered+emission+normal_map+transmission.frag.spv").exists()

    def test_all_extensions_forward(self, tmp_path):
        _compile_layered(
            tmp_path, "GltfForward",
            {"has_normal_map", "has_emission", "has_clearcoat", "has_sheen", "has_transmission"},
        )
        spv = tmp_path / "gltf_pbr_layered+clearcoat+emission+normal_map+sheen+transmission.frag.spv"
        assert spv.exists()

    def test_clearcoat_rt(self, tmp_path):
        _compile_layered(
            tmp_path, "GltfRT",
            {"has_normal_map", "has_emission", "has_clearcoat"},
            validate=False,
        )
        assert (tmp_path / "gltf_pbr_layered+clearcoat+emission+normal_map.rgen.spv").exists()
        assert (tmp_path / "gltf_pbr_layered+clearcoat+emission+normal_map.rchit.spv").exists()
        assert (tmp_path / "gltf_pbr_layered+clearcoat+emission+normal_map.rmiss.spv").exists()

    def test_all_extensions_rt(self, tmp_path):
        _compile_layered(
            tmp_path, "GltfRT",
            {"has_normal_map", "has_emission", "has_clearcoat", "has_sheen", "has_transmission"},
            validate=False,
        )
        spv = tmp_path / "gltf_pbr_layered+clearcoat+emission+normal_map+sheen+transmission.rgen.spv"
        assert spv.exists()

    def test_sheen_only_forward(self, tmp_path):
        """SheenChair needs has_normal_map + has_sheen (no emission)."""
        _compile_layered(
            tmp_path, "GltfForward",
            {"has_normal_map", "has_sheen"},
        )
        assert (tmp_path / "gltf_pbr_layered+normal_map+sheen.frag.spv").exists()

    def test_transmission_only_forward(self, tmp_path):
        """TransmissionTest needs has_emission + has_transmission (no normal_map)."""
        _compile_layered(
            tmp_path, "GltfForward",
            {"has_emission", "has_transmission"},
        )
        assert (tmp_path / "gltf_pbr_layered+emission+transmission.frag.spv").exists()


# ---------------------------------------------------------------------------
# Reflection JSON validation for extension textures
# ---------------------------------------------------------------------------

@requires_spirv_tools
class TestExtensionReflection:
    """Verify that reflection JSON includes correct extension texture bindings."""

    def test_clearcoat_reflection_has_coat_textures(self, tmp_path):
        _compile_layered(
            tmp_path, "GltfForward",
            {"has_normal_map", "has_emission", "has_clearcoat"},
        )
        refl = json.loads(
            (tmp_path / "gltf_pbr_layered+clearcoat+emission+normal_map.frag.json").read_text()
        )
        samplers = {
            b["name"]
            for bindings in refl["descriptor_sets"].values()
            for b in bindings
            if b.get("type") == "sampler"
        }
        assert "clearcoat_tex" in samplers
        assert "clearcoat_roughness_tex" in samplers

    def test_sheen_reflection_has_sheen_texture(self, tmp_path):
        _compile_layered(
            tmp_path, "GltfForward",
            {"has_normal_map", "has_emission", "has_sheen"},
        )
        refl = json.loads(
            (tmp_path / "gltf_pbr_layered+emission+normal_map+sheen.frag.json").read_text()
        )
        samplers = {
            b["name"]
            for bindings in refl["descriptor_sets"].values()
            for b in bindings
            if b.get("type") == "sampler"
        }
        assert "sheen_color_tex" in samplers

    def test_transmission_reflection_has_transmission_texture(self, tmp_path):
        _compile_layered(
            tmp_path, "GltfForward",
            {"has_normal_map", "has_emission", "has_transmission"},
        )
        refl = json.loads(
            (tmp_path / "gltf_pbr_layered+emission+normal_map+transmission.frag.json").read_text()
        )
        samplers = {
            b["name"]
            for bindings in refl["descriptor_sets"].values()
            for b in bindings
            if b.get("type") == "sampler"
        }
        assert "transmission_tex" in samplers

    def test_all_extensions_reflection_complete(self, tmp_path):
        _compile_layered(
            tmp_path, "GltfForward",
            {"has_normal_map", "has_emission", "has_clearcoat", "has_sheen", "has_transmission"},
        )
        refl = json.loads(
            (tmp_path / "gltf_pbr_layered+clearcoat+emission+normal_map+sheen+transmission.frag.json").read_text()
        )
        samplers = {
            b["name"]
            for bindings in refl["descriptor_sets"].values()
            for b in bindings
            if b.get("type") == "sampler"
        }
        # All base PBR textures
        assert "base_color_tex" in samplers
        assert "metallic_roughness_tex" in samplers
        assert "occlusion_tex" in samplers
        assert "normal_tex" in samplers
        assert "emissive_tex" in samplers
        # IBL textures
        assert "env_specular" in samplers
        assert "env_irradiance" in samplers
        assert "brdf_lut" in samplers
        # Extension textures
        assert "clearcoat_tex" in samplers
        assert "clearcoat_roughness_tex" in samplers
        assert "sheen_color_tex" in samplers
        assert "transmission_tex" in samplers

    def test_feature_flags_in_reflection(self, tmp_path):
        _compile_layered(
            tmp_path, "GltfForward",
            {"has_normal_map", "has_clearcoat", "has_sheen"},
        )
        refl = json.loads(
            (tmp_path / "gltf_pbr_layered+clearcoat+normal_map+sheen.frag.json").read_text()
        )
        assert refl["features"]["has_clearcoat"] is True
        assert refl["features"]["has_sheen"] is True
        assert refl["features"]["has_normal_map"] is True
        assert refl["features"]["has_emission"] is False
        assert refl["features"]["has_transmission"] is False


# ---------------------------------------------------------------------------
# glTF loader extension detection tests
# ---------------------------------------------------------------------------

class TestGltfExtensionDetection:
    """Verify that the Python glTF loader correctly detects KHR_materials_* extensions."""

    @pytest.fixture(autouse=True)
    def _check_assets(self):
        if not (ASSETS / "ClearCoatTest.glb").exists():
            pytest.skip("Khronos test assets not downloaded")

    def test_clearcoat_detected(self):
        from playground.gltf_loader import load_gltf
        scene = load_gltf(str(ASSETS / "ClearCoatTest.glb"))
        has_clearcoat = any("clearcoat" in m.extensions for m in scene.materials)
        assert has_clearcoat, "ClearCoatTest.glb should have clearcoat extension"

    def test_sheen_detected(self):
        from playground.gltf_loader import load_gltf
        scene = load_gltf(str(ASSETS / "SheenChair.glb"))
        has_sheen = any("sheen" in m.extensions for m in scene.materials)
        assert has_sheen, "SheenChair.glb should have sheen extension"

    def test_transmission_detected(self):
        from playground.gltf_loader import load_gltf
        scene = load_gltf(str(ASSETS / "TransmissionTest.glb"))
        has_trans = any("transmission" in m.extensions for m in scene.materials)
        assert has_trans, "TransmissionTest.glb should have transmission extension"

    def test_damaged_helmet_no_extensions(self):
        from playground.gltf_loader import load_gltf
        scene = load_gltf(str(ASSETS / "DamagedHelmet.glb"))
        exts = set()
        for m in scene.materials:
            exts.update(m.extensions.keys())
        # DamagedHelmet uses only base PBR, no extensions
        assert "clearcoat" not in exts
        assert "sheen" not in exts
        assert "transmission" not in exts


# ---------------------------------------------------------------------------
# Scene feature detection tests
# ---------------------------------------------------------------------------

class TestSceneFeatureDetection:
    """Verify that scene-level feature detection works correctly."""

    @pytest.fixture(autouse=True)
    def _check_assets(self):
        if not (ASSETS / "ClearCoatTest.glb").exists():
            pytest.skip("Khronos test assets not downloaded")

    def _detect_features(self, glb_path):
        from playground.gltf_loader import load_gltf
        scene = load_gltf(str(glb_path))
        features = set()
        for mat in scene.materials:
            if mat.normal_texture is not None:
                features.add("has_normal_map")
            if mat.emissive_texture is not None or any(e > 0 for e in mat.emissive):
                features.add("has_emission")
            if "clearcoat" in mat.extensions:
                features.add("has_clearcoat")
            if "sheen" in mat.extensions:
                features.add("has_sheen")
            if "transmission" in mat.extensions:
                features.add("has_transmission")
        return features

    def test_clearcoat_features(self):
        features = self._detect_features(ASSETS / "ClearCoatTest.glb")
        assert "has_clearcoat" in features
        assert "has_normal_map" in features
        assert "has_emission" in features

    def test_sheen_features(self):
        features = self._detect_features(ASSETS / "SheenChair.glb")
        assert "has_sheen" in features
        assert "has_normal_map" in features

    def test_transmission_features(self):
        features = self._detect_features(ASSETS / "TransmissionTest.glb")
        assert "has_transmission" in features
        assert "has_emission" in features

    def test_damaged_helmet_base_features(self):
        features = self._detect_features(ASSETS / "DamagedHelmet.glb")
        assert "has_normal_map" in features
        assert "has_emission" in features
        assert "has_clearcoat" not in features
        assert "has_sheen" not in features
        assert "has_transmission" not in features

    def test_pipeline_path_generation(self):
        features = self._detect_features(ASSETS / "ClearCoatTest.glb")
        suffix = "+".join(sorted(f.replace("has_", "") for f in features))
        expected = "shadercache/gltf_pbr_layered+clearcoat+emission+normal_map"
        assert f"shadercache/gltf_pbr_layered+{suffix}" == expected


# ---------------------------------------------------------------------------
# Screenshot rendering tests (Python engine, requires GPU)
# ---------------------------------------------------------------------------

def _has_wgpu():
    try:
        import wgpu  # noqa: F401
        return True
    except ImportError:
        return False


requires_wgpu = pytest.mark.skipif(
    not _has_wgpu(), reason="wgpu not available"
)


def _render_screenshot(scene_path, pipeline_base, output_path, width=512, height=512):
    """Render a screenshot using the Python engine, return pixel array."""
    from playground.engine import render
    pixels = render(
        str(scene_path), pipeline_base, str(output_path),
        width=width, height=height,
    )
    return pixels


def _check_screenshot_validity(pixels, min_coverage=0.10, name=""):
    """Basic validity checks on rendered screenshot pixels."""
    assert pixels is not None, f"{name}: render returned None"
    assert pixels.shape[0] > 0 and pixels.shape[1] > 0, f"{name}: empty image"

    # Check coverage: percentage of non-black pixels
    non_black = (pixels[:, :, :3].sum(axis=2) > 10).sum()
    total = pixels.shape[0] * pixels.shape[1]
    coverage = non_black / total
    assert coverage >= min_coverage, (
        f"{name}: coverage {coverage:.1%} below threshold {min_coverage:.0%}"
    )

    # Check not all-white (would indicate rendering failure)
    mean_brightness = pixels[:, :, :3].mean()
    assert mean_brightness < 250, f"{name}: all-white image (mean={mean_brightness:.0f})"

    # Check reasonable brightness range (not totally dark either)
    assert mean_brightness > 5, f"{name}: all-black image (mean={mean_brightness:.0f})"

    return coverage


@requires_wgpu
@requires_spirv_tools
class TestExtensionScreenshots:
    """Render Khronos test models with extension shaders and validate output."""

    @pytest.fixture(autouse=True)
    def _check_assets(self):
        if not (ASSETS / "ClearCoatTest.glb").exists():
            pytest.skip("Khronos test assets not downloaded")

    @pytest.fixture(autouse=True)
    def _ensure_shaders(self, tmp_path):
        """Compile required shader variants for screenshot tests."""
        # ClearCoatTest needs: clearcoat+emission+normal_map
        _compile_layered(
            tmp_path, "GltfForward",
            {"has_normal_map", "has_emission", "has_clearcoat"},
        )
        # SheenChair needs: normal_map+sheen
        _compile_layered(
            tmp_path, "GltfForward",
            {"has_normal_map", "has_sheen"},
        )
        # TransmissionTest needs: emission+transmission
        _compile_layered(
            tmp_path, "GltfForward",
            {"has_emission", "has_transmission"},
        )
        self._shader_dir = tmp_path

    def test_clearcoat_screenshot(self):
        pipeline = str(self._shader_dir / "gltf_pbr_layered+clearcoat+emission+normal_map")
        output = SCREENSHOTS / "test_clearcoat_python.png"
        pixels = _render_screenshot(
            ASSETS / "ClearCoatTest.glb", pipeline, output,
        )
        coverage = _check_screenshot_validity(pixels, min_coverage=0.30, name="ClearCoatTest")
        assert coverage > 0.50, f"ClearCoatTest coverage too low: {coverage:.1%}"

    def test_sheen_screenshot(self):
        pipeline = str(self._shader_dir / "gltf_pbr_layered+normal_map+sheen")
        output = SCREENSHOTS / "test_sheen_python.png"
        pixels = _render_screenshot(
            ASSETS / "SheenChair.glb", pipeline, output,
        )
        coverage = _check_screenshot_validity(pixels, min_coverage=0.10, name="SheenChair")

    def test_transmission_screenshot(self):
        pipeline = str(self._shader_dir / "gltf_pbr_layered+emission+transmission")
        output = SCREENSHOTS / "test_transmission_python.png"
        pixels = _render_screenshot(
            ASSETS / "TransmissionTest.glb", pipeline, output,
        )
        coverage = _check_screenshot_validity(pixels, min_coverage=0.20, name="TransmissionTest")

    def test_damaged_helmet_regression(self):
        """DamagedHelmet still renders correctly with base pipeline."""
        output = SCREENSHOTS / "test_gltf_extension_regression.png"
        pixels = _render_screenshot(
            ASSETS / "DamagedHelmet.glb", "shadercache/gltf_pbr", output,
        )
        coverage = _check_screenshot_validity(pixels, min_coverage=0.30, name="DamagedHelmet")
