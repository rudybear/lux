"""Tests for stdlib compose_pbr_layers and compositing functions."""

import pytest
from pathlib import Path
from luxc.compiler import compile_source
from luxc.builtins.types import clear_type_aliases

EXAMPLES = Path(__file__).parent.parent / "examples"

import subprocess

def _has_spirv_tools() -> bool:
    try:
        subprocess.run(["spirv-as", "--version"], capture_output=True)
        return True
    except FileNotFoundError:
        return False

requires_spirv_tools = pytest.mark.skipif(
    not _has_spirv_tools(), reason="spirv-as/spirv-val not found on PATH"
)


@requires_spirv_tools
class TestComposePbrLayers:
    """Unit tests for the compose_pbr_layers stdlib function."""

    def setup_method(self):
        clear_type_aliases()

    def teardown_method(self):
        clear_type_aliases()

    def test_compose_pbr_layers_base_only(self, tmp_path):
        """compose_pbr_layers with all optional layers zeroed = direct + IBL."""
        src = """
        import brdf;
        import color;
        import compositing;
        import ibl;

        fragment {
            in n: vec3;
            in v: vec3;
            in l: vec3;
            out color: vec4;
            fn main() {
                let direct: vec3 = vec3(0.5, 0.4, 0.3);
                let ambient: vec3 = vec3(0.1, 0.1, 0.1);
                let result: vec3 = compose_pbr_layers(
                    direct,
                    normalize(n), normalize(v), normalize(l),
                    vec3(0.8), 0.5,
                    0.0, 1.5, 0.0, vec3(1.0), 1000000.0,
                    ambient,
                    vec3(0.0), 0.0,
                    0.0, 0.0,
                    vec3(0.0),
                    vec3(0.0)
                );
                color = vec4(result, 1.0);
            }
        }
        """
        compile_source(src, "compose_base_only", tmp_path, validate=True)
        assert (tmp_path / "compose_base_only.frag.spv").exists()

    def test_compose_pbr_layers_full(self, tmp_path):
        """compose_pbr_layers with all layers active."""
        src = """
        import brdf;
        import color;
        import compositing;
        import ibl;

        fragment {
            in n: vec3;
            in v: vec3;
            in l: vec3;
            out color: vec4;
            fn main() {
                let direct: vec3 = vec3(0.5, 0.4, 0.3);
                let ambient: vec3 = vec3(0.1, 0.1, 0.1);
                let result: vec3 = compose_pbr_layers(
                    direct,
                    normalize(n), normalize(v), normalize(l),
                    vec3(0.8), 0.5,
                    0.5, 1.5, 0.0, vec3(1.0), 1000000.0,
                    ambient,
                    vec3(0.2, 0.1, 0.05), 0.5,
                    0.8, 0.1,
                    vec3(0.02),
                    vec3(1.0, 0.5, 0.0)
                );
                color = vec4(result, 1.0);
            }
        }
        """
        compile_source(src, "compose_full", tmp_path, validate=True)
        assert (tmp_path / "compose_full.frag.spv").exists()

    def test_compose_pbr_layers_transmission_disabled(self, tmp_path):
        """compose_pbr_layers with factor=0 transmission passthrough."""
        src = """
        import brdf;
        import color;
        import compositing;
        import ibl;

        fragment {
            in n: vec3;
            in v: vec3;
            in l: vec3;
            out color: vec4;
            fn main() {
                let direct: vec3 = vec3(0.5);
                let ambient: vec3 = vec3(0.1);
                let result: vec3 = compose_pbr_layers(
                    direct,
                    normalize(n), normalize(v), normalize(l),
                    vec3(0.8), 0.5,
                    0.0, 1.5, 0.0, vec3(1.0), 1000000.0,
                    ambient,
                    vec3(0.0), 0.0,
                    0.0, 0.0,
                    vec3(0.0),
                    vec3(0.0)
                );
                color = vec4(result, 1.0);
            }
        }
        """
        compile_source(src, "compose_no_trans", tmp_path, validate=True)
        assert (tmp_path / "compose_no_trans.frag.spv").exists()

    def test_compose_pbr_layers_sheen_disabled(self, tmp_path):
        """compose_pbr_layers with color=vec3(0) sheen passthrough."""
        src = """
        import brdf;
        import color;
        import compositing;
        import ibl;

        fragment {
            in n: vec3;
            in v: vec3;
            in l: vec3;
            out color: vec4;
            fn main() {
                let direct: vec3 = vec3(0.5);
                let ambient: vec3 = vec3(0.1);
                let result: vec3 = compose_pbr_layers(
                    direct,
                    normalize(n), normalize(v), normalize(l),
                    vec3(0.8), 0.5,
                    0.0, 1.5, 0.0, vec3(1.0), 1000000.0,
                    ambient,
                    vec3(0.0), 0.5,
                    0.0, 0.0,
                    vec3(0.0),
                    vec3(0.0)
                );
                color = vec4(result, 1.0);
            }
        }
        """
        compile_source(src, "compose_no_sheen", tmp_path, validate=True)
        assert (tmp_path / "compose_no_sheen.frag.spv").exists()

    def test_compose_pbr_layers_coat_disabled(self, tmp_path):
        """compose_pbr_layers with factor=0 coat passthrough."""
        src = """
        import brdf;
        import color;
        import compositing;
        import ibl;

        fragment {
            in n: vec3;
            in v: vec3;
            in l: vec3;
            out color: vec4;
            fn main() {
                let direct: vec3 = vec3(0.5);
                let ambient: vec3 = vec3(0.1);
                let result: vec3 = compose_pbr_layers(
                    direct,
                    normalize(n), normalize(v), normalize(l),
                    vec3(0.8), 0.5,
                    0.0, 1.5, 0.0, vec3(1.0), 1000000.0,
                    ambient,
                    vec3(0.0), 0.0,
                    0.0, 0.3,
                    vec3(0.0),
                    vec3(0.0)
                );
                color = vec4(result, 1.0);
            }
        }
        """
        compile_source(src, "compose_no_coat", tmp_path, validate=True)
        assert (tmp_path / "compose_no_coat.frag.spv").exists()


@requires_spirv_tools
class TestLayeredPipelineFullMatrix:
    """Full-matrix compilation regression tests for gltf_pbr_layered.lux.

    Tests all pipeline × mode × permutation combinations to ensure
    the refactored compose_pbr_layers produces valid SPIR-V.
    """

    def setup_method(self):
        clear_type_aliases()

    def teardown_method(self):
        clear_type_aliases()

    def _compile_layered(self, tmp_path, pipeline, features=None, bindless=False):
        """Helper to compile gltf_pbr_layered.lux with given options."""
        src = (EXAMPLES / "gltf_pbr_layered.lux").read_text()
        stem = f"layered_{pipeline}"
        if bindless:
            stem += "_bindless"
        if features:
            stem += "_" + "_".join(sorted(features))
        compile_source(
            src, stem, tmp_path, validate=True,
            source_dir=EXAMPLES,
            pipeline=pipeline,
            features=features or set(),
            bindless=bindless,
        )
        spv_files = list(tmp_path.glob("*.spv"))
        assert len(spv_files) >= 1, f"No .spv files for {stem}"
        return spv_files

    # --- GltfForward (raster) non-bindless ---

    def test_forward_base_only(self, tmp_path):
        self._compile_layered(tmp_path, "GltfForward")

    def test_forward_normal_emission(self, tmp_path):
        self._compile_layered(tmp_path, "GltfForward",
            features={"has_normal_map", "has_emission"})

    def test_forward_all_features(self, tmp_path):
        self._compile_layered(tmp_path, "GltfForward",
            features={"has_normal_map", "has_emission", "has_clearcoat",
                       "has_sheen", "has_transmission"})

    # --- GltfForward (raster) bindless ---

    def test_forward_bindless_base(self, tmp_path):
        self._compile_layered(tmp_path, "GltfForward", bindless=True)

    def test_forward_bindless_all(self, tmp_path):
        self._compile_layered(tmp_path, "GltfForward",
            features={"has_normal_map", "has_emission", "has_clearcoat",
                       "has_sheen", "has_transmission"},
            bindless=True)

    # --- GltfRT (raytracing) non-bindless ---

    def test_rt_base_only(self, tmp_path):
        self._compile_layered(tmp_path, "GltfRT")

    def test_rt_normal_emission(self, tmp_path):
        self._compile_layered(tmp_path, "GltfRT",
            features={"has_normal_map", "has_emission"})

    def test_rt_all_features(self, tmp_path):
        self._compile_layered(tmp_path, "GltfRT",
            features={"has_normal_map", "has_emission", "has_clearcoat",
                       "has_sheen", "has_transmission"})

    # --- GltfRT (raytracing) bindless ---

    def test_rt_bindless_base(self, tmp_path):
        self._compile_layered(tmp_path, "GltfRT", bindless=True)

    def test_rt_bindless_all(self, tmp_path):
        self._compile_layered(tmp_path, "GltfRT",
            features={"has_normal_map", "has_emission", "has_clearcoat",
                       "has_sheen", "has_transmission"},
            bindless=True)

    # --- GltfMesh (mesh shader) non-bindless ---

    def test_mesh_base_only(self, tmp_path):
        self._compile_layered(tmp_path, "GltfMesh")

    def test_mesh_normal_emission(self, tmp_path):
        self._compile_layered(tmp_path, "GltfMesh",
            features={"has_normal_map", "has_emission"})

    def test_mesh_all_features(self, tmp_path):
        self._compile_layered(tmp_path, "GltfMesh",
            features={"has_normal_map", "has_emission", "has_clearcoat",
                       "has_sheen", "has_transmission"})

    # --- GltfMesh (mesh shader) bindless ---

    def test_mesh_bindless_base(self, tmp_path):
        self._compile_layered(tmp_path, "GltfMesh", bindless=True)

    def test_mesh_bindless_all(self, tmp_path):
        self._compile_layered(tmp_path, "GltfMesh",
            features={"has_normal_map", "has_emission", "has_clearcoat",
                       "has_sheen", "has_transmission"},
            bindless=True)
