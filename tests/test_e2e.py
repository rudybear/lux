"""End-to-end tests: compile .lux files and validate with spirv-val."""

import subprocess
import pytest
from pathlib import Path
from luxc.compiler import compile_source
from luxc.builtins.types import clear_type_aliases

FIXTURES = Path(__file__).parent / "fixtures"
EXAMPLES = Path(__file__).parent.parent / "examples"


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
class TestFixtureCompilation:
    def test_minimal_vertex(self, tmp_path):
        src = (FIXTURES / "minimal_vertex.lux").read_text()
        compile_source(src, "minimal_vertex", tmp_path, validate=True)
        assert (tmp_path / "minimal_vertex.vert.spv").exists()

    def test_minimal_fragment(self, tmp_path):
        src = (FIXTURES / "minimal_fragment.lux").read_text()
        compile_source(src, "minimal_fragment", tmp_path, validate=True)
        assert (tmp_path / "minimal_fragment.frag.spv").exists()

    def test_hello_triangle(self, tmp_path):
        src = (FIXTURES / "hello_triangle.lux").read_text()
        compile_source(src, "hello_triangle", tmp_path, validate=True)
        assert (tmp_path / "hello_triangle.vert.spv").exists()
        assert (tmp_path / "hello_triangle.frag.spv").exists()


@requires_spirv_tools
class TestExampleCompilation:
    def test_hello_triangle_example(self, tmp_path):
        src = (EXAMPLES / "hello_triangle.lux").read_text()
        compile_source(src, "hello_triangle", tmp_path, validate=True)
        assert (tmp_path / "hello_triangle.vert.spv").exists()
        assert (tmp_path / "hello_triangle.frag.spv").exists()

    def test_pbr_basic_example(self, tmp_path):
        src = (EXAMPLES / "pbr_basic.lux").read_text()
        compile_source(src, "pbr_basic", tmp_path, validate=True)
        assert (tmp_path / "pbr_basic.vert.spv").exists()
        assert (tmp_path / "pbr_basic.frag.spv").exists()


@requires_spirv_tools
class TestAdditionalFixtures:
    def test_texture_sampling(self, tmp_path):
        src = """
        fragment {
            in uv: vec2;
            out color: vec4;
            sampler2d tex;
            fn main() {
                color = sample(tex, uv);
            }
        }
        """
        compile_source(src, "texture_test", tmp_path, validate=True)
        assert (tmp_path / "texture_test.frag.spv").exists()

    def test_multiple_uniforms(self, tmp_path):
        src = """
        vertex {
            in pos: vec3;
            uniform Transform { model: mat4, view: mat4, projection: mat4 }
            fn main() {
                builtin_position = projection * view * model * vec4(pos, 1.0);
            }
        }
        """
        compile_source(src, "multi_uniform", tmp_path, validate=True)
        assert (tmp_path / "multi_uniform.vert.spv").exists()

    def test_push_constants(self, tmp_path):
        src = """
        fragment {
            out color: vec4;
            push Material { base_color: vec3 }
            fn main() {
                color = vec4(base_color, 1.0);
            }
        }
        """
        compile_source(src, "push_test", tmp_path, validate=True)
        assert (tmp_path / "push_test.frag.spv").exists()

    def test_math_heavy(self, tmp_path):
        src = """
        fragment {
            in n: vec3;
            in l: vec3;
            out color: vec4;
            fn main() {
                let nn: vec3 = normalize(n);
                let nl: scalar = max(dot(nn, normalize(l)), 0.0);
                let diffuse: vec3 = vec3(nl);
                color = vec4(diffuse, 1.0);
            }
        }
        """
        compile_source(src, "math_heavy", tmp_path, validate=True)
        assert (tmp_path / "math_heavy.frag.spv").exists()

    def test_vec3_splat_constructor(self, tmp_path):
        src = """
        fragment {
            out color: vec4;
            fn main() {
                let v: vec3 = vec3(0.5);
                color = vec4(v, 1.0);
            }
        }
        """
        compile_source(src, "splat_test", tmp_path, validate=True)
        assert (tmp_path / "splat_test.frag.spv").exists()

    def test_swizzle_operations(self, tmp_path):
        src = """
        fragment {
            in v: vec4;
            out color: vec4;
            fn main() {
                let xyz: vec3 = v.xyz;
                let rg: vec2 = v.rg;
                color = vec4(xyz, 1.0);
            }
        }
        """
        compile_source(src, "swizzle_test", tmp_path, validate=True)
        assert (tmp_path / "swizzle_test.frag.spv").exists()

    def test_vec_scalar_division(self, tmp_path):
        """vec3 / scalar should splat scalar and use OpFDiv."""
        src = """
        fragment {
            in n: vec3;
            out color: vec4;
            fn main() {
                let v: vec3 = n / 2.0;
                let w: vec3 = n + 0.5;
                let x: vec3 = n - 0.1;
                color = vec4(v, 1.0);
            }
        }
        """
        compile_source(src, "vec_scalar_div", tmp_path, validate=True)
        assert (tmp_path / "vec_scalar_div.frag.spv").exists()


@requires_spirv_tools
class TestImportSystem:
    def setup_method(self):
        clear_type_aliases()

    def teardown_method(self):
        clear_type_aliases()

    def test_import_brdf(self, tmp_path):
        src = """
        import brdf;

        fragment {
            in n: vec3;
            in l: vec3;
            out color: vec4;
            fn main() {
                let albedo: vec3 = vec3(0.8, 0.2, 0.1);
                let ndotl: scalar = max(dot(n, l), 0.0);
                let result: vec3 = lambert_brdf(albedo, ndotl);
                color = vec4(result, 1.0);
            }
        }
        """
        compile_source(src, "import_brdf", tmp_path, validate=True)
        assert (tmp_path / "import_brdf.frag.spv").exists()

    def test_import_color(self, tmp_path):
        src = """
        import color;

        fragment {
            in c: vec3;
            out color: vec4;
            fn main() {
                let mapped: vec3 = tonemap_aces(c);
                let final_c: vec3 = linear_to_srgb(mapped);
                color = vec4(final_c, 1.0);
            }
        }
        """
        compile_source(src, "import_color", tmp_path, validate=True)
        assert (tmp_path / "import_color.frag.spv").exists()

    def test_import_brdf_full_pbr(self, tmp_path):
        """Full PBR pipeline using imported stdlib functions."""
        src = """
        import brdf;
        import color;

        vertex {
            in position: vec3;
            in normal: vec3;
            out frag_normal: vec3;
            out frag_pos: vec3;

            uniform MVP {
                model: mat4,
                view: mat4,
                projection: mat4,
            }

            fn main() {
                let wp: vec4 = model * vec4(position, 1.0);
                frag_pos = wp.xyz;
                frag_normal = normalize((model * vec4(normal, 0.0)).xyz);
                builtin_position = projection * view * wp;
            }
        }

        fragment {
            in frag_normal: vec3;
            in frag_pos: vec3;
            out color: vec4;

            push Params { view_pos: vec3, light_pos: vec3 }

            fn main() {
                let n: vec3 = normalize(frag_normal);
                let v: vec3 = normalize(view_pos - frag_pos);
                let l: vec3 = normalize(light_pos - frag_pos);
                let result: vec3 = pbr_brdf(n, v, l, vec3(0.8, 0.2, 0.1), 0.4, 0.0);
                let mapped: vec3 = tonemap_aces(result);
                color = vec4(linear_to_srgb(mapped), 1.0);
            }
        }
        """
        compile_source(src, "full_pbr", tmp_path, validate=True)
        assert (tmp_path / "full_pbr.vert.spv").exists()
        assert (tmp_path / "full_pbr.frag.spv").exists()

    def test_type_aliases_from_import(self, tmp_path):
        """Type aliases defined in imported stdlib should be usable."""
        src = """
        import brdf;

        fragment {
            in n: vec3;
            in l: vec3;
            out color: vec4;
            fn main() {
                let light: Radiance = vec3(1.0, 0.9, 0.8);
                let surface: Reflectance = vec3(0.5);
                let ndotl: scalar = max(dot(n, l), 0.0);
                let result: vec3 = surface * ndotl;
                color = vec4(result, 1.0);
            }
        }
        """
        compile_source(src, "alias_import", tmp_path, validate=True)
        assert (tmp_path / "alias_import.frag.spv").exists()
