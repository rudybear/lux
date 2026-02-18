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


@requires_spirv_tools
class TestSurfaceExpansion:
    def setup_method(self):
        clear_type_aliases()

    def teardown_method(self):
        clear_type_aliases()

    def test_surface_lambert(self, tmp_path):
        """Standalone surface with lambert BRDF generates fragment shader."""
        src = """
        import brdf;
        surface RedMatte {
            brdf: lambert(vec3(0.8, 0.2, 0.1)),
        }
        """
        compile_source(src, "surface_lambert", tmp_path, validate=True)
        assert (tmp_path / "surface_lambert.frag.spv").exists()

    def test_surface_pbr(self, tmp_path):
        """Surface with full PBR BRDF."""
        src = """
        import brdf;
        surface Metal {
            brdf: pbr(vec3(0.95, 0.64, 0.54), 0.3, 0.9),
        }
        """
        compile_source(src, "surface_pbr", tmp_path, validate=True)
        assert (tmp_path / "surface_pbr.frag.spv").exists()

    def test_surface_microfacet(self, tmp_path):
        """Surface with microfacet GGX specular only."""
        src = """
        import brdf;
        surface GlossyPlastic {
            brdf: microfacet_ggx(0.1, vec3(0.04)),
        }
        """
        compile_source(src, "surface_specular", tmp_path, validate=True)
        assert (tmp_path / "surface_specular.frag.spv").exists()

    def test_full_pipeline(self, tmp_path):
        """Full pipeline: geometry + surface generates vertex + fragment."""
        src = """
        import brdf;
        geometry Mesh {
            position: vec3,
            normal: vec3,
            transform: MVP {
                model: mat4,
                view: mat4,
                projection: mat4,
            }
            outputs {
                world_pos: (model * vec4(position, 1.0)).xyz,
                world_normal: normalize((model * vec4(normal, 0.0)).xyz),
                clip_pos: projection * view * model * vec4(position, 1.0),
            }
        }
        surface Material {
            brdf: pbr(vec3(0.8, 0.2, 0.1), 0.4, 0.0),
        }
        pipeline Forward {
            geometry: Mesh,
            surface: Material,
        }
        """
        compile_source(src, "full_pipeline", tmp_path, validate=True)
        assert (tmp_path / "full_pipeline.vert.spv").exists()
        assert (tmp_path / "full_pipeline.frag.spv").exists()

    def test_pbr_surface_example(self, tmp_path):
        """Compile the pbr_surface.lux example file."""
        src = (Path(__file__).parent.parent / "examples" / "pbr_surface.lux").read_text()
        compile_source(src, "pbr_surface", tmp_path, validate=True)
        assert (tmp_path / "pbr_surface.vert.spv").exists()
        assert (tmp_path / "pbr_surface.frag.spv").exists()


@requires_spirv_tools
class TestSDFStdlib:
    def setup_method(self):
        clear_type_aliases()

    def teardown_method(self):
        clear_type_aliases()

    def test_import_sdf(self, tmp_path):
        """Compile a fragment shader using SDF primitives and CSG."""
        src = """
        import sdf;

        fragment {
            in uv: vec2;
            out color: vec4;
            fn main() {
                let p: vec3 = vec3(uv - vec2(0.5), 0.0);
                let d1: scalar = sdf_sphere(p, 0.3);
                let d2: scalar = sdf_box(sdf_translate(p, vec3(0.2, 0.0, 0.0)), vec3(0.15, 0.15, 0.15));
                let d: scalar = sdf_smooth_union(d1, d2, 0.1);
                let c: scalar = smoothstep(0.01, 0.0, d);
                color = vec4(vec3(c), 1.0);
            }
        }
        """
        compile_source(src, "import_sdf", tmp_path, validate=True)
        assert (tmp_path / "import_sdf.frag.spv").exists()

    def test_sdf_example(self, tmp_path):
        """Compile the sdf_shapes.lux example file."""
        src = (Path(__file__).parent.parent / "examples" / "sdf_shapes.lux").read_text()
        compile_source(src, "sdf_shapes", tmp_path, validate=True)
        assert (tmp_path / "sdf_shapes.frag.spv").exists()


@requires_spirv_tools
class TestNoiseStdlib:
    def setup_method(self):
        clear_type_aliases()

    def teardown_method(self):
        clear_type_aliases()

    def test_import_noise(self, tmp_path):
        """Compile a fragment shader using noise functions."""
        src = """
        import noise;

        fragment {
            in uv: vec2;
            out color: vec4;
            fn main() {
                let n: scalar = fbm2d_4(uv * 8.0, 2.0, 0.5);
                let c: scalar = clamp(n + 0.5, 0.0, 1.0);
                color = vec4(vec3(c), 1.0);
            }
        }
        """
        compile_source(src, "import_noise", tmp_path, validate=True)
        assert (tmp_path / "import_noise.frag.spv").exists()

    def test_noise_example(self, tmp_path):
        """Compile the procedural_noise.lux example file."""
        src = (Path(__file__).parent.parent / "examples" / "procedural_noise.lux").read_text()
        compile_source(src, "procedural_noise", tmp_path, validate=True)
        assert (tmp_path / "procedural_noise.frag.spv").exists()


@requires_spirv_tools
class TestScheduleSystem:
    def setup_method(self):
        clear_type_aliases()

    def teardown_method(self):
        clear_type_aliases()

    def test_schedule_default(self, tmp_path):
        """Pipeline without schedule compiles with default behavior."""
        src = """
        import brdf;
        geometry Mesh {
            position: vec3,
            normal: vec3,
            transform: MVP {
                model: mat4,
                view: mat4,
                projection: mat4,
            }
            outputs {
                world_pos: (model * vec4(position, 1.0)).xyz,
                world_normal: normalize((model * vec4(normal, 0.0)).xyz),
                clip_pos: projection * view * model * vec4(position, 1.0),
            }
        }
        surface Material {
            brdf: pbr(vec3(0.8, 0.2, 0.1), 0.4, 0.0),
        }
        pipeline Forward {
            geometry: Mesh,
            surface: Material,
        }
        """
        compile_source(src, "sched_default", tmp_path, validate=True)
        assert (tmp_path / "sched_default.vert.spv").exists()
        assert (tmp_path / "sched_default.frag.spv").exists()

    def test_schedule_mobile(self, tmp_path):
        """Pipeline with fast distribution/geometry schedule compiles."""
        src = """
        import brdf;
        schedule Mobile {
            distribution: ggx_fast,
            geometry_term: smith_ggx_fast,
        }
        geometry Mesh {
            position: vec3,
            normal: vec3,
            transform: MVP {
                model: mat4,
                view: mat4,
                projection: mat4,
            }
            outputs {
                world_pos: (model * vec4(position, 1.0)).xyz,
                world_normal: normalize((model * vec4(normal, 0.0)).xyz),
                clip_pos: projection * view * model * vec4(position, 1.0),
            }
        }
        surface Material {
            brdf: pbr(vec3(0.8, 0.2, 0.1), 0.4, 0.0),
        }
        pipeline Forward {
            geometry: Mesh,
            surface: Material,
            schedule: Mobile,
        }
        """
        compile_source(src, "sched_mobile", tmp_path, validate=True)
        assert (tmp_path / "sched_mobile.vert.spv").exists()
        assert (tmp_path / "sched_mobile.frag.spv").exists()

    def test_schedule_high_quality(self, tmp_path):
        """Pipeline with schlick_roughness fresnel schedule compiles."""
        src = """
        import brdf;
        schedule HighQuality {
            fresnel: schlick_roughness,
        }
        geometry Mesh {
            position: vec3,
            normal: vec3,
            transform: MVP {
                model: mat4,
                view: mat4,
                projection: mat4,
            }
            outputs {
                world_pos: (model * vec4(position, 1.0)).xyz,
                world_normal: normalize((model * vec4(normal, 0.0)).xyz),
                clip_pos: projection * view * model * vec4(position, 1.0),
            }
        }
        surface Material {
            brdf: microfacet_ggx(0.3, vec3(0.04)),
        }
        pipeline Forward {
            geometry: Mesh,
            surface: Material,
            schedule: HighQuality,
        }
        """
        compile_source(src, "sched_hq", tmp_path, validate=True)
        assert (tmp_path / "sched_hq.vert.spv").exists()
        assert (tmp_path / "sched_hq.frag.spv").exists()

    def test_scheduled_pbr_example(self, tmp_path):
        """Compile the scheduled_pbr.lux example file."""
        src = (Path(__file__).parent.parent / "examples" / "scheduled_pbr.lux").read_text()
        compile_source(src, "scheduled_pbr", tmp_path, validate=True)
        assert (tmp_path / "scheduled_pbr.vert.spv").exists()
        assert (tmp_path / "scheduled_pbr.frag.spv").exists()


@requires_spirv_tools
class TestConstantFolding:
    def setup_method(self):
        clear_type_aliases()

    def teardown_method(self):
        clear_type_aliases()

    def test_arithmetic_folding(self, tmp_path):
        """Shader with 1.0 + 2.0 compiles (folded to 3.0)."""
        src = """
        fragment {
            out color: vec4;
            fn main() {
                let v: scalar = 1.0 + 2.0;
                color = vec4(v, v, v, 1.0);
            }
        }
        """
        compile_source(src, "fold_arith", tmp_path, validate=True)
        assert (tmp_path / "fold_arith.frag.spv").exists()

    def test_const_reference_folding(self, tmp_path):
        """Module const inlined and folded in shader."""
        src = """
        const HALF: scalar = 0.5;
        fragment {
            out color: vec4;
            fn main() {
                let v: scalar = HALF + HALF;
                color = vec4(v, v, v, 1.0);
            }
        }
        """
        compile_source(src, "fold_const", tmp_path, validate=True)
        assert (tmp_path / "fold_const.frag.spv").exists()

    def test_builtin_math_folding(self, tmp_path):
        """Builtin math calls on literals compile (folded)."""
        src = """
        fragment {
            out color: vec4;
            fn main() {
                let a: scalar = sin(0.0);
                let b: scalar = max(1.0, 2.0);
                color = vec4(a, b, 0.0, 1.0);
            }
        }
        """
        compile_source(src, "fold_builtin", tmp_path, validate=True)
        assert (tmp_path / "fold_builtin.frag.spv").exists()
