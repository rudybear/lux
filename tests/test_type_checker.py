"""Tests for the Lux type checker."""

import pytest
from luxc.parser.tree_builder import parse_lux
from luxc.analysis.type_checker import type_check, TypeCheckError
from luxc.analysis.layout_assigner import assign_layouts
from luxc.builtins.types import clear_type_aliases


class TestTypeCheckPositive:
    """Programs that should type-check successfully."""

    def test_minimal_vertex(self):
        src = """
        vertex {
            in position: vec3;
            out frag_color: vec3;
            fn main() {
                frag_color = position;
                builtin_position = vec4(position, 1.0);
            }
        }
        """
        m = parse_lux(src)
        type_check(m)  # should not raise

    def test_minimal_fragment(self):
        src = """
        fragment {
            in frag_color: vec3;
            out color: vec4;
            fn main() {
                color = vec4(frag_color, 1.0);
            }
        }
        """
        m = parse_lux(src)
        type_check(m)

    def test_let_stmt(self):
        src = """
        fragment {
            in n: vec3;
            out color: vec4;
            fn main() {
                let nn: vec3 = normalize(n);
                color = vec4(nn, 1.0);
            }
        }
        """
        m = parse_lux(src)
        type_check(m)

    def test_binary_ops(self):
        src = """
        fragment {
            in a: vec3;
            in b: vec3;
            out color: vec4;
            fn main() {
                let c: vec3 = a + b;
                let d: vec3 = a - b;
                let e: vec3 = a * 2.0;
                color = vec4(c, 1.0);
            }
        }
        """
        m = parse_lux(src)
        type_check(m)

    def test_matrix_multiply(self):
        src = """
        vertex {
            in position: vec3;
            uniform MVP { model: mat4, view: mat4, projection: mat4 }
            fn main() {
                let wp: vec4 = model * vec4(position, 1.0);
                builtin_position = projection * view * wp;
            }
        }
        """
        m = parse_lux(src)
        type_check(m)

    def test_swizzle_types(self):
        src = """
        fragment {
            in v: vec4;
            out color: vec4;
            fn main() {
                let xyz: vec3 = v.xyz;
                let x: scalar = v.x;
                color = vec4(xyz, x);
            }
        }
        """
        m = parse_lux(src)
        type_check(m)

    def test_dot_product(self):
        src = """
        fragment {
            in n: vec3;
            in l: vec3;
            out color: vec4;
            fn main() {
                let d: scalar = dot(n, l);
                color = vec4(d, d, d, 1.0);
            }
        }
        """
        m = parse_lux(src)
        type_check(m)

    def test_constants(self):
        src = """
        const PI: scalar = 3.14159;
        fragment {
            out color: vec4;
            fn main() {
                color = vec4(PI, 0.0, 0.0, 1.0);
            }
        }
        """
        m = parse_lux(src)
        type_check(m)

    def test_push_constants(self):
        src = """
        fragment {
            out color: vec4;
            push Camera { view_pos: vec3 }
            fn main() {
                color = vec4(view_pos, 1.0);
            }
        }
        """
        m = parse_lux(src)
        type_check(m)


class TestTypeCheckNegative:
    """Programs that should be rejected by the type checker."""

    def test_undefined_variable(self):
        src = """
        fragment {
            out color: vec4;
            fn main() {
                color = vec4(undefined_var, 1.0);
            }
        }
        """
        m = parse_lux(src)
        with pytest.raises(TypeCheckError, match="Undefined variable"):
            type_check(m)

    def test_type_mismatch_assign(self):
        src = """
        fragment {
            in n: vec3;
            out color: vec4;
            fn main() {
                color = n;
            }
        }
        """
        m = parse_lux(src)
        with pytest.raises(TypeCheckError, match="Type mismatch"):
            type_check(m)

    def test_unknown_function(self):
        src = """
        fragment {
            in n: vec3;
            out color: vec4;
            fn main() {
                let x: vec3 = unknown_func(n);
                color = vec4(x, 1.0);
            }
        }
        """
        m = parse_lux(src)
        with pytest.raises(TypeCheckError, match="No matching overload"):
            type_check(m)


class TestTypeAliases:
    """Tests for radiometric type aliases."""

    def setup_method(self):
        clear_type_aliases()

    def teardown_method(self):
        clear_type_aliases()

    def test_type_alias_resolves(self):
        src = """
        type Radiance = vec3;
        fragment {
            out color: vec4;
            fn main() {
                let light: Radiance = vec3(1.0, 0.9, 0.8);
                color = vec4(light, 1.0);
            }
        }
        """
        m = parse_lux(src)
        type_check(m)  # should not raise

    def test_type_alias_in_io(self):
        src = """
        type Radiance = vec3;
        vertex {
            in position: vec3;
            out light_out: Radiance;
            fn main() {
                light_out = vec3(1.0, 0.9, 0.8);
                builtin_position = vec4(position, 1.0);
            }
        }
        """
        m = parse_lux(src)
        type_check(m)

    def test_type_alias_compatibility(self):
        """Radiance (=vec3) can be assigned to vec3 output."""
        src = """
        type Radiance = vec3;
        fragment {
            in light: Radiance;
            out color: vec4;
            fn main() {
                let normalized: vec3 = normalize(light);
                color = vec4(normalized, 1.0);
            }
        }
        """
        m = parse_lux(src)
        type_check(m)

    def test_multiple_aliases_same_base(self):
        src = """
        type Radiance = vec3;
        type Reflectance = vec3;
        type Direction = vec3;
        type Normal = vec3;
        fragment {
            in light_dir: Direction;
            in n: Normal;
            out color: vec4;
            fn main() {
                let d: scalar = dot(light_dir, n);
                color = vec4(d, d, d, 1.0);
            }
        }
        """
        m = parse_lux(src)
        type_check(m)

    def test_unknown_alias_target_rejected(self):
        src = "type Bad = nonexistent_type;"
        m = parse_lux(src)
        with pytest.raises(TypeCheckError, match="Unknown target type"):
            type_check(m)

    def test_alias_in_function_param(self):
        src = """
        type Direction = vec3;
        fn helper(d: Direction) -> scalar {
            return length(d);
        }
        fragment {
            in n: vec3;
            out color: vec4;
            fn main() {
                let l: scalar = helper(n);
                color = vec4(l, l, l, 1.0);
            }
        }
        """
        m = parse_lux(src)
        type_check(m)


class TestLayoutAssigner:
    def test_input_locations(self):
        src = """
        vertex {
            in position: vec3;
            in normal: vec3;
            in uv: vec2;
            fn main() {
                builtin_position = vec4(position, 1.0);
            }
        }
        """
        m = parse_lux(src)
        type_check(m)
        assign_layouts(m)
        stage = m.stages[0]
        assert stage.inputs[0].location == 0
        assert stage.inputs[1].location == 1
        assert stage.inputs[2].location == 2

    def test_output_locations(self):
        src = """
        vertex {
            in pos: vec3;
            out a: vec3;
            out b: vec2;
            fn main() {
                a = pos;
                b = vec2(0.0, 0.0);
                builtin_position = vec4(pos, 1.0);
            }
        }
        """
        m = parse_lux(src)
        type_check(m)
        assign_layouts(m)
        stage = m.stages[0]
        assert stage.outputs[0].location == 0
        assert stage.outputs[1].location == 1

    def test_uniform_binding(self):
        src = """
        vertex {
            in pos: vec3;
            uniform MVP { model: mat4, view: mat4, projection: mat4 }
            fn main() {
                builtin_position = projection * view * model * vec4(pos, 1.0);
            }
        }
        """
        m = parse_lux(src)
        type_check(m)
        assign_layouts(m)
        ub = m.stages[0].uniforms[0]
        assert ub.set_number == 0
        assert ub.binding == 0
