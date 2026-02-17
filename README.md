# Lux — Math-First Shader Language

Lux is a shader language designed for humans and LLMs alike. Write rendering math directly, and the compiler handles GPU translation to SPIR-V for Vulkan.

No `layout(set=0, binding=1)`. No `gl_Position`. Just math.

```
vertex {
    in position: vec3;
    in color: vec3;
    out frag_color: vec3;

    fn main() {
        frag_color = color;
        builtin_position = vec4(position, 1.0);
    }
}

fragment {
    in frag_color: vec3;
    out color: vec4;

    fn main() {
        color = vec4(frag_color, 1.0);
    }
}
```

## Features

- **Math-first syntax** — `scalar` not `float`, `builtin_position` not `gl_Position`
- **Auto-layout** — locations, descriptor sets, and bindings assigned by declaration order
- **One file, multi-stage** — `vertex {}` and `fragment {}` blocks in a single `.lux` file
- **Rust-like types** — `: type` annotations, `let` bindings, `fn` functions
- **Full SPIR-V output** — compiles to validated `.spv` binaries via `spirv-as` + `spirv-val`
- **30+ built-in functions** — `normalize`, `dot`, `pow`, `mix`, `clamp`, `smoothstep`, texture sampling, etc.
- **Swizzle** — `v.xyz`, `v.rg`, `v.x`
- **Constructors** — `vec4(pos, 1.0)`, `vec3(0.5)` (splat)
- **Uniform blocks** with std140 layout, push constants, combined image samplers
- **User-defined functions** with inlining

## Prerequisites

- **Python 3.11+**
- **SPIR-V Tools** (`spirv-as`, `spirv-val`) — from the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) or [spirv-tools](https://github.com/KhronosGroup/SPIRV-Tools)

## Installation

```bash
git clone https://github.com/rudybear/lux.git
cd lux
pip install -e ".[dev]"
```

Verify:

```bash
python -m luxc --version
# luxc 0.1.0
```

## Usage

### Compile a shader

```bash
python -m luxc examples/hello_triangle.lux
# Wrote examples/hello_triangle.vert.spv
# Wrote examples/hello_triangle.frag.spv
```

### Options

```
python -m luxc input.lux [options]

Options:
  --emit-asm        Write .spvasm text files alongside .spv
  --dump-ast        Dump the AST as JSON and exit
  --no-validate     Skip spirv-val validation
  -o OUTPUT_DIR     Output directory (default: same as input)
  --version         Show version
```

### Inspect generated assembly

```bash
python -m luxc examples/hello_triangle.lux --emit-asm
cat examples/hello_triangle.vert.spvasm
```

## Language Reference

### Types

| Lux Type | SPIR-V | Notes |
|----------|--------|-------|
| `scalar` | `OpTypeFloat 32` | Always f32 |
| `int` / `uint` | `OpTypeInt 32` | Signed / unsigned |
| `bool` | `OpTypeBool` | |
| `vec2` / `vec3` / `vec4` | `OpTypeVector` | Float vectors |
| `ivec2/3/4` / `uvec2/3/4` | `OpTypeVector` | Integer vectors |
| `mat2` / `mat3` / `mat4` | `OpTypeMatrix` | Column-major |
| `sampler2d` | `OpTypeSampledImage` | Combined image sampler |

### Stage Blocks

```
vertex {
    in position: vec3;          // auto location=0
    in normal: vec3;            // auto location=1
    out frag_normal: vec3;      // auto location=0

    uniform MVP {               // auto set=0, binding=0
        model: mat4,
        view: mat4,
        projection: mat4,
    }

    push Camera { view_pos: vec3 }

    fn main() {
        builtin_position = projection * view * model * vec4(position, 1.0);
    }
}

fragment {
    in frag_normal: vec3;
    out color: vec4;

    sampler2d albedo_tex;       // auto set=0, binding=0

    fn main() {
        color = vec4(normalize(frag_normal), 1.0);
    }
}
```

### Built-in Functions

**GLSL.std.450**: `normalize`, `reflect`, `pow`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `exp`, `exp2`, `log`, `log2`, `sqrt`, `abs`, `sign`, `floor`, `ceil`, `fract`, `min`, `max`, `clamp`, `mix`, `step`, `smoothstep`, `length`, `distance`, `cross`, `fma`

**Native SPIR-V**: `dot` (OpDot)

**Texture**: `sample(tex, uv)` (OpImageSampleImplicitLod)

### User-Defined Functions

```
fn fresnel_schlick(cos_theta: scalar, f0: vec3) -> vec3 {
    return f0 + (vec3(1.0) - f0) * pow(1.0 - cos_theta, 5.0);
}
```

Module-level functions can be called from any stage block. They are inlined at the call site.

### Constants

```
const PI: scalar = 3.14159265;
```

## Compiler Pipeline

```
input.lux
  -> Lark Parser         (lux.lark grammar)
  -> Tree Builder         (Transformer -> AST dataclasses)
  -> Type Checker         (resolve types, check operators, validate)
  -> Layout Assigner      (auto-assign location/set/binding)
  -> SPIR-V Builder       (emit .spvasm text)
  -> spirv-as             (assemble to .spv binary)
  -> spirv-val            (validate)
  -> output: name.vert.spv, name.frag.spv
```

## Project Structure

```
luxc/
    __init__.py
    __main__.py              # python -m luxc
    cli.py                   # argparse CLI
    compiler.py              # pipeline orchestration
    grammar/
        lux.lark             # Lark EBNF grammar
    parser/
        ast_nodes.py         # AST dataclasses
        tree_builder.py      # Lark Transformer -> AST
    analysis/
        symbols.py           # Symbol table, scopes
        type_checker.py      # Type checking + overload resolution
        layout_assigner.py   # Auto-assign locations/bindings/sets
    builtins/
        types.py             # Built-in type definitions
        functions.py         # Built-in function signatures
    codegen/
        spirv_builder.py     # SPIR-V assembly text generator
        spirv_types.py       # Type registry + deduplication
        glsl_ext.py          # GLSL.std.450 instruction mappings
        spv_assembler.py     # spirv-as / spirv-val invocation
tests/
    test_parser.py
    test_type_checker.py
    test_codegen.py
    test_e2e.py
    fixtures/
        hello_triangle.lux
        minimal_vertex.lux
        minimal_fragment.lux
examples/
    hello_triangle.lux       # Simplest working program
    pbr_basic.lux            # Diffuse + specular + Fresnel
```

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

Requires `spirv-as` and `spirv-val` on PATH for end-to-end tests.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| `scalar` not `float` | Mathematical vocabulary | Reads naturally in equations |
| No layout qualifiers | Auto-assigned by order | Eliminates #1 source of GLSL bugs |
| `builtin_position` | Explicit naming | No magic globals, greppable |
| `:` type annotations | Rust-like syntax | LLMs generate this reliably |
| One file, multi-stage | `vertex {}` / `fragment {}` | Natural unit of compilation |
| Explicit types | No inference (v1) | LLMs produce more correct code |
| Direct AST to SPIR-V | No IR in v1 | Simpler, faster to working output |
| Function inlining | No SPIR-V OpFunctionCall | Simplifies codegen for v1 |

## License

MIT
