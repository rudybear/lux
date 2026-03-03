# Lux Shader Debugger Guide

The Lux CPU-side shader debugger lets you step through shader code, inspect variables, and detect NaN/Inf issues — all without a GPU. It works by interpreting the shader AST directly in Python, with the same math semantics as SPIR-V.

## Quick Start

```bash
# Interactive debugging (gdb-style REPL)
python -m luxc examples/debug_playground.lux --debug-run --stage fragment

# Batch mode — detect NaN/Inf issues
python -m luxc examples/debug_playground.lux --debug-run --stage fragment --batch --check-nan

# Batch mode — dump all variable values
python -m luxc examples/debug_playground.lux --debug-run --stage fragment --batch --dump-vars

# Batch mode — break at specific lines and dump state
python -m luxc examples/debug_playground.lux --debug-run --stage fragment --batch --break 100 --dump-at-break

# Custom inputs — override default values
python -m luxc examples/debug_playground.lux --debug-run --stage fragment --input examples/debug_playground_inputs.json --batch --dump-vars
```

## CLI Flags

| Flag | Description |
|------|-------------|
| `--debug-run` | Launch the CPU debugger instead of compiling |
| `--stage <name>` | Which stage to debug: `fragment`, `vertex`, `compute`, etc. (default: `fragment`) |
| `--batch` | Machine-readable JSON output mode (no interactive REPL) |
| `--dump-vars` | Record every variable assignment (batch mode) |
| `--check-nan` | Flag NaN/Inf values with source location (batch mode) |
| `--break <line>` | Set a breakpoint at a source line (repeatable) |
| `--dump-at-break` | Dump all visible variables when a breakpoint is hit |
| `--input <file.json>` | Load custom input values from JSON |
| `--pixel X,Y` | Debug a specific pixel (auto-computes uv, position, normal) |
| `--resolution WxH` | Screen resolution for `--pixel` (default: 1920x1080) |
| `--set VAR=VALUE` | Override a single input (repeatable, e.g. `--set roughness=0.1`) |
| `--export-inputs <file.json>` | Export default input values to JSON for editing |

## Interactive Commands

When running without `--batch`, you get an interactive REPL:

| Command | Short | Description |
|---------|-------|-------------|
| `start` | | Begin execution, stop at first statement |
| `run` | `r` | Run to completion (stops only at breakpoints) |
| `step` | `s` | Step into the next statement — shows new/changed variables |
| `next` | `n` | Step over (don't enter function calls) |
| `continue` | `c` | Continue to next breakpoint |
| `finish` | `f` | Run until current function returns |
| `break <line>` | `b <line>` | Set a breakpoint at a source line |
| `delete <id>` | `d <id>` | Remove a breakpoint |
| `print <var>` | `p <var>` | Print a variable's current value |
| `locals` | `l` | Show all variables in current scope |
| `list` | | Show full shader with `>` cursor and `*` breakpoint markers |
| `watch <expr>` | `w <expr>` | Add a watch expression |
| `source [line]` | `src [line]` | Show source code around current/specified line (5 lines) |
| `output` | | Show the current output variable |
| `breakpoints` | `info` | List all breakpoints |
| `backtrace` | `bt` | Show current position |
| `quit` | `q` | Exit the debugger |

When stepping, variable changes are shown automatically:
```
lux-debug> step
  + albedo = vec3(0.800, 0.800, 0.800) (vec3)    <- new variable
Stopped at line 88
 >   88 |         let n: vec3 = normalize(world_normal);
```

### Example Interactive Session

```
$ python -m luxc examples/debug_playground.lux --debug-run --stage fragment

lux-debug> start
Stopped at line 87
 >   87 |         let albedo: vec3 = sample(albedo_tex, uv).rgb;

lux-debug> step
Stopped at line 88
 >   88 |         let n: vec3 = normalize(world_normal);

lux-debug> step
Stopped at line 91
lux-debug> print albedo
  albedo = vec3(0.800, 0.800, 0.800) (vec3)
lux-debug> print n
  n = vec3(0.000, 0.000, 1.000) (vec3)
lux-debug> locals
  albedo = vec3(0.800, 0.800, 0.800)
  n = vec3(0.000, 0.000, 1.000)
  roughness = 0.500 (scalar)
  metallic = 0.000 (scalar)
  ...
lux-debug> break 113
Breakpoint 1 at line 113
lux-debug> continue
Hit breakpoint 1 at line 113
lux-debug> print d
  d = 0.141471 (scalar)
lux-debug> print g
  g = 0.640000 (scalar)
lux-debug> continue
Output: vec4(0.403, 0.381, 0.345, 1.000)
```

**Workflow tips:**
- Use `start` or `step` to begin stepping from line 1
- Use `break <line> + run` to jump directly to a specific location
- Use `step` inside functions to follow the call, `next` to skip over them
- Use `finish` to run until the current function returns

## Batch Mode JSON Output

The `--batch` flag outputs a JSON object to stdout:

```json
{
  "status": "completed",
  "statements_executed": 53,
  "nan_detected": true,
  "output": {
    "type": "vec4",
    "value": [0.403, 0.381, 0.345, 1.0]
  },
  "nan_events": [
    {
      "line": 55,
      "variable": "dir",
      "operation": "let",
      "value": {"type": "vec3", "value": ["NaN", "NaN", "NaN"]}
    }
  ],
  "variable_trace": [
    {
      "line": 84,
      "name": "albedo",
      "type": "vec3",
      "value": {"type": "vec3", "value": [0.8, 0.8, 0.8]}
    }
  ],
  "break_dumps": [
    {
      "line": 100,
      "variables": {
        "albedo": {"type": "vec3", "value": [0.8, 0.8, 0.8]},
        "roughness": {"type": "scalar", "value": 0.5}
      }
    }
  ]
}
```

### Fields

| Field | Present When | Description |
|-------|-------------|-------------|
| `status` | Always | `"completed"` or `"error"` |
| `statements_executed` | Always | Number of statements the interpreter ran |
| `nan_detected` | `--check-nan` | Whether any NaN/Inf was found |
| `output` | Always | The first output variable (e.g., `color`) |
| `nan_events` | `--check-nan` and NaN found | Array of NaN events with source line |
| `variable_trace` | `--dump-vars` | Array of every variable assignment |
| `break_dumps` | `--dump-at-break` | Scope snapshots at each breakpoint hit |
| `debug_prints` | If shader has `debug_print()` | Printf-style output |
| `assert_failures` | If asserts fail | Failed assertion messages |

## Debugging a Specific Pixel

Use `--pixel X,Y` to debug the shader at a specific screen position:

```bash
# Center of a 1920x1080 screen (default resolution)
python -m luxc examples/debug_playground.lux --debug-run --stage fragment --pixel 960,540

# Top-left corner of an 800x600 screen
python -m luxc examples/debug_playground.lux --debug-run --stage fragment --pixel 0,0 --resolution 800x600

# Edge pixel — steep viewing angle, good for testing Fresnel
python -m luxc examples/debug_playground.lux --debug-run --stage fragment --pixel 50,300 --resolution 800x600 --batch --check-nan
```

`--pixel X,Y` auto-computes from the pixel coordinates:

| Variable | Derivation |
|----------|-----------|
| `uv` | `((X+0.5)/W, (Y+0.5)/H)` — pixel center in [0,1] |
| `world_position` | Mapped to [-1,1] unit quad |
| `normal` / `world_normal` | Hemisphere: center faces camera, edges face sideways |

### Quick Inline Overrides with `--set`

Override individual variables without creating a JSON file:

```bash
# Mirror-smooth metal
python -m luxc examples/debug_playground.lux --debug-run --stage fragment --set roughness=0.01 --set metallic=1.0 --batch

# Custom normal direction
python -m luxc examples/debug_playground.lux --debug-run --stage fragment --set "normal=1,0,0" --batch --check-nan

# Combine pixel + material overrides
python -m luxc examples/debug_playground.lux --debug-run --stage fragment --pixel 100,200 --set roughness=0.0 --batch --check-nan
```

Supported value formats for `--set`:
- **Scalar:** `--set roughness=0.5`
- **Vector:** `--set "normal=0,1,0"` (comma-separated, no spaces)
- **Boolean:** `--set flag=true`

Multiple `--set` flags stack, and they merge with `--input` JSON if both are provided.

## Custom Inputs

The debugger provides sensible defaults for common shader inputs:

| Input Name | Default Value |
|------------|---------------|
| `position`, `world_position` | `vec3(0, 0, 0)` |
| `normal`, `world_normal` | `vec3(0, 0, 1)` — pointing toward camera |
| `uv`, `texcoord` | `vec2(0.5, 0.5)` — center of texture |
| `roughness` | `0.5` |
| `metallic` | `0.0` — dielectric |
| `exposure` | `1.0` |
| `ao`, `occlusion` | `1.0` — no occlusion |
| All `mat4` uniforms | Identity matrix |
| All `sampler2d` samples | `vec4(0.8, 0.8, 0.8, 1.0)` — neutral grey |

### Override with JSON

Export defaults, edit, then replay:

```bash
# Export
python -m luxc examples/debug_playground.lux --debug-run --export-inputs my_inputs.json

# Edit my_inputs.json to change values...

# Replay with custom inputs
python -m luxc examples/debug_playground.lux --debug-run --stage fragment --input my_inputs.json --batch --dump-vars
```

Input JSON format:

```json
{
  "world_normal": [0.0, 1.0, 0.0],
  "world_position": [1.0, 0.0, -2.0],
  "uv": [0.5, 0.5],
  "roughness": 0.35,
  "metallic": 0.0,
  "exposure": 1.5
}
```

Supported value formats:
- **Scalar:** `0.5` or `{"type": "scalar", "value": 0.5}`
- **Vector:** `[1.0, 2.0, 3.0]` or `{"type": "vec3", "value": [1.0, 2.0, 3.0]}`
- **Matrix:** `[[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]`
- **Integer:** `42` or `{"type": "int", "value": 42}`
- **Boolean:** `true` or `{"type": "bool", "value": true}`

## Debugging Strategies

### Finding NaN/Inf Sources

```bash
python -m luxc shader.lux --debug-run --stage fragment --batch --check-nan
```

Common NaN sources in shaders:
- `normalize(vec3(0, 0, 0))` — normalizing a zero vector
- `0.0 / 0.0` — zero divided by zero
- `sqrt(-x)` — square root of negative
- `pow(0.0, -1.0)` — zero to negative power
- `acos(1.0001)` — domain error in trig

The `nan_events` array tells you exactly which variable and which line.

### Simulating Different Pixels

Override inputs to simulate specific screen positions or geometry:

```json
{"world_normal": [1.0, 0.0, 0.0], "roughness": 0.01, "metallic": 1.0}
```

This simulates a highly metallic, mirror-smooth surface with a side-facing normal — a case that often triggers specular edge issues.

### Tracing the Full PBR Pipeline

```bash
python -m luxc examples/debug_playground.lux --debug-run --stage fragment --batch --dump-vars --check-nan
```

The `variable_trace` shows every intermediate value in execution order, letting you follow the math step by step:

```
albedo    = vec3(0.800, 0.800, 0.800)    # from texture
n         = vec3(0.000, 0.000, 1.000)    # surface normal
n_dot_l   = 0.333                         # Lambert factor
d         = 0.141                         # GGX NDF
g         = 0.640                         # Smith geometry
f         = vec3(0.040, 0.040, 0.040)    # Fresnel
specular  = vec3(0.003, 0.003, 0.003)    # BRDF specular
diffuse   = vec3(0.244, 0.244, 0.244)    # Lambert diffuse
direct    = vec3(0.247, 0.231, 0.206)    # Combined with light color
hdr_color = vec3(0.271, 0.255, 0.230)    # + rim + ambient
ldr_color = vec3(0.403, 0.381, 0.345)    # After ACES tonemap
```

### Breakpoint-Based Inspection

Set breakpoints at key stages and dump the full variable state:

```bash
python -m luxc examples/debug_playground.lux --debug-run --stage fragment \
  --batch --break 100 --break 113 --break 137 --dump-at-break
```

Line 100: after material setup (check f0, diffuse_color)
Line 113: after BRDF denominator (check specular term is finite)
Line 137: before tonemap (check HDR range is reasonable)

## Supported Language Features

The debugger supports the full Lux language except GPU-specific intrinsics:

- All arithmetic, comparison, and logical operators
- `let` bindings, `if`/`else`, `for`, `while`, `break`, `continue`
- User-defined functions with arguments and return values
- Vector/matrix constructors: `vec3(1.0)`, `mat4(1.0)`, `vec4(v3, 1.0)`
- Swizzle access: `.xyz`, `.rgb`, `.xy`, single component `.x`
- Index access: `v[0]`, `m[col][row]`
- Struct field access
- 44+ math builtins matching GLSL.std.450 semantics
- `debug_print()` and `assert()` statements
- Module-level constants
- Texture `sample()` (returns mock data)

### What's NOT Supported

- Ray tracing intrinsics (`traceRay`, etc.)
- Compute shader atomics and barriers
- Actual texture data (sampling returns constant grey)
- Bindless descriptor indexing at runtime
- `dFdx`/`dFdy` screen-space derivatives

## Architecture

```
luxc/debug/
    values.py        — Value types (LuxScalar, LuxVec, LuxMat, LuxInt, LuxBool, LuxStruct)
    environment.py   — Scoped variable storage with parent chain lookup
    builtins.py      — 44+ math builtins (sin, cos, normalize, dot, cross, mix, ...)
    interpreter.py   — Tree-walking AST evaluator (exec_stmt, eval_expr)
    debugger.py      — Breakpoints, stepping modes (step/next/continue), NaN detection
    io.py            — Input loading (JSON files, semantic defaults, reflection)
    cli.py           — Interactive REPL and batch mode orchestration
```

The debugger reuses the compiler's parser and type checker but skips codegen entirely. It runs the AST directly in Python, making it fast to iterate and independent of GPU drivers.
