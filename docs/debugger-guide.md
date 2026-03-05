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
| `--debug-print` | Preserve `debug_print`/`assert` without full `-g` debug mode |
| `--assert-kill` | Demote fragment invocation on assert failure (`OpDemoteToHelperInvocation`) |
| `--rich-debug` | Emit NonSemantic.Shader.DebugInfo.100 for RenderDoc step-through (implies `-g`) |

## Interactive Commands

When running without `--batch`, you get an interactive REPL:

### Execution Control

| Command | Short | Description |
|---------|-------|-------------|
| `start` | | Begin execution, stop at first statement |
| `run` | `r` | Run to completion (stops only at breakpoints) |
| `step` | `s` | Step into the next statement — shows new/changed variables |
| `next` | `n` | Step over (don't enter function calls) |
| `continue` | `c` | Continue to next breakpoint |
| `finish` | `f` | Run until current function returns |

### Breakpoints

| Command | Short | Description |
|---------|-------|-------------|
| `break <line>` | `b <line>` | Set a breakpoint at a source line |
| `break <line> if <expr>` | | Set a conditional breakpoint (stops only when `<expr>` is truthy) |
| `break-nan` | `bn` | Toggle break-on-NaN (stop when NaN/Inf is first produced) |
| `delete <id>` | `d <id>` | Remove a breakpoint |
| `breakpoints` | `info` | List all breakpoints with conditions and hit counts |

### Inspection

| Command | Short | Description |
|---------|-------|-------------|
| `print <var>` | `p <var>` | Print a variable's current value |
| `eval <expr>` | `e <expr>` | Evaluate any Lux expression in current scope |
| `locals` | `l` | Show all variables in current scope |
| `list` | | Show full shader with `>` cursor and `*` breakpoint markers |
| `watch <expr>` | `w <expr>` | Add a watch expression (evaluated at each stop) |
| `source [line]` | `src [line]` | Show source code around current/specified line (5 lines) |
| `output` | | Show the current output variable |
| `backtrace` | `bt` | Show current position |

### Time-Travel

| Command | Short | Description |
|---------|-------|-------------|
| `reverse-step` | `rs` | Step backwards one statement (restores previous state) |
| `reverse-continue` | `rc` | Run backwards until hitting a breakpoint |
| `goto <step>` | `g <step>` | Jump to a specific step number in execution history |
| `history [N]` | `hist [N]` | Show the last N execution steps (default: 10) |

### General

| Command | Short | Description |
|---------|-------|-------------|
| `quit` | `q` | Exit the debugger |
| `help` | `h` | Show all commands |

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

## Conditional Breakpoints

Breakpoints can include a condition expression. The debugger only stops when the condition evaluates to true.

```
lux-debug> break 100 if roughness < 0.1
Breakpoint 1 at line 100 (condition: roughness < 0.1)
lux-debug> break 55 if dot(n, l) > 0.9
Breakpoint 2 at line 55 (condition: dot(n, l) > 0.9)
```

The condition is parsed using the full Lux expression parser, so it supports all arithmetic, comparisons, function calls, field/swizzle access, and constructors. If the condition fails to parse or evaluate, the breakpoint is treated as unconditional (it always triggers).

The `hit_count` field on each `Breakpoint` tracks how many times the condition was met and the breakpoint fired.

**Programmatic:**
```python
dbg.add_breakpoint(100, condition="roughness < 0.1")
```

## Watch Expressions

The `watch` command evaluates expressions in the current interpreter environment at every stop.

```
lux-debug> watch x + y
Watch 1: x + y
lux-debug> watch dot(n, l)
Watch 2: dot(n, l)
lux-debug> watch length(world_position)
Watch 3: length(world_position)
```

When stepping, watched expressions are re-evaluated and the debugger highlights any that changed since the last stop:

```
lux-debug> step
  [watch 1] x + y = 7.000 (changed)
  [watch 2] dot(n, l) = 0.333
```

**Programmatic:**
```python
w = dbg.add_watch("x + y")
results = dbg.evaluate_watches()
# Returns: [(WatchEntry, LuxValue | None, changed: bool), ...]
```

## Expression Evaluation

Evaluate any Lux expression in the current scope without modifying state:

```
lux-debug> eval dot(n, l) * 2.0
  0.666 (scalar)
lux-debug> e vec3(roughness, metallic, ao)
  vec3(0.500, 0.000, 1.000)
lux-debug> e normalize(world_position - camera_pos)
  vec3(0.577, 0.577, 0.577)
```

Expressions are parsed by the full Lux parser (`luxc.debug.expr_parser.parse_debug_expr`) and evaluated in the interpreter's current environment, giving access to all variables, builtins, and user-defined functions.

## Break-on-NaN

Automatically stop execution the moment a NaN or Inf value is produced, rather than discovering it post-mortem.

**Interactive:**
```
lux-debug> break-nan
NaN break enabled — will stop when NaN/Inf is produced
lux-debug> run
NaN detected at line 55: dir = vec3(NaN, NaN, NaN)
Stopped at line 56
```

**Short form:** `bn`

When break-on-NaN triggers, the debugger switches to step mode so you can inspect the state immediately after the offending operation.

## Time-Travel Debugging

The debugger records execution history, allowing you to step backwards through previously executed statements and inspect past state.

**Example session:**
```
lux-debug> step
Stopped at line 100
lux-debug> step
Stopped at line 101
lux-debug> reverse-step
[time-travel] Restored to step 5, line 100
lux-debug> print roughness
  roughness = 0.500 (scalar)
lux-debug> history 5
  Step 1: line 87
  Step 2: line 88
  Step 3: line 91
  Step 4: line 95
  Step 5: line 100  <-- current
lux-debug> goto 2
[time-travel] Restored to step 2, line 88
```

Each `ExecutionSnapshot` contains:
- `step_index`: monotonically increasing step counter
- `line`: source line number
- `env_snapshot`: deep copy of all variables at that moment

History is capped at 10,000 steps by default to limit memory usage.

**Programmatic:**
```python
snap = dbg.reverse_step()       # ExecutionSnapshot | None
snap = dbg.reverse_continue()   # Run backwards to previous breakpoint
snap = dbg.goto_step(5)         # Jump to step 5
history = dbg.time_travel.get_recent(10)  # Last 10 snapshots
```

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

### Extended Format (Textures and Materials)

For bindless shaders, the input JSON can also specify texture paths and material overrides:

```json
{
  "textures": [
    "assets/textures/albedo.png",
    "assets/textures/normal.png",
    "assets/textures/roughness.png"
  ],
  "materials": [
    {
      "baseColorFactor": [1.0, 0.0, 0.0, 1.0],
      "metallicFactor": 1.0,
      "roughnessFactor": 0.01,
      "base_color_tex_index": 0,
      "normal_tex_index": 1
    }
  ],
  "world_normal": [0.0, 1.0, 0.0],
  "roughness": 0.35
}
```

Supported value formats:
- **Scalar:** `0.5` or `{"type": "scalar", "value": 0.5}`
- **Vector:** `[1.0, 2.0, 3.0]` or `{"type": "vec3", "value": [1.0, 2.0, 3.0]}`
- **Matrix:** `[[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]`
- **Integer:** `42` or `{"type": "int", "value": 42}`
- **Boolean:** `true` or `{"type": "bool", "value": true}`

## Real Texture Sampling

The debugger can load actual texture files from disk for `sample()` calls instead of returning constant grey.

**Requirements:** `pip install Pillow numpy`

**Via JSON input:**
```json
{
  "textures": ["textures/albedo.png", "textures/normal.png"]
}
```

**Via Python API:**
```python
from luxc.debug.values import load_image, LuxImage

img = load_image("textures/albedo.png", wrap="repeat")
# img.sample_bilinear(u, v) returns vec4 in [0,1]
```

Texture sampling features:
- Bilinear-filtered lookups (4-tap interpolation)
- Wrap modes: `repeat` (default) and `clamp`
- Supported formats: PNG, JPG, TGA, BMP (anything Pillow can open)
- `sample_bindless()` indexes into texture arrays for bindless shaders

## Heatmap Mode (Python API)

Run the shader across a grid of pixels and produce a PPM image visualizing output colors, NaN locations, performance hotspots, or variable magnitudes.

```python
from luxc.debug.heatmap import run_pixel_grid, write_ppm

source = open("examples/debug_playground.lux").read()

# Output color visualization
result = run_pixel_grid(source, "fragment", grid_size=(64, 64), mode="output")
write_ppm("color.ppm", result)

# NaN heatmap — red where NaN occurs, green where clean
result = run_pixel_grid(source, "fragment", grid_size=(32, 32), mode="nan", check_nan=True)
write_ppm("nan.ppm", result)

# Performance heatmap — blue (few statements) to red (many)
result = run_pixel_grid(source, "fragment", grid_size=(16, 16), mode="perf")
write_ppm("perf.ppm", result)

# Variable magnitude heatmap
result = run_pixel_grid(source, "fragment", grid_size=(32, 32),
                        mode="var:roughness", var_name="roughness")
write_ppm("rough.ppm", result)
```

**Modes:**

| Mode | Visualization |
|------|---------------|
| `output` | Shader output color (vec4 RGBA mapped to RGB) |
| `nan` | Red = NaN present, Green = clean |
| `perf` | Blue-to-red heatmap by statement count |
| `var:<name>` | Blue-to-red heatmap by variable magnitude |

**Output format:** P6 binary PPM (viewable in most image viewers, convertible with ImageMagick).

**Inspecting results:**
```python
for pixel in result.pixels:
    if pixel.nan_count > 0:
        print(f"NaN at ({pixel.x},{pixel.y}) uv=({pixel.u:.2f},{pixel.v:.2f})")
```

## Bindless Debugging

The debugger provides automatic defaults for bindless material and texture array patterns, matching the `BindlessMaterialData` struct layout.

**Automatic material defaults:**

When a shader declares a storage buffer with `BindlessMaterialData` structs, the debugger automatically provides a default material:

| Field | Default |
|-------|---------|
| `baseColorFactor` | `vec4(0.8, 0.8, 0.8, 1.0)` |
| `metallicFactor` | `0.0` (dielectric) |
| `roughnessFactor` | `0.5` |
| `ior` | `1.5` |
| `emissiveFactor` | `vec3(0.0)` |
| All `*_tex_index` fields | `-1` (no texture bound) |

**Texture array defaults:**

Bindless texture arrays are populated with neutral grey 1x1 images by default.

**List indexing for SSBOs:**

The interpreter supports list indexing for storage buffer arrays:
```
lux-debug> print materials[0].roughnessFactor
  0.500 (scalar)
lux-debug> print materials[1].baseColorFactor
  vec4(0.800, 0.800, 0.800, 1.000)
```

**Programmatic:**
```python
from luxc.debug.io import build_default_material
mat = build_default_material()
# Returns LuxStruct("BindlessMaterialData", {...})
```

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

**Interactive NaN hunting with break-on-NaN:**
```
lux-debug> break-nan
lux-debug> run
NaN detected at line 55: dir = vec3(NaN, NaN, NaN)
Stopped at line 56
lux-debug> eval diff
  vec3(0.000, 0.000, 0.000)
```

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

### Time-Travel Workflow

Use time-travel to compare variable states at different execution points without restarting:

```
lux-debug> break 100
lux-debug> run
Hit breakpoint at line 100
lux-debug> print specular
  vec3(0.003, 0.003, 0.003)
lux-debug> continue
Output: vec4(0.403, 0.381, 0.345, 1.000)
lux-debug> reverse-continue
[time-travel] Restored to step 23, line 100
lux-debug> print specular
  vec3(0.003, 0.003, 0.003)
lux-debug> goto 1
[time-travel] Restored to step 1, line 87
```

## Supported Language Features

The debugger supports the full Lux language except GPU-specific intrinsics:

- All arithmetic, comparison, and logical operators
- `let` bindings, `if`/`else`, `for`, `while`, `break`, `continue`
- User-defined functions with arguments and return values
- Vector/matrix constructors: `vec3(1.0)`, `mat4(1.0)`, `vec4(v3, 1.0)`
- Swizzle access: `.xyz`, `.rgb`, `.xy`, single component `.x`
- Index access: `v[0]`, `m[col][row]`, `list[i]` (for SSBO arrays)
- Struct field access
- 47 math builtins matching GLSL.std.450 semantics
- `debug_print()` and `assert()` statements
- Module-level constants
- Texture `sample()` with real image data (or mock grey fallback)
- `sample_bindless()` and `sample_bindless_lod()` for bindless texture arrays

### What's NOT Supported

- Ray tracing intrinsics (`traceRay`, etc.)
- Compute shader atomics and barriers
- `dFdx`/`dFdy` screen-space derivatives

## Architecture

```
luxc/debug/
    values.py        — Value types (LuxScalar, LuxVec, LuxMat, LuxInt, LuxBool, LuxStruct, LuxImage)
    environment.py   — Scoped variable storage with parent chain lookup
    builtins.py      — 47+ math builtins + texture-aware sample/sample_bindless
    interpreter.py   — Tree-walking AST evaluator (exec_stmt, eval_expr, list indexing)
    debugger.py      — Breakpoints (conditional), stepping, watches, break-on-NaN, time-travel
    expr_parser.py   — Debug expression parser (reuses full Lux parser)
    io.py            — Input loading (JSON, semantic defaults, textures, materials)
    heatmap.py       — Multi-pixel grid runner + PPM output
    cli.py           — Interactive REPL and batch mode orchestration
```

The debugger reuses the compiler's parser and type checker but skips codegen entirely. It runs the AST directly in Python, making it fast to iterate and independent of GPU drivers.

## Manual Testing

Run the automated P18.2 test suite:

```bash
# All checks (58 tests)
python tests/manual_test_p18_2.py --all

# Specific section
python tests/manual_test_p18_2.py --section a1    # Conditional breakpoints
python tests/manual_test_p18_2.py --section b     # Time-travel
python tests/manual_test_p18_2.py --section c     # Heatmap
python tests/manual_test_p18_2.py --section d     # Real textures
python tests/manual_test_p18_2.py --section cli   # CLI integration

# Interactive testing guide
python tests/manual_test_p18_2.py --guide
```
