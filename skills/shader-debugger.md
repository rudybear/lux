# Shader Debugger — CPU-Side AST Interpreter

## When to Apply
Use when diagnosing shader bugs that are hard to identify through static analysis:
- NaN/Inf producing black or flickering pixels
- Wrong colors or brightness issues
- Energy conservation violations
- Normal/tangent space confusion
- Suspicious intermediate values in complex BRDF chains

## The Debugger Tool

Lux includes a CPU-side AST interpreter (`--debug-run`) that executes shaders
without a GPU. It supports breakpoints, variable inspection, NaN detection, and
full variable tracing.

### Batch Mode (Recommended for AI)

Always start diagnosis with batch mode for machine-readable JSON output:

```bash
# Quick NaN check — first diagnostic step
luxc shader.lux --debug-run --stage fragment --batch --check-nan

# Full variable dump — trace every assignment
luxc shader.lux --debug-run --stage fragment --batch --dump-vars

# Break at a specific line and dump state
luxc shader.lux --debug-run --stage fragment --batch --break 42 --dump-at-break

# Multiple breakpoints
luxc shader.lux --debug-run --stage fragment --batch --break 42 --break 67 --dump-at-break

# Custom inputs
luxc shader.lux --debug-run --stage fragment --batch --check-nan --input inputs.json
```

### JSON Output Format

```json
{
  "status": "completed",
  "output": {"type": "vec4", "value": [0.523, 0.187, 0.112, 1.0]},
  "nan_detected": false,
  "statements_executed": 56,
  "variable_trace": [
    {"line": 12, "name": "albedo", "type": "vec3", "value": [0.8, 0.2, 0.1]},
    {"line": 15, "name": "roughness", "type": "scalar", "value": 0.35}
  ],
  "nan_events": [],
  "debug_prints": [],
  "assert_failures": []
}
```

### Custom Input JSON

Create an `inputs.json` file to override default values:

```json
{
  "normal": [0.0, 0.0, 1.0],
  "uv": [0.5, 0.5],
  "roughness": 0.3,
  "metallic": 1.0,
  "albedo": [0.9, 0.1, 0.1]
}
```

Default values: `position=(0,0,0)`, `normal=(0,0,1)`, `uv=(0.5,0.5)`,
`roughness=0.5`, `metallic=0.0`, `mat4=identity`, texture samples=`vec4(0.8,0.8,0.8,1.0)`.

## Diagnosis Patterns

### NaN Trace-Back

When `nan_detected: true`, check `nan_events` for the first occurrence:

1. Look at the `line` and `variable` in the first nan_event
2. Check the `operation` field (e.g., "let", "let(inf)")
3. Read the source at that line to identify the problematic expression
4. Common culprits: `sqrt(negative)`, `normalize(zero)`, `pow(0,0)`, `1/0`
5. Fix: add guards (`max(x, 0.0)`, length checks, epsilon clamps)

### Normal Flip Detection

Break at the BRDF evaluation line and check:
- `dot(N, L)` should be positive for front-facing surfaces
- If negative, normals might be flipped or in wrong space
- Check that `unpack_normal()` was applied to normal map samples

### Energy Audit

Use `--dump-vars` and check:
1. All BRDF lobe weights (diffuse + specular + coat + sheen) sum to <= 1.0
2. Final color components are in reasonable HDR range (0-10 for most scenes)
3. Fresnel terms are in [0,1] range
4. Albedo values are < 1.0 for non-emissive materials

### Wrong Color Diagnosis

1. Run `--batch --dump-vars` to get full trace
2. Find where color values diverge from expected
3. Check type coercions (int->float, swizzle order)
4. Verify texture sample returns (mock returns grey by default)

## Interactive Mode

For manual exploration:

```bash
luxc shader.lux --debug-run --stage fragment
```

Commands:
- `break <line>` / `b <line>` — set breakpoint
- `run` / `r` — run shader
- `step` / `s` — step one statement
- `next` / `n` — step over function calls
- `continue` / `c` — continue to next breakpoint
- `print <var>` / `p <var>` — show variable value
- `locals` / `l` — show all visible variables
- `source` / `src` — show current source context
- `output` — show current output value
- `quit` / `q` — exit

## Integration with --ai-critique

When running `--ai-critique`, the debugger batch mode should be the first
diagnostic step. Run `--batch --check-nan` to rule out NaN issues, then
`--batch --dump-vars` for full analysis if needed.

## Limitations

- Texture samples return mock data (grey) by default — override with `--input`
- No GPU-specific behavior (workgroup shared memory, barriers)
- Floating-point precision may differ slightly from GPU hardware
- Loop iteration limit: 10,000 (guards against infinite loops)
