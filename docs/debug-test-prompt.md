# Lux Shader Debugger — Test Prompt for Claude

Copy this entire prompt into a new Claude Code session to test the debugger. The session should demonstrate all debug features, find bugs, verify math, and explore edge cases.

---

## Prompt

You are testing the Lux shader CPU debugger — a tool that lets you step through shader code, inspect every variable, and detect NaN/Inf issues without a GPU. The debugger interprets the shader AST directly in Python with the same math semantics as SPIR-V.

Your task is to thoroughly exercise the debugger on the test shader and write a report of your findings.

### Background

The Lux shader compiler (`luxc`) has a `--debug-run` mode that launches a CPU-side AST interpreter instead of compiling to SPIR-V. It supports:

- **Interactive REPL** (`--debug-run --stage fragment`): gdb-style commands — `run`, `step`, `next`, `continue`, `break <line>`, `print <var>`, `locals`, `source`, `output`, `quit`
- **Batch mode** (`--batch`): JSON output for automation
- **NaN detection** (`--check-nan`): pinpoints exact line and variable where NaN/Inf originates
- **Variable tracing** (`--dump-vars`): records every variable assignment with value and source line
- **Breakpoints** (`--break <line> --dump-at-break`): dumps full variable scope at specified lines
- **Pixel debugging** (`--pixel X,Y`): auto-computes uv, position, normal from screen coordinates
- **Inline overrides** (`--set VAR=VALUE`): quick variable overrides without a JSON file
- **Custom inputs** (`--input <file.json>`): override input values from a JSON file

### Step 1: Read the test shader

Read `examples/debug_playground.lux` and understand what it does. It's a PBR fragment shader with:
- Fresnel-Schlick, GGX NDF, Smith geometry
- ACES tonemapping
- An intentional NaN trap in `compute_reflection()`
- 8 labeled stages in the main function

### Step 2: Run the default diagnostic

```bash
python -m luxc examples/debug_playground.lux --debug-run --stage fragment --batch --check-nan
```

Analyze the output:
- What is the final output color? Is it reasonable for a PBR shader?
- Were any NaN/Inf values detected? Where do they originate? Trace back through the code to explain the root cause.

### Step 3: Full variable trace

```bash
python -m luxc examples/debug_playground.lux --debug-run --stage fragment --batch --dump-vars
```

Walk through the variable trace and verify the PBR math step by step:
1. Are the dot products (n_dot_l, n_dot_v, n_dot_h) reasonable for the default geometry?
2. What is the GGX distribution value? Manually verify: `D = a2 / (PI * denom^2)` where `a2 = (roughness^2)^2` and `denom = n_dot_h^2 * (a2 - 1) + 1`
3. What is the Smith geometry term? Is g1=1.0 correct for head-on viewing (n_dot_v=1)?
4. Is the Fresnel term near f0=0.04 at normal incidence? Verify with Schlick's formula.
5. Is the specular-to-diffuse ratio physically plausible for a dielectric?
6. Does the ACES tonemap produce reasonable output (S-curve, boosts darks, compresses highlights)?

### Step 4: Simulate different pixels

Create input files and test edge cases:

**Test A: Metallic surface at grazing angle**
```bash
python -m luxc examples/debug_playground.lux --debug-run --stage fragment --input examples/debug_grazing_angle.json --batch --check-nan --dump-vars
```

Questions:
- How much does the Fresnel term increase vs the head-on case?
- What happens to the geometry term at n_dot_v near zero?
- Does the rim lighting term dominate? Why?
- What color is the output and does it make physical sense?

**Test B: Zero roughness**
```bash
python -m luxc examples/debug_playground.lux --debug-run --stage fragment --input examples/debug_zero_roughness.json --batch --check-nan --dump-vars
```

Questions:
- Does roughness=0 cause any NaN or Inf?
- What is the GGX distribution value when a2=0?
- Is the result physically correct for a perfect mirror? If not, what's wrong?
- What would you recommend as a fix?

**Test C: Pixel-specific debugging**

Use `--pixel` and `--set` to test specific screen locations without creating JSON files:

```bash
# Top-left corner — steep viewing angle
python -m luxc examples/debug_playground.lux --debug-run --stage fragment --pixel 0,0 --resolution 800x600 --batch --check-nan

# Center pixel with mirror-smooth metal
python -m luxc examples/debug_playground.lux --debug-run --stage fragment --pixel 400,300 --resolution 800x600 --set roughness=0.01 --set metallic=1.0 --batch --dump-vars

# Edge pixel — the hemisphere normal will be nearly perpendicular to view
python -m luxc examples/debug_playground.lux --debug-run --stage fragment --pixel 10,300 --resolution 800x600 --batch --check-nan
```

Questions:
- How does the output change across different screen positions?
- Does the edge pixel produce any NaN from the steep normal angle?
- How does the hemisphere normal derivation work?

**Test D: Create your own edge case**

Create a scenario YOU think might break the shader using `--set`. Ideas:
- `--set "normal=0,0,-1"` — normal pointing away from camera (backface)
- `--set roughness=0.0 --set metallic=1.0` — perfect metal mirror
- `--set exposure=100` — extreme exposure
- `--set "world_position=10000,10000,10000"` — far from camera

Run it and report what happens.

### Step 5: Breakpoint inspection

```bash
python -m luxc examples/debug_playground.lux --debug-run --stage fragment --batch --break 100 --break 119 --break 135 --dump-at-break
```

At each breakpoint:
- How many variables are in scope?
- What is the state of the BRDF computation?
- Can you identify the exact moment the specular and diffuse terms are combined?

### Step 6: Interactive session (if supported)

If you can interact with stdin, try:
```bash
python -m luxc examples/debug_playground.lux --debug-run --stage fragment
```

Walk through these commands:
```
break 100
run
locals
print f0
print diffuse_color
step
step
print n_dot_l
print n_dot_v
continue
output
```

### Step 7: Write your report

Create a file `docs/debug-test-report.md` with:

1. **Executive Summary**: One paragraph on what the debugger found
2. **NaN Analysis**: Root cause, propagation chain, recommended fix
3. **PBR Math Verification**: Table of key values with manual verification
4. **Edge Case Results**: What each custom input scenario revealed
5. **Debugger Feature Assessment**: Rate each feature (NaN detection, variable tracing, breakpoints, custom inputs, batch mode) on usefulness from 1-5 stars
6. **Bugs or Limitations Found**: Anything that didn't work as expected
7. **Feature Requests**: What would make the debugger even better?

### Important Notes

- Actually run every command — don't fake output
- The shader has an INTENTIONAL NaN bug. Finding it is part of the test.
- Default inputs: world_normal=(0,0,1), world_position=(0,0,0), roughness=0.5, metallic=0.0, exposure=1.0
- Texture samples always return vec4(0.8, 0.8, 0.8, 1.0) (mock data)
- The debugger supports 44+ math builtins matching GLSL.std.450

---

## Expected Outcomes

A successful test session should demonstrate:
- NaN detected at line 55 (`normalize(vec3(0))` in `compute_reflection`)
- Output color approximately (0.40, 0.38, 0.35) with default inputs
- Fresnel increasing from ~0.04 (normal) to ~0.73 (grazing angle)
- GGX D=0 with roughness=0 (physically wrong, needs min clamp)
- At least one creative edge case test
- A clear report documenting all findings
