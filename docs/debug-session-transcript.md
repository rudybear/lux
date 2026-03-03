# Lux Shader Debugger -- Session Transcript

A step-by-step debugging session analyzing `examples/debug_playground.lux`, a PBR
fragment shader with intentional edge cases, using the Lux CPU shader debugger.

---

## Session 1: Initial Scan -- Default Inputs

**Goal:** Run the shader with default inputs and see what comes out.

```
python -m luxc examples/debug_playground.lux --debug-run --stage fragment --batch
```

**Output:**
```json
{
  "status": "completed",
  "statements_executed": 53,
  "nan_detected": true,
  "output": {
    "type": "vec4",
    "value": [0.4025, 0.3805, 0.3455, 1.0]
  }
}
```

**Analysis:**

The shader ran 53 statements and produced a final color of approximately
`(0.40, 0.38, 0.35, 1.0)` -- a warm mid-gray with a slight orange tint. This is
a reasonable result for a PBR shader rendering a light gray surface (albedo 0.8)
lit by a warm directional light `(3.0, 2.8, 2.5)`, viewed head-on with default
roughness 0.5 and metallic 0.0.

However, the debugger immediately flagged `nan_detected: true`, even though the
final output is clean. This tells us NaN exists somewhere in the execution trace
but did not propagate into the final color output. Something to investigate.

---

## Session 2: NaN Detection -- Find the Bugs

**Goal:** Use `--check-nan` to pinpoint exactly where NaN values originate.

```
python -m luxc examples/debug_playground.lux --debug-run --stage fragment --batch --check-nan
```

**NaN Events:**
```
Event 1:  line 55, variable "dir"      -- normalize(diff) produces [NaN, NaN, NaN]
Event 2:  line 143, variable "bad_reflect" -- propagated from compute_reflection()
```

**Root Cause Analysis:**

The NaN originates on **line 55** inside `compute_reflection()`:

```lux
let diff: vec3 = incident - surface_normal;   // line 54
let dir: vec3 = normalize(diff);               // line 55 -- NaN!
```

The call site is on line 143:
```lux
let same_dir: vec3 = normalize(vec3(0.0, 0.0, 1.0));
let bad_reflect: vec3 = compute_reflection(same_dir, same_dir);
```

Here `incident == surface_normal`, so `diff = (0, 0, 0)`. The `normalize()`
function computes `v / length(v)`, and `length(vec3(0)) = 0`. Division by zero
produces `NaN` in all three components.

This is an intentional bug in the shader for demonstration purposes. The NaN is
stored in `bad_reflect` but is never used in the final color output, which is why
the final pixel looks correct. In a real shader, if `bad_reflect` were mixed into
the lighting calculation, the entire pixel would go black or produce visual
artifacts.

**Fix:** Guard against zero-length vectors before normalizing:
```lux
let len: scalar = length(diff);
let dir: vec3 = if len > 0.0001 { normalize(diff) } else { surface_normal };
```

---

## Session 3: Full Variable Trace -- Follow the Math

**Goal:** Dump every intermediate value and verify the PBR calculations are correct.

```
python -m luxc examples/debug_playground.lux --debug-run --stage fragment --batch --dump-vars
```

### Setup (lines 87-108)

| Variable | Value | Notes |
|----------|-------|-------|
| `albedo` | `(0.8, 0.8, 0.8)` | Default texture sample -- neutral gray |
| `n` | `(0, 0, 1)` | Normal pointing toward camera |
| `v` | `(0, 0, 1)` | View direction (camera at z=5, surface at origin) |
| `n_dot_v` | `1.0` | Head-on viewing angle |
| `light_dir` | `(0.667, 0.667, 0.333)` | Normalized `(1,1,0.5)` |
| `n_dot_l` | `0.333` | Light hits surface at ~70 degrees |
| `h` | `(0.408, 0.408, 0.817)` | Half-vector between view and light |
| `n_dot_h` | `0.816` | -- |
| `v_dot_h` | `0.816` | -- |

### Material (lines 95-97)

| Variable | Value | Notes |
|----------|-------|-------|
| `f0` | `(0.04, 0.04, 0.04)` | Dielectric F0 (metallic=0, so no albedo mix) |
| `diffuse_color` | `(0.8, 0.8, 0.8)` | Full albedo used for diffuse (1 - metallic = 1) |

**Verification:** For a non-metallic surface, `f0` should be 0.04 (standard
dielectric reflectance). Correct. `diffuse_color` should equal `albedo * (1 - metallic)
= 0.8 * 1.0 = 0.8`. Correct.

### BRDF (lines 111-122)

| Variable | Value | Notes |
|----------|-------|-------|
| `a` (GGX) | `0.25` | `roughness^2 = 0.5^2` |
| `a2` | `0.0625` | `a^2 = 0.25^2` |
| `d` (NDF) | `0.1415` | GGX distribution at this half angle |
| `k` (Smith) | `0.28125` | `((roughness+1)^2) / 8` |
| `g1` | `1.0` | Smith G1 for view (n_dot_v=1 -> perfect geometry term) |
| `g2` | `0.64` | Smith G1 for light |
| `g` (total) | `0.64` | `g1 * g2` |
| `f` (Fresnel) | `(0.0402, 0.0402, 0.0402)` | Near-normal -> nearly base F0=0.04 |
| `specular` | `(0.00273, 0.00273, 0.00273)` | Small specular contribution |
| `diffuse` | `(0.2444, 0.2444, 0.2444)` | Dominant contribution |
| `direct` | `(0.2471, 0.2307, 0.2060)` | Warm tint from light color |

**Verification -- Diffuse:** The diffuse BRDF is `kd * albedo / PI` where `kd = (1-F)*(1-metallic)`.
With `F ~ 0.04`, `kd ~ 0.96`, so `diffuse = 0.96 * 0.8 / 3.14159 = 0.2444`. Matches.

**Verification -- Specular:** For roughness=0.5 with a non-metallic surface viewed
head-on, the specular highlight should be subtle. The value 0.00273 (about 1% of
diffuse) is physically reasonable -- dielectrics reflect very little at normal
incidence.

**Verification -- Fresnel:** Schlick's approximation at normal incidence gives
`F = f0 + (1-f0) * (1 - cos_theta)^5`. With `v_dot_h = 0.816`:
`F = 0.04 + 0.96 * (1 - 0.816)^5 = 0.04 + 0.96 * 0.000208 = 0.0402`. Matches.

### Rim, Ambient, and Final Composite (lines 127-137)

| Variable | Value | Notes |
|----------|-------|-------|
| `rim_factor` | `0.0` | `(1 - n_dot_v)^4 = (1-1)^4 = 0` -- no rim at head-on |
| `rim` | `(0, 0, 0)` | No rim contribution |
| `ambient` | `(0.024, 0.024, 0.024)` | `0.03 * 0.8` |
| `hdr_color` | `(0.271, 0.255, 0.230)` | Sum of direct + rim + ambient |
| `exposed` | same as hdr_color | exposure=1.0, so no change |
| `ldr_color` | `(0.403, 0.381, 0.345)` | After ACES tonemapping |

**Verification -- Tonemapping:** The ACES curve is S-shaped. Input ~0.27 mapping
to output ~0.40 is in the expected range (the curve boosts darks and compresses
highlights). The warm-to-cool gradient across RGB channels persists correctly
through the tonemap.

**Conclusion:** All PBR math checks out. The lighting pipeline is physically
plausible and numerically stable for this default configuration.

---

## Session 4: Grazing Angle -- Metallic Surface, Near-Mirror

**Goal:** Test a metallic surface at a steep grazing angle. Does the Fresnel
effect increase? Is specular dominant?

**Input file** (`examples/debug_grazing_angle.json`):
```json
{
    "world_normal": [0.0, 1.0, 0.0],
    "world_position": [10.0, 0.0, 0.0],
    "roughness": 0.1,
    "metallic": 0.9,
    "exposure": 2.0
}
```

This places the camera at `(0, 0, 5)` looking at a point far to the right
`(10, 0, 0)`, creating a nearly perpendicular view angle relative to the
upward-facing normal.

```
python -m luxc examples/debug_playground.lux --debug-run --stage fragment --batch \
  --check-nan --dump-vars --input examples/debug_grazing_angle.json
```

**Output color:** `(0.375, 0.494, 0.581, 1.0)` -- a steel blue.

### Key Variables

| Variable | Value | Notes |
|----------|-------|-------|
| `v` | `(-0.894, 0.0, 0.447)` | View from the side -- steep angle |
| `n_dot_v` | `0.001` | Near-zero! Clamped by `max(..., 0.001)` |
| `f0` | `(0.724, 0.724, 0.724)` | `mix(0.04, 0.8, 0.9)` -- metallic dominates |
| `diffuse_color` | `(0.08, 0.08, 0.08)` | `albedo * (1-0.9) = 0.8 * 0.1` -- almost no diffuse |
| `f` (Fresnel) | `(0.731, 0.731, 0.731)` | Strong reflectance |
| `d` (NDF) | `8.9e-5` | Extremely tight specular lobe (roughness=0.1) |
| `g1` (geometry) | `0.00657` | Nearly zero -- extreme self-shadowing at grazing |
| `specular` | `(0.000144, 0.000144, 0.000144)` | Small due to geometry term |
| `diffuse` | `(0.000686, 0.000686, 0.000686)` | Negligible |
| `rim_factor` | `0.996` | Nearly maximum! `(1-0.001)^4 = 0.996` |
| `rim` | `(0.0996, 0.1494, 0.1992)` | Dominant contribution |
| `hdr_color` | `(0.125, 0.175, 0.225)` | Rim-dominated |
| `exposed` | `(0.251, 0.350, 0.449)` | Doubled by exposure=2.0 |
| `ldr_color` | `(0.375, 0.494, 0.581)` | Steel blue after tonemapping |

### Analysis

**Does the Fresnel effect increase at grazing angles?** Yes. The Fresnel term
jumped from 0.040 (head-on in Session 3) to 0.731 (grazing angle here). This is
the expected behavior -- Schlick's approximation approaches 1.0 at extreme grazing
angles. With `v_dot_h = 0.526`, the formula gives:
`F = 0.724 + 0.276 * (1-0.526)^5 = 0.724 + 0.276 * 0.024 = 0.731`. Correct.

**Is specular dominant for metallic=0.9?** Surprisingly, no -- not here. Even
though the material is metallic (high F0, suppressed diffuse), the geometry term
`g1` collapses to 0.007 at this extreme grazing angle. The Smith geometry function
aggressively attenuates when `n_dot_v` is near zero. This is physically correct:
at extreme grazing angles, surface micro-facets shadow each other.

**What dominates instead?** The rim lighting term. With `rim_factor = 0.996`, the
rim color `(0.1, 0.15, 0.2)` drives the final appearance, giving the blue tint.
In a production shader, this rim term would typically be Fresnel-weighted or
replaced by an environment map reflection.

**NaN status:** The same `normalize(vec3(0))` bug appears at lines 55/143. No new
NaN issues from the grazing angle inputs -- the `max(dot, 0.001)` guard on
`n_dot_v` prevents division by zero in the BRDF denominator.

---

## Session 5: Breakpoint Inspection -- BRDF State Snapshot

**Goal:** Set breakpoints at the specular calculation (line 119) and the final
composite (line 135) to inspect the full variable environment at those moments.

```
python -m luxc examples/debug_playground.lux --debug-run --stage fragment --batch \
  --break 119 --break 135 --dump-at-break
```

### Breakpoint 1: Line 119 (Specular calculation)

At this point, the BRDF terms have been computed but not yet combined:

```
Variables in scope: 24
Key state:
  roughness = 0.5,  metallic = 0.0
  n_dot_l   = 0.333
  n_dot_v   = 1.0
  n_dot_h   = 0.816
  v_dot_h   = 0.816
  d (NDF)   = 0.1415
  g (Smith)  = 0.64
  f (Fresnel) = (0.0402, 0.0402, 0.0402)
  numerator  = (0.00364, 0.00364, 0.00364)
  denominator = 1.333
```

The `denominator` equals `4 * n_dot_v * n_dot_l = 4 * 1.0 * 0.333 = 1.333`.
This is well away from zero, so the specular division is safe. The specular result
will be `0.00364 / (1.333 + 0.0001) = 0.00273`.

Note that `color` is `(0, 0, 0, 0)` at this point -- the output has not been
written yet.

### Breakpoint 2: Line 135 (Final composite)

All lighting terms are now computed:

```
Additional variables since breakpoint 1:
  specular  = (0.00273, 0.00273, 0.00273)
  kd        = (0.960, 0.960, 0.960)
  diffuse   = (0.244, 0.244, 0.244)
  direct    = (0.247, 0.231, 0.206)
  rim       = (0, 0, 0)
  ambient   = (0.024, 0.024, 0.024)
```

The `--dump-at-break` feature lets you inspect the exact state of every variable
at any source line, without needing to add print statements or recompile. This is
particularly useful for verifying intermediate BRDF values that would normally
be invisible.

---

## Session 6: Edge Case -- Zero Roughness

**Goal:** Test roughness=0.0. In the GGX distribution, `a2 = roughness^4 = 0`,
which makes the numerator zero. Does this cause division issues?

**Input file** (`examples/debug_zero_roughness.json`):
```json
{
    "world_normal": [0.0, 0.0, 1.0],
    "world_position": [0.0, 0.0, 0.0],
    "roughness": 0.0,
    "metallic": 0.0,
    "exposure": 1.0
}
```

```
python -m luxc examples/debug_playground.lux --debug-run --stage fragment --batch \
  --check-nan --dump-vars --input examples/debug_zero_roughness.json
```

**Output color:** `(0.399, 0.377, 0.342, 1.0)` -- very similar to Session 1.

### GGX Distribution with roughness=0

| Variable | Value |
|----------|-------|
| `a` | `0.0` (`roughness^2`) |
| `a2` | `0.0` (`a^2`) |
| `denom` | `0.333` (`n_dot_h^2 * (0 - 1) + 1 = 1 - 0.667 = 0.333`) |
| `d` (NDF) | `0.0` (`a2 / (PI * denom^2) = 0 / 0.349 = 0`) |

**Analysis:** When `roughness = 0`, the GGX numerator `a2` is zero, so the
distribution function returns `D = 0.0` regardless of angle. The denominator
`denom = n_dot_h^2 * (a2 - 1) + 1` simplifies to `1 - n_dot_h^2 * 1`, which
equals `1 - 0.667 = 0.333` -- safely non-zero.

This means: **no division-by-zero occurs, but the result is physically wrong.**
A perfect mirror (roughness=0) should concentrate all energy into a single
reflection direction (a Dirac delta). Instead, the GGX function returns zero,
killing all specular contribution:

```
specular = (0, 0, 0)    -- zero specular when roughness=0
diffuse  = (0.244, ...)  -- all contribution from diffuse
```

The output looks almost identical to the default case because the specular was
already small (0.00273) at roughness=0.5. The real problem is conceptual: a
mirror-smooth surface should have an intense, perfectly sharp specular highlight
when the view/light/normal align, not zero specular everywhere.

**The fix for production code:** Clamp roughness to a minimum value:
```lux
let clamped_roughness: scalar = max(roughness, 0.045);
```
This is standard practice in game engines (Unreal uses 0.045 as the minimum).

**NaN status:** No new NaN issues. The only NaN remains the intentional
`normalize(vec3(0))` bug on line 55.

---

## Summary of Findings

### Bug Found: normalize(vec3(0)) Produces NaN

- **Where:** Line 55 in `compute_reflection()`, called from line 143
- **Cause:** When `incident == surface_normal`, their difference is `vec3(0, 0, 0)`.
  `normalize(zero_vector)` computes `0 / 0 = NaN`.
- **Impact:** The NaN is stored in `bad_reflect` but not used in the final output.
  In a real shader, this would silently corrupt the pixel and potentially spread
  to neighboring pixels through post-processing.
- **Fix:** Check vector length before normalizing, or use a safe-normalize helper
  that returns a default direction for zero-length inputs.

### PBR Math Verification Results

All BRDF calculations were verified against manual computation:

- **Fresnel (Schlick):** Correct. Returns 0.04 at normal incidence, increases to
  0.73 at grazing angles. Behaves as expected for both dielectric (f0=0.04) and
  metallic (f0=0.72) configurations.
- **GGX Normal Distribution:** Correct for roughness > 0. Returns 0 at roughness=0
  (see edge case below).
- **Smith Geometry Function:** Correct. Properly attenuates at grazing angles
  (g1 drops to 0.007 when n_dot_v=0.001).
- **Energy Conservation:** The `kd = (1-F)*(1-metallic)` term correctly reduces
  diffuse when specular increases, maintaining energy conservation.
- **ACES Tonemapping:** Maps HDR values to [0,1] range correctly. Input 0.27 maps
  to output 0.40, consistent with the ACES curve.

### Edge Cases Discovered

1. **roughness=0 kills all specular:** The GGX distribution returns `D=0` when
   roughness is zero because `a2 = 0^4 = 0`. No NaN or Inf occurs, but the
   result is physically incorrect for a perfect mirror. Clamp roughness to a
   minimum (0.045) in production.

2. **Grazing angle geometry collapse:** At extreme grazing angles (`n_dot_v ~= 0`),
   the Smith geometry term collapses nearly to zero, even for metallic surfaces.
   The `max(dot, 0.001)` guard prevents division by zero but creates a very narrow
   valid range where the BRDF underestimates reflection. This is a known
   limitation of the Schlick-GGX model.

3. **Rim lighting dominance:** At grazing angles, the simple `pow(1-NdotV, 4)` rim
   term overwhelms the BRDF. This is artistically useful but not physically based.
   Consider weighting the rim by the Fresnel term for more realistic results.

### Debug Features That Made This Analysis Possible

The Lux CPU debugger provided several features that would be difficult or
impossible with traditional GPU shader debugging:

- **`--check-nan`**: Instantly pinpointed the exact line and variable where NaN
  originated, including the call chain. On a GPU, NaN typically manifests as a
  black pixel with no indication of which calculation went wrong.

- **`--dump-vars`**: Provided a complete trace of every intermediate value in
  execution order. This made it trivial to verify each step of the PBR math
  against known-correct formulas. No need for the "change one uniform, recompile,
  screenshot, compare" cycle.

- **`--break <line> --dump-at-break`**: Allowed inspection of the full variable
  environment at critical points in the shader (before specular division, before
  final composite). This is the shader equivalent of a debugger watchpoint.

- **`--input <json>`**: Custom inputs made it possible to test specific edge cases
  (grazing angles, zero roughness, metallic surfaces) without modifying the shader
  or setting up a rendering scene. Each test was a single command-line invocation.

- **Batch mode (`--batch`)**: All sessions ran non-interactively with structured
  JSON output, making the results easy to parse, compare, and document. This
  enables automated regression testing of shader behavior.

These tools transform shader debugging from an opaque, visual-only process into a
transparent, quantitative workflow. Every number is visible, every edge case is
testable, and every bug is traceable to its exact source line.
