# Lux Optimization Features

Comprehensive reference for the Lux shader compiler optimization pipeline.

## Pipeline Overview

```
Source (.lux)
    |
    v
  parse_lux()              -- Parser: tree_builder.py
    |
    v
  _resolve_imports()       -- Module system: compiler.py
    |
    v
  expand_surfaces()        -- Surface/pipeline expansion: surface_expander.py
    |
    v
  autodiff_expand()        -- Automatic differentiation: forward_diff.py
    |
    v
  type_check()             -- Type analysis: type_checker.py
    |
    v
  check_nan_warnings()     -- NaN/Inf static analysis (--warn-nan): nan_checker.py
    |
    v
  strip_debug_stmts()      -- Debug strip (release mode): compiler.py
    |
    v
  constant_fold()          -- Constant folding + strength reduction: const_fold.py
    |
    v
  inline_functions()       -- AST-level function inlining (release only): inline.py
    |
    v
  dead_code_elim()         -- Dead code elimination: dead_code.py
    |
    v
  cse()                    -- Common subexpression elimination: cse.py
    |
    v
  assign_layouts()         -- Descriptor set/binding assignment: layout_assigner.py
    |
    v
  generate_spirv()         -- SPIR-V codegen + mem2reg + const vector hoisting: spirv_builder.py
    |
    v
  assemble_and_validate()  -- spirv-as + spirv-val: spv_assembler.py
    |
    v
  run_spirv_opt()          -- Optional spirv-opt passes: spv_assembler.py
    |
    v
  generate_reflection()    -- Reflection JSON + performance hints: reflection.py
```

All AST-level optimizations run between type checking and layout assignment. SPIR-V-level optimizations run after assembly. Reflection metadata is emitted last, incorporating cost analysis from the final SPIR-V.

---

## 1. Constant Folding & Strength Reduction

**Source**: `luxc/optimization/const_fold.py`

Walks all function bodies and folds constant expressions at compile time.

### Constant Folding

| Category | Example | Result |
|----------|---------|--------|
| Arithmetic on literals | `1.0 + 2.0` | `3.0` |
| Const variable inlining | `const PI` reference | Literal `3.14159...` |
| Builtin calls on literals | `sin(0.0)` | `0.0` |
| Dead branch elimination | `if (true) { A } else { B }` | `A` |
| Ternary folding | `true ? a : b` | `a` |
| Double negation | `-(-x)` | `x` |
| Boolean double negation | `!(!x)` | `x` |

**Supported builtin functions** (folded when all arguments are literal):

- **1-arg**: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `exp`, `exp2`, `log`, `log2`, `sqrt`, `abs`, `floor`, `ceil`, `fract`, `sign`
- **2-arg**: `min`, `max`, `pow`, `step`
- **3-arg**: `mix`, `clamp`, `smoothstep`

### Algebraic Identity Simplification

| Pattern | Simplification |
|---------|---------------|
| `x * 1.0` | `x` |
| `x * 0.0` | `0.0` |
| `x / 1.0` | `x` |
| `x + 0.0` | `x` |
| `x - 0.0` | `x` |
| `length(v) * length(v)` | `dot(v, v)` |

### Strength Reduction

Reduces expensive operations to cheaper equivalents:

| Pattern | Replacement | Speedup |
|---------|-------------|---------|
| `pow(x, 0.0)` | `1.0` | Eliminates call |
| `pow(x, 1.0)` | `x` | Eliminates call |
| `pow(x, 2.0)` | `x * x` | 1 MUL vs transcendental |
| `pow(x, 3.0)` | `x * x * x` | 2 MULs vs transcendental |
| `pow(x, 0.5)` | `sqrt(x)` | HW sqrt vs general pow |
| `pow(x, -0.5)` | `inversesqrt(x)` | HW rsqrt vs general pow |

---

## 2. Dead Code Elimination

**Source**: `luxc/optimization/dead_code.py`

Iteratively removes unreferenced let bindings and assignments to unused variables.

### Algorithm

1. Collect all variable names that are **read** (referenced) in the function body
2. Stage output variables are always considered "used" (never eliminated)
3. A `let` binding is dead if its name is not in the used set
4. An assignment is dead if its target base variable is not in the used set
5. **Side-effect preservation**: if a dead binding's RHS contains side-effecting calls, the binding is converted to an `ExprStmt` (expression statement) rather than dropped entirely
6. Iterate until no more changes (up to 20 iterations for safety)

### Side-Effect Calls (Never Dropped)

Texture sampling, ray tracing, barriers, atomics, geometry output, and debug:

`sample`, `sample_lod`, `sample_bindless`, `sample_bindless_lod`, `sample_compare`, `sample_array`, `sample_grad`, `trace_ray`, `barrier`, `memoryBarrier`, `memoryBarrierShared`, `atomicAdd`, `atomicMin`, `atomicMax`, `atomicAnd`, `atomicOr`, `atomicXor`, `atomicExchange`, `atomicCompareExchange`, `emit_vertex`, `end_primitive`, `emit_mesh_tasks`, `debug_printf`, `imageStore`

### Scope Handling

- Recursively processes `if`, `for`, `while`, and `debug` block bodies
- Assignment target chains (`a[idx].x = ...`) are walked to collect index reads
- Protected variables: stage outputs in `main()` are never eliminated

---

## 3. Common Subexpression Elimination (CSE)

**Source**: `luxc/optimization/cse.py`

Identifies duplicate expression trees within function bodies and replaces redundant computations with synthetic let bindings.

### Algorithm

1. Walk each statement collecting all candidate sub-expressions with structural hashes and setter callbacks
2. Group by hash; confirm structural equality to guard against collisions
3. For each true duplicate set (2+ occurrences):
   - Create a synthetic `let _cse_N: <type> = <first_occurrence>`
   - Replace **every** occurrence (including the first) with a `VarRef` to the synthetic variable
   - Insert the let before the statement containing the first occurrence
4. Process largest expressions first (greedy, by node count) to avoid replacing sub-expressions of already-replaced expressions

### Constraints

| Constraint | Reason |
|------------|--------|
| Never CSE across control-flow boundaries | If/for/while bodies get independent CSE passes |
| Never CSE side-effecting calls | Texture sampling, atomics, barriers, etc. |
| Never CSE trivial leaves | `VarRef`, `NumberLit`, `BoolLit` not worth hoisting |
| Never CSE `SwizzleAccess` | Very cheap in SPIR-V (`OpCompositeExtract`); hoisting causes naga/wgpu issues |
| Never CSE struct-typed SSBO access | StorageBuffer layout decorations invalid for Function scope |
| Deep-copy before per-stage CSE | Pipeline expansion may share expression objects across stages |
| Monotonic counter across all stages | Guarantees unique `_cse_N` names across the entire module |

### Structural Hashing

Two-phase approach for performance:
- **Fast hash**: `_expr_hash()` computes a structural hash for each expression tree
- **Equality check**: `_exprs_equal()` confirms structural equality when hashes match

---

## 4. AST-Level Function Inlining

**Source**: `luxc/optimization/inline.py`

Inlines user-defined function calls at the AST level before CSE runs. This allows CSE to catch duplicate expressions that result from inlining the same function multiple times (e.g., `dot(N, Lo)` inlined 4 times from a PBR BRDF).

### How It Works

1. Build a lookup of all user-defined functions (module-level + per-stage)
2. Deep-copy function bodies before processing (pipeline expansion may share expression objects across stages)
3. For each `CallExpr` to a user-defined function:
   - Create unique let bindings for each parameter (`_inlN_paramName`)
   - Rename local variables with unique prefix to avoid conflicts
   - Replace the call with the function body's statements + return expression
4. Process non-main stage functions first (they may call each other), then main

### Inlineability Criteria

A function is inlined if it has a simple structure:
- A sequence of `LetStmt` nodes (optional `IfStmt`) followed by a `ReturnStmt`
- Recursive calls are not inlined (current function is excluded)

### Debug Mode

**Skipped entirely** in debug mode (`-g`). Debug builds use codegen-time inlining to preserve the call structure for debugger step-through in RenderDoc and similar tools.

### Impact

On a PBR fragment shader: enables CSE to deduplicate ~15-25 expressions from inlined light calculations that were previously invisible to CSE.

---

## 5. Mem2Reg — SSA Value Forwarding

**Source**: `luxc/codegen/spirv_builder.py` (codegen-time optimization)

Eliminates redundant `OpVariable`/`OpStore`/`OpLoad` sequences for single-assignment variables by tracking SSA values directly during SPIR-V code generation.

### Motivation

Without mem2reg, every `let` binding generates:
```
%ptr = OpVariable %_ptr_Function_float Function  ; allocate stack slot
       OpStore %ptr %value                        ; write value
%val = OpLoad %float %ptr                         ; read it back
```

With mem2reg, single-assignment variables skip the stack entirely — the computed value is used directly wherever the variable is referenced.

### Algorithm

1. **Pre-scan** (`_scan_mutable_vars`): Walk the function body to find variables targeted by `AssignStmt` or used as `ForStmt` loop variables — these are "mutable" and must keep `OpVariable`
2. **Forward-reference detection** (`_scan_forward_refs`): CSE may produce let bindings where one references another defined later in the statement list; these are forced to `OpVariable` to avoid use-before-definition
3. **SSA path**: For non-mutable variables, store the computed SSA ID in `_ssa_values[name]` instead of emitting `OpStore`. On read, return the SSA ID directly instead of emitting `OpLoad`
4. **Mutable path**: Variables in `_mutable_vars` use the existing `OpVariable`/`OpStore`/`OpLoad` path

### Debug Mode

**Skipped entirely** in debug mode. All variables get `OpVariable` for debugger variable inspection (RenderDoc, NSight). The debug flag causes `_scan_mutable_vars` to be skipped, treating all variables as mutable.

### Impact

On a PBR fragment shader: ~125 `OpVariable` → ~20 (only loop vars, accumulators), ~100 `OpLoad` eliminated. Single largest contributor to instruction count reduction.

---

## 6. Constant Vector Hoisting

**Source**: `luxc/codegen/spirv_builder.py` (`_gen_constructor`)

Replaces runtime `OpCompositeConstruct` with compile-time `OpConstantComposite` when all vector constructor arguments are `NumberLit` constants.

### Before

```
%c1 = OpConstant %float 1.0
%c2 = OpConstant %float 0.0
%v  = OpCompositeConstruct %v3float %c1 %c2 %c2   ; runtime instruction
```

### After

```
%v = OpConstantComposite %v3float %c1 %c2 %c2      ; compile-time, deduplicated
```

### Handles

- All vector types: `vec2`, `vec3`, `vec4`, `ivec*`, `uvec*`
- Splat constructors: `vec3(1.0)` → single constant replicated 3 times
- Automatic deduplication: `vec3(1.0)` appearing 6 times → 1 `OpConstantComposite`

### Debug Mode

**Applied in both debug and release** — constant composites are still inspectable in debuggers and don't eliminate any variable information.

### Impact

On a PBR fragment shader: 11 runtime `OpCompositeConstruct` → 2 deduplicated compile-time constants, saving ~45 instructions.

---

## 7. Cost Estimator & VGPR Analysis

**Source**: `luxc/analysis/cost_estimator.py`

Parses SPIR-V assembly text and produces cost metrics for estimating shader performance.

### Metrics Produced

| Metric | Description |
|--------|-------------|
| `instruction_count` | Total SPIR-V instructions |
| `alu_ops` | Arithmetic/logic operations (FAdd, FMul, Dot, MatrixTimesVector, etc.) |
| `texture_samples` | Texture fetch/sample operations |
| `branches` | Conditional branches and switches |
| `function_calls` | `OpFunctionCall` count |
| `temp_ids` | Total unique `%id` references (register pressure proxy) |
| `vgpr_estimate` | VGPR pressure: "low" (<30 ids/fn), "medium" (30-60), "high" (>60) |

### Op-Code Categories

**ALU**: `OpFAdd`, `OpFSub`, `OpFMul`, `OpFDiv`, `OpFMod`, `OpFNegate`, `OpDot`, `OpVectorTimesScalar`, `OpVectorTimesMatrix`, `OpMatrixTimesVector`, `OpMatrixTimesMatrix`, `OpIAdd`, `OpISub`, `OpIMul`, `OpSDiv`, `OpUDiv`

**Texture**: `OpImageSampleImplicitLod`, `OpImageSampleExplicitLod`, `OpImageSampleDrefImplicitLod`, `OpImageSampleDrefExplicitLod`, `OpImageFetch`

**Branch**: `OpBranchConditional`, `OpSwitch`

### Output Format

```
fragment: 312 instr, 89 ALU, 4 tex, VGPR: medium
```

---

## 8. SPIR-V Optimization Profiles

**Source**: `luxc/codegen/spv_assembler.py`

Two spirv-opt profiles are available, applied after SPIR-V assembly:

### Standard Optimization (`-O` / `--optimize`)

Runs `spirv-opt -O` which applies the default optimization recipe. Focuses on code size and general optimizations.

```bash
luxc shader.lux -O
```

### Performance Optimization (`--perf`)

Runs a curated set of performance-oriented passes:

| Pass | Purpose |
|------|---------|
| `--merge-blocks` | Merge basic blocks to reduce branch overhead |
| `--eliminate-dead-branches` | Remove unreachable branches |
| `--eliminate-dead-code-aggressive` | Aggressive dead code removal |
| `--private-to-local` | Promote private variables to local scope |
| `--scalar-replacement=100` | Break aggregates into scalars (threshold: 100) |
| `--ssa-rewrite` | Convert to SSA form for better optimization |
| `--ccp` | Conditional constant propagation |
| `--simplify-instructions` | Peephole simplifications |
| `--redundancy-elimination` | Remove redundant computations |
| `--combine-access-chains` | Merge chained access operations |
| `--vector-dce` | Dead code elimination on vector components |
| `--if-conversion` | Convert branches to select instructions |
| `--loop-unroll` | Unroll loops for throughput |
| `--strength-reduction` | Replace expensive ops with cheaper equivalents |

```bash
luxc shader.lux --perf
```

Both profiles operate in-place on the `.spv` binary. If spirv-opt fails, the original file is preserved. If spirv-opt is not found on PATH, a warning is emitted and compilation proceeds without optimization.

---

## 9. CLI Flags

**Source**: `luxc/cli.py`

### Optimization Flags

| Flag | Description |
|------|-------------|
| `-O` / `--optimize` | Run `spirv-opt -O` on generated SPIR-V binaries |
| `--perf` | Run performance-oriented spirv-opt passes (loop unroll, strength reduction) |
| `--analyze` | Print per-stage instruction cost analysis after compilation |
| `--warn-nan` | Enable static analysis warnings for potential NaN/Inf operations |

### Debug & Inspection Flags

| Flag | Description |
|------|-------------|
| `--dump-ast` | Dump the AST as JSON and exit (before codegen) |
| `--emit-asm` | Write `.spvasm` text assembly files alongside `.spv` |
| `-g` / `--debug` | Emit `OpLine`/`OpSource` debug info in SPIR-V |
| `--no-validate` | Skip `spirv-val` validation |
| `--no-reflection` | Skip `.lux.json` reflection metadata emission |

### Compilation Flags

| Flag | Description |
|------|-------------|
| `-o` / `--output-dir` | Output directory (default: same as input file) |
| `--pipeline NAME` | Compile only the named pipeline |
| `--features FEAT,...` | Enable compile-time features (comma-separated) |
| `--all-permutations` | Compile all 2^N feature permutations |
| `--list-features` | List available features and exit |
| `--define KEY=VALUE` | Define compile-time integer constant |
| `--bindless` | Emit bindless descriptor uber-shaders |
| `--watch` | Watch source files for changes and recompile |
| `--watch-poll MS` | Polling interval for watch mode (default: 500ms) |

### AI & Transpiler Flags

| Flag | Description |
|------|-------------|
| `--transpile` | Transpile GLSL input to Lux |
| `--ai DESCRIPTION` | Generate shader from natural language description |
| `--ai-from-image PATH` | Generate surface material from an image |
| `--ai-modify INSTRUCTION` | Modify existing material with AI |
| `--ai-batch DESCRIPTION` | Generate batch of materials |
| `--ai-critique FILE` | AI review of a `.lux` file |

### Example: Full Optimization Pipeline

```bash
# Compile with all optimizations and analysis
luxc shader.lux --perf --analyze --emit-asm

# Compile with NaN warnings and standard optimization
luxc shader.lux -O --warn-nan

# Watch mode with performance optimization
luxc shader.lux --perf --watch
```

---

## 10. Reflection & Performance Hints

**Source**: `luxc/codegen/reflection.py`

Every compiled stage produces a `.lux.json` sidecar file with full reflection metadata consumed by Python/wgpu, C++/Vulkan, and Rust/ash runtimes.

### Reflection Schema

```json
{
  "version": 1,
  "source": "shader.lux",
  "stage": "fragment",
  "execution_model": "Fragment",
  "inputs": [{"name": "uv", "type": "vec2", "location": 0}],
  "outputs": [{"name": "color", "type": "vec4", "location": 0}],
  "descriptor_sets": {
    "0": [
      {
        "binding": 0,
        "type": "uniform_buffer",
        "name": "Uniforms",
        "fields": [
          {"name": "mvp", "type": "mat4", "offset": 0, "size": 64}
        ],
        "size": 64,
        "stage_flags": ["fragment"]
      }
    ]
  },
  "push_constants": [],
  "vertex_attributes": [],
  "performance_hints": {
    "instruction_count": 312,
    "alu_ops": 89,
    "texture_samples": 4,
    "branches": 2,
    "function_calls": 1,
    "vgpr_estimate": "medium"
  }
}
```

### Performance Hints

The `performance_hints` section is populated by `add_performance_hints()`, which runs the cost estimator on the final SPIR-V assembly. This enables runtime systems to:

- **Budget GPU time**: know instruction count before dispatch
- **Select LOD**: choose shader complexity based on VGPR pressure
- **Profile**: compare instruction counts across optimization levels
- **Alert**: flag shaders exceeding ALU or texture budgets

### Descriptor Types Reflected

`uniform_buffer`, `sampler`, `sampled_image`, `sampled_cube_image`, `storage_image`, `storage_buffer`, `acceleration_structure`, `bindless_combined_image_sampler_array`

### Additional Metadata

- **Vertex attributes**: format, offset, stride for vertex shaders
- **Compute workgroup size**: from `--define workgroup_size_x/y/z`
- **Mesh shader output**: max vertices, max primitives, topology
- **Ray tracing**: payloads, hit attributes, callable data
- **Shared memory**: per-variable sizes, total bytes
- **Specialization constants**: names, types, spec IDs, defaults
- **Feature flags**: active/inactive state per feature

---

## 8. NaN/Inf Static Analysis

**Source**: `luxc/analysis/nan_checker.py`

Enabled with `--warn-nan`. Walks all functions after type checking and emits warnings for operations that may produce NaN or Inf at runtime:

- Division where the divisor could be zero
- `sqrt()` / `log()` of potentially negative values
- `asin()` / `acos()` of values outside [-1, 1]
- `pow()` with negative base and non-integer exponent

Warnings are emitted via Python's `warnings` module so they don't block compilation.

---

## 9. Benchmark Suite

**Source**: `tools/benchmark_suite.py`

Compiles curated shaders, extracts instruction counts from reflection JSON, and verifies they don't exceed established baselines.

```bash
# Run benchmarks against baselines
python tools/benchmark_suite.py

# Update baselines with current results
python tools/benchmark_suite.py --update-baselines

# JSON output for CI integration
python tools/benchmark_suite.py --json
```

Each benchmark defines a shader path, target stage, and soft instruction ceiling. Exceeding the ceiling is a warning (not a failure), enabling regression tracking without breaking CI.

---

## 10. Shader Bottleneck Analyzer

**Source**: `tools/shader_analysis.py`

Detects performance anti-patterns in compiled shaders using configurable thresholds:

| Metric | Threshold | Target |
|--------|-----------|--------|
| Fragment texture samples | >3 | Texture-bound detection |
| Fragment branches | >5 | Divergence warning |
| Fragment ALU (60fps) | >200 | ALU budget for 60fps |
| Fragment ALU (VR) | >100 | Tighter VR budget |
| Vertex ALU | >50 | Vertex processing budget |
| Float divisions | >4 | High `fdiv` count |
| Texture size | >4 MB | Large texture warning |

Produces `Finding` objects with severity (`info`, `warning`, `critical`), category, and metric values.

---

## 11. A/B Performance Experiment Runner

**Source**: `tools/perf_experiment.py`

Automates optimization comparison workflows:

```bash
python tools/perf_experiment.py \
    --shader examples/gltf_pbr_layered.lux \
    --scene sphere \
    --variants "baseline:" "optimized:-O" "aggressive:--perf" \
    --output perf_results/
```

For each variant:
1. Compiles the shader with the specified flags
2. Extracts instruction counts from reflection JSON
3. Captures rendered frame (if RenderDoc available)
4. Compares image quality between variants
5. Generates JSON report with per-variant metrics

---

## 12. Image Quality Metrics

**Source**: `tools/quality_metrics.py`

Compares baseline and candidate images using PSNR and SSIM. Requires only Pillow and numpy.

```bash
# Basic comparison
python tools/quality_metrics.py baseline.png candidate.png

# With difference heatmap
python tools/quality_metrics.py baseline.png candidate.png --diff diff_heatmap.png

# With pass/fail thresholds
python tools/quality_metrics.py baseline.png candidate.png \
    --psnr-threshold 35 --ssim-threshold 0.97
```

| Metric | Typical Pass Threshold | Description |
|--------|----------------------|-------------|
| PSNR | >35 dB | Peak signal-to-noise ratio (higher = more similar) |
| SSIM | >0.97 | Structural similarity index (1.0 = identical) |

The `--diff` flag writes a per-pixel difference heatmap PNG for visual inspection of where differences occur.

---

## 13. CPU-Side Rendering Analysis

**Source**: `tools/cpu_analysis.py`

Analyzes draw call patterns from RenderDoc capture JSON to identify CPU-side bottlenecks:

| Check | Threshold | Meaning |
|-------|-----------|---------|
| Draws per pipeline bind | <2.0 | Excessive pipeline switching |
| Draws per descriptor bind | <3.0 | Descriptor set churn |
| Buffer padding ratio | >15% | Wasted buffer memory |
| Redundant bind ratio | >10% | Unnecessary re-binds |

```bash
python tools/cpu_analysis.py metrics.json
python tools/cpu_analysis.py --format table metrics.json
```

---

## Optimization Workflow

### Development Cycle

```bash
# 1. Compile with analysis to see baseline costs
luxc shader.lux --analyze --emit-asm

# 2. Apply standard optimization
luxc shader.lux -O --analyze

# 3. Apply aggressive optimization
luxc shader.lux --perf --analyze

# 4. Compare instruction counts from reflection JSON
python tools/perf_experiment.py \
    --shader shader.lux \
    --variants "baseline:" "standard:-O" "perf:--perf"

# 5. Verify visual quality preservation
python tools/quality_metrics.py reference.png optimized.png \
    --psnr-threshold 40 --ssim-threshold 0.99
```

### CI Integration

```bash
# Run benchmark suite to catch regressions
python tools/benchmark_suite.py --json > benchmark_results.json

# Check shader quality
python tools/shader_analysis.py reflection.json

# Full test suite
pytest tests/  # 891 tests
```

### What the Optimizer Does (AST Level)

Before any SPIR-V is generated, the AST passes transform the program:

1. **Constant folding** evaluates `sin(0.0)` to `0.0`, replaces `pow(x, 2.0)` with `x * x`
2. **Dead code elimination** removes unused `let` bindings and dead assignments
3. **CSE** finds `dot(N, L)` computed twice and replaces both with a single `_cse_0` variable

These passes compose: constant folding may expose dead code, and dead code elimination may expose CSE opportunities.

### What spirv-opt Does (Binary Level)

After SPIR-V assembly, the optional spirv-opt passes perform lower-level transformations:

- **SSA rewrite**: canonical SSA form enables further optimization
- **Scalar replacement**: breaks structs into individual scalars
- **Loop unrolling**: eliminates loop overhead for small fixed-count loops
- **If-conversion**: replaces branches with `OpSelect` where profitable
- **Redundancy elimination**: CSE at the SPIR-V level
- **Strength reduction**: replaces expensive SPIR-V ops with cheaper equivalents
