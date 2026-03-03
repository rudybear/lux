# Usage

### Compile a shader

```bash
python -m luxc examples/hello_triangle.lux
# Wrote examples/hello_triangle.vert.spv
# Wrote examples/hello_triangle.frag.spv
```

### Compile a declarative surface pipeline

```bash
python -m luxc examples/pbr_surface.lux
# Wrote examples/pbr_surface.vert.spv
# Wrote examples/pbr_surface.frag.spv
```

### Compile a ray tracing pipeline

```bash
python -m luxc examples/rt_pathtracer.lux
# Wrote examples/rt_pathtracer.rgen.spv
# Wrote examples/rt_pathtracer.rchit.spv
# Wrote examples/rt_pathtracer.rmiss.spv
```

### Compile a mesh shader pipeline

```bash
python -m luxc examples/gltf_pbr_layered.lux --pipeline GltfMesh --features has_emission --define max_vertices=64 --define max_primitives=124 --define workgroup_size=32
# Wrote shadercache/gltf_pbr_layered+emission.mesh.spv
# Wrote shadercache/gltf_pbr_layered+emission.frag.spv
```

### Compile a compute shader

```bash
# 1D workgroup (data-parallel SSBO operations)
python -m luxc examples/compute_saxpy.lux --define workgroup_size=256
# Wrote examples/compute_saxpy.comp.spv

# 2D workgroup (image processing)
python -m luxc examples/compute_mandelbrot.lux --define workgroup_size_x=16 --define workgroup_size_y=16
# Wrote examples/compute_mandelbrot.comp.spv
```

### Watch mode (hot reload)

```bash
# Recompile on save — watches file + all imports
python -m luxc examples/pbr_surface.lux --watch
# Watching examples/pbr_surface.lux (+ 3 imports)...
# [12:34:56] Recompiled pbr_surface (2 stages) in 0.04s

# Custom poll interval (default 500ms)
python -m luxc examples/pbr_surface.lux --watch --watch-poll 200
```

### Optimize SPIR-V output

```bash
# Run spirv-opt for smaller binaries
python -m luxc examples/hello_triangle.lux -O
# Wrote examples/hello_triangle.vert.spv (optimized)
# Wrote examples/hello_triangle.frag.spv (optimized)
```

### Auto-type precision analysis

```bash
# Analyze variable precision (report only, no code changes)
python -m luxc examples/pbr_surface.lux --auto-type=report
# === Fragment Stage: Auto-Type Precision Report ===
# Variable              Type    Decision  Confidence  Reason
# ----------------------------------------------------------------
# roughness             scalar  fp16      1.00        both dynamic and static ranges fit fp16
# world_position        vec3    fp32      1.00        heuristic: name heuristic: position value
# ----------------------------------------------------------------
# Summary: 12/18 variables safe for fp16 (66.7%)

# Emit RelaxedPrecision decorations for fp16-safe variables
python -m luxc examples/pbr_surface.lux --auto-type=relaxed
# Wrote examples/pbr_surface.vert.spv
# Wrote examples/pbr_surface.frag.spv  (with OpDecorate RelaxedPrecision)
```

### Compile a specific pipeline

```bash
python -m luxc examples/gltf_pbr_layered.lux --pipeline GltfForward
# Wrote examples/gltf_pbr_layered.vert.spv
# Wrote examples/gltf_pbr_layered.frag.spv

python -m luxc examples/gltf_pbr_layered.lux --pipeline GltfRT
# Wrote examples/gltf_pbr_layered.rgen.spv
# Wrote examples/gltf_pbr_layered.rchit.spv
# Wrote examples/gltf_pbr_layered.rmiss.spv
```

### Compile with features

```bash
# List available features
python -m luxc examples/gltf_pbr_layered.lux --list-features
# Available features:
#   has_normal_map
#   has_clearcoat
#   has_sheen
#   has_emission

# Compile with specific features enabled
python -m luxc examples/gltf_pbr_layered.lux --pipeline GltfForward \
    --features has_normal_map,has_emission
# Wrote gltf_pbr_layered+emission+normal_map.vert.spv
# Wrote gltf_pbr_layered+emission+normal_map.frag.spv

# Compile base variant (no features)
python -m luxc examples/gltf_pbr_layered.lux --pipeline GltfForward
# Wrote gltf_pbr_layered.vert.spv
# Wrote gltf_pbr_layered.frag.spv

# Compile all 2^N permutations
python -m luxc examples/gltf_pbr_layered.lux --pipeline GltfForward \
    --all-permutations
# Compiling 16 permutations...
# Wrote gltf_pbr_layered.manifest.json
```

### Transpile GLSL to Lux

```bash
python -m luxc shader.glsl --transpile
# Transpiled: shader.lux
```

### Generate a shader with AI

```bash
# Using default provider (Anthropic Claude)
python -m luxc --ai "frosted glass with subsurface scattering" -o generated.lux

# Using local Ollama
python -m luxc --ai "glossy copper metal" --ai-provider ollama --ai-model llama3.2

# Using OpenAI
python -m luxc --ai "brushed steel" --ai-provider openai --ai-model gpt-4o

# Using any OpenAI-compatible endpoint
python -m luxc --ai "marble" --ai-provider openai --ai-base-url http://my-server:8080/v1

# Interactive setup wizard (saves to ~/.luxc/config.toml)
python -m luxc --ai-setup
```

### Options

```
python -m luxc input.lux [options]

Options:
  --emit-asm          Write .spvasm text files alongside .spv
  --dump-ast          Dump the AST as JSON and exit
  --no-validate       Skip spirv-val validation
  -o OUTPUT_DIR       Output directory (default: same as input)
  --pipeline NAME     Compile only the named pipeline
  --features FEAT,...   Enable compile-time features (comma-separated)
  --all-permutations    Compile all 2^N feature permutations
  --list-features       List available features and exit
  --define KEY=VALUE  Set compile-time parameter (e.g., --define max_vertices=64)
  --no-reflection     Skip .lux.json reflection metadata
  -g, --debug         Enable debug instrumentation (OpLine, OpName on locals, debug_print, assert, @[debug] blocks)
  --rich-debug        Emit NonSemantic.Shader.DebugInfo.100 for RenderDoc step-through (implies -g)
  --debug-print       Preserve debug_print/assert without full debug mode
  --assert-kill       Demote fragment invocation on assert failure (OpDemoteToHelperInvocation)
  --warn-nan          Static analysis warnings for risky float operations
  --debug-run         Launch CPU-side shader debugger (interactive REPL, no GPU required)
  --stage STAGE       Stage to debug with --debug-run (default: fragment)
  --batch             Batch mode for --debug-run (JSON output to stdout)
  --dump-vars         Trace all variable assignments (--batch)
  --check-nan         Detect NaN/Inf with source location (--batch)
  --break LINE        Set breakpoint at source line (repeatable, --batch)
  --dump-at-break     Dump all variables at breakpoint hits (--batch)
  --input FILE        Load custom input values from JSON for --debug-run
  --pixel X,Y         Debug a specific pixel (auto-computes uv, position, normal)
  --resolution WxH    Screen resolution for --pixel (default: 1920x1080)
  --set VAR=VALUE     Override a single input inline (repeatable)
  --export-inputs FILE  Export default input values to JSON
  -O, --optimize      Run spirv-opt on output binaries
  --perf              Run performance-oriented spirv-opt passes (loop unroll, strength reduction)
  --analyze           Print per-stage instruction cost analysis after compilation
  --auto-type MODE    Auto precision optimization (report=analyze only, relaxed=emit RelaxedPrecision)
  --watch             Watch input file for changes and recompile
  --watch-poll MS     Polling interval in milliseconds (default: 500)
  --bindless          Emit bindless descriptor uber-shaders
  --transpile         Transpile GLSL input to Lux
  --ai DESCRIPTION    Generate shader from natural language
  --ai-setup          Run interactive AI provider setup wizard
  --ai-provider NAME  Override AI provider (anthropic, openai, gemini, ollama, lm-studio)
  --ai-model MODEL    Model for AI generation (default: from config)
  --ai-base-url URL   Override AI provider base URL (OpenAI-compatible endpoints)
  --ai-no-verify      Skip compilation verification of AI output
  --ai-retries N      Max retry attempts for AI generation (default: 2)
  --ai-from-image IMG Generate surface material from a reference image
  --ai-modify TEXT    Modify existing material (e.g., --ai-modify "add weathering")
  --ai-batch DESC     Generate batch of materials (e.g., --ai-batch "medieval tavern")
  --ai-batch-count N  Number of materials in batch (default: AI decides)
  --ai-from-video VID Generate animated shader from video
  --ai-match-reference IMG  Iteratively match a reference image
  --ai-match-iterations N   Max refinement iterations for reference matching (default: 5)
  --ai-critique FILE  AI review of a .lux file
  --ai-skills SKILL,... Load specific skills into the AI prompt
  --ai-list-skills    List available AI skills and exit
  --version           Show version
```
