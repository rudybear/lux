# Lux Playground — WebGPU

Browser-based rendering engine for Lux-compiled shaders via WebGPU. Same compiler pipeline as the desktop viewers — `.lux` source compiles to SPIR-V, transpiles to WGSL via naga, and loads at runtime from reflection JSON. No embedded shaders.

## Quick Start

```bash
# 1. Install naga-cli (SPIR-V -> WGSL transpiler)
cargo install naga-cli@28.0.0

# 2. Compile Lux examples to WGSL
python tools/compile_webgpu.py

# 3. Install and run
cd playground_web
npm install
npm run dev
```

Open http://localhost:5173 in Chrome 113+, Firefox 130+, or Safari 18+.

## Architecture

```
.lux source
  -> luxc compiler (--target wgsl --webgpu)
  -> SPIR-V -> naga-cli -> WGSL + reflection JSON
  -> browser loads .wgsl + .json at runtime
  -> ReflectedPipeline creates WebGPU bind groups from reflection
  -> RenderEngine renders with reflection-driven uniform updates
```

The WebGPU viewer uses the exact same shader compilation pipeline as the C++/Vulkan and Rust/ash viewers. The `ReflectedPipeline` TypeScript class is a direct port of the Python `reflected_pipeline.py` — it reads descriptor set layouts, vertex attributes, and push constant emulation metadata from the compiler's reflection JSON to build WebGPU pipeline layouts and bind groups automatically.

### Push Constant Emulation

WebGPU has no push constants. The `--webgpu` compiler flag converts push constant blocks to uniform buffers at descriptor set N+1. The reflection JSON includes a `push_emulated` section with set/binding/fields metadata, which the viewer reads to create and update the emulated uniform buffer each frame.

### Key Files

| File | Purpose |
|------|---------|
| `src/main.ts` | Entry point, scene loading, render loop |
| `src/gpu-context.ts` | WebGPU device init, canvas configuration |
| `src/reflected-pipeline.ts` | Reflection JSON -> WebGPU pipeline (bind groups, vertex state) |
| `src/render-engine.ts` | Per-frame rendering, reflection-driven uniform updates |
| `src/shader-loader.ts` | Fetch `.wgsl` + `.json` from server |
| `src/gltf-pbr.ts` | glTF PBR pipeline using compiler-produced shaders |
| `src/gltf-loader.ts` | GLB parser (meshes, materials, textures, tangents) |
| `src/ibl-loader.ts` | Pre-computed IBL cubemap loader (specular, irradiance, BRDF LUT) |
| `src/procedural-ibl.ts` | Fallback procedural IBL generation (CPU) |
| `src/camera.ts` | Orbit camera with mouse/touch input |
| `src/ui.ts` | Sidebar controls (material sliders, scene selector, drag-and-drop) |
| `src/fallback-shaders.ts` | Embedded fallback PBR sphere shader |

## Features

- **Compiler-driven pipeline**: All shaders loaded from `.wgsl` + `.json` — same source, same compiler, same reflection as Vulkan/Metal
- **glTF PBR rendering**: DamagedHelmet default scene, drag-and-drop `.glb` files
- **Image-based lighting**: Pre-computed Pisa IBL cubemaps (specular + irradiance + BRDF LUT), matching the desktop C++ viewer
- **Interactive controls**: Metallic, roughness, exposure, light direction sliders
- **Orbit camera**: Mouse drag (rotate), scroll (zoom), middle-click (pan)
- **Reflection-driven uniforms**: Engine reads field names/offsets from reflection JSON — no hardcoded uniform layouts

## IBL Assets

The viewer loads pre-computed IBL from `public/assets/ibl/`:

| File | Format | Size | Description |
|------|--------|------|-------------|
| `specular.bin` | float16 RGBA | 256x256, 9 mips, 6 faces | GGX-filtered specular cubemap |
| `irradiance.bin` | float16 RGBA | 32x32, 6 faces | Cosine-weighted irradiance cubemap |
| `brdf_lut.bin` | float16 RG | 512x512 | Split-sum BRDF integration LUT |

These are the same assets used by the C++/Vulkan desktop viewer, generated from the Pisa HDR environment.

## Building Shaders

```bash
# Compile all compatible examples to WGSL
python tools/compile_webgpu.py

# Compile a specific file
python tools/compile_webgpu.py examples/gltf_pbr.lux
```

Output goes to `playground_web/public/shaders/`. Each shader produces:
- `{name}.{stage}.wgsl` — WGSL shader source
- `{name}.{stage}.json` — Reflection metadata (descriptor sets, vertex layout, push constants)

## Browser Requirements

- Chrome 113+ / Edge 113+
- Firefox 130+
- Safari 18+

WebGPU must be enabled. The viewer shows a fallback message if WebGPU is unavailable.
