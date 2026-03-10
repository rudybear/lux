# Screenshot Tests & Rendering Engines

The `playground/` directory contains screenshot tests that compile, render, and validate each example:

```bash
# Run individual tests
python playground/test_hello_triangle.py
python playground/test_pbr_surface.py
python playground/test_brdf_gallery.py
python playground/test_colorspace.py
python playground/test_texture_utils.py
python playground/test_autodiff.py
python playground/test_advanced_materials.py
python playground/test_rt_pathtracer.py
python playground/test_gltf_pbr.py
python playground/test_lighting.py
python playground/test_ibl.py
python playground/test_math_builtins.py
```

Each test compiles the shader, renders to a 512x512 PNG, and validates pixel-level properties (coverage, color distribution, spatial variation).

## Rendering Engines

Four rendering backends share the same compiled shaders and reflection JSON:

**Python (wgpu) — headless only:**
```bash
python -m playground.engine --scene assets/DamagedHelmet.glb --pipeline shadercache/gltf_pbr --output render.png
python -m playground.engine --scene sphere --pipeline shadercache/pbr_surface --output sphere.png
python -m playground.engine --scene assets/DamagedHelmet.glb --pipeline shadercache/gltf_pbr --ibl pisa --output render_ibl.png
```

**C++ (Vulkan/GLFW) — headless and interactive:**
```bash
# Headless render to PNG
playground_cpp/build/Release/lux-playground.exe --scene assets/DamagedHelmet.glb --pipeline shadercache/gltf_pbr_rt --ibl pisa --output render.png

# Interactive viewer (orbit camera: drag to rotate, scroll to zoom, ESC to exit)
playground_cpp/build/Release/lux-playground.exe --scene assets/DamagedHelmet.glb --pipeline shadercache/gltf_pbr_rt --ibl pisa --interactive
```

**C++ (Metal/GLFW — macOS only) — headless and interactive:**
```bash
# Build
cd playground_cpp && cmake -B build-metal -DLUX_METAL=ON && cmake --build build-metal

# Headless render to PNG (auto-detects pipeline from scene)
./build-metal/lux-playground-metal --scene assets/DamagedHelmet.glb --output render.png

# Interactive viewer (orbit camera: drag to rotate, scroll to zoom, ESC to exit)
./build-metal/lux-playground-metal --scene assets/DamagedHelmet.glb --interactive
```

The Metal backend transpiles SPIR-V shaders to MSL via SPIRV-Cross at runtime, supporting raster and mesh shader (Metal 3) pipelines. Same shader permutation system, glTF loading, IBL, and multi-material support as the Vulkan engines. Requires Metal-cpp headers vendored in `playground_cpp/deps/metal-cpp/` (download from [Apple](https://developer.apple.com/metal/cpp/)).

**Rust (ash/winit) — headless and interactive:**
```bash
# Headless render to PNG
playground_rust/target/release/lux-playground.exe --scene assets/DamagedHelmet.glb --pipeline shadercache/gltf_pbr_rt --ibl pisa --output render.png

# Interactive viewer (orbit camera: drag to rotate, scroll to zoom, ESC to exit)
playground_rust/target/release/lux-playground.exe --scene assets/DamagedHelmet.glb --pipeline shadercache/gltf_pbr_rt --ibl pisa --interactive
```

**Gaussian splatting (all engines):**
```bash
# Compile splat shaders + generate test data
python -m luxc examples/gaussian_splat.lux
python -m tools.generate_test_splats tests/assets/test_splats.glb

# C++ (Vulkan) — interactive
playground_cpp/build/Release/lux-playground.exe --scene tests/assets/test_splats.glb --pipeline examples/gaussian_splat --interactive

# Rust (ash) — interactive
cd playground_rust && cargo run --release -- --scene ../tests/assets/test_splats.glb --pipeline ../examples/gaussian_splat --interactive

# Python (CPU rasterizer) — interactive orbit viewer
python -m playground.render_harness --splat-scene tests/assets/test_splats.glb --interactive

# Python (CPU rasterizer) — headless PNG
python -m playground.render_harness --splat-scene tests/assets/test_splats.glb -o splat_render.png
```

The splat pipeline uses CPU-side depth sorting and instanced quad drawing. Input data is loaded from glTF files with the `KHR_gaussian_splatting` extension. The `tools/glb_to_ply.py` converter exports `.glb` splat data to standard `.ply` format for validation in external viewers (SuperSplat, gsplat.tech).

**Batch/shell scripts for quick launch:**
```bash
run_interactive_cpp.bat     # Interactive C++ raster viewer
run_interactive_rust.bat    # Interactive Rust raster viewer
run_interactive_cpp_rt.bat  # Interactive C++ RT viewer
run_interactive_rust_rt.bat # Interactive Rust RT viewer
run_interactive_metal.sh    # Interactive Metal raster viewer (macOS)
run_mesh_interactive_metal.sh  # Interactive Metal mesh shader viewer (macOS)
compile_gltf_rt.bat         # Compile + render glTF RT (headless, both engines)
compile_gltf_forward.bat    # Compile + render glTF raster (all 3 engines)
compile_gltf_all.bat        # Compile + render all pipeline variants
run_splat_cpp.bat            # Compile + render Gaussian splats (C++)
run_splat_rust.bat           # Compile + render Gaussian splats (Rust)
run_splat_python.bat         # Compile + render Gaussian splats (Python)
compile_mesh.bat            # Compile mesh shader pipeline
run_mesh_headless_cpp.bat   # Render mesh shaders headless (C++)
run_mesh_headless_rust.bat  # Render mesh shaders headless (Rust)
run_mesh_headless_metal.sh  # Render mesh shaders headless (Metal, macOS)
run_mesh_interactive_cpp.bat  # Interactive mesh shader viewer (C++)
run_mesh_interactive_rust.bat # Interactive mesh shader viewer (Rust)
```

**Common CLI flags** (C++, Metal, and Rust engines):
| Flag | Description |
|------|-------------|
| `--scene <PATH>` | Scene source: `.glb`/`.gltf` file, `sphere`, `triangle`, `fullscreen` |
| `--pipeline <BASE>` | Shader base path (auto-resolved from scene if omitted) |
| `--ibl <NAME>` | IBL environment name from `assets/ibl/` (default: auto-detect `pisa`) |
| `--interactive` | Open a window with orbit camera |
| `--width <N>` | Output width (default: 512, interactive: 1024) |
| `--height <N>` | Output height (default: 512, interactive: 768) |
| `--output <PATH>` | Output PNG path for headless mode |
| `--validation` | Force Vulkan validation layers ON in release builds |

## IBL Preprocessing

Convert HDR environment maps to pre-filtered cubemaps for image-based lighting:

```bash
# Download and preprocess a Khronos sample environment
python -m playground.preprocess_ibl --download neutral

# Process a custom HDR panorama
python -m playground.preprocess_ibl my_environment.hdr
```

Output: specular cubemap (6 faces x 5 mips), irradiance cubemap, and BRDF integration LUT.
