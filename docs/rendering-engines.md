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

The splat pipeline uses CPU-side depth sorting and instanced quad drawing. Input data is loaded from glTF files with the ratified `KHR_gaussian_splatting` extension: attribute semantics (`KHR_gaussian_splatting:ROTATION`/`SCALE`/`OPACITY`/`SH_DEGREE_N_COEF_M`) live directly in `primitive.attributes`, `mode` is `POINTS` (0), and `extensions.KHR_gaussian_splatting` carries `kernel`/`colorSpace`/`sortingMethod`/`projection`. `SCALE` and `OPACITY` are linear per the ratified spec; every loader (C++ `gltf_loader.cpp`, Rust `gltf_loader.rs`, the Python `playground/render_harness.py` harness, and `tools/glb_to_ply.py`) auto-detects log-space `SCALE` (any negative value — the encoding used by the currently-published Khronos conformance test assets, which predate an April-2026 spec editorial pass) versus ratified linear `SCALE` (all non-negative) and converts as needed before feeding the shared exp()/sigmoid()-based compute shader. All loaders also still accept the pre-ratification draft layout (ROTATION/SCALE/OPACITY nested, unprefixed, under `extensions.KHR_gaussian_splatting.attributes`) as a fallback — `tools/ply_to_gltf.py --legacy-layout` emits that layout for testing it. `tools/glb_to_ply.py` converts either layout's `.glb` splat data to standard `.ply` format for validation in external viewers (SuperSplat, gsplat.tech).

**Deferred rendering (all engines):**
```bash
# Compile deferred shaders (G-buffer vertex/fragment + lighting vertex/fragment)
python -m luxc examples/deferred_basic.lux

# C++ (Vulkan) — interactive deferred renderer
run_deferred_cpp.bat

# Rust (ash) — interactive deferred renderer
run_deferred_rust.bat

# Python (wgpu) — interactive deferred renderer
run_deferred_python.bat
```

The deferred pipeline uses the same `surface` + `lighting` declarations as forward rendering — `mode: deferred` auto-generates a G-buffer geometry pass (3 MRT: albedo+metallic, octahedron normal+roughness, emission+occlusion) and a fullscreen lighting pass with IBL + multi-light support.

**Batch/shell scripts for quick launch:**
```bash
run_interactive_cpp.bat     # Interactive C++ raster viewer
run_interactive_rust.bat    # Interactive Rust raster viewer
run_interactive_cpp_rt.bat  # Interactive C++ RT viewer
run_interactive_rust_rt.bat # Interactive Rust RT viewer
run_interactive_metal.sh    # Interactive Metal raster viewer (macOS)
run_mesh_interactive_metal.sh  # Interactive Metal mesh shader viewer (macOS)
run_deferred_cpp.bat         # Deferred renderer (C++)
run_deferred_rust.bat        # Deferred renderer (Rust)
run_deferred_python.bat      # Deferred renderer (Python)
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
| `--headless` | Offscreen render only (default) |
| `--validation` | Force Vulkan validation layers ON in release builds |
| `--time <SECONDS>` / `--frame <N>` | Dynamic (`motion: keyframes`) splats: pick the animation time for headless renders (frame -> time via `extras.fps` when present, else keyframe times) |
| `--jitter <JX> <JY>` | Sub-pixel jitter in pixels, applied to the splat projection matrix only (motion vectors always use the unjittered matrices) -- see the "DLSS Input-Contract Outputs" section of `docs/language-reference.md` |
| `--camera-json <FILE>` | Drive the splat camera from an OpenCV-convention `{viewmat_cv, K, width, height}` JSON file -- see `docs/lux-4d-spec.md` section 4 |
| `--camera-json-prev <FILE>` | Seed the motion-vector "previous frame" camera explicitly from a second camera JSON, instead of the default prev==curr (mv=0) on frame 1 (testing/tooling) |
| `--time-prev <SECONDS>` / `--frame-prev <N>` | Dynamic (`motion: keyframes`) splats only: evaluate the morph animation a *second* time, at this earlier time, into `splat_prev_pos` -- so a single headless render's motion vectors reflect real actor motion (`mv = projection(pos(time), cam) - projection(pos(time_prev), cam_prev)`), not just camera motion. Without this flag, `splat_prev_pos` defaults to the current frame's own (post-morph) position, i.e. `mv` is camera-motion-only (see `--camera-json-prev` above) -- this is still what sequential playback wants (`prev` = last rendered frame), it's specifically a single-shot-validation convenience. Combine with `--camera-json-prev` for a real actor-plus-camera delta between two arbitrary frames |
| `--output-aux <PREFIX>` | Splats compiled with `motion_vectors`/`expected_depth`: write `<PREFIX>_color.png`, `_color.npy` (float32 `[H,W,4]`, un-premultiplied RGBA read directly off the RGBA16F color attachment, no 8-bit quantization), `_depth.npy`, `_mv.npy` (float32, row-major `[H,W,C]`) plus normalized PNG previews |
| `--sort <MODE>` (Metal only) | `camera_distance` (default, Euclidean) or `view_depth` (gsplat's raw view-space z) -- a host-level mirror of the `splat` block's `sort` option (SPECIFICATION.md 12.8), since Metal's splat pipeline is hand-written MSL with no compiled `.lux` config to read `sort:` from. On Vulkan, `sort` is a compile-time `splat` block member instead (recompile with `sort: view_depth` to switch). **The Metal headless splat path never reads a compiled pipeline's `sort:` setting, even when `--pipeline <base>` names one** (splat scenes render from embedded MSL driven by the glTF scene data alone) -- so a `.lux` file declaring `sort: view_depth` (e.g. `examples/gaussian_splat_dlss.lux`) has **no effect** on `lux-playground-metal`'s own sort order; pass `--sort view_depth` explicitly on the Metal command line to match it: `./build-metal/lux-playground-metal --scene <clip>.glb --sort view_depth --output-aux <prefix>` (add `--camera-json`/`--jitter`/`--output-aux` etc. per the DLSS input-contract flags above). |
| `--reconstruct-dump <DIR>` | Run the compiled `reconstruct.{warp,apply,blend}.comp.spv` stages (SPECIFICATION.md 12.9) standalone against a directory of dumped `.npy` inputs -- no splat rendering, `--scene`/`--pipeline` are ignored. When the compiled pipeline (`--reconstruct-pipeline`) also has `bguv`/`memory` stages (a `memory` sub-block was declared), the memory path runs too and `out_f{t}.npy` reflects it. See the dump directory layout below |
| `--reconstruct-out <DIR>` | Output directory for `--reconstruct-dump`/`--dump-bg-features` (default: the dump directory itself) |
| `--reconstruct-pipeline <BASE>` | Compiled reconstruct pipeline base path for `--reconstruct-dump`/`--dump-bg-features` (default: `examples/reconstruct`) |
| `--dump-bg-features <DIR>` | Run only the `bguv`+`memory` stages (SPECIFICATION.md 12.9's scene-memory path) at *proxy* resolution against a dump directory's `bg_sphere.npy`/`texture.npy`/`k_params_f{t}.npy`/`cam_to_world_f{t}.npy`/`meta.json` -- no `--reconstruct-dump`, no splat rendering. Writes `bg_features_f{t}.npy` (`[proxy_h, proxy_w, memory.channels]`, the raw sampled scene-texture feature, pre-decode) per frame into the output directory, for the network host (which runs outside this pass) to concatenate onto `ParamPredUNet`'s other input channels |

**`--reconstruct-dump` directory layout** (all `.npy`, fp32, `[H, W, C]` row-major; `T` = `num_frames`, `t` = `0..T-1`):

| File | Shape | Notes |
|------|-------|-------|
| `meta.json` | -- | `{s, k, param_stride, hidden, proxy_w, proxy_h, target_w, target_h, net_w, net_h, num_frames}`, plus `{memory_channels, memory_hidden, tex_w, tex_h}` when the memory path is present |
| `proxy_color_f{t}.npy` | `[proxy_h, proxy_w, 3]` | this frame's (jittered) proxy colour |
| `mv_proxy_f{t}.npy` | `[proxy_h, proxy_w, 2]` | backward, jitter-free proxy-resolution motion vectors |
| `jitter_f{t}.npy` | `[2]` | raw, target-pixel-unit jitter |
| `packed_params_f{t}.npy` | `[net_h, net_w, sp*sp*k*k + sp*sp*n_blend + hidden]` (`sp = s*param_stride`, `n_blend` = 1 or 3) | `ParamPredUNet.forward_packed`'s own layout; `n_blend=3` (scene memory) iff `blend_logits`' own channel count says so |
| `disocc_f{t}.npy` | `[target_h, target_w]` | 1 = disoccluded (forces pure-spatial / zeroes the history share) |
| `bg_sphere.npy` *(memory only)* | `[4]` | `cx, cy, cz, r` -- `mobiledlss.datagen.make_clips`'s `bg_sphere` field, constant across the clip |
| `texture.npy` *(memory only)* | `[memory.channels, tex_h, tex_w]` | the per-scene feature texture (`SceneTexture.export`'s fp16 layout, upcast to fp32 on load) |
| `memory_head.npz` *(memory only)* | -- | `fc1_w [hidden, channels]`, `fc1_b [hidden]`, `fc2_w [3, hidden]`, `fc2_b [3]` -- `MemoryColorHead`'s shared-decoder weights |
| `k_params_f{t}.npy` *(memory only)* | `[4]` | `fx, fy, cx, cy` for this frame's *target*-resolution camera |
| `cam_to_world_f{t}.npy` *(memory only)* | `[4, 4]` | this frame's camera-to-world matrix (OpenCV convention) |

Writes `out_f{t}.npy` (`[target_h, target_w, 3]`) and `hidden_f{t}.npy` (`[target_h, target_w, hidden]`) per frame into the output directory. `prev_color`/`prev_hidden` are double-buffered host-side between frames (starting at zero, matching `mobiledlss.train.train.rollout`'s initial state); this port runs no network between frames, so `hidden_out` is a pure per-frame function of `packed_params_f{t}.npy` (see SPECIFICATION.md 12.9's scope note).

### Fused-GPU-compute ParamPredUNet (docs/lux-unet-spec.md, steps 1-3)

Hand-written (not yet a declarative `network` block -- step 4, only after these numbers are validated) generic conv kernels running `mobiledlss.train.model.ParamPredUNet`'s forward pass on GPU: `examples/unet_conv3x3_lrelu.lux`, `unet_conv3x3_s2_lrelu.lux`, `unet_upsample_concat_conv_lrelu.lux`, `unet_conv1x1.lux` (each a raw `compute { ... }` block, compiled like the `radix_sort_*.lux` examples: `python -m luxc examples/unet_conv3x3_lrelu.lux --define workgroup_size=256`, etc.). NHWC fp32 buffers (weights loaded from an fp16 blob and converted on load); Cin/Cout/spatial dims are runtime push-constant fields, so one compiled kernel handles every layer of that type. `playground_cpp/src/unet_runner.cpp` (Vulkan) / `metal_unet_runner.cpp` (Metal, via the same `ShaderTranspiler` as the reconstruct pass) chain the network's fixed 12-layer sequence host-side, reading a `mobiledlss/tools/export_lux_weights.py`-exported weight blob + manifest.

| Flag | Description |
|------|-------------|
| `--unet-input <NPY>` | The exact tensor `ParamPredUNet.stem` consumes -- `build_input_f{t}.npy` from `reconstruct_reference_dump.py`, `[net_h, net_w, 10+hidden]` |
| `--unet-weights <BIN>` | fp16 weight blob from `export_lux_weights.py --out-blob` |
| `--unet-manifest <JSON>` | Manifest from `export_lux_weights.py --out-manifest` (reads the `.layers.txt` sidecar written alongside it) |
| `--unet-output <NPY>` | Where to write the packed `forward_packed`-equivalent output, `[net_h, net_w, sp*sp*K*K + sp*sp + hidden]` |
| `--unet-kernel-dir <DIR>` | Directory holding the compiled `unet_*.comp.spv` kernels (default: `examples`) |

Only the 2-way blend (no scene memory) is covered, matching `checkpoints/expA_r1_spatial.pt`'s architecture exactly. Validated against the real checkpoint: max abs error on raw logits 9.4e-3 (< 2e-2 spec bound, fp16 weight quantisation), < 1.2e-3 after softmax/sigmoid (< 5e-3 bound) -- see `mobiledlss/tests/test_lux_unet.py`. Per-pass timing is logged per layer; current kernels are correctness-first (no threadgroup-memory tiling, no fp16-storage activations), so they do not yet beat Core ML's ANE floor -- see the commit message for measured numbers and the optimisation gap.

This table covers flags common to the Vulkan and Metal playgrounds; `--jitter`/`--camera-json`/`--camera-json-prev`/`--time-prev`/`--frame-prev`/`--output-aux` are implemented identically in both (`playground_cpp/src/main.cpp` and `playground_cpp/src/metal_main.cpp`), sharing their camera-math/`.npy`-writing/un-premultiply logic via `playground_cpp/src/dlss_io.h`.

## IBL Preprocessing

Convert HDR environment maps to pre-filtered cubemaps for image-based lighting:

```bash
# Download and preprocess a Khronos sample environment
python -m playground.preprocess_ibl --download neutral

# Process a custom HDR panorama
python -m playground.preprocess_ibl my_environment.hdr
```

Output: specular cubemap (6 faces x 5 mips), irradiance cubemap, and BRDF integration LUT.
