# Project Structure

```
luxc/
    __init__.py
    __main__.py              # python -m luxc
    cli.py                   # argparse CLI
    compiler.py              # pipeline orchestration
    ai/
        generate.py          # AI shader generation (multi-provider)
        system_prompt.py     # LLM system prompt
        config.py            # AIConfig + TOML persistence (~/.luxc/config.toml)
        setup.py             # interactive provider setup wizard
        providers/
            base.py          # abstract AIProvider interface
            anthropic.py     # Anthropic (Claude)
            openai_compat.py # OpenAI + Ollama + LM Studio + custom endpoints
            gemini.py        # Google Gemini
    autotype/
        types.py             # Interval, VarRange, PrecisionMap, Precision enum
        tracer.py            # dynamic range profiling via AST interpreter
        range_analysis.py    # static forward-dataflow interval propagation
        heuristics.py        # name/usage pattern classification
        classifier.py        # 3-signal precision classifier + run_auto_type_analysis()
        report.py            # text + JSON report formatting
    analysis/
        symbols.py           # symbol table, scopes
        type_checker.py      # type checking + overload resolution
        layout_assigner.py   # auto-assign locations/bindings/sets
        nan_checker.py       # static NaN/division-by-zero warnings (--warn-nan)
    autodiff/
        forward_diff.py      # forward-mode autodiff expansion
    builtins/
        types.py             # built-in type definitions
        functions.py         # built-in function signatures (40+)
    codegen/
        spirv_builder.py     # SPIR-V assembly text generator
        spirv_types.py       # type registry + deduplication
        glsl_ext.py          # GLSL.std.450 instruction mappings
        spv_assembler.py     # spirv-as / spirv-val invocation
        debug_info.py        # NonSemantic.Shader.DebugInfo.100 emission (--rich-debug)
    debug/
        values.py            # LuxScalar, LuxVec, LuxMat, LuxInt, LuxBool, LuxStruct
        interpreter.py       # tree-walking AST evaluator (44+ builtins)
        environment.py       # scoped variable storage with parent chain
        builtins.py          # GLSL.std.450 math builtins in Python
        debugger.py          # breakpoints, stepping, NaN/Inf detection
        cli.py               # interactive REPL + batch mode
        io.py                # input loading (JSON, semantic defaults)
    expansion/
        surface_expander.py  # surface/geometry/pipeline expansion
        deferred_expander.py # deferred pipeline expansion (G-buffer + lighting passes)
        splat_expander.py    # splat/gaussian_splat pipeline expansion (3-stage)
    features/
        __init__.py
        evaluator.py             # compile-time feature stripping
    grammar/
        lux.lark             # Lux EBNF grammar
        glsl_subset.lark     # GLSL transpiler grammar
    optimization/
        const_fold.py        # constant folding
    parser/
        ast_nodes.py         # AST dataclasses
        tree_builder.py      # Lark Transformer -> AST
    stdlib/
        brdf.lux             # 30+ BRDF functions
        sdf.lux              # 18 SDF primitives + CSG
        noise.lux            # 13 noise + FBM + Voronoi
        color.lux            # tonemapping + color space
        colorspace.lux       # HSV, contrast, saturation
        texture.lux          # normal maps, triplanar, UV utils
        ibl.lux              # image-based lighting + multi-scattering
        lighting.lux         # multi-light shading (directional, point, spot)
        shadow.lux           # shadow sampling (basic, PCF4, cascade selection, UV computation)
        toon.lux             # @layer cartoon cel-shading + rim lighting
        compositing.lux      # IBL multi-scattering, unified compose_pbr_layers compositing
        pbr_pipeline.lux     # PBR orchestration: pbr_shade() single-call entry point
        debug.lux            # debug visualization helpers (normal, depth, heatmap, checkerboard)
        gaussian.lux         # Gaussian splatting helpers (SH, covariance, quad radius)
        gbuffer.lux          # G-buffer utilities (octahedron encode/decode, pack/unpack, reconstruct)
        openpbr.lux          # OpenPBR Surface v1.1 material model (F82-tint, EON diffuse, coat darkening)
    transpiler/
        glsl_ast.py          # GLSL AST nodes
        glsl_parser.py       # GLSL subset parser
        glsl_to_lux.py       # GLSL -> Lux transpiler
examples/
    hello_triangle.lux       # simplest working program
    pbr_basic.lux            # manual PBR with uniforms + textures
    pbr_surface.lux          # declarative surface/geometry/pipeline
    scheduled_pbr.lux        # algorithm/schedule separation
    sdf_shapes.lux           # SDF stdlib demo
    procedural_noise.lux     # noise stdlib demo
    differentiable.lux       # @differentiable autodiff
    rt_pathtracer.lux        # declarative RT pipeline
    rt_manual.lux            # hand-written RT stages
    brdf_gallery.lux         # BRDF comparison (Lambert, GGX, PBR, clearcoat)
    colorspace_demo.lux      # HSV rainbow + colorspace transforms
    texture_demo.lux         # UV tiling, rotation, triplanar weights
    autodiff_demo.lux        # wave function + auto-generated derivative
    advanced_materials_demo.lux  # transmission, iridescence, dispersion, volume
    gltf_pbr.lux             # glTF 2.0 PBR with IBL + normal mapping
    gltf_pbr_rt.lux          # ray traced glTF PBR
    gltf_pbr_layered.lux     # layered surface — unified raster + RT pipeline
    cartoon_toon.lux         # @layer custom functions — cel-shading + rim lighting
    lighting_demo.lux        # multi-light PBR (directional, point, spot)
    multi_light_demo.lux     # multi-light with shadows (N-light loop, shadow maps)
    ibl_demo.lux             # image-based lighting demo
    math_builtins_demo.lux   # built-in function visualizer
    viz_transfer_functions.lux  # BRDF transfer function graphs (2x3 grid)
    viz_brdf_polar.lux       # polar lobe visualization (2x2 grid)
    viz_param_sweep.lux      # parameter sweep heatmaps (viridis)
    viz_furnace_test.lux     # white furnace test (energy conservation)
    viz_layer_energy.lux     # per-layer energy breakdown (stacked area)
    compute_mandelbrot.lux   # compute shader: Mandelbrot + storage image
    compute_histogram.lux    # shared memory: histogram + atomics
    compute_reduction.lux    # parallel reduction: barrier-synchronized sum
    debug_features_demo.lux  # debug_print, assert, @[debug], semantic types
    gaussian_splat.lux       # Gaussian splatting: splat decl + 3-stage pipeline
    openpbr_carpaint.lux     # OpenPBR: metallic car paint with clearcoat + thin-film
    openpbr_velvet.lux       # OpenPBR: velvet with fuzz layer + Oren-Nayar diffuse
    openpbr_glass.lux        # OpenPBR: amber glass with transmission + volume absorption
    deferred_basic.lux       # deferred rendering: G-buffer + lighting from standard glTF PBR declarations
    openpbr_simple.lux       # OpenPBR: simple no-texture surface for quick testing
    openpbr_reference.lux    # OpenPBR ASWF reference: car paint (exact values from OpenPBR viewer)
    openpbr_ref_aluminum.lux # OpenPBR ASWF reference: brushed aluminum (metalness=1)
    openpbr_ref_pearl.lux    # OpenPBR ASWF reference: pearl (coat + thin-film iridescence)
    openpbr_ref_velvet.lux   # OpenPBR ASWF reference: velvet (fuzz layer, dark base)
    debug_playground.lux     # CPU debugger playground: PBR + NaN trap
    debug_playground_inputs.json  # custom input values for debug_playground
playground/
    engine.py                # unified rendering engine (scene/pipeline separation)
    deferred_engine.py       # wgpu deferred renderer (G-buffer + lighting, reflection-driven)
    reflected_pipeline.py    # reflection-driven descriptor binding
    gltf_loader.py           # glTF 2.0 scene loader (meshes, materials, textures)
    scene_utils.py           # procedural scene generators
    preprocess_ibl.py        # HDR/EXR -> cubemap + irradiance + BRDF LUT
    test_*.py                # screenshot tests (15 tests)
assets/                      # glTF models, IBL maps (downloaded separately, gitignored)
playground_cpp/
    src/                     # native Vulkan C++ renderer (raster + RT + mesh + splat + deferred, GLFW)
    src/editor_ui.h/cpp      # interactive scene editor UI (Dear ImGui: scene tree, properties, pipeline hot-swap)
    src/deferred_renderer.cpp/h  # deferred renderer (G-buffer MRT + fullscreen lighting)
    src/splat_renderer.cpp/h # Gaussian splat renderer (compute + instanced draw)
    src/metal_*.cpp/h/mm     # native Metal renderer (raster + mesh, macOS, SPIRV-Cross)
playground_rust/
    src/                     # native Vulkan Rust renderer (ash, winit, raster + RT + mesh + splat + deferred)
    src/deferred_renderer.rs # deferred renderer (G-buffer MRT + fullscreen lighting)
    src/splat_renderer.rs    # Gaussian splat renderer (compute + instanced draw)
run_interactive_cpp.bat      # launch interactive C++ raster viewer
run_interactive_rust.bat     # launch interactive Rust raster viewer
run_interactive_cpp_rt.bat   # launch interactive C++ RT viewer
run_interactive_rust_rt.bat  # launch interactive Rust RT viewer
run_interactive_metal.sh     # launch interactive Metal raster viewer (macOS)
run_interactive_cartoon_*.bat  # cartoon shader viewers (raster + RT, C++ + Rust)
compile_mesh.bat             # compile mesh shader pipeline
run_mesh_headless_*.bat      # headless mesh shader rendering (C++ + Rust)
run_mesh_headless_metal.sh   # headless Metal mesh shader rendering (macOS)
run_mesh_interactive_*.bat   # interactive mesh shader viewers (C++ + Rust)
run_mesh_interactive_metal.sh  # interactive Metal mesh shader viewer (macOS)
run_deferred_cpp.bat         # Deferred renderer (C++)
run_deferred_rust.bat        # Deferred renderer (Rust)
run_deferred_python.bat      # Deferred renderer (Python)
run_splat_cpp.bat            # Gaussian splat viewer (C++)
run_splat_rust.bat           # Gaussian splat viewer (Rust)
run_splat_python.bat         # Gaussian splat viewer (Python)
compile_gltf_*.bat           # compile + render glTF pipelines (forward, RT, layered)
screenshots/                 # rendered gallery screenshots
shadercache/                 # compiled SPV + reflection JSON (generated, gitignored)
tests/
    test_parser.py
    test_type_checker.py
    test_codegen.py
    test_e2e.py
    test_autodiff.py
    test_transpiler.py
    test_ai_generate.py
    test_ai_config.py
    test_ai_providers.py
    test_training_data.py
    test_p5_builtins.py
    test_p5_3_advanced.py
    test_p6_raytracing.py
    test_custom_layers.py
    test_gltf_extensions.py
    test_mesh_shader.py         # P13 mesh shader tests (compilation, expansion, reflection)
    test_brdf_visualization.py  # P15 BRDF visualization tests (10 tests)
    test_lighting_block.py      # P17.1 lighting block tests (21 tests)
    test_multi_light.py         # P17.2 multi-light + shadow tests (40 tests)
    test_debug_features.py      # P20 debug instrumentation + semantic types (22 tests)
    test_debugger.py            # CPU shader debugger tests (62 tests)
    test_autotype.py            # auto-type precision optimization tests (50 tests)
    test_gaussian_splatting.py  # Gaussian splatting tests (33 tests: parser, config, expansion, compilation)
    test_stdlib_refactoring.py  # Shared stdlib refactoring tests (13 tests: compose_pbr_layers, coat IBL, helpers)
    test_deferred_rendering.py  # Deferred rendering tests (expansion, compilation, reflection, edge cases)
    test_openpbr.py             # OpenPBR material model tests (35 tests: parser, stdlib, expansion, compilation, imports)
    test_khr_splat_conformance.py  # KHR_gaussian_splatting conformance tests (226 tests: asset loading, data validation, compilation)
    test_ply_to_gltf.py         # PLY-to-glTF converter tests (73 tests: parsing, transforms, round-trip, edge cases)
tools/
    generate_training_data.py
    generate_test_splats.py  # generate test Gaussian splat .glb files (KHR_gaussian_splatting)
    ply_to_gltf.py           # enhanced PLY-to-glTF converter (coord transform, opacity/scale, SH auto-detect, verify, batch)
    glb_to_ply.py            # convert KHR_gaussian_splatting .glb to standard .ply for external viewers
    debug_splats.py          # debug visualization for Gaussian splat data
    visualize_brdf.py        # BRDF visualization CLI (compile + render + composite)
```
