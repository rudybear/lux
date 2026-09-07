#include "metal_context.h"
#include "metal_bridge.h"
#include "metal_raster_renderer.h"
#include "metal_mesh_renderer.h"
#include "metal_splat_renderer.h"
#include "metal_splat_luxc_renderer.h"
#include "metal_scene_manager.h"
#include "metal_renderer_interface.h"
#include "metal_screenshot.h"
#include "dlss_io.h"
#include "metal_reconstruct_runner.h"
#include "metal_unet_runner.h"
#include "reflected_pipeline.h"
#include "scene_light.h"
#include "camera.h"
#include "editor_panels.h"
#include "material_ubo.h"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_metal.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <iostream>
#include <string>
#include <memory>
#include <filesystem>
#include <algorithm>
#include <type_traits>
#include <cstring>
#include <stdexcept>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>

namespace fs = std::filesystem;

// --------------------------------------------------------------------------
// Orbit camera for interactive mode
// --------------------------------------------------------------------------

struct OrbitCamera {
    float yaw = 0.0f;
    float pitch = 0.15f;
    float distance = 3.0f;
    glm::vec3 target{0.0f};
    glm::vec3 up{0.0f, 1.0f, 0.0f};
    float fovY = glm::radians(45.0f);
    float nearPlane = 0.1f;
    float farPlane = 100.0f;

    bool dragging = false;
    double lastX = 0, lastY = 0;

    glm::vec3 getEye() const {
        float x = distance * cosf(pitch) * sinf(yaw);
        float y = distance * sinf(pitch);
        float z = distance * cosf(pitch) * cosf(yaw);
        return target + glm::vec3(x, y, z);
    }

    void initFromAutoCamera(const glm::vec3& eye, const glm::vec3& tgt,
                            const glm::vec3& u, float far_) {
        target = tgt;
        up = u;
        farPlane = far_;
        glm::vec3 dir = eye - tgt;
        distance = glm::length(dir);
        if (distance > 0.001f) {
            dir /= distance;
            pitch = asinf(glm::clamp(dir.y, -1.0f, 1.0f));
            yaw = atan2f(dir.x, dir.z);
        }
    }
};

static OrbitCamera g_orbit;

// --- Dynamic (4D) splats: interactive playback state ---
// Space toggles play/pause; [ and ] step one keyframe when paused.
struct MorphPlayback {
    bool paused = true;
    float time = 0.0f;
    // Debounce state for edge-triggered polling (glfwGetKey is level-triggered).
    bool spaceWasDown = false, leftBracketWasDown = false, rightBracketWasDown = false;
};
static MorphPlayback g_morph;
static EditorPanels* g_editorPanels = nullptr;

static void mouseButtonCallback(GLFWwindow* window, int button, int action, int /*mods*/) {
    // Let ImGui handle input first when editor is active
    if (g_editorPanels && g_editorPanels->wantCaptureMouse()) return;

    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        g_orbit.dragging = (action == GLFW_PRESS);
        if (g_orbit.dragging) {
            glfwGetCursorPos(window, &g_orbit.lastX, &g_orbit.lastY);
        }
    }
}

static void cursorPosCallback(GLFWwindow* /*window*/, double xpos, double ypos) {
    if (g_editorPanels && g_editorPanels->wantCaptureMouse()) return;

    if (!g_orbit.dragging) return;
    float dx = static_cast<float>(xpos - g_orbit.lastX);
    float dy = static_cast<float>(ypos - g_orbit.lastY);
    g_orbit.lastX = xpos;
    g_orbit.lastY = ypos;
    g_orbit.yaw += dx * 0.005f;
    g_orbit.pitch += dy * 0.005f;
    g_orbit.pitch = glm::clamp(g_orbit.pitch, -1.5f, 1.5f);
}

static void scrollCallback(GLFWwindow* /*window*/, double /*xoff*/, double yoff) {
    if (g_editorPanels && g_editorPanels->wantCaptureMouse()) return;

    g_orbit.distance *= (1.0f - 0.1f * static_cast<float>(yoff));
    g_orbit.distance = glm::clamp(g_orbit.distance, 0.01f, 1000.0f);
}

// --------------------------------------------------------------------------
// CLI argument parsing
// --------------------------------------------------------------------------

struct CLIOptions {
    std::string shaderBase;
    // True iff shaderBase was set from an explicit `--pipeline <base>` (or
    // positional) CLI argument, as opposed to resolveDefaultPipeline()'s
    // generic per-scene-type guess (see the splat-scene fallback below --
    // that guess is always "examples/gltf_pbr" for ANY .glb, since pipeline
    // resolution runs before the scene is loaded and can't yet know whether
    // the file turns out to contain gaussian splat data).
    bool shaderBaseExplicit = false;
    bool splatBackendExplicit = false;
    std::string sceneSource;
    std::string iblName;
    std::string forceMode;
    uint32_t width = 512;
    uint32_t height = 512;
    std::string output = "output.png";
    bool interactive = false;
    bool headless = true;
    bool demoLights = false;
    bool sponzaLights = false;
    bool editor = false;
    // Dynamic (4D) Gaussian splats: pick an animation time for headless
    // renders and the initial interactive pose. --frame maps to time via
    // mesh.extras.fps when present, else the animation's own keyframe times
    // (see MetalSplatRenderer::frameToTime / splatFrameToTime).
    bool hasTime = false;
    float time = 0.0f;
    bool hasFrame = false;
    int frame = 0;

    // DLSS input-contract outputs (docs/lux-4d-spec.md sections 3-4).
    float jitterX = 0.0f, jitterY = 0.0f;
    std::string outputAuxPrefix;
    std::string cameraJsonPath;
    std::string cameraJsonPrevPath;

    // Motion vectors: evaluate the morph at a *different* previous time
    // for a real positional MV delta from one headless render (instead of
    // the default prev-positions == current-positions "camera-only" mv).
    // See docs/lux-4d-spec.md section 3's --time-prev/--frame-prev follow-up.
    bool hasTimePrev = false;
    float timePrev = 0.0f;
    bool hasFramePrev = false;
    int framePrev = 0;

    // Verification hook (tests/test_metal_morph_gpu.py): dumps the
    // GPU-morph-updated splat_pos/rot/sh0 buffers to <prefix>_{pos,rot,sh0}.npy
    // right after setMorphTime(), for bit-exactness comparison against the
    // Python reference evaluator (playground/dynamic_splat_gltf.py).
    std::string dumpSplatBuffersPrefix;

    // Sort convention (SPECIFICATION.md 12.8's `sort` splat option, mirrored
    // as a host-level flag here since Metal's splat pipeline is hand-written
    // MSL with no compiled .lux splat config to read `sort:` from).
    bool sortByViewDepth = false;  // false = camera_distance (default) -- --splat-backend hand only

    // --splat-backend hand|luxc. "luxc" (default, since it now meets the
    // Metal-vs-Vulkan parity bar -- see metal_splat_luxc_renderer.h's class
    // comment) runs the actual compiled examples/<pipeline>.{comp,vert,
    // frag,morph.comp}.spv (+ the shared GPU radix sort) transpiled to MSL,
    // i.e. the exact same shader Vulkan runs. "hand" is MetalSplatRenderer's
    // own embedded-MSL implementation (no compiled .lux source of truth,
    // kept selectable for comparison/fallback).
    std::string splatBackend = "luxc";

    // Reconstruction pass (docs/lux-reconstruct-spec.md): see main.cpp's
    // identical Vulkan-side flags / runReconstructDump().
    std::string reconstructDumpDir;
    std::string reconstructOutDir;
    std::string reconstructPipeline = "examples/reconstruct";
    // Scene-memory path only (SPECIFICATION.md 12.9): runs bguv+memory at
    // proxy resolution, for the network host to consume the sampled
    // scene-texture features. Mutually exclusive with reconstructDumpDir.
    std::string dumpBgFeaturesDir;

    // Fused-GPU-compute ParamPredUNet (docs/lux-unet-spec.md).
    std::string unetInput;
    std::string unetWeights;
    std::string unetManifest;
    std::string unetOutput;
    std::string unetKernelDir = "examples";

    // --bench <N>: steady-state GPU-timing bench mode (bench/
    // lux_perf_ablation.md milestone 1 in mobiledlss). Splat scenes only
    // (--splat-backend luxc). Renders a fixed warm-up, then N measured
    // frames with a slowly orbiting camera, reporting median/p90 GPU ms
    // (both the fused single-command-buffer total -- the number directly
    // comparable to MetalSplatter's own gpuStartTime/gpuEndTime
    // methodology -- and a separate stage-split diagnostic sub-run) plus
    // CPU wall ms. 0 (default) = disabled, normal single-shot render.
    int benchFrames = 0;
    // Degrees of camera yaw advanced per measured frame (a "slowly
    // orbiting camera" per the task spec -- enough to exercise the sort
    // stage's view-change-triggered rescheduling without being a
    // discontinuous camera jump between frames).
    float benchOrbitDegPerFrame = 0.5f;
    // Optional opt-in sort scheduling during the bench run (milestone 2(a):
    // port of splat_renderer.h's Vulkan setSortSchedule() to Metal). 0 (the
    // struct default below) means "don't call setSortSchedule() at all" --
    // i.e. the exactness-preserving every-frame-sort default stays active,
    // matching what a plain --bench run (no extra flags) measures.
    bool benchSortSchedule = false;
    uint32_t benchSortEveryNFrames = 4;
    float benchSortViewThresholdDeg = 2.0f;
};

static void printUsage(const char* program) {
    std::cout << "Usage: " << program << " [OPTIONS]\n"
              << "\nOptions:\n"
              << "  --scene <SOURCE>       Scene: sphere, fullscreen, triangle, or .glb/.gltf path\n"
              << "  --pipeline <BASE>      Compiled shader base path\n"
              << "  --ibl <NAME>           IBL environment name\n"
              << "  --mode <MODE>          Rendering mode: mesh\n"
              << "  --width <N>            Output width (default: 512)\n"
              << "  --height <N>           Output height (default: 512)\n"
              << "  --output <PATH>        Output PNG path (default: output.png)\n"
              << "  --interactive          Open GLFW preview window\n"
              << "  --editor               Open interactive editor with ImGui overlay\n"
              << "  --headless             Offscreen render only (default)\n"
              << "  --demo-lights          Add 3 demo lights (directional + point + spot) with shadows\n"
              << "  --sponza-lights        Sponza courtyard lights (sun + torch + accent)\n"
              << "  --time <SECONDS>       Dynamic splats: animation time (headless + initial interactive pose)\n"
              << "  --frame <N>            Dynamic splats: animation frame index (extras.fps, else keyframe times)\n"
              << "  --jitter <JX> <JY>     Sub-pixel jitter in pixels (splat projection matrix only)\n"
              << "  --output-aux <PREFIX>  Write <PREFIX>_color.png, _depth.npy, _mv.npy, _fg.npy (if foreground_coverage) + PNG previews\n"
              << "  --camera-json <FILE>   Drive the splat camera from {viewmat_cv, K, width, height}\n"
              << "  --camera-json-prev <FILE> Seed the mv \"previous frame\" camera explicitly (testing)\n"
              << "  --time-prev <SECONDS> / --frame-prev <N>  Dynamic splats: evaluate the morph at\n"
              << "                         this previous time into splat_prev_pos, so mv reflects real\n"
              << "                         actor motion too (default: camera-only mv)\n"
              << "  --sort <MODE>          camera_distance (default, Euclidean) or view_depth (gsplat's z)\n"
              << "  --bench <N>            Steady-state GPU-timing bench: N frames, orbiting camera,\n"
              << "                         real MTLCommandBuffer GPU timestamps (splat + --splat-backend luxc only)\n"
              << "  --bench-orbit-deg <D>  Camera yaw degrees advanced per bench frame (default 0.5)\n"
              << "  --bench-sort-schedule <everyN> <deg>  Opt into sort scheduling during --bench\n"
              << "  --help                 Show this help message\n"
              << std::endl;
}

static CLIOptions parseArgs(int argc, char* argv[]) {
    CLIOptions opts;

    if (argc < 2) {
        printUsage(argv[0]);
        std::exit(1);
    }

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            std::exit(0);
        } else if (arg == "--scene" && i + 1 < argc) {
            opts.sceneSource = argv[++i];
        } else if (arg == "--pipeline" && i + 1 < argc) {
            opts.shaderBase = argv[++i];
            opts.shaderBaseExplicit = true;
        } else if (arg == "--ibl" && i + 1 < argc) {
            opts.iblName = argv[++i];
        } else if (arg == "--mode" && i + 1 < argc) {
            opts.forceMode = argv[++i];
        } else if (arg == "--width" && i + 1 < argc) {
            opts.width = static_cast<uint32_t>(std::stoi(argv[++i]));
        } else if (arg == "--height" && i + 1 < argc) {
            opts.height = static_cast<uint32_t>(std::stoi(argv[++i]));
        } else if (arg == "--output" && i + 1 < argc) {
            opts.output = argv[++i];
        } else if (arg == "--interactive") {
            opts.interactive = true;
            opts.headless = false;
        } else if (arg == "--editor") {
            opts.editor = true;
            opts.interactive = true;
            opts.headless = false;
        } else if (arg == "--headless") {
            opts.headless = true;
            opts.interactive = false;
        } else if (arg == "--demo-lights") {
            opts.demoLights = true;
        } else if (arg == "--sponza-lights") {
            opts.sponzaLights = true;
        } else if (arg == "--time" && i + 1 < argc) {
            opts.time = std::stof(argv[++i]);
            opts.hasTime = true;
        } else if (arg == "--frame" && i + 1 < argc) {
            opts.frame = std::stoi(argv[++i]);
            opts.hasFrame = true;
        } else if (arg == "--jitter" && i + 2 < argc) {
            opts.jitterX = std::stof(argv[++i]);
            opts.jitterY = std::stof(argv[++i]);
        } else if (arg == "--output-aux" && i + 1 < argc) {
            opts.outputAuxPrefix = argv[++i];
        } else if (arg == "--camera-json" && i + 1 < argc) {
            opts.cameraJsonPath = argv[++i];
        } else if (arg == "--camera-json-prev" && i + 1 < argc) {
            opts.cameraJsonPrevPath = argv[++i];
        } else if (arg == "--time-prev" && i + 1 < argc) {
            opts.timePrev = std::stof(argv[++i]);
            opts.hasTimePrev = true;
        } else if (arg == "--frame-prev" && i + 1 < argc) {
            opts.framePrev = std::stoi(argv[++i]);
            opts.hasFramePrev = true;
        } else if (arg == "--dump-splat-buffers" && i + 1 < argc) {
            opts.dumpSplatBuffersPrefix = argv[++i];
        } else if (arg == "--bench" && i + 1 < argc) {
            opts.benchFrames = std::stoi(argv[++i]);
        } else if (arg == "--bench-orbit-deg" && i + 1 < argc) {
            opts.benchOrbitDegPerFrame = std::stof(argv[++i]);
        } else if (arg == "--bench-sort-schedule" && i + 2 < argc) {
            opts.benchSortSchedule = true;
            opts.benchSortEveryNFrames = static_cast<uint32_t>(std::stoi(argv[++i]));
            opts.benchSortViewThresholdDeg = std::stof(argv[++i]);
        } else if (arg == "--sort" && i + 1 < argc) {
            std::string mode = argv[++i];
            if (mode == "view_depth") opts.sortByViewDepth = true;
            else if (mode == "camera_distance") opts.sortByViewDepth = false;
            else {
                std::cerr << "Unknown --sort mode: " << mode
                          << " (expected camera_distance or view_depth)" << std::endl;
                std::exit(1);
            }
        } else if (arg == "--splat-backend" && i + 1 < argc) {
            opts.splatBackend = argv[++i];
            opts.splatBackendExplicit = true;
            if (opts.splatBackend != "hand" && opts.splatBackend != "luxc") {
                std::cerr << "Unknown --splat-backend: " << opts.splatBackend
                          << " (expected hand or luxc)" << std::endl;
                std::exit(1);
            }
        } else if (arg == "--reconstruct-dump" && i + 1 < argc) {
            opts.reconstructDumpDir = argv[++i];
        } else if (arg == "--reconstruct-out" && i + 1 < argc) {
            opts.reconstructOutDir = argv[++i];
        } else if (arg == "--reconstruct-pipeline" && i + 1 < argc) {
            opts.reconstructPipeline = argv[++i];
        } else if (arg == "--dump-bg-features" && i + 1 < argc) {
            opts.dumpBgFeaturesDir = argv[++i];
        } else if (arg == "--unet-input" && i + 1 < argc) {
            opts.unetInput = argv[++i];
        } else if (arg == "--unet-weights" && i + 1 < argc) {
            opts.unetWeights = argv[++i];
        } else if (arg == "--unet-manifest" && i + 1 < argc) {
            opts.unetManifest = argv[++i];
        } else if (arg == "--unet-output" && i + 1 < argc) {
            opts.unetOutput = argv[++i];
        } else if (arg == "--unet-kernel-dir" && i + 1 < argc) {
            opts.unetKernelDir = argv[++i];
        } else if (arg[0] != '-') {
            opts.shaderBase = arg;
            opts.shaderBaseExplicit = true;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            printUsage(argv[0]);
            std::exit(1);
        }
    }

    if (!opts.reconstructDumpDir.empty() || !opts.unetInput.empty() || !opts.dumpBgFeaturesDir.empty()) {
        return opts;
    }

    if (opts.sceneSource.empty()) {
        if (!opts.shaderBase.empty()) {
            opts.sceneSource = "sphere";
        } else {
            std::cerr << "Error: --scene is required\n";
            printUsage(argv[0]);
            std::exit(1);
        }
    }

    return opts;
}

// --------------------------------------------------------------------------
// Demo lights setup
// --------------------------------------------------------------------------

static void setupDemoLights(MetalSceneManager& scene) {
    scene.clearLights();

    // Light 1: warm directional (sun-like), casts shadow
    SceneLight sun;
    sun.type = SceneLight::Directional;
    sun.direction = glm::normalize(glm::vec3(0.6f, -0.8f, 0.4f));
    sun.color = glm::vec3(1.0f, 0.95f, 0.85f);
    sun.intensity = 1.2f;
    sun.castsShadow = true;
    scene.addLight(sun);

    // Light 2: blue point light (left side)
    SceneLight blue;
    blue.type = SceneLight::Point;
    blue.position = glm::vec3(-2.0f, 1.0f, 1.0f);
    blue.color = glm::vec3(0.3f, 0.5f, 1.0f);
    blue.intensity = 3.0f;
    blue.range = 10.0f;
    scene.addLight(blue);

    // Light 3: red spot light (right side), casts shadow
    SceneLight spot;
    spot.type = SceneLight::Spot;
    spot.position = glm::vec3(2.5f, 2.0f, 1.5f);
    spot.direction = glm::normalize(glm::vec3(-1.0f, -1.0f, -0.5f));
    spot.color = glm::vec3(1.0f, 0.3f, 0.2f);
    spot.intensity = 5.0f;
    spot.range = 15.0f;
    spot.innerConeAngle = 0.2f;
    spot.outerConeAngle = 0.5f;
    spot.castsShadow = true;
    scene.addLight(spot);

    std::cout << "[metal] Demo lights: 3 lights (directional+shadow, point, spot+shadow)"
              << std::endl;
}

// --------------------------------------------------------------------------
// Pipeline resolution
// --------------------------------------------------------------------------

static std::string resolveDefaultPipeline(const std::string& scene) {
    namespace fs = std::filesystem;

    // glTF scenes: prefer layered permutation pipeline, fall back to basic gltf_pbr
    if (MetalSceneManager::isGltfFile(scene)) {
        if (fs::exists("shadercache/gltf_pbr_layered.manifest.json"))
            return "shadercache/gltf_pbr_layered";
        if (fs::exists("shadercache/gltf_pbr.frag.spv"))
            return "shadercache/gltf_pbr";
        return "examples/gltf_pbr";
    }
    if (scene == "fullscreen")
        throw std::runtime_error("--pipeline required for fullscreen scenes");
    if (scene == "triangle") {
        if (fs::exists("shadercache/hello_triangle.vert.spv"))
            return "shadercache/hello_triangle";
        return "examples/hello_triangle";
    }
    // PBR sphere
    if (fs::exists("shadercache/pbr_basic.vert.spv"))
        return "shadercache/pbr_basic";
    return "examples/pbr_basic";
}

static std::string detectRenderPath(const std::string& base, const std::string& forceMode = "",
                                     const std::string& scene = "") {
    // Scene type overrides shader-based detection for special scenes
    if (scene == "triangle") return "triangle";
    if (scene == "fullscreen") return "fullscreen";

    if (forceMode == "mesh" && fs::exists(base + ".mesh.spv") && fs::exists(base + ".frag.spv")) return "mesh";
    if (fs::exists(base + ".vert.spv") && fs::exists(base + ".frag.spv")) return "raster";
    if (fs::exists(base + ".mesh.spv") && fs::exists(base + ".frag.spv")) return "mesh";
    if (fs::exists(base + ".frag.spv")) return "fullscreen";
    throw std::runtime_error("No shader files found for: " + base);
}

// --------------------------------------------------------------------------
// Sponza courtyard lights: sun + torch + blue accent
// --------------------------------------------------------------------------

static void setupSponzaLights(MetalSceneManager& scene) {
    scene.clearLights();

    // Sun: warm directional light, casts shadow
    SceneLight sun;
    sun.type = SceneLight::Directional;
    sun.direction = glm::normalize(glm::vec3(0.5f, -0.7f, 0.3f));
    sun.color = glm::vec3(1.0f, 0.95f, 0.85f);
    sun.intensity = 8.0f;
    sun.castsShadow = true;
    scene.addLight(sun);

    // Torch: orange spot light inside the courtyard, casts shadow
    SceneLight torch;
    torch.type = SceneLight::Spot;
    torch.position = glm::vec3(0.0f, 600.0f, 0.0f);
    torch.direction = glm::normalize(glm::vec3(0.0f, -200.0f, 0.0f) - torch.position);
    torch.color = glm::vec3(1.0f, 0.7f, 0.3f);
    torch.intensity = 500000.0f;
    torch.range = 3000.0f;
    torch.innerConeAngle = 0.3f;
    torch.outerConeAngle = 0.7f;
    torch.castsShadow = true;
    scene.addLight(torch);

    // Accent: blue point light (no shadow)
    SceneLight accent;
    accent.type = SceneLight::Point;
    accent.position = glm::vec3(-500.0f, 400.0f, -300.0f);
    accent.color = glm::vec3(0.3f, 0.5f, 1.0f);
    accent.intensity = 100000.0f;
    accent.range = 2000.0f;
    scene.addLight(accent);

    std::cout << "[metal] Sponza lights: 3 lights (sun+shadow, torch+shadow, blue accent)" << std::endl;
}

// --------------------------------------------------------------------------
// --bench N: steady-state GPU-timing bench mode (bench/lux_perf_ablation.md
// milestone 1 in mobiledlss). See CLIOptions::benchFrames's comment.
// --------------------------------------------------------------------------

static void computeMedianP90(std::vector<double> v, double& median, double& p90) {
    if (v.empty()) { median = p90 = 0.0; return; }
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    median = (n % 2 == 0) ? 0.5 * (v[n / 2 - 1] + v[n / 2]) : v[n / 2];
    size_t p90idx = std::min(n - 1, static_cast<size_t>(0.9 * static_cast<double>(n)));
    p90 = v[p90idx];
}

// Runs the bench loop against a MetalSplatLuxcRenderer: a fixed warm-up,
// then benchFrames measured frames with a slowly orbiting camera (reusing
// the scene's own auto-computed eye/target/up/far, exactly like
// --interactive's initial pose). Reports median/p90 of:
//  - the fused single-command-buffer real GPU total (getLastGpuTotalMs(),
//    directly comparable to MetalSplatter's own gpuStartTime/gpuEndTime
//    methodology -- see bench/lux_perf_ablation.md in mobiledlss)
//  - CPU wall time per render() call
//  - a separate, smaller stage-split diagnostic sub-run (renderProfiled())
static void runSplatBench(MetalSplatLuxcRenderer& splatR, MetalContext& ctx,
                           MetalSceneManager& scene, const CLIOptions& opts) {
    if (opts.benchSortSchedule) {
        splatR.setSortSchedule(opts.benchSortEveryNFrames, opts.benchSortViewThresholdDeg);
        std::cout << "[bench] sort schedule: every " << opts.benchSortEveryNFrames
                  << " frames OR " << opts.benchSortViewThresholdDeg << " deg view change"
                  << std::endl;
    } else {
        std::cout << "[bench] sort schedule: every frame (default, exactness-preserving)"
                  << std::endl;
    }

    OrbitCamera orbit;
    orbit.initFromAutoCamera(scene.getAutoEye(), scene.getAutoTarget(), scene.getAutoUp(),
                              scene.getAutoFar());
    float aspect = static_cast<float>(splatR.getWidth()) / static_cast<float>(splatR.getHeight());
    float degPerFrame = opts.benchOrbitDegPerFrame;

    auto setFrameCamera = [&](int frameIdx) {
        orbit.yaw = glm::radians(degPerFrame) * static_cast<float>(frameIdx);
        glm::vec3 eye = orbit.getEye();
        splatR.updateCamera(eye, orbit.target, orbit.up, orbit.fovY, aspect,
                             orbit.nearPlane, orbit.farPlane);
    };

    const int kWarmup = 10;
    for (int i = 0; i < kWarmup; ++i) {
        setFrameCamera(i);
        splatR.render(ctx);
    }

    std::vector<double> gpuTotalMs, cpuWallMs;
    gpuTotalMs.reserve(opts.benchFrames);
    cpuWallMs.reserve(opts.benchFrames);
    for (int i = 0; i < opts.benchFrames; ++i) {
        setFrameCamera(kWarmup + i);
        auto t0 = std::chrono::steady_clock::now();
        splatR.render(ctx);
        auto t1 = std::chrono::steady_clock::now();
        cpuWallMs.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        gpuTotalMs.push_back(splatR.getLastGpuTotalMs());
    }

    double gpuMedian, gpuP90, cpuMedian, cpuP90;
    computeMedianP90(gpuTotalMs, gpuMedian, gpuP90);
    computeMedianP90(cpuWallMs, cpuMedian, cpuP90);

    // Stage-split diagnostic sub-run: fewer frames (extra CPU<->GPU round
    // trips per frame make this slower), same camera path continued from
    // where the fused-total loop left off, same sort schedule.
    const int kProfiledFrames = std::min(opts.benchFrames, 30);
    std::vector<double> preMs, sortMs, drawMs, totalMs;
    for (int i = 0; i < kProfiledFrames; ++i) {
        setFrameCamera(kWarmup + opts.benchFrames + i);
        double pre, srt, drw, tot;
        splatR.renderProfiled(ctx, &pre, &srt, &drw, &tot);
        preMs.push_back(pre); sortMs.push_back(srt); drawMs.push_back(drw); totalMs.push_back(tot);
    }
    double preMed, preP90, sortMed, sortP90, drawMed, drawP90, totMed, totP90;
    computeMedianP90(preMs, preMed, preP90);
    computeMedianP90(sortMs, sortMed, sortP90);
    computeMedianP90(drawMs, drawMed, drawP90);
    computeMedianP90(totalMs, totMed, totP90);

    std::cout << "[bench] ===== RESULTS (" << splatR.getWidth() << "x" << splatR.getHeight()
              << ", " << opts.benchFrames << " frames, " << kWarmup << " warmup) =====" << std::endl;
    std::cout << "[bench] fused_total_gpu_ms median=" << gpuMedian << " p90=" << gpuP90 << std::endl;
    std::cout << "[bench] cpu_wall_ms median=" << cpuMedian << " p90=" << cpuP90 << std::endl;
    std::cout << "[bench] stage_split (own-cmdbuf-per-stage, " << kProfiledFrames
              << " frames, includes extra CPU<->GPU sync overhead not present in fused_total):"
              << std::endl;
    std::cout << "[bench]   preprocess_gpu_ms median=" << preMed << " p90=" << preP90 << std::endl;
    std::cout << "[bench]   sort_gpu_ms       median=" << sortMed << " p90=" << sortP90 << std::endl;
    std::cout << "[bench]   draw_gpu_ms       median=" << drawMed << " p90=" << drawP90 << std::endl;
    std::cout << "[bench]   split_total_gpu_ms median=" << totMed << " p90=" << totP90 << std::endl;
}

// --------------------------------------------------------------------------
// Splat rendering (shared between --splat-backend hand and luxc): both
// MetalSplatRenderer and MetalSplatLuxcRenderer expose the identical public
// method surface (see metal_splat_luxc_renderer.h's class comment), so the
// whole camera-bridge/morph/DLSS-aux-dump sequence is written once here and
// templated on the renderer type; `initFn` is the one call whose signature
// genuinely differs between the two (luxc's init() also takes shaderBase).
// --------------------------------------------------------------------------

template <typename Renderer, typename InitFn>
static int runSplatBranch(MetalContext& ctx, MetalSceneManager& scene, const CLIOptions& opts,
                           NS::AutoreleasePool* pool, const char* backendLabel, InitFn&& initFn) {
    std::cout << "[metal] Detected gaussian splat data, using " << backendLabel << std::endl;
    auto splatR = std::make_unique<Renderer>();
    splatR->setSortByViewDepth(opts.sortByViewDepth);
    auto tInitStart = std::chrono::steady_clock::now();
    initFn(*splatR);
    auto tInitEnd = std::chrono::steady_clock::now();
    std::cout << "[metal] [timing] splat init (buffers+pipelines): "
              << std::chrono::duration<double, std::milli>(tInitEnd - tInitStart).count()
              << "ms" << std::endl;

    if (splatR->hasMotion()) {
        float t = 0.0f;
        if (opts.hasFrame) t = splatR->frameToTime(opts.frame);
        else if (opts.hasTime) t = opts.time;
        std::cout << "[metal] Dynamic splats: evaluating at t=" << t << "s"
                  << (opts.hasFrame ? " (--frame " + std::to_string(opts.frame) + ")" : "")
                  << std::endl;
        auto tMorphStart = std::chrono::steady_clock::now();
        splatR->setMorphTime(t);
        auto tMorphEnd = std::chrono::steady_clock::now();
        std::cout << "[metal] [timing] morph (GPU compute, incl. wait): "
                  << std::chrono::duration<double, std::milli>(tMorphEnd - tMorphStart).count()
                  << "ms" << std::endl;

        if (!opts.dumpSplatBuffersPrefix.empty()) {
            uint32_t n = scene.getSplatData().num_splats;
            std::vector<float> pos(splatR->debugPosBufferPtr(), splatR->debugPosBufferPtr() + n * 4);
            std::vector<float> rot(splatR->debugRotBufferPtr(), splatR->debugRotBufferPtr() + n * 4);
            std::vector<float> sh0(splatR->debugSh0BufferPtr(), splatR->debugSh0BufferPtr() + n * 4);
            DlssIO::writeNpyFloat32(opts.dumpSplatBuffersPrefix + "_pos.npy", pos, {n, 4});
            DlssIO::writeNpyFloat32(opts.dumpSplatBuffersPrefix + "_rot.npy", rot, {n, 4});
            DlssIO::writeNpyFloat32(opts.dumpSplatBuffersPrefix + "_sh0.npy", sh0, {n, 4});
            std::cout << "[metal] Dumped splat buffers: " << opts.dumpSplatBuffersPrefix
                      << "_{pos,rot,sh0}.npy" << std::endl;
        }
    } else if (opts.hasTime || opts.hasFrame) {
        std::cerr << "[warn] --time/--frame given but scene has no morph-target animation" << std::endl;
    }

    // Diagnostic: LUX_DEBUG_MORPH_BENCH=<N> simulates N frames of a
    // continuous playback loop's per-frame morph evaluation pattern
    // (setMorphTime(tCur) then seedPreviousMorphTime(tPrev), the sequence
    // playground_ios/SplatView.mm's continuous rendering loop uses) inside
    // this single process/renderer instance, and reports per-call timing --
    // for measuring the fix in seedPreviousMorphTime()'s header comment
    // (only the first call should do real GPU work; frames 2..N should be
    // ~free). Exits early (no actual render) since this is a pure timing
    // probe. Opt-in, zero effect on default behavior.
    if (splatR->hasMotion()) {
        if (const char* benchN = std::getenv("LUX_DEBUG_MORPH_BENCH")) {
            int n = std::atoi(benchN);
            float duration = splatR->animationDuration();
            double totalMs = 0.0;
            for (int i = 0; i < n; ++i) {
                float tCur = duration > 0.0f ? std::fmod(static_cast<float>(i) * 0.033f, duration) : 0.0f;
                float tPrev = duration > 0.0f ? std::fmod(static_cast<float>(i - 1) * 0.033f + duration, duration) : 0.0f;
                auto t0 = std::chrono::steady_clock::now();
                splatR->setMorphTime(tCur);
                splatR->seedPreviousMorphTime(tPrev);
                auto t1 = std::chrono::steady_clock::now();
                double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                totalMs += ms;
                std::cout << "[metal] [timing] morph bench frame " << i << ": " << ms << "ms" << std::endl;
            }
            std::cout << "[metal] [timing] morph bench: " << n << " frames, total " << totalMs
                      << "ms, mean " << (n > 0 ? totalMs / n : 0.0) << "ms/frame" << std::endl;
            pool->release();
            return 0;
        }
    }

    // Debug: same LUX_DEBUG_SPLAT_DUMP convention as the Vulkan
    // SplatRenderer::render() -- dumps the first N post-morph splat
    // attributes exactly as they're about to be handed to the preprocess
    // stage, for diffing across all three backends. Metal's setMorphTime()
    // is already synchronous (writes straight into shared-storage buffers,
    // no deferred GPU dispatch to wait on), so no extra sync is needed here
    // (unlike Vulkan's, which forces an early command-buffer submit).
    if (const char* dumpPath = std::getenv("LUX_DEBUG_SPLAT_DUMP")) {
        uint32_t n = std::min<uint32_t>(16, scene.getSplatData().num_splats);
        std::ofstream out(dumpPath);
        out << "# backend=" << opts.splatBackend << " n=" << n << " time=" << splatR->currentMorphTimeSeconds() << "\n";
        const float* pos = splatR->debugPosBufferPtr();
        for (uint32_t i = 0; i < n; ++i)
            out << "pos " << i << " " << pos[i*4] << " " << pos[i*4+1] << " " << pos[i*4+2] << " " << pos[i*4+3] << "\n";
        const float* rot = splatR->debugRotBufferPtr();
        for (uint32_t i = 0; i < n; ++i)
            out << "rot_xyzw " << i << " " << rot[i*4] << " " << rot[i*4+1] << " " << rot[i*4+2] << " " << rot[i*4+3] << "\n";
        const float* scl = splatR->debugScaleBufferPtr();
        for (uint32_t i = 0; i < n; ++i)
            out << "scale_log " << i << " " << scl[i*4] << " " << scl[i*4+1] << " " << scl[i*4+2] << "\n";
        const float* op = splatR->debugOpacityBufferPtr();
        for (uint32_t i = 0; i < n; ++i)
            out << "opacity_logit " << i << " " << op[i] << "\n";
        const float* sh0 = splatR->debugSh0BufferPtr();
        for (uint32_t i = 0; i < n; ++i)
            out << "sh0 " << i << " " << sh0[i*4] << " " << sh0[i*4+1] << " " << sh0[i*4+2] << "\n";
        std::cout << "[debug-splat] dumped " << n << " splats to " << dumpPath << std::endl;
    }

    // --- Camera bridge (docs/lux-4d-spec.md section 4) ---
    if (!opts.cameraJsonPath.empty()) {
        DlssIO::CameraJsonData camJson;
        if (!DlssIO::loadCameraJson(opts.cameraJsonPath, camJson)) {
            std::cerr << "[error] Failed to parse --camera-json file: "
                      << opts.cameraJsonPath << std::endl;
            return 1;
        }
        glm::mat4 viewGl = DlssIO::cvViewToGl(camJson.viewmatCv);
        glm::vec3 eye = glm::vec3(glm::inverse(viewGl) * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
        float fx = camJson.K[0], cx = camJson.K[2];
        float fy = camJson.K[4], cy = camJson.K[5];
        glm::mat4 proj = DlssIO::buildIntrinsicsProjection(
            fx, fy, cx, cy,
            static_cast<float>(camJson.width), static_cast<float>(camJson.height),
            0.01f, 1000.0f, /*metalYConvention=*/Renderer::kMetalYConvention);
        splatR->updateCameraExplicit(eye, viewGl, proj, fx, fy);
        std::cout << "[metal] --camera-json applied: fx=" << fx << " fy=" << fy
                  << " cx=" << cx << " cy=" << cy
                  << " (" << camJson.width << "x" << camJson.height << ")" << std::endl;

        if (!opts.cameraJsonPrevPath.empty()) {
            DlssIO::CameraJsonData prevJson;
            if (!DlssIO::loadCameraJson(opts.cameraJsonPrevPath, prevJson)) {
                std::cerr << "[error] Failed to parse --camera-json-prev file: "
                          << opts.cameraJsonPrevPath << std::endl;
                return 1;
            }
            glm::mat4 prevViewGl = DlssIO::cvViewToGl(prevJson.viewmatCv);
            glm::mat4 prevProj = DlssIO::buildIntrinsicsProjection(
                prevJson.K[0], prevJson.K[4], prevJson.K[2], prevJson.K[5],
                static_cast<float>(prevJson.width), static_cast<float>(prevJson.height),
                0.01f, 1000.0f, /*metalYConvention=*/Renderer::kMetalYConvention);
            splatR->setPreviousCameraExplicit(prevViewGl, prevProj);
            std::cout << "[metal] --camera-json-prev applied" << std::endl;
        }
    }

    // --- Motion vectors: real previous-time morph evaluation ---
    if (opts.hasTimePrev || opts.hasFramePrev) {
        float tPrev = opts.hasFramePrev ? splatR->frameToTime(opts.framePrev) : opts.timePrev;
        std::cout << "[metal] --time-prev/--frame-prev applied: evaluating morph at t=" << tPrev
                  << "s for splat_prev_pos" << std::endl;
        splatR->seedPreviousMorphTime(tPrev);
    }

    if (opts.jitterX != 0.0f || opts.jitterY != 0.0f) {
        splatR->setJitter(opts.jitterX, opts.jitterY);
    }

    if (opts.benchFrames > 0) {
        if constexpr (std::is_same_v<Renderer, MetalSplatLuxcRenderer>) {
            runSplatBench(*splatR, ctx, scene, opts);
        } else {
            std::cerr << "[error] --bench requires --splat-backend luxc" << std::endl;
            return 1;
        }
    }

    splatR->render(ctx);

    MetalScreenshot::saveTextureToPNG(ctx, splatR->getOutputTexture(),
                                       splatR->getWidth(), splatR->getHeight(),
                                       opts.output);

    // --- Headless aux dumps (docs/lux-4d-spec.md section 3) ---
    if (!opts.outputAuxPrefix.empty()) {
        uint32_t w = splatR->getWidth(), h = splatR->getHeight();
        std::string colorPath = opts.outputAuxPrefix + "_color.png";
        MetalScreenshot::saveTextureToPNG(ctx, splatR->getOutputTexture(), w, h, colorPath);

        std::vector<float> colorF32;
        {
            auto raw = MetalScreenshot::readTextureRaw(ctx, splatR->getOutputTexture(), w, h, 8);
            std::vector<uint8_t> unusedRgba8;
            colorF32 = DlssIO::convertRgba16fColorAttachment(raw, w, h, unusedRgba8);
            DlssIO::writeNpyFloat32(opts.outputAuxPrefix + "_color.npy", colorF32, {h, w, 4});
        }

        // MetalSplatRenderer (hand-written "hand" backend) and
        // MetalSplatLuxcRenderer (luxc-compiled backend) diverge here:
        // MetalSplatLuxcRenderer packs mv/depth into an RGBA32Float out_aux
        // texture (bench/lux_perf_ablation.md task 2) with GENUINE alpha
        // in its own `.w` (required for correct hardware blend
        // accumulation -- un-premultiplying via out_color's alpha instead
        // was tried and is a real, measured correctness bug, not a valid
        // shortcut, see getAuxTexture()'s header comment) and, only when
        // foreground_coverage is set, a SEPARATE RGBA16Float out_fg
        // texture (fg*alpha, alpha) that can't share out_aux's lanes.
        // MetalSplatRenderer still uses its original two-texture (RG32Float
        // motion, R32Float/RG32Float depth) design, each carrying its own
        // alpha. `if constexpr` keeps this shared template working for
        // both without forcing MetalSplatRenderer onto the new layout (out
        // of scope for task 2 -- only splat_renderer.cpp/
        // metal_splat_luxc_renderer.cpp were asked to change).
        if constexpr (std::is_same_v<Renderer, MetalSplatLuxcRenderer>) {
            if (splatR->hasMotionVectors() || splatR->hasExpectedDepth()) {
                constexpr uint32_t C = Renderer::kAuxChannels;
                bool half = splatR->auxPrecisionHalf();
                auto raw = MetalScreenshot::readTextureRaw(ctx, splatR->getAuxTexture(), w, h, half ? C * 2 : C * 4);
                std::vector<float> auxF32(static_cast<size_t>(w) * h * C);
                if (half) {
                    for (size_t i = 0; i < auxF32.size(); ++i) {
                        uint16_t h16;
                        std::memcpy(&h16, raw.data() + i * 2, 2);
                        auxF32[i] = DlssIO::halfToFloat(h16);
                    }
                } else {
                    std::memcpy(auxF32.data(), raw.data(), raw.size());
                }
                std::vector<float> auxAlpha(static_cast<size_t>(w) * h);
                for (size_t i = 0; i < auxAlpha.size(); ++i) {
                    auxAlpha[i] = auxF32[i * C + 3];
                }
                if (splatR->hasExpectedDepth()) {
                    std::vector<float> depthPremul(static_cast<size_t>(w) * h);
                    for (size_t i = 0; i < depthPremul.size(); ++i) {
                        depthPremul[i] = auxF32[i * C + 2];
                    }
                    auto depth = DlssIO::unpremultiplyByAlpha(depthPremul, auxAlpha, w, h, 1);
                    DlssIO::writeNpyFloat32(opts.outputAuxPrefix + "_depth.npy", depth, {h, w});
                    DlssIO::writeNormalizedPreviewPNG(opts.outputAuxPrefix + "_depth_preview.png", depth, w, h, 1);

                    // foreground_coverage: a SEPARATE out_fg texture,
                    // ALWAYS RGBA16Float (fg*alpha at .x, alpha at .w, 4
                    // channels) regardless of aux_precision -- see
                    // metal_splat_luxc_renderer.h's getFgFormat() comment
                    // for why a 2-channel "half" out_fg is unsafe.
                    if (splatR->hasForegroundCoverage()) {
                        uint32_t FC = splatR->getFgChannels();
                        auto rawFg = MetalScreenshot::readTextureRaw(ctx, splatR->getFgTexture(), w, h, FC * 2);
                        std::vector<float> fgF32(static_cast<size_t>(w) * h * FC);
                        for (size_t i = 0; i < fgF32.size(); ++i) {
                            uint16_t h16;
                            std::memcpy(&h16, rawFg.data() + i * 2, 2);
                            fgF32[i] = DlssIO::halfToFloat(h16);
                        }
                        std::vector<float> fgPremul(static_cast<size_t>(w) * h);
                        std::vector<float> fgAlpha(static_cast<size_t>(w) * h);
                        for (size_t i = 0; i < fgPremul.size(); ++i) {
                            fgPremul[i] = fgF32[i * FC + 0];
                            fgAlpha[i] = fgF32[i * FC + (FC - 1)];
                        }
                        auto fg = DlssIO::unpremultiplyByAlpha(fgPremul, fgAlpha, w, h, 1);
                        DlssIO::writeNpyFloat32(opts.outputAuxPrefix + "_fg.npy", fg, {h, w});
                        DlssIO::writeNormalizedPreviewPNG(opts.outputAuxPrefix + "_fg_preview.png", fg, w, h, 1);
                    }
                }
                if (splatR->hasMotionVectors()) {
                    std::vector<float> mvPremul(static_cast<size_t>(w) * h * 2);
                    for (size_t i = 0; i < auxAlpha.size(); ++i) {
                        mvPremul[i * 2 + 0] = auxF32[i * C + 0];
                        mvPremul[i * 2 + 1] = auxF32[i * C + 1];
                    }
                    auto mv = DlssIO::unpremultiplyByAlpha(mvPremul, auxAlpha, w, h, 2);
                    DlssIO::writeNpyFloat32(opts.outputAuxPrefix + "_mv.npy", mv, {h, w, 2});
                    DlssIO::writeNormalizedPreviewPNG(opts.outputAuxPrefix + "_mv_preview.png", mv, w, h, 2);
                }
            }
        } else {
            if (splatR->hasExpectedDepth()) {
                // MetalSplatRenderer's expected-depth target is RG32Float (2
                // floats/pixel: depth*alpha, alpha); alpha is always the
                // LAST channel.
                constexpr uint32_t C = Renderer::kExpectedDepthChannels;
                auto raw = MetalScreenshot::readTextureRaw(ctx, splatR->getExpectedDepthTexture(), w, h, C * 4);
                std::vector<float> chans(static_cast<size_t>(w) * h * C);
                std::memcpy(chans.data(), raw.data(), raw.size());
                std::vector<float> depthPremul(static_cast<size_t>(w) * h);
                std::vector<float> depthAlpha(static_cast<size_t>(w) * h);
                for (size_t i = 0; i < depthPremul.size(); ++i) {
                    depthPremul[i] = chans[i * C + 0];
                    depthAlpha[i] = chans[i * C + (C - 1)];
                }
                auto depth = DlssIO::unpremultiplyByAlpha(depthPremul, depthAlpha, w, h, 1);
                DlssIO::writeNpyFloat32(opts.outputAuxPrefix + "_depth.npy", depth, {h, w});
                DlssIO::writeNormalizedPreviewPNG(opts.outputAuxPrefix + "_depth_preview.png", depth, w, h, 1);

                if (splatR->hasForegroundCoverage()) {
                    std::vector<float> fgPremul(static_cast<size_t>(w) * h);
                    for (size_t i = 0; i < fgPremul.size(); ++i) {
                        fgPremul[i] = chans[i * C + 1];
                    }
                    auto fg = DlssIO::unpremultiplyByAlpha(fgPremul, depthAlpha, w, h, 1);
                    DlssIO::writeNpyFloat32(opts.outputAuxPrefix + "_fg.npy", fg, {h, w});
                    DlssIO::writeNormalizedPreviewPNG(opts.outputAuxPrefix + "_fg_preview.png", fg, w, h, 1);
                }
            }
            if (splatR->hasMotionVectors()) {
                auto raw = MetalScreenshot::readTextureRaw(ctx, splatR->getMotionTexture(), w, h, 16);
                std::vector<float> rgba(static_cast<size_t>(w) * h * 4);
                std::memcpy(rgba.data(), raw.data(), raw.size());
                std::vector<float> mvPremul(static_cast<size_t>(w) * h * 2);
                std::vector<float> mvAlpha(static_cast<size_t>(w) * h);
                for (size_t i = 0; i < mvAlpha.size(); ++i) {
                    mvPremul[i * 2 + 0] = rgba[i * 4 + 0];
                    mvPremul[i * 2 + 1] = rgba[i * 4 + 1];
                    mvAlpha[i] = rgba[i * 4 + 3];
                }
                auto mv = DlssIO::unpremultiplyByAlpha(mvPremul, mvAlpha, w, h, 2);
                DlssIO::writeNpyFloat32(opts.outputAuxPrefix + "_mv.npy", mv, {h, w, 2});
                DlssIO::writeNormalizedPreviewPNG(opts.outputAuxPrefix + "_mv_preview.png", mv, w, h, 2);
            }
        }
        std::cout << "[metal] Wrote aux dumps: " << opts.outputAuxPrefix
                  << "_{color.png,color.npy,depth.npy,mv.npy"
                  << (splatR->hasForegroundCoverage() ? ",fg.npy" : "")
                  << ",*_preview.png}" << std::endl;
    }

    splatR->cleanup();
    scene.cleanup();
    ctx.cleanup();
    pool->release();
    std::cout << "[metal] Done." << std::endl;
    return 0;
}

// --------------------------------------------------------------------------
// Headless rendering
// --------------------------------------------------------------------------

static int runHeadless(const CLIOptions& opts) {
    std::cout << "[metal] Headless render: scene=\"" << opts.sceneSource
              << "\" (" << opts.width << "x" << opts.height << ")" << std::endl;

    // Metal-cpp requires an autorelease pool for temporary Objective-C objects
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

    MetalContext ctx;
    try {
        ctx.initHeadless();
    } catch (const std::exception& e) {
        std::cerr << "[error] Failed to initialize Metal: " << e.what() << std::endl;
        pool->release();
        return 1;
    }

    try {
        MetalSceneManager scene;
        auto tLoadStart = std::chrono::steady_clock::now();
        scene.loadScene(opts.sceneSource);
        auto tLoadEnd = std::chrono::steady_clock::now();
        std::cout << "[metal] [timing] scene load: "
                  << std::chrono::duration<double, std::milli>(tLoadEnd - tLoadStart).count()
                  << "ms" << std::endl;

        // Check for gaussian splat data early -- splat scenes use embedded MSL
        // (hand backend) or the luxc-compiled pipeline (luxc backend, default).
        if (scene.hasSplatData()) {
            // resolveDefaultPipeline() ran before the scene was loaded and had
            // no way to know this .glb would turn out to hold splat data --
            // for any glTF scene it always guesses "examples/gltf_pbr" (a
            // mesh pipeline with no .comp stage at all). If the caller didn't
            // explicitly ask for --splat-backend luxc (or an explicit
            // --pipeline that might target a real splat pipeline), fall back
            // to the hand-written backend here: it's the one Metal splat
            // backend that has no compiled-.lux source of truth and therefore
            // needs no --pipeline at all, matching this binary's behavior
            // before the luxc backend became the default (commit 08cfe8e).
            bool useHandBackend = (opts.splatBackend == "hand") ||
                (!opts.splatBackendExplicit && !opts.shaderBaseExplicit);
            if (useHandBackend) {
                return runSplatBranch<MetalSplatRenderer>(ctx, scene, opts, pool, "MetalSplatRenderer (hand)",
                    [&](MetalSplatRenderer& r) { r.init(ctx, scene.getSplatData(), opts.width, opts.height); });
            }
            if (opts.shaderBase.empty()) {
                std::cerr << "[error] --splat-backend luxc requires --pipeline <base> "
                             "(e.g. --pipeline examples/gaussian_splat_dlss)" << std::endl;
                return 1;
            }
            return runSplatBranch<MetalSplatLuxcRenderer>(ctx, scene, opts, pool, "MetalSplatLuxcRenderer (luxc)",
                [&](MetalSplatLuxcRenderer& r) {
                    r.init(ctx, scene.getSplatData(), opts.shaderBase, opts.width, opts.height);
                });
        }


        // Non-splat path: populate lights from glTF scene data
        if (scene.hasGltfScene()) {
            scene.populateLightsFromGltf(scene.getGltfScene());
        }

        // Set up demo/sponza lights if requested (overrides glTF lights)
        bool useSponzaLights = opts.sponzaLights;
        if (!useSponzaLights && !opts.demoLights) {
            std::string sceneLower = opts.sceneSource;
            std::transform(sceneLower.begin(), sceneLower.end(), sceneLower.begin(), ::tolower);
            if (sceneLower.find("ponza") != std::string::npos) {
                useSponzaLights = true;
                std::cout << "[metal] Auto-detected Sponza scene, enabling sponza lights" << std::endl;
            }
        }
        if (useSponzaLights) {
            setupSponzaLights(scene);
            scene.overrideAutoCamera(
                glm::vec3(-1200.0f, 200.0f, 0.0f),
                glm::vec3(200.0f, 200.0f, 0.0f),
                glm::vec3(0.0f, 1.0f, 0.0f),
                5000.0f);
        } else if (opts.demoLights) {
            setupDemoLights(scene);
        }

        // Non-splat path: resolve shader pipeline
        std::string renderPath = detectRenderPath(opts.shaderBase, opts.forceMode, opts.sceneSource);
        bool needMesh = (renderPath == "mesh");

        std::cout << "[metal] pipeline=\"" << opts.shaderBase
                  << "\" path=" << renderPath << std::endl;

        if (needMesh && !ctx.meshShaderSupported) {
            std::cerr << "[error] Mesh shader pipeline requested but Metal 3 mesh shaders not supported." << std::endl;
            scene.cleanup();
            ctx.cleanup();
            pool->release();
            return 1;
        }


        int vertexStride = scene.hasGltfScene() ? 48 : 32;
        scene.uploadToGPU(ctx, vertexStride);
        scene.uploadTextures(ctx);
        scene.loadIBLAssets(ctx, opts.iblName);

        // Resolve permutation
        std::string resolvedBase = opts.shaderBase;
        if (scene.hasGltfScene()) {
            ShaderManifest manifest = tryLoadManifest(opts.shaderBase);
            bool hasMultiMat = scene.getGltfScene().materials.size() > 1;
            if (!manifest.permutations.empty() && !hasMultiMat) {
                auto features = scene.detectSceneFeatures();
                std::string suffix = findPermutationSuffix(manifest, features);
                if (!suffix.empty()) {
                    std::string ext = needMesh ? ".mesh.spv" : ".vert.spv";
                    if (fs::exists(opts.shaderBase + suffix + ext)) {
                        resolvedBase = opts.shaderBase + suffix;
                        std::cout << "[metal] Resolved permutation: " << suffix << std::endl;
                    }
                }
            }
        }

        std::unique_ptr<IMetalRenderer> renderer;
        if (needMesh) {
            auto meshR = std::make_unique<MetalMeshRenderer>();
            meshR->init(ctx, scene, resolvedBase, opts.width, opts.height);
            renderer = std::move(meshR);
        } else {
            auto raster = std::make_unique<MetalRasterRenderer>();
            raster->init(ctx, scene, opts.shaderBase, renderPath, opts.width, opts.height);
            renderer = std::move(raster);
        }

        renderer->render(ctx);

        MetalScreenshot::saveTextureToPNG(ctx, renderer->getOutputTexture(),
                                           renderer->getWidth(), renderer->getHeight(),
                                           opts.output);

        renderer->cleanup();
        scene.cleanup();
    } catch (const std::exception& e) {
        std::cerr << "[error] Rendering failed: " << e.what() << std::endl;
        ctx.cleanup();
        pool->release();
        return 1;
    }

    ctx.cleanup();
    pool->release();
    std::cout << "[metal] Done." << std::endl;
    return 0;
}

// --------------------------------------------------------------------------
// Interactive rendering
// --------------------------------------------------------------------------

static int runInteractive(CLIOptions opts) {
    if (opts.width == 512 && opts.height == 512) {
        opts.width = opts.editor ? 1280 : 1024;
        opts.height = opts.editor ? 800 : 768;
    }

    std::cout << "[metal] Interactive render: scene=\"" << opts.sceneSource
              << (opts.editor ? " (editor)" : "")
              << " (" << opts.width << "x" << opts.height << ")" << std::endl;

    if (!glfwInit()) {
        std::cerr << "[error] Failed to initialize GLFW" << std::endl;
        return 1;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    const char* windowTitle = opts.editor ? "Lux Editor (Metal)" : "Lux Playground (Metal)";
    GLFWwindow* window = glfwCreateWindow(
        static_cast<int>(opts.width), static_cast<int>(opts.height),
        windowTitle, nullptr, nullptr);
    if (!window) {
        std::cerr << "[error] Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return 1;
    }

    MetalContext ctx;
    try {
        ctx.init(window);
    } catch (const std::exception& e) {
        std::cerr << "[error] Failed to initialize Metal: " << e.what() << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    // Initialize editor UI (if --editor)
    EditorPanels editorPanels;
    bool editorActive = opts.editor;
    if (editorActive) {
        try {
            IMGUI_CHECKVERSION();
            ImGui::CreateContext();

            ImGuiIO& io = ImGui::GetIO();
            io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

            ImGui::StyleColorsDark();
            ImGuiStyle& style = ImGui::GetStyle();
            style.WindowRounding = 4.0f;
            style.FrameRounding = 2.0f;
            style.GrabRounding = 2.0f;
            style.WindowBorderSize = 1.0f;
            style.FrameBorderSize = 0.0f;
            style.Alpha = 0.95f;

            ImVec4* colors = style.Colors;
            colors[ImGuiCol_WindowBg] = ImVec4(0.08f, 0.08f, 0.12f, 0.94f);
            colors[ImGuiCol_TitleBg] = ImVec4(0.10f, 0.10f, 0.16f, 1.0f);
            colors[ImGuiCol_TitleBgActive] = ImVec4(0.15f, 0.15f, 0.25f, 1.0f);
            colors[ImGuiCol_Header] = ImVec4(0.20f, 0.20f, 0.35f, 0.50f);
            colors[ImGuiCol_HeaderHovered] = ImVec4(0.30f, 0.30f, 0.50f, 0.70f);
            colors[ImGuiCol_HeaderActive] = ImVec4(0.35f, 0.35f, 0.55f, 0.90f);
            colors[ImGuiCol_Button] = ImVec4(0.20f, 0.22f, 0.35f, 0.80f);
            colors[ImGuiCol_ButtonHovered] = ImVec4(0.30f, 0.32f, 0.50f, 1.0f);
            colors[ImGuiCol_ButtonActive] = ImVec4(0.35f, 0.38f, 0.60f, 1.0f);
            colors[ImGuiCol_FrameBg] = ImVec4(0.12f, 0.12f, 0.20f, 0.80f);
            colors[ImGuiCol_FrameBgHovered] = ImVec4(0.18f, 0.18f, 0.30f, 0.80f);
            colors[ImGuiCol_FrameBgActive] = ImVec4(0.22f, 0.22f, 0.38f, 0.80f);
            colors[ImGuiCol_SliderGrab] = ImVec4(0.40f, 0.45f, 0.70f, 1.0f);
            colors[ImGuiCol_SliderGrabActive] = ImVec4(0.50f, 0.55f, 0.80f, 1.0f);
            colors[ImGuiCol_CheckMark] = ImVec4(0.50f, 0.60f, 1.0f, 1.0f);
            colors[ImGuiCol_Separator] = ImVec4(0.25f, 0.25f, 0.40f, 0.50f);
            colors[ImGuiCol_Tab] = ImVec4(0.15f, 0.15f, 0.25f, 0.90f);
            colors[ImGuiCol_TabHovered] = ImVec4(0.30f, 0.30f, 0.50f, 1.0f);
            colors[ImGuiCol_TabActive] = ImVec4(0.25f, 0.25f, 0.45f, 1.0f);
            colors[ImGuiCol_MenuBarBg] = ImVec4(0.10f, 0.10f, 0.16f, 1.0f);

            ImGui_ImplGlfw_InitForOther(window, true);
            ImGui_ImplMetal_Init(ctx.device);

            editorPanels.getState().currentPipelineBase = opts.shaderBase;
            editorPanels.scanPipelines("shadercache");
            editorPanels.scanPipelines("examples");
            g_editorPanels = &editorPanels;

            std::cout << "[editor] ImGui Metal backend initialized successfully" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[error] Failed to initialize editor: " << e.what() << std::endl;
            editorActive = false;
        }
    }


    MetalSceneManager scene;
    std::unique_ptr<IMetalRenderer> renderer;

    try {
        scene.loadScene(opts.sceneSource);

        // Check for gaussian splat data early — splat scenes use embedded MSL
        if (scene.hasSplatData()) {
            std::cout << "[metal] Detected gaussian splat data, using MetalSplatRenderer" << std::endl;
            auto splatR = std::make_unique<MetalSplatRenderer>();
            splatR->init(ctx, scene.getSplatData(), opts.width, opts.height);
            if (splatR->hasMotion()) {
                if (opts.hasFrame) g_morph.time = splatR->frameToTime(opts.frame);
                else if (opts.hasTime) g_morph.time = opts.time;
                splatR->setMorphTime(g_morph.time);
                std::cout << "[metal] Dynamic splats: t=" << g_morph.time << "s / "
                          << splatR->animationDuration() << "s. Space=play/pause, "
                          << "[ / ] = step one keyframe." << std::endl;
            }
            renderer = std::move(splatR);
        } else {
            // Non-splat: populate lights from glTF
            if (scene.hasGltfScene()) {
                scene.populateLightsFromGltf(scene.getGltfScene());
            }

            // Set up demo/sponza lights if requested
            bool useSponzaLightsI = opts.sponzaLights;
            if (!useSponzaLightsI && !opts.demoLights) {
                std::string sceneLower = opts.sceneSource;
                std::transform(sceneLower.begin(), sceneLower.end(), sceneLower.begin(), ::tolower);
                if (sceneLower.find("ponza") != std::string::npos) {
                    useSponzaLightsI = true;
                    std::cout << "[metal] Auto-detected Sponza scene, enabling sponza lights" << std::endl;
                }
            }
            if (useSponzaLightsI) {
                setupSponzaLights(scene);
                scene.overrideAutoCamera(
                    glm::vec3(-1200.0f, 200.0f, 0.0f),
                    glm::vec3(200.0f, 200.0f, 0.0f),
                    glm::vec3(0.0f, 1.0f, 0.0f),
                    5000.0f);
            } else if (opts.demoLights) {
                setupDemoLights(scene);
            }

            // Resolve pipeline
            std::string renderPath = detectRenderPath(opts.shaderBase, opts.forceMode, opts.sceneSource);
            bool needMesh = (renderPath == "mesh");

            std::cout << "[metal] pipeline=\"" << opts.shaderBase
                      << "\" path=" << renderPath << std::endl;

            if (needMesh && !ctx.meshShaderSupported) {
                throw std::runtime_error("Mesh shader pipeline requested but Metal 3 mesh shaders not supported");
            }

            int vertexStride = scene.hasGltfScene() ? 48 : 32;
            scene.uploadToGPU(ctx, vertexStride);
            scene.uploadTextures(ctx);
            scene.loadIBLAssets(ctx, opts.iblName);

            std::string resolvedBase = opts.shaderBase;
            if (scene.hasGltfScene()) {
                ShaderManifest manifest = tryLoadManifest(opts.shaderBase);
                bool hasMultiMat = scene.getGltfScene().materials.size() > 1;
                if (!manifest.permutations.empty() && !hasMultiMat) {
                    auto features = scene.detectSceneFeatures();
                    std::string suffix = findPermutationSuffix(manifest, features);
                    if (!suffix.empty()) {
                        std::string ext = needMesh ? ".mesh.spv" : ".vert.spv";
                        if (fs::exists(opts.shaderBase + suffix + ext)) {
                            resolvedBase = opts.shaderBase + suffix;
                        }
                    }
                }
            }

            if (needMesh) {
                auto meshR = std::make_unique<MetalMeshRenderer>();
                meshR->init(ctx, scene, resolvedBase, opts.width, opts.height);
                renderer = std::move(meshR);
            } else {
                auto raster = std::make_unique<MetalRasterRenderer>();
                raster->init(ctx, scene, opts.shaderBase, renderPath, opts.width, opts.height);
                renderer = std::move(raster);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[error] Failed to initialize renderer: " << e.what() << std::endl;
        if (editorActive) {
            ImGui_ImplMetal_Shutdown();
            ImGui_ImplGlfw_Shutdown();
            ImGui::DestroyContext();
        }
        ctx.cleanup();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    // Initialize orbit camera from splat bounding box if no scene bounds
    if (scene.hasSplatData() && !scene.hasSceneBounds()) {
        const auto& sd = scene.getSplatData();
        glm::vec3 minB(1e9f), maxB(-1e9f);
        for (uint32_t i = 0; i < sd.num_splats; ++i) {
            float x = sd.positions[i * 4 + 0];
            float y = sd.positions[i * 4 + 1];
            float z = sd.positions[i * 4 + 2];
            minB = glm::min(minB, glm::vec3(x, y, z));
            maxB = glm::max(maxB, glm::vec3(x, y, z));
        }
        glm::vec3 center = (minB + maxB) * 0.5f;
        float radius = glm::length(maxB - minB) * 0.5f;
        if (radius < 0.001f) radius = 1.0f;
        glm::vec3 eye = center + glm::vec3(0.0f, radius * 0.5f, radius * 2.5f);
        g_orbit.initFromAutoCamera(eye, center, glm::vec3(0, 1, 0), radius * 10.0f);
    }

    // Initialize orbit camera
    if (scene.hasSceneBounds()) {
        g_orbit.initFromAutoCamera(scene.getAutoEye(), scene.getAutoTarget(),
                                   scene.getAutoUp(), scene.getAutoFar());
    }

    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);
    glfwSetScrollCallback(window, scrollCallback);

    std::cout << "[metal] Starting render loop. Press ESC or close the window to quit." << std::endl;
    std::cout << "[metal] Mouse: drag to orbit, scroll to zoom." << std::endl;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // Check for ESC key (only when editor is not capturing keyboard)
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            if (!editorActive || !editorPanels.wantCaptureKeyboard()) {
                glfwSetWindowShouldClose(window, GLFW_TRUE);
                continue;
            }
        }

        int fbWidth, fbHeight;
        glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
        if (fbWidth == 0 || fbHeight == 0) {
            glfwWaitEvents();
            continue;
        }

        // Update drawable size
        updateDrawableSize(ctx.metalLayer, window);

        // --- Dynamic splats: playback keys + advance time ---
        // Space = pause/resume, [ / ] = step one keyframe (only while paused).
        auto* splatRenderer = dynamic_cast<MetalSplatRenderer*>(renderer.get());
        if (splatRenderer && splatRenderer->hasMotion() &&
            (!editorActive || !editorPanels.wantCaptureKeyboard())) {
            bool spaceDown = glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS;
            if (spaceDown && !g_morph.spaceWasDown) {
                g_morph.paused = !g_morph.paused;
                std::cout << "[metal] Dynamic splats: " << (g_morph.paused ? "paused" : "playing")
                          << " at t=" << splatRenderer->currentMorphTimeSeconds() << "s" << std::endl;
            }
            g_morph.spaceWasDown = spaceDown;

            bool leftDown = glfwGetKey(window, GLFW_KEY_LEFT_BRACKET) == GLFW_PRESS;
            if (leftDown && !g_morph.leftBracketWasDown) {
                g_morph.paused = true;
                splatRenderer->stepKeyframe(-1);
                std::cout << "[metal] Dynamic splats: stepped to t="
                          << splatRenderer->currentMorphTimeSeconds() << "s" << std::endl;
            }
            g_morph.leftBracketWasDown = leftDown;

            bool rightDown = glfwGetKey(window, GLFW_KEY_RIGHT_BRACKET) == GLFW_PRESS;
            if (rightDown && !g_morph.rightBracketWasDown) {
                g_morph.paused = true;
                splatRenderer->stepKeyframe(1);
                std::cout << "[metal] Dynamic splats: stepped to t="
                          << splatRenderer->currentMorphTimeSeconds() << "s" << std::endl;
            }
            g_morph.rightBracketWasDown = rightDown;

            if (!g_morph.paused) {
                static double lastFrameTime = glfwGetTime();
                double now = glfwGetTime();
                float dt = static_cast<float>(now - lastFrameTime);
                lastFrameTime = now;
                float duration = splatRenderer->animationDuration();
                float t = splatRenderer->currentMorphTimeSeconds() + dt;
                if (duration > 0.0f && t > duration) t = std::fmod(t, duration);  // loop
                splatRenderer->setMorphTime(t);
            }
        }

        // Update camera
        float aspect = static_cast<float>(fbWidth) / static_cast<float>(fbHeight);
        renderer->updateCamera(g_orbit.getEye(), g_orbit.target, g_orbit.up,
                              g_orbit.fovY, aspect, g_orbit.nearPlane, g_orbit.farPlane);

        // Get next drawable and render
        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
        CA::MetalDrawable* drawable = ctx.metalLayer->nextDrawable();
        if (drawable) {
            if (editorActive) {
                // ---- Editor mode: render scene, then overlay ImGui ----
                auto* drawableTex = drawable->texture();

                // Build ImGui render pass descriptor first (needed for NewFrame)
                auto* imguiRPDesc = MTL::RenderPassDescriptor::alloc()->init();
                auto* imguiColorAtt = imguiRPDesc->colorAttachments()->object(0);
                imguiColorAtt->setTexture(drawableTex);
                imguiColorAtt->setLoadAction(MTL::LoadActionLoad);   // preserve scene
                imguiColorAtt->setStoreAction(MTL::StoreActionStore);

                // Begin ImGui frame (Metal backend needs the render pass descriptor)
                ImGui_ImplMetal_NewFrame(imguiRPDesc);
                ImGui_ImplGlfw_NewFrame();
                ImGui::NewFrame();

                // Draw editor panels
                editorPanels.drawPanels(scene);
                editorPanels.updateFPS();

                // Finalize ImGui frame (builds draw data)
                ImGui::Render();

                // Step 1: Render the 3D scene to drawable (no present, synchronous)
                renderer->renderToDrawableNoPresent(ctx, drawable);

                // Step 2: ImGui overlay pass on top of the scene
                auto* imguiCmdBuf = ctx.beginCommandBuffer();
                auto* imguiEnc = imguiCmdBuf->renderCommandEncoder(imguiRPDesc);

                ImGui_ImplMetal_RenderDrawData(ImGui::GetDrawData(), imguiCmdBuf, imguiEnc);

                imguiEnc->endEncoding();
                imguiCmdBuf->presentDrawable(drawable);
                imguiCmdBuf->commit();

                imguiRPDesc->release();
            } else {
                // Non-editor mode: just render the scene directly
                renderer->renderToDrawable(ctx, drawable);
            }
        }
        pool->release();

        // Handle editor glTF load requests
        if (editorActive && editorPanels.getState().gltfLoadRequested) {
            editorPanels.getState().gltfLoadRequested = false;
            std::string newGltf = editorPanels.getState().pendingGltfPath;
            std::cout << "[editor] Loading glTF: " << newGltf << std::endl;

            // Destroy old renderer
            if (renderer) {
                renderer->cleanup();
                renderer.reset();
            }

            // Reload scene
            try {
                scene.cleanup();
                scene.loadScene(newGltf);

                int vertexStride = scene.hasGltfScene() ? 48 : 32;
                scene.uploadToGPU(ctx, vertexStride);
                scene.uploadTextures(ctx);
                scene.loadIBLAssets(ctx, opts.iblName);

                // Rebuild renderer with current pipeline
                std::string rp = detectRenderPath(opts.shaderBase, "", newGltf);
                auto raster = std::make_unique<MetalRasterRenderer>();
                raster->init(ctx, scene, opts.shaderBase, rp, opts.width, opts.height);
                renderer = std::move(raster);

                // Reset camera to new scene bounds
                if (scene.hasSceneBounds()) {
                    g_orbit.initFromAutoCamera(
                        scene.getAutoEye(), scene.getAutoTarget(),
                        scene.getAutoUp(), scene.getAutoFar());
                }

                // Re-init editor state from new scene
                editorPanels.getState().nodeTransforms.clear();
                editorPanels.getState().materialOverrides.clear();
                editorPanels.getState().selectedNodeIndex = -1;
                editorPanels.getState().selectedMaterialIndex = -1;
                editorPanels.resetSceneState();

                opts.sceneSource = newGltf;
                std::cout << "[editor] glTF load successful" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "[editor] glTF load failed: " << e.what() << std::endl;
            }
        }

        // Handle editor material property changes
        if (editorActive && renderer && scene.hasGltfScene()) {
            auto& state = editorPanels.getState();
            auto* rasterR = dynamic_cast<MetalRasterRenderer*>(renderer.get());
            if (rasterR) {
                for (int mi = 0; mi < static_cast<int>(state.materialOverrides.size()); mi++) {
                    auto& ov = state.materialOverrides[mi];
                    if (ov.modified) {
                        ov.modified = false;
                        MaterialUBOData data{};
                        data.baseColorFactor = glm::vec4(ov.baseColor[0], ov.baseColor[1],
                                                         ov.baseColor[2], ov.baseColor[3]);
                        data.metallicFactor = ov.metallic;
                        data.roughnessFactor = ov.roughness;
                        data.emissiveFactor = glm::vec3(ov.emissive[0], ov.emissive[1], ov.emissive[2]);
                        data.emissiveStrength = 1.0f;
                        // Preserve other fields from original material
                        if (mi < static_cast<int>(scene.getGltfScene().materials.size())) {
                            const auto& mat = scene.getGltfScene().materials[mi];
                            data.ior = mat.ior;
                            data.clearcoatFactor = mat.clearcoatFactor;
                            data.clearcoatRoughnessFactor = mat.clearcoatRoughnessFactor;
                            data.sheenColorFactor = mat.sheenColorFactor;
                            data.sheenRoughnessFactor = mat.sheenRoughnessFactor;
                            data.transmissionFactor = mat.transmissionFactor;
                        }
                        rasterR->updateMaterialUBO(mi, data);
                    }
                }
            }
        }

        // Handle editor pipeline reload requests
        if (editorActive && editorPanels.getState().pipelineReloadRequested) {
            editorPanels.getState().pipelineReloadRequested = false;
            std::string newBase = editorPanels.getState().pendingPipelinePath;
            std::cout << "[editor] Hot-swap pipeline to: " << newBase << std::endl;

            // Destroy old renderer
            if (renderer) {
                renderer->cleanup();
                renderer.reset();
            }

            // Create new raster renderer with the new pipeline
            try {
                std::string newPath = detectRenderPath(newBase, "", opts.sceneSource);
                auto raster = std::make_unique<MetalRasterRenderer>();
                raster->init(ctx, scene, newBase, newPath, opts.width, opts.height);
                renderer = std::move(raster);

                opts.shaderBase = newBase;
                editorPanels.getState().currentPipelineBase = newBase;
                std::cout << "[editor] Pipeline hot-swap successful" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "[editor] Pipeline hot-swap failed: " << e.what() << std::endl;
            }
        }
    }

    // Cleanup
    if (editorActive) {
        ImGui_ImplMetal_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        g_editorPanels = nullptr;
    }

    renderer->cleanup();
    scene.cleanup();
    ctx.cleanup();

    glfwDestroyWindow(window);
    glfwTerminate();

    std::cout << "[metal] Done." << std::endl;
    return 0;
}

// --------------------------------------------------------------------------
// Entry point
// --------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    CLIOptions opts = parseArgs(argc, argv);

    if (!opts.reconstructDumpDir.empty()) {
        return runReconstructDumpMetal(opts.reconstructDumpDir, opts.reconstructOutDir,
                                        opts.reconstructPipeline);
    }

    if (!opts.dumpBgFeaturesDir.empty()) {
        return runDumpBgFeaturesMetal(opts.dumpBgFeaturesDir, opts.reconstructOutDir,
                                       opts.reconstructPipeline);
    }

    if (!opts.unetInput.empty()) {
        return runUnetDumpMetal(opts.unetInput, opts.unetWeights, opts.unetManifest,
                                 opts.unetOutput, opts.unetKernelDir);
    }

    if (opts.shaderBase.empty()) {
        try {
            opts.shaderBase = resolveDefaultPipeline(opts.sceneSource);
            std::cout << "[metal] Resolved pipeline: " << opts.shaderBase << std::endl;
        } catch (const std::exception& e) {
            // Pipeline resolution failure is OK for splat-only glTF scenes —
            // they use embedded MSL compute shaders, no external pipeline needed.
            if (MetalSceneManager::isGltfFile(opts.sceneSource)) {
                std::cout << "[metal] No shader pipeline found (may be splat-only scene)" << std::endl;
            } else {
                std::cerr << "[error] " << e.what() << std::endl;
                return 1;
            }
        }
    }

    try {
        if (opts.interactive) {
            return runInteractive(opts);
        } else {
            return runHeadless(opts);
        }
    } catch (const std::exception& e) {
        std::cerr << "[fatal] Unhandled exception: " << e.what() << std::endl;
        return 1;
    }
}
