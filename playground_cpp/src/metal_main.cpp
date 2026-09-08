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
#include "metal_live_reconstruct.h"
#include "orbit_camera.h"
#include "reflected_pipeline.h"
#include "scene_light.h"
#include "camera.h"
#include "editor_panels.h"
#include "material_ubo.h"

// --live-dump's PNG writer (proxy/bilinear/recon/target/absdiff frames) --
// implementation already lives in metal_screenshot.cpp's
// STB_IMAGE_WRITE_IMPLEMENTATION translation unit; this is just the
// declaration.
#include "stb_image_write.h"

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
#include <atomic>
#include <dispatch/dispatch.h>
#include <type_traits>
#include <cstring>
#include <stdexcept>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <set>

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
    // Set when the user actually passes --width/--height on the command
    // line (as opposed to `width`/`height` above sitting at their struct
    // defaults) -- --live-bench/--live-psnr use this to tell "the user
    // asked for a specific output size" apart from "just use the
    // live-chain's own 960x540 default" (see CLIOptions::liveTargetW's
    // comment and resolveLiveResolutions()).
    bool widthExplicit = false;
    bool heightExplicit = false;
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
    // Motion-aware extension (perf; bench/lux_perf_ablation.md in
    // mobiledlss, "Motion-aware schedule"): world-unit camera-translation
    // threshold, on top of the rotation/frame-budget ones above. 0
    // (default) disables the translation check specifically; the schedule
    // is always safe for dynamic scenes regardless via the unconditional
    // morph-time-changed check in setSortSchedule() itself.
    float benchSortTranslateThreshold = 0.0f;

    // Colour attachment format experiment (--splat-backend luxc only; see
    // MetalSplatLuxcRenderer::setColorFormat8BitExperiment()'s comment).
    // Default false = unchanged RGBA16Float colour-only behavior.
    bool colorFormat8Bit = false;

    // --live-bench <N>: headless benchmark of the SAME per-frame live-
    // inference chain playground_ios/Source/SplatView.mm runs in
    // Reconstruction mode (MetalLiveReconstruct: proxy render -> net input
    // assembly -> MPSGraph UNet -> reconstruct-with-memory), driven by the
    // identical orbit camera (LiveOrbitCamera, orbit_camera.h) -- so this
    // number is directly comparable to the iPad's own on-device log. 0
    // (default) = disabled. Independent of --scene/--pipeline (which drive
    // the generic splat/mesh code paths above) -- uses its own --live-*
    // flags instead, since it needs a proxy AND a target resolution plus
    // the exported UNet/memory-head/texture asset bundle, not a single
    // scene+pipeline+resolution. --width/--height ARE consulted, though
    // (see resolveLiveResolutions()): when the user passes either one
    // explicitly, it becomes the live chain's TARGET (output) size and the
    // proxy is derived as half that (rounded down) -- otherwise (the
    // default) liveProxyW/H and liveTargetW/H below apply unchanged, byte-
    // identical to before --width/--height were wired in here.
    int liveBenchFrames = 0;
    // --live-bench-pipelined (perf; playground_ios/Source/SplatView.mm's
    // Reconstruction-mode host-wait removal, mobiledlss task): when combined
    // with --live-bench <N>, runs an ADDITIONAL pass through the SAME N
    // frames using the 2-deep-in-flight submission pattern the iOS app now
    // uses (dispatch_semaphore_t, addCompletedHandler instead of
    // waitUntilCompleted, MetalSplatLuxcRenderer::encodeFrame's
    // frameInFlightIndex ping-ponging prevCameraBuffer_) -- reports
    // cmdbufs/waits per frame, CPU encode ms, real GPU ms and wall-clock fps
    // next to the existing always-wait baseline block, on the SAME machine/
    // scene/history continuation, so the two numbers are directly
    // comparable. Mac stand-in for the on-device A/B this task originally
    // wanted from the iPad (disconnected) -- this class (metal_live_
    // reconstruct.*) is unchanged between the two hosts, so a real overlap
    // win here is strong evidence the same win holds on-device. Default
    // false = unchanged (baseline-only) --live-bench behavior.
    bool liveBenchPipelined = false;
    // Sibling-repo asset bundle (mobiledlss/demo/ios_assets/, playground_ios's
    // own bundled resources -- see the Xcode project's ../../mobiledlss/
    // demo/ios_assets/... file references): texture.npy/unet_weights.*/
    // memory_head.npz live under exported/, bg_sphere.npy and the scene
    // .glb directly under this dir. Relative default assumes CWD ==
    // lux-4dgs repo root (matches every other relative path this binary
    // already reads, e.g. "examples/..." -- and tests/test_dlss_outputs.py's
    // own subprocess cwd=REPO_ROOT convention).
    std::string liveAssetsDir = "../mobiledlss/demo/ios_assets";
    // Defaults to <liveAssetsDir>/juggle_p0.8_stride4.glb (the pruned
    // proxy/reconstruction scene -- same one SplatView.mm's kSceneAssetName
    // loads) when left empty.
    std::string liveSceneSource;
    // Overrides the exported/{texture.npy,unet_weights.*,memory_head.npz} bundle's directory
    // independent of --live-assets-dir/--live-scene/--live-target-scene, so a different
    // checkpoint's export (e.g. demo/ios_assets/exported_p0.99/) can be benchmarked against
    // the SAME bg_sphere.npy + scene .glb layout under --live-assets-dir, without copying
    // those in too. Defaults to "" (empty), meaning <liveAssetsDir>/exported.
    std::string liveExportedDir;
    std::string livePipeline = "examples/gaussian_splat_dlss";
    // Default proxy/target resolution -- what --live-bench/--live-psnr use
    // when --width/--height are NOT given (see resolveLiveResolutions()).
    // Matches the net trained at 480x270 -> 960x540.
    uint32_t liveProxyW = 480, liveProxyH = 270;
    uint32_t liveTargetW = 960, liveTargetH = 540;
    // Overrides MetalLiveReconstruct::InitParams::paramStride (default 2, matching the
    // checkpoints exported so far) -- ps=1 checkpoints (e.g. expY_mem3_juggle_p0.9x/p1.0's
    // full-res nets) need this set to 1 or NetInputAssembly derives the net resolution at
    // the wrong stride for the exported unet_weights.* manifest (which carries its own
    // paramStride, read separately by MetalUnetRunner -- see metal_unet_runner.cpp's
    // ManifestHeader parse). Independent of --live-assets-dir/--live-exported-dir; the user
    // must keep this in sync with the checkpoint's `config['param_stride']`.
    uint32_t liveParamStride = 2;

    // --live-psnr <N>: N-frame continuous Reconstruction-vs-Target PSNR
    // gate (the Mac-side counterpart of playground_ios/Source/SplatView.mm's
    // LUX_PSNR_FRAMES on-device capture -- same orbit camera, same "no
    // history reset, continuous from frame 0" convention) -- verifies the
    // shared MetalLiveReconstruct port numerically, independent of any
    // iPad. Target mode's own unpruned scene (kSceneAssetNameFullTarget's
    // iOS rationale: Target must NOT render the pruned scene the network
    // reconstructs *toward*, or the reference itself is missing 80% of the
    // background). 0 (default) = disabled.
    int livePsnrFrames = 0;
    std::string liveTargetSceneSource;  // defaults to <assets-dir>/juggle_full_stride4.glb

    // --live-dump <DIR>: reproduces playground_ios/Source/SplatView.mm's 4
    // display modes as PNGs, headless, for visual bug-hunting without a
    // device attached (mobiledlss/reports/ios_modes_mac/). Requires
    // --live-psnr <N> (piggybacks on its already-continuous, no-history-
    // reset Reconstruction-vs-Target rollout and per-frame Target render --
    // this flag only adds PNG writes, no extra chain state). For each
    // dumped frame `f`, writes proxy_fFF.png (native 480x270 proxy colour,
    // un-premultiplied -- the pixels the Proxy display mode's 2x
    // nearest-neighbor upscale reads from), bilinear_fFF.png (2x upscale
    // via the SAME upscale_proxy GPU kernel SplatView.mm's Bicubic/
    // "Bilinear" mode dispatches -- see kLiveDumpUpscaleMSL below, kept
    // byte-identical to SplatView.mm's kUpscaleMSL since SplatView.mm
    // itself is off-limits to edit here), recon_fFF.png (MetalLiveReconstruct's
    // chain output, target-res, already straight non-premultiplied RGB --
    // same texture Reconstruction mode blits to the drawable), target_fFF.png
    // (the unpruned full-scene reference render), and
    // absdiff_recon_target_fFF.png (|recon-target| RGB, gain-boosted --
    // see writeAbsDiffPng()'s comment). 0/empty = disabled (default).
    std::string liveDumpDir;
    // Comma-separated frame indices/ranges to actually write (e.g.
    // "0-11,60-71") -- the --live-psnr rollout itself still runs
    // continuously from frame 0 through livePsnrFrames-1 regardless (real,
    // uninterrupted recurrent history), this only controls which of those
    // frames get PNGs written to --live-dump's directory. Empty (default)
    // = dump every frame in the run.
    std::string liveDumpFramesSpec;

    // --live-interactive: opens a real GLFW/Metal window and runs the same
    // shared MetalLiveReconstruct chain live, so the 4 SplatView.mm display
    // modes can be inspected on a Mac with no iPad attached -- see
    // runLiveInteractiveMetal()'s comment. Independent of the generic
    // --interactive path (which needs --scene/--pipeline; this is
    // self-contained like --live-bench/--live-psnr, own --live-* asset
    // flags).
    bool liveInteractive = false;
    // --live-shot <DIR>: with --live-interactive, auto-cycle all 4 modes
    // (Proxy/Bilinear/Reconstruction/Target -- 2s each so the recurrent
    // chain has real time to settle), write one PNG of the actual
    // on-screen drawable content per mode, then exit -- for reviewing the
    // interactive path's real display output without sitting at the
    // window. Empty (default) = disabled (normal interactive session,
    // runs until the window is closed/ESC).
    std::string liveShotDir;
};

static void printUsage(const char* program) {
    std::cout << "Usage: " << program << " [OPTIONS]\n"
              << "\nOptions:\n"
              << "  --scene <SOURCE>       Scene: sphere, fullscreen, triangle, or .glb/.gltf path\n"
              << "  --pipeline <BASE>      Compiled shader base path\n"
              << "  --ibl <NAME>           IBL environment name\n"
              << "  --mode <MODE>          Rendering mode: mesh\n"
              << "  --width <N>            Output width (default: 512; also sets --live-bench/--live-psnr's\n"
              << "                         target size when given -- see --live-bench below)\n"
              << "  --height <N>           Output height (default: 512; see --width)\n"
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
              << "  --bench-sort-schedule <everyN> <deg> [translate]  Opt into sort scheduling during --bench\n"
              << "  --live-bench <N>       Headless bench of the live-inference chain (proxy render +\n"
              << "                         net input assembly + MPSGraph UNet + reconstruct-with-memory) --\n"
              << "                         the same chain playground_ios/Source/SplatView.mm runs, driven\n"
              << "                         by the identical orbit camera. Reports fused CPU/GPU ms + fps and\n"
              << "                         a per-stage GPU breakdown. Default proxy/target res is\n"
              << "                         480x270 -> 960x540; pass --width/--height to use a different\n"
              << "                         target (output) size instead -- the proxy becomes half that\n"
              << "                         (net res is derived automatically). --live-psnr honors the\n"
              << "                         same --width/--height override.\n"
              << "  --live-bench-pipelined With --live-bench <N>: also runs the same N frames through a\n"
              << "                         2-deep-in-flight submission (semaphore + addCompletedHandler,\n"
              << "                         no waitUntilCompleted) instead of the baseline's wait-every-\n"
              << "                         frame loop, and prints cmdbufs/waits, CPU encode ms, GPU ms and\n"
              << "                         wall-clock fps for both so they can be compared directly.\n"
              << "  --live-assets-dir <DIR> exported/{texture.npy,unet_weights.*,memory_head.npz} +\n"
              << "                         bg_sphere.npy + the scene .glb (default: ../mobiledlss/demo/ios_assets)\n"
              << "  --live-scene <PATH>    Override the --live-bench scene .glb (default: <assets-dir>/juggle_p0.8_stride4.glb)\n"
              << "  --live-pipeline <BASE> Override the --live-bench compiled splat pipeline (default: examples/gaussian_splat_dlss)\n"
              << "  --live-exported-dir <DIR>  Override just the exported/ subdir (texture.npy/unet_weights.*/\n"
              << "                         memory_head.npz) independent of --live-assets-dir, so a\n"
              << "                         different checkpoint export can reuse the same bg_sphere.npy/\n"
              << "                         scene .glb (default: <assets-dir>/exported)\n"
              << "  --live-param-stride <N>  Net param_stride the exported checkpoint was trained/exported\n"
              << "                         at (default: 2) -- must match the checkpoint config, e.g. 1 for\n"
              << "                         expY_mem3_juggle_p0.9/p0.95/p0.99/p1.0\n"
              << "  --live-psnr <N>        N-frame continuous Reconstruction-vs-Target PSNR gate (Mac-side\n"
              << "                         counterpart of SplatView.mm's LUX_PSNR_FRAMES) -- verifies the\n"
              << "                         shared live-inference port numerically.\n"
              << "  --live-target-scene <PATH>  Override the --live-psnr Target-mode (unpruned) scene .glb\n"
              << "                         (default: <assets-dir>/juggle_full_stride4.glb)\n"
              << "  --live-dump <DIR>      Requires --live-psnr <N>: dump PNGs of the 4 SplatView.mm\n"
              << "                         display modes (proxy/bilinear/recon/target + an absdiff\n"
              << "                         preview) for each captured frame, headless -- for visually\n"
              << "                         diffing the app's display modes without a device.\n"
              << "  --live-dump-frames <SPEC>  Comma-separated frame indices/ranges to write with\n"
              << "                         --live-dump, e.g. \"0-11,60-71\" (default: every frame in\n"
              << "                         the --live-psnr run)\n"
              << "  --live-interactive     Open a real GLFW/Metal window running the live-inference\n"
              << "                         chain (same shared MetalLiveReconstruct as the iPad app), with\n"
              << "                         the SAME 4 display modes SplatView.mm has -- keys 1-4 select\n"
              << "                         Proxy/Bilinear/Reconstruction/Target directly, space cycles.\n"
              << "                         Title bar shows mode/fps/GPU ms. ESC or close the window to quit.\n"
              << "  --live-shot <DIR>      With --live-interactive: auto-cycle all 4 modes (2s each),\n"
              << "                         write one PNG of the actual on-screen drawable per mode to\n"
              << "                         <DIR>/{proxy,bilinear,recon,target}.png, then exit.\n"
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
            opts.widthExplicit = true;
        } else if (arg == "--height" && i + 1 < argc) {
            opts.height = static_cast<uint32_t>(std::stoi(argv[++i]));
            opts.heightExplicit = true;
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
        } else if (arg == "--color-format-8bit") {
            opts.colorFormat8Bit = true;
        } else if (arg == "--bench-sort-schedule" && i + 2 < argc) {
            opts.benchSortSchedule = true;
            opts.benchSortEveryNFrames = static_cast<uint32_t>(std::stoi(argv[++i]));
            opts.benchSortViewThresholdDeg = std::stof(argv[++i]);
            // Optional 3rd arg: translation threshold (world units). Only
            // consumed if present and not itself the next flag.
            if (i + 1 < argc) {
                std::string maybeTranslate = argv[i + 1];
                if (!maybeTranslate.empty() && maybeTranslate.substr(0, 2) != "--") {
                    opts.benchSortTranslateThreshold = std::stof(argv[++i]);
                }
            }
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
        } else if (arg == "--live-bench" && i + 1 < argc) {
            opts.liveBenchFrames = std::stoi(argv[++i]);
        } else if (arg == "--live-bench-pipelined") {
            opts.liveBenchPipelined = true;
        } else if (arg == "--live-assets-dir" && i + 1 < argc) {
            opts.liveAssetsDir = argv[++i];
        } else if (arg == "--live-scene" && i + 1 < argc) {
            opts.liveSceneSource = argv[++i];
        } else if (arg == "--live-pipeline" && i + 1 < argc) {
            opts.livePipeline = argv[++i];
        } else if (arg == "--live-exported-dir" && i + 1 < argc) {
            opts.liveExportedDir = argv[++i];
        } else if (arg == "--live-param-stride" && i + 1 < argc) {
            opts.liveParamStride = static_cast<uint32_t>(std::stoi(argv[++i]));
        } else if (arg == "--live-psnr" && i + 1 < argc) {
            opts.livePsnrFrames = std::stoi(argv[++i]);
        } else if (arg == "--live-target-scene" && i + 1 < argc) {
            opts.liveTargetSceneSource = argv[++i];
        } else if (arg == "--live-dump" && i + 1 < argc) {
            opts.liveDumpDir = argv[++i];
        } else if (arg == "--live-dump-frames" && i + 1 < argc) {
            opts.liveDumpFramesSpec = argv[++i];
        } else if (arg == "--live-interactive") {
            opts.liveInteractive = true;
        } else if (arg == "--live-shot" && i + 1 < argc) {
            opts.liveShotDir = argv[++i];
        } else if (arg[0] != '-') {
            opts.shaderBase = arg;
            opts.shaderBaseExplicit = true;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            printUsage(argv[0]);
            std::exit(1);
        }
    }

    if (!opts.reconstructDumpDir.empty() || !opts.unetInput.empty() || !opts.dumpBgFeaturesDir.empty() ||
        opts.liveBenchFrames > 0 || opts.livePsnrFrames > 0 || opts.liveInteractive) {
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
        splatR.setSortSchedule(opts.benchSortEveryNFrames, opts.benchSortViewThresholdDeg,
                                opts.benchSortTranslateThreshold);
        std::cout << "[bench] sort schedule: every " << opts.benchSortEveryNFrames
                  << " frames OR " << opts.benchSortViewThresholdDeg << " deg view change"
                  << " OR " << opts.benchSortTranslateThreshold << " world-unit translate"
                  << " OR any morph-time change"
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

// Resolves the live chain's proxy/target resolution from CLIOptions, for
// both --live-bench and --live-psnr. Default (no --width/--height on the
// command line) reproduces the old hardcoded behavior exactly:
// liveProxyW/H x liveTargetW/H (480x270 -> 960x540). When the user passes
// --width and/or --height explicitly, THAT becomes the live chain's target
// (output) resolution instead, and the proxy is derived as half of it
// (integer division, rounded down) -- matching this measurement's
// "proxy at half res" convention (see bench/lux_perf_ablation.md in
// mobiledlss). The net's own resolution is NOT set here: MetalLiveReconstruct
// ::init() -> NetInputAssembly::init() derives netW/H from proxyW/H itself
// (round proxy up to a multiple of 8*paramStride, then divide by
// paramStride -- e.g. proxy 480x270, paramStride=2 -> net 240x136), and
// that derivation was already fully general (parameterized on proxyW/H,
// not hardcoded), so any proxy size handed to it here "just works" -- see
// net_input_assembly.mm's NetInputAssembly::init(). Likewise
// MPSGraphUNet::init() rebuilds its MPSGraph graph for whatever netW/H it's
// given (the model is fully convolutional -- mobiledlss/train/model.py),
// and the scene texture / bg-sphere UV sampling in NetInputAssembly /
// LiveReconstructPass is resolution-independent (UV-based, not a fixed
// pixel grid), so no further plumbing is needed for --live-bench/
// --live-psnr to run the whole chain at an arbitrary output size.
static void resolveLiveResolutions(const CLIOptions& opts, uint32_t& proxyW, uint32_t& proxyH, uint32_t& targetW,
                                    uint32_t& targetH) {
    if (opts.widthExplicit || opts.heightExplicit) {
        targetW = opts.width;
        targetH = opts.height;
        proxyW = targetW / 2;
        proxyH = targetH / 2;
    } else {
        proxyW = opts.liveProxyW;
        proxyH = opts.liveProxyH;
        targetW = opts.liveTargetW;
        targetH = opts.liveTargetH;
    }
}

// --------------------------------------------------------------------------
// --live-bench N: headless benchmark of the SAME per-frame live-inference
// chain playground_ios/Source/SplatView.mm runs in Reconstruction mode
// (MetalLiveReconstruct: proxy render -> net input assembly -> MPSGraph
// UNet -> reconstruct-with-memory), driven by the identical orbit camera
// (LiveOrbitCamera, orbit_camera.h) at the identical proxy/target
// resolutions -- so this Mac number is directly comparable to the iPad's
// own on-device log. Self-contained (own MetalContext/MetalSceneManager,
// like runReconstructDumpMetal/runUnetDumpMetal), independent of the
// generic --scene/--pipeline splat/mesh code paths above (--width/--height
// ARE consulted -- see resolveLiveResolutions()).
// --------------------------------------------------------------------------

static int runLiveBenchMetal(const CLIOptions& opts) {
    MetalContext ctx;
    ctx.initHeadless();

    std::string sceneSource =
        opts.liveSceneSource.empty() ? (opts.liveAssetsDir + "/juggle_p0.8_stride4.glb") : opts.liveSceneSource;

    MetalSceneManager scene;
    scene.loadScene(sceneSource);
    if (!scene.hasSplatData()) {
        std::cerr << "[live-bench] scene has no KHR_gaussian_splatting data: " << sceneSource << std::endl;
        return 1;
    }

    uint32_t proxyW, proxyH, targetW, targetH;
    resolveLiveResolutions(opts, proxyW, proxyH, targetW, targetH);

    MetalLiveReconstruct live;
    MetalLiveReconstruct::InitParams params;
    params.shaderBase = opts.livePipeline;
    params.proxyW = proxyW;
    params.proxyH = proxyH;
    params.targetW = targetW;
    params.targetH = targetH;
    params.paramStride = opts.liveParamStride;
    params.hiddenChannels = 8;
    const std::string liveExportedDir =
        opts.liveExportedDir.empty() ? (opts.liveAssetsDir + "/exported") : opts.liveExportedDir;
    params.textureNpyPath = liveExportedDir + "/texture.npy";
    params.bgSphereNpyPath = opts.liveAssetsDir + "/bg_sphere.npy";
    params.unetWeightsBinPath = liveExportedDir + "/unet_weights.fp16.bin";
    params.unetLayersTxtPath = liveExportedDir + "/unet_weights.layers.txt";
    params.memoryHeadNpzPath = liveExportedDir + "/memory_head.npz";
    // (4, 2.0f) -- perf; bench/lux_perf_ablation.md's "Motion-aware
    // schedule" (see runLivePsnrMetal's identical comment and
    // MetalSplatLuxcRenderer::setSortSchedule()'s header for the full
    // story): the un-fixed version of this schedule badly broke
    // reconstruction quality on THIS dynamic (actor-motion) scene
    // (periodic ~15dB collapses), because it only watched camera
    // rotation, with no signal for the scene content itself moving.
    // setSortSchedule() now also re-sorts on any currentMorphTime_ change,
    // which fires every frame this continuously-animating scene actually
    // renders -- so numerically this measures IDENTICALLY to the
    // un-scheduled every-frame-sort default (verified via --live-psnr:
    // frame-by-frame PSNR bit-for-bit unchanged) since this benchmark's
    // scene never holds still long enough for the schedule to skip a
    // sort. The schedule now being safe to opt into here at all -- rather
    // than left at the exactness-preserving default out of necessity -- is
    // itself the fix; a scene with actual static-camera-and-content holds
    // would see the amortized win this schedule was designed for.
    params.sortEveryNFrames = 4;
    params.sortViewThresholdDeg = 2.0f;

    auto tInit0 = std::chrono::steady_clock::now();
    live.init(ctx, scene.getSplatData(), params);
    auto tInit1 = std::chrono::steady_clock::now();
    std::cout << "[live-bench] init: " << std::chrono::duration<double, std::milli>(tInit1 - tInit0).count()
              << "ms, netRes=" << live.getNetW() << "x" << live.getNetH()
              << " splats=" << scene.getSplatData().num_splats << std::endl;

    auto buildFrameInputs = [&](int frame) -> MetalLiveReconstruct::FrameInputs {
        MetalLiveReconstruct::FrameInputs in;
        int prevFrame = frame > 0 ? frame - 1 : 0;
        auto toCamFrame = [](const LiveOrbitCamera::Result& r) {
            return MetalLiveReconstruct::CameraFrame{r.eye, r.r, r.u, r.f, r.viewGl, r.proj, r.fx, r.fy};
        };
        in.proxyCur = toCamFrame(
            LiveOrbitCamera::compute(frame, live.getProxyW(), live.getProxyH(), MetalSplatLuxcRenderer::kMetalYConvention));
        in.proxyPrev = toCamFrame(LiveOrbitCamera::compute(prevFrame, live.getProxyW(), live.getProxyH(),
                                                            MetalSplatLuxcRenderer::kMetalYConvention));
        in.targetCur = toCamFrame(LiveOrbitCamera::compute(frame, live.getTargetW(), live.getTargetH(),
                                                            MetalSplatLuxcRenderer::kMetalYConvention));
        bool hasMotion = live.proxyRenderer().hasMotion();
        in.morphTimeCur = hasMotion ? live.proxyRenderer().frameToTime(frame) : 0.0f;
        in.morphTimePrev = hasMotion ? live.proxyRenderer().frameToTime(prevFrame) : 0.0f;
        float jxTarget, jyTarget;
        LiveOrbitCamera::taaJitterTargetPx(frame, /*period=*/16, jxTarget, jyTarget);
        float scaleP = static_cast<float>(live.getProxyW()) / static_cast<float>(live.getTargetW());
        in.jitterTargetX = jxTarget;
        in.jitterTargetY = jyTarget;
        in.jitterProxyX = jxTarget * scaleP;
        in.jitterProxyY = jyTarget * scaleP;
        return in;
    };

    // Every frame gets its OWN autoreleasepool (matches playground_ios/
    // Source/SplatView.mm's -tick: convention exactly): ctx.beginCommandBuffer()
    // (MTLCommandQueue.commandBuffer) returns an AUTORELEASED command buffer
    // with no explicit retain (see MetalContext::beginCommandBuffer()), and
    // MPSGraphUNet::encode() creates a bunch of short-lived MPSGraph
    // objects of its own inside its own inner @autoreleasepool -- both need
    // a live pool somewhere on the stack for the whole frame, which nothing
    // upstream of this early-return CLI mode otherwise provides (unlike the
    // generic scene/mesh code paths below, which run inside main()'s own
    // long-lived top-level pool). Without this, the very first frame
    // reliably crashes (EXC_BAD_ACCESS in objc_msgSend) once the UNet
    // stage's own inner pool drains and frees the command buffer object
    // out from under the still-in-progress frame.
    const int kWarmup = 10;
    for (int i = 0; i < kWarmup; ++i) {
        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
        auto in = buildFrameInputs(i);
        auto* cmdBuf = ctx.beginCommandBuffer();
        auto* cur = live.encodeFrame(ctx, cmdBuf, in);
        cur->commit();
        cur->waitUntilCompleted();
        cur->release();  // balances MPSGraphUNet::encode()'s extra retain -- see its own doc comment.
        pool->release();
    }

    // Fused (production) path: one command buffer/frame, zero inter-stage
    // host waits -- exactly what SplatView.mm's Reconstruction mode does.
    std::vector<double> gpuTotalMs, cpuWallMs;
    gpuTotalMs.reserve(static_cast<size_t>(opts.liveBenchFrames));
    cpuWallMs.reserve(static_cast<size_t>(opts.liveBenchFrames));
    for (int i = 0; i < opts.liveBenchFrames; ++i) {
        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
        auto in = buildFrameInputs(kWarmup + i);
        auto t0 = std::chrono::steady_clock::now();
        auto* cmdBuf = ctx.beginCommandBuffer();
        auto* cur = live.encodeFrame(ctx, cmdBuf, in);
        cur->commit();
        cur->waitUntilCompleted();
        auto t1 = std::chrono::steady_clock::now();
        cpuWallMs.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        // See MetalLiveReconstruct::gpuMsAcrossPossibleSplit()'s own comment:
        // `cur`'s own GPUStartTime()/GPUEndTime() alone can badly
        // under-report the frame's real GPU cost whenever MPSGraph's UNet
        // internally commitAndContinue-split the command buffer.
        gpuTotalMs.push_back(MetalLiveReconstruct::gpuMsAcrossPossibleSplit(cmdBuf, cur));
        cur->release();  // balances MPSGraphUNet::encode()'s extra retain -- see its own doc comment.
        pool->release();
    }

    double gpuMedian, gpuP90, cpuMedian, cpuP90;
    computeMedianP90(gpuTotalMs, gpuMedian, gpuP90);
    computeMedianP90(cpuWallMs, cpuMedian, cpuP90);

    // Per-stage GPU breakdown (own-cmdbuf-per-stage diagnostic path,
    // encodeFrameProfiled()) -- continues the SAME history state (no
    // resetHistory()); fewer frames since each pays 3 extra CPU<->GPU round
    // trips vs. the fused path above.
    const int kProfiledFrames = std::min(opts.liveBenchFrames, 30);
    std::vector<double> proxyMs, inputMs, netMs, reconMs, splitTotalMs;
    for (int i = 0; i < kProfiledFrames; ++i) {
        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
        auto in = buildFrameInputs(kWarmup + opts.liveBenchFrames + i);
        auto r = live.encodeFrameProfiled(ctx, in);
        proxyMs.push_back(r.proxyGpuMs);
        inputMs.push_back(r.inputGpuMs);
        netMs.push_back(r.netGpuMs);
        reconMs.push_back(r.reconGpuMs);
        splitTotalMs.push_back(r.totalGpuMs);
        pool->release();
    }
    double proxyMed, proxyP90, inputMed, inputP90, netMed, netP90, reconMed, reconP90, splitMed, splitP90;
    computeMedianP90(proxyMs, proxyMed, proxyP90);
    computeMedianP90(inputMs, inputMed, inputP90);
    computeMedianP90(netMs, netMed, netP90);
    computeMedianP90(reconMs, reconMed, reconP90);
    computeMedianP90(splitTotalMs, splitMed, splitP90);

    std::cout << "[live-bench] ===== RESULTS (proxy " << live.getProxyW() << "x" << live.getProxyH()
              << " -> target " << live.getTargetW() << "x" << live.getTargetH() << ", net " << live.getNetW()
              << "x" << live.getNetH() << ", " << opts.liveBenchFrames << " frames, " << kWarmup
              << " warmup) =====" << std::endl;
    std::cout << "[live-bench] fused_total_gpu_ms   median=" << gpuMedian << " p90=" << gpuP90 << std::endl;
    std::cout << "[live-bench] cpu_wall_ms          median=" << cpuMedian << " p90=" << cpuP90 << "  ("
              << (cpuMedian > 0.0 ? 1000.0 / cpuMedian : 0.0) << " fps)" << std::endl;
    std::cout << "[live-bench] stage_split (own-cmdbuf-per-stage, " << kProfiledFrames
              << " frames, includes extra CPU<->GPU sync overhead not present in fused_total):" << std::endl;
    std::cout << "[live-bench]   proxy_gpu_ms       median=" << proxyMed << " p90=" << proxyP90 << std::endl;
    std::cout << "[live-bench]   input_gpu_ms       median=" << inputMed << " p90=" << inputP90 << std::endl;
    std::cout << "[live-bench]   net_gpu_ms         median=" << netMed << " p90=" << netP90 << std::endl;
    std::cout << "[live-bench]   recon_gpu_ms       median=" << reconMed << " p90=" << reconP90 << std::endl;
    std::cout << "[live-bench]   split_total_gpu_ms median=" << splitMed << " p90=" << splitP90 << std::endl;

    // --live-bench-pipelined: SAME live/scene/history (continues on from the
    // frame index the two blocks above already consumed -- no resetHistory()
    // -- so this is a like-for-like continuation of the same run, not a
    // fresh warm-up), but submitted the way playground_ios/Source/
    // SplatView.mm's Reconstruction mode now does: 2 frames' command buffers
    // may be outstanding on the GPU at once (dispatch_semaphore_t, count 2),
    // no waitUntilCompleted anywhere in the loop, GPU timing read back
    // asynchronously via addCompletedHandler, and MetalSplatLuxcRenderer's
    // prevCameraBuffer_ ping-ponged via frameInFlightIndex=i&1 (see its own
    // doc comment for why a single shared buffer would race under overlap).
    // Real per-frame throughput under overlap isn't "time for one iteration
    // of this loop" (that's just CPU encode time now, not GPU-bound) -- it's
    // wall-clock across the WHOLE batch, N frames / total_seconds.
    if (opts.liveBenchPipelined) {
        const int N = opts.liveBenchFrames;
        const int frameBase = kWarmup + opts.liveBenchFrames + kProfiledFrames;
        dispatch_semaphore_t sem = dispatch_semaphore_create(2);
        std::vector<double> gpuTotalMsP(static_cast<size_t>(N), 0.0);
        std::vector<double> cpuEncodeMsP(static_cast<size_t>(N), 0.0);
        std::vector<int> cmdbufCountP(static_cast<size_t>(N), 0);
        std::atomic<int> completedCount{0};

        auto tBatch0 = std::chrono::steady_clock::now();
        for (int i = 0; i < N; ++i) {
            NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
            // Bounds outstanding GPU work to 2 frames -- blocks here only
            // once frame i-2's command buffer hasn't completed yet by the
            // time frame i wants to start encoding (the intended, expected
            // "semaphore wait is fine" throttle -- NOT counted as a
            // waitUntilCompleted-style host wait in the per-frame stats
            // below, matching SplatView.mm's own accounting).
            dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
            auto in = buildFrameInputs(frameBase + i);
            auto t0 = std::chrono::steady_clock::now();
            auto* cmdBuf = ctx.beginCommandBuffer();
            // Own +1 on cmdBuf -- see the crash this fixed, below. `cur`
            // already carries its own extra +1 from MPSGraphUNet::encode()'s
            // __bridge_retained (balanced by cur->release() below); cmdBuf
            // gets none from anywhere, and ctx.beginCommandBuffer() itself
            // returns an AUTORELEASED pointer (MTLCommandQueue::commandBuffer
            // semantics), autoreleased into THIS iteration's own pool, which
            // drains (pool->release()) before this frame's completion
            // handler can possibly fire.
            cmdBuf->retain();
            auto* cur = live.encodeFrame(ctx, cmdBuf, in, static_cast<uint32_t>(i & 1));
            cmdbufCountP[static_cast<size_t>(i)] = (cur != cmdBuf) ? 2 : 1;
            // Registered BEFORE commit() (required -- see MTLCommandBuffer's
            // addCompletedHandler docs).
            //
            // First cut of this got EXC_BAD_ACCESS (objc_msgSend) here on
            // the SPLIT path (MPSGraph's commitAndContinue, `cur != cmdBuf`):
            // committing a command buffer does NOT, on its own, keep it
            // alive until every handler on it has run -- only a handler
            // registered ON THAT SPECIFIC BUFFER gets that guarantee (from
            // the runtime holding it live for ITS OWN dispatch). Nothing
            // else held a strong ref to `cmdBuf` past this iteration's own
            // NS::AutoreleasePool draining a few lines down (pool->release()),
            // so by the time `cur`'s completion handler ran -- after cmdBuf
            // had ALREADY completed and this loop had moved on several
            // frames -- cmdBuf was a dangling pointer. Passed the same crash
            // to playground_ios/Source/SplatView.mm's real device
            // implementation before it ever got the chance to run there --
            // this class of bug is exactly what MPSGraphUNet::encode()'s own
            // __bridge_retained fix (see its doc comment) already covers for
            // `cur`; cmdBuf just needed the same treatment.
            cur->addCompletedHandler([&gpuTotalMsP, &completedCount, &sem, cmdBuf, cur, i](MTL::CommandBuffer*) {
                gpuTotalMsP[static_cast<size_t>(i)] = MetalLiveReconstruct::gpuMsAcrossPossibleSplit(cmdBuf, cur);
                completedCount.fetch_add(1, std::memory_order_relaxed);
                cmdBuf->release();  // balances the retain() above.
                dispatch_semaphore_signal(sem);
            });
            cur->commit();
            auto t1 = std::chrono::steady_clock::now();
            cpuEncodeMsP[static_cast<size_t>(i)] = std::chrono::duration<double, std::milli>(t1 - t0).count();
            cur->release();  // balances MPSGraphUNet::encode()'s extra retain -- see its own doc comment.
            pool->release();
        }
        // Drain the up-to-2 frames still in flight after the loop -- once
        // both have signalled, completedCount==N and gpuTotalMsP/cmdbufCountP
        // are fully populated.
        dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
        dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
        auto tBatch1 = std::chrono::steady_clock::now();
        double batchSeconds = std::chrono::duration<double>(tBatch1 - tBatch0).count();
        double wallFps = batchSeconds > 0.0 ? static_cast<double>(N) / batchSeconds : 0.0;

        double gpuMedianP, gpuP90P, cpuMedianP, cpuP90P;
        computeMedianP90(gpuTotalMsP, gpuMedianP, gpuP90P);
        computeMedianP90(cpuEncodeMsP, cpuMedianP, cpuP90P);
        double cmdbufMean = 0.0;
        for (int c : cmdbufCountP) cmdbufMean += c;
        cmdbufMean /= std::max(1, N);

        std::cout << "[live-bench] ===== PIPELINED (2 frames in flight, " << N << " frames, completed="
                  << completedCount.load() << "/" << N << ") =====" << std::endl;
        std::cout << "[live-bench] wall_fps            " << wallFps << "  (" << batchSeconds * 1000.0
                  << "ms / " << N << " frames, real overlapped throughput)" << std::endl;
        std::cout << "[live-bench] fused_total_gpu_ms   median=" << gpuMedianP << " p90=" << gpuP90P
                  << "  (vs. baseline median=" << gpuMedian << ")" << std::endl;
        std::cout << "[live-bench] cpu_encode_ms        median=" << cpuMedianP << " p90=" << cpuP90P
                  << "  (vs. baseline cpu_wall_ms median=" << cpuMedian << " -- baseline's cpu_wall_ms"
                  << " includes its waitUntilCompleted(), this doesn't)" << std::endl;
        std::cout << "[live-bench] cmdbufs/frame        mean=" << cmdbufMean
                  << "  waits/frame=0 (host waitUntilCompleted count; the 2-deep semaphore wait above"
                  << " is intentional backpressure, not counted)" << std::endl;

        // Correctness cross-check (the thing that actually matters here --
        // a broken prevCameraBuffer_ double-buffer, or the cmdBuf
        // use-after-free this file's own comment above describes hitting
        // first, would show up as WRONG pixels on some frames, not
        // necessarily a crash). Two brand-new MetalLiveReconstruct
        // instances, each with its own fresh from-frame-0 history (not the
        // benchmark runs' already-warmed-up state above) -- one driven
        // sequentially (wait-every-frame, frameInFlightIndex always 0, i.e.
        // today's already-shipped behavior), one driven the new pipelined
        // way (2 in flight, index alternating) -- over the IDENTICAL frame
        // range and camera/morph schedule. Metal compute is deterministic
        // for identical inputs and identical per-queue submission order (the
        // only thing overlap changes is how far ahead the CPU encodes, not
        // command execution order), so these should match closely; any real
        // desync between prevCameraBuffer_[0]/[1] would show up as a wrong
        // (not just noisy) frame's motion vectors, i.e. a localized spike in
        // recon_rgba_max_abs_diff, not a small uniform rmse.
        {
            const int kCheckFrames = 12;
            MetalLiveReconstruct liveSeq, livePipe;
            liveSeq.init(ctx, scene.getSplatData(), params);
            livePipe.init(ctx, scene.getSplatData(), params);

            std::vector<std::vector<float>> seqColor(static_cast<size_t>(kCheckFrames));
            for (int i = 0; i < kCheckFrames; ++i) {
                NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
                auto in = buildFrameInputs(i);
                auto* cmdBuf = ctx.beginCommandBuffer();
                auto* cur = liveSeq.encodeFrame(ctx, cmdBuf, in, 0);
                cur->commit();
                cur->waitUntilCompleted();
                std::vector<uint8_t> unusedRgba8;
                auto raw = MetalScreenshot::readTextureRaw(ctx, liveSeq.getReconOutputTexture(), liveSeq.getTargetW(),
                                                             liveSeq.getTargetH(), 8);
                seqColor[static_cast<size_t>(i)] =
                    DlssIO::convertRgba16fColorAttachment(raw, liveSeq.getTargetW(), liveSeq.getTargetH(), unusedRgba8);
                cur->release();
                pool->release();
            }

            dispatch_semaphore_t semC = dispatch_semaphore_create(2);
            std::vector<std::vector<float>> pipeColor(static_cast<size_t>(kCheckFrames));
            std::atomic<int> pipeCompleted{0};
            uint32_t checkTw = livePipe.getTargetW(), checkTh = livePipe.getTargetH();
            for (int i = 0; i < kCheckFrames; ++i) {
                NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
                dispatch_semaphore_wait(semC, DISPATCH_TIME_FOREVER);
                auto in = buildFrameInputs(i);
                auto* cmdBuf = ctx.beginCommandBuffer();
                cmdBuf->retain();  // see the cmdBuf use-after-free comment above.
                auto* cur = livePipe.encodeFrame(ctx, cmdBuf, in, static_cast<uint32_t>(i & 1));
                MTL::Texture* reconTex = livePipe.getReconOutputTexture();
                cur->addCompletedHandler([&pipeColor, &pipeCompleted, &semC, &ctx, cmdBuf, reconTex, checkTw,
                                           checkTh, i](MTL::CommandBuffer*) {
                    NS::AutoreleasePool* hpool = NS::AutoreleasePool::alloc()->init();
                    std::vector<uint8_t> unusedRgba8;
                    auto raw = MetalScreenshot::readTextureRaw(ctx, reconTex, checkTw, checkTh, 8);
                    pipeColor[static_cast<size_t>(i)] =
                        DlssIO::convertRgba16fColorAttachment(raw, checkTw, checkTh, unusedRgba8);
                    pipeCompleted.fetch_add(1, std::memory_order_relaxed);
                    cmdBuf->release();
                    hpool->release();
                    dispatch_semaphore_signal(semC);
                });
                cur->commit();
                cur->release();
                pool->release();
            }
            dispatch_semaphore_wait(semC, DISPATCH_TIME_FOREVER);
            dispatch_semaphore_wait(semC, DISPATCH_TIME_FOREVER);

            double maxAbsDiff = 0.0, sumSqDiff = 0.0;
            size_t n = 0;
            for (int i = 0; i < kCheckFrames; ++i) {
                if (seqColor[static_cast<size_t>(i)].size() != pipeColor[static_cast<size_t>(i)].size() ||
                    seqColor[static_cast<size_t>(i)].empty()) {
                    continue;
                }
                for (size_t k = 0; k < seqColor[static_cast<size_t>(i)].size(); ++k) {
                    double d = static_cast<double>(seqColor[static_cast<size_t>(i)][k]) -
                               static_cast<double>(pipeColor[static_cast<size_t>(i)][k]);
                    maxAbsDiff = std::max(maxAbsDiff, std::abs(d));
                    sumSqDiff += d * d;
                    n++;
                }
            }
            double rmse = n > 0 ? std::sqrt(sumSqDiff / static_cast<double>(n)) : -1.0;
            std::cout << "[live-bench] ===== CORRECTNESS CROSS-CHECK (sequential vs pipelined, " << kCheckFrames
                      << " frames, fresh history each) =====" << std::endl;
            std::cout << "[live-bench] recon_rgba max_abs_diff=" << maxAbsDiff << " rmse=" << rmse
                      << " (pipeCompleted=" << pipeCompleted.load() << "/" << kCheckFrames << ", compared "
                      << n << " values)" << std::endl;
        }

        return 0;
    }

    return 0;
}

// RGB-only PSNR (alpha ignored) -- matches playground_ios/Source/SplatView.mm's
// psnrRgb()/mobiledlss/scripts/eval_rollout.py::_psnr_frame's convention:
// mse = mean((pred-target)**2) over RGB, psnr = 10*log10(1/mse). `pred`/
// `target` are both [h,w,4] float32, un-premultiplied.
static float psnrRgb(const std::vector<float>& pred, const std::vector<float>& target, uint32_t w, uint32_t h) {
    double se = 0.0;
    size_t n = static_cast<size_t>(w) * h;
    for (size_t i = 0; i < n; i++) {
        for (int c = 0; c < 3; c++) {
            double d = static_cast<double>(pred[i * 4 + c]) - static_cast<double>(target[i * 4 + c]);
            se += d * d;
        }
    }
    double mse = se / static_cast<double>(n * 3);
    if (mse <= 0.0) return INFINITY;
    return static_cast<float>(10.0 * std::log10(1.0 / mse));
}

// --------------------------------------------------------------------------
// --live-dump support (runLivePsnrMetal only) -- see CLIOptions::liveDumpDir's
// comment for the flag contract.
// --------------------------------------------------------------------------

// Verbatim copy of playground_ios/Source/SplatView.mm's kUpscaleMSL
// upscale_proxy kernel -- SplatView.mm is READ-ONLY for this change (owned
// by another agent editing the shared live-reconstruct chain), so this is
// duplicated rather than shared, and MUST be kept byte-identical to that
// copy for --live-dump's bilinear_fFF.png to be a faithful reproduction of
// the app's own Bicubic/"Bilinear" display mode (same bilinear sampling +
// alpha un-premultiply + forced-opaque-alpha convention).
static const char* kLiveDumpUpscaleMSL = R"(
#include <metal_stdlib>
using namespace metal;
kernel void upscale_proxy(texture2d<float, access::read> src [[texture(0)]],
                           texture2d<float, access::write> dst [[texture(1)]],
                           constant uint& bilinear [[buffer(0)]],
                           constant float2& scale [[buffer(1)]],
                           uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    uint srcW = src.get_width(), srcH = src.get_height();
    float4 rgba;
    if (bilinear == 0) {
        uint sx = min(uint(float(gid.x) / scale.x), srcW - 1);
        uint sy = min(uint(float(gid.y) / scale.y), srcH - 1);
        rgba = src.read(uint2(sx, sy));
    } else {
        float fx = (float(gid.x) + 0.5) / scale.x - 0.5;
        float fy = (float(gid.y) + 0.5) / scale.y - 0.5;
        int x0 = int(floor(fx)); int y0 = int(floor(fy));
        float tx = fx - float(x0); float ty = fy - float(y0);
        int x0c = clamp(x0, 0, int(srcW) - 1); int x1c = clamp(x0 + 1, 0, int(srcW) - 1);
        int y0c = clamp(y0, 0, int(srcH) - 1); int y1c = clamp(y0 + 1, 0, int(srcH) - 1);
        float4 c00 = src.read(uint2(x0c, y0c));
        float4 c10 = src.read(uint2(x1c, y0c));
        float4 c01 = src.read(uint2(x0c, y1c));
        float4 c11 = src.read(uint2(x1c, y1c));
        rgba = mix(mix(c00, c10, tx), mix(c01, c11, tx), ty);
    }
    float a = rgba.a;
    float3 rgb = (a > 1e-6) ? rgba.rgb / a : float3(0.0);
    dst.write(float4(rgb, 1.0), gid);
}
)";

// Parses a --live-dump-frames spec ("0-11,60-71") into the set of frame
// indices to dump. Empty spec = every frame in [0, totalFrames).
static std::set<int> parseFrameSpec(const std::string& spec, int totalFrames) {
    std::set<int> out;
    if (spec.empty()) {
        for (int i = 0; i < totalFrames; ++i) out.insert(i);
        return out;
    }
    std::stringstream ss(spec);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.empty()) continue;
        auto dash = tok.find('-', 1);  // skip a leading '-' (not supported/expected, but avoid misparsing)
        if (dash == std::string::npos) {
            out.insert(std::stoi(tok));
        } else {
            int a = std::stoi(tok.substr(0, dash));
            int b = std::stoi(tok.substr(dash + 1));
            for (int i = a; i <= b; ++i) out.insert(i);
        }
    }
    return out;
}

// Reads an RGBA16Float texture back and returns its un-premultiplied
// float32 [h,w,4] RGBA (DlssIO::convertRgba16fColorAttachment's convention
// -- alpha itself is NOT forced/altered here, just rgb divided out of it).
static std::vector<float> readRgba16fAsFloat(MetalContext& ctx, MTL::Texture* tex, uint32_t w, uint32_t h) {
    auto raw = MetalScreenshot::readTextureRaw(ctx, tex, w, h, 8);
    std::vector<uint8_t> unusedRgba8;
    return DlssIO::convertRgba16fColorAttachment(raw, w, h, unusedRgba8);
}

// Writes un-premultiplied [h,w,4] float RGBA data as an 8-bit PNG with
// alpha FORCED to opaque (255) -- matches kLiveDumpUpscaleMSL's/SplatView.mm's
// own "force alpha=1.0 for opaque display" convention (the drawable is
// always shown opaque; a background pixel with true alpha~0 has
// rgb already zeroed by the un-premultiply's a>1e-6 branch, so forcing
// alpha=255 here reproduces exactly what the app puts on screen, not a
// half-transparent PNG that would misrepresent it).
static void writeFloatRgbaPng(const std::string& path, const std::vector<float>& rgba, uint32_t w, uint32_t h) {
    std::vector<uint8_t> out(static_cast<size_t>(w) * h * 4);
    size_t n = static_cast<size_t>(w) * h;
    for (size_t i = 0; i < n; ++i) {
        out[i * 4 + 0] = DlssIO::floatToUnorm8Rounded(rgba[i * 4 + 0]);
        out[i * 4 + 1] = DlssIO::floatToUnorm8Rounded(rgba[i * 4 + 1]);
        out[i * 4 + 2] = DlssIO::floatToUnorm8Rounded(rgba[i * 4 + 2]);
        out[i * 4 + 3] = 255;
    }
    if (!stbi_write_png(path.c_str(), static_cast<int>(w), static_cast<int>(h), 4, out.data(),
                         static_cast<int>(w) * 4)) {
        std::cerr << "[live-dump] failed to write " << path << std::endl;
    }
}

// |a-b| RGB preview, gain-boosted (default 4x) for visibility -- a
// well-converged reconstruction's raw per-pixel diff is small and would
// otherwise look almost entirely black; alpha forced opaque like
// writeFloatRgbaPng() above. `a`/`b` are both un-premultiplied [h,w,4]
// float RGBA (e.g. reconF32/targetF32, already computed for the PSNR math
// below -- this reuses them rather than re-reading the textures).
static void writeAbsDiffPng(const std::string& path, const std::vector<float>& a, const std::vector<float>& b,
                             uint32_t w, uint32_t h, float gain) {
    std::vector<uint8_t> out(static_cast<size_t>(w) * h * 4);
    size_t n = static_cast<size_t>(w) * h;
    for (size_t i = 0; i < n; ++i) {
        for (int c = 0; c < 3; ++c) {
            float d = std::fabs(a[i * 4 + c] - b[i * 4 + c]) * gain;
            out[i * 4 + c] = DlssIO::floatToUnorm8Rounded(d);
        }
        out[i * 4 + 3] = 255;
    }
    if (!stbi_write_png(path.c_str(), static_cast<int>(w), static_cast<int>(h), 4, out.data(),
                         static_cast<int>(w) * 4)) {
        std::cerr << "[live-dump] failed to write " << path << std::endl;
    }
}

// --------------------------------------------------------------------------
// --live-psnr N: N-frame continuous Reconstruction-vs-Target PSNR gate --
// the Mac-side counterpart of playground_ios/Source/SplatView.mm's
// LUX_PSNR_FRAMES on-device capture. Runs the SAME MetalLiveReconstruct
// chain --live-bench does (continuous from frame 0, no history reset) PLUS
// a second, separate full-res MetalSplatLuxcRenderer against Target mode's
// own UNPRUNED scene (see CLIOptions::liveTargetSceneSource's comment),
// and reports RGB PSNR of the reconstruction against that Target reference
// every frame -- self-contained, own MetalContext, like runLiveBenchMetal.
// --------------------------------------------------------------------------

static int runLivePsnrMetal(const CLIOptions& opts) {
    MetalContext ctx;
    ctx.initHeadless();

    std::string proxySceneSource =
        opts.liveSceneSource.empty() ? (opts.liveAssetsDir + "/juggle_p0.8_stride4.glb") : opts.liveSceneSource;
    std::string targetSceneSource = opts.liveTargetSceneSource.empty()
                                         ? (opts.liveAssetsDir + "/juggle_full_stride4.glb")
                                         : opts.liveTargetSceneSource;

    MetalSceneManager sceneProxy, sceneTarget;
    sceneProxy.loadScene(proxySceneSource);
    sceneTarget.loadScene(targetSceneSource);
    if (!sceneProxy.hasSplatData() || !sceneTarget.hasSplatData()) {
        std::cerr << "[live-psnr] scene has no KHR_gaussian_splatting data" << std::endl;
        return 1;
    }

    uint32_t proxyW, proxyH, targetW, targetH;
    resolveLiveResolutions(opts, proxyW, proxyH, targetW, targetH);

    MetalLiveReconstruct live;
    MetalLiveReconstruct::InitParams params;
    params.shaderBase = opts.livePipeline;
    params.proxyW = proxyW;
    params.proxyH = proxyH;
    params.targetW = targetW;
    params.targetH = targetH;
    params.paramStride = opts.liveParamStride;
    params.hiddenChannels = 8;
    const std::string liveExportedDir =
        opts.liveExportedDir.empty() ? (opts.liveAssetsDir + "/exported") : opts.liveExportedDir;
    params.textureNpyPath = liveExportedDir + "/texture.npy";
    params.bgSphereNpyPath = opts.liveAssetsDir + "/bg_sphere.npy";
    params.unetWeightsBinPath = liveExportedDir + "/unet_weights.fp16.bin";
    params.unetLayersTxtPath = liveExportedDir + "/unet_weights.layers.txt";
    params.memoryHeadNpzPath = liveExportedDir + "/memory_head.npz";
    // (4, 2.0f) -- perf; bench/lux_perf_ablation.md's "Motion-aware
    // schedule": this used to badly break reconstruction quality on this
    // DYNAMIC (actor-motion) scene -- periodic ~15dB collapses on 3 of
    // every 4 frames (17.8dB vs a stable ~33dB with every-frame sort) --
    // because the schedule's view-change threshold only re-sorted on
    // CAMERA rotation, with no signal for the scene content itself moving
    // (exactly what reorders back-to-front blend order here even with
    // zero camera motion). setSortSchedule() now ALSO re-sorts on any
    // currentMorphTime_ change (see its header comment), which fires
    // every frame this scene actually animates -- degenerating this
    // config back to "sort every frame" for this dynamic scene (safe,
    // matching the un-scheduled baseline's PSNR) while still allowing a
    // static-camera hold on a static scene to skip via the
    // rotation/frame-budget checks.
    params.sortEveryNFrames = 4;
    params.sortViewThresholdDeg = 2.0f;
    live.init(ctx, sceneProxy.getSplatData(), params);

    // Target mode: same luxc pipeline, full (unpruned) scene, full target
    // resolution, no jitter -- mirrors SplatView.mm's _splatRTarget exactly.
    // Same finding as above applies to Target's own display quality --
    // left un-scheduled (exactness-preserving default, no explicit
    // setSortSchedule() call) rather than reproducing the same bug.
    MetalSplatLuxcRenderer target;
    target.init(ctx, sceneTarget.getSplatData(), opts.livePipeline, targetW, targetH);

    std::cout << "[live-psnr] proxy splats=" << sceneProxy.getSplatData().num_splats
              << " target splats=" << sceneTarget.getSplatData().num_splats
              << " netRes=" << live.getNetW() << "x" << live.getNetH() << std::endl;

    // --live-dump setup: own upscale_proxy compute pipeline (kLiveDumpUpscaleMSL)
    // + a scratch target-res RGBA16Float texture for the bilinear preview.
    // Built once, up front, so the per-frame loop below only pays for a
    // dispatch + readback on the actually-selected dump frames.
    const bool dumping = !opts.liveDumpDir.empty();
    std::set<int> dumpFrameSet;
    MTL::ComputePipelineState* dumpUpscalePipeline = nullptr;
    MTL::Texture* dumpBilinearTex = nullptr;
    if (dumping) {
        std::error_code ec;
        fs::create_directories(opts.liveDumpDir, ec);
        dumpFrameSet = parseFrameSpec(opts.liveDumpFramesSpec, opts.livePsnrFrames);

        NS::Error* shaderErr = nullptr;
        auto* src = NS::String::string(kLiveDumpUpscaleMSL, NS::UTF8StringEncoding);
        auto* compileOpts = MTL::CompileOptions::alloc()->init();
        auto* lib = ctx.device->newLibrary(src, compileOpts, &shaderErr);
        compileOpts->release();
        if (!lib) {
            std::cerr << "[live-dump] upscale shader compile failed: "
                      << (shaderErr ? shaderErr->localizedDescription()->utf8String() : "?") << std::endl;
            return 1;
        }
        auto* fn = lib->newFunction(NS::String::string("upscale_proxy", NS::UTF8StringEncoding));
        dumpUpscalePipeline = ctx.device->newComputePipelineState(fn, &shaderErr);
        fn->release();
        lib->release();
        if (!dumpUpscalePipeline) {
            std::cerr << "[live-dump] failed to create upscale pipeline" << std::endl;
            return 1;
        }

        auto* texDesc =
            MTL::TextureDescriptor::texture2DDescriptor(MTL::PixelFormatRGBA16Float, targetW, targetH, false);
        texDesc->setUsage(MTL::TextureUsageShaderRead | MTL::TextureUsageShaderWrite);
        texDesc->setStorageMode(MTL::StorageModePrivate);
        dumpBilinearTex = ctx.newTexture(texDesc);

        std::cout << "[live-dump] enabled, dir=" << opts.liveDumpDir << " frames=" << dumpFrameSet.size()
                  << std::endl;
    }

    std::vector<float> psnrLog;
    for (int frame = 0; frame < opts.livePsnrFrames; ++frame) {
        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
        int prevFrame = frame > 0 ? frame - 1 : 0;
        auto toCamFrame = [](const LiveOrbitCamera::Result& r) {
            return MetalLiveReconstruct::CameraFrame{r.eye, r.r, r.u, r.f, r.viewGl, r.proj, r.fx, r.fy};
        };
        MetalLiveReconstruct::FrameInputs in;
        in.proxyCur = toCamFrame(LiveOrbitCamera::compute(frame, live.getProxyW(), live.getProxyH(),
                                                           MetalSplatLuxcRenderer::kMetalYConvention));
        in.proxyPrev = toCamFrame(LiveOrbitCamera::compute(prevFrame, live.getProxyW(), live.getProxyH(),
                                                            MetalSplatLuxcRenderer::kMetalYConvention));
        LiveOrbitCamera::Result targetCamRaw = LiveOrbitCamera::compute(
            frame, live.getTargetW(), live.getTargetH(), MetalSplatLuxcRenderer::kMetalYConvention);
        in.targetCur = toCamFrame(targetCamRaw);
        bool hasMotion = live.proxyRenderer().hasMotion();
        in.morphTimeCur = hasMotion ? live.proxyRenderer().frameToTime(frame) : 0.0f;
        in.morphTimePrev = hasMotion ? live.proxyRenderer().frameToTime(prevFrame) : 0.0f;
        float jxTarget, jyTarget;
        LiveOrbitCamera::taaJitterTargetPx(frame, /*period=*/16, jxTarget, jyTarget);
        float scaleP = static_cast<float>(live.getProxyW()) / static_cast<float>(live.getTargetW());
        in.jitterTargetX = jxTarget;
        in.jitterTargetY = jyTarget;
        in.jitterProxyX = jxTarget * scaleP;
        in.jitterProxyY = jyTarget * scaleP;

        auto* cmdBuf = ctx.beginCommandBuffer();
        auto* cur = live.encodeFrame(ctx, cmdBuf, in);
        cur->commit();
        cur->waitUntilCompleted();
        cur->release();

        // Target reference: ground truth, no jitter, own morph evaluation.
        target.updateCameraExplicit(targetCamRaw.eye, targetCamRaw.viewGl, targetCamRaw.proj, targetCamRaw.fx,
                                     targetCamRaw.fy);
        target.setJitter(0.0f, 0.0f);
        if (target.hasMotion()) target.setMorphTime(in.morphTimeCur);
        target.render(ctx);

        std::vector<uint8_t> unusedRgba8;
        auto reconRaw =
            MetalScreenshot::readTextureRaw(ctx, live.getReconOutputTexture(), live.getTargetW(), live.getTargetH(), 8);
        auto reconF32 = DlssIO::convertRgba16fColorAttachment(reconRaw, live.getTargetW(), live.getTargetH(), unusedRgba8);
        auto targetRaw =
            MetalScreenshot::readTextureRaw(ctx, target.getOutputTexture(), live.getTargetW(), live.getTargetH(), 8);
        auto targetF32 = DlssIO::convertRgba16fColorAttachment(targetRaw, live.getTargetW(), live.getTargetH(), unusedRgba8);

        if (dumping && dumpFrameSet.count(frame)) {
            char suffix[8];
            std::snprintf(suffix, sizeof(suffix), "%02d", frame);
            const std::string base = opts.liveDumpDir + "/";

            // proxy_fFF.png: native 480x270 proxy colour, un-premultiplied --
            // the Proxy display mode just nearest-neighbor-upscales these
            // same pixels 2x to the drawable (kLiveDumpUpscaleMSL's
            // bilinear==0 branch), so this is what that mode's content
            // actually is before that upscale.
            auto proxyF32 = readRgba16fAsFloat(ctx, live.proxyRenderer().getOutputTexture(), live.getProxyW(),
                                                live.getProxyH());
            writeFloatRgbaPng(base + "proxy_f" + suffix + ".png", proxyF32, live.getProxyW(), live.getProxyH());

            // bilinear_fFF.png: actual GPU dispatch of the same kernel/params
            // (bilinear=1, scale=target/proxy) SplatView.mm's Bicubic
            // ("Bilinear") display mode uses -- bit-exact with the app, not
            // a CPU approximation.
            {
                auto* dcmd = ctx.beginCommandBuffer();
                auto* enc = dcmd->computeCommandEncoder();
                enc->setComputePipelineState(dumpUpscalePipeline);
                enc->setTexture(live.proxyRenderer().getOutputTexture(), 0);
                enc->setTexture(dumpBilinearTex, 1);
                uint32_t bilinear = 1;
                std::array<float, 2> scale = {
                    static_cast<float>(live.getTargetW()) / static_cast<float>(live.getProxyW()),
                    static_cast<float>(live.getTargetH()) / static_cast<float>(live.getProxyH())};
                enc->setBytes(&bilinear, sizeof(bilinear), 0);
                enc->setBytes(scale.data(), sizeof(float) * 2, 1);
                MTL::Size grid(live.getTargetW(), live.getTargetH(), 1);
                NS::UInteger tew = dumpUpscalePipeline->threadExecutionWidth();
                NS::UInteger maxT = dumpUpscalePipeline->maxTotalThreadsPerThreadgroup();
                NS::UInteger th = maxT / tew;
                if (th == 0) th = 1;
                MTL::Size tg(tew, th, 1);
                enc->dispatchThreads(grid, tg);
                enc->endEncoding();
                ctx.submitAndWait(dcmd);
            }
            auto bilinearF32 = readRgba16fAsFloat(ctx, dumpBilinearTex, live.getTargetW(), live.getTargetH());
            writeFloatRgbaPng(base + "bilinear_f" + suffix + ".png", bilinearF32, live.getTargetW(),
                               live.getTargetH());

            // recon_fFF.png / target_fFF.png: reuse the PSNR math's own
            // readbacks (reconF32/targetF32) -- already exactly what
            // Reconstruction mode blits to the drawable / what Target mode
            // (unpruned scene) renders.
            writeFloatRgbaPng(base + "recon_f" + suffix + ".png", reconF32, live.getTargetW(), live.getTargetH());
            writeFloatRgbaPng(base + "target_f" + suffix + ".png", targetF32, live.getTargetW(), live.getTargetH());
            writeAbsDiffPng(base + "absdiff_recon_target_f" + suffix + ".png", reconF32, targetF32,
                             live.getTargetW(), live.getTargetH(), /*gain=*/4.0f);

            std::cout << "[live-dump] frame=" << frame << " wrote 5 PNGs" << std::endl;
        }

        float psnr = psnrRgb(reconF32, targetF32, live.getTargetW(), live.getTargetH());
        psnrLog.push_back(psnr);
        std::cout << "[live-psnr] frame=" << (frame + 1) << "/" << opts.livePsnrFrames << " psnr_recon_db=" << psnr
                  << std::endl;
        pool->release();
    }

    if (!psnrLog.empty()) {
        double sum = 0.0;
        for (float v : psnrLog) sum += v;
        std::cout << "[live-psnr] COMPLETE. mean=" << (sum / psnrLog.size()) << "dB last=" << psnrLog.back()
                  << "dB (" << psnrLog.size() << " frames)" << std::endl;
    }

    return 0;
}

// --------------------------------------------------------------------------
// --live-interactive: a real GLFW/Metal window running the SAME shared
// MetalLiveReconstruct chain --live-bench/--live-psnr do, live, with the
// exact 4 display modes playground_ios/Source/SplatView.mm's DisplayMode
// enum has (Proxy/Bilinear("Bicubic")/Reconstruction/Target) -- keys 1-4
// select a mode directly, space cycles, so the visual bugs a device would
// show can be inspected on a Mac with nobody's iPad attached. Self-contained
// (own GLFW window/MetalContext), like runLiveBenchMetal/runLivePsnrMetal.
//
// Display-mode parity with the app: Reconstruction blits
// live.getReconOutputTexture() straight to the drawable (SplatView.mm's own
// "already target-res, already straight non-premultiplied RGB" shortcut);
// Proxy/Bilinear/Target all go through kLiveDumpUpscaleMSL (byte-identical
// to SplatView.mm's kUpscaleMSL -- see its own comment above), same
// scale/bilinear-flag convention (scale=1 nearest for Target's identity
// un-premultiply "copy", scale=target/proxy for Proxy(nearest)/
// Bilinear(bilinear=1)).
//
// --live-shot <DIR>: auto-cycles all 4 modes (2s each) and captures one PNG
// of the ACTUAL drawable content per mode (not an offline re-render --
// exercises the identical blit/upscale-kernel dispatch a human sitting at
// the window would see), then exits -- see the capture block below.
// --------------------------------------------------------------------------

static const char* kLiveInteractiveModeNames[4] = {"Proxy", "Bilinear", "Reconstruction", "Target"};
static const char* kLiveInteractiveModeFiles[4] = {"proxy", "bilinear", "recon", "target"};

static int runLiveInteractiveMetal(CLIOptions opts) {
    uint32_t proxyW, proxyH, targetW, targetH;
    resolveLiveResolutions(opts, proxyW, proxyH, targetW, targetH);

    std::string proxySceneSource =
        opts.liveSceneSource.empty() ? (opts.liveAssetsDir + "/juggle_p0.8_stride4.glb") : opts.liveSceneSource;
    std::string targetSceneSource = opts.liveTargetSceneSource.empty()
                                         ? (opts.liveAssetsDir + "/juggle_full_stride4.glb")
                                         : opts.liveTargetSceneSource;

    MetalSceneManager sceneProxy, sceneTarget;
    sceneProxy.loadScene(proxySceneSource);
    sceneTarget.loadScene(targetSceneSource);
    if (!sceneProxy.hasSplatData() || !sceneTarget.hasSplatData()) {
        std::cerr << "[live-interactive] scene has no KHR_gaussian_splatting data" << std::endl;
        return 1;
    }

    if (!glfwInit()) {
        std::cerr << "[error] Failed to initialize GLFW" << std::endl;
        return 1;
    }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    // Keep glfwGetFramebufferSize() == the window's point size (no implicit
    // 2x on Retina): the live chain's textures (recon/proxy/target) are all
    // fixed at proxyW/H x targetW/H from init() below, and the Reconstruction
    // display path is a raw size-matched blit -- a HiDPI-scaled framebuffer
    // would silently only fill the drawable's top-left corner.
    glfwWindowHint(GLFW_COCOA_RETINA_FRAMEBUFFER, GLFW_FALSE);

    GLFWwindow* window = glfwCreateWindow(static_cast<int>(targetW), static_cast<int>(targetH),
                                           "Lux Live (Metal)", nullptr, nullptr);
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
    // Required for the Proxy/Bilinear/Target display path below (a compute
    // kernel WRITES directly into drawable->texture(), and --live-shot
    // blit-reads it back) -- CAMetalLayer.framebufferOnly defaults to true,
    // which restricts its drawables to framebuffer-only (render-pass) use;
    // a compute-kernel access::write (or a blit read) into that texture is
    // silently invalid without this. Exactly the same requirement/fix
    // SplatView.mm's own metalLayer.framebufferOnly = NO already documents
    // for the identical reason.
    ctx.metalLayer->setFramebufferOnly(false);
    // Required for Reconstruction mode's display path: MetalContext::init()
    // leaves the GLFW window's CAMetalLayer at its default BGRA8Unorm
    // (8-bit) pixel format, but live.getReconOutputTexture() is
    // RGBA16Float (16-bit/channel) -- a raw blitCommandEncoder copy between
    // mismatched-bit-depth pixel formats reinterprets bytes rather than
    // converting them, producing corrupted/tiled garbage (reproduced:
    // without this line, recon.png came out as scrambled magenta noise).
    // SplatView.mm sets `metalLayer.pixelFormat = MTLPixelFormatRGBA16Float`
    // for exactly this reason (its own "already target-res RGBA16Float --
    // straight blit into the (same-format) drawable" comment) -- match it
    // here so the blit is actually same-format, like the app's.
    ctx.metalLayer->setPixelFormat(MTL::PixelFormatRGBA16Float);

    MetalLiveReconstruct live;
    MetalLiveReconstruct::InitParams params;
    params.shaderBase = opts.livePipeline;
    params.proxyW = proxyW;
    params.proxyH = proxyH;
    params.targetW = targetW;
    params.targetH = targetH;
    params.paramStride = opts.liveParamStride;
    params.hiddenChannels = 8;
    const std::string liveExportedDir =
        opts.liveExportedDir.empty() ? (opts.liveAssetsDir + "/exported") : opts.liveExportedDir;
    params.textureNpyPath = liveExportedDir + "/texture.npy";
    params.bgSphereNpyPath = opts.liveAssetsDir + "/bg_sphere.npy";
    params.unetWeightsBinPath = liveExportedDir + "/unet_weights.fp16.bin";
    params.unetLayersTxtPath = liveExportedDir + "/unet_weights.layers.txt";
    params.memoryHeadNpzPath = liveExportedDir + "/memory_head.npz";
    params.sortEveryNFrames = 4;  // (perf) -- see runLiveBenchMetal's identical comment.
    params.sortViewThresholdDeg = 2.0f;
    live.init(ctx, sceneProxy.getSplatData(), params);

    // Target mode: its own second luxc renderer against the unpruned scene,
    // full target resolution, no jitter -- mirrors SplatView.mm's
    // _splatRTarget exactly.
    MetalSplatLuxcRenderer target;
    target.init(ctx, sceneTarget.getSplatData(), opts.livePipeline, targetW, targetH);

    int loopFrames = 480;  // SplatView.mm's own fallback before hasMotion() is known.
    if (target.hasMotion()) {
        loopFrames = std::max(1, static_cast<int>(std::round(target.animationDuration() * 30.0f)));  // kSimFps
    }

    // upscale_proxy compute pipeline (Proxy/Bilinear/Target display path) --
    // see kLiveDumpUpscaleMSL's own comment.
    MTL::ComputePipelineState* upscalePipeline = nullptr;
    {
        NS::Error* shaderErr = nullptr;
        auto* src = NS::String::string(kLiveDumpUpscaleMSL, NS::UTF8StringEncoding);
        auto* compileOpts = MTL::CompileOptions::alloc()->init();
        auto* lib = ctx.device->newLibrary(src, compileOpts, &shaderErr);
        compileOpts->release();
        if (!lib) {
            std::cerr << "[live-interactive] upscale shader compile failed: "
                      << (shaderErr ? shaderErr->localizedDescription()->utf8String() : "?") << std::endl;
            return 1;
        }
        auto* fn = lib->newFunction(NS::String::string("upscale_proxy", NS::UTF8StringEncoding));
        upscalePipeline = ctx.device->newComputePipelineState(fn, &shaderErr);
        fn->release();
        lib->release();
        if (!upscalePipeline) {
            std::cerr << "[live-interactive] failed to create upscale pipeline" << std::endl;
            return 1;
        }
    }

    const bool autoShot = !opts.liveShotDir.empty();
    if (autoShot) {
        std::error_code ec;
        fs::create_directories(opts.liveShotDir, ec);
    }

    // Mode indices match SplatView.mm's DisplayMode enum: 0=Proxy,
    // 1=Bilinear("Bicubic"), 2=Reconstruction, 3=Target. App default is
    // Reconstruction; --live-shot always starts at Proxy (mode 0) so the
    // 4-shot cycle below is deterministic regardless.
    int mode = autoShot ? 0 : 2;
    int lastMode = mode;
    int frame = 0;
    bool key1Prev = false, key2Prev = false, key3Prev = false, key4Prev = false, spacePrev = false;
    double modeStartTime = glfwGetTime();
    bool capturedThisMode = false;

    double fpsWindowStart = glfwGetTime();
    int fpsFrameCount = 0;
    double fps = 0.0;
    double lastGpuMs = 0.0;
    double lastTitleUpdate = 0.0;

    std::cout << "[live-interactive] ready: proxy=" << proxyW << "x" << proxyH << " target=" << targetW << "x"
              << targetH << " netRes=" << live.getNetW() << "x" << live.getNetH() << " loopFrames=" << loopFrames
              << (autoShot ? "  (--live-shot: auto-cycling, will exit after 4 modes)" : "") << std::endl;
    std::cout << "[live-interactive] keys: 1=Proxy 2=Bilinear 3=Reconstruction 4=Target space=cycle ESC=quit"
              << std::endl;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
            continue;
        }

        if (!autoShot) {
            bool k1 = glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS;
            bool k2 = glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS;
            bool k3 = glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS;
            bool k4 = glfwGetKey(window, GLFW_KEY_4) == GLFW_PRESS;
            bool sp = glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS;
            if (k1 && !key1Prev) mode = 0;
            if (k2 && !key2Prev) mode = 1;
            if (k3 && !key3Prev) mode = 2;
            if (k4 && !key4Prev) mode = 3;
            if (sp && !spacePrev) mode = (mode + 1) % 4;
            key1Prev = k1;
            key2Prev = k2;
            key3Prev = k3;
            key4Prev = k4;
            spacePrev = sp;
        }

        int fbW, fbH;
        glfwGetFramebufferSize(window, &fbW, &fbH);
        if (fbW == 0 || fbH == 0) {
            glfwWaitEvents();
            continue;
        }
        updateDrawableSize(ctx.metalLayer, window);

        // Mirrors SplatView.mm's tick: -- entering Reconstruction from any
        // other mode resets the recurrent history (stale otherwise, by
        // however many frames were spent displaying a mode that doesn't run
        // the chain).
        bool enteringRecon = (mode == 2 && lastMode != 2);
        if (enteringRecon) live.resetHistory();
        lastMode = mode;

        int prevFrame = frame > 0 ? frame - 1 : 0;
        bool hasMotion = live.proxyRenderer().hasMotion();
        float tCur = hasMotion ? live.proxyRenderer().frameToTime(frame) : 0.0f;
        float tPrev = hasMotion ? live.proxyRenderer().frameToTime(prevFrame) : 0.0f;
        float jxTarget, jyTarget;
        LiveOrbitCamera::taaJitterTargetPx(frame, /*period=*/16, jxTarget, jyTarget);
        float scaleP = static_cast<float>(proxyW) / static_cast<float>(targetW);
        float jxProxy = jxTarget * scaleP, jyProxy = jyTarget * scaleP;

        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

        MTL::Texture* displaySrc = nullptr;  // Proxy/Bilinear/Target path
        bool useReconBlit = false;           // Reconstruction path
        double gpuMs = 0.0;

        if (mode == 3) {  // Target
            LiveOrbitCamera::Result cam =
                LiveOrbitCamera::compute(frame, targetW, targetH, MetalSplatLuxcRenderer::kMetalYConvention);
            target.updateCameraExplicit(cam.eye, cam.viewGl, cam.proj, cam.fx, cam.fy);
            target.setJitter(0.0f, 0.0f);
            if (target.hasMotion()) target.setMorphTime(tCur);
            target.render(ctx);
            gpuMs = target.getLastGpuTotalMs();
            displaySrc = target.getOutputTexture();
        } else if (mode == 0 || mode == 1) {  // Proxy / Bilinear
            LiveOrbitCamera::Result cam =
                LiveOrbitCamera::compute(frame, proxyW, proxyH, MetalSplatLuxcRenderer::kMetalYConvention);
            live.proxyRenderer().updateCameraExplicit(cam.eye, cam.viewGl, cam.proj, cam.fx, cam.fy);
            live.proxyRenderer().setJitter(jxProxy, jyProxy);
            if (hasMotion) live.proxyRenderer().setMorphTime(tCur);
            live.proxyRenderer().render(ctx);
            gpuMs = live.proxyRenderer().getLastGpuTotalMs();
            displaySrc = live.proxyRenderer().getOutputTexture();
        } else {  // Reconstruction
            auto toCamFrame = [](const LiveOrbitCamera::Result& r) {
                return MetalLiveReconstruct::CameraFrame{r.eye, r.r, r.u, r.f, r.viewGl, r.proj, r.fx, r.fy};
            };
            MetalLiveReconstruct::FrameInputs in;
            in.proxyCur = toCamFrame(
                LiveOrbitCamera::compute(frame, proxyW, proxyH, MetalSplatLuxcRenderer::kMetalYConvention));
            in.proxyPrev = toCamFrame(
                LiveOrbitCamera::compute(prevFrame, proxyW, proxyH, MetalSplatLuxcRenderer::kMetalYConvention));
            in.targetCur = toCamFrame(
                LiveOrbitCamera::compute(frame, targetW, targetH, MetalSplatLuxcRenderer::kMetalYConvention));
            in.morphTimeCur = tCur;
            in.morphTimePrev = tPrev;
            in.jitterProxyX = jxProxy;
            in.jitterProxyY = jyProxy;
            in.jitterTargetX = jxTarget;
            in.jitterTargetY = jyTarget;

            auto* cmdBuf = ctx.beginCommandBuffer();
            auto* cur = live.encodeFrame(ctx, cmdBuf, in);
            cur->commit();
            cur->waitUntilCompleted();
            gpuMs = MetalLiveReconstruct::gpuMsAcrossPossibleSplit(cmdBuf, cur);
            cur->release();
            useReconBlit = true;
        }

        CA::MetalDrawable* drawable = ctx.metalLayer->nextDrawable();
        if (drawable) {
            auto* dcmd = ctx.beginCommandBuffer();
            if (useReconBlit) {
                auto* blit = dcmd->blitCommandEncoder();
                blit->copyFromTexture(live.getReconOutputTexture(), 0, 0, MTL::Origin(0, 0, 0),
                                       MTL::Size(targetW, targetH, 1), drawable->texture(), 0, 0,
                                       MTL::Origin(0, 0, 0));
                blit->endEncoding();
            } else {
                float scaleXY =
                    (mode == 3) ? 1.0f : static_cast<float>(targetW) / static_cast<float>(proxyW);
                std::array<float, 2> scale = {scaleXY, scaleXY};
                uint32_t bilinear = (mode == 1) ? 1 : 0;
                auto* enc = dcmd->computeCommandEncoder();
                enc->setComputePipelineState(upscalePipeline);
                enc->setTexture(displaySrc, 0);
                enc->setTexture(drawable->texture(), 1);
                enc->setBytes(&bilinear, sizeof(bilinear), 0);
                enc->setBytes(scale.data(), sizeof(float) * 2, 1);
                MTL::Size grid(targetW, targetH, 1);
                NS::UInteger tew = upscalePipeline->threadExecutionWidth();
                NS::UInteger maxT = upscalePipeline->maxTotalThreadsPerThreadgroup();
                NS::UInteger th = maxT / tew;
                if (th == 0) th = 1;
                MTL::Size tg(tew, th, 1);
                enc->dispatchThreads(grid, tg);
                enc->endEncoding();
            }
            dcmd->presentDrawable(drawable);
            dcmd->commit();
            dcmd->waitUntilCompleted();

            fpsFrameCount++;
            double now = glfwGetTime();
            if (now - fpsWindowStart >= 0.5) {
                fps = fpsFrameCount / (now - fpsWindowStart);
                fpsFrameCount = 0;
                fpsWindowStart = now;
            }
            lastGpuMs = gpuMs;
            if (now - lastTitleUpdate >= 0.1) {
                lastTitleUpdate = now;
                char title[256];
                std::snprintf(title, sizeof(title), "Lux Live [%s] fps=%.1f gpu=%.2fms frame=%d",
                              kLiveInteractiveModeNames[mode], fps, lastGpuMs, frame);
                glfwSetWindowTitle(window, title);
            }

            if (autoShot) {
                double elapsedInMode = now - modeStartTime;
                if (!capturedThisMode && elapsedInMode >= 1.9) {
                    std::string path =
                        opts.liveShotDir + "/" + kLiveInteractiveModeFiles[mode] + ".png";
                    try {
                        MetalScreenshot::saveTextureToPNG(ctx, drawable->texture(), targetW, targetH, path);
                        std::cout << "[live-shot] mode=" << kLiveInteractiveModeNames[mode] << " -> " << path
                                  << std::endl;
                    } catch (const std::exception& e) {
                        std::cerr << "[live-shot] capture failed for mode " << kLiveInteractiveModeNames[mode]
                                  << ": " << e.what() << std::endl;
                    }
                    capturedThisMode = true;
                }
                if (elapsedInMode >= 2.0) {
                    mode++;
                    modeStartTime = now;
                    capturedThisMode = false;
                    if (mode >= 4) {
                        glfwSetWindowShouldClose(window, GLFW_TRUE);
                    }
                }
            }
        }
        pool->release();

        frame = (frame + 1) % loopFrames;
    }

    ctx.cleanup();
    glfwDestroyWindow(window);
    glfwTerminate();
    std::cout << "[live-interactive] done." << std::endl;
    return 0;
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
    if (opts.colorFormat8Bit) {
        if constexpr (std::is_same_v<Renderer, MetalSplatLuxcRenderer>) {
            splatR->setColorFormat8BitExperiment(true);
        } else {
            std::cerr << "[warn] --color-format-8bit requires --splat-backend luxc; ignored" << std::endl;
        }
    }
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
    } else {
        // Regression fix: headless one-shot splat renders with no
        // --camera-json previously left the camera at whatever the
        // renderer's own default-constructed view/projection matrices
        // were (MetalSplatLuxcRenderer: identity glm::mat4 for both --
        // see its header's viewMatrix_/projMatrix_ member defaults) --
        // an identity view+projection does NOT frame the scene, so the
        // splat quads project outside the NDC cube and get frustum-culled
        // entirely, producing a blank (all-zero) image. --bench (and the
        // interactive GLFW path) both already set a real camera from the
        // scene's own auto-computed bounding-box eye/target/up/far before
        // their first render() call, which is why this gap was invisible
        // there; --camera-json also fixed it for callers that pass one
        // (e.g. mobiledlss's verify_lux_vs_gsplat.py). Plain
        // `--pipeline examples/gaussian_splat_dlss --frame N` with no
        // other camera flag had no fallback at all. Match Vulkan's
        // main.cpp (syncSplatCamera lambda / SplatRenderer's own internal
        // splat-bounding-box default) by applying the same scene auto-
        // camera used by --bench/interactive here too.
        float aspect = static_cast<float>(opts.width) / static_cast<float>(opts.height);
        splatR->updateCamera(scene.getAutoEye(), scene.getAutoTarget(), scene.getAutoUp(),
                              glm::radians(45.0f), aspect, 0.1f, scene.getAutoFar());
        std::cout << "[metal] No --camera-json given: using scene auto-camera "
                  << "(eye=" << scene.getAutoEye().x << "," << scene.getAutoEye().y << ","
                  << scene.getAutoEye().z << ")" << std::endl;
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

    if (opts.liveBenchFrames > 0) {
        return runLiveBenchMetal(opts);
    }

    if (opts.livePsnrFrames > 0) {
        return runLivePsnrMetal(opts);
    }

    if (opts.liveInteractive) {
        return runLiveInteractiveMetal(opts);
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
