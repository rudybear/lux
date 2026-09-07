// Stage A of the mobile-DLSS Android live-rendering demo: NativeActivity
// entry point that reuses playground_cpp's Vulkan splat pipeline (by
// reference -- SceneManager/SplatRenderer/gltf_loader/camera/dlss_io/
// spv_loader are compiled unmodified from ../../../lux-4dgs/playground_cpp/src)
// to live-render the pruned PanopticSports juggle scene with a synthetic
// orbit camera, mirroring mobiledlss/datagen/camera.py::orbit_path /
// look_at exactly (see buildOrbitEyeAndView below) -- the same recipe
// playground_ios/Source/SplatView.h validated on iPad (docs/rendering-engines.md
// "Stage A of the mobile-DLSS live demo").
//
// Android specifics vs. the desktop/iOS ports:
//  - No GLFW: android_vulkan_context.{h,cpp} (this directory) fills in the
//    shared VulkanContext struct via VK_KHR_android_surface + ANativeWindow.
//  - No argv: paths (scene .glb, examples/*.spv, shaders/radix_sort/*.spv)
//    are resolved relative to the app's external files dir, which the app
//    chdir()s into at startup; build.sh's `adb push` step lays out the
//    same relative directory structure the desktop CLI expects.

// NOTE: VMA_IMPLEMENTATION lives in android_vulkan_context.cpp, and
// CGLTF_IMPLEMENTATION / STB_IMAGE_IMPLEMENTATION already live in
// playground_cpp/src/gltf_loader.cpp (reused unmodified) -- defining any of
// them again here would duplicate symbols at link time.

#include <android_native_app_glue.h>
#include <android/log.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "vulkan_context.h"
#include "scene_manager.h"
#include "splat_renderer.h"
#include "dlss_io.h"
#include "android_vulkan_context.h"
#include "input_assembly.h"
#include "net_runner.h"
#include "reconstruct_pass.h"
#include "gpu_probe.h"

#define LOG_TAG "lux_android"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

// Display resolution for Stage A (task spec: 960x540 display target). Stage
// 2 (docs/rendering-engines.md "Android live-rendering demo") adds a second,
// quarter-area "proxy" resolution -- exactly half linear (480x270 = 960x540
// / 2) -- rendered by an independent SplatRenderer instance with TAA jitter;
// kWidth/kHeight above remain the unjittered "Target" (full-res) pass.
constexpr uint32_t kWidth = 960;
constexpr uint32_t kHeight = 540;
constexpr uint32_t kProxyWidth = 480;
constexpr uint32_t kProxyHeight = 270;

// Which image gets blitted to the swapchain this frame. Stage 6
// (docs/rendering-engines.md): tap-to-cycle Proxy -> Bicubic ->
// Reconstruction -> Target -> Proxy (any ACTION_UP touch event advances
// one step, see onInputEvent()). Proxy/Bicubic/Reconstruction all drive
// the SAME jittered proxyRenderer_ + InputAssembly + NetRunner pipeline
// (see renderFrame()'s `usesProxyRenderer`) and differ only in what's
// displayed: Proxy blits the raw 480x270 proxy colour (hardware-upscaled
// by the swapchain blit's own VK_FILTER_LINEAR); Bicubic does a real
// Catmull-Rom bicubic upsample to target res on the CPU (bicubicUpsample2x)
// -- the classic non-ML upsample baseline DLSS-style demos compare
// against; Reconstruction runs the full net + reconstruct-with-memory
// pass per displayed frame (via reconstruct_pass.h's runReconstructDump,
// reused from Stage 5, over a fresh 1-frame dump directory each call --
// simple and correct, at real disk-I/O + full-GPU-drain cost per stage,
// not latency-optimized here). Target renders the full-res unjittered
// scene directly, independent of the proxy pipeline entirely.
enum class DemoMode { Proxy, Bicubic, Reconstruction, Target };

DemoMode nextDemoMode(DemoMode m) {
    switch (m) {
        case DemoMode::Proxy: return DemoMode::Bicubic;
        case DemoMode::Bicubic: return DemoMode::Reconstruction;
        case DemoMode::Reconstruction: return DemoMode::Target;
        case DemoMode::Target: return DemoMode::Proxy;
    }
    return DemoMode::Proxy;
}

const char* demoModeName(DemoMode m) {
    switch (m) {
        case DemoMode::Proxy: return "PROXY";
        case DemoMode::Bicubic: return "BICUBIC";
        case DemoMode::Reconstruction: return "RECONSTRUCTION";
        case DemoMode::Target: return "TARGET";
    }
    return "?";
}

// Stage 6: real Catmull-Rom bicubic upsample (the classic non-ML baseline),
// CPU-side -- `src` is [srcH,srcW,3] (readColorRgb's output convention),
// clamped at the borders. Not restricted to an exact integer scale factor
// (dstW/dstH need not be 2x srcW/srcH) even though this demo only ever
// calls it at exactly 2x (proxy 480x270 -> target 960x540).
std::vector<float> bicubicUpsample2x(const std::vector<float>& src, uint32_t srcW, uint32_t srcH,
                                      uint32_t dstW, uint32_t dstH) {
    auto catmullRom = [](float p0, float p1, float p2, float p3, float t) {
        float t2 = t * t, t3 = t2 * t;
        return 0.5f * ((2.0f * p1) + (-p0 + p2) * t + (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3) * t2 +
                       (-p0 + 3.0f * p1 - 3.0f * p2 + p3) * t3);
    };
    auto tap = [&](int x, int y, int c) -> float {
        x = std::clamp(x, 0, static_cast<int>(srcW) - 1);
        y = std::clamp(y, 0, static_cast<int>(srcH) - 1);
        return src[(static_cast<size_t>(y) * srcW + x) * 3 + c];
    };
    std::vector<float> out(static_cast<size_t>(dstW) * dstH * 3);
    float sx = static_cast<float>(srcW) / static_cast<float>(dstW);
    float sy = static_cast<float>(srcH) / static_cast<float>(dstH);
    for (uint32_t dy = 0; dy < dstH; dy++) {
        float fy = (dy + 0.5f) * sy - 0.5f;
        int y1 = static_cast<int>(std::floor(fy));
        float ty = fy - y1;
        for (uint32_t dx = 0; dx < dstW; dx++) {
            float fx = (dx + 0.5f) * sx - 0.5f;
            int x1 = static_cast<int>(std::floor(fx));
            float tx = fx - x1;
            for (int c = 0; c < 3; c++) {
                float rows[4];
                for (int r = -1; r <= 2; r++) {
                    rows[r + 1] = catmullRom(tap(x1 - 1, y1 + r, c), tap(x1, y1 + r, c),
                                              tap(x1 + 1, y1 + r, c), tap(x1 + 2, y1 + r, c), tx);
                }
                float v = catmullRom(rows[0], rows[1], rows[2], rows[3], ty);
                out[(static_cast<size_t>(dy) * dstW + dx) * 3 + c] = std::clamp(v, 0.0f, 1.0f);
            }
        }
    }
    return out;
}

// Halton(2,3) TAA jitter, exactly mirroring mobiledlss/datagen/camera.py::
// halton/taa_jitter: 1-indexed low-discrepancy sequence, period 16, output
// centred in [-0.5, +0.5]. docs/lux-reconstruct-spec.md's input-contract
// table defines this jitter in TARGET pixels; the proxy pass (this stage)
// scales it by (proxyWidth/targetWidth) == 0.5 here before feeding it to
// SplatRenderer::setJitter(), which expects pixels in the renderer's OWN
// (proxy) resolution -- "0.5*jitter in proxy px" per the task brief.
float haltonSequence(int index, int base) {
    float f = 1.0f, r = 0.0f;
    int i = index;
    while (i > 0) {
        f /= static_cast<float>(base);
        r += f * static_cast<float>(i % base);
        i /= base;
    }
    return r;
}

constexpr int kJitterPeriod = 16;

glm::vec2 taaJitterTargetPixels(int frameIndex) {
    int i = (frameIndex % kJitterPeriod) + 1;
    return glm::vec2(haltonSequence(i, 2) - 0.5f, haltonSequence(i, 3) - 0.5f);
}

// Orbit parameters, matching mobiledlss/datagen/camera.py::orbit_path exactly:
// radius 2.0, elevation 15deg, 0.6deg/frame, PanopticSports y-down world (up=(0,-1,0)).
constexpr float kOrbitRadius = 2.0f;
constexpr float kElevationDeg = 15.0f;
constexpr float kDegPerFrame = 0.6f;
constexpr float kFovYDeg = 50.0f;

// Builds the OpenCV-convention world->camera view matrix exactly like
// mobiledlss/datagen/camera.py::look_at (x right, y down, z forward; world
// up = (0,-1,0) for the y-down PanopticSports scene), row-major for
// DlssIO::cvViewToGl (docs/lux-4d-spec.md section 4 camera bridge -- the
// same path main.cpp's --camera-json flag drives).
std::array<float, 16> buildCvViewRowMajor(glm::vec3 eye, glm::vec3 target, glm::vec3 upWorld) {
    glm::vec3 f = glm::normalize(target - eye);
    glm::vec3 r = glm::normalize(glm::cross(-upWorld, f));  // right = down x forward
    glm::vec3 u = glm::cross(f, r);                          // camera y = forward x right
    // R rows are camera axes in world coords: [r; u; f]
    std::array<float, 16> m{};
    m[0] = r.x; m[1] = r.y; m[2] = r.z;  m[3] = -glm::dot(r, eye);
    m[4] = u.x; m[5] = u.y; m[6] = u.z;  m[7] = -glm::dot(u, eye);
    m[8] = f.x; m[9] = f.y; m[10] = f.z; m[11] = -glm::dot(f, eye);
    m[12] = 0;  m[13] = 0;  m[14] = 0;   m[15] = 1;
    return m;
}

struct OrbitFrame {
    glm::vec3 eye;
    glm::mat4 viewGl;
    glm::mat4 proj;
    float fx, fy;
    // World-space camera basis (right, up/down, forward) -- Stage 3's input
    // assembly needs these directly for its bg-sphere UV ray math
    // (mobiledlss/datagen/camera.py::sphere_uv's `dirs_cam @ c2w[:3,:3].T`):
    // buildCvViewRowMajor's local r/u/f ARE exactly camera-to-world's
    // rotation columns (its rows are world->camera, so world->camera's
    // transpose -- i.e. camera->world -- has r/u/f as columns), so no
    // separate camera-to-world matrix construction is needed.
    glm::vec3 rAxis, uAxis, fAxis;
};

// Actor/action center for THIS scene (juggle_p0.8_stride4.glb), taken
// verbatim from lux-4dgs/tests/assets/cameras/generate_juggle_camera.py
// ("Actor/action center taken as world (-0.22, -1.04, 0.09) (given)") --
// that script is the already-validated (docs/lux-4d-spec.md section D)
// reproduction of mobiledlss/datagen/camera.py::orbit_path/look_at for this
// exact glb, so this is the known-good center rather than
// SceneManager::computeAutoCamera's whole-scene (person + dome background)
// bounding-box centroid, which sits far outside the room the person is
// standing in and produced a wildly wrong orbit (camera outside the whole
// dome looking at it from ~15 units away) the first time this was tried.
constexpr glm::vec3 kJuggleActorCenter(-0.22f, -1.04f, 0.09f);

OrbitFrame computeOrbitFrame(float frameIndex, uint32_t width, uint32_t height) {
    float az = glm::radians(frameIndex * kDegPerFrame);
    float el = glm::radians(kElevationDeg);
    glm::vec3 center = kJuggleActorCenter;
    glm::vec3 eye = center + kOrbitRadius * glm::vec3(cosf(el) * cosf(az), -sinf(el), cosf(el) * sinf(az));
    glm::vec3 upWorld(0.0f, -1.0f, 0.0f);  // PanopticSports y-down, matches generate_juggle_camera.py

    auto viewCv = buildCvViewRowMajor(eye, center, upWorld);
    glm::mat4 viewGl = DlssIO::cvViewToGl(viewCv);

    float fy = 0.5f * static_cast<float>(height) / tanf(glm::radians(kFovYDeg) * 0.5f);
    float fx = fy;
    float cx = width * 0.5f, cy = height * 0.5f;
    glm::mat4 proj = DlssIO::buildIntrinsicsProjection(fx, fy, cx, cy,
                                                        static_cast<float>(width), static_cast<float>(height),
                                                        0.01f, 100.0f);

    // Same r/u/f the view matrix itself was built from (see buildCvViewRowMajor).
    glm::vec3 f = glm::normalize(center - eye);
    glm::vec3 r = glm::normalize(glm::cross(-upWorld, f));
    glm::vec3 u = glm::cross(f, r);
    return {eye, viewGl, proj, fx, fy, r, u, f};
}

struct AppState {
    struct android_app* app = nullptr;
    bool vulkanReady = false;
    bool windowInitialized = false;

    VulkanContext ctx;
    SceneManager scene;  // owns the Target (full-res, jitter-free) SplatRenderer

    // Stage 2: independent proxy-resolution SplatRenderer (480x270, jittered)
    // constructed directly from scene.getGltfScene().splat_data -- a second
    // GPU-resident copy of the splat buffers/pipelines at a different
    // resolution, entirely separate from SceneManager's Target renderer.
    std::unique_ptr<SplatRenderer> proxyRenderer;

    // Stage 3: assembles ParamPredUNet's 26-channel input tensor from the
    // proxy renderer's per-frame attachments (docs/rendering-engines.md).
    // param_stride=2, hidden=8 match the checkpoint this demo targets
    // (expY_mem3_juggle_p0.8_ps2.pt -- task brief's "State" section).
    InputAssembly inputAssembly;
    static constexpr uint32_t kParamStride = 2;
    static constexpr uint32_t kHiddenChannels = 8;
    static constexpr uint32_t kTexChannels = 8;

    // Stage 4: TFLite C API + GPU delegate net inference (docs/rendering-
    // engines.md). Runs once on the frame-30 dump for validation (see
    // dumpProxyDebugFrame's call site) -- not yet driven every frame
    // (Stage 5 wires the recurrent hidden state + per-frame invocation).
    NetRunner netRunner;
    bool netRunnerReady = false;
    // Recurrent hidden state (net-res, hiddenChannels) fed back frame to
    // frame once the Stage 5 net-warmup window starts; empty == zero.
    std::vector<float> hiddenState;

    // Stage 5 (docs/rendering-engines.md): live reconstruct-with-memory
    // validation against the REAL juggle scene (not the synthetic clip
    // Stage 5's runReconstructDump() startup pass above already validated
    // bit-for-bit). kNetWarmupStart..kReconWindowStart gives the recurrent
    // hidden state 16 frames to leave its zero-initial condition before the
    // 16-frame capture window [kReconWindowStart, +kReconWindowFrames)
    // that's actually dumped and reconstructed/compared -- the app starts
    // in DemoMode::Proxy and only advances on a tap (Stage 6), so both
    // ranges see `isProxy` (== mode != Target) true throughout by default,
    // with no auto-cycle to coordinate against.
    static constexpr int kNetWarmupStart = 34;
    static constexpr int kReconWindowStart = 50;
    static constexpr int kReconWindowFrames = 16;
    bool stage5Done = false;

    DemoMode mode = DemoMode::Proxy;
    // Stage 6: tap-to-cycle (see onInputEvent()) sets this from the input
    // thread/callback; renderFrame() consumes+clears it at the top of the
    // next frame it renders (single-threaded render loop, no lock needed --
    // android_native_app_glue's ALooper_pollOnce/source->process() and
    // renderFrame() both run on the same thread, see android_main()).
    bool pendingModeAdvance = false;

    // Stage 6: persistent target-res (960x540) RGBA32F image Bicubic/
    // Reconstruction modes upload their CPU-computed colour into before
    // blitting to the swapchain (see uploadRgbToDisplayImage/
    // blitImageToSwapchain below) -- neither mode has its result as a
    // ready-made VkImage the way Proxy/Target's own SplatRenderer output
    // already is.
    VkImage displayImage = VK_NULL_HANDLE;
    VmaAllocation displayAlloc = VK_NULL_HANDLE;

    VkSemaphore imageAvailableSem = VK_NULL_HANDLE;
    VkSemaphore renderFinishedSem = VK_NULL_HANDLE;
    VkFence inFlightFence = VK_NULL_HANDLE;
    VkCommandBuffer blitCmd = VK_NULL_HANDLE;

    int frameCounter = 0;

    std::chrono::high_resolution_clock::time_point fpsWindowStart;
    int fpsWindowFrames = 0;
    float lastFps = 0.0f;

    bool initFailed = false;

    // --- Stage 1 timing breakdown (docs/rendering-engines.md "Android
    // live-rendering demo") -- CPU wall-clock per phase (ms) accumulated
    // over a 60-frame window, plus the GPU timestamp-query breakdown from
    // SplatRenderer::lastGpuTimingsMs() (preprocess/sort/draw), logged
    // together every ~60 frames via LOGI so `adb logcat -d | grep TIMING`
    // gives a periodic snapshot without a blocking logcat stream.
    static constexpr int kTimingWindowFrames = 60;
    int timingFrameCount = 0;
    double sumCpuWaitFenceMs = 0.0;   // vkWaitForFences (outer loop, waits for prior frame's blit+present)
    double sumCpuRenderMs = 0.0;      // wall time of splatR->render() (preprocess+sort+draw, incl. its 2 internal vkQueueWaitIdle round trips)
    double sumCpuBlitPresentMs = 0.0; // blit cmd record+submit+vkQueuePresentKHR (does not itself block)
    double sumCpuFrameMs = 0.0;       // total renderFrame() wall time
    double sumGpuPreprocessMs = 0.0;
    double sumGpuSortMs = 0.0;
    double sumGpuDrawMs = 0.0;
    int gpuTimingValidFrames = 0;
};

std::string basePath(AppState* state) {
    // internalDataPath (/data/user/0/<pkg>/files, i.e. getFilesDir()) rather
    // than externalDataPath: adb push into
    // /sdcard/Android/data/<pkg>/files produced files this app's own
    // process couldn't fopen() (cgltf_parse_file came back
    // cgltf_result_file_not_found == 6 despite `adb shell ls` showing the
    // right size/rw-rw-rw perms) -- Android 11+ scoped storage enforces
    // per-app isolation on that path at a layer below plain POSIX
    // permission bits, and files written by the shell UID don't reliably
    // get labeled for the app UID to read even inside its "own" external
    // dir. push_assets.sh instead stages via /data/local/tmp + `run-as`
    // into this internal dir, which the app's UID owns outright.
    return state->app->activity->internalDataPath;
}

void initRenderer(AppState* state) {
    std::string base = basePath(state);
    if (chdir(base.c_str()) != 0) {
        LOGE("chdir(%s) failed", base.c_str());
    } else {
        LOGI("chdir to %s", base.c_str());
    }
    mkdir((base + "/dump").c_str(), 0755);  // Stage 2 validation dump target (dumpProxyDebugFrame)

    // Task 3 (docs/rendering-engines.md, GPU-delegate correctness bisect):
    // runs before anything else (no Vulkan/scene dependency at all) and is
    // a no-op if files/probes doesn't exist -- see gpu_probe.h.
    runGpuProbes(base + "/probes");

    try {
        AndroidVulkan::init(state->ctx, state->app->window, false);
        AndroidVulkan::createSwapchain(state->ctx, kWidth, kHeight);

        std::string scenePath = base + "/scene/juggle_p0.8_stride4.glb";
        LOGI("Loading scene: %s", scenePath.c_str());
        state->scene.loadScene(state->ctx, scenePath);

        int vertexStride = 32;  // pure splat scene, no glTF mesh triangles
        state->scene.uploadToGPU(state->ctx, vertexStride);
        state->scene.uploadTextures(state->ctx);

        if (!state->scene.hasSplatData()) {
            LOGE("Scene has no gaussian splat data!");
            state->initFailed = true;
            return;
        }

        std::string shaderBase = base + "/examples/gaussian_splat_dlss";
        state->scene.initSplatRenderer(state->ctx, shaderBase, kWidth, kHeight);
        state->scene.getSplatRenderer()->setGpuTimingEnabled(state->ctx, true);

        // Stage 2: second, independent SplatRenderer at proxy resolution,
        // built directly from the same CPU-side splat_data the Target
        // renderer above was built from (SplatRenderer::init() uploads its
        // own copy of the buffers -- the two renderers share no GPU state).
        state->proxyRenderer = std::make_unique<SplatRenderer>();
        state->proxyRenderer->init(state->ctx, state->scene.getGltfScene().splat_data,
                                    shaderBase, kProxyWidth, kProxyHeight);
        state->proxyRenderer->setGpuTimingEnabled(state->ctx, true);

        // Stage 3: input-assembly GLSL compute pipeline (assets pushed by
        // push_assets.sh: assets/texture.npy, assets/bg_sphere.npy,
        // shaders_ia/input_assembly_{unpremul_depth,assemble}.comp.spv).
        state->inputAssembly.init(state->ctx, base + "/assets/texture.npy", base + "/assets/bg_sphere.npy",
                                   kProxyWidth, kProxyHeight, AppState::kParamStride, AppState::kHiddenChannels,
                                   base + "/shaders_ia");
        LOGI("InputAssembly initialized: net=%ux%u channels=%u",
             state->inputAssembly.getNetW(), state->inputAssembly.getNetH(), state->inputAssembly.getChannels());

        // Stage 4: TFLite net (best-effort -- GPU delegate availability
        // varies by driver; log and continue the rest of the demo on failure).
        try {
            state->netRunner.init(base + "/assets/unet_ps2_mem.tflite",
                                   state->inputAssembly.getNetW(), state->inputAssembly.getNetH(),
                                   AppState::kParamStride, AppState::kHiddenChannels, AppState::kTexChannels);
            state->netRunnerReady = true;
            LOGI("NetRunner initialized: gpu_delegate=%d output_channels=%u",
                 state->netRunner.isGpuDelegateActive(), state->netRunner.getOutputChannels());
        } catch (const std::exception& e) {
            LOGE("NetRunner init failed (continuing without Stage 4): %s", e.what());
            state->netRunnerReady = false;
        }

        // Stage 5 validation: runs playground_cpp/src/reconstruct_runner.cpp's
        // runReconstructDump() -- reused BY REFERENCE, unmodified, exactly
        // as docs/rendering-engines.md's Stage 5 plan allows -- once at
        // startup, against a 16-frame synthetic dump (mobiledlss/tools/
        // reconstruct_reference_dump.py --checkpoint expY_mem3_juggle_p0.8_ps2.pt
        // --memory --net-h 136 --net-w 240 --frames 16) pushed onto the
        // device by push_assets.sh, using the new examples/
        // reconstruct_mem_ps2 pipeline (s=2 k=4 param_stride=2 hidden=8
        // memory channels=8/hidden=16 -- this checkpoint's exact config,
        // which neither examples/reconstruct_mem.lux [ps=1] nor
        // examples/reconstruct_ps2.lux [no memory] alone covers). Desktop
        // (`playground_cpp/build/lux-playground --reconstruct-dump`) was
        // already checked bit-for-bit against mobiledlss.train.reconstruct
        // (max|diff| ~4e-5 over all 16 frames/hidden state, docs/
        // rendering-engines.md's commit for this stage has the full table)
        // -- this on-device run re-validates the identical SPIR-V pipeline
        // on the Mali-G715 itself.
        {
            std::string reconDumpDir = base + "/recon_dump";
            std::string reconOutDir = base + "/recon_out";
            struct stat st{};
            if (stat(reconDumpDir.c_str(), &st) == 0) {
                auto t0 = std::chrono::high_resolution_clock::now();
                int rc = runReconstructDump(state->ctx, reconDumpDir, reconOutDir,
                                             base + "/examples/reconstruct_mem_ps2");
                double ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
                LOGI("Stage5 runReconstructDump: rc=%d elapsed=%.1fms out=%s", rc, ms, reconOutDir.c_str());
            } else {
                LOGI("Stage5: %s not found on device, skipping runReconstructDump validation pass", reconDumpDir.c_str());
            }
        }

        // NOTE: deliberately NOT using scene.getAutoTarget()/getAutoEye()
        // here -- SceneManager::computeAutoCamera frames the WHOLE scene's
        // bounding box (person + the ~2.5-3.2-unit PanopticSports dome
        // background), whose centroid sits far outside the room the person
        // actually stands in. First attempt used that (radius ~15.8) and
        // produced a camera looking at the entire dome from outside as a
        // tiny blurry blob. computeOrbitFrame() instead uses
        // kJuggleActorCenter, matching generate_juggle_camera.py exactly.
        LOGI("Splat renderer initialized: %ux%u, actor center=(%.3f,%.3f,%.3f) radius=%.3f",
             kWidth, kHeight, kJuggleActorCenter.x, kJuggleActorCenter.y, kJuggleActorCenter.z, kOrbitRadius);

        // Stage 6: persistent target-res display image for Bicubic/
        // Reconstruction modes (see AppState::displayImage's comment).
        {
            VkImageCreateInfo imgInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            imgInfo.imageType = VK_IMAGE_TYPE_2D;
            imgInfo.format = VK_FORMAT_R32G32B32A32_SFLOAT;
            imgInfo.extent = {kWidth, kHeight, 1};
            imgInfo.mipLevels = 1;
            imgInfo.arrayLayers = 1;
            imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
            imgInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            VmaAllocationCreateInfo allocInfo{};
            allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
            vmaCreateImage(state->ctx.allocator, &imgInfo, &allocInfo, &state->displayImage, &state->displayAlloc, nullptr);
        }

        VkSemaphoreCreateInfo semInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        vkCreateSemaphore(state->ctx.device, &semInfo, nullptr, &state->imageAvailableSem);
        vkCreateSemaphore(state->ctx.device, &semInfo, nullptr, &state->renderFinishedSem);
        VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        vkCreateFence(state->ctx.device, &fenceInfo, nullptr, &state->inFlightFence);

        state->fpsWindowStart = std::chrono::high_resolution_clock::now();
        state->vulkanReady = true;
    } catch (const std::exception& e) {
        LOGE("Renderer init failed: %s", e.what());
        state->initFailed = true;
    }
}

// Stage 1 helper: milliseconds between two high_resolution_clock points.
inline double msSince(std::chrono::high_resolution_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
}

// --------------------------------------------------------------------------
// Stage 5 helpers: live reconstruct-with-memory validation against the real
// juggle scene (docs/rendering-engines.md). Writes a mobiledlss/tools/
// reconstruct_reference_dump.py-shaped directory from ACTUAL on-device
// render/InputAssembly/NetRunner output over kReconWindowFrames consecutive
// orbit frames, then replays it once through reconstruct_pass.h's
// runReconstructDump() (same reused-by-reference code already bit-for-bit
// validated against mobiledlss.train.reconstruct on the synthetic dump at
// startup -- see initRenderer()'s comment).
// --------------------------------------------------------------------------

// Replicate-pads [h,w,c] -> [targetH,w,c] by repeating the last row (proxy
// 270 -> 272 rows, matching netH*paramStride -- the same "135->136" pad
// convention InputAssembly's own subtap clamp applies internally, just
// materialized here as real padded rows since reconstruct_pass.h's
// runReconstructDump reads proxy_color/mv_proxy straight off disk at the
// meta.json-declared padded proxy resolution).
std::vector<float> padRowsReplicate(const std::vector<float>& src, uint32_t h, uint32_t w, uint32_t c, uint32_t targetH) {
    std::vector<float> out(static_cast<size_t>(targetH) * w * c);
    for (uint32_t y = 0; y < targetH; y++) {
        uint32_t srcY = std::min(y, h - 1);
        std::memcpy(out.data() + static_cast<size_t>(y) * w * c,
                    src.data() + static_cast<size_t>(srcY) * w * c, static_cast<size_t>(w) * c * sizeof(float));
    }
    return out;
}

void writeMetaJson(const std::string& path, int s, int k, int paramStride, int hidden,
                    int proxyW, int proxyH, int targetW, int targetH, int netW, int netH, int numFrames,
                    int memChannels, int memHidden, int texW, int texH) {
    FILE* f = fopen(path.c_str(), "w");
    if (!f) {
        LOGE("writeMetaJson: failed to open %s", path.c_str());
        return;
    }
    fprintf(f,
            "{\"s\":%d,\"k\":%d,\"param_stride\":%d,\"hidden\":%d,\"proxy_w\":%d,\"proxy_h\":%d,"
            "\"target_w\":%d,\"target_h\":%d,\"net_w\":%d,\"net_h\":%d,\"num_frames\":%d,"
            "\"memory_channels\":%d,\"memory_hidden\":%d,\"tex_w\":%d,\"tex_h\":%d}\n",
            s, k, paramStride, hidden, proxyW, proxyH, targetW, targetH, netW, netH, numFrames,
            memChannels, memHidden, texW, texH);
    fclose(f);
}

// Copies a small static asset file (bg_sphere.npy / texture.npy /
// memory_head.npz) into the live dump directory, once, so runReconstructDump
// finds everything it needs (bg_sphere.npy/texture.npy/memory_head.npz)
// alongside the per-frame files in the SAME directory.
void copyFile(const std::string& src, const std::string& dst) {
    FILE* in = fopen(src.c_str(), "rb");
    if (!in) {
        LOGE("copyFile: failed to open src %s", src.c_str());
        return;
    }
    FILE* out = fopen(dst.c_str(), "wb");
    if (!out) {
        LOGE("copyFile: failed to open dst %s", dst.c_str());
        fclose(in);
        return;
    }
    char buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), in)) > 0) fwrite(buf, 1, n, out);
    fclose(in);
    fclose(out);
}

// --------------------------------------------------------------------------
// Stage 2 validation: one-shot proxy-frame dump (colour/MV/depth), to be
// diffed against the Mac Vulkan CLI (playground_cpp/build/lux-playground
// --output-aux) run with the same --camera-json/--camera-json-prev/--time/
// --time-prev/--jitter. New code, entirely local to this file -- not a
// copy of playground_cpp/src/screenshot.cpp's Screenshot::readImageRaw()
// (same ~30-line vkCmdCopyImageToBuffer-via-staging-buffer pattern, just
// inlined here) to avoid pulling that translation unit's
// stbi_write_png() reference into the Android link (nothing else in this
// app's CMakeLists defines STB_IMAGE_WRITE_IMPLEMENTATION, unlike the
// desktop CLI's vulkan_context.cpp, which Android doesn't compile). Reuses
// DlssIO::convertRgba16fColorAttachment / unpremultiplyByAlpha /
// writeNpyFloat32 by reference (dlss_io.cpp is already in the Android
// CMakeLists) so the on-disk format is byte-for-byte the same code path
// the Mac CLI's --output-aux uses.
// --------------------------------------------------------------------------

std::vector<uint8_t> readImageRawAndroid(VulkanContext& ctx, VkImage image,
                                          uint32_t width, uint32_t height,
                                          uint32_t bytesPerPixel, VkImageLayout currentLayout) {
    VkDeviceSize imageSize = static_cast<VkDeviceSize>(width) * height * bytesPerPixel;

    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = imageSize;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;
    allocInfo.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    VkBuffer stagingBuffer;
    VmaAllocation stagingAllocation;
    if (vmaCreateBuffer(ctx.allocator, &bufferInfo, &allocInfo, &stagingBuffer, &stagingAllocation, nullptr) != VK_SUCCESS) {
        throw std::runtime_error("readImageRawAndroid: failed to create staging buffer");
    }

    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
    if (currentLayout != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
        VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barrier.oldLayout = currentLayout;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &barrier);
    }
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {width, height, 1};
    vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, stagingBuffer, 1, &region);
    ctx.endSingleTimeCommands(cmd);

    void* mapped = nullptr;
    vmaMapMemory(ctx.allocator, stagingAllocation, &mapped);
    std::vector<uint8_t> pixels(imageSize);
    memcpy(pixels.data(), mapped, imageSize);
    vmaUnmapMemory(ctx.allocator, stagingAllocation);
    vmaDestroyBuffer(ctx.allocator, stagingBuffer, stagingAllocation);
    return pixels;
}

void dumpProxyDebugFrame(VulkanContext& ctx, SplatRenderer* splatR, const std::string& outDir,
                          const std::string& tag, int frameIndex, float t, float jx, float jy) {
    uint32_t w = splatR->getWidth(), h = splatR->getHeight();
    LOGI("DUMP tag=%s frame=%d t=%.6f jitter=(%.6f,%.6f) size=%ux%u dir=%s",
         tag.c_str(), frameIndex, t, jx, jy, w, h, outDir.c_str());

    auto rawColor = readImageRawAndroid(ctx, splatR->getOutputImage(), w, h, 8,
                                         VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    std::vector<uint8_t> unusedRgba8;
    auto colorF32 = DlssIO::convertRgba16fColorAttachment(rawColor, w, h, unusedRgba8);
    DlssIO::writeNpyFloat32(outDir + "/android_" + tag + "_color.npy", colorF32, {h, w, 4});

    if (splatR->hasExpectedDepth()) {
        auto raw = readImageRawAndroid(ctx, splatR->getExpectedDepthImage(), w, h, 16,
                                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        std::vector<float> rgba(static_cast<size_t>(w) * h * 4);
        memcpy(rgba.data(), raw.data(), raw.size());
        std::vector<float> depthPremul(static_cast<size_t>(w) * h), depthAlpha(static_cast<size_t>(w) * h);
        for (size_t i = 0; i < depthPremul.size(); ++i) {
            depthPremul[i] = rgba[i * 4 + 0];
            depthAlpha[i] = rgba[i * 4 + 3];
        }
        auto depth = DlssIO::unpremultiplyByAlpha(depthPremul, depthAlpha, w, h, 1);
        DlssIO::writeNpyFloat32(outDir + "/android_" + tag + "_depth.npy", depth, {h, w});

        // foreground_coverage packs into out_depth's .g channel (index 1),
        // same alpha (.w) as depth -- see SPECIFICATION.md 12.8 /
        // playground_cpp/src/main.cpp's --output-aux _fg.npy path (identical
        // extraction, reused here for Stage-3 fg validation).
        if (splatR->hasForegroundCoverage()) {
            std::vector<float> fgPremul(static_cast<size_t>(w) * h);
            for (size_t i = 0; i < fgPremul.size(); ++i) {
                fgPremul[i] = rgba[i * 4 + 1];
            }
            auto fg = DlssIO::unpremultiplyByAlpha(fgPremul, depthAlpha, w, h, 1);
            DlssIO::writeNpyFloat32(outDir + "/android_" + tag + "_fg.npy", fg, {h, w});
        }
    }
    if (splatR->hasMotionVectors()) {
        auto raw = readImageRawAndroid(ctx, splatR->getMotionImage(), w, h, 16,
                                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        std::vector<float> rgba(static_cast<size_t>(w) * h * 4);
        memcpy(rgba.data(), raw.data(), raw.size());
        std::vector<float> mvPremul(static_cast<size_t>(w) * h * 2), mvAlpha(static_cast<size_t>(w) * h);
        for (size_t i = 0; i < mvAlpha.size(); ++i) {
            mvPremul[i * 2 + 0] = rgba[i * 4 + 0];
            mvPremul[i * 2 + 1] = rgba[i * 4 + 1];
            mvAlpha[i] = rgba[i * 4 + 3];
        }
        auto mv = DlssIO::unpremultiplyByAlpha(mvPremul, mvAlpha, w, h, 2);
        DlssIO::writeNpyFloat32(outDir + "/android_" + tag + "_mv.npy", mv, {h, w, 2});
    }
    LOGI("DUMP done: %s/android_%s_{color,depth,mv}.npy", outDir.c_str(), tag.c_str());
}

// Stage 5: un-premultiplied RGB color only, at whatever resolution splatR
// currently renders (used both for proxy_color_f{t}.npy at proxy res and
// the Target-render ground truth at target res -- see the header comment
// above writeMetaJson()). DlssIO::convertRgba16fColorAttachment already
// returns un-premultiplied RGBA (see dumpProxyDebugFrame's identical call),
// so only the alpha channel needs dropping here.
std::vector<float> readColorRgb(VulkanContext& ctx, SplatRenderer* splatR) {
    uint32_t w = splatR->getWidth(), h = splatR->getHeight();
    auto rawColor = readImageRawAndroid(ctx, splatR->getOutputImage(), w, h, 8, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    std::vector<uint8_t> unusedRgba8;
    auto colorF32 = DlssIO::convertRgba16fColorAttachment(rawColor, w, h, unusedRgba8);  // [h,w,4]
    std::vector<float> rgb(static_cast<size_t>(w) * h * 3);
    for (size_t i = 0; i < static_cast<size_t>(w) * h; i++) {
        rgb[i * 3 + 0] = colorF32[i * 4 + 0];
        rgb[i * 3 + 1] = colorF32[i * 4 + 1];
        rgb[i * 3 + 2] = colorF32[i * 4 + 2];
    }
    return rgb;
}

// Stage 5: proxy_h,proxy_w-resolution un-premultiplied MV (backward, jitter
// -free -- SplatRenderer's out_motion is always jitter-free by construction,
// see setJitter()'s header comment), matching mv_proxy_f{t}.npy's contract.
std::vector<float> readMvProxy(VulkanContext& ctx, SplatRenderer* splatR) {
    uint32_t w = splatR->getWidth(), h = splatR->getHeight();
    auto raw = readImageRawAndroid(ctx, splatR->getMotionImage(), w, h, 16, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    std::vector<float> rgba(static_cast<size_t>(w) * h * 4);
    memcpy(rgba.data(), raw.data(), raw.size());
    std::vector<float> mvPremul(static_cast<size_t>(w) * h * 2), mvAlpha(static_cast<size_t>(w) * h);
    for (size_t i = 0; i < mvAlpha.size(); ++i) {
        mvPremul[i * 2 + 0] = rgba[i * 4 + 0];
        mvPremul[i * 2 + 1] = rgba[i * 4 + 1];
        mvAlpha[i] = rgba[i * 4 + 3];
    }
    return DlssIO::unpremultiplyByAlpha(mvPremul, mvAlpha, w, h, 2);
}

// Stage 6: uploads a CPU-side [h,w,3] RGB buffer (Bicubic's or
// Reconstruction's result) into AppState::displayImage, leaving it in
// TRANSFER_SRC_OPTIMAL (ready for blitImageToSwapchain below). One-shot
// synchronous staging-buffer upload -- correctness/simplicity over
// per-frame latency, matching every other new Stage 3-5 GPU path in this
// file.
void uploadRgbToDisplayImage(VulkanContext& ctx, VkImage image, const std::vector<float>& rgb, uint32_t w, uint32_t h) {
    std::vector<float> rgba(static_cast<size_t>(w) * h * 4, 1.0f);
    for (size_t i = 0; i < static_cast<size_t>(w) * h; i++) {
        rgba[i * 4 + 0] = rgb[i * 3 + 0];
        rgba[i * 4 + 1] = rgb[i * 3 + 1];
        rgba[i * 4 + 2] = rgb[i * 3 + 2];
    }
    VkDeviceSize sizeBytes = rgba.size() * sizeof(float);
    VkBufferCreateInfo bufInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufInfo.size = sizeBytes;
    bufInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;
    allocInfo.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    VkBuffer stagingBuffer;
    VmaAllocation stagingAlloc;
    vmaCreateBuffer(ctx.allocator, &bufInfo, &allocInfo, &stagingBuffer, &stagingAlloc, nullptr);
    void* mapped = nullptr;
    vmaMapMemory(ctx.allocator, stagingAlloc, &mapped);
    memcpy(mapped, rgba.data(), sizeBytes);
    vmaUnmapMemory(ctx.allocator, stagingAlloc);

    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
    VkImageMemoryBarrier toDst{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    toDst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    toDst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toDst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toDst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toDst.image = image;
    toDst.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    toDst.srcAccessMask = 0;
    toDst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &toDst);

    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {w, h, 1};
    vkCmdCopyBufferToImage(cmd, stagingBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    VkImageMemoryBarrier toSrc = toDst;
    toSrc.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toSrc.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    toSrc.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    toSrc.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &toSrc);
    ctx.endSingleTimeCommands(cmd);

    vmaDestroyBuffer(ctx.allocator, stagingBuffer, stagingAlloc);
}

// Stage 6: generic version of SplatRenderer::blitToSwapchain -- same
// barrier/blit pattern (see splat_renderer.cpp), parameterized on an
// arbitrary source VkImage (already TRANSFER_SRC_OPTIMAL) instead of a
// SplatRenderer's own colorImage_, since Bicubic/Reconstruction display
// AppState::displayImage rather than any SplatRenderer's output.
void blitImageToSwapchain(VkCommandBuffer cmd, VkImage srcImage, uint32_t srcW, uint32_t srcH,
                           VkImage swapImage, VkExtent2D extent) {
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = swapImage;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &barrier);

    VkImageBlit blitRegion{};
    blitRegion.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blitRegion.srcOffsets[1] = {static_cast<int32_t>(srcW), static_cast<int32_t>(srcH), 1};
    blitRegion.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blitRegion.dstOffsets[1] = {static_cast<int32_t>(extent.width), static_cast<int32_t>(extent.height), 1};
    vkCmdBlitImage(cmd, srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, swapImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                  1, &blitRegion, VK_FILTER_LINEAR);

    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = 0;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &barrier);
}

// Task 2 (Reconstruction-mode time budget): everything runReconstructionModeFrame
// does beyond the net run itself (which fills its own NetRunner::RunTimingsMs).
struct ReconModeFrameTimingsMs {
    NetRunner::RunTimingsMs net;
    double dumpWriteMs = 0.0;   // proxy_color/mv_proxy/jitter/packed_params/disocc/k_params/cam_to_world npy writes
    ReconstructTimingsMs reconstruct;  // runReconstructDump's own breakdown (setup/read/upload/dispatch/download/write/teardown)
    double outputReadMs = 0.0;  // reading out_f0.npy back off disk
};

// Stage 6: runs the Reconstruction mode's full net + reconstruct-with-
// memory pass for exactly ONE frame (fresh 1-frame dump dir each call --
// see DemoMode's header comment) and returns its [kHeight,kWidth,3]
// output cropped from the pass's own padded 960x544 (same crop Stage 5's
// PSNR comparison uses). Reuses the SAME NetRunner hidden-state feedback
// (state->hiddenState) the Stage 5 warmup/capture window already
// established -- calling this repeatedly (every displayed frame while in
// Reconstruction mode) keeps that recurrence live and running. `outTimings`
// (optional) receives task 2's breakdown of everything below.
std::vector<float> runReconstructionModeFrame(AppState* state, VulkanContext& ctx, SplatRenderer* splatR,
                                               const OrbitFrame& frame, float jitterTargetX, float jitterTargetY,
                                               ReconModeFrameTimingsMs* outTimings = nullptr) {
    NetRunner::RunTimingsMs t{};
    const float* hiddenPtr = state->hiddenState.empty() ? nullptr : state->hiddenState.data();
    const float* out = state->netRunner.run(state->inputAssembly.getOutputHostPtr(), hiddenPtr, t);
    if (outTimings != nullptr) outTimings->net = t;
    uint32_t netW = state->inputAssembly.getNetW(), netH = state->inputAssembly.getNetH();
    uint32_t outCh = state->netRunner.getOutputChannels();
    uint32_t hiddenCh = AppState::kHiddenChannels;
    state->hiddenState.assign(static_cast<size_t>(netW) * netH * hiddenCh, 0.0f);
    for (size_t p = 0; p < static_cast<size_t>(netW) * netH; p++) {
        memcpy(&state->hiddenState[p * hiddenCh], &out[p * outCh + (outCh - hiddenCh)], hiddenCh * sizeof(float));
    }

    std::string dumpDir = basePath(state) + "/live_recon_dump_1f";
    static bool staticsCopied = false;
    if (!staticsCopied) {
        mkdir(dumpDir.c_str(), 0755);
        copyFile(basePath(state) + "/assets/bg_sphere.npy", dumpDir + "/bg_sphere.npy");
        copyFile(basePath(state) + "/assets/texture.npy", dumpDir + "/texture.npy");
        copyFile(basePath(state) + "/assets/memory_head.npz", dumpDir + "/memory_head.npz");
        staticsCopied = true;
    }
    uint32_t paddedProxyH = netH * AppState::kParamStride;
    uint32_t targetW = kWidth, targetH = paddedProxyH * 2;
    writeMetaJson(dumpDir + "/meta.json", 2, 4, AppState::kParamStride, hiddenCh, kProxyWidth, paddedProxyH,
                  targetW, targetH, netW, netH, /*numFrames=*/1, AppState::kTexChannels, 16, 512, 256);

    auto tDumpWrite = std::chrono::high_resolution_clock::now();
    auto colorProxy = readColorRgb(ctx, splatR);
    auto colorPadded = padRowsReplicate(colorProxy, kProxyHeight, kProxyWidth, 3, paddedProxyH);
    DlssIO::writeNpyFloat32(dumpDir + "/proxy_color_f0.npy", colorPadded, {paddedProxyH, kProxyWidth, 3});
    auto mvProxy = readMvProxy(ctx, splatR);
    auto mvPadded = padRowsReplicate(mvProxy, kProxyHeight, kProxyWidth, 2, paddedProxyH);
    DlssIO::writeNpyFloat32(dumpDir + "/mv_proxy_f0.npy", mvPadded, {paddedProxyH, kProxyWidth, 2});
    std::vector<float> jitterArr = {jitterTargetX, jitterTargetY};
    DlssIO::writeNpyFloat32(dumpDir + "/jitter_f0.npy", jitterArr, {2});
    std::vector<float> packed(out, out + static_cast<size_t>(netW) * netH * outCh);
    DlssIO::writeNpyFloat32(dumpDir + "/packed_params_f0.npy", packed, {netH, netW, outCh});

    const float* netTensor = state->inputAssembly.getOutputHostPtr();
    uint32_t ch26 = state->inputAssembly.getChannels();
    std::vector<float> disoccTarget(static_cast<size_t>(targetW) * targetH);
    uint32_t upFactor = AppState::kParamStride * 2;
    for (uint32_t y = 0; y < targetH; y++) {
        uint32_t gy = std::min(y / upFactor, netH - 1);
        for (uint32_t x = 0; x < targetW; x++) {
            uint32_t gx = std::min(x / upFactor, netW - 1);
            disoccTarget[static_cast<size_t>(y) * targetW + x] = netTensor[(static_cast<size_t>(gy) * netW + gx) * ch26 + 6];
        }
    }
    DlssIO::writeNpyFloat32(dumpDir + "/disocc_f0.npy", disoccTarget, {targetH, targetW});

    OrbitFrame reconTargetFrame = computeOrbitFrame(static_cast<float>(state->frameCounter), targetW, targetH);
    std::vector<float> kParams = {reconTargetFrame.fx, reconTargetFrame.fy, targetW * 0.5f, targetH * 0.5f};
    DlssIO::writeNpyFloat32(dumpDir + "/k_params_f0.npy", kParams, {4});
    std::vector<float> camToWorld(16, 0.0f);
    camToWorld[0] = reconTargetFrame.rAxis.x; camToWorld[1] = reconTargetFrame.uAxis.x; camToWorld[2] = reconTargetFrame.fAxis.x; camToWorld[3] = reconTargetFrame.eye.x;
    camToWorld[4] = reconTargetFrame.rAxis.y; camToWorld[5] = reconTargetFrame.uAxis.y; camToWorld[6] = reconTargetFrame.fAxis.y; camToWorld[7] = reconTargetFrame.eye.y;
    camToWorld[8] = reconTargetFrame.rAxis.z; camToWorld[9] = reconTargetFrame.uAxis.z; camToWorld[10] = reconTargetFrame.fAxis.z; camToWorld[11] = reconTargetFrame.eye.z;
    camToWorld[15] = 1.0f;
    DlssIO::writeNpyFloat32(dumpDir + "/cam_to_world_f0.npy", camToWorld, {4, 4});
    if (outTimings != nullptr) outTimings->dumpWriteMs = msSince(tDumpWrite);

    std::string outDir = basePath(state) + "/live_recon_out_1f";
    int rc = runReconstructDump(ctx, dumpDir, outDir, basePath(state) + "/examples/reconstruct_mem_ps2",
                                 outTimings != nullptr ? &outTimings->reconstruct : nullptr);
    if (rc != 0) {
        LOGE("Reconstruction mode: runReconstructDump failed (rc=%d)", rc);
        return std::vector<float>(static_cast<size_t>(kWidth) * kHeight * 3, 0.0f);
    }
    auto tOutputRead = std::chrono::high_resolution_clock::now();
    DlssIO::NpyArray outArr = DlssIO::readNpyFloat32(outDir + "/out_f0.npy");  // [targetH,targetW,3]
    if (outTimings != nullptr) outTimings->outputReadMs = msSince(tOutputRead);
    std::vector<float> cropped(static_cast<size_t>(kWidth) * kHeight * 3);
    for (uint32_t y = 0; y < kHeight; y++) {
        memcpy(&cropped[static_cast<size_t>(y) * kWidth * 3], &outArr.data[static_cast<size_t>(y) * targetW * 3],
               static_cast<size_t>(kWidth) * 3 * sizeof(float));
    }
    return cropped;
}

void renderFrame(AppState* state) {
    if (!state->vulkanReady) return;
    VulkanContext& ctx = state->ctx;

    auto tFrameStart = std::chrono::high_resolution_clock::now();

    auto tWaitFence = std::chrono::high_resolution_clock::now();
    vkWaitForFences(ctx.device, 1, &state->inFlightFence, VK_TRUE, UINT64_MAX);
    double waitFenceMs = msSince(tWaitFence);
    if (state->blitCmd != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(ctx.device, ctx.commandPool, 1, &state->blitCmd);
        state->blitCmd = VK_NULL_HANDLE;
    }

    uint32_t imageIndex = 0;
    VkResult acquireResult = vkAcquireNextImageKHR(ctx.device, ctx.swapchain, UINT64_MAX,
                                                    state->imageAvailableSem, VK_NULL_HANDLE, &imageIndex);
    if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
        vkDeviceWaitIdle(ctx.device);
        for (auto& iv : ctx.swapchainImageViews) vkDestroyImageView(ctx.device, iv, nullptr);
        ctx.swapchainImageViews.clear();
        vkDestroySwapchainKHR(ctx.device, ctx.swapchain, nullptr);
        ctx.swapchain = VK_NULL_HANDLE;
        AndroidVulkan::createSwapchain(ctx, kWidth, kHeight);
        return;
    }
    if (acquireResult != VK_SUCCESS && acquireResult != VK_SUBOPTIMAL_KHR) {
        LOGE("vkAcquireNextImageKHR failed: %d", acquireResult);
        return;
    }
    vkResetFences(ctx.device, 1, &state->inFlightFence);

    // --- Stage 6: tap-to-cycle (see onInputEvent()); no auto-cycle ---
    if (state->pendingModeAdvance) {
        state->pendingModeAdvance = false;
        state->mode = nextDemoMode(state->mode);
        // Don't mix a partial window's samples across the mode switch --
        // different modes use different renderers/resolutions/code paths.
        state->timingFrameCount = 0;
        state->sumCpuWaitFenceMs = state->sumCpuRenderMs = state->sumCpuBlitPresentMs = state->sumCpuFrameMs = 0.0;
        state->sumGpuPreprocessMs = state->sumGpuSortMs = state->sumGpuDrawMs = 0.0;
        state->gpuTimingValidFrames = 0;
        LOGI("mode switch -> %s", demoModeName(state->mode));
    }

    // Proxy/Bicubic/Reconstruction all drive the same jittered proxy
    // pipeline (InputAssembly + NetRunner); only Target renders the
    // separate full-res unjittered scene -- see DemoMode's comment.
    bool isProxy = (state->mode != DemoMode::Target);
    SplatRenderer* splatR = isProxy ? state->proxyRenderer.get() : state->scene.getSplatRenderer();
    uint32_t activeW = isProxy ? kProxyWidth : kWidth;
    uint32_t activeH = isProxy ? kProxyHeight : kHeight;

    OrbitFrame frame = computeOrbitFrame(static_cast<float>(state->frameCounter), activeW, activeH);
    splatR->updateCameraExplicit(frame.eye, frame.viewGl, frame.proj, frame.fx, frame.fy);

    // --- Stage 2: Halton(2,3) TAA jitter, proxy pass only (docs/lux-
    // reconstruct-spec.md's jitter is defined in TARGET px; scale by
    // proxyWidth/targetWidth == 0.5 to get proxy-px jitter for
    // setJitter(), which rasterizes with it but leaves the unjittered
    // view/proj SplatRenderer::render() already carries forward for motion
    // vectors -- see setJitter()'s header comment). Target mode is never
    // jittered (setJitter defaults to 0,0 and is never called on it).
    float jitterPxX = 0.0f, jitterPxY = 0.0f;
    float jitterTargetX = 0.0f, jitterTargetY = 0.0f;  // Stage 5: reconstruct-spec's jitter, target px
    if (isProxy) {
        glm::vec2 jTarget = taaJitterTargetPixels(state->frameCounter);
        jitterTargetX = jTarget.x;
        jitterTargetY = jTarget.y;
        float scale = static_cast<float>(kProxyWidth) / static_cast<float>(kWidth);  // 0.5
        jitterPxX = jTarget.x * scale;
        jitterPxY = jTarget.y * scale;
        splatR->setJitter(jitterPxX, jitterPxY);
    }

    float morphT = 0.0f;
    if (splatR->hasMotion()) {
        morphT = std::fmod(state->frameCounter * (1.0f / 30.0f), std::max(splatR->animationDuration(), 0.001f));
        splatR->setMorphTime(morphT);
    }

    auto tRender = std::chrono::high_resolution_clock::now();
    splatR->render(ctx);
    double renderMs = msSince(tRender);

    // Task 2 (Reconstruction-mode time budget, docs/rendering-engines.md):
    // filled in by the input_assembly.run() call just below when isProxy
    // (always true in Reconstruction mode), left zero otherwise. Declared
    // here (not inside the `if (isProxy)` block) so it's still in scope for
    // the RECON_TIMING log after the Stage 6 dispatch below.
    InputAssembly::Timings iaTimings;

    // --- Stage 3: input assembly, proxy-mode frames only (Target mode has
    // no jitter/history contract to feed the network -- it exists purely
    // as this demo's ground-truth comparison target, per DemoMode's
    // comment). hiddenIn=nullptr until Stage 5 wires the real recurrent
    // state through; every frame still runs the full pass (not just the
    // dump frame) so its own timing shows up in profiling passes later.
    if (isProxy) {
        float cx = static_cast<float>(kProxyWidth) * 0.5f, cy = static_cast<float>(kProxyHeight) * 0.5f;
        state->inputAssembly.run(ctx, splatR->getOutputImage(), splatR->getExpectedDepthImage(),
                                  splatR->getMotionImage(), kProxyWidth, kProxyHeight, nullptr,
                                  frame.eye.x, frame.eye.y, frame.eye.z,
                                  frame.rAxis.x, frame.rAxis.y, frame.rAxis.z,
                                  frame.uAxis.x, frame.uAxis.y, frame.uAxis.z,
                                  frame.fAxis.x, frame.fAxis.y, frame.fAxis.z,
                                  frame.fx, frame.fy, cx, cy, jitterPxX, jitterPxY, &iaTimings);
    }

    // --- Stage 5: net inference with recurrent hidden feedback + live
    // reconstruct-with-memory capture (docs/rendering-engines.md). Runs the
    // net every proxy frame from kNetWarmupStart on (letting the zero-
    // initial hidden state settle for kReconWindowStart-kNetWarmupStart
    // frames before anything is dumped), and writes one
    // reconstruct_reference_dump.py-shaped frame per proxy frame during
    // [kReconWindowStart, +kReconWindowFrames) -- see the AppState field
    // comments. Runs unconditionally within that range regardless of
    // `state->mode` being Proxy (guaranteed true there, see the comment),
    // but the Target-render ground-truth capture below is independent of
    // the display auto-cycle entirely (renders scene.getSplatRenderer()
    // on-demand at the SAME orbit frame index, whatever `state->mode`
    // currently is).
    if (isProxy && state->netRunnerReady && state->frameCounter >= AppState::kNetWarmupStart &&
        state->frameCounter < AppState::kReconWindowStart + AppState::kReconWindowFrames) {
        NetRunner::RunTimingsMs t{};
        const float* hiddenPtr = state->hiddenState.empty() ? nullptr : state->hiddenState.data();
        const float* out = state->netRunner.run(state->inputAssembly.getOutputHostPtr(), hiddenPtr, t);
        uint32_t netW = state->inputAssembly.getNetW(), netH = state->inputAssembly.getNetH();
        uint32_t outCh = state->netRunner.getOutputChannels();
        uint32_t hiddenCh = AppState::kHiddenChannels;

        // Carry hidden_raw (the LAST hiddenCh channels of the packed output)
        // forward as next frame's hidden_in.
        state->hiddenState.assign(static_cast<size_t>(netW) * netH * hiddenCh, 0.0f);
        for (size_t p = 0; p < static_cast<size_t>(netW) * netH; p++) {
            memcpy(&state->hiddenState[p * hiddenCh], &out[p * outCh + (outCh - hiddenCh)], hiddenCh * sizeof(float));
        }

        int t0 = AppState::kReconWindowStart;
        int idx = state->frameCounter - t0;
        if (idx >= 0 && idx < AppState::kReconWindowFrames) {
            std::string dumpDir = basePath(state) + "/live_recon_dump";
            if (idx == 0) {
                mkdir(dumpDir.c_str(), 0755);
                copyFile(basePath(state) + "/assets/bg_sphere.npy", dumpDir + "/bg_sphere.npy");
                copyFile(basePath(state) + "/assets/texture.npy", dumpDir + "/texture.npy");
                copyFile(basePath(state) + "/assets/memory_head.npz", dumpDir + "/memory_head.npz");
                // proxy padded to netH*paramStride (272), matching the
                // "135->136"-style replicate pad InputAssembly applies
                // internally (see padRowsReplicate's comment).
                writeMetaJson(dumpDir + "/meta.json", /*s=*/2, /*k=*/4, AppState::kParamStride, hiddenCh,
                              kProxyWidth, netH * AppState::kParamStride, kWidth, netH * AppState::kParamStride * 2,
                              netW, netH, AppState::kReconWindowFrames, AppState::kTexChannels,
                              /*memHidden=*/16, /*texW=*/512, /*texH=*/256);
            }
            std::string suf = "_f" + std::to_string(idx) + ".npy";

            uint32_t paddedProxyH = netH * AppState::kParamStride;
            auto colorProxy = readColorRgb(ctx, splatR);  // [proxyH,proxyW,3] @ true 270 rows
            auto colorPadded = padRowsReplicate(colorProxy, kProxyHeight, kProxyWidth, 3, paddedProxyH);
            DlssIO::writeNpyFloat32(dumpDir + "/proxy_color" + suf, colorPadded, {paddedProxyH, kProxyWidth, 3});

            auto mvProxy = readMvProxy(ctx, splatR);
            auto mvPadded = padRowsReplicate(mvProxy, kProxyHeight, kProxyWidth, 2, paddedProxyH);
            DlssIO::writeNpyFloat32(dumpDir + "/mv_proxy" + suf, mvPadded, {paddedProxyH, kProxyWidth, 2});

            std::vector<float> jitterArr = {jitterTargetX, jitterTargetY};
            DlssIO::writeNpyFloat32(dumpDir + "/jitter" + suf, jitterArr, {2});

            std::vector<float> packed(out, out + static_cast<size_t>(netW) * netH * outCh);
            DlssIO::writeNpyFloat32(dumpDir + "/packed_params" + suf, packed, {netH, netW, outCh});

            // disocc at target res: nearest-upsample the net-res disocc
            // channel (Stage 3's InputAssembly tensor, channel 6) by
            // s*param_stride=4 -- see the header comment above
            // writeMetaJson() re: this approximation vs. the iOS reference's
            // proxy-res version.
            uint32_t targetW = kWidth, targetH = paddedProxyH * 2;
            const float* netTensor = state->inputAssembly.getOutputHostPtr();
            uint32_t ch26 = state->inputAssembly.getChannels();
            std::vector<float> disoccTarget(static_cast<size_t>(targetW) * targetH);
            uint32_t upFactor = AppState::kParamStride * 2;  // s * param_stride
            for (uint32_t y = 0; y < targetH; y++) {
                uint32_t gy = std::min(y / upFactor, netH - 1);
                for (uint32_t x = 0; x < targetW; x++) {
                    uint32_t gx = std::min(x / upFactor, netW - 1);
                    disoccTarget[static_cast<size_t>(y) * targetW + x] = netTensor[(static_cast<size_t>(gy) * netW + gx) * ch26 + 6];
                }
            }
            DlssIO::writeNpyFloat32(dumpDir + "/disocc" + suf, disoccTarget, {targetH, targetW});

            // k_params/cam_to_world at the reconstruct pass's OWN target
            // resolution (960x544, from padding -- see writeMetaJson's
            // header comment), a separate camera evaluation from the
            // displayed Target renderer's own 960x540 camera (computeOrbitFrame
            // is a pure function of (frameIndex,width,height); only height differs).
            OrbitFrame reconTargetFrame = computeOrbitFrame(static_cast<float>(state->frameCounter), targetW, targetH);
            std::vector<float> kParams = {reconTargetFrame.fx, reconTargetFrame.fy,
                                           targetW * 0.5f, targetH * 0.5f};
            DlssIO::writeNpyFloat32(dumpDir + "/k_params" + suf, kParams, {4});

            std::vector<float> camToWorld(16, 0.0f);
            camToWorld[0] = reconTargetFrame.rAxis.x; camToWorld[1] = reconTargetFrame.uAxis.x; camToWorld[2] = reconTargetFrame.fAxis.x; camToWorld[3] = reconTargetFrame.eye.x;
            camToWorld[4] = reconTargetFrame.rAxis.y; camToWorld[5] = reconTargetFrame.uAxis.y; camToWorld[6] = reconTargetFrame.fAxis.y; camToWorld[7] = reconTargetFrame.eye.y;
            camToWorld[8] = reconTargetFrame.rAxis.z; camToWorld[9] = reconTargetFrame.uAxis.z; camToWorld[10] = reconTargetFrame.fAxis.z; camToWorld[11] = reconTargetFrame.eye.z;
            camToWorld[15] = 1.0f;
            DlssIO::writeNpyFloat32(dumpDir + "/cam_to_world" + suf, camToWorld, {4, 4});

            // Ground-truth Target render at the SAME orbit frame index, for
            // the PSNR comparison -- independent of state->mode/the display
            // auto-cycle (a direct render() call on the Target SplatRenderer
            // instance, not a mode switch).
            SplatRenderer* targetR = state->scene.getSplatRenderer();
            OrbitFrame targetFrame = computeOrbitFrame(static_cast<float>(state->frameCounter), kWidth, kHeight);
            targetR->updateCameraExplicit(targetFrame.eye, targetFrame.viewGl, targetFrame.proj, targetFrame.fx, targetFrame.fy);
            if (targetR->hasMotion()) targetR->setMorphTime(morphT);
            targetR->render(ctx);
            auto targetColor = readColorRgb(ctx, targetR);  // [540,960,3]
            DlssIO::writeNpyFloat32(dumpDir + "/target_color" + suf, targetColor, {kHeight, kWidth, 3});

            LOGI("Stage5 capture frame %d/%d (app frame %d)", idx + 1, AppState::kReconWindowFrames, state->frameCounter);

            if (idx == AppState::kReconWindowFrames - 1 && !state->stage5Done) {
                state->stage5Done = true;
                std::string outDir = basePath(state) + "/live_recon_out";
                auto tRecon = std::chrono::high_resolution_clock::now();
                int rc = runReconstructDump(ctx, dumpDir, outDir, basePath(state) + "/examples/reconstruct_mem_ps2");
                double reconMs = msSince(tRecon);
                LOGI("Stage5 LIVE runReconstructDump: rc=%d elapsed=%.1fms out=%s", rc, reconMs, outDir.c_str());
            }
        }
    }

    // --- Stage 2 validation dump: one fixed proxy frame, diffed against
    // the Mac Vulkan CLI with the same camera/jitter/morph-time (see
    // dumpProxyDebugFrame's comment above). frame 30 is well past the
    // orbit's start (0.6deg/frame) and the scene's motion, so both MV and
    // colour carry real signal, not the degenerate all-zero first frame.
    //
    // VALIDATION RESULT (frames 1, 3, and 30 all checked against
    // playground_cpp/build/lux-playground --output-aux with matching
    // --camera-json/--camera-json-prev/--time/--time-prev/--jitter):
    // colour and expected-depth match closely (mean abs diff ~0.0016 and
    // ~1e-6 respectively -- consistent with ordinary cross-GPU float
    // rounding). Motion vectors do NOT match -- Android's mv is 3-6 orders
    // of magnitude too large (e.g. frame 30: max |mv| ~5.0e6 vs the Mac
    // CLI's ~8.6, over essentially the whole image, including
    // fully-opaque/high-confidence pixels) EVEN AT FRAME 1, the very first
    // frame with nonzero motion right after the firstMvFrame_ seed -- so
    // this is not slow numerical drift across many frames. Since
    // out_depth is computed by the exact same preprocess dispatch +
    // fragment shader and matches essentially exactly, the current-frame
    // camera/position math is provably correct; the bug is isolated to
    // whatever feeds "previous" camera/position into that dispatch when
    // it's populated by SplatRenderer::render()'s automatic per-frame
    // carry-forward (prevViewMatrix_/prevProjMatrixUnjittered_/
    // prevPosBuffer_), as opposed to the one-shot
    // setPreviousCameraExplicit()/seedPreviousMorphTime() path the
    // existing lux-4dgs tests (test_dlss_outputs.py) exercise instead --
    // this app may be the first continuous multi-frame (30+ render() calls
    // in one process) exerciser of that carry-forward path. NOT
    // (confirmed) Android/Mali-specific: not re-tested against the desktop
    // CLI's own interactive GLFW loop (playground_cpp/src/main.cpp) in
    // this pass. This blocks trusting motion vectors in Stage 3's 26-ch
    // input assembly and especially Stage 5's warp/reprojection until
    // root-caused -- flagged to the task owner rather than guessed at
    // further here.
    constexpr int kDumpFrame = 30;
    // Diagnostic (see the MV bug investigation above): also dump the
    // SceneManager-owned TARGET renderer at frame 181 (its own 2nd-ever
    // render() call, within the first target window 180-359) to check
    // whether the automatic prev-camera/prev-pos carry-forward is broken
    // for ANY continuously-run SplatRenderer, or specific to the
    // proxyRenderer_ this app constructs directly from splat_data.
    constexpr int kDumpFrameTarget = 181;
    // Stage 3 validation needs frame (kDumpFrame - 1)'s proxy depth too
    // (disocclusion_mask's "previous frame" input) -- dumped under its own
    // tag so it doesn't disturb the existing frame-30 comparison.
    if (isProxy && state->frameCounter == kDumpFrame - 1) {
        dumpProxyDebugFrame(ctx, splatR, basePath(state) + "/dump", "proxy_prev", state->frameCounter, morphT, jitterPxX, jitterPxY);
    }
    if (isProxy && state->frameCounter == kDumpFrame) {
        dumpProxyDebugFrame(ctx, splatR, basePath(state) + "/dump", "proxy", state->frameCounter, morphT, jitterPxX, jitterPxY);
        // Stage 3 validation: dump the packed 26-ch input tensor for the
        // SAME frame, to diff against mobiledlss.train.model.build_input
        // fed the Mac CLI's dumped proxy color/depth/mv for this frame.
        state->inputAssembly.dumpToNpy(basePath(state) + "/dump/android_input_tensor.npy");
        LOGI("DUMP input_tensor: %s/dump/android_input_tensor.npy (net=%ux%u ch=%u)",
             basePath(state).c_str(), state->inputAssembly.getNetW(), state->inputAssembly.getNetH(),
             state->inputAssembly.getChannels());

        // Stage 4: run the TFLite net on this same dumped tensor (zero
        // hidden-in, matching Stage 3/4's validation config) and dump its
        // packed output for comparison against torch_packed_output_f30.npy.
        if (state->netRunnerReady) {
            NetRunner::RunTimingsMs t{};
            const float* out = state->netRunner.run(state->inputAssembly.getOutputHostPtr(), nullptr, t);
            uint32_t netW = state->inputAssembly.getNetW(), netH = state->inputAssembly.getNetH();
            uint32_t outCh = state->netRunner.getOutputChannels();
            std::vector<float> outCopy(out, out + static_cast<size_t>(netW) * netH * outCh);
            DlssIO::writeNpyFloat32(basePath(state) + "/dump/android_net_output.npy", outCopy, {netH, netW, outCh});
            LOGI("NET timing (ms): adapter=%.3f upload=%.3f infer=%.3f download=%.3f total=%.3f "
                 "gpu_delegate=%d out_ch=%u",
                 t.adapterMs, t.uploadMs, t.inferMs, t.downloadMs,
                 t.adapterMs + t.uploadMs + t.inferMs + t.downloadMs,
                 state->netRunner.isGpuDelegateActive(), outCh);
        }
    } else if (!isProxy && state->frameCounter == kDumpFrameTarget) {
        dumpProxyDebugFrame(ctx, splatR, basePath(state) + "/dump", "target", state->frameCounter, morphT, jitterPxX, jitterPxY);
    }

    // --- Stage 6: Bicubic/Reconstruction produce their displayed image on
    // the CPU/via a separate dump-and-reconstruct round trip (see
    // bicubicUpsample2x/runReconstructionModeFrame above), then upload it
    // into AppState::displayImage for the blit below; Proxy/Target blit
    // straight from their own SplatRenderer's colorImage_, unchanged.
    double stage6Ms = 0.0;
    if (state->mode == DemoMode::Bicubic) {
        auto tS6 = std::chrono::high_resolution_clock::now();
        auto colorProxy = readColorRgb(ctx, splatR);
        auto upsampled = bicubicUpsample2x(colorProxy, kProxyWidth, kProxyHeight, kWidth, kHeight);
        uploadRgbToDisplayImage(ctx, state->displayImage, upsampled, kWidth, kHeight);
        stage6Ms = msSince(tS6);
    } else if (state->mode == DemoMode::Reconstruction) {
        auto tS6 = std::chrono::high_resolution_clock::now();
        ReconModeFrameTimingsMs reconT;
        auto reconColor = runReconstructionModeFrame(state, ctx, splatR, frame, jitterTargetX, jitterTargetY, &reconT);
        uploadRgbToDisplayImage(ctx, state->displayImage, reconColor, kWidth, kHeight);
        stage6Ms = msSince(tS6);
        // Task 2 (docs/rendering-engines.md, Reconstruction-mode time
        // budget): proxy render is `renderMs` (above); input assembly is
        // `iaTimings` (readback = 3x vkCmdCopyImageToBuffer, compute = the
        // 2 GLSL dispatches); net is `reconT.net` (adapter=CPU repack,
        // upload/infer/download=TFLite); the rest is runReconstructionModeFrame's
        // own file-based round trip through reconstruct_pass.cpp's
        // runReconstructDump (dumpWrite=write dump npys, reconstruct.*=its
        // internal setup/read/upload/dispatch[cpu+gpu]/download/write/
        // teardown breakdown, outputRead=read the final out_f0.npy back).
        const ReconstructTimingsMs& rt = reconT.reconstruct;
        double reconstructTotalMs = rt.setupMs + rt.fileReadMs + rt.uploadMs + rt.dispatchCpuMs +
                                     rt.downloadMs + rt.fileWriteMs + rt.teardownMs;
        LOGI("RECON_TIMING (ms) proxy_render=%.1f | ia_readback=%.1f ia_compute=%.1f | "
             "net_adapter=%.1f net_upload=%.1f net_infer=%.1f net_download=%.1f | "
             "dump_write=%.1f | recon_setup=%.1f recon_read=%.1f recon_upload=%.1f "
             "recon_dispatch_cpu=%.1f recon_dispatch_gpu=%.1f recon_download=%.1f "
             "recon_write=%.1f recon_teardown=%.1f recon_total=%.1f | output_read=%.1f | "
             "stage6_total=%.1f",
             renderMs, iaTimings.readbackMs, iaTimings.computeMs,
             reconT.net.adapterMs, reconT.net.uploadMs, reconT.net.inferMs, reconT.net.downloadMs,
             reconT.dumpWriteMs, rt.setupMs, rt.fileReadMs, rt.uploadMs, rt.dispatchCpuMs, rt.dispatchGpuMs,
             rt.downloadMs, rt.fileWriteMs, rt.teardownMs, reconstructTotalMs, reconT.outputReadMs, stage6Ms);
    }
    if (stage6Ms > 0.0) {
        LOGI("Stage6 %s frame render: %.1fms", demoModeName(state->mode), stage6Ms);
    }

    auto tBlit = std::chrono::high_resolution_clock::now();
    state->blitCmd = ctx.beginSingleTimeCommands();
    if (state->mode == DemoMode::Bicubic || state->mode == DemoMode::Reconstruction) {
        blitImageToSwapchain(state->blitCmd, state->displayImage, kWidth, kHeight,
                              ctx.swapchainImages[imageIndex], ctx.swapchainExtent);
    } else {
        splatR->blitToSwapchain(ctx, state->blitCmd, ctx.swapchainImages[imageIndex], ctx.swapchainExtent);
    }
    vkEndCommandBuffer(state->blitCmd);

    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = &state->imageAvailableSem;
    submitInfo.pWaitDstStageMask = &waitStage;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &state->blitCmd;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &state->renderFinishedSem;
    vkQueueSubmit(ctx.graphicsQueue, 1, &submitInfo, state->inFlightFence);

    VkPresentInfoKHR presentInfo{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &state->renderFinishedSem;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &ctx.swapchain;
    presentInfo.pImageIndices = &imageIndex;
    VkResult presentResult = vkQueuePresentKHR(ctx.graphicsQueue, &presentInfo);
    // NOTE: only recreate on OUT_OF_DATE, not SUBOPTIMAL -- this app
    // deliberately forces preTransform=IDENTITY at swapchain-creation time
    // (AndroidVulkan::createSwapchain, see the c1539ff pre-rotation fix)
    // while the device's surface capabilities report currentTransform=0x2
    // (ROTATE_90) for this orientation, so vkQueuePresentKHR legitimately
    // returns VK_SUBOPTIMAL_KHR on every single frame forever (the spec's
    // definition of "suboptimal" is exactly this: presentation still
    // succeeds correctly, just not through the ideal/most-efficient
    // compositor path). Treating that as "must recreate" like OUT_OF_DATE
    // caused a full vkDeviceWaitIdle + imageview/swapchain
    // destroy-and-recreate cycle EVERY frame -- confirmed via the Stage 1
    // timing breakdown (docs/rendering-engines.md) to cost ~30ms/frame
    // (~25% of the ~120ms frame budget) for zero benefit, since the very
    // next present is suboptimal again regardless. Real resizes/rotations
    // still get caught by the VK_ERROR_OUT_OF_DATE_KHR paths (both here and
    // in the vkAcquireNextImageKHR check above).
    if (presentResult == VK_ERROR_OUT_OF_DATE_KHR) {
        vkDeviceWaitIdle(ctx.device);
        for (auto& iv : ctx.swapchainImageViews) vkDestroyImageView(ctx.device, iv, nullptr);
        ctx.swapchainImageViews.clear();
        vkDestroySwapchainKHR(ctx.device, ctx.swapchain, nullptr);
        ctx.swapchain = VK_NULL_HANDLE;
        AndroidVulkan::createSwapchain(ctx, kWidth, kHeight);
    }

    double blitPresentMs = msSince(tBlit) - 0.0;  // encode+submit+present call (non-blocking; actual GPU blit cost is hidden behind next frame's wait-fence)
    double frameMs = msSince(tFrameStart);
    if (state->mode == DemoMode::Reconstruction) {
        // Companion to the RECON_TIMING line above (present cost is only
        // known after vkQueuePresentKHR, later in renderFrame() than
        // Stage 6) -- cpu_wait_fence is the PRIOR frame's blit+present
        // finishing (this frame's own present cost is non-blocking and
        // shows up as the NEXT frame's cpu_wait_fence instead, same as the
        // Stage-1 TIMING block above).
        LOGI("RECON_TIMING_PRESENT (ms) cpu_wait_fence=%.1f blit_present_submit=%.1f frame_total=%.1f",
             waitFenceMs, blitPresentMs, frameMs);
    }

    // --- Stage 1 timing breakdown accounting ---
    state->sumCpuWaitFenceMs += waitFenceMs;
    state->sumCpuRenderMs += renderMs;
    state->sumCpuBlitPresentMs += blitPresentMs;
    state->sumCpuFrameMs += frameMs;
    SplatRenderer::GpuTimingsMs gt = splatR->lastGpuTimingsMs();
    if (gt.valid) {
        state->sumGpuPreprocessMs += gt.preprocessMs;
        state->sumGpuSortMs += gt.sortMs;
        state->sumGpuDrawMs += gt.drawMs;
        state->gpuTimingValidFrames++;
    }
    state->timingFrameCount++;
    if (state->timingFrameCount >= AppState::kTimingWindowFrames) {
        int n = state->timingFrameCount;
        int gn = std::max(state->gpuTimingValidFrames, 1);
        LOGI("TIMING mode=%s %ux%u avg-over-%d-frames (ms): cpu_wait_fence=%.2f cpu_render(preprocess+sort+draw)=%.2f "
             "cpu_blit_present=%.2f cpu_frame_total=%.2f | gpu_preprocess=%.2f gpu_sort=%.2f gpu_draw=%.2f "
             "gpu_valid_frames=%d/%d | sync_stalls_per_frame: outer_vkWaitForFences=1 "
             "inner_vkQueueWaitIdle=%d (splat_renderer.cpp render(): 1 after preprocess [MV drain, "
             "hasMotionVectors=%d] + 1 final submit)",
             demoModeName(state->mode), activeW, activeH,
             n, state->sumCpuWaitFenceMs / n, state->sumCpuRenderMs / n,
             state->sumCpuBlitPresentMs / n, state->sumCpuFrameMs / n,
             state->sumGpuPreprocessMs / gn, state->sumGpuSortMs / gn, state->sumGpuDrawMs / gn,
             state->gpuTimingValidFrames, n,
             splatR->hasMotionVectors() ? 2 : 1, splatR->hasMotionVectors() ? 1 : 0);
        state->timingFrameCount = 0;
        state->sumCpuWaitFenceMs = state->sumCpuRenderMs = state->sumCpuBlitPresentMs = state->sumCpuFrameMs = 0.0;
        state->sumGpuPreprocessMs = state->sumGpuSortMs = state->sumGpuDrawMs = 0.0;
        state->gpuTimingValidFrames = 0;
    }

    state->frameCounter++;
    state->fpsWindowFrames++;
    auto now = std::chrono::high_resolution_clock::now();
    float elapsed = std::chrono::duration<float>(now - state->fpsWindowStart).count();
    if (elapsed >= 1.0f) {
        state->lastFps = state->fpsWindowFrames / elapsed;
        LOGI("FPS: %.1f (frame %d)", state->lastFps, state->frameCounter);
        state->fpsWindowFrames = 0;
        state->fpsWindowStart = now;
    }
}

void cleanupRenderer(AppState* state) {
    if (!state->vulkanReady) return;
    vkDeviceWaitIdle(state->ctx.device);
    if (state->blitCmd) vkFreeCommandBuffers(state->ctx.device, state->ctx.commandPool, 1, &state->blitCmd);
    if (state->imageAvailableSem) vkDestroySemaphore(state->ctx.device, state->imageAvailableSem, nullptr);
    if (state->renderFinishedSem) vkDestroySemaphore(state->ctx.device, state->renderFinishedSem, nullptr);
    if (state->inFlightFence) vkDestroyFence(state->ctx.device, state->inFlightFence, nullptr);
    if (state->scene.getSplatRenderer()) state->scene.getSplatRenderer()->cleanup(state->ctx);
    if (state->proxyRenderer) { state->proxyRenderer->cleanup(state->ctx); state->proxyRenderer.reset(); }
    if (state->displayImage) vmaDestroyImage(state->ctx.allocator, state->displayImage, state->displayAlloc);
    AndroidVulkan::cleanup(state->ctx);
    state->vulkanReady = false;
}

// Stage 6: tap-to-cycle input handler (android_native_app_glue's
// AInputQueue-backed callback, set as app->onInputEvent below). Any
// ACTION_UP motion event is treated as a tap (no drag/swipe
// discrimination -- a deliberate simplification given this demo's only
// interaction is "advance the mode"; `adb shell input tap X Y` drives the
// same path for scripted testing). Sets a flag consumed at the top of the
// next renderFrame() -- both this callback and renderFrame() run on the
// same thread (see android_main()'s loop), so no synchronization is needed.
int32_t onInputEvent(struct android_app* app, AInputEvent* event) {
    AppState* state = static_cast<AppState*>(app->userData);
    if (AInputEvent_getType(event) == AINPUT_EVENT_TYPE_MOTION) {
        int32_t action = AMotionEvent_getAction(event) & AMOTION_EVENT_ACTION_MASK;
        if (action == AMOTION_EVENT_ACTION_UP) {
            state->pendingModeAdvance = true;
            return 1;
        }
    }
    return 0;
}

void onAppCmd(struct android_app* app, int32_t cmd) {
    AppState* state = static_cast<AppState*>(app->userData);
    switch (cmd) {
        case APP_CMD_INIT_WINDOW:
            if (app->window != nullptr && !state->windowInitialized) {
                state->windowInitialized = true;
                initRenderer(state);
            }
            break;
        case APP_CMD_TERM_WINDOW:
            cleanupRenderer(state);
            state->windowInitialized = false;
            break;
        default:
            break;
    }
}

}  // namespace

void android_main(struct android_app* app) {
    AppState state;
    state.app = app;
    app->userData = &state;
    app->onAppCmd = onAppCmd;
    app->onInputEvent = onInputEvent;

    while (true) {
        int events;
        struct android_poll_source* source;
        // Non-blocking poll while rendering, blocking poll while paused (no
        // window / renderer not ready yet). Recomputed on EVERY inner-loop
        // iteration (not hoisted above the loop) -- vulkanReady can flip
        // from false to true mid-loop (inside source->process() ->
        // onAppCmd() -> initRenderer()), and a stale -1 timeout captured
        // before that would block on ALooper_pollOnce() forever afterward
        // since no further input/window events arrive without user
        // interaction, silently starving renderFrame() below.
        while (ALooper_pollOnce(state.vulkanReady ? 0 : -1, nullptr, &events,
                                 reinterpret_cast<void**>(&source)) >= 0) {
            if (source != nullptr) source->process(app, source);
            if (app->destroyRequested != 0) {
                cleanupRenderer(&state);
                return;
            }
        }
        if (state.initFailed) {
            LOGE("Renderer failed to initialize; idling.");
            state.initFailed = false;  // log once
        }
        renderFrame(&state);
    }
}
