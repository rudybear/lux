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
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "vulkan_context.h"
#include "scene_manager.h"
#include "splat_renderer.h"
#include "dlss_io.h"
#include "android_vulkan_context.h"

#define LOG_TAG "lux_android"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

// Display resolution for Stage A (task spec: 960x540 display target).
constexpr uint32_t kWidth = 960;
constexpr uint32_t kHeight = 540;

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
    return {eye, viewGl, proj, fx, fy};
}

struct AppState {
    struct android_app* app = nullptr;
    bool vulkanReady = false;
    bool windowInitialized = false;

    VulkanContext ctx;
    SceneManager scene;

    VkSemaphore imageAvailableSem = VK_NULL_HANDLE;
    VkSemaphore renderFinishedSem = VK_NULL_HANDLE;
    VkFence inFlightFence = VK_NULL_HANDLE;
    VkCommandBuffer blitCmd = VK_NULL_HANDLE;

    int frameCounter = 0;

    std::chrono::high_resolution_clock::time_point fpsWindowStart;
    int fpsWindowFrames = 0;
    float lastFps = 0.0f;

    bool initFailed = false;
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

void renderFrame(AppState* state) {
    if (!state->vulkanReady) return;
    VulkanContext& ctx = state->ctx;

    vkWaitForFences(ctx.device, 1, &state->inFlightFence, VK_TRUE, UINT64_MAX);
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

    SplatRenderer* splatR = state->scene.getSplatRenderer();

    OrbitFrame frame = computeOrbitFrame(static_cast<float>(state->frameCounter), kWidth, kHeight);
    splatR->updateCameraExplicit(frame.eye, frame.viewGl, frame.proj, frame.fx, frame.fy);

    if (splatR->hasMotion()) {
        float t = std::fmod(state->frameCounter * (1.0f / 30.0f), std::max(splatR->animationDuration(), 0.001f));
        splatR->setMorphTime(t);
    }

    splatR->render(ctx);

    state->blitCmd = ctx.beginSingleTimeCommands();
    splatR->blitToSwapchain(ctx, state->blitCmd, ctx.swapchainImages[imageIndex], ctx.swapchainExtent);
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
    if (presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR) {
        vkDeviceWaitIdle(ctx.device);
        for (auto& iv : ctx.swapchainImageViews) vkDestroyImageView(ctx.device, iv, nullptr);
        ctx.swapchainImageViews.clear();
        vkDestroySwapchainKHR(ctx.device, ctx.swapchain, nullptr);
        ctx.swapchain = VK_NULL_HANDLE;
        AndroidVulkan::createSwapchain(ctx, kWidth, kHeight);
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
    AndroidVulkan::cleanup(state->ctx);
    state->vulkanReady = false;
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
