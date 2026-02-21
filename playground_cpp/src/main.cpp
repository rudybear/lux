#include "vulkan_context.h"
#include "spv_loader.h"
#include "raster_renderer.h"
#include "rt_renderer.h"
#include "scene_manager.h"
#include "renderer_interface.h"
#include "reflected_pipeline.h"
#include "scene.h"
#include "camera.h"
#include "screenshot.h"

#include <GLFW/glfw3.h>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <filesystem>
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace fs = std::filesystem;

// --------------------------------------------------------------------------
// Orbit camera for interactive mode
// --------------------------------------------------------------------------

struct OrbitCamera {
    float yaw = 0.0f;        // horizontal angle (radians)
    float pitch = 0.15f;     // vertical angle (radians), slight default elevation
    float distance = 3.0f;   // distance from target
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

static void mouseButtonCallback(GLFWwindow* window, int button, int action, int /*mods*/) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        g_orbit.dragging = (action == GLFW_PRESS);
        if (g_orbit.dragging) {
            glfwGetCursorPos(window, &g_orbit.lastX, &g_orbit.lastY);
        }
    }
}

static void cursorPosCallback(GLFWwindow* /*window*/, double xpos, double ypos) {
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
    g_orbit.distance *= (1.0f - 0.1f * static_cast<float>(yoff));
    g_orbit.distance = glm::clamp(g_orbit.distance, 0.01f, 1000.0f);
}

// --------------------------------------------------------------------------
// CLI argument parsing (scene/pipeline architecture)
// --------------------------------------------------------------------------

struct CLIOptions {
    std::string shaderBase;    // --pipeline value (resolved from scene if not given)
    std::string sceneSource;   // --scene value (required)
    uint32_t width = 512;
    uint32_t height = 512;
    std::string output = "output.png";
    bool interactive = false;
    bool headless = true;
};

static void printUsage(const char* program) {
    std::cout << "Usage: " << program << " [OPTIONS] [shader_base]\n"
              << "\n"
              << "Arguments:\n"
              << "  [shader_base]          Base path for .spv shader files (positional, legacy)\n"
              << "\n"
              << "Options:\n"
              << "  --scene <SOURCE>       Scene source: sphere, fullscreen, triangle, or path to .glb/.gltf\n"
              << "  --pipeline <BASE>      Compiled shader base path (auto-resolved from scene if omitted)\n"
              << "  --mode <MODE>          (Legacy) Rendering mode: triangle|fullscreen|pbr|rt\n"
              << "  --width <N>            Output width in pixels (default: 512)\n"
              << "  --height <N>           Output height in pixels (default: 512)\n"
              << "  --output <PATH>        Output PNG path (default: output.png)\n"
              << "  --interactive          Open GLFW preview window\n"
              << "  --headless             Offscreen render only (default)\n"
              << "  --help                 Show this help message\n"
              << std::endl;
}

static CLIOptions parseArgs(int argc, char* argv[]) {
    CLIOptions opts;
    std::string legacyMode;

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
        } else if (arg == "--mode" && i + 1 < argc) {
            // Legacy --mode support: map to --scene
            i++;
            legacyMode = argv[i];
            std::transform(legacyMode.begin(), legacyMode.end(), legacyMode.begin(), ::tolower);
        } else if (arg == "--width" && i + 1 < argc) {
            i++;
            opts.width = static_cast<uint32_t>(std::stoi(argv[i]));
        } else if (arg == "--height" && i + 1 < argc) {
            i++;
            opts.height = static_cast<uint32_t>(std::stoi(argv[i]));
        } else if (arg == "--output" && i + 1 < argc) {
            i++;
            opts.output = argv[i];
        } else if (arg == "--interactive") {
            opts.interactive = true;
            opts.headless = false;
        } else if (arg == "--headless") {
            opts.headless = true;
            opts.interactive = false;
        } else if (arg[0] != '-') {
            opts.shaderBase = arg;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            printUsage(argv[0]);
            std::exit(1);
        }
    }

    // Legacy --mode backwards compatibility: map mode to scene source
    if (!legacyMode.empty() && opts.sceneSource.empty()) {
        if (legacyMode == "triangle")        opts.sceneSource = "triangle";
        else if (legacyMode == "fullscreen") opts.sceneSource = "fullscreen";
        else if (legacyMode == "pbr")        opts.sceneSource = "sphere";
        else if (legacyMode == "rt")         opts.sceneSource = "sphere";
        else {
            std::cerr << "Unknown legacy mode: " << legacyMode << std::endl;
            std::exit(1);
        }
        std::cout << "[info] Legacy --mode \"" << legacyMode
                  << "\" mapped to --scene \"" << opts.sceneSource << "\"" << std::endl;
    }

    // Backwards compat: if no --scene, treat positional arg as pipeline base and use "sphere" as scene
    if (opts.sceneSource.empty()) {
        if (!opts.shaderBase.empty()) {
            opts.sceneSource = "sphere";  // default scene
        } else {
            std::cerr << "Error: --scene is required\n";
            printUsage(argv[0]);
            std::exit(1);
        }
    }

    return opts;
}

// --------------------------------------------------------------------------
// Pipeline resolution and render path detection
// --------------------------------------------------------------------------

static std::string resolveDefaultPipeline(const std::string& scene) {
    if (scene.size() > 4 && (scene.substr(scene.size()-4) == ".glb" || scene.substr(scene.size()-5) == ".gltf"))
        return "examples/gltf_pbr";
    if (scene == "fullscreen")
        throw std::runtime_error("--pipeline required for fullscreen scenes");
    if (scene == "triangle")
        return "examples/hello_triangle";
    return "examples/pbr_basic";
}

static std::string detectRenderPath(const std::string& base) {
    if (fs::exists(base + ".rgen.spv")) return "rt";
    if (fs::exists(base + ".vert.spv") && fs::exists(base + ".frag.spv")) return "raster";
    if (fs::exists(base + ".frag.spv")) return "fullscreen";
    throw std::runtime_error("No shader files found for: " + base);
}

// --------------------------------------------------------------------------
// Headless rendering
// --------------------------------------------------------------------------

static int runHeadless(const CLIOptions& opts) {
    std::string renderPath = detectRenderPath(opts.shaderBase);
    bool needRT = (renderPath == "rt");
    VkPipelineStageFlags dstStage = needRT
        ? VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR
        : VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

    std::cout << "[info] Headless render: scene=\"" << opts.sceneSource
              << "\" pipeline=\"" << opts.shaderBase
              << "\" path=" << renderPath
              << " (" << opts.width << "x" << opts.height << ")" << std::endl;

    // Initialize Vulkan context
    VulkanContext ctx;
    try {
        ctx.init(needRT, true);
    } catch (const std::exception& e) {
        std::cerr << "[error] Failed to initialize Vulkan: " << e.what() << std::endl;
        return 1;
    }

    if (needRT && !ctx.supportsRT()) {
        std::cerr << "[error] RT mode requested but ray tracing is not supported on this device." << std::endl;
        ctx.cleanup();
        return 1;
    }

    try {
        // Shared scene setup
        SceneManager scene;
        scene.loadScene(ctx, opts.sceneSource);

        // Determine vertex stride: for raster path, check if reflection wants 48-byte stride
        int vertexStride = 32;
        if (!needRT && renderPath == "raster") {
            // Check if vertex reflection requires 48-byte stride
            std::string vertJsonPath = opts.shaderBase + ".vert.json";
            if (fs::exists(vertJsonPath)) {
                auto vertRefl = parseReflectionJson(vertJsonPath);
                if (vertRefl.vertex_stride == 48) {
                    vertexStride = 48;
                }
            }
        }

        scene.uploadToGPU(ctx, vertexStride);
        scene.uploadTextures(ctx);
        scene.loadIBLAssets(ctx, dstStage);

        std::unique_ptr<IRenderer> renderer;
        if (needRT) {
            auto rt = std::make_unique<RTRenderer>();
            rt->init(ctx, opts.shaderBase + ".rgen.spv", opts.shaderBase + ".rmiss.spv",
                     opts.shaderBase + ".rchit.spv", opts.width, opts.height, scene);
            renderer = std::move(rt);
        } else {
            auto raster = std::make_unique<RasterRenderer>();
            raster->init(ctx, scene, opts.shaderBase, renderPath, opts.width, opts.height);
            renderer = std::move(raster);
        }

        renderer->render(ctx);

        Screenshot::saveImageToPNG(ctx, renderer->getOutputImage(), renderer->getOutputFormat(),
                                    renderer->getWidth(), renderer->getHeight(),
                                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, opts.output);

        renderer->cleanup(ctx);
        scene.cleanup(ctx);
    } catch (const std::exception& e) {
        std::cerr << "[error] Rendering failed: " << e.what() << std::endl;
        ctx.cleanup();
        return 1;
    }

    ctx.cleanup();
    std::cout << "[info] Done." << std::endl;
    return 0;
}

// --------------------------------------------------------------------------
// Interactive rendering
// --------------------------------------------------------------------------

static int runInteractive(CLIOptions opts) {
    // Use larger window for interactive mode if user didn't specify
    if (opts.width == 512 && opts.height == 512) {
        opts.width = 1024;
        opts.height = 768;
    }
    std::string renderPath = detectRenderPath(opts.shaderBase);
    bool needRT = (renderPath == "rt");
    VkPipelineStageFlags dstStage = needRT
        ? VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR
        : VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

    std::cout << "[info] Interactive render: scene=\"" << opts.sceneSource
              << "\" pipeline=\"" << opts.shaderBase
              << "\" path=" << renderPath
              << " (" << opts.width << "x" << opts.height << ")" << std::endl;

    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "[error] Failed to initialize GLFW" << std::endl;
        return 1;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    GLFWwindow* window = glfwCreateWindow(
        static_cast<int>(opts.width), static_cast<int>(opts.height),
        "Lux Playground", nullptr, nullptr);
    if (!window) {
        std::cerr << "[error] Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return 1;
    }

    // Initialize Vulkan context with surface
    VulkanContext ctx;
    try {
        ctx.init(needRT, false, window);
    } catch (const std::exception& e) {
        std::cerr << "[error] Failed to initialize Vulkan: " << e.what() << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    if (needRT && !ctx.supportsRT()) {
        std::cerr << "[error] RT mode requested but ray tracing is not supported on this device." << std::endl;
        ctx.cleanup();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    // Create swapchain
    try {
        ctx.createSwapchain(opts.width, opts.height);
    } catch (const std::exception& e) {
        std::cerr << "[error] Failed to create swapchain: " << e.what() << std::endl;
        ctx.cleanup();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    // Create synchronization objects
    VkSemaphore imageAvailableSem = VK_NULL_HANDLE;
    VkSemaphore renderFinishedSem = VK_NULL_HANDLE;
    VkFence inFlightFence = VK_NULL_HANDLE;

    VkSemaphoreCreateInfo semInfo = {};
    semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    vkCreateSemaphore(ctx.device, &semInfo, nullptr, &imageAvailableSem);
    vkCreateSemaphore(ctx.device, &semInfo, nullptr, &renderFinishedSem);

    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    vkCreateFence(ctx.device, &fenceInfo, nullptr, &inFlightFence);

    // Initialize shared scene
    SceneManager scene;
    std::unique_ptr<IRenderer> renderer;
    bool useRT = needRT;

    try {
        scene.loadScene(ctx, opts.sceneSource);

        // Determine vertex stride for raster
        int vertexStride = 32;
        if (!useRT && renderPath == "raster") {
            std::string vertJsonPath = opts.shaderBase + ".vert.json";
            if (fs::exists(vertJsonPath)) {
                auto vertRefl = parseReflectionJson(vertJsonPath);
                if (vertRefl.vertex_stride == 48) {
                    vertexStride = 48;
                }
            }
        }

        scene.uploadToGPU(ctx, vertexStride);
        scene.uploadTextures(ctx);
        scene.loadIBLAssets(ctx, dstStage);

        if (useRT) {
            auto rt = std::make_unique<RTRenderer>();
            std::string rgenPath = opts.shaderBase + ".rgen.spv";
            std::string rmissPath = opts.shaderBase + ".rmiss.spv";
            std::string rchitPath = opts.shaderBase + ".rchit.spv";
            rt->init(ctx, rgenPath, rmissPath, rchitPath,
                     opts.width, opts.height, scene);
            renderer = std::move(rt);
        } else {
            auto raster = std::make_unique<RasterRenderer>();
            raster->init(ctx, scene, opts.shaderBase, renderPath,
                         opts.width, opts.height);
            renderer = std::move(raster);
        }
    } catch (const std::exception& e) {
        std::cerr << "[error] Failed to initialize renderer: " << e.what() << std::endl;
        vkDestroySemaphore(ctx.device, imageAvailableSem, nullptr);
        vkDestroySemaphore(ctx.device, renderFinishedSem, nullptr);
        vkDestroyFence(ctx.device, inFlightFence, nullptr);
        ctx.cleanup();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    // Initialize orbit camera from scene auto-camera
    if (scene.hasSceneBounds()) {
        g_orbit.initFromAutoCamera(
            scene.getAutoEye(),
            scene.getAutoTarget(),
            scene.getAutoUp(),
            scene.getAutoFar());
    }

    // Register GLFW input callbacks
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);
    glfwSetScrollCallback(window, scrollCallback);

    std::cout << "[info] Starting render loop. Press ESC or close the window to quit." << std::endl;
    std::cout << "[info] Mouse: drag to orbit, scroll to zoom." << std::endl;

    VkCommandBuffer rtBlitCmd = VK_NULL_HANDLE; // Track RT blit cmd for deferred free

    // Render loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // Check for ESC key
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
            continue;
        }

        // Check for minimized window
        int fbWidth, fbHeight;
        glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
        if (fbWidth == 0 || fbHeight == 0) {
            glfwWaitEvents();
            continue;
        }

        // Wait for previous frame to finish
        vkWaitForFences(ctx.device, 1, &inFlightFence, VK_TRUE, UINT64_MAX);

        // Free previous RT blit command buffer (now safe after fence)
        if (rtBlitCmd != VK_NULL_HANDLE) {
            vkFreeCommandBuffers(ctx.device, ctx.commandPool, 1, &rtBlitCmd);
            rtBlitCmd = VK_NULL_HANDLE;
        }

        // Acquire swapchain image
        uint32_t imageIndex;
        VkResult acquireResult = vkAcquireNextImageKHR(
            ctx.device, ctx.swapchain, UINT64_MAX,
            imageAvailableSem, VK_NULL_HANDLE, &imageIndex);

        if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
            // Recreate swapchain
            vkDeviceWaitIdle(ctx.device);
            for (auto& iv : ctx.swapchainImageViews) {
                vkDestroyImageView(ctx.device, iv, nullptr);
            }
            ctx.swapchainImageViews.clear();
            vkDestroySwapchainKHR(ctx.device, ctx.swapchain, nullptr);
            ctx.swapchain = VK_NULL_HANDLE;
            ctx.createSwapchain(static_cast<uint32_t>(fbWidth), static_cast<uint32_t>(fbHeight));
            continue;
        }
        if (acquireResult != VK_SUCCESS && acquireResult != VK_SUBOPTIMAL_KHR) {
            std::cerr << "[error] Failed to acquire swapchain image" << std::endl;
            break;
        }

        vkResetFences(ctx.device, 1, &inFlightFence);

        // Update orbit camera matrices each frame
        {
            float aspect = static_cast<float>(ctx.swapchainExtent.width) /
                           static_cast<float>(ctx.swapchainExtent.height);
            renderer->updateCamera(ctx, g_orbit.getEye(), g_orbit.target,
                                   g_orbit.up, g_orbit.fovY, aspect,
                                   g_orbit.nearPlane, g_orbit.farPlane);
        }

        if (useRT) {
            // RT rendering: render to storage image (synchronous)
            renderer->render(ctx);

            // Blit RT output to swapchain with proper semaphore synchronization
            rtBlitCmd = ctx.beginSingleTimeCommands();
            VkCommandBuffer cmd = rtBlitCmd;

            renderer->blitToSwapchain(ctx, cmd, ctx.swapchainImages[imageIndex], ctx.swapchainExtent);

            vkEndCommandBuffer(cmd);

            // Submit with semaphore sync: wait on imageAvailable, signal renderFinished + fence
            VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            VkSubmitInfo submitInfo = {};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.waitSemaphoreCount = 1;
            submitInfo.pWaitSemaphores = &imageAvailableSem;
            submitInfo.pWaitDstStageMask = &waitStage;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &cmd;
            submitInfo.signalSemaphoreCount = 1;
            submitInfo.pSignalSemaphores = &renderFinishedSem;

            vkQueueSubmit(ctx.graphicsQueue, 1, &submitInfo, inFlightFence);
        } else {
            // Raster rendering to swapchain
            // Cast to RasterRenderer for renderToSwapchain (uses internal command buffer)
            auto* rasterPtr = static_cast<RasterRenderer*>(renderer.get());
            rasterPtr->renderToSwapchain(ctx,
                ctx.swapchainImages[imageIndex],
                ctx.swapchainImageViews[imageIndex],
                ctx.swapchainFormat, ctx.swapchainExtent,
                imageAvailableSem, renderFinishedSem, inFlightFence);
        }

        // Present (wait for rendering to finish before presenting)
        VkPresentInfoKHR presentInfo = {};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &renderFinishedSem;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &ctx.swapchain;
        presentInfo.pImageIndices = &imageIndex;

        VkResult presentResult = vkQueuePresentKHR(ctx.graphicsQueue, &presentInfo);
        if (presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR) {
            // Recreate swapchain on next frame
            vkDeviceWaitIdle(ctx.device);
            for (auto& iv : ctx.swapchainImageViews) {
                vkDestroyImageView(ctx.device, iv, nullptr);
            }
            ctx.swapchainImageViews.clear();
            vkDestroySwapchainKHR(ctx.device, ctx.swapchain, nullptr);
            ctx.swapchain = VK_NULL_HANDLE;
            glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
            ctx.createSwapchain(static_cast<uint32_t>(fbWidth), static_cast<uint32_t>(fbHeight));
        }
    }

    // Cleanup
    vkDeviceWaitIdle(ctx.device);

    renderer->cleanup(ctx);
    scene.cleanup(ctx);

    vkDestroySemaphore(ctx.device, imageAvailableSem, nullptr);
    vkDestroySemaphore(ctx.device, renderFinishedSem, nullptr);
    vkDestroyFence(ctx.device, inFlightFence, nullptr);

    ctx.cleanup();
    glfwDestroyWindow(window);
    glfwTerminate();

    std::cout << "[info] Done." << std::endl;
    return 0;
}

// --------------------------------------------------------------------------
// Entry point
// --------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    CLIOptions opts = parseArgs(argc, argv);

    // Resolve pipeline from scene if not explicitly given
    if (opts.shaderBase.empty()) {
        try {
            opts.shaderBase = resolveDefaultPipeline(opts.sceneSource);
            std::cout << "[info] Resolved pipeline: " << opts.shaderBase << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[error] " << e.what() << std::endl;
            return 1;
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
