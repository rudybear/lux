#include "vulkan_context.h"
#include "spv_loader.h"
#include "raster_renderer.h"
#include "rt_renderer.h"
#include "scene.h"
#include "camera.h"
#include "screenshot.h"

#include <GLFW/glfw3.h>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <cstring>

namespace fs = std::filesystem;

// --------------------------------------------------------------------------
// Rendering mode enumeration
// --------------------------------------------------------------------------

enum class PlaygroundMode {
    Triangle,
    Fullscreen,
    PBR,
    RT
};

static const char* modeToString(PlaygroundMode mode) {
    switch (mode) {
        case PlaygroundMode::Triangle:   return "triangle";
        case PlaygroundMode::Fullscreen: return "fullscreen";
        case PlaygroundMode::PBR:       return "pbr";
        case PlaygroundMode::RT:        return "rt";
    }
    return "unknown";
}

// --------------------------------------------------------------------------
// CLI argument parsing
// --------------------------------------------------------------------------

struct CLIOptions {
    std::string shaderBase;
    PlaygroundMode mode = PlaygroundMode::Triangle;
    bool modeSpecified = false;
    uint32_t width = 512;
    uint32_t height = 512;
    std::string output = "output.png";
    bool interactive = false;
    bool headless = true;
};

static void printUsage(const char* program) {
    std::cout << "Usage: " << program << " [OPTIONS] <shader_base>\n"
              << "\n"
              << "Arguments:\n"
              << "  <shader_base>   Base path for .spv shader files\n"
              << "\n"
              << "Options:\n"
              << "  --mode <MODE>   Rendering mode: triangle|fullscreen|pbr|rt\n"
              << "                  (auto-detected from available .spv files if omitted)\n"
              << "  --width <N>     Output width in pixels (default: 512)\n"
              << "  --height <N>    Output height in pixels (default: 512)\n"
              << "  --output <PATH> Output PNG path (default: output.png)\n"
              << "  --interactive   Open GLFW preview window\n"
              << "  --headless      Offscreen render only (default)\n"
              << "  --help          Show this help message\n"
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
        } else if (arg == "--mode" && i + 1 < argc) {
            i++;
            std::string modeStr = argv[i];
            std::transform(modeStr.begin(), modeStr.end(), modeStr.begin(), ::tolower);
            if (modeStr == "triangle") {
                opts.mode = PlaygroundMode::Triangle;
            } else if (modeStr == "fullscreen") {
                opts.mode = PlaygroundMode::Fullscreen;
            } else if (modeStr == "pbr") {
                opts.mode = PlaygroundMode::PBR;
            } else if (modeStr == "rt") {
                opts.mode = PlaygroundMode::RT;
            } else {
                std::cerr << "Unknown mode: " << modeStr << std::endl;
                std::exit(1);
            }
            opts.modeSpecified = true;
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

    if (opts.shaderBase.empty()) {
        std::cerr << "Error: No shader base path specified." << std::endl;
        printUsage(argv[0]);
        std::exit(1);
    }

    return opts;
}

// --------------------------------------------------------------------------
// Auto-detection of rendering mode from available .spv files
// --------------------------------------------------------------------------

static PlaygroundMode autoDetectMode(const std::string& shaderBase) {
    // Check for RT shaders
    if (fs::exists(shaderBase + ".rgen.spv")) {
        std::cout << "[info] Auto-detected mode: RT (found .rgen.spv)" << std::endl;
        return PlaygroundMode::RT;
    }

    bool hasVert = fs::exists(shaderBase + ".vert.spv");
    bool hasFrag = fs::exists(shaderBase + ".frag.spv");

    if (hasVert && hasFrag) {
        // Check if PBR mode (look for PBR indicators in the shader base name)
        std::string baseLower = shaderBase;
        std::transform(baseLower.begin(), baseLower.end(), baseLower.begin(), ::tolower);
        if (baseLower.find("pbr") != std::string::npos ||
            baseLower.find("material") != std::string::npos ||
            baseLower.find("sphere") != std::string::npos) {
            std::cout << "[info] Auto-detected mode: PBR (found .vert.spv + .frag.spv with PBR indicator)" << std::endl;
            return PlaygroundMode::PBR;
        }
        std::cout << "[info] Auto-detected mode: Triangle (found .vert.spv + .frag.spv)" << std::endl;
        return PlaygroundMode::Triangle;
    }

    if (hasFrag) {
        std::cout << "[info] Auto-detected mode: Fullscreen (found .frag.spv only)" << std::endl;
        return PlaygroundMode::Fullscreen;
    }

    std::cerr << "[error] Could not auto-detect mode. No shader files found for base: "
              << shaderBase << std::endl;
    std::cerr << "Searched for:" << std::endl;
    std::cerr << "  " << shaderBase << ".rgen.spv (RT)" << std::endl;
    std::cerr << "  " << shaderBase << ".vert.spv (Triangle/PBR)" << std::endl;
    std::cerr << "  " << shaderBase << ".frag.spv (Fullscreen/Triangle/PBR)" << std::endl;
    std::exit(1);
}

// --------------------------------------------------------------------------
// Headless rendering
// --------------------------------------------------------------------------

static int runHeadless(const CLIOptions& opts) {
    PlaygroundMode mode = opts.modeSpecified ? opts.mode : autoDetectMode(opts.shaderBase);
    bool needRT = (mode == PlaygroundMode::RT);

    std::cout << "[info] Headless mode: " << modeToString(mode)
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
        if (mode == PlaygroundMode::RT) {
            // RT rendering
            std::string rgenPath = opts.shaderBase + ".rgen.spv";
            std::string rmissPath = opts.shaderBase + ".rmiss.spv";
            std::string rchitPath = opts.shaderBase + ".rchit.spv";

            RTRenderer renderer;
            renderer.init(ctx, rgenPath, rmissPath, rchitPath, opts.width, opts.height);
            renderer.render(ctx);

            // Save screenshot
            Screenshot::saveImageToPNG(ctx,
                renderer.getOutputImage(), renderer.getOutputFormat(),
                renderer.getWidth(), renderer.getHeight(),
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                opts.output);

            renderer.cleanup(ctx);
        } else {
            // Raster rendering
            RasterMode rasterMode;
            std::string vertPath;
            std::string fragPath = opts.shaderBase + ".frag.spv";

            switch (mode) {
                case PlaygroundMode::Triangle:
                    rasterMode = RasterMode::Triangle;
                    vertPath = opts.shaderBase + ".vert.spv";
                    break;
                case PlaygroundMode::Fullscreen:
                    rasterMode = RasterMode::Fullscreen;
                    // vertPath left empty; RasterRenderer uses built-in fullscreen vert
                    break;
                case PlaygroundMode::PBR:
                    rasterMode = RasterMode::PBR;
                    vertPath = opts.shaderBase + ".vert.spv";
                    break;
                default:
                    rasterMode = RasterMode::Triangle;
                    vertPath = opts.shaderBase + ".vert.spv";
                    break;
            }

            RasterRenderer renderer;
            renderer.init(ctx, rasterMode, vertPath, fragPath, opts.width, opts.height);
            renderer.render(ctx);

            // Save screenshot
            Screenshot::saveImageToPNG(ctx,
                renderer.getOffscreenImage(), renderer.getOffscreenFormat(),
                renderer.getWidth(), renderer.getHeight(),
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                opts.output);

            renderer.cleanup(ctx);
        }
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

static int runInteractive(const CLIOptions& opts) {
    PlaygroundMode mode = opts.modeSpecified ? opts.mode : autoDetectMode(opts.shaderBase);
    bool needRT = (mode == PlaygroundMode::RT);

    std::cout << "[info] Interactive mode: " << modeToString(mode)
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

    // Initialize renderer
    RasterRenderer rasterRenderer;
    RTRenderer rtRenderer;
    bool useRT = (mode == PlaygroundMode::RT);

    try {
        if (useRT) {
            std::string rgenPath = opts.shaderBase + ".rgen.spv";
            std::string rmissPath = opts.shaderBase + ".rmiss.spv";
            std::string rchitPath = opts.shaderBase + ".rchit.spv";
            rtRenderer.init(ctx, rgenPath, rmissPath, rchitPath, opts.width, opts.height);
        } else {
            RasterMode rasterMode;
            std::string vertPath;
            std::string fragPath = opts.shaderBase + ".frag.spv";

            switch (mode) {
                case PlaygroundMode::Triangle:
                    rasterMode = RasterMode::Triangle;
                    vertPath = opts.shaderBase + ".vert.spv";
                    break;
                case PlaygroundMode::Fullscreen:
                    rasterMode = RasterMode::Fullscreen;
                    break;
                case PlaygroundMode::PBR:
                    rasterMode = RasterMode::PBR;
                    vertPath = opts.shaderBase + ".vert.spv";
                    break;
                default:
                    rasterMode = RasterMode::Triangle;
                    vertPath = opts.shaderBase + ".vert.spv";
                    break;
            }

            rasterRenderer.init(ctx, rasterMode, vertPath, fragPath, opts.width, opts.height);
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

    std::cout << "[info] Starting render loop. Press ESC or close the window to quit." << std::endl;

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

        if (useRT) {
            // RT rendering: render to storage image, then copy to swapchain
            rtRenderer.render(ctx);

            // Copy RT output to swapchain image
            VkCommandBuffer cmd = ctx.beginSingleTimeCommands();

            // Transition swapchain image to TRANSFER_DST
            VkImageMemoryBarrier barrier = {};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = ctx.swapchainImages[imageIndex];
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            barrier.subresourceRange.baseMipLevel = 0;
            barrier.subresourceRange.levelCount = 1;
            barrier.subresourceRange.baseArrayLayer = 0;
            barrier.subresourceRange.layerCount = 1;
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 0, nullptr, 0, nullptr, 1, &barrier);

            // Blit RT output to swapchain
            VkImageBlit blitRegion = {};
            blitRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            blitRegion.srcSubresource.layerCount = 1;
            blitRegion.srcOffsets[1] = {
                static_cast<int32_t>(rtRenderer.getWidth()),
                static_cast<int32_t>(rtRenderer.getHeight()), 1
            };
            blitRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            blitRegion.dstSubresource.layerCount = 1;
            blitRegion.dstOffsets[1] = {
                static_cast<int32_t>(ctx.swapchainExtent.width),
                static_cast<int32_t>(ctx.swapchainExtent.height), 1
            };

            vkCmdBlitImage(cmd,
                rtRenderer.getOutputImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                ctx.swapchainImages[imageIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1, &blitRegion, VK_FILTER_LINEAR);

            // Transition swapchain image to PRESENT
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = 0;

            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                0, 0, nullptr, 0, nullptr, 1, &barrier);

            ctx.endSingleTimeCommands(cmd);
        } else {
            // Raster rendering to swapchain
            rasterRenderer.renderToSwapchain(ctx,
                ctx.swapchainImages[imageIndex],
                ctx.swapchainImageViews[imageIndex],
                ctx.swapchainFormat, ctx.swapchainExtent,
                imageAvailableSem, renderFinishedSem, inFlightFence);
        }

        // Present
        VkPresentInfoKHR presentInfo = {};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
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

    if (useRT) {
        rtRenderer.cleanup(ctx);
    } else {
        rasterRenderer.cleanup(ctx);
    }

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
