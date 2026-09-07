#include "android_vulkan_context.h"

#define VK_USE_PLATFORM_ANDROID_KHR
#include <vulkan/vulkan.h>
#include <android/native_window.h>
#include <android/log.h>

#include "VkBootstrap.h"

// VMA_IMPLEMENTATION / STB_IMAGE_WRITE_IMPLEMENTATION live here (exactly one
// TU), matching the desktop convention in vulkan_context.cpp (excluded from
// the Android build because it #includes GLFW).
//
// The NDK's libvulkan.so stub linked at build time only statically exports
// the subset of core entry points guaranteed by ANDROID_PLATFORM (here
// android-29, ~Vulkan 1.1); it does NOT export functions promoted to core
// in 1.2/1.3 (vkGetBufferDeviceAddress, vkGetDeviceBufferMemoryRequirements,
// etc.) even though the actual on-device driver (Tensor G4 / Mali-G715,
// Vulkan 1.4) supports them -- those must be resolved dynamically via
// vkGetInstanceProcAddr/vkGetDeviceProcAddr instead of static linking.
// VMA_DYNAMIC_VULKAN_FUNCTIONS makes VMA do exactly that internally.
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <stdexcept>
#include <cstring>

#define LOG_TAG "lux_android"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {
PFN_vkGetBufferDeviceAddress g_pfnGetBufferDeviceAddress = nullptr;
}  // namespace

namespace AndroidVulkan {

void init(VulkanContext& ctx, ANativeWindow* window, bool forceValidation) {
    vkb::InstanceBuilder instanceBuilder;
    instanceBuilder.set_app_name("lux-playground-android")
        .require_api_version(1, 2, 0);

#ifndef NDEBUG
    instanceBuilder.request_validation_layers(true);
#else
    if (forceValidation) {
        instanceBuilder.request_validation_layers(true);
    }
#endif

    auto instanceResult = instanceBuilder.build();
    if (!instanceResult) {
        throw std::runtime_error("Failed to create Vulkan instance: " +
                                  instanceResult.error().message());
    }
    auto vkbInstance = instanceResult.value();
    ctx.instance = vkbInstance.instance;
    ctx.debugMessenger = vkbInstance.debug_messenger;

    // VK_KHR_android_surface surface from the ANativeWindow (vk-bootstrap's
    // InstanceBuilder auto-enables VK_KHR_surface + VK_KHR_android_surface
    // for a non-headless build when __ANDROID__ is defined).
    VkAndroidSurfaceCreateInfoKHR surfaceInfo{VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR};
    surfaceInfo.window = window;
    if (vkCreateAndroidSurfaceKHR(ctx.instance, &surfaceInfo, nullptr, &ctx.surface) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create VK_KHR_android_surface surface");
    }

    vkb::PhysicalDeviceSelector selector(vkbInstance);
    selector.set_minimum_version(1, 2);
    selector.set_surface(ctx.surface);

    auto physResult = selector.select();
    if (!physResult) {
        throw std::runtime_error("Failed to select physical device: " +
                                  physResult.error().message());
    }
    auto vkbPhysDevice = physResult.value();
    ctx.physicalDevice = vkbPhysDevice.physical_device;

    LOGI("Selected physical device: %s", vkbPhysDevice.properties.deviceName);
    LOGI("maxPushConstantsSize = %u bytes", vkbPhysDevice.properties.limits.maxPushConstantsSize);

    vkb::DeviceBuilder deviceBuilder(vkbPhysDevice);
    VkPhysicalDeviceVulkan12Features features12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    features12.bufferDeviceAddress = VK_TRUE;
    features12.descriptorIndexing = VK_TRUE;
    deviceBuilder.add_pNext(&features12);
    auto deviceResult = deviceBuilder.build();
    if (!deviceResult) {
        throw std::runtime_error("Failed to create logical device: " +
                                  deviceResult.error().message());
    }
    auto vkbDevice = deviceResult.value();
    ctx.device = vkbDevice.device;

    auto queueResult = vkbDevice.get_queue(vkb::QueueType::graphics);
    if (!queueResult) {
        throw std::runtime_error("Failed to get graphics queue");
    }
    ctx.graphicsQueue = queueResult.value();
    ctx.graphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.queueFamilyIndex = ctx.graphicsQueueFamily;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(ctx.device, &poolInfo, nullptr, &ctx.commandPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create command pool");
    }

    VmaVulkanFunctions vmaFuncs{};
    vmaFuncs.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    vmaFuncs.vkGetDeviceProcAddr = vkGetDeviceProcAddr;

    VmaAllocatorCreateInfo allocatorInfo{};
    allocatorInfo.physicalDevice = ctx.physicalDevice;
    allocatorInfo.device = ctx.device;
    allocatorInfo.instance = ctx.instance;
    allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_2;
    allocatorInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    allocatorInfo.pVulkanFunctions = &vmaFuncs;
    if (vmaCreateAllocator(&allocatorInfo, &ctx.allocator) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create VMA allocator");
    }

    ctx.pfnCmdBeginDebugUtilsLabel = reinterpret_cast<PFN_vkCmdBeginDebugUtilsLabelEXT>(
        vkGetDeviceProcAddr(ctx.device, "vkCmdBeginDebugUtilsLabelEXT"));
    ctx.pfnCmdEndDebugUtilsLabel = reinterpret_cast<PFN_vkCmdEndDebugUtilsLabelEXT>(
        vkGetDeviceProcAddr(ctx.device, "vkCmdEndDebugUtilsLabelEXT"));

    // vkGetBufferDeviceAddress was promoted to core in Vulkan 1.2; the NDK's
    // android-29 libvulkan.so stub doesn't statically export it (see the
    // VMA_DYNAMIC_VULKAN_FUNCTIONS comment above), so resolve it dynamically
    // for VulkanContext::getBufferDeviceAddress() below.
    g_pfnGetBufferDeviceAddress = reinterpret_cast<PFN_vkGetBufferDeviceAddress>(
        vkGetDeviceProcAddr(ctx.device, "vkGetBufferDeviceAddress"));

    LOGI("Android Vulkan context initialized (queue family %u)", ctx.graphicsQueueFamily);
}

void createSwapchain(VulkanContext& ctx, uint32_t width, uint32_t height) {
    // Android Vulkan pre-rotation gotcha: vk-bootstrap's SwapchainBuilder
    // defaults preTransform to VkSurfaceCapabilitiesKHR::currentTransform
    // when unset. Declaring preTransform == currentTransform tells the
    // platform "the pixels I hand you are ALREADY rotated by that much",
    // so the compositor does NOT apply any further correcting rotation --
    // but our render+blit path (splat_renderer.cpp's blitToSwapchain, an
    // axis-aligned vkCmdBlitImage, reused unmodified) never actually
    // pre-rotates anything. On this device currentTransform came back as a
    // 90 degree rotation (Mali-G715 pixel 9 pro xl, landscape-locked
    // activity on a portrait-native panel), so the untouched content got
    // displayed literally sideways -- a correctly-shaped landscape buffer
    // (matching ANativeWindow's 2244x1008) with its CONTENT still oriented
    // for a 0-degree transform. Force preTransform = IDENTITY explicitly
    // instead: this is always among a Vulkan surface's supportedTransforms
    // and tells the compositor to do the (cheap, GPU/hwcomposer-side)
    // rotate itself, matching what our simple axis-aligned blit assumes.
    VkSurfaceCapabilitiesKHR caps{};
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(ctx.physicalDevice, ctx.surface, &caps);
    LOGI("Surface currentTransform=0x%x, forcing preTransform=IDENTITY", caps.currentTransform);

    vkb::SwapchainBuilder swapchainBuilder(ctx.physicalDevice, ctx.device, ctx.surface);
    auto swapResult = swapchainBuilder
        .set_desired_format({VK_FORMAT_R8G8B8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
        .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
        .set_desired_extent(width, height)
        .set_pre_transform_flags(VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
        .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
        .build();
    if (!swapResult) {
        throw std::runtime_error("Failed to create swapchain: " + swapResult.error().message());
    }
    auto vkbSwapchain = swapResult.value();
    ctx.swapchain = vkbSwapchain.swapchain;
    ctx.swapchainFormat = vkbSwapchain.image_format;
    ctx.swapchainExtent = vkbSwapchain.extent;
    ctx.swapchainImages = vkbSwapchain.get_images().value();
    ctx.swapchainImageViews = vkbSwapchain.get_image_views().value();
    LOGI("Swapchain created: %ux%u, format=%d, %zu images", ctx.swapchainExtent.width,
         ctx.swapchainExtent.height, ctx.swapchainFormat, ctx.swapchainImages.size());
}

void cleanup(VulkanContext& ctx) {
    if (ctx.device) vkDeviceWaitIdle(ctx.device);
    for (auto& iv : ctx.swapchainImageViews) vkDestroyImageView(ctx.device, iv, nullptr);
    ctx.swapchainImageViews.clear();
    if (ctx.swapchain) { vkDestroySwapchainKHR(ctx.device, ctx.swapchain, nullptr); ctx.swapchain = VK_NULL_HANDLE; }
    if (ctx.allocator) { vmaDestroyAllocator(ctx.allocator); ctx.allocator = VK_NULL_HANDLE; }
    if (ctx.commandPool) { vkDestroyCommandPool(ctx.device, ctx.commandPool, nullptr); ctx.commandPool = VK_NULL_HANDLE; }
    if (ctx.device) { vkDestroyDevice(ctx.device, nullptr); ctx.device = VK_NULL_HANDLE; }
    if (ctx.surface) { vkDestroySurfaceKHR(ctx.instance, ctx.surface, nullptr); ctx.surface = VK_NULL_HANDLE; }
    if (ctx.debugMessenger) {
        auto func = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(ctx.instance, "vkDestroyDebugUtilsMessengerEXT"));
        if (func) func(ctx.instance, ctx.debugMessenger, nullptr);
        ctx.debugMessenger = VK_NULL_HANDLE;
    }
    if (ctx.instance) { vkDestroyInstance(ctx.instance, nullptr); ctx.instance = VK_NULL_HANDLE; }
}

}  // namespace AndroidVulkan

// --- VulkanContext member definitions ---
// These live in vulkan_context.cpp on desktop, which also #includes GLFW
// and can't be compiled for Android; the bodies are unrelated to
// windowing so they're duplicated here verbatim.

VkCommandBuffer VulkanContext::beginSingleTimeCommands() {
    VkCommandBufferAllocateInfo allocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device, &allocInfo, &cmd);

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);
    return cmd;
}

void VulkanContext::endSingleTimeCommands(VkCommandBuffer cmd) {
    vkEndCommandBuffer(cmd);

    VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);

    vkFreeCommandBuffers(device, commandPool, 1, &cmd);
}

VkDeviceAddress VulkanContext::getBufferDeviceAddress(VkBuffer buffer) {
    VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    info.buffer = buffer;
    if (!g_pfnGetBufferDeviceAddress) return 0;
    return g_pfnGetBufferDeviceAddress(device, &info);
}

void VulkanContext::cmdBeginLabel(VkCommandBuffer cmd, const char* name, float r, float g, float b, float a) {
    if (pfnCmdBeginDebugUtilsLabel) {
        VkDebugUtilsLabelEXT label{VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT};
        label.pLabelName = name;
        label.color[0] = r; label.color[1] = g; label.color[2] = b; label.color[3] = a;
        pfnCmdBeginDebugUtilsLabel(cmd, &label);
    }
}

void VulkanContext::cmdEndLabel(VkCommandBuffer cmd) {
    if (pfnCmdEndDebugUtilsLabel) {
        pfnCmdEndDebugUtilsLabel(cmd);
    }
}
