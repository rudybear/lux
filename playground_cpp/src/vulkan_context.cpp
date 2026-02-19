#include "vulkan_context.h"
#include "VkBootstrap.h"

#include <GLFW/glfw3.h>
#include <iostream>
#include <stdexcept>
#include <cstring>

// VMA implementation — define in exactly one translation unit
#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

// stb_image_write implementation — define in exactly one translation unit
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT type,
    const VkDebugUtilsMessengerCallbackDataEXT* callbackData,
    void* userData) {
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        std::cerr << "[Vulkan] " << callbackData->pMessage << std::endl;
    }
    return VK_FALSE;
}

void VulkanContext::init(bool enableRT, bool headless, GLFWwindow* window) {
    // Build instance
    vkb::InstanceBuilder instanceBuilder;
    instanceBuilder.set_app_name("lux-playground")
        .require_api_version(1, 2, 0)
        .set_debug_callback(debugCallback);

#ifndef NDEBUG
    instanceBuilder.request_validation_layers(true);
#endif

    auto instanceResult = instanceBuilder.build();
    if (!instanceResult) {
        throw std::runtime_error("Failed to create Vulkan instance: " +
                                 instanceResult.error().message());
    }

    auto vkbInstance = instanceResult.value();
    instance = vkbInstance.instance;
    debugMessenger = vkbInstance.debug_messenger;

    // Create surface if not headless
    if (!headless && window) {
        VkResult surfResult = glfwCreateWindowSurface(instance, window, nullptr, &surface);
        if (surfResult != VK_SUCCESS) {
            throw std::runtime_error("Failed to create window surface");
        }
    }

    // Select physical device
    vkb::PhysicalDeviceSelector selector(vkbInstance);
    selector.set_minimum_version(1, 2);

    if (surface != VK_NULL_HANDLE) {
        selector.set_surface(surface);
    } else {
        selector.defer_surface_initialization();
    }

    // Request RT extensions if enabled
    if (enableRT) {
        selector.add_desired_extension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
        selector.add_desired_extension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
        selector.add_desired_extension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
        selector.add_desired_extension(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    }

    auto physResult = selector.select();
    if (!physResult) {
        throw std::runtime_error("Failed to select physical device: " +
                                 physResult.error().message());
    }

    auto vkbPhysDevice = physResult.value();
    physicalDevice = vkbPhysDevice.physical_device;

    // Check which RT extensions are actually available
    bool hasAccelStruct = false;
    bool hasRTPipeline = false;
    if (enableRT) {
        auto extensions = vkbPhysDevice.get_extensions();
        for (const auto& ext : extensions) {
            if (ext == VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)
                hasAccelStruct = true;
            if (ext == VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME)
                hasRTPipeline = true;
        }
        rtSupported = hasAccelStruct && hasRTPipeline;
    }

    if (enableRT && !rtSupported) {
        std::cout << "[warn] Ray tracing extensions not available on this GPU. "
                  << "RT features will be disabled." << std::endl;
    }

    // Build device
    vkb::DeviceBuilder deviceBuilder(vkbPhysDevice);

    // Enable Vulkan 1.2 features
    VkPhysicalDeviceVulkan12Features features12 = {};
    features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    features12.bufferDeviceAddress = VK_TRUE;
    features12.descriptorIndexing = VK_TRUE;
    deviceBuilder.add_pNext(&features12);

    // Enable RT features if supported
    VkPhysicalDeviceAccelerationStructureFeaturesKHR accelFeatures = {};
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtFeatures = {};

    if (rtSupported) {
        accelFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
        accelFeatures.accelerationStructure = VK_TRUE;
        deviceBuilder.add_pNext(&accelFeatures);

        rtFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
        rtFeatures.rayTracingPipeline = VK_TRUE;
        deviceBuilder.add_pNext(&rtFeatures);
    }

    auto deviceResult = deviceBuilder.build();
    if (!deviceResult) {
        throw std::runtime_error("Failed to create logical device: " +
                                 deviceResult.error().message());
    }

    auto vkbDevice = deviceResult.value();
    device = vkbDevice.device;

    // Get graphics queue
    auto queueResult = vkbDevice.get_queue(vkb::QueueType::graphics);
    if (!queueResult) {
        throw std::runtime_error("Failed to get graphics queue");
    }
    graphicsQueue = queueResult.value();

    auto queueIdxResult = vkbDevice.get_queue_index(vkb::QueueType::graphics);
    graphicsQueueFamily = queueIdxResult.value();

    // Create command pool
    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = graphicsQueueFamily;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create command pool");
    }

    // Create VMA allocator
    VmaAllocatorCreateInfo allocatorInfo = {};
    allocatorInfo.physicalDevice = physicalDevice;
    allocatorInfo.device = device;
    allocatorInfo.instance = instance;
    allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_2;
    allocatorInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;

    if (vmaCreateAllocator(&allocatorInfo, &allocator) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create VMA allocator");
    }

    // Load RT function pointers if supported
    if (rtSupported) {
        pfnCreateAccelerationStructureKHR = reinterpret_cast<PFN_vkCreateAccelerationStructureKHR>(
            vkGetDeviceProcAddr(device, "vkCreateAccelerationStructureKHR"));
        pfnDestroyAccelerationStructureKHR = reinterpret_cast<PFN_vkDestroyAccelerationStructureKHR>(
            vkGetDeviceProcAddr(device, "vkDestroyAccelerationStructureKHR"));
        pfnGetAccelerationStructureBuildSizesKHR = reinterpret_cast<PFN_vkGetAccelerationStructureBuildSizesKHR>(
            vkGetDeviceProcAddr(device, "vkGetAccelerationStructureBuildSizesKHR"));
        pfnCmdBuildAccelerationStructuresKHR = reinterpret_cast<PFN_vkCmdBuildAccelerationStructuresKHR>(
            vkGetDeviceProcAddr(device, "vkCmdBuildAccelerationStructuresKHR"));
        pfnGetAccelerationStructureDeviceAddressKHR = reinterpret_cast<PFN_vkGetAccelerationStructureDeviceAddressKHR>(
            vkGetDeviceProcAddr(device, "vkGetAccelerationStructureDeviceAddressKHR"));
        pfnCreateRayTracingPipelinesKHR = reinterpret_cast<PFN_vkCreateRayTracingPipelinesKHR>(
            vkGetDeviceProcAddr(device, "vkCreateRayTracingPipelinesKHR"));
        pfnGetRayTracingShaderGroupHandlesKHR = reinterpret_cast<PFN_vkGetRayTracingShaderGroupHandlesKHR>(
            vkGetDeviceProcAddr(device, "vkGetRayTracingShaderGroupHandlesKHR"));
        pfnCmdTraceRaysKHR = reinterpret_cast<PFN_vkCmdTraceRaysKHR>(
            vkGetDeviceProcAddr(device, "vkCmdTraceRaysKHR"));

        // Query RT pipeline properties
        rtPipelineProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
        VkPhysicalDeviceProperties2 props2 = {};
        props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        props2.pNext = &rtPipelineProperties;
        vkGetPhysicalDeviceProperties2(physicalDevice, &props2);

        // Verify all function pointers loaded
        if (!pfnCreateAccelerationStructureKHR || !pfnCmdTraceRaysKHR ||
            !pfnCreateRayTracingPipelinesKHR || !pfnGetRayTracingShaderGroupHandlesKHR) {
            std::cout << "[warn] Some RT function pointers failed to load. "
                      << "Disabling RT support." << std::endl;
            rtSupported = false;
        }
    }
}

void VulkanContext::createSwapchain(uint32_t width, uint32_t height) {
    vkb::SwapchainBuilder swapchainBuilder(physicalDevice, device, surface);
    auto swapResult = swapchainBuilder
        .set_desired_format({VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR})
        .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
        .set_desired_extent(width, height)
        .build();

    if (!swapResult) {
        throw std::runtime_error("Failed to create swapchain: " +
                                 swapResult.error().message());
    }

    auto vkbSwapchain = swapResult.value();
    swapchain = vkbSwapchain.swapchain;
    swapchainFormat = vkbSwapchain.image_format;
    swapchainExtent = vkbSwapchain.extent;
    swapchainImages = vkbSwapchain.get_images().value();
    swapchainImageViews = vkbSwapchain.get_image_views().value();
}

void VulkanContext::cleanup() {
    if (device) {
        vkDeviceWaitIdle(device);
    }

    for (auto& iv : swapchainImageViews) {
        vkDestroyImageView(device, iv, nullptr);
    }
    swapchainImageViews.clear();

    if (swapchain) {
        vkDestroySwapchainKHR(device, swapchain, nullptr);
        swapchain = VK_NULL_HANDLE;
    }

    if (allocator) {
        vmaDestroyAllocator(allocator);
        allocator = VK_NULL_HANDLE;
    }

    if (commandPool) {
        vkDestroyCommandPool(device, commandPool, nullptr);
        commandPool = VK_NULL_HANDLE;
    }

    if (device) {
        vkDestroyDevice(device, nullptr);
        device = VK_NULL_HANDLE;
    }

    if (surface) {
        vkDestroySurfaceKHR(instance, surface, nullptr);
        surface = VK_NULL_HANDLE;
    }

    if (debugMessenger) {
        auto func = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
        if (func) {
            func(instance, debugMessenger, nullptr);
        }
        debugMessenger = VK_NULL_HANDLE;
    }

    if (instance) {
        vkDestroyInstance(instance, nullptr);
        instance = VK_NULL_HANDLE;
    }
}

VkCommandBuffer VulkanContext::beginSingleTimeCommands() {
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device, &allocInfo, &cmd);

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    return cmd;
}

void VulkanContext::endSingleTimeCommands(VkCommandBuffer cmd) {
    vkEndCommandBuffer(cmd);

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);

    vkFreeCommandBuffers(device, commandPool, 1, &cmd);
}

VkDeviceAddress VulkanContext::getBufferDeviceAddress(VkBuffer buffer) {
    VkBufferDeviceAddressInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    info.buffer = buffer;
    return vkGetBufferDeviceAddress(device, &info);
}
