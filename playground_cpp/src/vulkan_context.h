#pragma once

#include <vulkan/vulkan.h>
#include "vk_mem_alloc.h"
#include <vector>
#include <string>
#include <optional>

struct GLFWwindow;

class VulkanContext {
public:
    // Core Vulkan objects
    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    uint32_t graphicsQueueFamily = 0;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VmaAllocator allocator = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;

    // Swapchain (interactive mode only)
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    std::vector<VkImage> swapchainImages;
    std::vector<VkImageView> swapchainImageViews;
    VkFormat swapchainFormat = VK_FORMAT_B8G8R8A8_UNORM;
    VkExtent2D swapchainExtent = {0, 0};

    // Bindless descriptor support
    bool bindlessSupported = false;

    // Ray tracing support
    bool rtSupported = false;

    // RT function pointers
    PFN_vkCreateAccelerationStructureKHR pfnCreateAccelerationStructureKHR = nullptr;
    PFN_vkDestroyAccelerationStructureKHR pfnDestroyAccelerationStructureKHR = nullptr;
    PFN_vkGetAccelerationStructureBuildSizesKHR pfnGetAccelerationStructureBuildSizesKHR = nullptr;
    PFN_vkCmdBuildAccelerationStructuresKHR pfnCmdBuildAccelerationStructuresKHR = nullptr;
    PFN_vkGetAccelerationStructureDeviceAddressKHR pfnGetAccelerationStructureDeviceAddressKHR = nullptr;
    PFN_vkCreateRayTracingPipelinesKHR pfnCreateRayTracingPipelinesKHR = nullptr;
    PFN_vkGetRayTracingShaderGroupHandlesKHR pfnGetRayTracingShaderGroupHandlesKHR = nullptr;
    PFN_vkCmdTraceRaysKHR pfnCmdTraceRaysKHR = nullptr;

    // RT properties
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtPipelineProperties = {};

    // Mesh shader support
    bool meshShaderSupported = false;

    // Mesh shader function pointer
    PFN_vkCmdDrawMeshTasksEXT pfnCmdDrawMeshTasksEXT = nullptr;

    // Mesh shader properties
    VkPhysicalDeviceMeshShaderPropertiesEXT meshShaderProperties = {};

    // Debug utils label function pointers (for RenderDoc markers)
    PFN_vkCmdBeginDebugUtilsLabelEXT pfnCmdBeginDebugUtilsLabel = nullptr;
    PFN_vkCmdEndDebugUtilsLabelEXT pfnCmdEndDebugUtilsLabel = nullptr;

    // Debug utils label helpers
    void cmdBeginLabel(VkCommandBuffer cmd, const char* name, float r=0, float g=1, float b=0, float a=1);
    void cmdEndLabel(VkCommandBuffer cmd);

    // Check mesh shader support
    bool supportsMeshShader() const { return meshShaderSupported; }

    // Check bindless descriptor support
    bool supportsBindless() const { return bindlessSupported; }

    // Initialize Vulkan context
    void init(bool enableRT, bool headless, GLFWwindow* window = nullptr, bool forceValidation = false);

    // Create swapchain for interactive mode
    void createSwapchain(uint32_t width, uint32_t height);

    // Cleanup
    void cleanup();

    // Check RT support
    bool supportsRT() const { return rtSupported; }

    // Helper: begin/end single-time command buffer
    VkCommandBuffer beginSingleTimeCommands();
    void endSingleTimeCommands(VkCommandBuffer cmd);

    // Helper: get buffer device address
    VkDeviceAddress getBufferDeviceAddress(VkBuffer buffer);
};
