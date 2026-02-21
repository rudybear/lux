#pragma once

#include <vulkan/vulkan.h>
#include "vulkan_context.h"
#include <glm/glm.hpp>

class IRenderer {
public:
    virtual void render(VulkanContext& ctx) = 0;
    virtual void updateCamera(VulkanContext& ctx, glm::vec3 eye, glm::vec3 target,
                               glm::vec3 up, float fovY, float aspect,
                               float nearPlane, float farPlane) = 0;
    virtual void blitToSwapchain(VulkanContext& ctx, VkCommandBuffer cmd,
                                  VkImage swapImage, VkExtent2D extent) = 0;
    virtual VkImage getOutputImage() const = 0;
    virtual VkFormat getOutputFormat() const = 0;
    virtual uint32_t getWidth() const = 0;
    virtual uint32_t getHeight() const = 0;
    virtual void cleanup(VulkanContext& ctx) = 0;
    virtual ~IRenderer() = default;
};
