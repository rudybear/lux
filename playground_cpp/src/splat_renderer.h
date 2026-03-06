#pragma once

#include <vulkan/vulkan.h>
#include "vk_mem_alloc.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <string>
#include <cstdint>

struct VulkanContext;
struct GaussianSplatData;

class SplatRenderer {
public:
    SplatRenderer() = default;
    ~SplatRenderer();

    void init(VulkanContext& ctx, const GaussianSplatData& data,
              const std::string& shaderBase, uint32_t width, uint32_t height);

    void updateCamera(glm::vec3 eye, glm::vec3 target, glm::vec3 up,
                      float fovY, float aspect, float nearPlane, float farPlane);

    void render(VulkanContext& ctx);

    void blitToSwapchain(VulkanContext& ctx, VkCommandBuffer cmd,
                         VkImage swapImage, VkExtent2D extent);

    void cleanup(VulkanContext& ctx);

    VkImage getOutputImage() const { return colorImage_; }
    VkFormat getOutputFormat() const { return VK_FORMAT_R8G8B8A8_UNORM; }
    uint32_t getWidth() const { return width_; }
    uint32_t getHeight() const { return height_; }

private:
    uint32_t width_ = 0, height_ = 0;
    uint32_t numSplats_ = 0;

    // Offscreen render targets
    VkImage colorImage_ = VK_NULL_HANDLE;
    VmaAllocation colorAlloc_ = VK_NULL_HANDLE;
    VkImageView colorView_ = VK_NULL_HANDLE;

    VkImage depthImage_ = VK_NULL_HANDLE;
    VmaAllocation depthAlloc_ = VK_NULL_HANDLE;
    VkImageView depthView_ = VK_NULL_HANDLE;

    VkRenderPass renderPass_ = VK_NULL_HANDLE;
    VkFramebuffer framebuffer_ = VK_NULL_HANDLE;

    // Pipelines
    VkPipeline computePipeline_ = VK_NULL_HANDLE;
    VkPipeline renderPipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout computeLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout renderLayout_ = VK_NULL_HANDLE;

    // Descriptor sets
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout computeSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout renderSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSet computeDescSet_ = VK_NULL_HANDLE;
    VkDescriptorSet renderDescSet_ = VK_NULL_HANDLE;

    // Splat GPU buffers (VMA)
    VkBuffer posBuffer_ = VK_NULL_HANDLE;       VmaAllocation posAlloc_ = VK_NULL_HANDLE;
    VkBuffer rotBuffer_ = VK_NULL_HANDLE;       VmaAllocation rotAlloc_ = VK_NULL_HANDLE;
    VkBuffer scaleBuffer_ = VK_NULL_HANDLE;     VmaAllocation scaleAlloc_ = VK_NULL_HANDLE;
    VkBuffer opacityBuffer_ = VK_NULL_HANDLE;   VmaAllocation opacityAlloc_ = VK_NULL_HANDLE;
    std::vector<VkBuffer> shBuffers_;
    std::vector<VmaAllocation> shAllocs_;

    // Projected output buffers
    VkBuffer projCenterBuffer_ = VK_NULL_HANDLE;  VmaAllocation projCenterAlloc_ = VK_NULL_HANDLE;
    VkBuffer projConicBuffer_ = VK_NULL_HANDLE;   VmaAllocation projConicAlloc_ = VK_NULL_HANDLE;
    VkBuffer projColorBuffer_ = VK_NULL_HANDLE;   VmaAllocation projColorAlloc_ = VK_NULL_HANDLE;

    // Sort buffers
    VkBuffer sortKeysBuffer_ = VK_NULL_HANDLE;      VmaAllocation sortKeysAlloc_ = VK_NULL_HANDLE;
    VkBuffer sortedIndicesBuffer_ = VK_NULL_HANDLE;  VmaAllocation sortedIndicesAlloc_ = VK_NULL_HANDLE;

    // Visible count (atomic counter)
    VkBuffer visibleCountBuffer_ = VK_NULL_HANDLE;  VmaAllocation visibleCountAlloc_ = VK_NULL_HANDLE;

    // Camera state
    glm::mat4 viewMatrix_{1.0f};
    glm::mat4 projMatrix_{1.0f};
    glm::vec3 camPos_{0.0f, 0.0f, 3.0f};
    float focalX_ = 256.0f;
    float focalY_ = 256.0f;
    uint32_t shDegree_ = 0;

    // Cached positions for CPU sort
    std::vector<float> hostPositions_;

    // Helpers
    void createOffscreenTarget(VulkanContext& ctx);
    void createRenderPass(VkDevice device);
    void createFramebuffer(VkDevice device);
    void createPipelines(VkDevice device, const std::string& shaderBase);
    void createBuffers(VulkanContext& ctx, const GaussianSplatData& data);
};
