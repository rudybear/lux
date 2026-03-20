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

/// RT Gaussian Splat Renderer — renders Gaussian splat scenes using
/// ray tracing (3DGRT algorithm) instead of rasterization.
///
/// Uses an AABB BLAS per Gaussian with analytical ray-Gaussian intersection
/// in the intersection shader. Multi-round closest-hit accumulation in the
/// raygen shader evaluates SH for view-dependent color and alpha-composites
/// front-to-back.
///
/// Shader stages: raygen (.rgen) + intersection (.rint) + closest_hit (.rchit) + miss (.rmiss)
class RTSplatRenderer {
public:
    RTSplatRenderer() = default;
    ~RTSplatRenderer();

    /// Initialize the RT splat renderer.
    /// @param ctx       Vulkan context (device, queues, VMA allocator)
    /// @param data      Loaded Gaussian splat data (positions, rotations, scales, opacity, SH coeffs)
    /// @param shaderBase Base path to compiled shaders (e.g. "examples/gaussian_splat_rt")
    /// @param width     Output image width
    /// @param height    Output image height
    void init(VulkanContext& ctx, const GaussianSplatData& data,
              const std::string& shaderBase, uint32_t width, uint32_t height);

    void updateCamera(glm::vec3 eye, glm::vec3 target, glm::vec3 up,
                      float fovY, float aspect, float nearPlane, float farPlane);

    void render(VulkanContext& ctx);

    void blitToSwapchain(VulkanContext& ctx, VkCommandBuffer cmd,
                         VkImage swapImage, VkExtent2D extent);

    void cleanup(VulkanContext& ctx);

    VkImage getOutputImage() const { return storageImage_; }
    VkFormat getOutputFormat() const { return VK_FORMAT_R8G8B8A8_UNORM; }
    uint32_t getWidth() const { return width_; }
    uint32_t getHeight() const { return height_; }

private:
    uint32_t width_ = 0, height_ = 0;
    uint32_t numSplats_ = 0;
    uint32_t shDegree_ = 0;

    // --- Storage image (RT output) ---
    VkImage storageImage_ = VK_NULL_HANDLE;
    VmaAllocation storageAlloc_ = VK_NULL_HANDLE;
    VkImageView storageView_ = VK_NULL_HANDLE;

    // --- Acceleration structures ---
    VkAccelerationStructureKHR blas_ = VK_NULL_HANDLE;
    VkBuffer blasBuffer_ = VK_NULL_HANDLE;
    VmaAllocation blasAlloc_ = VK_NULL_HANDLE;

    VkAccelerationStructureKHR tlas_ = VK_NULL_HANDLE;
    VkBuffer tlasBuffer_ = VK_NULL_HANDLE;
    VmaAllocation tlasAlloc_ = VK_NULL_HANDLE;

    VkBuffer instanceBuffer_ = VK_NULL_HANDLE;
    VmaAllocation instanceAlloc_ = VK_NULL_HANDLE;

    VkBuffer blasScratchBuffer_ = VK_NULL_HANDLE;
    VmaAllocation blasScratchAlloc_ = VK_NULL_HANDLE;
    VkBuffer tlasScratchBuffer_ = VK_NULL_HANDLE;
    VmaAllocation tlasScratchAlloc_ = VK_NULL_HANDLE;

    // --- AABB buffer for BLAS ---
    VkBuffer aabbBuffer_ = VK_NULL_HANDLE;
    VmaAllocation aabbAlloc_ = VK_NULL_HANDLE;

    // --- RT pipeline ---
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;

    // --- Descriptor sets ---
    // Set 0: raygen (camera, tlas, image, SH buffers, pos)
    // Set 1: intersection (splat_pos, rot, scale, opacity)
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout0_ = VK_NULL_HANDLE;  // raygen
    VkDescriptorSetLayout setLayout1_ = VK_NULL_HANDLE;  // intersection
    VkDescriptorSet descSet0_ = VK_NULL_HANDLE;
    VkDescriptorSet descSet1_ = VK_NULL_HANDLE;

    // --- Shader Binding Table ---
    VkBuffer sbtBuffer_ = VK_NULL_HANDLE;
    VmaAllocation sbtAlloc_ = VK_NULL_HANDLE;
    VkStridedDeviceAddressRegionKHR raygenSBT_ = {};
    VkStridedDeviceAddressRegionKHR missSBT_ = {};
    VkStridedDeviceAddressRegionKHR hitSBT_ = {};
    VkStridedDeviceAddressRegionKHR callableSBT_ = {};

    // --- Camera UBO ---
    VkBuffer cameraBuffer_ = VK_NULL_HANDLE;
    VmaAllocation cameraAlloc_ = VK_NULL_HANDLE;

    // Camera state
    glm::mat4 invView_{1.0f};
    glm::mat4 invProj_{1.0f};

    // --- Splat GPU buffers ---
    VkBuffer posBuffer_ = VK_NULL_HANDLE;       VmaAllocation posAlloc_ = VK_NULL_HANDLE;
    VkBuffer rotBuffer_ = VK_NULL_HANDLE;       VmaAllocation rotAlloc_ = VK_NULL_HANDLE;
    VkBuffer scaleBuffer_ = VK_NULL_HANDLE;     VmaAllocation scaleAlloc_ = VK_NULL_HANDLE;
    VkBuffer opacityBuffer_ = VK_NULL_HANDLE;   VmaAllocation opacityAlloc_ = VK_NULL_HANDLE;
    std::vector<VkBuffer> shBuffers_;
    std::vector<VmaAllocation> shAllocs_;

    // --- Shader modules ---
    VkShaderModule rgenModule_ = VK_NULL_HANDLE;
    VkShaderModule rintModule_ = VK_NULL_HANDLE;
    VkShaderModule rchitModule_ = VK_NULL_HANDLE;
    VkShaderModule rmissModule_ = VK_NULL_HANDLE;

    // --- Internal methods ---
    void createStorageImage(VulkanContext& ctx);
    void createBuffers(VulkanContext& ctx, const GaussianSplatData& data);
    void computeAABBs(const GaussianSplatData& data, std::vector<VkAabbPositionsKHR>& aabbs);
    void createBLAS(VulkanContext& ctx, const std::vector<VkAabbPositionsKHR>& aabbs);
    void createTLAS(VulkanContext& ctx);
    void createRTPipeline(VulkanContext& ctx, const std::string& shaderBase);
    void createSBT(VulkanContext& ctx);
    void createDescriptorSet(VulkanContext& ctx);
    void updateCameraUBO(VulkanContext& ctx);
};
