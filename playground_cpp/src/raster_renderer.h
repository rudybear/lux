#pragma once

#include <vulkan/vulkan.h>
#include "vk_mem_alloc.h"
#include "vulkan_context.h"
#include "scene.h"
#include "gltf_loader.h"
#include "reflected_pipeline.h"
#include <string>
#include <vector>
#include <unordered_map>

class RasterRenderer {
public:
    // Three-phase initialization: scene/pipeline architecture
    void init(VulkanContext& ctx, const std::string& sceneSource,
              const std::string& pipelineBase, const std::string& renderPath,
              uint32_t width, uint32_t height);

    // Three-phase rendering methods
    void uploadScene(VulkanContext& ctx, const std::string& sceneSource);
    void createPipeline(VulkanContext& ctx, const std::string& pipelineBase, const std::string& renderPath);
    void bindSceneToPipeline(VulkanContext& ctx);

    void render(VulkanContext& ctx);

    // Get the offscreen image for screenshot
    VkImage getOffscreenImage() const { return offscreenImage; }
    VkFormat getOffscreenFormat() const { return VK_FORMAT_R8G8B8A8_UNORM; }
    uint32_t getWidth() const { return renderWidth; }
    uint32_t getHeight() const { return renderHeight; }

    // Render to a swapchain image (interactive mode)
    void renderToSwapchain(VulkanContext& ctx, VkImage swapImage, VkImageView swapView,
                           VkFormat swapFormat, VkExtent2D extent,
                           VkSemaphore waitSem, VkSemaphore signalSem, VkFence fence);

    void cleanup(VulkanContext& ctx);

private:
    std::string m_sceneSource;
    std::string m_renderPath;  // "raster", "fullscreen", "rt"
    std::string m_pipelineBase; // pipeline base path for reflection JSON lookup
    uint32_t renderWidth = 512;
    uint32_t renderHeight = 512;

    // Offscreen render target
    VkImage offscreenImage = VK_NULL_HANDLE;
    VmaAllocation offscreenAllocation = VK_NULL_HANDLE;
    VkImageView offscreenImageView = VK_NULL_HANDLE;

    // Depth buffer (raster scenes with depth)
    VkImage depthImage = VK_NULL_HANDLE;
    VmaAllocation depthAllocation = VK_NULL_HANDLE;
    VkImageView depthImageView = VK_NULL_HANDLE;
    bool m_needsDepth = false;

    // Render pass and framebuffer
    VkRenderPass renderPass = VK_NULL_HANDLE;
    VkFramebuffer framebuffer = VK_NULL_HANDLE;

    // Pipeline
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;

    // Reflection-driven descriptor sets (dynamic, not hardcoded)
    std::unordered_map<int, VkDescriptorSetLayout> reflectedSetLayouts;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    std::unordered_map<int, VkDescriptorSet> reflectedDescSets;

    // Uniform buffers (created based on reflection)
    VkBuffer mvpBuffer = VK_NULL_HANDLE;
    VmaAllocation mvpAllocation = VK_NULL_HANDLE;
    VkBuffer lightBuffer = VK_NULL_HANDLE;
    VmaAllocation lightAllocation = VK_NULL_HANDLE;

    // Mesh data (scene GPU resources)
    GPUMesh mesh = {};

    // Textures: keyed by binding name from reflection JSON
    // (e.g. "base_color_tex", "normal_tex", "metallic_roughness_tex", etc.)
    std::unordered_map<std::string, GPUTexture> namedTextures;

    // IBL cubemap textures: keyed by "env_specular", "env_irradiance", "brdf_lut"
    std::unordered_map<std::string, GPUTexture> iblTextures;

    // Default 1x1 white texture for missing bindings
    GPUTexture defaultWhiteTexture = {};

    // Triangle vertex buffer
    VkBuffer triangleVB = VK_NULL_HANDLE;
    VmaAllocation triangleVBAllocation = VK_NULL_HANDLE;

    // Shader modules
    VkShaderModule vertModule = VK_NULL_HANDLE;
    VkShaderModule fragModule = VK_NULL_HANDLE;

    // Reflection data cached from JSON parsing
    ReflectionData vertReflection;
    ReflectionData fragReflection;

    // Push constant data (from reflection, filled with scene lighting)
    std::vector<uint8_t> pushConstantData;
    VkShaderStageFlags pushConstantStageFlags = 0;
    uint32_t pushConstantSize = 0;

    // glTF scene data (for texture extraction)
    GltfScene m_gltfScene;
    bool m_hasGltfScene = false;

    void createOffscreenTarget(VulkanContext& ctx);
    void createRenderPass(VulkanContext& ctx);
    void createFramebuffer(VulkanContext& ctx);
    void createPipelineTriangle(VulkanContext& ctx);
    void createPipelineFullscreen(VulkanContext& ctx);
    void createPipelinePBR(VulkanContext& ctx);
    void setupPBRResources(VulkanContext& ctx);

    // Reflection-driven descriptor setup
    void setupReflectedDescriptors(VulkanContext& ctx);
    void createDefaultWhiteTexture(VulkanContext& ctx);
    void uploadGltfTextures(VulkanContext& ctx);
    void loadIBLAssets(VulkanContext& ctx, const std::string& iblName);
    GPUTexture uploadCubemapF16(VulkanContext& ctx, uint32_t faceSize, uint32_t mipCount,
                                const std::vector<uint16_t>& data);
    GPUTexture& getTextureForBinding(const std::string& name);
};
