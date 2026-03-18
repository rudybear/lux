#include "splat_renderer.h"
#include "vulkan_context.h"
#include "gltf_loader.h"
#include "spv_loader.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;

// --------------------------------------------------------------------------
// SH degree helpers
// --------------------------------------------------------------------------

static uint32_t numShCoeffsForDegree(uint32_t degree) {
    // degree 0: 1 (DC), degree 1: 4, degree 2: 9, degree 3: 16
    switch (degree) {
        case 0: return 1;
        case 1: return 4;
        case 2: return 9;
        case 3: return 16;
        default: return 1;
    }
}

static uint32_t readShaderShDegree(const std::string& shaderBase) {
    // Read the gaussian_splatting.sh_degree from the compute reflection JSON
    std::string jsonPath = shaderBase + ".comp.json";
    if (!fs::exists(jsonPath)) return 0;
    std::ifstream f(jsonPath);
    if (!f.is_open()) return 0;
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    // Find "gaussian_splatting" section, then "sh_degree" within it
    auto gsPos = content.find("\"gaussian_splatting\"");
    if (gsPos == std::string::npos) return 0;
    auto shPos = content.find("\"sh_degree\"", gsPos);
    if (shPos == std::string::npos) return 0;
    auto colonPos = content.find(':', shPos + 11);
    if (colonPos == std::string::npos) return 0;
    colonPos++;
    while (colonPos < content.size() && (content[colonPos] == ' ' || content[colonPos] == '\t'))
        colonPos++;
    return static_cast<uint32_t>(std::atoi(content.c_str() + colonPos));
}

// --------------------------------------------------------------------------
// VMA buffer helper
// --------------------------------------------------------------------------

static void createVmaBuffer(VmaAllocator allocator, VkDeviceSize size,
                             VkBufferUsageFlags usage, VmaMemoryUsage memUsage,
                             VkBuffer& buffer, VmaAllocation& allocation) {
    VkBufferCreateInfo bufInfo = {};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size = std::max(size, VkDeviceSize(16)); // avoid zero-size buffers
    bufInfo.usage = usage;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = memUsage;

    if (vmaCreateBuffer(allocator, &bufInfo, &allocInfo, &buffer, &allocation, nullptr) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create VMA buffer");
    }
}

static void uploadVmaBuffer(VmaAllocator allocator, VmaAllocation allocation,
                             const void* data, VkDeviceSize size) {
    void* mapped = nullptr;
    vmaMapMemory(allocator, allocation, &mapped);
    std::memcpy(mapped, data, static_cast<size_t>(size));
    vmaUnmapMemory(allocator, allocation);
}

static void destroyVmaBuffer(VmaAllocator allocator, VkBuffer& buffer, VmaAllocation& alloc) {
    if (buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(allocator, buffer, alloc);
        buffer = VK_NULL_HANDLE;
        alloc = VK_NULL_HANDLE;
    }
}

// --------------------------------------------------------------------------
// Destructor
// --------------------------------------------------------------------------

SplatRenderer::~SplatRenderer() {
    // Resources should be cleaned up via cleanup(ctx) before destruction.
    // This is a safety net — VMA allocator is gone by now so we can't free here.
}

// --------------------------------------------------------------------------
// Offscreen render target (VMA)
// --------------------------------------------------------------------------

void SplatRenderer::createOffscreenTarget(VulkanContext& ctx) {
    // Color image
    VkImageCreateInfo imageInfo = {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    imageInfo.extent = {width_, height_, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    vmaCreateImage(ctx.allocator, &imageInfo, &allocInfo,
                   &colorImage_, &colorAlloc_, nullptr);

    VkImageViewCreateInfo viewInfo = {};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = colorImage_;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.layerCount = 1;

    vkCreateImageView(ctx.device, &viewInfo, nullptr, &colorView_);

    // Depth image
    VkImageCreateInfo depthInfo = imageInfo;
    depthInfo.format = VK_FORMAT_D32_SFLOAT;
    depthInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    vmaCreateImage(ctx.allocator, &depthInfo, &allocInfo,
                   &depthImage_, &depthAlloc_, nullptr);

    VkImageViewCreateInfo depthViewInfo = viewInfo;
    depthViewInfo.image = depthImage_;
    depthViewInfo.format = VK_FORMAT_D32_SFLOAT;
    depthViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

    vkCreateImageView(ctx.device, &depthViewInfo, nullptr, &depthView_);
}

// --------------------------------------------------------------------------
// Render pass (color + depth, finalLayout = TRANSFER_SRC for blit/screenshot)
// --------------------------------------------------------------------------

void SplatRenderer::createRenderPass(VkDevice device) {
    VkAttachmentDescription attachments[2] = {};

    // Color
    attachments[0].format = VK_FORMAT_R8G8B8A8_UNORM;
    attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[0].finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

    // Depth
    attachments[1].format = VK_FORMAT_D32_SFLOAT;
    attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorRef = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depthRef = {1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;
    subpass.pDepthStencilAttachment = &depthRef;

    VkSubpassDependency deps[2] = {};
    deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    deps[0].dstSubpass = 0;
    deps[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                           VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    deps[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                           VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    deps[0].srcAccessMask = 0;
    deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    deps[1].srcSubpass = 0;
    deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    deps[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    deps[1].dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
    deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    deps[1].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    VkRenderPassCreateInfo rpInfo = {};
    rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpInfo.attachmentCount = 2;
    rpInfo.pAttachments = attachments;
    rpInfo.subpassCount = 1;
    rpInfo.pSubpasses = &subpass;
    rpInfo.dependencyCount = 2;
    rpInfo.pDependencies = deps;

    vkCreateRenderPass(device, &rpInfo, nullptr, &renderPass_);
}

// --------------------------------------------------------------------------
// Render pass (LOAD variant: color loaded, for compositing on background)
// --------------------------------------------------------------------------

void SplatRenderer::createRenderPassLoad(VkDevice device) {
    VkAttachmentDescription attachments[2] = {};

    // Color — LOAD existing contents (background was blitted in)
    attachments[0].format = VK_FORMAT_R8G8B8A8_UNORM;
    attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[0].initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    attachments[0].finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

    // Depth — still CLEAR (splats have their own depth)
    attachments[1].format = VK_FORMAT_D32_SFLOAT;
    attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorRef = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depthRef = {1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;
    subpass.pDepthStencilAttachment = &depthRef;

    VkSubpassDependency deps[2] = {};
    deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    deps[0].dstSubpass = 0;
    deps[0].srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT |
                           VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                           VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    deps[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                           VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    deps[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                            VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    deps[1].srcSubpass = 0;
    deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    deps[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    deps[1].dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
    deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    deps[1].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    VkRenderPassCreateInfo rpInfo = {};
    rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpInfo.attachmentCount = 2;
    rpInfo.pAttachments = attachments;
    rpInfo.subpassCount = 1;
    rpInfo.pSubpasses = &subpass;
    rpInfo.dependencyCount = 2;
    rpInfo.pDependencies = deps;

    vkCreateRenderPass(device, &rpInfo, nullptr, &renderPassLoad_);
}

// --------------------------------------------------------------------------
// Render pass (LOAD variant: both color AND depth loaded, for full hybrid compositing)
// --------------------------------------------------------------------------

void SplatRenderer::createRenderPassLoadDepth(VkDevice device) {
    VkAttachmentDescription attachments[2] = {};

    // Color — LOAD existing contents (background was blitted in)
    attachments[0].format = VK_FORMAT_R8G8B8A8_UNORM;
    attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[0].initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    attachments[0].finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

    // Depth — LOAD existing depth from raster pass (splats depth-test against mesh)
    attachments[1].format = VK_FORMAT_D32_SFLOAT;
    attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorRef = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depthRef = {1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;
    subpass.pDepthStencilAttachment = &depthRef;

    VkSubpassDependency deps[2] = {};
    deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    deps[0].dstSubpass = 0;
    deps[0].srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT |
                           VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                           VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    deps[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                           VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    deps[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                            VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;

    deps[1].srcSubpass = 0;
    deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    deps[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    deps[1].dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
    deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    deps[1].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    VkRenderPassCreateInfo rpInfo = {};
    rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpInfo.attachmentCount = 2;
    rpInfo.pAttachments = attachments;
    rpInfo.subpassCount = 1;
    rpInfo.pSubpasses = &subpass;
    rpInfo.dependencyCount = 2;
    rpInfo.pDependencies = deps;

    vkCreateRenderPass(device, &rpInfo, nullptr, &renderPassLoadDepth_);
}

// --------------------------------------------------------------------------
// Framebuffer
// --------------------------------------------------------------------------

void SplatRenderer::createFramebuffer(VkDevice device) {
    VkImageView fbViews[2] = {colorView_, depthView_};

    VkFramebufferCreateInfo fbInfo = {};
    fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fbInfo.renderPass = renderPass_;
    fbInfo.attachmentCount = 2;
    fbInfo.pAttachments = fbViews;
    fbInfo.width = width_;
    fbInfo.height = height_;
    fbInfo.layers = 1;

    vkCreateFramebuffer(device, &fbInfo, nullptr, &framebuffer_);
}

void SplatRenderer::createFramebufferLoad(VkDevice device) {
    VkImageView fbViews[2] = {colorView_, depthView_};

    VkFramebufferCreateInfo fbInfo = {};
    fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fbInfo.renderPass = renderPassLoad_;
    fbInfo.attachmentCount = 2;
    fbInfo.pAttachments = fbViews;
    fbInfo.width = width_;
    fbInfo.height = height_;
    fbInfo.layers = 1;

    vkCreateFramebuffer(device, &fbInfo, nullptr, &framebufferLoad_);
}

void SplatRenderer::createFramebufferLoadDepth(VkDevice device) {
    VkImageView fbViews[2] = {colorView_, depthView_};

    VkFramebufferCreateInfo fbInfo = {};
    fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fbInfo.renderPass = renderPassLoadDepth_;
    fbInfo.attachmentCount = 2;
    fbInfo.pAttachments = fbViews;
    fbInfo.width = width_;
    fbInfo.height = height_;
    fbInfo.layers = 1;

    vkCreateFramebuffer(device, &fbInfo, nullptr, &framebufferLoadDepth_);
}

// --------------------------------------------------------------------------
// Pipeline creation
// --------------------------------------------------------------------------

void SplatRenderer::createPipelines(VkDevice device, const std::string& shaderBase) {
    // --- Descriptor set layouts ---

    // Compute: 4 input + N SH coefficients + 6 output SSBOs
    // Output order: proj_center, proj_conic, proj_color, sort_keys, sorted_indices, visible_count
    uint32_t numShCoeffs = numShCoeffsForDegree(shaderShDegree_);
    uint32_t numComputeBindings = 4 + numShCoeffs + 6;
    std::vector<VkDescriptorSetLayoutBinding> computeBindings(numComputeBindings);
    for (uint32_t i = 0; i < numComputeBindings; ++i) {
        computeBindings[i] = {};
        computeBindings[i].binding = i;
        computeBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        computeBindings[i].descriptorCount = 1;
        computeBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo computeLayoutInfo = {};
    computeLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    computeLayoutInfo.bindingCount = static_cast<uint32_t>(computeBindings.size());
    computeLayoutInfo.pBindings = computeBindings.data();
    vkCreateDescriptorSetLayout(device, &computeLayoutInfo, nullptr, &computeSetLayout_);

    // Render: 4 SSBOs (projected_centers, conics, colors, sorted_indices)
    std::vector<VkDescriptorSetLayoutBinding> renderBindings(4);
    for (uint32_t i = 0; i < 4; ++i) {
        renderBindings[i] = {};
        renderBindings[i].binding = i;
        renderBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        renderBindings[i].descriptorCount = 1;
        renderBindings[i].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    }

    VkDescriptorSetLayoutCreateInfo renderLayoutInfo = {};
    renderLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    renderLayoutInfo.bindingCount = static_cast<uint32_t>(renderBindings.size());
    renderLayoutInfo.pBindings = renderBindings.data();
    vkCreateDescriptorSetLayout(device, &renderLayoutInfo, nullptr, &renderSetLayout_);

    // --- Pipeline layouts ---

    // Compute push constants: view(64) + proj(64) + camPos(12) + pad(4) + focal(8) + screen(8) + numSplats(4) + pad(12) = 176 bytes
    VkPushConstantRange computePush = {};
    computePush.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    computePush.offset = 0;
    computePush.size = 176;

    VkPipelineLayoutCreateInfo computePipeLayoutInfo = {};
    computePipeLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    computePipeLayoutInfo.setLayoutCount = 1;
    computePipeLayoutInfo.pSetLayouts = &computeSetLayout_;
    computePipeLayoutInfo.pushConstantRangeCount = 1;
    computePipeLayoutInfo.pPushConstantRanges = &computePush;
    vkCreatePipelineLayout(device, &computePipeLayoutInfo, nullptr, &computeLayout_);

    // Render push constants: screen_size(8) + visible_count(4) + alpha_cutoff(4) = 16 bytes
    VkPushConstantRange renderPush = {};
    renderPush.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    renderPush.offset = 0;
    renderPush.size = 16;

    VkPipelineLayoutCreateInfo renderPipeLayoutInfo = {};
    renderPipeLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    renderPipeLayoutInfo.setLayoutCount = 1;
    renderPipeLayoutInfo.pSetLayouts = &renderSetLayout_;
    renderPipeLayoutInfo.pushConstantRangeCount = 1;
    renderPipeLayoutInfo.pPushConstantRanges = &renderPush;
    vkCreatePipelineLayout(device, &renderPipeLayoutInfo, nullptr, &renderLayout_);

    // --- Load compute shader ---
    auto compCode = SpvLoader::loadSPIRV(shaderBase + ".comp.spv");
    VkShaderModule compModule = SpvLoader::createShaderModule(device, compCode);

    VkComputePipelineCreateInfo compPipeInfo = {};
    compPipeInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    compPipeInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    compPipeInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    compPipeInfo.stage.module = compModule;
    compPipeInfo.stage.pName = "main";
    compPipeInfo.layout = computeLayout_;

    vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &compPipeInfo, nullptr, &computePipeline_);
    vkDestroyShaderModule(device, compModule, nullptr);

    // --- Load graphics shaders ---
    auto vertCode = SpvLoader::loadSPIRV(shaderBase + ".vert.spv");
    auto fragCode = SpvLoader::loadSPIRV(shaderBase + ".frag.spv");
    VkShaderModule vertModule = SpvLoader::createShaderModule(device, vertCode);
    VkShaderModule fragModule = SpvLoader::createShaderModule(device, fragCode);

    VkPipelineShaderStageCreateInfo shaderStages[2] = {};
    shaderStages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    shaderStages[0].module = vertModule;
    shaderStages[0].pName = "main";
    shaderStages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    shaderStages[1].module = fragModule;
    shaderStages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo vertexInput = {};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport viewport = {};
    viewport.width = static_cast<float>(width_);
    viewport.height = static_cast<float>(height_);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor = {};
    scissor.extent = {width_, height_};

    VkPipelineViewportStateCreateInfo viewportState = {};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterizer = {};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisampling = {};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil = {};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_FALSE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

    VkPipelineColorBlendAttachmentState blendAttachment = {};
    blendAttachment.blendEnable = VK_TRUE;
    blendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;  // premultiplied alpha
    blendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
    blendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    blendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
    blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                     VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo colorBlending = {};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &blendAttachment;

    VkGraphicsPipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInput;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.layout = renderLayout_;
    pipelineInfo.renderPass = renderPass_;
    pipelineInfo.subpass = 0;

    vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &renderPipeline_);

    vkDestroyShaderModule(device, vertModule, nullptr);
    vkDestroyShaderModule(device, fragModule, nullptr);

    // --- Descriptor pool ---
    // Need descriptors for: compute + render + sort (2*2 histogram + 1*2 prefix + 2*5 scatter = 16)
    VkDescriptorPoolSize poolSize = {};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = numComputeBindings + 4 + 16 + 4; // compute + render + sort + margin

    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 2 + 5;  // compute, render + 2 histogram + 1 prefix_sum + 2 scatter
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;

    vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool_);

    VkDescriptorSetLayout layouts[2] = {computeSetLayout_, renderSetLayout_};
    VkDescriptorSetAllocateInfo setAllocInfo = {};
    setAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    setAllocInfo.descriptorPool = descriptorPool_;
    setAllocInfo.descriptorSetCount = 2;
    setAllocInfo.pSetLayouts = layouts;

    VkDescriptorSet sets[2];
    vkAllocateDescriptorSets(device, &setAllocInfo, sets);
    computeDescSet_ = sets[0];
    renderDescSet_ = sets[1];
}

// --------------------------------------------------------------------------
// GPU radix sort pipeline creation
// --------------------------------------------------------------------------

static VkDescriptorSetLayout createSortSetLayout(VkDevice device, uint32_t numBindings) {
    std::vector<VkDescriptorSetLayoutBinding> bindings(numBindings);
    for (uint32_t i = 0; i < numBindings; ++i) {
        bindings[i] = {};
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = numBindings;
    layoutInfo.pBindings = bindings.data();

    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &layout);
    return layout;
}

static VkPipelineLayout createSortPipelineLayout(VkDevice device, VkDescriptorSetLayout setLayout) {
    VkPushConstantRange pushRange = {};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = 8;  // SortPush: num_elements(4) + bit_offset(4)

    VkPipelineLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &setLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushRange;

    VkPipelineLayout layout = VK_NULL_HANDLE;
    vkCreatePipelineLayout(device, &layoutInfo, nullptr, &layout);
    return layout;
}

static VkPipeline createSortComputePipeline(VkDevice device, VkPipelineLayout layout,
                                              const std::string& spvPath) {
    auto code = SpvLoader::loadSPIRV(spvPath);
    VkShaderModule module = SpvLoader::createShaderModule(device, code);

    VkComputePipelineCreateInfo pipeInfo = {};
    pipeInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeInfo.stage.module = module;
    pipeInfo.stage.pName = "main";
    pipeInfo.layout = layout;

    VkPipeline pipeline = VK_NULL_HANDLE;
    vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipeline);
    vkDestroyShaderModule(device, module, nullptr);
    return pipeline;
}

void SplatRenderer::createSortPipelines(VkDevice device) {
    // Descriptor set layouts
    // histogram.comp: binding 0 = keys_in, binding 1 = histograms
    sortHistogramSetLayout_ = createSortSetLayout(device, 2);
    // prefix_sum.comp: binding 0 = histograms, binding 1 = partition_sums
    sortPrefixSumSetLayout_ = createSortSetLayout(device, 2);
    // scatter.comp: binding 0-4 = keys_in, keys_out, vals_in, vals_out, histograms
    sortScatterSetLayout_ = createSortSetLayout(device, 5);

    // Pipeline layouts
    sortHistogramLayout_ = createSortPipelineLayout(device, sortHistogramSetLayout_);
    sortPrefixSumLayout_ = createSortPipelineLayout(device, sortPrefixSumSetLayout_);
    sortScatterLayout_ = createSortPipelineLayout(device, sortScatterSetLayout_);

    // Compute pipelines (load pre-compiled SPIR-V from shaders/radix_sort/)
    std::string sortDir = "shaders/radix_sort/";
    sortHistogramPipeline_ = createSortComputePipeline(device, sortHistogramLayout_, sortDir + "histogram.comp.spv");
    sortPrefixSumPipeline_ = createSortComputePipeline(device, sortPrefixSumLayout_, sortDir + "prefix_sum.comp.spv");
    sortScatterPipeline_ = createSortComputePipeline(device, sortScatterLayout_, sortDir + "scatter.comp.spv");

    // Allocate sort descriptor sets from the shared pool
    VkDescriptorSetLayout sortLayouts[5] = {
        sortHistogramSetLayout_,   // A->B
        sortHistogramSetLayout_,   // B->A
        sortPrefixSumSetLayout_,   // single
        sortScatterSetLayout_,     // A->B
        sortScatterSetLayout_,     // B->A
    };
    VkDescriptorSetAllocateInfo sortAllocInfo = {};
    sortAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    sortAllocInfo.descriptorPool = descriptorPool_;
    sortAllocInfo.descriptorSetCount = 5;
    sortAllocInfo.pSetLayouts = sortLayouts;

    VkDescriptorSet sortSets[5];
    vkAllocateDescriptorSets(device, &sortAllocInfo, sortSets);
    sortHistogramDescSets_[0] = sortSets[0];
    sortHistogramDescSets_[1] = sortSets[1];
    sortPrefixSumDescSet_ = sortSets[2];
    sortScatterDescSets_[0] = sortSets[3];
    sortScatterDescSets_[1] = sortSets[4];

    std::cout << "[info] GPU radix sort pipelines created" << std::endl;
}

// --------------------------------------------------------------------------
// Buffer creation and data upload
// --------------------------------------------------------------------------

void SplatRenderer::createBuffers(VulkanContext& ctx, const GaussianSplatData& data) {
    numSplats_ = data.num_splats;
    shDegree_ = data.sh_degree;
    if (numSplats_ == 0) return;

    VkBufferUsageFlags ssbo = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    // Input buffers (CPU-visible for upload)
    // Positions: loader stores as vec4 (x,y,z,1) already
    const auto& pos4 = data.positions;  // already numSplats * 4 floats

    createVmaBuffer(ctx.allocator, pos4.size() * sizeof(float), ssbo,
                    VMA_MEMORY_USAGE_CPU_TO_GPU, posBuffer_, posAlloc_);
    uploadVmaBuffer(ctx.allocator, posAlloc_, pos4.data(), pos4.size() * sizeof(float));

    // Rotations (vec4 xyzw)
    createVmaBuffer(ctx.allocator, numSplats_ * 4 * sizeof(float), ssbo,
                    VMA_MEMORY_USAGE_CPU_TO_GPU, rotBuffer_, rotAlloc_);
    uploadVmaBuffer(ctx.allocator, rotAlloc_, data.rotations.data(),
                    numSplats_ * 4 * sizeof(float));

    // Scales as vec4 (x,y,z,0)
    std::vector<float> scale4(numSplats_ * 4);
    for (uint32_t i = 0; i < numSplats_; ++i) {
        scale4[i * 4 + 0] = data.scales[i * 3 + 0];
        scale4[i * 4 + 1] = data.scales[i * 3 + 1];
        scale4[i * 4 + 2] = data.scales[i * 3 + 2];
        scale4[i * 4 + 3] = 0.0f;
    }
    createVmaBuffer(ctx.allocator, scale4.size() * sizeof(float), ssbo,
                    VMA_MEMORY_USAGE_CPU_TO_GPU, scaleBuffer_, scaleAlloc_);
    uploadVmaBuffer(ctx.allocator, scaleAlloc_, scale4.data(), scale4.size() * sizeof(float));

    // Opacities
    createVmaBuffer(ctx.allocator, numSplats_ * sizeof(float), ssbo,
                    VMA_MEMORY_USAGE_CPU_TO_GPU, opacityBuffer_, opacityAlloc_);
    uploadVmaBuffer(ctx.allocator, opacityAlloc_, data.opacities.data(),
                    numSplats_ * sizeof(float));

    // SH coefficient buffers — pad vec3 data to vec4 for std430 alignment
    for (const auto& coeffs : data.sh_coefficients) {
        VkBuffer buf = VK_NULL_HANDLE;
        VmaAllocation alloc = VK_NULL_HANDLE;
        // Shader expects vec4 per splat; source data may be vec3
        uint32_t srcFloatsPerSplat = coeffs.empty() ? 0 : static_cast<uint32_t>(coeffs.size() / numSplats_);
        if (srcFloatsPerSplat == 3) {
            // Pad vec3 → vec4
            std::vector<float> padded(numSplats_ * 4, 0.0f);
            for (uint32_t i = 0; i < numSplats_; ++i) {
                padded[i * 4 + 0] = coeffs[i * 3 + 0];
                padded[i * 4 + 1] = coeffs[i * 3 + 1];
                padded[i * 4 + 2] = coeffs[i * 3 + 2];
            }
            createVmaBuffer(ctx.allocator, padded.size() * sizeof(float), ssbo,
                            VMA_MEMORY_USAGE_CPU_TO_GPU, buf, alloc);
            uploadVmaBuffer(ctx.allocator, alloc, padded.data(), padded.size() * sizeof(float));
        } else {
            VkDeviceSize sz = coeffs.empty() ? sizeof(float) : coeffs.size() * sizeof(float);
            createVmaBuffer(ctx.allocator, sz, ssbo, VMA_MEMORY_USAGE_CPU_TO_GPU, buf, alloc);
            if (!coeffs.empty()) {
                uploadVmaBuffer(ctx.allocator, alloc, coeffs.data(), coeffs.size() * sizeof(float));
            }
        }
        shBuffers_.push_back(buf);
        shAllocs_.push_back(alloc);
    }

    // Projected output buffers (GPU only, written by compute)
    createVmaBuffer(ctx.allocator, numSplats_ * 4 * sizeof(float), ssbo,
                    VMA_MEMORY_USAGE_GPU_ONLY, projCenterBuffer_, projCenterAlloc_);
    createVmaBuffer(ctx.allocator, numSplats_ * 4 * sizeof(float), ssbo,
                    VMA_MEMORY_USAGE_GPU_ONLY, projConicBuffer_, projConicAlloc_);
    createVmaBuffer(ctx.allocator, numSplats_ * 4 * sizeof(float), ssbo,
                    VMA_MEMORY_USAGE_GPU_ONLY, projColorBuffer_, projColorAlloc_);

    // Sort keys (buffer A, GPU only)
    createVmaBuffer(ctx.allocator, numSplats_ * sizeof(uint32_t), ssbo,
                    VMA_MEMORY_USAGE_GPU_ONLY, sortKeysBuffer_, sortKeysAlloc_);

    // Sorted indices (buffer A, GPU only — GPU radix sort, no CPU upload)
    createVmaBuffer(ctx.allocator, numSplats_ * sizeof(uint32_t), ssbo,
                    VMA_MEMORY_USAGE_GPU_ONLY, sortedIndicesBuffer_, sortedIndicesAlloc_);

    // Visible count (GPU only, atomic counter)
    createVmaBuffer(ctx.allocator, sizeof(uint32_t), ssbo,
                    VMA_MEMORY_USAGE_GPU_ONLY, visibleCountBuffer_, visibleCountAlloc_);

    // Ping-pong sort buffers (buffer B)
    createVmaBuffer(ctx.allocator, numSplats_ * sizeof(uint32_t), ssbo,
                    VMA_MEMORY_USAGE_GPU_ONLY, sortKeysBBuffer_, sortKeysBAlloc_);
    createVmaBuffer(ctx.allocator, numSplats_ * sizeof(uint32_t), ssbo,
                    VMA_MEMORY_USAGE_GPU_ONLY, sortValsBBuffer_, sortValsBAlloc_);

    // Radix sort histogram and partition buffers
    static const uint32_t SORT_TILE_SIZE = 3840;
    static const uint32_t PREFIX_SUM_BLOCK_SIZE = 2048;
    sortNumWg_ = (numSplats_ + SORT_TILE_SIZE - 1) / SORT_TILE_SIZE;
    uint32_t histogramSize = std::max(256u * sortNumWg_ * 4u, 4u);
    createVmaBuffer(ctx.allocator, histogramSize, ssbo,
                    VMA_MEMORY_USAGE_GPU_ONLY, histogramBuffer_, histogramAlloc_);
    uint32_t totalHistEntries = 256 * sortNumWg_;
    uint32_t numPartitions = (totalHistEntries + PREFIX_SUM_BLOCK_SIZE - 1) / PREFIX_SUM_BLOCK_SIZE;
    uint32_t partitionSumsSize = std::max(numPartitions * 4u, 4u);
    createVmaBuffer(ctx.allocator, partitionSumsSize, ssbo,
                    VMA_MEMORY_USAGE_GPU_ONLY, partitionSumsBuffer_, partitionSumsAlloc_);

    std::cout << "[info] GPU radix sort: " << numSplats_ << " splats, "
              << sortNumWg_ << " workgroups, " << totalHistEntries << " histogram entries, "
              << numPartitions << " partitions" << std::endl;

    // --- Update descriptor sets ---
    auto writeSSBO = [&](VkDescriptorSet set, uint32_t binding, VkBuffer buffer, VkDeviceSize size) {
        VkDescriptorBufferInfo bufInfo = {};
        bufInfo.buffer = buffer;
        bufInfo.offset = 0;
        bufInfo.range = size;

        VkWriteDescriptorSet write = {};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = set;
        write.dstBinding = binding;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &bufInfo;

        vkUpdateDescriptorSets(ctx.device, 1, &write, 0, nullptr);
    };

    // Compute set layout:
    //   0: positions, 1: rotations, 2: scales, 3: opacities,
    //   4..4+numShCoeffs-1: SH coefficient buffers,
    //   4+numShCoeffs: projected_centers, +1: conics, +2: colors, +3: sort_keys,
    //   +4: sorted_indices, +5: visible_count
    uint32_t numShCoeffs = numShCoeffsForDegree(shaderShDegree_);
    uint32_t outputBase = 4 + numShCoeffs;

    writeSSBO(computeDescSet_, 0, posBuffer_, numSplats_ * 4 * sizeof(float));
    writeSSBO(computeDescSet_, 1, rotBuffer_, numSplats_ * 4 * sizeof(float));
    writeSSBO(computeDescSet_, 2, scaleBuffer_, numSplats_ * 4 * sizeof(float));
    writeSSBO(computeDescSet_, 3, opacityBuffer_, numSplats_ * sizeof(float));

    // Bind SH coefficient buffers: use scene data where available, zero-filled dummy otherwise.
    // The sh3 shader evaluates ALL degrees unconditionally (no runtime branching on sh_degree),
    // so missing SH buffers MUST contain zeros (not random data).
    for (uint32_t i = 0; i < numShCoeffs; ++i) {
        VkBuffer shBuf;
        VkDeviceSize shSize;
        if (i < static_cast<uint32_t>(shBuffers_.size())) {
            shBuf = shBuffers_[i];
            shSize = numSplats_ * 4 * sizeof(float);
        } else {
            // Create zero-filled dummy buffer for missing SH coefficients
            VkBuffer dummyBuf = VK_NULL_HANDLE;
            VmaAllocation dummyAlloc = VK_NULL_HANDLE;
            VkDeviceSize dummySize = numSplats_ * 4 * sizeof(float);
            createVmaBuffer(ctx.allocator, dummySize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                            VMA_MEMORY_USAGE_CPU_TO_GPU, dummyBuf, dummyAlloc);
            // Zero-fill: VMA CPU_TO_GPU memory is not guaranteed to be zeroed
            void* mapped = nullptr;
            vmaMapMemory(ctx.allocator, dummyAlloc, &mapped);
            std::memset(mapped, 0, static_cast<size_t>(dummySize));
            vmaUnmapMemory(ctx.allocator, dummyAlloc);
            shBuffers_.push_back(dummyBuf);
            shAllocs_.push_back(dummyAlloc);
            shBuf = dummyBuf;
            shSize = dummySize;
        }
        writeSSBO(computeDescSet_, 4 + i, shBuf, std::max(shSize, VkDeviceSize(sizeof(float))));
    }

    writeSSBO(computeDescSet_, outputBase + 0, projCenterBuffer_, numSplats_ * 4 * sizeof(float));
    writeSSBO(computeDescSet_, outputBase + 1, projConicBuffer_, numSplats_ * 4 * sizeof(float));
    writeSSBO(computeDescSet_, outputBase + 2, projColorBuffer_, numSplats_ * 4 * sizeof(float));
    writeSSBO(computeDescSet_, outputBase + 3, sortKeysBuffer_, numSplats_ * sizeof(uint32_t));
    writeSSBO(computeDescSet_, outputBase + 4, sortedIndicesBuffer_, numSplats_ * sizeof(uint32_t));
    writeSSBO(computeDescSet_, outputBase + 5, visibleCountBuffer_, sizeof(uint32_t));

    // Render set: projected_centers(0), conics(1), colors(2), sorted_indices(3)
    writeSSBO(renderDescSet_, 0, projCenterBuffer_, numSplats_ * 4 * sizeof(float));
    writeSSBO(renderDescSet_, 1, projConicBuffer_, numSplats_ * 4 * sizeof(float));
    writeSSBO(renderDescSet_, 2, projColorBuffer_, numSplats_ * 4 * sizeof(float));
    writeSSBO(renderDescSet_, 3, sortedIndicesBuffer_, numSplats_ * sizeof(uint32_t));

    // --- Sort descriptor sets ---
    VkDeviceSize sortBufSize = numSplats_ * sizeof(uint32_t);
    VkDeviceSize histBufSize = histogramSize;
    VkDeviceSize partBufSize = partitionSumsSize;

    // Histogram sets: binding 0 = keys_in, binding 1 = histograms
    // [0] = A->B (reads keys A)
    writeSSBO(sortHistogramDescSets_[0], 0, sortKeysBuffer_, sortBufSize);
    writeSSBO(sortHistogramDescSets_[0], 1, histogramBuffer_, histBufSize);
    // [1] = B->A (reads keys B)
    writeSSBO(sortHistogramDescSets_[1], 0, sortKeysBBuffer_, sortBufSize);
    writeSSBO(sortHistogramDescSets_[1], 1, histogramBuffer_, histBufSize);

    // Prefix sum set: binding 0 = histograms, binding 1 = partition_sums
    writeSSBO(sortPrefixSumDescSet_, 0, histogramBuffer_, histBufSize);
    writeSSBO(sortPrefixSumDescSet_, 1, partitionSumsBuffer_, partBufSize);

    // Scatter sets: binding 0=keys_in, 1=keys_out, 2=vals_in, 3=vals_out, 4=histograms
    // [0] = A->B
    writeSSBO(sortScatterDescSets_[0], 0, sortKeysBuffer_, sortBufSize);
    writeSSBO(sortScatterDescSets_[0], 1, sortKeysBBuffer_, sortBufSize);
    writeSSBO(sortScatterDescSets_[0], 2, sortedIndicesBuffer_, sortBufSize);
    writeSSBO(sortScatterDescSets_[0], 3, sortValsBBuffer_, sortBufSize);
    writeSSBO(sortScatterDescSets_[0], 4, histogramBuffer_, histBufSize);
    // [1] = B->A
    writeSSBO(sortScatterDescSets_[1], 0, sortKeysBBuffer_, sortBufSize);
    writeSSBO(sortScatterDescSets_[1], 1, sortKeysBuffer_, sortBufSize);
    writeSSBO(sortScatterDescSets_[1], 2, sortValsBBuffer_, sortBufSize);
    writeSSBO(sortScatterDescSets_[1], 3, sortedIndicesBuffer_, sortBufSize);
    writeSSBO(sortScatterDescSets_[1], 4, histogramBuffer_, histBufSize);
}

// --------------------------------------------------------------------------
// Init
// --------------------------------------------------------------------------

void SplatRenderer::init(VulkanContext& ctx, const GaussianSplatData& data,
                          const std::string& shaderBase, uint32_t width, uint32_t height) {
    width_ = width;
    height_ = height;

    // Read shader's compiled SH degree from reflection JSON (determines descriptor layout)
    shaderShDegree_ = readShaderShDegree(shaderBase);
    // Scene's actual SH degree (for push constant — determines which SH coefficients to evaluate)
    shDegree_ = data.sh_degree;

    std::cout << "[info] Shader SH degree: " << shaderShDegree_
              << ", scene SH degree: " << shDegree_
              << " (" << numShCoeffsForDegree(shaderShDegree_) << " SH bindings)" << std::endl;

    createOffscreenTarget(ctx);
    createRenderPass(ctx.device);
    createRenderPassLoad(ctx.device);
    createRenderPassLoadDepth(ctx.device);
    createFramebuffer(ctx.device);
    createFramebufferLoad(ctx.device);
    createFramebufferLoadDepth(ctx.device);
    createPipelines(ctx.device, shaderBase);
    createSortPipelines(ctx.device);
    createBuffers(ctx, data);

    // Robust camera from IQR-based bounds (handles outlier splats)
    if (data.num_splats > 0) {
        uint32_t n = data.num_splats;

        // Collect per-axis coordinates
        std::vector<float> xs(n), ys(n), zs(n);
        glm::vec3 minB(1e9f), maxB(-1e9f);
        for (uint32_t i = 0; i < n; ++i) {
            xs[i] = data.positions[i * 4 + 0];
            ys[i] = data.positions[i * 4 + 1];
            zs[i] = data.positions[i * 4 + 2];
            minB = glm::min(minB, glm::vec3(xs[i], ys[i], zs[i]));
            maxB = glm::max(maxB, glm::vec3(xs[i], ys[i], zs[i]));
        }

        float fullRadius = glm::length(maxB - minB) * 0.5f;
        if (fullRadius < 0.001f) fullRadius = 1.0f;

        // Use IQR camera only for large scenes with significant outliers
        // (IQR extent < 50% of full bbox extent = lots of sky/background splats)
        bool useIqr = false;
        float maxIqrExt = 0.0f;
        if (n >= 1000) {
            std::sort(xs.begin(), xs.end());
            std::sort(ys.begin(), ys.end());
            std::sort(zs.begin(), zs.end());
            uint32_t p25 = n * 25 / 100, p75 = n * 75 / 100;
            glm::vec3 extent(xs[p75] - xs[p25], ys[p75] - ys[p25], zs[p75] - zs[p25]);
            maxIqrExt = glm::max(extent.x, glm::max(extent.y, extent.z));
            float fullExtent = glm::max(maxB.x - minB.x, glm::max(maxB.y - minB.y, maxB.z - minB.z));
            useIqr = (maxIqrExt > 0.001f && maxIqrExt < fullExtent * 0.25f);
        }

        if (!useIqr) {
            glm::vec3 center = (minB + maxB) * 0.5f;
            glm::vec3 extent = maxB - minB;
            // Detect flat/planar scenes: if one axis has near-zero extent,
            // place camera perpendicular to the plane for a straight-on view.
            // Otherwise, add a slight Y elevation for a more natural viewing angle.
            glm::vec3 camOffset;
            glm::vec3 upVec(0.0f, 1.0f, 0.0f);
            float maxExt = glm::max(extent.x, glm::max(extent.y, extent.z));
            if (extent.z < maxExt * 0.01f) {
                // Flat in Z — place camera along +Z axis (looking at XY plane)
                camOffset = glm::vec3(0.0f, 0.0f, fullRadius * 2.5f);
            } else if (extent.y < maxExt * 0.01f) {
                // Flat in Y — place camera along +Y axis
                camOffset = glm::vec3(0.0f, fullRadius * 3.0f, 0.0f);
                upVec = glm::vec3(0.0f, 0.0f, -1.0f);
            } else if (extent.x < maxExt * 0.01f) {
                // Flat in X — place camera along +X axis
                camOffset = glm::vec3(fullRadius * 3.0f, 0.0f, 0.0f);
            } else {
                // 3D scene — use elevated camera
                camOffset = glm::vec3(0.0f, fullRadius * 0.5f, fullRadius * 2.5f);
            }
            camPos_ = center + camOffset;
            float aspect = static_cast<float>(width) / static_cast<float>(height);
            float fov = glm::radians(45.0f);
            viewMatrix_ = glm::lookAt(camPos_, center, upVec);
            projMatrix_ = glm::perspective(fov, aspect, 0.01f, fullRadius * 10.0f);
            projMatrix_[1][1] *= -1.0f;
            focalY_ = 0.5f * static_cast<float>(height) / tanf(fov * 0.5f);
            focalX_ = focalY_;
            std::cout << "[info] Splat bounds: min=(" << minB.x << "," << minB.y << "," << minB.z
                      << ") max=(" << maxB.x << "," << maxB.y << "," << maxB.z
                      << ") center=(" << center.x << "," << center.y << "," << center.z
                      << ") radius=" << fullRadius
                      << " cam=(" << camPos_.x << "," << camPos_.y << "," << camPos_.z
                      << ") shBufs=" << data.sh_coefficients.size() << std::endl;
        } else {
            // IQR-based robust camera for large real-world scenes
            // (xs, ys, zs already sorted above)
            uint32_t p25 = n * 25 / 100, p75 = n * 75 / 100;
            glm::vec3 center(xs[n / 2], ys[n / 2], zs[n / 2]);
            glm::vec3 extent(xs[p75] - xs[p25], ys[p75] - ys[p25], zs[p75] - zs[p25]);

            // Detect up axis (shortest IQR extent)
            int upIdx = 1; // default Y-up
            glm::vec3 upVec(0.0f, 1.0f, 0.0f);
            if (extent.y <= extent.x && extent.y <= extent.z) {
                upIdx = 1; upVec = glm::vec3(0.0f, -1.0f, 0.0f); // Y shortest, COLMAP Y-down
            } else if (extent.z <= extent.x && extent.z <= extent.y) {
                upIdx = 2; upVec = glm::vec3(0.0f, 0.0f, 1.0f);
            } else {
                upIdx = 0; upVec = glm::vec3(1.0f, 0.0f, 0.0f);
            }

            float maxIqr = glm::max(extent.x, glm::max(extent.y, extent.z));
            if (maxIqr < 0.001f) maxIqr = 1.0f;
            float camDist = maxIqr * 0.4f;

            // Build eye offset on ground axes with slight elevation
            glm::vec3 off(0.0f);
            int groundAxes[2];
            int gi = 0;
            for (int i = 0; i < 3; ++i) {
                if (i != upIdx) groundAxes[gi++] = i;
            }
            off[groundAxes[0]] = camDist * 0.7f;
            off[groundAxes[1]] = camDist * 0.7f;
            off[upIdx] = extent[upIdx] * 0.1f;
            camPos_ = center + off;

            float aspect = static_cast<float>(width) / static_cast<float>(height);
            float fov = glm::radians(45.0f);
            viewMatrix_ = glm::lookAt(camPos_, center, upVec);
            projMatrix_ = glm::perspective(fov, aspect, 0.01f, fullRadius * 5.0f);
            projMatrix_[1][1] *= -1.0f;
            focalY_ = 0.5f * static_cast<float>(height) / tanf(fov * 0.5f);
            focalX_ = focalY_;

            std::cout << "[info] Splat IQR camera: center=(" << center.x << "," << center.y << "," << center.z
                      << ") extent=(" << extent.x << "," << extent.y << "," << extent.z
                      << ") iqr=" << maxIqr << " up=" << upIdx
                      << " cam=(" << camPos_.x << "," << camPos_.y << "," << camPos_.z
                      << ") shBufs=" << data.sh_coefficients.size() << std::endl;
        }
    }

    std::cout << "[info] SplatRenderer initialized: " << numSplats_ << " splats, "
              << width << "x" << height << " (GPU radix sort)" << std::endl;
}

// --------------------------------------------------------------------------
// Update camera
// --------------------------------------------------------------------------

void SplatRenderer::updateCamera(glm::vec3 eye, glm::vec3 target, glm::vec3 up,
                                  float fovY, float aspect, float nearPlane, float farPlane) {
    camPos_ = eye;
    viewMatrix_ = glm::lookAt(eye, target, up);
    projMatrix_ = glm::perspective(fovY, aspect, nearPlane, farPlane);
    projMatrix_[1][1] *= -1.0f; // Vulkan Y-flip

    // Focal lengths: fy = h/(2*tan(fov_y/2)), fx = fy for square pixels
    focalY_ = 0.5f * static_cast<float>(height_) / tanf(fovY * 0.5f);
    focalX_ = focalY_;
}

// --------------------------------------------------------------------------
// Render
// --------------------------------------------------------------------------

void SplatRenderer::render(VulkanContext& ctx) {
    if (numSplats_ == 0) return;

    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();

    // --- Compute dispatch (projection + sort key generation) ---
    struct ComputePush {
        float view[16];       // offset 0
        float proj[16];       // offset 64
        float camPos[3];      // offset 128
        float _pad0;          // offset 140
        float screenW, screenH; // offset 144 (screen_size)
        uint32_t numSplats;   // offset 152 (total_splats)
        float focalX;         // offset 156
        float focalY;         // offset 160
        int32_t shDegree;     // offset 164
        float _pad1[2];       // offset 168 (pad to 176)
    } push = {};

    std::memcpy(push.view, &viewMatrix_[0][0], 64);
    std::memcpy(push.proj, &projMatrix_[0][0], 64);
    push.camPos[0] = camPos_.x;
    push.camPos[1] = camPos_.y;
    push.camPos[2] = camPos_.z;
    push.screenW = static_cast<float>(width_);
    push.screenH = static_cast<float>(height_);
    push.numSplats = numSplats_;
    push.focalX = focalX_;
    push.focalY = focalY_;
    push.shDegree = static_cast<int32_t>(shDegree_);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computeLayout_,
                            0, 1, &computeDescSet_, 0, nullptr);
    vkCmdPushConstants(cmd, computeLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(push), &push);

    uint32_t groupCount = (numSplats_ + 255) / 256;
    vkCmdDispatch(cmd, groupCount, 1, 1);

    // Barrier: compute writes -> sort reads
    VkMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &barrier, 0, nullptr, 0, nullptr);

    // --- GPU Radix Sort (4 passes, 8 bits per pass = 32-bit keys) ---
    {
        static const uint32_t PREFIX_SUM_BLOCK_SIZE = 2048;
        uint32_t numElements = numSplats_;
        uint32_t numWg = sortNumWg_;
        uint32_t totalHistogram = 256 * numWg;
        uint32_t numParts = (totalHistogram + PREFIX_SUM_BLOCK_SIZE - 1) / PREFIX_SUM_BLOCK_SIZE;

        VkMemoryBarrier sortBarrier = {};
        sortBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        sortBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        sortBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

        struct SortPush {
            uint32_t numElements;
            uint32_t bitOffset;
        };

        for (uint32_t pass = 0; pass < 4; ++pass) {
            uint32_t bitOffset = pass * 8;
            uint32_t ping = pass % 2;  // 0 = A->B, 1 = B->A

            // --- Phase 1: Histogram ---
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sortHistogramPipeline_);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sortHistogramLayout_,
                                    0, 1, &sortHistogramDescSets_[ping], 0, nullptr);
            SortPush histPush = {numElements, bitOffset};
            vkCmdPushConstants(cmd, sortHistogramLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(histPush), &histPush);
            vkCmdDispatch(cmd, numWg, 1, 1);

            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 1, &sortBarrier, 0, nullptr, 0, nullptr);

            // --- Phase 2: Prefix Sum (3 sub-passes) ---
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sortPrefixSumPipeline_);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sortPrefixSumLayout_,
                                    0, 1, &sortPrefixSumDescSet_, 0, nullptr);

            // Sub-pass 0: Local scan
            SortPush psPush0 = {totalHistogram, 0};
            vkCmdPushConstants(cmd, sortPrefixSumLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(psPush0), &psPush0);
            vkCmdDispatch(cmd, numParts, 1, 1);

            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 1, &sortBarrier, 0, nullptr, 0, nullptr);

            // Sub-pass 1: Spine scan
            SortPush psPush1 = {numParts, 1};
            vkCmdPushConstants(cmd, sortPrefixSumLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(psPush1), &psPush1);
            vkCmdDispatch(cmd, 1, 1, 1);

            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 1, &sortBarrier, 0, nullptr, 0, nullptr);

            // Sub-pass 2: Propagate
            SortPush psPush2 = {totalHistogram, 2};
            vkCmdPushConstants(cmd, sortPrefixSumLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(psPush2), &psPush2);
            vkCmdDispatch(cmd, numParts, 1, 1);

            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 1, &sortBarrier, 0, nullptr, 0, nullptr);

            // --- Phase 3: Scatter ---
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sortScatterPipeline_);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sortScatterLayout_,
                                    0, 1, &sortScatterDescSets_[ping], 0, nullptr);
            SortPush scatterPush = {numElements, bitOffset};
            vkCmdPushConstants(cmd, sortScatterLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(scatterPush), &scatterPush);
            vkCmdDispatch(cmd, numWg, 1, 1);

            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 1, &sortBarrier, 0, nullptr, 0, nullptr);
        }
    }

    // After 4 passes (even count), sorted results are in buffer A
    // (sortKeysBuffer_, sortedIndicesBuffer_) which is what render reads.

    // Barrier: sort compute -> vertex/fragment shader reads
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         0, 1, &barrier, 0, nullptr, 0, nullptr);

    // --- Render pass ---
    // Use LOAD render pass when a background has been blitted in (hybrid compositing)
    VkClearValue clearValues[2] = {};
    clearValues[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
    clearValues[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo rpBegin = {};
    rpBegin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    if (hasBackgroundDepth_) {
        // Full hybrid: both color and depth loaded from raster pass
        rpBegin.renderPass = renderPassLoadDepth_;
        rpBegin.framebuffer = framebufferLoadDepth_;
    } else if (hasBackground_) {
        // Color-only compositing: color loaded, depth cleared
        rpBegin.renderPass = renderPassLoad_;
        rpBegin.framebuffer = framebufferLoad_;
    } else {
        rpBegin.renderPass = renderPass_;
        rpBegin.framebuffer = framebuffer_;
    }
    rpBegin.renderArea.extent = {width_, height_};
    rpBegin.clearValueCount = 2;
    rpBegin.pClearValues = clearValues;

    vkCmdBeginRenderPass(cmd, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);

    // Reset background flags after use (one-shot per render call)
    hasBackground_ = false;
    hasBackgroundDepth_ = false;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, renderPipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, renderLayout_,
                            0, 1, &renderDescSet_, 0, nullptr);

    // Push constants: screen_size(vec2) + visible_count(uint) + alpha_cutoff(float) = 16 bytes
    struct RenderPush {
        float screenW, screenH;
        uint32_t visibleCount;
        float alphaCutoff;
    } renderPush = {};
    renderPush.screenW = static_cast<float>(width_);
    renderPush.screenH = static_cast<float>(height_);
    renderPush.visibleCount = numSplats_;  // all splats visible (CPU sort)
    renderPush.alphaCutoff = 0.004f;
    vkCmdPushConstants(cmd, renderLayout_,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(renderPush), &renderPush);

    // Instanced draw: 6 vertices (quad) x numSplats instances
    vkCmdDraw(cmd, 6, numSplats_, 0, 0);

    vkCmdEndRenderPass(cmd);

    // Submit and wait
    ctx.endSingleTimeCommands(cmd);
}

// --------------------------------------------------------------------------
// Blit to swapchain
// --------------------------------------------------------------------------

void SplatRenderer::blitToSwapchain(VulkanContext& ctx, VkCommandBuffer cmd,
                                     VkImage swapImage, VkExtent2D extent) {
    // Transition swapchain image: UNDEFINED -> TRANSFER_DST
    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = swapImage;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    // Blit offscreen -> swapchain
    VkImageBlit blitRegion = {};
    blitRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blitRegion.srcSubresource.layerCount = 1;
    blitRegion.srcOffsets[1] = {static_cast<int32_t>(width_),
                                static_cast<int32_t>(height_), 1};
    blitRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blitRegion.dstSubresource.layerCount = 1;
    blitRegion.dstOffsets[1] = {static_cast<int32_t>(extent.width),
                                static_cast<int32_t>(extent.height), 1};

    vkCmdBlitImage(cmd,
        colorImage_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        swapImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1, &blitRegion, VK_FILTER_LINEAR);

    // Transition swapchain image: TRANSFER_DST -> PRESENT_SRC
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = 0;

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);
}

// --------------------------------------------------------------------------
// Blit to swapchain (compositing mode — from PRESENT_SRC, not UNDEFINED)
// --------------------------------------------------------------------------

void SplatRenderer::blitToSwapchainComposite(VulkanContext& ctx, VkCommandBuffer cmd,
                                              VkImage swapImage, VkExtent2D extent) {
    // Transition swapchain image: PRESENT_SRC -> TRANSFER_DST
    // (scene has already been blitted to the swapchain before us)
    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = swapImage;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    // Blit offscreen -> swapchain
    VkImageBlit blitRegion = {};
    blitRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blitRegion.srcSubresource.layerCount = 1;
    blitRegion.srcOffsets[1] = {static_cast<int32_t>(width_),
                                static_cast<int32_t>(height_), 1};
    blitRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blitRegion.dstSubresource.layerCount = 1;
    blitRegion.dstOffsets[1] = {static_cast<int32_t>(extent.width),
                                static_cast<int32_t>(extent.height), 1};

    vkCmdBlitImage(cmd,
        colorImage_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        swapImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1, &blitRegion, VK_FILTER_LINEAR);

    // Transition swapchain image: TRANSFER_DST -> PRESENT_SRC
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = 0;

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);
}

// --------------------------------------------------------------------------
// Preload background image (blit external image into splat color target)
// --------------------------------------------------------------------------

void SplatRenderer::preloadBackground(VulkanContext& ctx, VkImage srcImage, VkFormat /*srcFormat*/,
                                       uint32_t srcWidth, uint32_t srcHeight) {
    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();

    // Transition splat color image: UNDEFINED -> TRANSFER_DST
    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = colorImage_;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    // Blit source image -> splat color image
    VkImageBlit blitRegion = {};
    blitRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blitRegion.srcSubresource.layerCount = 1;
    blitRegion.srcOffsets[1] = {static_cast<int32_t>(srcWidth),
                                static_cast<int32_t>(srcHeight), 1};
    blitRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blitRegion.dstSubresource.layerCount = 1;
    blitRegion.dstOffsets[1] = {static_cast<int32_t>(width_),
                                static_cast<int32_t>(height_), 1};

    vkCmdBlitImage(cmd,
        srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        colorImage_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1, &blitRegion, VK_FILTER_LINEAR);

    // Transition splat color image: TRANSFER_DST -> COLOR_ATTACHMENT_OPTIMAL
    // (the LOAD render pass expects this layout as initialLayout)
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    ctx.endSingleTimeCommands(cmd);

    hasBackground_ = true;
    std::cout << "[info] Preloaded background image into splat color target ("
              << srcWidth << "x" << srcHeight << " -> "
              << width_ << "x" << height_ << ")" << std::endl;
}

// --------------------------------------------------------------------------
// Preload depth from raster pass (copy raster depth into splat depth buffer)
// --------------------------------------------------------------------------

void SplatRenderer::preloadDepth(VulkanContext& ctx, VkImage srcDepthImage,
                                  uint32_t srcWidth, uint32_t srcHeight) {
    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();

    // Transition raster depth: DEPTH_STENCIL_ATTACHMENT_OPTIMAL -> TRANSFER_SRC
    VkImageMemoryBarrier barriers[2] = {};

    barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barriers[0].oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    barriers[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[0].image = srcDepthImage;
    barriers[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    barriers[0].subresourceRange.levelCount = 1;
    barriers[0].subresourceRange.layerCount = 1;
    barriers[0].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    barriers[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    // Transition splat depth: UNDEFINED -> TRANSFER_DST
    barriers[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barriers[1].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barriers[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barriers[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[1].image = depthImage_;
    barriers[1].subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    barriers[1].subresourceRange.levelCount = 1;
    barriers[1].subresourceRange.layerCount = 1;
    barriers[1].srcAccessMask = 0;
    barriers[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 2, barriers);

    // Copy depth image (vkCmdCopyImage requires matching dimensions, or use blit)
    if (srcWidth == width_ && srcHeight == height_) {
        VkImageCopy copyRegion = {};
        copyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        copyRegion.srcSubresource.layerCount = 1;
        copyRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        copyRegion.dstSubresource.layerCount = 1;
        copyRegion.extent = {srcWidth, srcHeight, 1};

        vkCmdCopyImage(cmd,
            srcDepthImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            depthImage_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &copyRegion);
    } else {
        // Blit with nearest filter for depth (no linear interpolation on depth)
        VkImageBlit blitRegion = {};
        blitRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        blitRegion.srcSubresource.layerCount = 1;
        blitRegion.srcOffsets[1] = {static_cast<int32_t>(srcWidth),
                                    static_cast<int32_t>(srcHeight), 1};
        blitRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        blitRegion.dstSubresource.layerCount = 1;
        blitRegion.dstOffsets[1] = {static_cast<int32_t>(width_),
                                    static_cast<int32_t>(height_), 1};

        vkCmdBlitImage(cmd,
            srcDepthImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            depthImage_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &blitRegion, VK_FILTER_NEAREST);
    }

    // Transition splat depth: TRANSFER_DST -> DEPTH_STENCIL_ATTACHMENT_OPTIMAL
    // (the LoadDepth render pass expects this layout as initialLayout)
    VkImageMemoryBarrier depthBarrier = {};
    depthBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    depthBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    depthBarrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    depthBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    depthBarrier.image = depthImage_;
    depthBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    depthBarrier.subresourceRange.levelCount = 1;
    depthBarrier.subresourceRange.layerCount = 1;
    depthBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    depthBarrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                  VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
        0, 0, nullptr, 0, nullptr, 1, &depthBarrier);

    ctx.endSingleTimeCommands(cmd);

    hasBackgroundDepth_ = true;
    std::cout << "[info] Preloaded raster depth into splat depth buffer ("
              << srcWidth << "x" << srcHeight << " -> "
              << width_ << "x" << height_ << ")" << std::endl;
}

// --------------------------------------------------------------------------
// Cleanup
// --------------------------------------------------------------------------

void SplatRenderer::cleanup(VulkanContext& ctx) {
    vkDeviceWaitIdle(ctx.device);

    // Framebuffer and render pass
    if (framebufferLoadDepth_ != VK_NULL_HANDLE) vkDestroyFramebuffer(ctx.device, framebufferLoadDepth_, nullptr);
    if (renderPassLoadDepth_ != VK_NULL_HANDLE)  vkDestroyRenderPass(ctx.device, renderPassLoadDepth_, nullptr);
    if (framebufferLoad_ != VK_NULL_HANDLE) vkDestroyFramebuffer(ctx.device, framebufferLoad_, nullptr);
    if (renderPassLoad_ != VK_NULL_HANDLE)  vkDestroyRenderPass(ctx.device, renderPassLoad_, nullptr);
    if (framebuffer_ != VK_NULL_HANDLE) vkDestroyFramebuffer(ctx.device, framebuffer_, nullptr);
    if (renderPass_ != VK_NULL_HANDLE)  vkDestroyRenderPass(ctx.device, renderPass_, nullptr);

    // Image views
    if (colorView_ != VK_NULL_HANDLE) vkDestroyImageView(ctx.device, colorView_, nullptr);
    if (depthView_ != VK_NULL_HANDLE) vkDestroyImageView(ctx.device, depthView_, nullptr);

    // Images (VMA)
    if (colorImage_ != VK_NULL_HANDLE) vmaDestroyImage(ctx.allocator, colorImage_, colorAlloc_);
    if (depthImage_ != VK_NULL_HANDLE) vmaDestroyImage(ctx.allocator, depthImage_, depthAlloc_);

    // Pipelines
    if (computePipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(ctx.device, computePipeline_, nullptr);
    if (renderPipeline_ != VK_NULL_HANDLE)  vkDestroyPipeline(ctx.device, renderPipeline_, nullptr);
    if (computeLayout_ != VK_NULL_HANDLE)   vkDestroyPipelineLayout(ctx.device, computeLayout_, nullptr);
    if (renderLayout_ != VK_NULL_HANDLE)    vkDestroyPipelineLayout(ctx.device, renderLayout_, nullptr);

    // Sort pipelines
    if (sortHistogramPipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(ctx.device, sortHistogramPipeline_, nullptr);
    if (sortPrefixSumPipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(ctx.device, sortPrefixSumPipeline_, nullptr);
    if (sortScatterPipeline_ != VK_NULL_HANDLE)   vkDestroyPipeline(ctx.device, sortScatterPipeline_, nullptr);
    if (sortHistogramLayout_ != VK_NULL_HANDLE)   vkDestroyPipelineLayout(ctx.device, sortHistogramLayout_, nullptr);
    if (sortPrefixSumLayout_ != VK_NULL_HANDLE)   vkDestroyPipelineLayout(ctx.device, sortPrefixSumLayout_, nullptr);
    if (sortScatterLayout_ != VK_NULL_HANDLE)     vkDestroyPipelineLayout(ctx.device, sortScatterLayout_, nullptr);

    // Descriptor layouts and pool
    if (computeSetLayout_ != VK_NULL_HANDLE)       vkDestroyDescriptorSetLayout(ctx.device, computeSetLayout_, nullptr);
    if (renderSetLayout_ != VK_NULL_HANDLE)        vkDestroyDescriptorSetLayout(ctx.device, renderSetLayout_, nullptr);
    if (sortHistogramSetLayout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(ctx.device, sortHistogramSetLayout_, nullptr);
    if (sortPrefixSumSetLayout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(ctx.device, sortPrefixSumSetLayout_, nullptr);
    if (sortScatterSetLayout_ != VK_NULL_HANDLE)   vkDestroyDescriptorSetLayout(ctx.device, sortScatterSetLayout_, nullptr);
    if (descriptorPool_ != VK_NULL_HANDLE)         vkDestroyDescriptorPool(ctx.device, descriptorPool_, nullptr);

    // Buffers (VMA)
    destroyVmaBuffer(ctx.allocator, posBuffer_, posAlloc_);
    destroyVmaBuffer(ctx.allocator, rotBuffer_, rotAlloc_);
    destroyVmaBuffer(ctx.allocator, scaleBuffer_, scaleAlloc_);
    destroyVmaBuffer(ctx.allocator, opacityBuffer_, opacityAlloc_);
    destroyVmaBuffer(ctx.allocator, projCenterBuffer_, projCenterAlloc_);
    destroyVmaBuffer(ctx.allocator, projConicBuffer_, projConicAlloc_);
    destroyVmaBuffer(ctx.allocator, projColorBuffer_, projColorAlloc_);
    destroyVmaBuffer(ctx.allocator, sortKeysBuffer_, sortKeysAlloc_);
    destroyVmaBuffer(ctx.allocator, sortedIndicesBuffer_, sortedIndicesAlloc_);
    destroyVmaBuffer(ctx.allocator, visibleCountBuffer_, visibleCountAlloc_);
    destroyVmaBuffer(ctx.allocator, sortKeysBBuffer_, sortKeysBAlloc_);
    destroyVmaBuffer(ctx.allocator, sortValsBBuffer_, sortValsBAlloc_);
    destroyVmaBuffer(ctx.allocator, histogramBuffer_, histogramAlloc_);
    destroyVmaBuffer(ctx.allocator, partitionSumsBuffer_, partitionSumsAlloc_);

    for (size_t i = 0; i < shBuffers_.size(); ++i) {
        destroyVmaBuffer(ctx.allocator, shBuffers_[i], shAllocs_[i]);
    }
    shBuffers_.clear();
    shAllocs_.clear();

    // Zero out all handles
    framebufferLoadDepth_ = VK_NULL_HANDLE;
    renderPassLoadDepth_ = VK_NULL_HANDLE;
    framebufferLoad_ = VK_NULL_HANDLE;
    renderPassLoad_ = VK_NULL_HANDLE;
    framebuffer_ = VK_NULL_HANDLE;
    renderPass_ = VK_NULL_HANDLE;
    hasBackground_ = false;
    hasBackgroundDepth_ = false;
    colorView_ = VK_NULL_HANDLE;
    depthView_ = VK_NULL_HANDLE;
    colorImage_ = VK_NULL_HANDLE;
    depthImage_ = VK_NULL_HANDLE;
    computePipeline_ = VK_NULL_HANDLE;
    renderPipeline_ = VK_NULL_HANDLE;
    computeLayout_ = VK_NULL_HANDLE;
    renderLayout_ = VK_NULL_HANDLE;
    computeSetLayout_ = VK_NULL_HANDLE;
    renderSetLayout_ = VK_NULL_HANDLE;
    sortHistogramPipeline_ = VK_NULL_HANDLE;
    sortPrefixSumPipeline_ = VK_NULL_HANDLE;
    sortScatterPipeline_ = VK_NULL_HANDLE;
    sortHistogramLayout_ = VK_NULL_HANDLE;
    sortPrefixSumLayout_ = VK_NULL_HANDLE;
    sortScatterLayout_ = VK_NULL_HANDLE;
    sortHistogramSetLayout_ = VK_NULL_HANDLE;
    sortPrefixSumSetLayout_ = VK_NULL_HANDLE;
    sortScatterSetLayout_ = VK_NULL_HANDLE;
    descriptorPool_ = VK_NULL_HANDLE;
}
