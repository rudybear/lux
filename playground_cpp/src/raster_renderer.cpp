#include "raster_renderer.h"
#include "spv_loader.h"
#include "camera.h"
#include "gltf_loader.h"
#include "reflected_pipeline.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <stdexcept>
#include <iostream>
#include <cstring>
#include <array>
#include <fstream>
#include <sstream>
#include <filesystem>

// --------------------------------------------------------------------------
// Offscreen render target creation
// --------------------------------------------------------------------------

void RasterRenderer::createOffscreenTarget(VulkanContext& ctx) {
    // Color attachment
    VkImageCreateInfo imageInfo = {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    imageInfo.extent = {renderWidth, renderHeight, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    vmaCreateImage(ctx.allocator, &imageInfo, &allocInfo,
                   &offscreenImage, &offscreenAllocation, nullptr);

    VkImageViewCreateInfo viewInfo = {};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = offscreenImage;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    vkCreateImageView(ctx.device, &viewInfo, nullptr, &offscreenImageView);

    // Depth attachment (raster scenes with 3D geometry)
    if (m_needsDepth) {
        VkImageCreateInfo depthInfo = imageInfo;
        depthInfo.format = VK_FORMAT_D32_SFLOAT;
        depthInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

        vmaCreateImage(ctx.allocator, &depthInfo, &allocInfo,
                       &depthImage, &depthAllocation, nullptr);

        VkImageViewCreateInfo depthViewInfo = viewInfo;
        depthViewInfo.image = depthImage;
        depthViewInfo.format = VK_FORMAT_D32_SFLOAT;
        depthViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

        vkCreateImageView(ctx.device, &depthViewInfo, nullptr, &depthImageView);
    }
}

// --------------------------------------------------------------------------
// Render pass creation
// --------------------------------------------------------------------------

void RasterRenderer::createRenderPass(VulkanContext& ctx) {
    std::vector<VkAttachmentDescription> attachments;
    std::vector<VkAttachmentReference> colorRefs;
    VkAttachmentReference depthRef = {};

    // Color attachment
    VkAttachmentDescription colorAtt = {};
    colorAtt.format = VK_FORMAT_R8G8B8A8_UNORM;
    colorAtt.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAtt.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    attachments.push_back(colorAtt);

    VkAttachmentReference colorRef = {};
    colorRef.attachment = 0;
    colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorRefs.push_back(colorRef);

    // Depth attachment (raster scenes with 3D geometry)
    if (m_needsDepth) {
        VkAttachmentDescription depthAtt = {};
        depthAtt.format = VK_FORMAT_D32_SFLOAT;
        depthAtt.samples = VK_SAMPLE_COUNT_1_BIT;
        depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depthAtt.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        attachments.push_back(depthAtt);

        depthRef.attachment = 1;
        depthRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    }

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = static_cast<uint32_t>(colorRefs.size());
    subpass.pColorAttachments = colorRefs.data();
    if (m_needsDepth) {
        subpass.pDepthStencilAttachment = &depthRef;
    }

    VkSubpassDependency dependency = {};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                              VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                              VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                               VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo rpInfo = {};
    rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    rpInfo.pAttachments = attachments.data();
    rpInfo.subpassCount = 1;
    rpInfo.pSubpasses = &subpass;
    rpInfo.dependencyCount = 1;
    rpInfo.pDependencies = &dependency;

    if (vkCreateRenderPass(ctx.device, &rpInfo, nullptr, &renderPass) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create render pass");
    }
}

// --------------------------------------------------------------------------
// Framebuffer creation
// --------------------------------------------------------------------------

void RasterRenderer::createFramebuffer(VulkanContext& ctx) {
    std::vector<VkImageView> attachments = {offscreenImageView};
    if (m_needsDepth) {
        attachments.push_back(depthImageView);
    }

    VkFramebufferCreateInfo fbInfo = {};
    fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fbInfo.renderPass = renderPass;
    fbInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    fbInfo.pAttachments = attachments.data();
    fbInfo.width = renderWidth;
    fbInfo.height = renderHeight;
    fbInfo.layers = 1;

    if (vkCreateFramebuffer(ctx.device, &fbInfo, nullptr, &framebuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create framebuffer");
    }
}

// --------------------------------------------------------------------------
// Triangle pipeline
// --------------------------------------------------------------------------

void RasterRenderer::createPipelineTriangle(VulkanContext& ctx) {
    // Pipeline layout: no descriptor sets
    VkPipelineLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;

    vkCreatePipelineLayout(ctx.device, &layoutInfo, nullptr, &pipelineLayout);

    // Shader stages
    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertModule;
    stages[0].pName = "main";

    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragModule;
    stages[1].pName = "main";

    // Vertex input: position (vec3, offset 0) + color (vec3, offset 12)
    VkVertexInputBindingDescription binding = {};
    binding.binding = 0;
    binding.stride = sizeof(TriangleVertex); // 24 bytes
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attrs[2] = {};
    attrs[0].location = 0;
    attrs[0].binding = 0;
    attrs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attrs[0].offset = 0;

    attrs[1].location = 1;
    attrs[1].binding = 0;
    attrs[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attrs[1].offset = 12;

    VkPipelineVertexInputStateCreateInfo vertexInput = {};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount = 1;
    vertexInput.pVertexBindingDescriptions = &binding;
    vertexInput.vertexAttributeDescriptionCount = 2;
    vertexInput.pVertexAttributeDescriptions = attrs;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport viewport = {};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(renderWidth);
    viewport.height = static_cast<float>(renderHeight);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor = {};
    scissor.offset = {0, 0};
    scissor.extent = {renderWidth, renderHeight};

    VkPipelineViewportStateCreateInfo viewportState = {};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterizer = {};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo multisampling = {};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                                          VK_COLOR_COMPONENT_G_BIT |
                                          VK_COLOR_COMPONENT_B_BIT |
                                          VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlending = {};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    VkGraphicsPipelineCreateInfo pipeInfo = {};
    pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeInfo.stageCount = 2;
    pipeInfo.pStages = stages;
    pipeInfo.pVertexInputState = &vertexInput;
    pipeInfo.pInputAssemblyState = &inputAssembly;
    pipeInfo.pViewportState = &viewportState;
    pipeInfo.pRasterizationState = &rasterizer;
    pipeInfo.pMultisampleState = &multisampling;
    pipeInfo.pColorBlendState = &colorBlending;
    pipeInfo.layout = pipelineLayout;
    pipeInfo.renderPass = renderPass;
    pipeInfo.subpass = 0;

    if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &pipeInfo,
                                  nullptr, &pipeline) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create triangle graphics pipeline");
    }
}

// --------------------------------------------------------------------------
// Fullscreen pipeline
// --------------------------------------------------------------------------

void RasterRenderer::createPipelineFullscreen(VulkanContext& ctx) {
    // Pipeline layout: no descriptor sets
    VkPipelineLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;

    vkCreatePipelineLayout(ctx.device, &layoutInfo, nullptr, &pipelineLayout);

    // Shader stages
    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertModule;
    stages[0].pName = "main";

    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragModule;
    stages[1].pName = "main";

    // No vertex input (uses gl_VertexIndex)
    VkPipelineVertexInputStateCreateInfo vertexInput = {};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport viewport = {};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(renderWidth);
    viewport.height = static_cast<float>(renderHeight);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor = {};
    scissor.offset = {0, 0};
    scissor.extent = {renderWidth, renderHeight};

    VkPipelineViewportStateCreateInfo viewportState = {};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterizer = {};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE;

    VkPipelineMultisampleStateCreateInfo multisampling = {};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                                          VK_COLOR_COMPONENT_G_BIT |
                                          VK_COLOR_COMPONENT_B_BIT |
                                          VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlending = {};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    VkGraphicsPipelineCreateInfo pipeInfo = {};
    pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeInfo.stageCount = 2;
    pipeInfo.pStages = stages;
    pipeInfo.pVertexInputState = &vertexInput;
    pipeInfo.pInputAssemblyState = &inputAssembly;
    pipeInfo.pViewportState = &viewportState;
    pipeInfo.pRasterizationState = &rasterizer;
    pipeInfo.pMultisampleState = &multisampling;
    pipeInfo.pColorBlendState = &colorBlending;
    pipeInfo.layout = pipelineLayout;
    pipeInfo.renderPass = renderPass;
    pipeInfo.subpass = 0;

    if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &pipeInfo,
                                  nullptr, &pipeline) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create fullscreen graphics pipeline");
    }
}

// --------------------------------------------------------------------------
// Default 1x1 white texture for missing texture bindings
// --------------------------------------------------------------------------

void RasterRenderer::createDefaultWhiteTexture(VulkanContext& ctx) {
    std::vector<uint8_t> white = {255, 255, 255, 255};
    defaultWhiteTexture = Scene::uploadTexture(ctx.allocator, ctx.device, ctx.commandPool,
                                                ctx.graphicsQueue, white, 1, 1);
    // Black texture for emissive (no emission by default)
    std::vector<uint8_t> black = {0, 0, 0, 255};
    defaultBlackTexture = Scene::uploadTexture(ctx.allocator, ctx.device, ctx.commandPool,
                                                ctx.graphicsQueue, black, 1, 1);
    // Flat normal texture (128,128,255) = tangent-space (0,0,1)
    std::vector<uint8_t> flatNormal = {128, 128, 255, 255};
    defaultNormalTexture = Scene::uploadTexture(ctx.allocator, ctx.device, ctx.commandPool,
                                                ctx.graphicsQueue, flatNormal, 1, 1);
    std::cout << "[info] Created default fallback textures (white, black, flat-normal)" << std::endl;
}

// --------------------------------------------------------------------------
// Upload glTF texture images to GPU
// --------------------------------------------------------------------------

void RasterRenderer::uploadGltfTextures(VulkanContext& ctx) {
    if (!m_hasGltfScene || m_gltfScene.materials.empty()) return;

    // Use the first material (most common case for single-material models)
    auto& mat = m_gltfScene.materials[0];

    auto uploadIfValid = [&](const GltfTextureData& texData, const std::string& name) {
        if (texData.valid()) {
            auto gpuTex = Scene::uploadTexture(ctx.allocator, ctx.device, ctx.commandPool,
                                                ctx.graphicsQueue, texData.pixels,
                                                static_cast<uint32_t>(texData.width),
                                                static_cast<uint32_t>(texData.height));
            namedTextures[name] = gpuTex;
            std::cout << "[info] Uploaded GPU texture: " << name
                      << " (" << texData.width << "x" << texData.height << ")" << std::endl;
        }
    };

    uploadIfValid(mat.base_color_tex, "base_color_tex");
    uploadIfValid(mat.normal_tex, "normal_tex");
    uploadIfValid(mat.metallic_roughness_tex, "metallic_roughness_tex");
    uploadIfValid(mat.occlusion_tex, "occlusion_tex");
    uploadIfValid(mat.emissive_tex, "emissive_tex");
}

// --------------------------------------------------------------------------
// Cubemap upload (F16 RGBA data, 6 faces, N mip levels)
// --------------------------------------------------------------------------

GPUTexture RasterRenderer::uploadCubemapF16(VulkanContext& ctx, uint32_t faceSize,
                                             uint32_t mipCount,
                                             const std::vector<uint16_t>& data) {
    GPUTexture tex;
    tex.width = faceSize;
    tex.height = faceSize;

    // Create cubemap image
    VkImageCreateInfo imageInfo = {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    imageInfo.extent = {faceSize, faceSize, 1};
    imageInfo.mipLevels = mipCount;
    imageInfo.arrayLayers = 6;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;

    VmaAllocationCreateInfo imageAllocInfo = {};
    imageAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    VkResult result = vmaCreateImage(ctx.allocator, &imageInfo, &imageAllocInfo,
                                     &tex.image, &tex.allocation, nullptr);
    if (result != VK_SUCCESS) {
        std::cerr << "[warn] Failed to create cubemap image" << std::endl;
        return tex;
    }

    // Create staging buffer for all data
    VkDeviceSize dataSize = data.size() * sizeof(uint16_t);
    VkBuffer stagingBuffer;
    VmaAllocation stagingAlloc;
    VkBufferCreateInfo stagingInfo = {};
    stagingInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    stagingInfo.size = dataSize;
    stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VmaAllocationCreateInfo stagingAllocInfo = {};
    stagingAllocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;
    vmaCreateBuffer(ctx.allocator, &stagingInfo, &stagingAllocInfo,
                    &stagingBuffer, &stagingAlloc, nullptr);

    void* mapped;
    vmaMapMemory(ctx.allocator, stagingAlloc, &mapped);
    memcpy(mapped, data.data(), dataSize);
    vmaUnmapMemory(ctx.allocator, stagingAlloc);

    // Transition image to TRANSFER_DST
    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();

    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = tex.image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = mipCount;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 6;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    // Copy each face+mip from the staging buffer
    // Data layout: for each mip level, for each face: faceWidth*faceHeight*4 half-floats
    VkDeviceSize offset = 0;
    for (uint32_t mip = 0; mip < mipCount; mip++) {
        uint32_t mipSize = std::max(1u, faceSize >> mip);
        VkDeviceSize faceBytes = static_cast<VkDeviceSize>(mipSize) * mipSize * 4 * sizeof(uint16_t);

        for (uint32_t face = 0; face < 6; face++) {
            VkBufferImageCopy region = {};
            region.bufferOffset = offset;
            region.bufferRowLength = 0;
            region.bufferImageHeight = 0;
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.mipLevel = mip;
            region.imageSubresource.baseArrayLayer = face;
            region.imageSubresource.layerCount = 1;
            region.imageOffset = {0, 0, 0};
            region.imageExtent = {mipSize, mipSize, 1};

            vkCmdCopyBufferToImage(cmd, stagingBuffer, tex.image,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
            offset += faceBytes;
        }
    }

    // Transition to SHADER_READ_ONLY
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    ctx.endSingleTimeCommands(cmd);

    vmaDestroyBuffer(ctx.allocator, stagingBuffer, stagingAlloc);

    // Create cube image view
    VkImageViewCreateInfo viewInfo = {};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = tex.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
    viewInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = mipCount;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 6;

    vkCreateImageView(ctx.device, &viewInfo, nullptr, &tex.imageView);

    // Create sampler with linear filtering + mipmap
    VkSamplerCreateInfo samplerInfo = {};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = static_cast<float>(mipCount);

    vkCreateSampler(ctx.device, &samplerInfo, nullptr, &tex.sampler);

    return tex;
}

// --------------------------------------------------------------------------
// IBL asset loading
// --------------------------------------------------------------------------

void RasterRenderer::loadIBLAssets(VulkanContext& ctx, const std::string& iblName) {
    namespace fs = std::filesystem;

    std::string iblDir = "playground/assets/ibl/" + iblName;
    std::string manifestPath = iblDir + "/manifest.json";

    if (!fs::exists(manifestPath)) {
        std::cout << "[info] No IBL assets found at " << manifestPath
                  << ", skipping IBL loading" << std::endl;
        return;
    }

    // Read manifest JSON
    std::ifstream manifestFile(manifestPath);
    if (!manifestFile.is_open()) {
        std::cerr << "[warn] Failed to open IBL manifest: " << manifestPath << std::endl;
        return;
    }
    std::stringstream ss;
    ss << manifestFile.rdbuf();
    std::string manifestContent = ss.str();

    // Simple JSON parsing for nested manifest fields
    // Manifest structure: { "specular": { "face_size": N, "mip_count": N }, "irradiance": { "face_size": N }, "brdf_lut": { "size": N } }
    auto extractInt = [&](const std::string& json, const std::string& key) -> int {
        auto pos = json.find("\"" + key + "\"");
        if (pos == std::string::npos) return 0;
        pos = json.find(":", pos);
        if (pos == std::string::npos) return 0;
        pos++;
        while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' || json[pos] == '\n' || json[pos] == '\r')) pos++;
        return std::stoi(json.substr(pos));
    };

    // Find the "specular" section and extract face_size and mip_count within it
    auto extractNestedInt = [&](const std::string& json, const std::string& section, const std::string& key) -> int {
        auto secPos = json.find("\"" + section + "\"");
        if (secPos == std::string::npos) return 0;
        auto bracePos = json.find("{", secPos);
        if (bracePos == std::string::npos) return 0;
        // Find the closing brace
        int depth = 1;
        size_t endPos = bracePos + 1;
        while (endPos < json.size() && depth > 0) {
            if (json[endPos] == '{') depth++;
            else if (json[endPos] == '}') depth--;
            endPos++;
        }
        std::string sub = json.substr(bracePos, endPos - bracePos);
        return extractInt(sub, key);
    };

    int specSize = extractNestedInt(manifestContent, "specular", "face_size");
    int specMips = extractNestedInt(manifestContent, "specular", "mip_count");
    int irrSize = extractNestedInt(manifestContent, "irradiance", "face_size");
    int brdfSize = extractNestedInt(manifestContent, "brdf_lut", "size");

    if (specSize <= 0 || specMips <= 0 || irrSize <= 0 || brdfSize <= 0) {
        std::cerr << "[warn] Invalid IBL manifest values" << std::endl;
        return;
    }

    std::cout << "[info] Loading IBL assets: " << iblName
              << " (spec=" << specSize << "x" << specSize << " mips=" << specMips
              << ", irr=" << irrSize << "x" << irrSize
              << ", brdf=" << brdfSize << "x" << brdfSize << ")" << std::endl;

    // Load specular cubemap
    {
        std::string specPath = iblDir + "/specular.bin";
        std::ifstream specFile(specPath, std::ios::binary | std::ios::ate);
        if (specFile.is_open()) {
            size_t fileSize = specFile.tellg();
            specFile.seekg(0);
            std::vector<uint16_t> specData(fileSize / sizeof(uint16_t));
            specFile.read(reinterpret_cast<char*>(specData.data()), fileSize);

            auto specTex = uploadCubemapF16(ctx, static_cast<uint32_t>(specSize),
                                             static_cast<uint32_t>(specMips), specData);
            if (specTex.image != VK_NULL_HANDLE) {
                iblTextures["env_specular"] = specTex;
                std::cout << "[info] Loaded IBL specular cubemap" << std::endl;
            }
        } else {
            std::cerr << "[warn] Failed to open " << specPath << std::endl;
        }
    }

    // Load irradiance cubemap
    {
        std::string irrPath = iblDir + "/irradiance.bin";
        std::ifstream irrFile(irrPath, std::ios::binary | std::ios::ate);
        if (irrFile.is_open()) {
            size_t fileSize = irrFile.tellg();
            irrFile.seekg(0);
            std::vector<uint16_t> irrData(fileSize / sizeof(uint16_t));
            irrFile.read(reinterpret_cast<char*>(irrData.data()), fileSize);

            auto irrTex = uploadCubemapF16(ctx, static_cast<uint32_t>(irrSize), 1, irrData);
            if (irrTex.image != VK_NULL_HANDLE) {
                iblTextures["env_irradiance"] = irrTex;
                std::cout << "[info] Loaded IBL irradiance cubemap" << std::endl;
            }
        } else {
            std::cerr << "[warn] Failed to open " << irrPath << std::endl;
        }
    }

    // Load BRDF LUT (2D texture, RG16F data padded to RGBA16F)
    {
        std::string brdfPath = iblDir + "/brdf_lut.bin";
        std::ifstream brdfFile(brdfPath, std::ios::binary | std::ios::ate);
        if (brdfFile.is_open()) {
            size_t fileSize = brdfFile.tellg();
            brdfFile.seekg(0);
            std::vector<uint16_t> rawData(fileSize / sizeof(uint16_t));
            brdfFile.read(reinterpret_cast<char*>(rawData.data()), fileSize);

            // Determine if data is RG (2 channels) or RGBA (4 channels)
            size_t totalPixels = static_cast<size_t>(brdfSize) * brdfSize;
            std::vector<uint16_t> rgbaData;

            if (rawData.size() == totalPixels * 2) {
                // RG data - pad to RGBA
                rgbaData.resize(totalPixels * 4);
                for (size_t p = 0; p < totalPixels; p++) {
                    rgbaData[p * 4 + 0] = rawData[p * 2 + 0];
                    rgbaData[p * 4 + 1] = rawData[p * 2 + 1];
                    rgbaData[p * 4 + 2] = 0;
                    rgbaData[p * 4 + 3] = 0x3C00; // 1.0 in half-float
                }
            } else {
                // Already RGBA
                rgbaData = rawData;
            }

            // Upload as 2D texture (not cubemap)
            GPUTexture brdfTex;
            brdfTex.width = static_cast<uint32_t>(brdfSize);
            brdfTex.height = static_cast<uint32_t>(brdfSize);

            VkDeviceSize imgSize = rgbaData.size() * sizeof(uint16_t);

            VkBuffer stagBuf;
            VmaAllocation stagAlloc;
            VkBufferCreateInfo stagInfo = {};
            stagInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            stagInfo.size = imgSize;
            stagInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            VmaAllocationCreateInfo stagAllocInfo = {};
            stagAllocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;
            vmaCreateBuffer(ctx.allocator, &stagInfo, &stagAllocInfo,
                            &stagBuf, &stagAlloc, nullptr);

            void* mapped;
            vmaMapMemory(ctx.allocator, stagAlloc, &mapped);
            memcpy(mapped, rgbaData.data(), imgSize);
            vmaUnmapMemory(ctx.allocator, stagAlloc);

            VkImageCreateInfo imgInfo = {};
            imgInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            imgInfo.imageType = VK_IMAGE_TYPE_2D;
            imgInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
            imgInfo.extent = {static_cast<uint32_t>(brdfSize), static_cast<uint32_t>(brdfSize), 1};
            imgInfo.mipLevels = 1;
            imgInfo.arrayLayers = 1;
            imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
            imgInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

            VmaAllocationCreateInfo imgAllocInfo = {};
            imgAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

            vmaCreateImage(ctx.allocator, &imgInfo, &imgAllocInfo,
                           &brdfTex.image, &brdfTex.allocation, nullptr);

            VkCommandBuffer cmd = ctx.beginSingleTimeCommands();

            VkImageMemoryBarrier barrier = {};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = brdfTex.image;
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

            VkBufferImageCopy region = {};
            region.bufferOffset = 0;
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.mipLevel = 0;
            region.imageSubresource.baseArrayLayer = 0;
            region.imageSubresource.layerCount = 1;
            region.imageExtent = {static_cast<uint32_t>(brdfSize), static_cast<uint32_t>(brdfSize), 1};

            vkCmdCopyBufferToImage(cmd, stagBuf, brdfTex.image,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0, 0, nullptr, 0, nullptr, 1, &barrier);

            ctx.endSingleTimeCommands(cmd);
            vmaDestroyBuffer(ctx.allocator, stagBuf, stagAlloc);

            VkImageViewCreateInfo viewInfo = {};
            viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            viewInfo.image = brdfTex.image;
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
            viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            viewInfo.subresourceRange.baseMipLevel = 0;
            viewInfo.subresourceRange.levelCount = 1;
            viewInfo.subresourceRange.baseArrayLayer = 0;
            viewInfo.subresourceRange.layerCount = 1;

            vkCreateImageView(ctx.device, &viewInfo, nullptr, &brdfTex.imageView);

            VkSamplerCreateInfo samplerInfo = {};
            samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            samplerInfo.magFilter = VK_FILTER_LINEAR;
            samplerInfo.minFilter = VK_FILTER_LINEAR;
            samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            samplerInfo.anisotropyEnable = VK_FALSE;
            samplerInfo.maxAnisotropy = 1.0f;
            samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
            samplerInfo.unnormalizedCoordinates = VK_FALSE;
            samplerInfo.compareEnable = VK_FALSE;
            samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

            vkCreateSampler(ctx.device, &samplerInfo, nullptr, &brdfTex.sampler);

            iblTextures["brdf_lut"] = brdfTex;
            std::cout << "[info] Loaded IBL BRDF LUT" << std::endl;
        } else {
            std::cerr << "[warn] Failed to open " << brdfPath << std::endl;
        }
    }

    std::cout << "[info] IBL loading complete: " << iblTextures.size() << " texture(s)" << std::endl;
}

// --------------------------------------------------------------------------
// Get texture for a binding name, falling back to default white
// --------------------------------------------------------------------------

GPUTexture& RasterRenderer::getTextureForBinding(const std::string& name) {
    auto it = namedTextures.find(name);
    if (it != namedTextures.end()) {
        return it->second;
    }

    // For "albedo_tex" used by pbr_surface, try "base_color_tex" as well
    if (name == "albedo_tex") {
        it = namedTextures.find("base_color_tex");
        if (it != namedTextures.end()) {
            return it->second;
        }
    }

    // Use semantically correct defaults for missing textures:
    // - emissive: black (no emission) — white would wash out everything
    // - normal: flat normal (128,128,255) — white would tilt all normals 45°
    // - everything else: white (identity for multiplicative textures)
    if (name == "emissive_tex") {
        return defaultBlackTexture;
    }
    if (name == "normal_tex") {
        return defaultNormalTexture;
    }

    return defaultWhiteTexture;
}

// --------------------------------------------------------------------------
// PBR resources: uniforms only (descriptors are now reflection-driven)
// --------------------------------------------------------------------------

void RasterRenderer::setupPBRResources(VulkanContext& ctx) {
    float aspect = static_cast<float>(renderWidth) / static_cast<float>(renderHeight);

    // Create MVP uniform buffer (3 x mat4 = 192 bytes)
    struct MVPData {
        glm::mat4 model;
        glm::mat4 view;
        glm::mat4 projection;
    };
    MVPData mvpData;
    if (m_hasSceneBounds) {
        // Use auto-camera computed from scene bounds
        mvpData.model = glm::mat4(1.0f); // identity - transforms already baked into vertices
        mvpData.view = glm::lookAt(m_autoEye, m_autoTarget, m_autoUp);
        mvpData.projection = glm::perspective(glm::radians(45.0f), aspect, 0.1f, m_autoFar);
        mvpData.projection[1][1] *= -1.0f; // Vulkan Y-flip
    } else {
        auto cam = Camera::getDefaultMatrices(aspect);
        mvpData = {cam.model, cam.view, cam.projection};
    }

    VkBufferCreateInfo mvpBufInfo = {};
    mvpBufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    mvpBufInfo.size = sizeof(MVPData);
    mvpBufInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

    VmaAllocationCreateInfo mvpAllocInfo = {};
    mvpAllocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;

    vmaCreateBuffer(ctx.allocator, &mvpBufInfo, &mvpAllocInfo,
                    &mvpBuffer, &mvpAllocation, nullptr);

    void* mapped;
    vmaMapMemory(ctx.allocator, mvpAllocation, &mapped);
    memcpy(mapped, &mvpData, sizeof(MVPData));
    vmaUnmapMemory(ctx.allocator, mvpAllocation);

    // Create Light uniform buffer (32 bytes: vec3 light_dir padded + vec3 view_pos padded)
    struct LightData {
        glm::vec3 lightDir;
        float _pad0;
        glm::vec3 viewPos;
        float _pad1;
    };

    glm::vec3 lightDir = glm::normalize(glm::vec3(1.0f, 0.8f, 0.6f));
    glm::vec3 viewPos = m_hasSceneBounds ? m_autoEye : Camera::DEFAULT_EYE;
    LightData lightData = {lightDir, 0.0f, viewPos, 0.0f};

    VkBufferCreateInfo lightBufInfo = {};
    lightBufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    lightBufInfo.size = sizeof(LightData);
    lightBufInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

    VmaAllocationCreateInfo lightAllocInfo = {};
    lightAllocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;

    vmaCreateBuffer(ctx.allocator, &lightBufInfo, &lightAllocInfo,
                    &lightBuffer, &lightAllocation, nullptr);

    vmaMapMemory(ctx.allocator, lightAllocation, &mapped);
    memcpy(mapped, &lightData, sizeof(LightData));
    vmaUnmapMemory(ctx.allocator, lightAllocation);

    // Generate sphere mesh only if no mesh was already uploaded (e.g. from glTF)
    if (mesh.vertexBuffer == VK_NULL_HANDLE) {
        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices;
        Scene::generateSphere(32, 32, vertices, indices);

        // Check if vertex reflection requires 48-byte stride (with tangent)
        if (vertReflection.vertex_stride == 48) {
            // Pad sphere vertices to 48 bytes: pos(3)+normal(3)+uv(2)+tangent(4) = 12 floats
            std::vector<float> vdata;
            vdata.reserve(vertices.size() * 12);
            for (auto& v : vertices) {
                vdata.push_back(v.position.x);
                vdata.push_back(v.position.y);
                vdata.push_back(v.position.z);
                vdata.push_back(v.normal.x);
                vdata.push_back(v.normal.y);
                vdata.push_back(v.normal.z);
                vdata.push_back(v.uv.x);
                vdata.push_back(v.uv.y);
                // Default tangent
                vdata.push_back(1.0f);
                vdata.push_back(0.0f);
                vdata.push_back(0.0f);
                vdata.push_back(1.0f);
            }

            // Upload raw 48-byte vertex data
            VkDeviceSize vbufSize = vdata.size() * sizeof(float);
            VkBufferCreateInfo vbufInfo = {};
            vbufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            vbufInfo.size = vbufSize;
            vbufInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                             VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                             VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
            VmaAllocationCreateInfo vbufAllocInfo = {};
            vbufAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
            vmaCreateBuffer(ctx.allocator, &vbufInfo, &vbufAllocInfo,
                            &mesh.vertexBuffer, &mesh.vertexAllocation, nullptr);

            VkBuffer vStagingBuf;
            VmaAllocation vStagingAlloc;
            VkBufferCreateInfo vStgInfo = {};
            vStgInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            vStgInfo.size = vbufSize;
            vStgInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            VmaAllocationCreateInfo vStgAllocInfo = {};
            vStgAllocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;
            vmaCreateBuffer(ctx.allocator, &vStgInfo, &vStgAllocInfo,
                            &vStagingBuf, &vStagingAlloc, nullptr);

            void* vMapped;
            vmaMapMemory(ctx.allocator, vStagingAlloc, &vMapped);
            memcpy(vMapped, vdata.data(), vbufSize);
            vmaUnmapMemory(ctx.allocator, vStagingAlloc);

            // Upload index data
            VkDeviceSize ibufSize = indices.size() * sizeof(uint32_t);
            VkBufferCreateInfo ibufInfo = {};
            ibufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            ibufInfo.size = ibufSize;
            ibufInfo.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                             VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                             VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
            VmaAllocationCreateInfo ibufAllocInfo = {};
            ibufAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
            vmaCreateBuffer(ctx.allocator, &ibufInfo, &ibufAllocInfo,
                            &mesh.indexBuffer, &mesh.indexAllocation, nullptr);

            VkBuffer iStagingBuf;
            VmaAllocation iStagingAlloc;
            VkBufferCreateInfo iStgInfo = {};
            iStgInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            iStgInfo.size = ibufSize;
            iStgInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            VmaAllocationCreateInfo iStgAllocInfo = {};
            iStgAllocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;
            vmaCreateBuffer(ctx.allocator, &iStgInfo, &iStgAllocInfo,
                            &iStagingBuf, &iStagingAlloc, nullptr);

            void* iMapped;
            vmaMapMemory(ctx.allocator, iStagingAlloc, &iMapped);
            memcpy(iMapped, indices.data(), ibufSize);
            vmaUnmapMemory(ctx.allocator, iStagingAlloc);

            VkCommandBuffer copyCmd = ctx.beginSingleTimeCommands();
            VkBufferCopy vCopy = {}; vCopy.size = vbufSize;
            vkCmdCopyBuffer(copyCmd, vStagingBuf, mesh.vertexBuffer, 1, &vCopy);
            VkBufferCopy iCopy = {}; iCopy.size = ibufSize;
            vkCmdCopyBuffer(copyCmd, iStagingBuf, mesh.indexBuffer, 1, &iCopy);
            ctx.endSingleTimeCommands(copyCmd);

            vmaDestroyBuffer(ctx.allocator, vStagingBuf, vStagingAlloc);
            vmaDestroyBuffer(ctx.allocator, iStagingBuf, iStagingAlloc);

            mesh.vertexCount = static_cast<uint32_t>(vertices.size());
            mesh.indexCount = static_cast<uint32_t>(indices.size());

            std::cout << "[info] Uploaded sphere mesh with 48-byte stride" << std::endl;
        } else {
            mesh = Scene::uploadMesh(ctx.allocator, ctx.device, ctx.commandPool,
                                     ctx.graphicsQueue, vertices, indices);
        }
    }

    // Create default 1x1 white texture for missing bindings
    createDefaultWhiteTexture(ctx);

    // Upload glTF textures if available
    uploadGltfTextures(ctx);

    // Auto-detect and load IBL assets for glTF scenes
    if (m_hasGltfScene) {
        std::string iblNames[] = {"pisa", "neutral"};
        for (auto& name : iblNames) {
            std::string testPath = "playground/assets/ibl/" + name + "/manifest.json";
            std::ifstream testFile(testPath);
            if (testFile.good()) {
                loadIBLAssets(ctx, name);
                break;
            }
        }
    }

    // If no named textures at all (non-glTF scene), create a procedural one as "albedo_tex"/"base_color_tex"
    if (namedTextures.empty()) {
        std::cout << "Generating procedural texture..." << std::endl;
        auto texPixels = Scene::generateProceduralTexture(512);
        auto gpuTex = Scene::uploadTexture(ctx.allocator, ctx.device, ctx.commandPool,
                                            ctx.graphicsQueue, texPixels, 512, 512);
        namedTextures["albedo_tex"] = gpuTex;
        namedTextures["base_color_tex"] = gpuTex;
    }

}

// --------------------------------------------------------------------------
// Reflection-driven descriptor set creation
// --------------------------------------------------------------------------

void RasterRenderer::setupReflectedDescriptors(VulkanContext& ctx) {
    // Use reflected_pipeline utilities to create descriptor set layouts from reflection JSON
    std::vector<ReflectionData> stages = {vertReflection, fragReflection};
    reflectedSetLayouts = createDescriptorSetLayoutsMultiStage(ctx.device, stages);

    // Get merged binding info for pool sizing and descriptor writes
    auto mergedBindings = getMergedBindings(stages);

    // Count pool sizes from reflection data
    std::unordered_map<VkDescriptorType, uint32_t> typeCounts;
    for (auto& b : mergedBindings) {
        VkDescriptorType vkType;
        if (b.type == "uniform_buffer") vkType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        else if (b.type == "sampler") vkType = VK_DESCRIPTOR_TYPE_SAMPLER;
        else if (b.type == "sampled_image") vkType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        else if (b.type == "sampled_cube_image") vkType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        else if (b.type == "storage_image") vkType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        else if (b.type == "acceleration_structure") vkType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        else vkType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        typeCounts[vkType]++;
    }

    std::vector<VkDescriptorPoolSize> poolSizes;
    for (auto& [type, count] : typeCounts) {
        poolSizes.push_back({type, count});
    }

    uint32_t maxSets = static_cast<uint32_t>(reflectedSetLayouts.size());

    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = maxSets;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();

    vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr, &descriptorPool);

    // Allocate descriptor sets
    for (auto& [setIdx, layout] : reflectedSetLayouts) {
        VkDescriptorSetAllocateInfo allocSetInfo = {};
        allocSetInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocSetInfo.descriptorPool = descriptorPool;
        allocSetInfo.descriptorSetCount = 1;
        allocSetInfo.pSetLayouts = &layout;

        VkDescriptorSet set;
        vkAllocateDescriptorSets(ctx.device, &allocSetInfo, &set);
        reflectedDescSets[setIdx] = set;
    }

    // Write descriptor sets based on reflection binding names
    // We need to keep descriptor info structs alive until vkUpdateDescriptorSets returns
    struct DescWriteInfo {
        VkDescriptorBufferInfo bufferInfo;
        VkDescriptorImageInfo imageInfo;
    };
    std::vector<DescWriteInfo> writeInfos(mergedBindings.size());
    std::vector<VkWriteDescriptorSet> writes;

    for (size_t i = 0; i < mergedBindings.size(); i++) {
        auto& b = mergedBindings[i];
        auto setIt = reflectedDescSets.find(b.set);
        if (setIt == reflectedDescSets.end()) continue;

        VkWriteDescriptorSet w = {};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = setIt->second;
        w.dstBinding = static_cast<uint32_t>(b.binding);
        w.descriptorCount = 1;

        if (b.type == "uniform_buffer") {
            w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            // Match by name: "MVP" -> mvpBuffer, "Light" -> lightBuffer
            if (b.name == "MVP") {
                writeInfos[i].bufferInfo = {mvpBuffer, 0, static_cast<VkDeviceSize>(b.size > 0 ? b.size : 192)};
            } else if (b.name == "Light") {
                writeInfos[i].bufferInfo = {lightBuffer, 0, static_cast<VkDeviceSize>(b.size > 0 ? b.size : 32)};
            } else {
                // Unknown UBO name - use MVP as fallback
                writeInfos[i].bufferInfo = {mvpBuffer, 0, static_cast<VkDeviceSize>(b.size > 0 ? b.size : 192)};
            }
            w.pBufferInfo = &writeInfos[i].bufferInfo;
        } else if (b.type == "sampler") {
            w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
            // Check IBL textures first, then regular textures
            auto iblIt = iblTextures.find(b.name);
            if (iblIt != iblTextures.end()) {
                writeInfos[i].imageInfo = {};
                writeInfos[i].imageInfo.sampler = iblIt->second.sampler;
            } else {
                auto& tex = getTextureForBinding(b.name);
                writeInfos[i].imageInfo = {};
                writeInfos[i].imageInfo.sampler = tex.sampler;
            }
            w.pImageInfo = &writeInfos[i].imageInfo;
        } else if (b.type == "sampled_image" || b.type == "sampled_cube_image") {
            w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            // Check IBL textures first, then regular textures
            auto iblIt = iblTextures.find(b.name);
            if (iblIt != iblTextures.end()) {
                writeInfos[i].imageInfo = {};
                writeInfos[i].imageInfo.imageView = iblIt->second.imageView;
                writeInfos[i].imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            } else {
                auto& tex = getTextureForBinding(b.name);
                writeInfos[i].imageInfo = {};
                writeInfos[i].imageInfo.imageView = tex.imageView;
                writeInfos[i].imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            }
            w.pImageInfo = &writeInfos[i].imageInfo;
        } else {
            continue; // Skip unsupported types
        }

        writes.push_back(w);
    }


    if (!writes.empty()) {
        vkUpdateDescriptorSets(ctx.device, static_cast<uint32_t>(writes.size()),
                               writes.data(), 0, nullptr);
    }

    std::cout << "[info] Reflection-driven descriptors created: "
              << reflectedSetLayouts.size() << " set(s), "
              << mergedBindings.size() << " binding(s)" << std::endl;
}

// --------------------------------------------------------------------------
// PBR pipeline (reflection-driven layout)
// --------------------------------------------------------------------------

void RasterRenderer::createPipelinePBR(VulkanContext& ctx) {
    // Pipeline layout from reflection data
    pipelineLayout = createReflectedPipelineLayoutMultiStage(
        ctx.device, reflectedSetLayouts,
        {vertReflection, fragReflection});

    // Shader stages
    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertModule;
    stages[0].pName = "main";

    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragModule;
    stages[1].pName = "main";

    // Vertex input from reflection data (supports both 32-byte and 48-byte stride)
    auto reflectedInput = createReflectedVertexInput(vertReflection);

    VkPipelineVertexInputStateCreateInfo vertexInput = {};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount = 1;
    vertexInput.pVertexBindingDescriptions = &reflectedInput.binding;
    vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(reflectedInput.attributes.size());
    vertexInput.pVertexAttributeDescriptions = reflectedInput.attributes.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport viewport = {};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(renderWidth);
    viewport.height = static_cast<float>(renderHeight);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor = {};
    scissor.offset = {0, 0};
    scissor.extent = {renderWidth, renderHeight};

    VkPipelineViewportStateCreateInfo viewportState = {};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterizer = {};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo multisampling = {};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil = {};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                                          VK_COLOR_COMPONENT_G_BIT |
                                          VK_COLOR_COMPONENT_B_BIT |
                                          VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlending = {};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    VkGraphicsPipelineCreateInfo pipeInfo = {};
    pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeInfo.stageCount = 2;
    pipeInfo.pStages = stages;
    pipeInfo.pVertexInputState = &vertexInput;
    pipeInfo.pInputAssemblyState = &inputAssembly;
    pipeInfo.pViewportState = &viewportState;
    pipeInfo.pRasterizationState = &rasterizer;
    pipeInfo.pMultisampleState = &multisampling;
    pipeInfo.pDepthStencilState = &depthStencil;
    pipeInfo.pColorBlendState = &colorBlending;
    pipeInfo.layout = pipelineLayout;
    pipeInfo.renderPass = renderPass;
    pipeInfo.subpass = 0;

    if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &pipeInfo,
                                  nullptr, &pipeline) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create PBR graphics pipeline");
    }
}

// --------------------------------------------------------------------------
// Helper: detect if scene source is a glTF file
// --------------------------------------------------------------------------

static bool isGltfFile(const std::string& source) {
    if (source.size() > 4 && source.substr(source.size()-4) == ".glb") return true;
    if (source.size() > 5 && source.substr(source.size()-5) == ".gltf") return true;
    return false;
}

// --------------------------------------------------------------------------
// Three-phase initialization
// --------------------------------------------------------------------------

void RasterRenderer::init(VulkanContext& ctx, const std::string& sceneSource,
                          const std::string& pipelineBase, const std::string& renderPath,
                          uint32_t width, uint32_t height) {
    m_sceneSource = sceneSource;
    m_renderPath = renderPath;
    m_pipelineBase = pipelineBase;
    renderWidth = width;
    renderHeight = height;

    // Determine if depth buffer is needed based on scene
    m_needsDepth = (sceneSource == "sphere" || isGltfFile(sceneSource));

    // Phase 1: Upload scene GPU resources
    uploadScene(ctx, sceneSource);

    // Create offscreen target (depends on depth requirement)
    createOffscreenTarget(ctx);
    createRenderPass(ctx);
    createFramebuffer(ctx);

    // Phase 2: Create pipeline from shader files
    createPipeline(ctx, pipelineBase, renderPath);

    // Phase 3: Bind scene resources to pipeline descriptor sets
    bindSceneToPipeline(ctx);
}

// --------------------------------------------------------------------------
// Phase 1: Upload scene GPU resources
// --------------------------------------------------------------------------

void RasterRenderer::uploadScene(VulkanContext& ctx, const std::string& sceneSource) {
    if (sceneSource == "sphere") {
        // Generate sphere mesh + uniforms (descriptors created later via reflection)
        setupPBRResources(ctx);
    } else if (sceneSource == "triangle") {
        // Generate triangle mesh
        std::vector<TriangleVertex> triVerts;
        Scene::generateTriangle(triVerts);

        VkDeviceSize bufSize = triVerts.size() * sizeof(TriangleVertex);
        VkBufferCreateInfo bufInfo = {};
        bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufInfo.size = bufSize;
        bufInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;

        VmaAllocationCreateInfo allocInfo = {};
        allocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;

        vmaCreateBuffer(ctx.allocator, &bufInfo, &allocInfo,
                        &triangleVB, &triangleVBAllocation, nullptr);

        void* mapped;
        vmaMapMemory(ctx.allocator, triangleVBAllocation, &mapped);
        memcpy(mapped, triVerts.data(), bufSize);
        vmaUnmapMemory(ctx.allocator, triangleVBAllocation);
    } else if (sceneSource == "fullscreen") {
        // No mesh, no depth buffer for fullscreen scenes
    } else if (isGltfFile(sceneSource)) {
        // Load glTF scene and upload mesh + textures
        std::cout << "[info] Loading glTF scene: " << sceneSource << std::endl;
        m_gltfScene = loadGltf(sceneSource);
        m_hasGltfScene = true;
        auto drawItems = flattenScene(m_gltfScene);

        if (!drawItems.empty()) {
            // Pack ALL meshes with world transforms into 48-byte stride vertices
            std::vector<float> vdata;
            std::vector<uint32_t> allIndices;
            uint32_t vertexOffset = 0;
            glm::vec3 bboxMin(FLT_MAX), bboxMax(-FLT_MAX);
            int positiveNormals = 0, negativeNormals = 0;

            for (auto& item : drawItems) {
                auto& gmesh = m_gltfScene.meshes[item.meshIndex];
                glm::mat4 world = item.worldTransform;
                glm::mat3 upper3x3 = glm::mat3(world);
                glm::mat3 normalMat = glm::transpose(glm::inverse(upper3x3));

                for (size_t i = 0; i < gmesh.vertices.size(); i++) {
                    auto& gv = gmesh.vertices[i];
                    // Transform position
                    glm::vec4 pos4 = world * glm::vec4(gv.position, 1.0f);
                    glm::vec3 pos(pos4);
                    // Transform normal
                    glm::vec3 norm = glm::normalize(normalMat * gv.normal);
                    // Track bounds
                    bboxMin = glm::min(bboxMin, pos);
                    bboxMax = glm::max(bboxMax, pos);

                    vdata.push_back(pos.x);
                    vdata.push_back(pos.y);
                    vdata.push_back(pos.z);
                    vdata.push_back(norm.x);
                    vdata.push_back(norm.y);
                    vdata.push_back(norm.z);
                    vdata.push_back(gv.uv.x);
                    vdata.push_back(gv.uv.y);
                    if (gmesh.hasTangents) {
                        glm::vec3 tang = glm::normalize(upper3x3 * glm::vec3(gv.tangent));
                        vdata.push_back(tang.x);
                        vdata.push_back(tang.y);
                        vdata.push_back(tang.z);
                        vdata.push_back(gv.tangent.w);
                    } else {
                        vdata.push_back(1.0f);
                        vdata.push_back(0.0f);
                        vdata.push_back(0.0f);
                        vdata.push_back(1.0f);
                    }
                }

                // Add indices with offset
                for (auto idx : gmesh.indices) {
                    allIndices.push_back(idx + vertexOffset);
                }
                vertexOffset += static_cast<uint32_t>(gmesh.vertices.size());
            }

            // Compute auto-camera from scene bounds using node transform
            m_sceneBboxMin = bboxMin;
            m_sceneBboxMax = bboxMax;
            m_hasSceneBounds = true;
            {
                glm::vec3 center = (bboxMin + bboxMax) * 0.5f;
                glm::vec3 extent = bboxMax - bboxMin;

                // Determine camera direction from first node's transform
                // (matches Python engine and Rust PersistentRenderer)
                glm::vec3 camDir(0.0f, 0.0f, 1.0f);   // default: +Z
                glm::vec3 camUp(0.0f, 1.0f, 0.0f);     // default: +Y

                if (!drawItems.empty()) {
                    glm::mat4 world = drawItems[0].worldTransform;
                    glm::mat3 upper3x3(world);
                    // Normalize rows to get rotation only
                    glm::vec3 row0 = glm::vec3(upper3x3[0]);
                    glm::vec3 row1 = glm::vec3(upper3x3[1]);
                    glm::vec3 row2 = glm::vec3(upper3x3[2]);
                    float n0 = glm::length(row0), n1 = glm::length(row1), n2 = glm::length(row2);
                    if (n0 > 1e-8f) upper3x3[0] /= n0;
                    if (n1 > 1e-8f) upper3x3[1] /= n1;
                    if (n2 > 1e-8f) upper3x3[2] /= n2;

                    // Blender front (-Y local) and up (+Z local) in world space
                    // Row-vector convention: v' = v @ R^T = transpose(R) * v
                    glm::mat3 Rt = glm::transpose(upper3x3);
                    glm::vec3 front = Rt * glm::vec3(0.0f, -1.0f, 0.0f);
                    glm::vec3 upW = Rt * glm::vec3(0.0f, 0.0f, 1.0f);

                    float fl = glm::length(front);
                    float ul = glm::length(upW);
                    if (fl > 0.5f && ul > 0.5f) {
                        glm::vec3 candidateDir = front / fl;
                        glm::vec3 candidateUp = upW / ul;
                        if (std::abs(candidateDir.y) < 0.9f) {
                            camDir = candidateDir;
                            camUp = candidateUp;
                            if (std::abs(glm::dot(camDir, camUp)) > 0.9f) {
                                camUp = glm::vec3(0.0f, 1.0f, 0.0f);
                            }
                        }
                    }
                }

                // Compute perpendicular extent for distance calculation
                glm::vec3 viewRight = glm::cross(-camDir, camUp);
                float vrLen = glm::length(viewRight);
                if (vrLen < 1e-6f) {
                    camUp = glm::vec3(0.0f, 0.0f, 1.0f);
                    viewRight = glm::cross(-camDir, camUp);
                    vrLen = glm::length(viewRight);
                }
                viewRight /= vrLen;
                glm::vec3 viewUp = glm::normalize(glm::cross(viewRight, -camDir));

                float projRight = std::abs(viewRight.x) * extent.x + std::abs(viewRight.y) * extent.y + std::abs(viewRight.z) * extent.z;
                float projUp = std::abs(viewUp.x) * extent.x + std::abs(viewUp.y) * extent.y + std::abs(viewUp.z) * extent.z;
                float maxPerpExtent = glm::max(projRight, projUp);

                float fovY = glm::radians(45.0f);
                float distance = (maxPerpExtent / 2.0f) / std::tan(fovY / 2.0f) * 1.1f;

                float elevRad = glm::radians(5.0f);
                glm::vec3 eye = center + distance * camDir + distance * std::sin(elevRad) * viewUp;

                m_autoEye = eye;
                m_autoTarget = center;
                m_autoUp = camUp;
                m_autoFar = glm::max(distance * 3.0f, 100.0f);
                std::cout << "[info] Auto-camera: eye=(" << eye.x << "," << eye.y << "," << eye.z
                          << ") target=(" << center.x << "," << center.y << "," << center.z << ")" << std::endl;
            }

            // Upload raw vertex data directly (bypasses Scene::uploadMesh which uses 32-byte Vertex)
            VkDeviceSize vbufSize = vdata.size() * sizeof(float);
            VkBufferCreateInfo vbufInfo = {};
            vbufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            vbufInfo.size = vbufSize;
            vbufInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                             VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                             VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;

            VmaAllocationCreateInfo vbufAllocInfo = {};
            vbufAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

            vmaCreateBuffer(ctx.allocator, &vbufInfo, &vbufAllocInfo,
                            &mesh.vertexBuffer, &mesh.vertexAllocation, nullptr);

            // Staging buffer for vertex data
            VkBuffer vStagingBuffer;
            VmaAllocation vStagingAlloc;
            VkBufferCreateInfo vStagingInfo = {};
            vStagingInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            vStagingInfo.size = vbufSize;
            vStagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            VmaAllocationCreateInfo vStagingAllocInfo = {};
            vStagingAllocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;
            vmaCreateBuffer(ctx.allocator, &vStagingInfo, &vStagingAllocInfo,
                            &vStagingBuffer, &vStagingAlloc, nullptr);

            void* vMapped;
            vmaMapMemory(ctx.allocator, vStagingAlloc, &vMapped);
            memcpy(vMapped, vdata.data(), vbufSize);
            vmaUnmapMemory(ctx.allocator, vStagingAlloc);

            // Upload index data
            VkDeviceSize ibufSize = allIndices.size() * sizeof(uint32_t);
            VkBufferCreateInfo ibufInfo = {};
            ibufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            ibufInfo.size = ibufSize;
            ibufInfo.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                             VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                             VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;

            VmaAllocationCreateInfo ibufAllocInfo = {};
            ibufAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

            vmaCreateBuffer(ctx.allocator, &ibufInfo, &ibufAllocInfo,
                            &mesh.indexBuffer, &mesh.indexAllocation, nullptr);

            VkBuffer iStagingBuffer;
            VmaAllocation iStagingAlloc;
            VkBufferCreateInfo iStagingInfo = {};
            iStagingInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            iStagingInfo.size = ibufSize;
            iStagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            VmaAllocationCreateInfo iStagingAllocInfo = {};
            iStagingAllocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;
            vmaCreateBuffer(ctx.allocator, &iStagingInfo, &iStagingAllocInfo,
                            &iStagingBuffer, &iStagingAlloc, nullptr);

            void* iMapped;
            vmaMapMemory(ctx.allocator, iStagingAlloc, &iMapped);
            memcpy(iMapped, allIndices.data(), ibufSize);
            vmaUnmapMemory(ctx.allocator, iStagingAlloc);

            // Copy staging -> device
            VkCommandBuffer copyCmd = ctx.beginSingleTimeCommands();
            VkBufferCopy vCopy = {};
            vCopy.size = vbufSize;
            vkCmdCopyBuffer(copyCmd, vStagingBuffer, mesh.vertexBuffer, 1, &vCopy);
            VkBufferCopy iCopy = {};
            iCopy.size = ibufSize;
            vkCmdCopyBuffer(copyCmd, iStagingBuffer, mesh.indexBuffer, 1, &iCopy);
            ctx.endSingleTimeCommands(copyCmd);

            vmaDestroyBuffer(ctx.allocator, vStagingBuffer, vStagingAlloc);
            vmaDestroyBuffer(ctx.allocator, iStagingBuffer, iStagingAlloc);

            mesh.vertexCount = vertexOffset;
            mesh.indexCount = static_cast<uint32_t>(allIndices.size());


            std::cout << "[info] Uploaded glTF mesh with 48-byte stride ("
                      << mesh.vertexCount << " verts, " << mesh.indexCount << " indices)" << std::endl;
        }

        // Setup PBR uniforms and upload glTF textures
        setupPBRResources(ctx);
    } else {
        // Default: treat as sphere scene
        setupPBRResources(ctx);
    }

    std::cout << "[info] Scene uploaded: " << sceneSource << std::endl;
}

// --------------------------------------------------------------------------
// Phase 2: Create pipeline from shader files
// --------------------------------------------------------------------------

void RasterRenderer::createPipeline(VulkanContext& ctx, const std::string& pipelineBase,
                                     const std::string& renderPath) {
    namespace fs = std::filesystem;
    std::string fragPath = pipelineBase + ".frag.spv";

    if (renderPath == "fullscreen") {
        // Fullscreen: load frag.spv only, use built-in vertex shader
        const auto& fullscreenSpv = SpvLoader::getFullscreenVertSPIRV();
        vertModule = SpvLoader::createShaderModule(ctx.device, fullscreenSpv);
        auto fragCode = SpvLoader::loadSPIRV(fragPath);
        fragModule = SpvLoader::createShaderModule(ctx.device, fragCode);
        createPipelineFullscreen(ctx);
    } else if (renderPath == "raster") {
        // Raster: load vert.spv + frag.spv
        std::string vertPath = pipelineBase + ".vert.spv";
        auto vertCode = SpvLoader::loadSPIRV(vertPath);
        vertModule = SpvLoader::createShaderModule(ctx.device, vertCode);
        auto fragCode = SpvLoader::loadSPIRV(fragPath);
        fragModule = SpvLoader::createShaderModule(ctx.device, fragCode);

        // Parse reflection JSON for descriptor set layouts
        std::string vertJsonPath = pipelineBase + ".vert.json";
        std::string fragJsonPath = pipelineBase + ".frag.json";

        if (fs::exists(vertJsonPath) && fs::exists(fragJsonPath)) {
            std::cout << "[info] Loading reflection JSON: " << vertJsonPath << std::endl;
            vertReflection = parseReflectionJson(vertJsonPath);
            std::cout << "[info] Loading reflection JSON: " << fragJsonPath << std::endl;
            fragReflection = parseReflectionJson(fragJsonPath);
        }

        // Choose pipeline type based on scene
        if (m_sceneSource == "triangle") {
            createPipelineTriangle(ctx);
        } else {
            // Setup reflection-driven descriptors before creating pipeline
            // (pipeline layout needs descriptor set layouts)
            setupReflectedDescriptors(ctx);
            createPipelinePBR(ctx);
        }
    } else {
        throw std::runtime_error("Unsupported render path for raster renderer: " + renderPath);
    }

    std::cout << "[info] Pipeline created: " << pipelineBase << " (path=" << renderPath << ")" << std::endl;
}

// --------------------------------------------------------------------------
// Phase 3: Bind scene resources to pipeline descriptor sets
// --------------------------------------------------------------------------

void RasterRenderer::bindSceneToPipeline(VulkanContext& ctx) {
    // Build push constant data from reflection (now that reflection JSON is loaded)
    glm::vec3 lightDir = glm::normalize(glm::vec3(1.0f, 0.8f, 0.6f));
    auto buildPushData = [&](const ReflectionData& refl) {
        for (auto& pc : refl.push_constants) {
            if (pc.size <= 0) continue;
            pushConstantSize = std::max(pushConstantSize, static_cast<uint32_t>(pc.size));
            pushConstantData.resize(pushConstantSize, 0);
            pushConstantStageFlags |= (refl.stage == "vertex")
                ? VK_SHADER_STAGE_VERTEX_BIT : VK_SHADER_STAGE_FRAGMENT_BIT;
            for (auto& f : pc.fields) {
                if (f.name == "light_dir" && static_cast<uint32_t>(f.offset) + 12 <= pushConstantSize) {
                    memcpy(pushConstantData.data() + f.offset, &lightDir, sizeof(glm::vec3));
                } else if (f.name == "view_pos" && static_cast<uint32_t>(f.offset) + 12 <= pushConstantSize) {
                    glm::vec3 eye = m_hasSceneBounds ? m_autoEye : Camera::DEFAULT_EYE;
                    memcpy(pushConstantData.data() + f.offset, &eye, sizeof(glm::vec3));
                }
            }
        }
    };
    buildPushData(vertReflection);
    buildPushData(fragReflection);

    std::cout << "[info] Scene bound to pipeline" << std::endl;
}

// --------------------------------------------------------------------------
// Rendering
// --------------------------------------------------------------------------

void RasterRenderer::render(VulkanContext& ctx) {
    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();

    // Set clear values
    std::vector<VkClearValue> clearValues;
    VkClearValue colorClear = {};
    if (m_needsDepth) {
        colorClear.color = {{0.05f, 0.05f, 0.08f, 1.0f}};
    } else {
        colorClear.color = {{0.0f, 0.0f, 0.0f, 1.0f}};
    }
    clearValues.push_back(colorClear);

    if (m_needsDepth) {
        VkClearValue depthClear = {};
        depthClear.depthStencil = {1.0f, 0};
        clearValues.push_back(depthClear);
    }

    VkRenderPassBeginInfo rpBeginInfo = {};
    rpBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpBeginInfo.renderPass = renderPass;
    rpBeginInfo.framebuffer = framebuffer;
    rpBeginInfo.renderArea.offset = {0, 0};
    rpBeginInfo.renderArea.extent = {renderWidth, renderHeight};
    rpBeginInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
    rpBeginInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(cmd, &rpBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

    // Push constants (if any)
    if (pushConstantSize > 0 && pushConstantStageFlags != 0) {
        vkCmdPushConstants(cmd, pipelineLayout, pushConstantStageFlags,
                           0, pushConstantSize, pushConstantData.data());
    }

    if (m_sceneSource == "triangle") {
        // Triangle scene: bind triangle vertex buffer and draw
        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &triangleVB, &offset);
        vkCmdDraw(cmd, 3, 1, 0, 0);
    } else if (m_sceneSource == "fullscreen" || m_renderPath == "fullscreen") {
        // Fullscreen scene: draw fullscreen triangle (no vertex buffer)
        vkCmdDraw(cmd, 3, 1, 0, 0);
    } else {
        // Sphere, glTF, or default PBR scene: bind reflection-driven descriptors
        // Bind descriptor sets in order
        int maxSet = -1;
        for (auto& [idx, _] : reflectedDescSets) {
            maxSet = std::max(maxSet, idx);
        }
        for (int i = 0; i <= maxSet; i++) {
            auto it = reflectedDescSets.find(i);
            if (it != reflectedDescSets.end()) {
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                        pipelineLayout, static_cast<uint32_t>(i),
                                        1, &it->second, 0, nullptr);
            }
        }

        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &mesh.vertexBuffer, &offset);
        vkCmdBindIndexBuffer(cmd, mesh.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, mesh.indexCount, 1, 0, 0, 0);
    }

    vkCmdEndRenderPass(cmd);
    ctx.endSingleTimeCommands(cmd);
}

// --------------------------------------------------------------------------
// Interactive mode: render to swapchain
// --------------------------------------------------------------------------

void RasterRenderer::renderToSwapchain(VulkanContext& ctx,
                                       VkImage swapImage, VkImageView swapView,
                                       VkFormat swapFormat, VkExtent2D extent,
                                       VkSemaphore waitSem, VkSemaphore signalSem,
                                       VkFence fence) {
    // Record all commands (offscreen render + blit to swapchain) into a single
    // command buffer and submit with proper semaphore/fence synchronization.

    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = ctx.commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(ctx.device, &allocInfo, &cmd);

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    // --- Offscreen render pass ---
    std::vector<VkClearValue> clearValues;
    VkClearValue colorClear = {};
    if (m_needsDepth) {
        colorClear.color = {{0.05f, 0.05f, 0.08f, 1.0f}};
    } else {
        colorClear.color = {{0.0f, 0.0f, 0.0f, 1.0f}};
    }
    clearValues.push_back(colorClear);
    if (m_needsDepth) {
        VkClearValue depthClear = {};
        depthClear.depthStencil = {1.0f, 0};
        clearValues.push_back(depthClear);
    }

    VkRenderPassBeginInfo rpBeginInfo = {};
    rpBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpBeginInfo.renderPass = renderPass;
    rpBeginInfo.framebuffer = framebuffer;
    rpBeginInfo.renderArea.offset = {0, 0};
    rpBeginInfo.renderArea.extent = {renderWidth, renderHeight};
    rpBeginInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
    rpBeginInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(cmd, &rpBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

    if (pushConstantSize > 0 && pushConstantStageFlags != 0) {
        vkCmdPushConstants(cmd, pipelineLayout, pushConstantStageFlags,
                           0, pushConstantSize, pushConstantData.data());
    }

    if (m_sceneSource == "triangle") {
        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &triangleVB, &offset);
        vkCmdDraw(cmd, 3, 1, 0, 0);
    } else if (m_sceneSource == "fullscreen" || m_renderPath == "fullscreen") {
        vkCmdDraw(cmd, 3, 1, 0, 0);
    } else {
        int maxSet = -1;
        for (auto& [idx, _] : reflectedDescSets) {
            maxSet = std::max(maxSet, idx);
        }
        for (int i = 0; i <= maxSet; i++) {
            auto it = reflectedDescSets.find(i);
            if (it != reflectedDescSets.end()) {
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                        pipelineLayout, static_cast<uint32_t>(i),
                                        1, &it->second, 0, nullptr);
            }
        }
        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &mesh.vertexBuffer, &offset);
        vkCmdBindIndexBuffer(cmd, mesh.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, mesh.indexCount, 1, 0, 0, 0);
    }

    vkCmdEndRenderPass(cmd);

    // --- Blit offscreen -> swapchain ---

    // Transition swapchain image: UNDEFINED -> TRANSFER_DST
    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = swapImage;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    VkImageBlit blitRegion = {};
    blitRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blitRegion.srcSubresource.layerCount = 1;
    blitRegion.srcOffsets[1] = {static_cast<int32_t>(renderWidth),
                                static_cast<int32_t>(renderHeight), 1};
    blitRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blitRegion.dstSubresource.layerCount = 1;
    blitRegion.dstOffsets[1] = {static_cast<int32_t>(extent.width),
                                static_cast<int32_t>(extent.height), 1};

    vkCmdBlitImage(cmd,
        offscreenImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
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

    vkEndCommandBuffer(cmd);

    // Submit with proper semaphore/fence synchronization
    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = &waitSem;
    submitInfo.pWaitDstStageMask = &waitStage;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &signalSem;

    vkQueueSubmit(ctx.graphicsQueue, 1, &submitInfo, fence);
}

// --------------------------------------------------------------------------
// --------------------------------------------------------------------------
// Orbit camera: update MVP + Light uniform buffers per frame
// --------------------------------------------------------------------------

void RasterRenderer::updateCamera(VulkanContext& ctx, const glm::vec3& eye,
                                  const glm::vec3& target, const glm::vec3& up,
                                  float fovY, float aspect, float nearPlane, float farPlane) {
    if (!mvpBuffer || !lightBuffer) return;

    // Update MVP
    struct MVPData { glm::mat4 model, view, projection; };
    MVPData mvp;
    mvp.model = glm::mat4(1.0f);
    mvp.view = Camera::lookAt(eye, target, up);
    mvp.projection = Camera::perspective(fovY, aspect, nearPlane, farPlane);

    void* mapped;
    vmaMapMemory(ctx.allocator, mvpAllocation, &mapped);
    memcpy(mapped, &mvp, sizeof(MVPData));
    vmaUnmapMemory(ctx.allocator, mvpAllocation);

    // Update Light uniform (preserve light_dir, update view_pos)
    struct LightData {
        glm::vec3 lightDir; float _pad0;
        glm::vec3 viewPos; float _pad1;
    };
    LightData light;
    vmaMapMemory(ctx.allocator, lightAllocation, &mapped);
    memcpy(&light, mapped, sizeof(LightData));
    light.viewPos = eye;
    memcpy(mapped, &light, sizeof(LightData));
    vmaUnmapMemory(ctx.allocator, lightAllocation);
}


// Cleanup
// --------------------------------------------------------------------------

void RasterRenderer::cleanup(VulkanContext& ctx) {
    vkDeviceWaitIdle(ctx.device);

    if (pipeline) vkDestroyPipeline(ctx.device, pipeline, nullptr);
    if (pipelineLayout) vkDestroyPipelineLayout(ctx.device, pipelineLayout, nullptr);
    if (framebuffer) vkDestroyFramebuffer(ctx.device, framebuffer, nullptr);
    if (renderPass) vkDestroyRenderPass(ctx.device, renderPass, nullptr);

    if (vertModule) vkDestroyShaderModule(ctx.device, vertModule, nullptr);
    if (fragModule) vkDestroyShaderModule(ctx.device, fragModule, nullptr);

    if (offscreenImageView) vkDestroyImageView(ctx.device, offscreenImageView, nullptr);
    if (offscreenImage) vmaDestroyImage(ctx.allocator, offscreenImage, offscreenAllocation);

    if (depthImageView) vkDestroyImageView(ctx.device, depthImageView, nullptr);
    if (depthImage) vmaDestroyImage(ctx.allocator, depthImage, depthAllocation);

    if (triangleVB) vmaDestroyBuffer(ctx.allocator, triangleVB, triangleVBAllocation);

    if (mvpBuffer) vmaDestroyBuffer(ctx.allocator, mvpBuffer, mvpAllocation);
    if (lightBuffer) vmaDestroyBuffer(ctx.allocator, lightBuffer, lightAllocation);

    if (descriptorPool) vkDestroyDescriptorPool(ctx.device, descriptorPool, nullptr);
    for (auto& [_, layout] : reflectedSetLayouts) {
        vkDestroyDescriptorSetLayout(ctx.device, layout, nullptr);
    }
    reflectedSetLayouts.clear();
    reflectedDescSets.clear();

    Scene::destroyMesh(ctx.allocator, mesh);

    // Destroy named textures
    // Track already-destroyed textures to avoid double-free when multiple names
    // point to the same GPUTexture (e.g. albedo_tex and base_color_tex for procedural)
    std::vector<VkImage> destroyed;
    for (auto& [name, tex] : namedTextures) {
        if (tex.image != VK_NULL_HANDLE) {
            bool alreadyDestroyed = false;
            for (auto img : destroyed) {
                if (img == tex.image) { alreadyDestroyed = true; break; }
            }
            if (!alreadyDestroyed) {
                destroyed.push_back(tex.image);
                Scene::destroyTexture(ctx.allocator, ctx.device, tex);
            }
        }
    }
    namedTextures.clear();

    // Destroy IBL textures
    for (auto& [name, tex] : iblTextures) {
        if (tex.image != VK_NULL_HANDLE) {
            Scene::destroyTexture(ctx.allocator, ctx.device, tex);
        }
    }
    iblTextures.clear();

    Scene::destroyTexture(ctx.allocator, ctx.device, defaultWhiteTexture);
    Scene::destroyTexture(ctx.allocator, ctx.device, defaultBlackTexture);
    Scene::destroyTexture(ctx.allocator, ctx.device, defaultNormalTexture);
}
