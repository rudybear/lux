#include "deferred_renderer.h"
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
#include <filesystem>

namespace fs = std::filesystem;

// --------------------------------------------------------------------------
// G-buffer image creation (3 color + 1 depth)
// --------------------------------------------------------------------------

void DeferredRenderer::createGBufferImages(VulkanContext& ctx) {
    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    // RT0: RGBA16F (base color + metallic, full precision)
    {
        VkImageCreateInfo imageInfo = {};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        imageInfo.extent = {m_renderWidth, m_renderHeight, 1};
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        vmaCreateImage(ctx.allocator, &imageInfo, &allocInfo,
                       &m_gbufRT0Image, &m_gbufRT0Alloc, nullptr);

        VkImageViewCreateInfo viewInfo = {};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = m_gbufRT0Image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;
        vkCreateImageView(ctx.device, &viewInfo, nullptr, &m_gbufRT0View);
    }

    // RT1: RGBA16F (oct normals + roughness)
    {
        VkImageCreateInfo imageInfo = {};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        imageInfo.extent = {m_renderWidth, m_renderHeight, 1};
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        vmaCreateImage(ctx.allocator, &imageInfo, &allocInfo,
                       &m_gbufRT1Image, &m_gbufRT1Alloc, nullptr);

        VkImageViewCreateInfo viewInfo = {};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = m_gbufRT1Image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;
        vkCreateImageView(ctx.device, &viewInfo, nullptr, &m_gbufRT1View);
    }

    // RT2: RGBA16F (metallic, roughness, emissive, etc.)
    {
        VkImageCreateInfo imageInfo = {};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        imageInfo.extent = {m_renderWidth, m_renderHeight, 1};
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        vmaCreateImage(ctx.allocator, &imageInfo, &allocInfo,
                       &m_gbufRT2Image, &m_gbufRT2Alloc, nullptr);

        VkImageViewCreateInfo viewInfo = {};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = m_gbufRT2Image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;
        vkCreateImageView(ctx.device, &viewInfo, nullptr, &m_gbufRT2View);
    }

    // Depth: D32_SFLOAT
    {
        VkImageCreateInfo imageInfo = {};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.format = VK_FORMAT_D32_SFLOAT;
        imageInfo.extent = {m_renderWidth, m_renderHeight, 1};
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        vmaCreateImage(ctx.allocator, &imageInfo, &allocInfo,
                       &m_gbufDepthImage, &m_gbufDepthAlloc, nullptr);

        VkImageViewCreateInfo viewInfo = {};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = m_gbufDepthImage;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = VK_FORMAT_D32_SFLOAT;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;
        vkCreateImageView(ctx.device, &viewInfo, nullptr, &m_gbufDepthView);
    }

    // Create G-buffer sampler for lighting pass
    {
        VkSamplerCreateInfo samplerInfo = {};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = 0.0f;
        vkCreateSampler(ctx.device, &samplerInfo, nullptr, &m_gbufSampler);
    }
}

// --------------------------------------------------------------------------
// G-buffer render pass (3 color + 1 depth attachment)
// --------------------------------------------------------------------------

void DeferredRenderer::createGBufferRenderPass(VulkanContext& ctx) {
    std::array<VkAttachmentDescription, 4> attachments = {};

    // RT0: RGBA16F (base color + metallic)
    attachments[0].format = VK_FORMAT_R16G16B16A16_SFLOAT;
    attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[0].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    // RT1: RGBA16F (octahedron normal + roughness)
    attachments[1].format = VK_FORMAT_R16G16B16A16_SFLOAT;
    attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[1].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    // RT2: RGBA16F
    attachments[2].format = VK_FORMAT_R16G16B16A16_SFLOAT;
    attachments[2].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[2].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[2].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[2].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[2].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[2].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[2].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    // Depth: D32_SFLOAT
    attachments[3].format = VK_FORMAT_D32_SFLOAT;
    attachments[3].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[3].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[3].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[3].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[3].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[3].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[3].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    std::array<VkAttachmentReference, 3> colorRefs = {};
    colorRefs[0] = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    colorRefs[1] = {1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    colorRefs[2] = {2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

    VkAttachmentReference depthRef = {};
    depthRef.attachment = 3;
    depthRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = static_cast<uint32_t>(colorRefs.size());
    subpass.pColorAttachments = colorRefs.data();
    subpass.pDepthStencilAttachment = &depthRef;

    VkSubpassDependency dependencies[2] = {};

    // EXTERNAL -> 0
    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0;
    dependencies[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                   VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                   VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependencies[0].srcAccessMask = 0;
    dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    // 0 -> EXTERNAL: G-buffer outputs must be readable by lighting pass fragment shader
    dependencies[1].srcSubpass = 0;
    dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                   VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    VkRenderPassCreateInfo rpInfo = {};
    rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    rpInfo.pAttachments = attachments.data();
    rpInfo.subpassCount = 1;
    rpInfo.pSubpasses = &subpass;
    rpInfo.dependencyCount = 2;
    rpInfo.pDependencies = dependencies;

    if (vkCreateRenderPass(ctx.device, &rpInfo, nullptr, &m_gbufRenderPass) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create G-buffer render pass");
    }
}

// --------------------------------------------------------------------------
// G-buffer framebuffer
// --------------------------------------------------------------------------

void DeferredRenderer::createGBufferFramebuffer(VulkanContext& ctx) {
    std::array<VkImageView, 4> attachments = {
        m_gbufRT0View,
        m_gbufRT1View,
        m_gbufRT2View,
        m_gbufDepthView
    };

    VkFramebufferCreateInfo fbInfo = {};
    fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fbInfo.renderPass = m_gbufRenderPass;
    fbInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    fbInfo.pAttachments = attachments.data();
    fbInfo.width = m_renderWidth;
    fbInfo.height = m_renderHeight;
    fbInfo.layers = 1;

    if (vkCreateFramebuffer(ctx.device, &fbInfo, nullptr, &m_gbufFramebuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create G-buffer framebuffer");
    }
}

// --------------------------------------------------------------------------
// G-buffer pipeline
// --------------------------------------------------------------------------

void DeferredRenderer::createGBufferPipeline(VulkanContext& ctx) {
    // Pipeline layout from reflection data
    m_gbufPipelineLayout = createReflectedPipelineLayoutMultiStage(
        ctx.device, m_gbufSetLayouts,
        {m_gbufVertRefl, m_gbufFragRefl});

    // Shader stages
    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = m_gbufVertModule;
    stages[0].pName = "main";

    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = m_gbufFragModule;
    stages[1].pName = "main";

    // Vertex input from reflection
    int bufferStride = (m_scene && m_scene->hasGltfScene()) ? 48 : 0;
    auto reflectedInput = createReflectedVertexInput(m_gbufVertRefl, bufferStride);

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
    viewport.width = static_cast<float>(m_renderWidth);
    viewport.height = static_cast<float>(m_renderHeight);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor = {};
    scissor.offset = {0, 0};
    scissor.extent = {m_renderWidth, m_renderHeight};

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

    // 3 color blend attachments (all blend disabled — G-buffer writes raw values)
    std::array<VkPipelineColorBlendAttachmentState, 3> colorBlendAttachments = {};
    for (auto& att : colorBlendAttachments) {
        att.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                             VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        att.blendEnable = VK_FALSE;
    }

    VkPipelineColorBlendStateCreateInfo colorBlending = {};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = static_cast<uint32_t>(colorBlendAttachments.size());
    colorBlending.pAttachments = colorBlendAttachments.data();

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
    pipeInfo.layout = m_gbufPipelineLayout;
    pipeInfo.renderPass = m_gbufRenderPass;
    pipeInfo.subpass = 0;

    if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &pipeInfo,
                                  nullptr, &m_gbufPipeline) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create G-buffer graphics pipeline");
    }
}

// --------------------------------------------------------------------------
// G-buffer descriptor setup (reflection-driven, same pattern as RasterRenderer)
// --------------------------------------------------------------------------

void DeferredRenderer::setupGBufferDescriptors(VulkanContext& ctx) {
    std::vector<ReflectionData> stages = {m_gbufVertRefl, m_gbufFragRefl};
    m_gbufSetLayouts = createDescriptorSetLayoutsMultiStage(ctx.device, stages);
    auto mergedBindings = getMergedBindings(stages);

    // Identify set indices
    int vertSetIdx = 0;
    int fragSetIdx = 1;
    for (auto& b : mergedBindings) {
        if (b.name == "MVP") vertSetIdx = b.set;
        if (b.name == "Material") fragSetIdx = b.set;
    }

    size_t numMaterials = 1;
    if (m_scene && m_scene->hasGltfScene()) {
        numMaterials = std::max(size_t(1), m_scene->getGltfScene().materials.size());
    }

    // Count pool sizes
    std::unordered_map<VkDescriptorType, uint32_t> typeCounts;
    for (auto& b : mergedBindings) {
        VkDescriptorType vkType;
        if (b.type == "uniform_buffer") vkType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        else if (b.type == "sampler") vkType = VK_DESCRIPTOR_TYPE_SAMPLER;
        else if (b.type == "sampled_image") vkType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        else if (b.type == "sampled_cube_image") vkType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        else if (b.type == "storage_image") vkType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        else if (b.type == "storage_buffer") vkType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        else vkType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;

        uint32_t multiplier = (b.set == fragSetIdx) ? static_cast<uint32_t>(numMaterials) : 1;
        typeCounts[vkType] += multiplier;
    }

    std::vector<VkDescriptorPoolSize> poolSizes;
    for (auto& [type, count] : typeCounts) {
        poolSizes.push_back({type, count});
    }

    uint32_t maxSets = 1 + static_cast<uint32_t>(numMaterials);

    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = maxSets;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr, &m_gbufDescPool);

    // Allocate vertex descriptor set (set 0)
    {
        auto layoutIt = m_gbufSetLayouts.find(vertSetIdx);
        if (layoutIt != m_gbufSetLayouts.end()) {
            VkDescriptorSetAllocateInfo allocSetInfo = {};
            allocSetInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            allocSetInfo.descriptorPool = m_gbufDescPool;
            allocSetInfo.descriptorSetCount = 1;
            allocSetInfo.pSetLayouts = &layoutIt->second;
            vkAllocateDescriptorSets(ctx.device, &allocSetInfo, &m_gbufVertDescSet);
        }
    }

    // Write vertex set (set 0): MVP + Light/SceneLight
    {
        struct DescWriteInfo { VkDescriptorBufferInfo bufferInfo; VkDescriptorImageInfo imageInfo; };
        std::vector<DescWriteInfo> writeInfos;
        writeInfos.reserve(mergedBindings.size());
        std::vector<VkWriteDescriptorSet> writes;
        writes.reserve(mergedBindings.size());

        for (auto& b : mergedBindings) {
            if (b.set != vertSetIdx) continue;
            DescWriteInfo info = {};
            VkWriteDescriptorSet w = {};
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = m_gbufVertDescSet;
            w.dstBinding = static_cast<uint32_t>(b.binding);
            w.descriptorCount = 1;

            if (b.type == "uniform_buffer") {
                w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                if (b.name == "MVP") {
                    info.bufferInfo = {m_mvpBuffer, 0, static_cast<VkDeviceSize>(b.size > 0 ? b.size : 192)};
                } else if (b.name == "Light") {
                    info.bufferInfo = {m_lightBuffer, 0, static_cast<VkDeviceSize>(b.size > 0 ? b.size : 32)};
                } else if (b.name == "SceneLight" && m_hasMultiLight && m_sceneLightUBO != VK_NULL_HANDLE) {
                    info.bufferInfo = {m_sceneLightUBO, 0, static_cast<VkDeviceSize>(b.size > 0 ? b.size : 32)};
                } else {
                    continue;
                }
                writeInfos.push_back(info);
                w.pBufferInfo = &writeInfos.back().bufferInfo;
            } else if (b.type == "storage_buffer") {
                if (b.name == "lights" && m_hasMultiLight && m_lightsSSBO != VK_NULL_HANDLE) {
                    w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    info.bufferInfo = {m_lightsSSBO, 0, VK_WHOLE_SIZE};
                    writeInfos.push_back(info);
                    w.pBufferInfo = &writeInfos.back().bufferInfo;
                } else {
                    continue;
                }
            } else {
                continue;
            }
            writes.push_back(w);
        }
        if (!writes.empty()) {
            vkUpdateDescriptorSets(ctx.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        }
    }

    // Allocate per-material fragment descriptor sets (set 1)
    auto fragLayoutIt = m_gbufSetLayouts.find(fragSetIdx);
    if (fragLayoutIt != m_gbufSetLayouts.end()) {
        m_gbufPerMaterialDescSets.resize(numMaterials);
        m_gbufPerMaterialBuffers.resize(numMaterials);
        m_gbufPerMaterialAllocations.resize(numMaterials);

        for (size_t mi = 0; mi < numMaterials; mi++) {
            // Create Material UBO for this material
            MaterialUBOData materialData{};
            if (m_scene && m_scene->hasGltfScene() && mi < m_scene->getGltfScene().materials.size()) {
                const auto& mat = m_scene->getGltfScene().materials[mi];
                materialData.baseColorFactor = mat.baseColor;
                materialData.metallicFactor = mat.metallic;
                materialData.roughnessFactor = mat.roughness;
                materialData.emissiveFactor = mat.emissive;
                materialData.emissiveStrength = mat.emissiveStrength;
                materialData.ior = mat.ior;
                materialData.clearcoatFactor = mat.clearcoatFactor;
                materialData.clearcoatRoughnessFactor = mat.clearcoatRoughnessFactor;
                materialData.sheenColorFactor = mat.sheenColorFactor;
                materialData.sheenRoughnessFactor = mat.sheenRoughnessFactor;
                materialData.transmissionFactor = mat.transmissionFactor;
                materialData.baseColorUvSt = glm::vec4(mat.base_color_uv_xform.offset, mat.base_color_uv_xform.scale);
                materialData.normalUvSt = glm::vec4(mat.normal_uv_xform.offset, mat.normal_uv_xform.scale);
                materialData.mrUvSt = glm::vec4(mat.metallic_roughness_uv_xform.offset, mat.metallic_roughness_uv_xform.scale);
                materialData.baseColorUvRot = mat.base_color_uv_xform.rotation;
                materialData.normalUvRot = mat.normal_uv_xform.rotation;
                materialData.mrUvRot = mat.metallic_roughness_uv_xform.rotation;
            }

            VkBufferCreateInfo matBufInfo = {};
            matBufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            matBufInfo.size = sizeof(MaterialUBOData);
            matBufInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
            VmaAllocationCreateInfo matAllocInfo = {};
            matAllocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
            vmaCreateBuffer(ctx.allocator, &matBufInfo, &matAllocInfo,
                            &m_gbufPerMaterialBuffers[mi], &m_gbufPerMaterialAllocations[mi], nullptr);

            void* mapped;
            vmaMapMemory(ctx.allocator, m_gbufPerMaterialAllocations[mi], &mapped);
            memcpy(mapped, &materialData, sizeof(MaterialUBOData));
            vmaUnmapMemory(ctx.allocator, m_gbufPerMaterialAllocations[mi]);

            // Allocate descriptor set
            VkDescriptorSetAllocateInfo allocSetInfo = {};
            allocSetInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            allocSetInfo.descriptorPool = m_gbufDescPool;
            allocSetInfo.descriptorSetCount = 1;
            allocSetInfo.pSetLayouts = &fragLayoutIt->second;
            vkAllocateDescriptorSets(ctx.device, &allocSetInfo, &m_gbufPerMaterialDescSets[mi]);

            // Write descriptors for this material
            struct DescWriteInfo { VkDescriptorBufferInfo bufferInfo; VkDescriptorImageInfo imageInfo; };
            std::vector<DescWriteInfo> writeInfos;
            writeInfos.reserve(mergedBindings.size());
            std::vector<VkWriteDescriptorSet> writes;
            writes.reserve(mergedBindings.size());

            auto& perMatTextures = m_scene->getPerMaterialTextures();

            for (auto& b : mergedBindings) {
                if (b.set != fragSetIdx) continue;
                DescWriteInfo info = {};
                VkWriteDescriptorSet w = {};
                w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w.dstSet = m_gbufPerMaterialDescSets[mi];
                w.dstBinding = static_cast<uint32_t>(b.binding);
                w.descriptorCount = 1;

                if (b.type == "uniform_buffer") {
                    w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                    if (b.name == "Material") {
                        info.bufferInfo = {m_gbufPerMaterialBuffers[mi], 0, sizeof(MaterialUBOData)};
                    } else if (b.name == "Light" && m_lightBuffer != VK_NULL_HANDLE) {
                        info.bufferInfo = {m_lightBuffer, 0, static_cast<VkDeviceSize>(b.size > 0 ? b.size : 32)};
                    } else if (b.name == "SceneLight" && m_hasMultiLight && m_sceneLightUBO != VK_NULL_HANDLE) {
                        info.bufferInfo = {m_sceneLightUBO, 0, static_cast<VkDeviceSize>(b.size > 0 ? b.size : 32)};
                    } else {
                        continue;
                    }
                    writeInfos.push_back(info);
                    w.pBufferInfo = &writeInfos.back().bufferInfo;
                } else if (b.type == "storage_buffer") {
                    if (b.name == "lights" && m_hasMultiLight && m_lightsSSBO != VK_NULL_HANDLE) {
                        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                        info.bufferInfo = {m_lightsSSBO, 0, VK_WHOLE_SIZE};
                        writeInfos.push_back(info);
                        w.pBufferInfo = &writeInfos.back().bufferInfo;
                    } else {
                        continue;
                    }
                } else if (b.type == "sampler") {
                    w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
                    auto& iblTextures = m_scene->getIBLTextures();
                    auto iblIt = iblTextures.find(b.name);
                    if (iblIt != iblTextures.end()) {
                        info.imageInfo = {};
                        info.imageInfo.sampler = iblIt->second.sampler;
                    } else {
                        auto& tex = m_scene->getTextureForBinding(b.name);
                        info.imageInfo = {};
                        info.imageInfo.sampler = tex.sampler;
                    }
                    writeInfos.push_back(info);
                    w.pImageInfo = &writeInfos.back().imageInfo;
                } else if (b.type == "sampled_image" || b.type == "sampled_cube_image") {
                    w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                    auto& iblTextures = m_scene->getIBLTextures();
                    auto iblIt = iblTextures.find(b.name);
                    if (iblIt != iblTextures.end()) {
                        info.imageInfo = {};
                        info.imageInfo.imageView = iblIt->second.imageView;
                        info.imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                    } else {
                        GPUTexture* texPtr = nullptr;
                        if (mi < perMatTextures.size()) {
                            auto it = perMatTextures[mi].find(b.name);
                            if (it != perMatTextures[mi].end()) {
                                texPtr = const_cast<GPUTexture*>(&it->second);
                            }
                        }
                        if (texPtr && texPtr->imageView != VK_NULL_HANDLE) {
                            info.imageInfo = {};
                            info.imageInfo.imageView = texPtr->imageView;
                            info.imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                        } else {
                            auto& tex = m_scene->getTextureForBinding(b.name);
                            info.imageInfo = {};
                            info.imageInfo.imageView = tex.imageView;
                            info.imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                        }
                    }
                    writeInfos.push_back(info);
                    w.pImageInfo = &writeInfos.back().imageInfo;
                } else {
                    continue;
                }
                writes.push_back(w);
            }

            if (!writes.empty()) {
                vkUpdateDescriptorSets(ctx.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
            }
        }
    }

    std::cout << "[info] G-buffer descriptors created: " << numMaterials
              << " material(s), " << mergedBindings.size() << " binding(s) each" << std::endl;
}

// --------------------------------------------------------------------------
// Lighting output image
// --------------------------------------------------------------------------

void DeferredRenderer::createLightingOutputImage(VulkanContext& ctx) {
    VkImageCreateInfo imageInfo = {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    imageInfo.extent = {m_renderWidth, m_renderHeight, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    vmaCreateImage(ctx.allocator, &imageInfo, &allocInfo,
                   &m_lightOutputImage, &m_lightOutputAlloc, nullptr);

    VkImageViewCreateInfo viewInfo = {};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = m_lightOutputImage;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;
    vkCreateImageView(ctx.device, &viewInfo, nullptr, &m_lightOutputView);
}

// --------------------------------------------------------------------------
// Lighting render pass (1 color attachment, no depth)
// --------------------------------------------------------------------------

void DeferredRenderer::createLightingRenderPass(VulkanContext& ctx) {
    VkAttachmentDescription colorAtt = {};
    colorAtt.format = VK_FORMAT_R8G8B8A8_UNORM;
    colorAtt.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAtt.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

    VkAttachmentReference colorRef = {};
    colorRef.attachment = 0;
    colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;

    VkSubpassDependency dependencies[2] = {};

    // EXTERNAL -> 0
    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0;
    dependencies[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    // 0 -> EXTERNAL: ensure color writes are visible to subsequent blit/transfer
    dependencies[1].srcSubpass = 0;
    dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[1].dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
    dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[1].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    VkRenderPassCreateInfo rpInfo = {};
    rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpInfo.attachmentCount = 1;
    rpInfo.pAttachments = &colorAtt;
    rpInfo.subpassCount = 1;
    rpInfo.pSubpasses = &subpass;
    rpInfo.dependencyCount = 2;
    rpInfo.pDependencies = dependencies;

    if (vkCreateRenderPass(ctx.device, &rpInfo, nullptr, &m_lightRenderPass) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create lighting render pass");
    }
}

// --------------------------------------------------------------------------
// Lighting framebuffer
// --------------------------------------------------------------------------

void DeferredRenderer::createLightingFramebuffer(VulkanContext& ctx) {
    VkFramebufferCreateInfo fbInfo = {};
    fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fbInfo.renderPass = m_lightRenderPass;
    fbInfo.attachmentCount = 1;
    fbInfo.pAttachments = &m_lightOutputView;
    fbInfo.width = m_renderWidth;
    fbInfo.height = m_renderHeight;
    fbInfo.layers = 1;

    if (vkCreateFramebuffer(ctx.device, &fbInfo, nullptr, &m_lightFramebuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create lighting framebuffer");
    }
}

// --------------------------------------------------------------------------
// Lighting pipeline (fullscreen triangle, no vertex input)
// --------------------------------------------------------------------------

void DeferredRenderer::createLightingPipeline(VulkanContext& ctx) {
    // Pipeline layout from reflection data
    m_lightPipelineLayout = createReflectedPipelineLayoutMultiStage(
        ctx.device, m_lightSetLayouts,
        {m_lightVertRefl, m_lightFragRefl});

    // Shader stages
    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = m_lightVertModule;
    stages[0].pName = "main";

    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = m_lightFragModule;
    stages[1].pName = "main";

    // No vertex input (fullscreen triangle uses vertex_index)
    VkPipelineVertexInputStateCreateInfo vertexInput = {};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport viewport = {};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(m_renderWidth);
    viewport.height = static_cast<float>(m_renderHeight);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor = {};
    scissor.offset = {0, 0};
    scissor.extent = {m_renderWidth, m_renderHeight};

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

    // 1 color blend attachment (blend disabled)
    VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
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
    pipeInfo.layout = m_lightPipelineLayout;
    pipeInfo.renderPass = m_lightRenderPass;
    pipeInfo.subpass = 0;

    if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &pipeInfo,
                                  nullptr, &m_lightPipeline) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create lighting graphics pipeline");
    }
}

// --------------------------------------------------------------------------
// Lighting descriptor setup
// --------------------------------------------------------------------------

void DeferredRenderer::setupLightingDescriptors(VulkanContext& ctx) {
    std::vector<ReflectionData> stages = {m_lightVertRefl, m_lightFragRefl};
    m_lightSetLayouts = createDescriptorSetLayoutsMultiStage(ctx.device, stages);
    auto mergedBindings = getMergedBindings(stages);

    // Count pool sizes
    std::unordered_map<VkDescriptorType, uint32_t> typeCounts;
    for (auto& b : mergedBindings) {
        VkDescriptorType vkType;
        if (b.type == "uniform_buffer") vkType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        else if (b.type == "sampler") vkType = VK_DESCRIPTOR_TYPE_SAMPLER;
        else if (b.type == "sampled_image") vkType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        else if (b.type == "sampled_cube_image") vkType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        else if (b.type == "combined_image_sampler") vkType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        else if (b.type == "storage_buffer") vkType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        else vkType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        typeCounts[vkType]++;
    }

    std::vector<VkDescriptorPoolSize> poolSizes;
    for (auto& [type, count] : typeCounts) {
        poolSizes.push_back({type, count});
    }

    // Determine how many sets we need
    int maxSetIdx = -1;
    for (auto& [idx, _] : m_lightSetLayouts) { maxSetIdx = std::max(maxSetIdx, idx); }
    uint32_t maxSets = static_cast<uint32_t>(maxSetIdx + 1);
    if (maxSets == 0) maxSets = 1;

    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = maxSets;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr, &m_lightDescPool);

    // Allocate all descriptor sets
    for (int setIdx = 0; setIdx <= maxSetIdx; setIdx++) {
        auto layoutIt = m_lightSetLayouts.find(setIdx);
        if (layoutIt == m_lightSetLayouts.end()) continue;

        VkDescriptorSetAllocateInfo allocSetInfo = {};
        allocSetInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocSetInfo.descriptorPool = m_lightDescPool;
        allocSetInfo.descriptorSetCount = 1;
        allocSetInfo.pSetLayouts = &layoutIt->second;

        VkDescriptorSet ds;
        VkResult result = vkAllocateDescriptorSets(ctx.device, &allocSetInfo, &ds);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate lighting descriptor set " + std::to_string(setIdx));
        }
        m_lightDescSets[setIdx] = ds;
    }

    // Write all descriptor sets
    struct DescWriteInfo { VkDescriptorBufferInfo bufferInfo; VkDescriptorImageInfo imageInfo; };
    std::vector<DescWriteInfo> writeInfos;
    writeInfos.reserve(mergedBindings.size());
    std::vector<VkWriteDescriptorSet> writes;
    writes.reserve(mergedBindings.size());

    // Map G-buffer texture names to their image views and layouts
    // The lighting fragment shader references these as sampler2d bindings
    auto getGBufferView = [&](const std::string& name) -> std::pair<VkImageView, VkImageLayout> {
        if (name == "gbuf_tex0" || name == "gbuffer_albedo" || name == "gbuf_rt0")
            return {m_gbufRT0View, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        if (name == "gbuf_tex1" || name == "gbuffer_normal" || name == "gbuf_rt1")
            return {m_gbufRT1View, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        if (name == "gbuf_tex2" || name == "gbuffer_material" || name == "gbuf_rt2")
            return {m_gbufRT2View, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        if (name == "gbuf_depth" || name == "gbuffer_depth")
            return {m_gbufDepthView, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};
        return {VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
    };

    auto isGBufferBinding = [&](const std::string& name) -> bool {
        return name == "gbuf_tex0" || name == "gbuffer_albedo" || name == "gbuf_rt0" ||
               name == "gbuf_tex1" || name == "gbuffer_normal" || name == "gbuf_rt1" ||
               name == "gbuf_tex2" || name == "gbuffer_material" || name == "gbuf_rt2" ||
               name == "gbuf_depth" || name == "gbuffer_depth";
    };

    for (auto& b : mergedBindings) {
        auto dsIt = m_lightDescSets.find(b.set);
        if (dsIt == m_lightDescSets.end()) continue;

        if (b.type == "uniform_buffer") {
            DescWriteInfo info = {};
            VkWriteDescriptorSet w = {};
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = dsIt->second;
            w.dstBinding = static_cast<uint32_t>(b.binding);
            w.descriptorCount = 1;
            w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;

            if (b.name == "DeferredCamera") {
                info.bufferInfo = {m_deferredCameraBuffer, 0, static_cast<VkDeviceSize>(b.size > 0 ? b.size : 80)};
            } else if (b.name == "MVP") {
                info.bufferInfo = {m_mvpBuffer, 0, static_cast<VkDeviceSize>(b.size > 0 ? b.size : 192)};
            } else if (b.name == "Light") {
                info.bufferInfo = {m_lightBuffer, 0, static_cast<VkDeviceSize>(b.size > 0 ? b.size : 32)};
            } else if (b.name == "SceneLight" && m_hasMultiLight && m_sceneLightUBO != VK_NULL_HANDLE) {
                info.bufferInfo = {m_sceneLightUBO, 0, static_cast<VkDeviceSize>(b.size > 0 ? b.size : 32)};
            } else {
                continue;
            }
            writeInfos.push_back(info);
            w.pBufferInfo = &writeInfos.back().bufferInfo;
            writes.push_back(w);
        } else if (b.type == "storage_buffer") {
            if (b.name == "lights" && m_hasMultiLight && m_lightsSSBO != VK_NULL_HANDLE) {
                DescWriteInfo info = {};
                info.bufferInfo = {m_lightsSSBO, 0, VK_WHOLE_SIZE};
                writeInfos.push_back(info);

                VkWriteDescriptorSet w = {};
                w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w.dstSet = dsIt->second;
                w.dstBinding = static_cast<uint32_t>(b.binding);
                w.descriptorCount = 1;
                w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                w.pBufferInfo = &writeInfos.back().bufferInfo;
                writes.push_back(w);
            }
        } else if (b.type == "sampler") {
            DescWriteInfo info = {};
            info.imageInfo = {};
            if (isGBufferBinding(b.name)) {
                info.imageInfo.sampler = m_gbufSampler;
            } else {
                auto& iblTextures = m_scene->getIBLTextures();
                auto iblIt = iblTextures.find(b.name);
                if (iblIt != iblTextures.end()) {
                    info.imageInfo.sampler = iblIt->second.sampler;
                } else {
                    auto& tex = m_scene->getTextureForBinding(b.name);
                    info.imageInfo.sampler = tex.sampler;
                }
            }
            writeInfos.push_back(info);

            VkWriteDescriptorSet w = {};
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = dsIt->second;
            w.dstBinding = static_cast<uint32_t>(b.binding);
            w.descriptorCount = 1;
            w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
            w.pImageInfo = &writeInfos.back().imageInfo;
            writes.push_back(w);
        } else if (b.type == "sampled_image" || b.type == "sampled_cube_image") {
            DescWriteInfo info = {};
            info.imageInfo = {};

            auto [gbufView, gbufLayout] = getGBufferView(b.name);
            if (gbufView != VK_NULL_HANDLE) {
                info.imageInfo.imageView = gbufView;
                info.imageInfo.imageLayout = gbufLayout;
            } else {
                auto& iblTextures = m_scene->getIBLTextures();
                auto iblIt = iblTextures.find(b.name);
                if (iblIt != iblTextures.end()) {
                    info.imageInfo.imageView = iblIt->second.imageView;
                    info.imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                } else {
                    auto& tex = m_scene->getTextureForBinding(b.name);
                    info.imageInfo.imageView = tex.imageView;
                    info.imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                }
            }
            writeInfos.push_back(info);

            VkWriteDescriptorSet w = {};
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = dsIt->second;
            w.dstBinding = static_cast<uint32_t>(b.binding);
            w.descriptorCount = 1;
            w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            w.pImageInfo = &writeInfos.back().imageInfo;
            writes.push_back(w);
        } else if (b.type == "combined_image_sampler") {
            // Some shaders may use combined image samplers for G-buffer textures
            DescWriteInfo info = {};
            info.imageInfo = {};
            info.imageInfo.sampler = m_gbufSampler;

            auto [gbufView, gbufLayout] = getGBufferView(b.name);
            if (gbufView != VK_NULL_HANDLE) {
                info.imageInfo.imageView = gbufView;
                info.imageInfo.imageLayout = gbufLayout;
            } else {
                auto& tex = m_scene->getTextureForBinding(b.name);
                info.imageInfo.imageView = tex.imageView;
                info.imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                info.imageInfo.sampler = tex.sampler;
            }
            writeInfos.push_back(info);

            VkWriteDescriptorSet w = {};
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = dsIt->second;
            w.dstBinding = static_cast<uint32_t>(b.binding);
            w.descriptorCount = 1;
            w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w.pImageInfo = &writeInfos.back().imageInfo;
            writes.push_back(w);
        }
    }

    if (!writes.empty()) {
        vkUpdateDescriptorSets(ctx.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

    std::cout << "[info] Lighting descriptors created: " << mergedBindings.size()
              << " binding(s)" << std::endl;
}

// --------------------------------------------------------------------------
// PBR resources setup (MVP, Light, Material uniform buffers)
// --------------------------------------------------------------------------

void DeferredRenderer::setupPBRResources(VulkanContext& ctx) {
    float aspect = static_cast<float>(m_renderWidth) / static_cast<float>(m_renderHeight);

    // Create MVP uniform buffer (3 x mat4 = 192 bytes)
    struct MVPData {
        glm::mat4 model;
        glm::mat4 view;
        glm::mat4 projection;
    };
    MVPData mvpData;
    if (m_scene->hasSceneBounds()) {
        mvpData.model = glm::mat4(1.0f);
        mvpData.view = glm::lookAt(m_scene->getAutoEye(), m_scene->getAutoTarget(), m_scene->getAutoUp());
        mvpData.projection = glm::perspective(glm::radians(45.0f), aspect, 0.1f, m_scene->getAutoFar());
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
                    &m_mvpBuffer, &m_mvpAlloc, nullptr);

    void* mapped;
    vmaMapMemory(ctx.allocator, m_mvpAlloc, &mapped);
    memcpy(mapped, &mvpData, sizeof(MVPData));
    vmaUnmapMemory(ctx.allocator, m_mvpAlloc);

    // Create Light uniform buffer (32 bytes)
    struct LightData {
        glm::vec3 lightDir;
        float _pad0;
        glm::vec3 viewPos;
        float _pad1;
    };

    glm::vec3 lightDir = glm::normalize(glm::vec3(1.0f, 0.8f, 0.6f));
    glm::vec3 viewPos = m_scene->hasSceneBounds() ? m_scene->getAutoEye() : Camera::DEFAULT_EYE;
    LightData lightData = {lightDir, 0.0f, viewPos, 0.0f};

    VkBufferCreateInfo lightBufInfo = {};
    lightBufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    lightBufInfo.size = sizeof(LightData);
    lightBufInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

    VmaAllocationCreateInfo lightAllocInfo = {};
    lightAllocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;

    vmaCreateBuffer(ctx.allocator, &lightBufInfo, &lightAllocInfo,
                    &m_lightBuffer, &m_lightAlloc, nullptr);

    vmaMapMemory(ctx.allocator, m_lightAlloc, &mapped);
    memcpy(mapped, &lightData, sizeof(LightData));
    vmaUnmapMemory(ctx.allocator, m_lightAlloc);

    // Create Material uniform buffer (144 bytes, std140 layout)
    MaterialUBOData materialData{};
    if (m_scene && m_scene->hasGltfScene()) {
        const auto& mat = m_scene->getGltfScene().materials[0];
        materialData.baseColorFactor = mat.baseColor;
        materialData.metallicFactor = mat.metallic;
        materialData.roughnessFactor = mat.roughness;
        materialData.emissiveFactor = mat.emissive;
        materialData.emissiveStrength = mat.emissiveStrength;
        materialData.ior = mat.ior;
        materialData.clearcoatFactor = mat.clearcoatFactor;
        materialData.clearcoatRoughnessFactor = mat.clearcoatRoughnessFactor;
        materialData.sheenColorFactor = mat.sheenColorFactor;
        materialData.sheenRoughnessFactor = mat.sheenRoughnessFactor;
        materialData.transmissionFactor = mat.transmissionFactor;
    }

    VkBufferCreateInfo matBufInfo = {};
    matBufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    matBufInfo.size = sizeof(MaterialUBOData);
    matBufInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

    VmaAllocationCreateInfo matAllocInfo = {};
    matAllocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;

    vmaCreateBuffer(ctx.allocator, &matBufInfo, &matAllocInfo,
                    &m_materialBuffer, &m_materialAlloc, nullptr);

    vmaMapMemory(ctx.allocator, m_materialAlloc, &mapped);
    memcpy(mapped, &materialData, sizeof(MaterialUBOData));
    vmaUnmapMemory(ctx.allocator, m_materialAlloc);
}

// --------------------------------------------------------------------------
// Multi-light resources
// --------------------------------------------------------------------------

void DeferredRenderer::setupMultiLightResources(VulkanContext& ctx) {
    // Check both gbuf and light fragment reflections for "lights" storage buffer
    m_hasMultiLight = false;
    auto checkReflection = [&](const ReflectionData& refl) {
        for (auto& [setIdx, bindings] : refl.descriptor_sets) {
            for (auto& b : bindings) {
                if (b.name == "lights" && b.type == "storage_buffer") {
                    m_hasMultiLight = true;
                    return;
                }
            }
        }
    };
    checkReflection(m_gbufFragRefl);
    checkReflection(m_lightFragRefl);

    if (m_hasMultiLight && m_scene) {
        auto lightBuf = m_scene->packLightsBuffer();
        if (lightBuf.empty()) {
            lightBuf.resize(16, 0.0f);
        }

        VkBufferCreateInfo ssboInfo = {};
        ssboInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        ssboInfo.size = lightBuf.size() * sizeof(float);
        ssboInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

        VmaAllocationCreateInfo ssboAllocInfo = {};
        ssboAllocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;

        vmaCreateBuffer(ctx.allocator, &ssboInfo, &ssboAllocInfo,
                        &m_lightsSSBO, &m_lightsSSBOAlloc, nullptr);

        void* mapped;
        vmaMapMemory(ctx.allocator, m_lightsSSBOAlloc, &mapped);
        memcpy(mapped, lightBuf.data(), lightBuf.size() * sizeof(float));
        vmaUnmapMemory(ctx.allocator, m_lightsSSBOAlloc);

        // Create SceneLight UBO
        struct SceneLightUBO {
            glm::vec3 viewPos;
            int32_t lightCount;
        };
        glm::vec3 viewPos = m_scene->hasSceneBounds() ? m_scene->getAutoEye() : Camera::DEFAULT_EYE;
        SceneLightUBO sceneLightData;
        sceneLightData.viewPos = viewPos;
        sceneLightData.lightCount = m_scene->getLightCount();

        VkBufferCreateInfo slBufInfo = {};
        slBufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        slBufInfo.size = sizeof(SceneLightUBO);
        slBufInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

        VmaAllocationCreateInfo slAllocInfo = {};
        slAllocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;

        vmaCreateBuffer(ctx.allocator, &slBufInfo, &slAllocInfo,
                        &m_sceneLightUBO, &m_sceneLightUBOAlloc, nullptr);

        vmaMapMemory(ctx.allocator, m_sceneLightUBOAlloc, &mapped);
        memcpy(mapped, &sceneLightData, sizeof(SceneLightUBO));
        vmaUnmapMemory(ctx.allocator, m_sceneLightUBOAlloc);

        std::cout << "[info] Multi-light SSBO created: " << m_scene->getLightCount()
                  << " light(s), " << lightBuf.size() * sizeof(float) << " bytes" << std::endl;
    }
}

// --------------------------------------------------------------------------
// DeferredCamera UBO setup
// --------------------------------------------------------------------------

void DeferredRenderer::setupDeferredCameraUBO(VulkanContext& ctx) {
    // DeferredCamera UBO layout (std140):
    //   mat4 inv_view_proj   (offset 0, 64 bytes)
    //   vec3 view_pos        (offset 64, 12 bytes)
    //   float _pad           (offset 76, 4 bytes)
    // Total: 80 bytes

    struct DeferredCameraData {
        glm::mat4 invViewProj;
        glm::vec3 viewPos;
        float _pad;
    };

    float aspect = static_cast<float>(m_renderWidth) / static_cast<float>(m_renderHeight);
    glm::mat4 view, proj;
    glm::vec3 viewPos;

    if (m_scene->hasSceneBounds()) {
        viewPos = m_scene->getAutoEye();
        view = glm::lookAt(m_scene->getAutoEye(), m_scene->getAutoTarget(), m_scene->getAutoUp());
        proj = glm::perspective(glm::radians(45.0f), aspect, 0.1f, m_scene->getAutoFar());
        proj[1][1] *= -1.0f; // Vulkan Y-flip
    } else {
        viewPos = Camera::DEFAULT_EYE;
        view = Camera::lookAt(Camera::DEFAULT_EYE, Camera::DEFAULT_TARGET, Camera::DEFAULT_UP);
        proj = Camera::perspective(Camera::DEFAULT_FOV, aspect, Camera::DEFAULT_NEAR, Camera::DEFAULT_FAR);
    }

    DeferredCameraData camData;
    camData.invViewProj = glm::inverse(proj * view);
    camData.viewPos = viewPos;
    camData._pad = 0.0f;

    VkBufferCreateInfo bufInfo = {};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size = sizeof(DeferredCameraData);
    bufInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;

    vmaCreateBuffer(ctx.allocator, &bufInfo, &allocInfo,
                    &m_deferredCameraBuffer, &m_deferredCameraAlloc, nullptr);

    void* mapped;
    vmaMapMemory(ctx.allocator, m_deferredCameraAlloc, &mapped);
    memcpy(mapped, &camData, sizeof(DeferredCameraData));
    vmaUnmapMemory(ctx.allocator, m_deferredCameraAlloc);
}

// --------------------------------------------------------------------------
// Record G-buffer draw commands
// --------------------------------------------------------------------------

void DeferredRenderer::recordGBufferCommands(VkCommandBuffer cmd) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_gbufPipeline);

    // Push constants
    if (m_pushConstantSize > 0 && m_pushConstantStageFlags != 0) {
        vkCmdPushConstants(cmd, m_gbufPipelineLayout, m_pushConstantStageFlags,
                           0, m_pushConstantSize, m_pushConstantData.data());
    }

    const auto& mesh = m_scene->getMesh();
    if (m_gbufVertDescSet != VK_NULL_HANDLE) {
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                m_gbufPipelineLayout, 0, 1, &m_gbufVertDescSet, 0, nullptr);
    }

    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &mesh.vertexBuffer, &offset);
    vkCmdBindIndexBuffer(cmd, mesh.indexBuffer, 0, VK_INDEX_TYPE_UINT32);

    const auto& drawRanges = m_scene->getDrawRanges();
    if (!drawRanges.empty() && !m_gbufPerMaterialDescSets.empty()) {
        int currentMat = -1;
        for (auto& range : drawRanges) {
            if (range.materialIndex != currentMat) {
                currentMat = range.materialIndex;
                int matIdx = std::min(currentMat, static_cast<int>(m_gbufPerMaterialDescSets.size()) - 1);
                if (matIdx >= 0) {
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                            m_gbufPipelineLayout, 1, 1,
                                            &m_gbufPerMaterialDescSets[matIdx], 0, nullptr);
                }
            }
            vkCmdDrawIndexed(cmd, range.indexCount, 1, range.indexOffset, 0, 0);
        }
    } else {
        // Bind the first per-material descriptor set (set 1) for single-draw path
        if (!m_gbufPerMaterialDescSets.empty()) {
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    m_gbufPipelineLayout, 1, 1,
                                    &m_gbufPerMaterialDescSets[0], 0, nullptr);
        }
        vkCmdDrawIndexed(cmd, mesh.indexCount, 1, 0, 0, 0);
    }
}

// --------------------------------------------------------------------------
// Initialization
// --------------------------------------------------------------------------

void DeferredRenderer::init(VulkanContext& ctx, SceneManager& scene,
                            const std::string& pipelineBase,
                            uint32_t width, uint32_t height) {
    m_scene = &scene;
    m_pipelineBase = pipelineBase;
    m_renderWidth = width;
    m_renderHeight = height;

    // 1. Load 4 shader modules
    std::string gbufVertPath = pipelineBase + ".gbuf.vert.spv";
    std::string gbufFragPath = pipelineBase + ".gbuf.frag.spv";
    std::string lightVertPath = pipelineBase + ".light.vert.spv";
    std::string lightFragPath = pipelineBase + ".light.frag.spv";

    auto gbufVertCode = SpvLoader::loadSPIRV(gbufVertPath);
    m_gbufVertModule = SpvLoader::createShaderModule(ctx.device, gbufVertCode);
    auto gbufFragCode = SpvLoader::loadSPIRV(gbufFragPath);
    m_gbufFragModule = SpvLoader::createShaderModule(ctx.device, gbufFragCode);
    auto lightVertCode = SpvLoader::loadSPIRV(lightVertPath);
    m_lightVertModule = SpvLoader::createShaderModule(ctx.device, lightVertCode);
    auto lightFragCode = SpvLoader::loadSPIRV(lightFragPath);
    m_lightFragModule = SpvLoader::createShaderModule(ctx.device, lightFragCode);

    std::cout << "[info] Loaded 4 deferred shader modules" << std::endl;

    // 2. Load reflection JSON for all 4 stages
    std::string gbufVertJson = pipelineBase + ".gbuf.vert.json";
    std::string gbufFragJson = pipelineBase + ".gbuf.frag.json";
    std::string lightVertJson = pipelineBase + ".light.vert.json";
    std::string lightFragJson = pipelineBase + ".light.frag.json";

    if (fs::exists(gbufVertJson)) {
        std::cout << "[info] Loading reflection JSON: " << gbufVertJson << std::endl;
        m_gbufVertRefl = parseReflectionJson(gbufVertJson);
    }
    if (fs::exists(gbufFragJson)) {
        std::cout << "[info] Loading reflection JSON: " << gbufFragJson << std::endl;
        m_gbufFragRefl = parseReflectionJson(gbufFragJson);
    }
    if (fs::exists(lightVertJson)) {
        std::cout << "[info] Loading reflection JSON: " << lightVertJson << std::endl;
        m_lightVertRefl = parseReflectionJson(lightVertJson);
    }
    if (fs::exists(lightFragJson)) {
        std::cout << "[info] Loading reflection JSON: " << lightFragJson << std::endl;
        m_lightFragRefl = parseReflectionJson(lightFragJson);
    }

    // 3. Setup PBR resources (MVP, Light, Material UBOs)
    setupPBRResources(ctx);

    // 4. Setup DeferredCamera UBO
    setupDeferredCameraUBO(ctx);

    // 5. Detect multi-light
    setupMultiLightResources(ctx);

    // 6. Create G-buffer images (3 color + depth)
    createGBufferImages(ctx);

    // 7. Create G-buffer render pass
    createGBufferRenderPass(ctx);

    // 8. Create G-buffer framebuffer
    createGBufferFramebuffer(ctx);

    // 9. Setup G-buffer descriptors (reflection-driven)
    setupGBufferDescriptors(ctx);

    // 10. Create G-buffer pipeline
    createGBufferPipeline(ctx);

    // 11. Create lighting output image
    createLightingOutputImage(ctx);

    // 12. Create lighting render pass
    createLightingRenderPass(ctx);

    // 13. Create lighting framebuffer
    createLightingFramebuffer(ctx);

    // 14. Setup lighting descriptors
    setupLightingDescriptors(ctx);

    // 15. Create lighting pipeline
    createLightingPipeline(ctx);

    // 16. Setup push constants (G-buffer pass)
    {
        glm::vec3 lightDir = glm::normalize(glm::vec3(1.0f, 0.8f, 0.6f));
        auto buildPushData = [&](const ReflectionData& refl) {
            for (auto& pc : refl.push_constants) {
                if (pc.size <= 0) continue;
                m_pushConstantSize = std::max(m_pushConstantSize, static_cast<uint32_t>(pc.size));
                m_pushConstantData.resize(m_pushConstantSize, 0);
                m_pushConstantStageFlags |= (refl.stage == "vertex")
                    ? VK_SHADER_STAGE_VERTEX_BIT : VK_SHADER_STAGE_FRAGMENT_BIT;
                for (auto& f : pc.fields) {
                    if (f.name == "light_dir" && static_cast<uint32_t>(f.offset) + 12 <= m_pushConstantSize) {
                        memcpy(m_pushConstantData.data() + f.offset, &lightDir, sizeof(glm::vec3));
                    } else if (f.name == "view_pos" && static_cast<uint32_t>(f.offset) + 12 <= m_pushConstantSize) {
                        glm::vec3 eye = m_scene->hasSceneBounds() ? m_scene->getAutoEye() : Camera::DEFAULT_EYE;
                        memcpy(m_pushConstantData.data() + f.offset, &eye, sizeof(glm::vec3));
                    }
                }
            }
        };
        buildPushData(m_gbufVertRefl);
        buildPushData(m_gbufFragRefl);
    }

    std::cout << "[info] Deferred renderer initialized: " << pipelineBase
              << " (" << m_renderWidth << "x" << m_renderHeight << ")" << std::endl;
}

// --------------------------------------------------------------------------
// Rendering (both passes)
// --------------------------------------------------------------------------

void DeferredRenderer::render(VulkanContext& ctx) {
    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();

    // === Pass 1: G-buffer ===
    {
        std::array<VkClearValue, 4> clearValues = {};
        clearValues[0].color = {{0.0f, 0.0f, 0.0f, 0.0f}};
        clearValues[1].color = {{0.0f, 0.0f, 0.0f, 0.0f}};
        clearValues[2].color = {{0.0f, 0.0f, 0.0f, 0.0f}};
        clearValues[3].depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo rpBeginInfo = {};
        rpBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpBeginInfo.renderPass = m_gbufRenderPass;
        rpBeginInfo.framebuffer = m_gbufFramebuffer;
        rpBeginInfo.renderArea.offset = {0, 0};
        rpBeginInfo.renderArea.extent = {m_renderWidth, m_renderHeight};
        rpBeginInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        rpBeginInfo.pClearValues = clearValues.data();

        ctx.cmdBeginLabel(cmd, "G-Buffer Pass", 0.2f, 0.6f, 0.8f, 1.0f);
        vkCmdBeginRenderPass(cmd, &rpBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
        recordGBufferCommands(cmd);
        vkCmdEndRenderPass(cmd);
        ctx.cmdEndLabel(cmd);
    }

    // === Transition G-buffer images: COLOR_ATTACHMENT -> SHADER_READ_ONLY ===
    {
        std::array<VkImageMemoryBarrier, 4> barriers = {};

        // Color images
        VkImage colorImages[3] = {m_gbufRT0Image, m_gbufRT1Image, m_gbufRT2Image};
        for (int i = 0; i < 3; i++) {
            barriers[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barriers[i].oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            barriers[i].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[i].image = colorImages[i];
            barriers[i].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            barriers[i].subresourceRange.baseMipLevel = 0;
            barriers[i].subresourceRange.levelCount = 1;
            barriers[i].subresourceRange.baseArrayLayer = 0;
            barriers[i].subresourceRange.layerCount = 1;
            barriers[i].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            barriers[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        }

        // Depth image
        barriers[3].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barriers[3].oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        barriers[3].newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        barriers[3].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[3].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[3].image = m_gbufDepthImage;
        barriers[3].subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        barriers[3].subresourceRange.baseMipLevel = 0;
        barriers[3].subresourceRange.levelCount = 1;
        barriers[3].subresourceRange.baseArrayLayer = 0;
        barriers[3].subresourceRange.layerCount = 1;
        barriers[3].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        barriers[3].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 0, nullptr, 0, nullptr,
            static_cast<uint32_t>(barriers.size()), barriers.data());
    }

    // === Pass 2: Lighting ===
    {
        VkClearValue clearValue = {};
        clearValue.color = {{0.0f, 0.0f, 0.0f, 1.0f}};

        VkRenderPassBeginInfo rpBeginInfo = {};
        rpBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpBeginInfo.renderPass = m_lightRenderPass;
        rpBeginInfo.framebuffer = m_lightFramebuffer;
        rpBeginInfo.renderArea.offset = {0, 0};
        rpBeginInfo.renderArea.extent = {m_renderWidth, m_renderHeight};
        rpBeginInfo.clearValueCount = 1;
        rpBeginInfo.pClearValues = &clearValue;

        ctx.cmdBeginLabel(cmd, "Lighting Pass", 0.8f, 0.8f, 0.2f, 1.0f);
        vkCmdBeginRenderPass(cmd, &rpBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_lightPipeline);

        // Bind all lighting descriptor sets
        int maxSet = -1;
        for (auto& [idx, _] : m_lightDescSets) { maxSet = std::max(maxSet, idx); }
        if (maxSet >= 0) {
            std::vector<VkDescriptorSet> allSets;
            for (int i = 0; i <= maxSet; i++) {
                auto it = m_lightDescSets.find(i);
                if (it != m_lightDescSets.end()) {
                    allSets.push_back(it->second);
                } else {
                    allSets.push_back(VK_NULL_HANDLE);
                }
            }
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    m_lightPipelineLayout, 0,
                                    static_cast<uint32_t>(allSets.size()),
                                    allSets.data(), 0, nullptr);
        }

        // Draw fullscreen triangle (3 vertices, no vertex buffer)
        vkCmdDraw(cmd, 3, 1, 0, 0);

        vkCmdEndRenderPass(cmd);
        ctx.cmdEndLabel(cmd);
    }

    ctx.endSingleTimeCommands(cmd);
}

// --------------------------------------------------------------------------
// Update camera
// --------------------------------------------------------------------------

void DeferredRenderer::updateCamera(VulkanContext& ctx, glm::vec3 eye, glm::vec3 target,
                                    glm::vec3 up, float fovY, float aspect,
                                    float nearPlane, float farPlane) {
    if (!m_mvpBuffer || !m_lightBuffer) return;

    // Update MVP
    struct MVPData { glm::mat4 model, view, projection; };
    MVPData mvp;
    mvp.model = glm::mat4(1.0f);
    mvp.view = Camera::lookAt(eye, target, up);
    mvp.projection = Camera::perspective(fovY, aspect, nearPlane, farPlane);

    void* mapped;
    vmaMapMemory(ctx.allocator, m_mvpAlloc, &mapped);
    memcpy(mapped, &mvp, sizeof(MVPData));
    vmaUnmapMemory(ctx.allocator, m_mvpAlloc);

    // Update Light uniform (preserve light_dir, update view_pos)
    struct LightData {
        glm::vec3 lightDir; float _pad0;
        glm::vec3 viewPos; float _pad1;
    };
    LightData light;
    vmaMapMemory(ctx.allocator, m_lightAlloc, &mapped);
    memcpy(&light, mapped, sizeof(LightData));
    light.viewPos = eye;
    memcpy(mapped, &light, sizeof(LightData));
    vmaUnmapMemory(ctx.allocator, m_lightAlloc);

    // Update DeferredCamera UBO
    if (m_deferredCameraBuffer != VK_NULL_HANDLE) {
        struct DeferredCameraData {
            glm::mat4 invViewProj;
            glm::vec3 viewPos;
            float _pad;
        };
        DeferredCameraData camData;
        camData.invViewProj = glm::inverse(mvp.projection * mvp.view);
        camData.viewPos = eye;
        camData._pad = 0.0f;

        vmaMapMemory(ctx.allocator, m_deferredCameraAlloc, &mapped);
        memcpy(mapped, &camData, sizeof(DeferredCameraData));
        vmaUnmapMemory(ctx.allocator, m_deferredCameraAlloc);
    }

    // Update SceneLight UBO viewPos for multi-light mode
    if (m_hasMultiLight && m_sceneLightUBO != VK_NULL_HANDLE) {
        struct SceneLightUBO {
            glm::vec3 viewPos;
            int32_t lightCount;
        };
        vmaMapMemory(ctx.allocator, m_sceneLightUBOAlloc, &mapped);
        SceneLightUBO* sl = static_cast<SceneLightUBO*>(mapped);
        sl->viewPos = eye;
        vmaUnmapMemory(ctx.allocator, m_sceneLightUBOAlloc);
    }
}

// --------------------------------------------------------------------------
// Blit to swapchain
// --------------------------------------------------------------------------

void DeferredRenderer::blitToSwapchain(VulkanContext& ctx, VkCommandBuffer cmd,
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
    blitRegion.srcOffsets[1] = {static_cast<int32_t>(m_renderWidth),
                                static_cast<int32_t>(m_renderHeight), 1};
    blitRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blitRegion.dstSubresource.layerCount = 1;
    blitRegion.dstOffsets[1] = {static_cast<int32_t>(extent.width),
                                static_cast<int32_t>(extent.height), 1};

    vkCmdBlitImage(cmd,
        m_lightOutputImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
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
// Interactive mode: render to swapchain
// --------------------------------------------------------------------------

void DeferredRenderer::renderToSwapchain(VulkanContext& ctx,
                                          VkImage swapImage, VkImageView /*swapView*/,
                                          VkFormat /*swapFormat*/, VkExtent2D extent,
                                          VkSemaphore waitSem, VkSemaphore signalSem,
                                          VkFence fence) {
    // Free previous frame's command buffer
    if (m_interactiveCmdBuffer != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(ctx.device, ctx.commandPool, 1, &m_interactiveCmdBuffer);
        m_interactiveCmdBuffer = VK_NULL_HANDLE;
    }

    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = ctx.commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(ctx.device, &allocInfo, &cmd);
    m_interactiveCmdBuffer = cmd;

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    // === Pass 1: G-buffer ===
    {
        std::array<VkClearValue, 4> clearValues = {};
        clearValues[0].color = {{0.0f, 0.0f, 0.0f, 0.0f}};
        clearValues[1].color = {{0.0f, 0.0f, 0.0f, 0.0f}};
        clearValues[2].color = {{0.0f, 0.0f, 0.0f, 0.0f}};
        clearValues[3].depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo rpBeginInfo = {};
        rpBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpBeginInfo.renderPass = m_gbufRenderPass;
        rpBeginInfo.framebuffer = m_gbufFramebuffer;
        rpBeginInfo.renderArea.offset = {0, 0};
        rpBeginInfo.renderArea.extent = {m_renderWidth, m_renderHeight};
        rpBeginInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        rpBeginInfo.pClearValues = clearValues.data();

        ctx.cmdBeginLabel(cmd, "G-Buffer Pass", 0.2f, 0.6f, 0.8f, 1.0f);
        vkCmdBeginRenderPass(cmd, &rpBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
        recordGBufferCommands(cmd);
        vkCmdEndRenderPass(cmd);
        ctx.cmdEndLabel(cmd);
    }

    // === Transition G-buffer images ===
    {
        std::array<VkImageMemoryBarrier, 4> barriers = {};
        VkImage colorImages[3] = {m_gbufRT0Image, m_gbufRT1Image, m_gbufRT2Image};
        for (int i = 0; i < 3; i++) {
            barriers[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barriers[i].oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            barriers[i].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[i].image = colorImages[i];
            barriers[i].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            barriers[i].subresourceRange.baseMipLevel = 0;
            barriers[i].subresourceRange.levelCount = 1;
            barriers[i].subresourceRange.baseArrayLayer = 0;
            barriers[i].subresourceRange.layerCount = 1;
            barriers[i].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            barriers[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        }
        barriers[3].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barriers[3].oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        barriers[3].newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        barriers[3].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[3].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[3].image = m_gbufDepthImage;
        barriers[3].subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        barriers[3].subresourceRange.baseMipLevel = 0;
        barriers[3].subresourceRange.levelCount = 1;
        barriers[3].subresourceRange.baseArrayLayer = 0;
        barriers[3].subresourceRange.layerCount = 1;
        barriers[3].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        barriers[3].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 0, nullptr, 0, nullptr,
            static_cast<uint32_t>(barriers.size()), barriers.data());
    }

    // === Pass 2: Lighting ===
    {
        VkClearValue clearValue = {};
        clearValue.color = {{0.0f, 0.0f, 0.0f, 1.0f}};

        VkRenderPassBeginInfo rpBeginInfo = {};
        rpBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpBeginInfo.renderPass = m_lightRenderPass;
        rpBeginInfo.framebuffer = m_lightFramebuffer;
        rpBeginInfo.renderArea.offset = {0, 0};
        rpBeginInfo.renderArea.extent = {m_renderWidth, m_renderHeight};
        rpBeginInfo.clearValueCount = 1;
        rpBeginInfo.pClearValues = &clearValue;

        ctx.cmdBeginLabel(cmd, "Lighting Pass", 0.8f, 0.8f, 0.2f, 1.0f);
        vkCmdBeginRenderPass(cmd, &rpBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_lightPipeline);

        int maxSet = -1;
        for (auto& [idx, _] : m_lightDescSets) { maxSet = std::max(maxSet, idx); }
        if (maxSet >= 0) {
            std::vector<VkDescriptorSet> allSets;
            for (int i = 0; i <= maxSet; i++) {
                auto it = m_lightDescSets.find(i);
                if (it != m_lightDescSets.end()) {
                    allSets.push_back(it->second);
                } else {
                    allSets.push_back(VK_NULL_HANDLE);
                }
            }
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    m_lightPipelineLayout, 0,
                                    static_cast<uint32_t>(allSets.size()),
                                    allSets.data(), 0, nullptr);
        }

        vkCmdDraw(cmd, 3, 1, 0, 0);

        vkCmdEndRenderPass(cmd);
        ctx.cmdEndLabel(cmd);
    }

    // Blit to swapchain
    blitToSwapchain(ctx, cmd, swapImage, extent);

    vkEndCommandBuffer(cmd);

    // Submit
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
// Cleanup
// --------------------------------------------------------------------------

void DeferredRenderer::cleanup(VulkanContext& ctx) {
    vkDeviceWaitIdle(ctx.device);

    // Interactive command buffer
    if (m_interactiveCmdBuffer != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(ctx.device, ctx.commandPool, 1, &m_interactiveCmdBuffer);
        m_interactiveCmdBuffer = VK_NULL_HANDLE;
    }

    // Lighting pipeline
    if (m_lightPipeline) vkDestroyPipeline(ctx.device, m_lightPipeline, nullptr);
    if (m_lightPipelineLayout) vkDestroyPipelineLayout(ctx.device, m_lightPipelineLayout, nullptr);
    if (m_lightFramebuffer) vkDestroyFramebuffer(ctx.device, m_lightFramebuffer, nullptr);
    if (m_lightRenderPass) vkDestroyRenderPass(ctx.device, m_lightRenderPass, nullptr);
    if (m_lightVertModule) vkDestroyShaderModule(ctx.device, m_lightVertModule, nullptr);
    if (m_lightFragModule) vkDestroyShaderModule(ctx.device, m_lightFragModule, nullptr);

    // Lighting output image
    if (m_lightOutputView) vkDestroyImageView(ctx.device, m_lightOutputView, nullptr);
    if (m_lightOutputImage) vmaDestroyImage(ctx.allocator, m_lightOutputImage, m_lightOutputAlloc);

    // Lighting descriptors
    if (m_lightDescPool) vkDestroyDescriptorPool(ctx.device, m_lightDescPool, nullptr);
    for (auto& [_, layout] : m_lightSetLayouts) {
        vkDestroyDescriptorSetLayout(ctx.device, layout, nullptr);
    }
    m_lightSetLayouts.clear();
    m_lightDescSets.clear();

    // G-buffer pipeline
    if (m_gbufPipeline) vkDestroyPipeline(ctx.device, m_gbufPipeline, nullptr);
    if (m_gbufPipelineLayout) vkDestroyPipelineLayout(ctx.device, m_gbufPipelineLayout, nullptr);
    if (m_gbufFramebuffer) vkDestroyFramebuffer(ctx.device, m_gbufFramebuffer, nullptr);
    if (m_gbufRenderPass) vkDestroyRenderPass(ctx.device, m_gbufRenderPass, nullptr);
    if (m_gbufVertModule) vkDestroyShaderModule(ctx.device, m_gbufVertModule, nullptr);
    if (m_gbufFragModule) vkDestroyShaderModule(ctx.device, m_gbufFragModule, nullptr);

    // G-buffer images
    if (m_gbufRT0View) vkDestroyImageView(ctx.device, m_gbufRT0View, nullptr);
    if (m_gbufRT0Image) vmaDestroyImage(ctx.allocator, m_gbufRT0Image, m_gbufRT0Alloc);
    if (m_gbufRT1View) vkDestroyImageView(ctx.device, m_gbufRT1View, nullptr);
    if (m_gbufRT1Image) vmaDestroyImage(ctx.allocator, m_gbufRT1Image, m_gbufRT1Alloc);
    if (m_gbufRT2View) vkDestroyImageView(ctx.device, m_gbufRT2View, nullptr);
    if (m_gbufRT2Image) vmaDestroyImage(ctx.allocator, m_gbufRT2Image, m_gbufRT2Alloc);
    if (m_gbufDepthView) vkDestroyImageView(ctx.device, m_gbufDepthView, nullptr);
    if (m_gbufDepthImage) vmaDestroyImage(ctx.allocator, m_gbufDepthImage, m_gbufDepthAlloc);

    // G-buffer sampler
    if (m_gbufSampler) vkDestroySampler(ctx.device, m_gbufSampler, nullptr);

    // G-buffer descriptors
    for (size_t i = 0; i < m_gbufPerMaterialBuffers.size(); i++) {
        if (m_gbufPerMaterialBuffers[i]) {
            vmaDestroyBuffer(ctx.allocator, m_gbufPerMaterialBuffers[i], m_gbufPerMaterialAllocations[i]);
        }
    }
    m_gbufPerMaterialBuffers.clear();
    m_gbufPerMaterialAllocations.clear();
    m_gbufPerMaterialDescSets.clear();

    if (m_gbufDescPool) vkDestroyDescriptorPool(ctx.device, m_gbufDescPool, nullptr);
    for (auto& [_, layout] : m_gbufSetLayouts) {
        vkDestroyDescriptorSetLayout(ctx.device, layout, nullptr);
    }
    m_gbufSetLayouts.clear();

    // Uniform buffers
    if (m_mvpBuffer) vmaDestroyBuffer(ctx.allocator, m_mvpBuffer, m_mvpAlloc);
    if (m_lightBuffer) vmaDestroyBuffer(ctx.allocator, m_lightBuffer, m_lightAlloc);
    if (m_materialBuffer) vmaDestroyBuffer(ctx.allocator, m_materialBuffer, m_materialAlloc);
    if (m_deferredCameraBuffer) vmaDestroyBuffer(ctx.allocator, m_deferredCameraBuffer, m_deferredCameraAlloc);

    // Multi-light buffers
    if (m_lightsSSBO) vmaDestroyBuffer(ctx.allocator, m_lightsSSBO, m_lightsSSBOAlloc);
    if (m_sceneLightUBO) vmaDestroyBuffer(ctx.allocator, m_sceneLightUBO, m_sceneLightUBOAlloc);
    m_lightsSSBO = VK_NULL_HANDLE;
    m_sceneLightUBO = VK_NULL_HANDLE;
    m_hasMultiLight = false;
}
