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

    VkSubpassDependency dependencies[2] = {};

    // EXTERNAL -> 0: ensure prior commands complete before rendering
    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0;
    dependencies[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                   VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                   VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependencies[0].srcAccessMask = 0;
    dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    // 0 -> EXTERNAL: ensure color writes are visible to subsequent blit/transfer
    dependencies[1].srcSubpass = 0;
    dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[1].dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
    dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[1].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    VkRenderPassCreateInfo rpInfo = {};
    rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    rpInfo.pAttachments = attachments.data();
    rpInfo.subpassCount = 1;
    rpInfo.pSubpasses = &subpass;
    rpInfo.dependencyCount = 2;
    rpInfo.pDependencies = dependencies;

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
    if (m_scene->hasSceneBounds()) {
        // Use auto-camera computed from scene bounds
        mvpData.model = glm::mat4(1.0f); // identity - transforms already baked into vertices
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
    glm::vec3 viewPos = m_scene->hasSceneBounds() ? m_scene->getAutoEye() : Camera::DEFAULT_EYE;
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

    // Create Material uniform buffer (80 bytes, std140 layout)
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
                    &m_materialBuffer, &m_materialAllocation, nullptr);

    vmaMapMemory(ctx.allocator, m_materialAllocation, &mapped);
    memcpy(mapped, &materialData, sizeof(MaterialUBOData));
    vmaUnmapMemory(ctx.allocator, m_materialAllocation);

}

// --------------------------------------------------------------------------
// Reflection-driven descriptor set creation
// --------------------------------------------------------------------------

void RasterRenderer::setupReflectedDescriptors(VulkanContext& ctx) {
    std::vector<ReflectionData> stages = {vertReflection, fragReflection};
    reflectedSetLayouts = createDescriptorSetLayoutsMultiStage(ctx.device, stages);
    auto mergedBindings = getMergedBindings(stages);

    // Identify set indices
    int vertSetIdx = 0;
    int fragSetIdx = 1;
    int bindlessTexSetIdx = -1;
    for (auto& b : mergedBindings) {
        if (b.name == "MVP") vertSetIdx = b.set;
        if (b.name == "Material" || b.name == "materials") fragSetIdx = b.set;
        if (b.type == "bindless_combined_image_sampler_array") bindlessTexSetIdx = b.set;
    }

    if (m_bindlessMode) {
        // =====================================================================
        // BINDLESS MODE: sets 0 (MVP), 1 (Light + materials SSBO + IBL), 2 (textures[])
        // =====================================================================
        uint32_t actualTexCount = m_bindlessTextures.textureCount;
        if (actualTexCount == 0) actualTexCount = 1;  // need at least 1 for valid descriptor

        // Count pool sizes
        std::unordered_map<VkDescriptorType, uint32_t> typeCounts;
        for (auto& b : mergedBindings) {
            VkDescriptorType vkType = bindingTypeToVkDescriptorType(b.type);
            if (b.type == "bindless_combined_image_sampler_array") {
                typeCounts[vkType] += actualTexCount;
            } else {
                typeCounts[vkType]++;
            }
        }

        std::vector<VkDescriptorPoolSize> poolSizes;
        for (auto& [type, count] : typeCounts) {
            poolSizes.push_back({type, count});
        }

        // Determine how many sets we need
        int maxSetIdx = -1;
        for (auto& [idx, _] : reflectedSetLayouts) { maxSetIdx = std::max(maxSetIdx, idx); }
        uint32_t maxSets = static_cast<uint32_t>(maxSetIdx + 1);

        VkDescriptorPoolCreateInfo poolInfo = {};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
        poolInfo.maxSets = maxSets;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr, &descriptorPool);

        // Allocate all descriptor sets
        for (int setIdx = 0; setIdx <= maxSetIdx; setIdx++) {
            auto layoutIt = reflectedSetLayouts.find(setIdx);
            if (layoutIt == reflectedSetLayouts.end()) continue;

            VkDescriptorSetAllocateInfo allocSetInfo = {};
            allocSetInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            allocSetInfo.descriptorPool = descriptorPool;
            allocSetInfo.descriptorSetCount = 1;
            allocSetInfo.pSetLayouts = &layoutIt->second;

            // For the bindless texture set, use variable descriptor count
            VkDescriptorSetVariableDescriptorCountAllocateInfo variableCountInfo = {};
            uint32_t variableCount = actualTexCount;
            if (setIdx == bindlessTexSetIdx) {
                variableCountInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO;
                variableCountInfo.descriptorSetCount = 1;
                variableCountInfo.pDescriptorCounts = &variableCount;
                allocSetInfo.pNext = &variableCountInfo;
            }

            VkDescriptorSet ds;
            VkResult result = vkAllocateDescriptorSets(ctx.device, &allocSetInfo, &ds);
            if (result != VK_SUCCESS) {
                throw std::runtime_error("Failed to allocate bindless descriptor set " + std::to_string(setIdx));
            }
            reflectedDescSets[setIdx] = ds;

            if (setIdx == vertSetIdx) m_vertexDescSet = ds;
        }

        // Write all descriptor sets
        struct DescWriteInfo { VkDescriptorBufferInfo bufferInfo; VkDescriptorImageInfo imageInfo; };
        std::vector<DescWriteInfo> writeInfos;
        writeInfos.reserve(mergedBindings.size());
        std::vector<VkWriteDescriptorSet> writes;
        writes.reserve(mergedBindings.size());
        // Reserve space for bindless texture image infos
        std::vector<VkDescriptorImageInfo> bindlessImageInfos;

        for (auto& b : mergedBindings) {
            auto dsIt = reflectedDescSets.find(b.set);
            if (dsIt == reflectedDescSets.end()) continue;

            if (b.type == "uniform_buffer") {
                DescWriteInfo info = {};
                VkWriteDescriptorSet w = {};
                w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w.dstSet = dsIt->second;
                w.dstBinding = static_cast<uint32_t>(b.binding);
                w.descriptorCount = 1;
                w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;

                if (b.name == "MVP") {
                    info.bufferInfo = {mvpBuffer, 0, static_cast<VkDeviceSize>(b.size > 0 ? b.size : 192)};
                } else if (b.name == "Light") {
                    info.bufferInfo = {lightBuffer, 0, static_cast<VkDeviceSize>(b.size > 0 ? b.size : 32)};
                } else {
                    continue;
                }
                writeInfos.push_back(info);
                w.pBufferInfo = &writeInfos.back().bufferInfo;
                writes.push_back(w);
            } else if (b.type == "storage_buffer") {
                // Materials SSBO
                if (m_bindlessMaterials.buffer != VK_NULL_HANDLE) {
                    DescWriteInfo info = {};
                    info.bufferInfo = {m_bindlessMaterials.buffer, 0, VK_WHOLE_SIZE};
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
                auto& iblTextures = m_scene->getIBLTextures();
                auto iblIt = iblTextures.find(b.name);
                if (iblIt != iblTextures.end()) {
                    info.imageInfo.sampler = iblIt->second.sampler;
                } else {
                    auto& tex = m_scene->getTextureForBinding(b.name);
                    info.imageInfo.sampler = tex.sampler;
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
                writeInfos.push_back(info);

                VkWriteDescriptorSet w = {};
                w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w.dstSet = dsIt->second;
                w.dstBinding = static_cast<uint32_t>(b.binding);
                w.descriptorCount = 1;
                w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                w.pImageInfo = &writeInfos.back().imageInfo;
                writes.push_back(w);
            } else if (b.type == "bindless_combined_image_sampler_array") {
                // Write ALL texture imageViews/samplers as combined image samplers
                if (m_bindlessTextures.textureCount > 0) {
                    bindlessImageInfos.resize(m_bindlessTextures.textureCount);
                    for (uint32_t ti = 0; ti < m_bindlessTextures.textureCount; ti++) {
                        bindlessImageInfos[ti] = {};
                        bindlessImageInfos[ti].imageView = m_bindlessTextures.imageViews[ti];
                        bindlessImageInfos[ti].sampler = m_bindlessTextures.samplers[ti];
                        bindlessImageInfos[ti].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                    }

                    VkWriteDescriptorSet w = {};
                    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    w.dstSet = dsIt->second;
                    w.dstBinding = static_cast<uint32_t>(b.binding);
                    w.dstArrayElement = 0;
                    w.descriptorCount = m_bindlessTextures.textureCount;
                    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    w.pImageInfo = bindlessImageInfos.data();
                    writes.push_back(w);
                }
            }
        }

        if (!writes.empty()) {
            vkUpdateDescriptorSets(ctx.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        }

        std::cout << "[info] Bindless descriptors created: " << mergedBindings.size()
                  << " binding(s), " << m_bindlessTextures.textureCount << " texture(s), "
                  << m_bindlessMaterials.materialCount << " material(s)" << std::endl;
        return;
    }

    // =====================================================================
    // NON-BINDLESS MODE: original per-material descriptor path
    // =====================================================================

    size_t numMaterials = 1;
    if (m_scene && m_scene->hasGltfScene()) {
        numMaterials = std::max(size_t(1), m_scene->getGltfScene().materials.size());
    }

    // Count pool sizes: vertex set bindings x1, fragment set bindings x numMaterials
    std::unordered_map<VkDescriptorType, uint32_t> typeCounts;
    for (auto& b : mergedBindings) {
        VkDescriptorType vkType;
        if (b.type == "uniform_buffer") vkType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        else if (b.type == "sampler") vkType = VK_DESCRIPTOR_TYPE_SAMPLER;
        else if (b.type == "sampled_image") vkType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        else if (b.type == "sampled_cube_image") vkType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        else if (b.type == "storage_image") vkType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        else if (b.type == "storage_buffer") vkType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        else if (b.type == "acceleration_structure") vkType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        else vkType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;

        uint32_t multiplier = (b.set == fragSetIdx) ? static_cast<uint32_t>(numMaterials) : 1;
        typeCounts[vkType] += multiplier;
    }

    std::vector<VkDescriptorPoolSize> poolSizes;
    for (auto& [type, count] : typeCounts) {
        poolSizes.push_back({type, count});
    }

    uint32_t maxSets = 1 + static_cast<uint32_t>(numMaterials); // 1 vertex + N fragment

    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = maxSets;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr, &descriptorPool);

    // Allocate vertex descriptor set (set 0) once
    {
        auto layoutIt = reflectedSetLayouts.find(vertSetIdx);
        if (layoutIt != reflectedSetLayouts.end()) {
            VkDescriptorSetAllocateInfo allocSetInfo = {};
            allocSetInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            allocSetInfo.descriptorPool = descriptorPool;
            allocSetInfo.descriptorSetCount = 1;
            allocSetInfo.pSetLayouts = &layoutIt->second;
            vkAllocateDescriptorSets(ctx.device, &allocSetInfo, &m_vertexDescSet);
            reflectedDescSets[vertSetIdx] = m_vertexDescSet;
        }
    }

    // Write vertex set (set 0): MVP + Light
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
            w.dstSet = m_vertexDescSet;
            w.dstBinding = static_cast<uint32_t>(b.binding);
            w.descriptorCount = 1;

            if (b.type == "uniform_buffer") {
                w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                if (b.name == "MVP") {
                    info.bufferInfo = {mvpBuffer, 0, static_cast<VkDeviceSize>(b.size > 0 ? b.size : 192)};
                } else if (b.name == "Light") {
                    info.bufferInfo = {lightBuffer, 0, static_cast<VkDeviceSize>(b.size > 0 ? b.size : 32)};
                } else {
                    continue;
                }
                writeInfos.push_back(info);
                w.pBufferInfo = &writeInfos.back().bufferInfo;
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
    auto fragLayoutIt = reflectedSetLayouts.find(fragSetIdx);
    if (fragLayoutIt != reflectedSetLayouts.end()) {
        m_perMaterialDescSets.resize(numMaterials);
        m_perMaterialBuffers.resize(numMaterials);
        m_perMaterialAllocations.resize(numMaterials);

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
                // UV transform data
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
                            &m_perMaterialBuffers[mi], &m_perMaterialAllocations[mi], nullptr);

            void* mapped;
            vmaMapMemory(ctx.allocator, m_perMaterialAllocations[mi], &mapped);
            memcpy(mapped, &materialData, sizeof(MaterialUBOData));
            vmaUnmapMemory(ctx.allocator, m_perMaterialAllocations[mi]);

            // Allocate descriptor set for this material
            VkDescriptorSetAllocateInfo allocSetInfo = {};
            allocSetInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            allocSetInfo.descriptorPool = descriptorPool;
            allocSetInfo.descriptorSetCount = 1;
            allocSetInfo.pSetLayouts = &fragLayoutIt->second;
            vkAllocateDescriptorSets(ctx.device, &allocSetInfo, &m_perMaterialDescSets[mi]);

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
                w.dstSet = m_perMaterialDescSets[mi];
                w.dstBinding = static_cast<uint32_t>(b.binding);
                w.descriptorCount = 1;

                if (b.type == "uniform_buffer") {
                    w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                    if (b.name == "Material") {
                        info.bufferInfo = {m_perMaterialBuffers[mi], 0, sizeof(MaterialUBOData)};
                    } else if (b.name == "Light" && lightBuffer != VK_NULL_HANDLE) {
                        info.bufferInfo = {lightBuffer, 0, static_cast<VkDeviceSize>(b.size > 0 ? b.size : 32)};
                    } else {
                        continue;
                    }
                    writeInfos.push_back(info);
                    w.pBufferInfo = &writeInfos.back().bufferInfo;
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
                    // Check IBL first
                    auto& iblTextures = m_scene->getIBLTextures();
                    auto iblIt = iblTextures.find(b.name);
                    if (iblIt != iblTextures.end()) {
                        info.imageInfo = {};
                        info.imageInfo.imageView = iblIt->second.imageView;
                        info.imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                    } else {
                        // Check per-material textures
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
                            // Fallback to defaults
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

    std::cout << "[info] Per-material descriptors created: " << numMaterials
              << " material(s), " << mergedBindings.size() << " binding(s) each" << std::endl;
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
    // For glTF scenes, buffer is always 48-byte stride; override pipeline stride to match.
    int bufferStride = (m_scene && m_scene->hasGltfScene()) ? 48 : 0;
    auto reflectedInput = createReflectedVertexInput(vertReflection, bufferStride);

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
// Initialization
// --------------------------------------------------------------------------

void RasterRenderer::init(VulkanContext& ctx, SceneManager& scene,
                          const std::string& pipelineBase, const std::string& renderPath,
                          uint32_t width, uint32_t height) {
    m_scene = &scene;
    m_renderPath = renderPath;
    m_pipelineBase = pipelineBase;
    renderWidth = width;
    renderHeight = height;

    const std::string& sceneSource = m_scene->getSceneSource();

    // Determine if depth buffer is needed based on scene
    m_needsDepth = (sceneSource == "sphere" || SceneManager::isGltfFile(sceneSource));

    // Upload scene resources
    if (sceneSource == "triangle") {
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
    } else if (sceneSource != "fullscreen") {
        // For sphere and glTF scenes, setup PBR resources (uniforms)
        setupPBRResources(ctx);
    }

    // Create offscreen target (depends on depth requirement)
    createOffscreenTarget(ctx);
    createRenderPass(ctx);
    createFramebuffer(ctx);

    // Create pipeline from shader files
    namespace fs = std::filesystem;
    std::string fragPath = pipelineBase + ".frag.spv";

    if (renderPath == "fullscreen") {
        const auto& fullscreenSpv = SpvLoader::getFullscreenVertSPIRV();
        vertModule = SpvLoader::createShaderModule(ctx.device, fullscreenSpv);
        auto fragCode = SpvLoader::loadSPIRV(fragPath);
        fragModule = SpvLoader::createShaderModule(ctx.device, fragCode);
        createPipelineFullscreen(ctx);
    } else if (renderPath == "raster") {
        // Check for manifest to enable multi-pipeline mode
        ShaderManifest manifest = tryLoadManifest(pipelineBase);
        bool hasMultipleMaterials = m_scene->hasGltfScene() &&
            m_scene->getGltfScene().materials.size() > 1;

        // Probe for bindless shader variant: load base reflection JSON first to check
        bool bindlessAvailable = false;
        {
            std::string probeFragJson = pipelineBase + ".frag.json";
            if (fs::exists(probeFragJson)) {
                auto probeRefl = parseReflectionJson(probeFragJson);
                if (probeRefl.bindlessEnabled && ctx.bindlessSupported) {
                    bindlessAvailable = true;
                }
            }
        }

        if (bindlessAvailable) {
            // Bindless mode: single uber-shader pipeline for all materials
            m_bindlessMode = true;
            std::cout << "[info] Bindless mode enabled" << std::endl;

            std::string vertPath = pipelineBase + ".vert.spv";
            fragPath = pipelineBase + ".frag.spv";
            auto vertCode = SpvLoader::loadSPIRV(vertPath);
            vertModule = SpvLoader::createShaderModule(ctx.device, vertCode);
            auto fragCode = SpvLoader::loadSPIRV(fragPath);
            fragModule = SpvLoader::createShaderModule(ctx.device, fragCode);

            std::string vertJsonPath = pipelineBase + ".vert.json";
            std::string fragJsonPath = pipelineBase + ".frag.json";

            if (fs::exists(vertJsonPath) && fs::exists(fragJsonPath)) {
                std::cout << "[info] Loading reflection JSON: " << vertJsonPath << std::endl;
                vertReflection = parseReflectionJson(vertJsonPath);
                std::cout << "[info] Loading reflection JSON: " << fragJsonPath << std::endl;
                fragReflection = parseReflectionJson(fragJsonPath);
            }

            // Build bindless texture array and materials SSBO
            m_bindlessTextures = m_scene->buildBindlessTextureArray(ctx);
            m_bindlessMaterials = m_scene->buildMaterialsSSBO(ctx, m_bindlessTextures);

            // Use single pipeline path
            setupReflectedDescriptors(ctx);
            createPipelinePBR(ctx);
        } else if (!manifest.permutations.empty() && hasMultipleMaterials) {
            // Multi-pipeline mode: create one pipeline per permutation used
            setupMultiPipeline(ctx, manifest);
        } else {
            // Single-pipeline mode: resolve permutation from manifest if available
            std::string resolvedBase = pipelineBase;
            if (!manifest.permutations.empty() && m_scene->hasGltfScene()) {
                // Auto-detect features for single-material scene and resolve permutation
                auto sceneFeatures = m_scene->detectSceneFeatures();
                std::string suffix = findPermutationSuffix(manifest, sceneFeatures);
                if (!suffix.empty()) {
                    std::string candidateBase = pipelineBase + suffix;
                    if (fs::exists(candidateBase + ".vert.spv") && fs::exists(candidateBase + ".frag.spv")) {
                        resolvedBase = candidateBase;
                        std::cout << "[info] Resolved single-material permutation: " << suffix << std::endl;
                    }
                }
            }

            std::string vertPath = resolvedBase + ".vert.spv";
            fragPath = resolvedBase + ".frag.spv";
            auto vertCode = SpvLoader::loadSPIRV(vertPath);
            vertModule = SpvLoader::createShaderModule(ctx.device, vertCode);
            auto fragCode = SpvLoader::loadSPIRV(fragPath);
            fragModule = SpvLoader::createShaderModule(ctx.device, fragCode);

            std::string vertJsonPath = resolvedBase + ".vert.json";
            std::string fragJsonPath = resolvedBase + ".frag.json";

            if (fs::exists(vertJsonPath) && fs::exists(fragJsonPath)) {
                std::cout << "[info] Loading reflection JSON: " << vertJsonPath << std::endl;
                vertReflection = parseReflectionJson(vertJsonPath);
                std::cout << "[info] Loading reflection JSON: " << fragJsonPath << std::endl;
                fragReflection = parseReflectionJson(fragJsonPath);
            }

            if (sceneSource == "triangle") {
                createPipelineTriangle(ctx);
            } else {
                setupReflectedDescriptors(ctx);
                createPipelinePBR(ctx);
            }
        }
    } else {
        throw std::runtime_error("Unsupported render path for raster renderer: " + renderPath);
    }

    // Bind scene resources to pipeline (push constants)
    if (m_bindlessMode) {
        // Bindless push constants: model (mat4, 64 bytes) + material_index (uint, 4 bytes)
        // The push constant block is 80 bytes (padded) per reflection
        for (auto& pc : vertReflection.push_constants) {
            if (pc.size <= 0) continue;
            pushConstantSize = std::max(pushConstantSize, static_cast<uint32_t>(pc.size));
            pushConstantStageFlags |= VK_SHADER_STAGE_VERTEX_BIT;
        }
        for (auto& pc : fragReflection.push_constants) {
            if (pc.size <= 0) continue;
            pushConstantSize = std::max(pushConstantSize, static_cast<uint32_t>(pc.size));
            pushConstantStageFlags |= VK_SHADER_STAGE_FRAGMENT_BIT;
        }
        pushConstantData.resize(pushConstantSize, 0);

        // Set model matrix to identity initially
        glm::mat4 identity(1.0f);
        if (pushConstantSize >= 64) {
            memcpy(pushConstantData.data(), &identity, sizeof(glm::mat4));
        }
        // material_index at offset 64 defaults to 0 (already zeroed)
    } else if (m_multiPipeline) {
        // In multi-pipeline mode, gather push constants from the first permutation
        glm::vec3 lightDir = glm::normalize(glm::vec3(1.0f, 0.8f, 0.6f));
        if (!m_permutations.empty()) {
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
                            glm::vec3 eye = m_scene->hasSceneBounds() ? m_scene->getAutoEye() : Camera::DEFAULT_EYE;
                            memcpy(pushConstantData.data() + f.offset, &eye, sizeof(glm::vec3));
                        }
                    }
                }
            };
            buildPushData(m_permutations[0].vertRefl);
            buildPushData(m_permutations[0].fragRefl);
        }
    } else {
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
                        glm::vec3 eye = m_scene->hasSceneBounds() ? m_scene->getAutoEye() : Camera::DEFAULT_EYE;
                        memcpy(pushConstantData.data() + f.offset, &eye, sizeof(glm::vec3));
                    }
                }
            }
        };
        buildPushData(vertReflection);
        buildPushData(fragReflection);
    }

    std::cout << "[info] Raster renderer initialized: " << pipelineBase
              << " (path=" << renderPath
              << (m_bindlessMode ? ", bindless" : "")
              << (m_multiPipeline ? ", multi-pipeline" : "")
              << ")" << std::endl;
}

// --------------------------------------------------------------------------
// Multi-pipeline setup: one pipeline per permutation
// --------------------------------------------------------------------------

void RasterRenderer::setupMultiPipeline(VulkanContext& ctx, const ShaderManifest& manifest) {
    namespace fs = std::filesystem;
    m_multiPipeline = true;

    // Group materials by feature set
    auto groups = m_scene->groupMaterialsByFeatures();
    size_t totalMaterials = m_scene->getGltfScene().materials.size();

    // Build materialToPermutation mapping
    m_materialToPermutation.resize(totalMaterials, 0);

    int permIdx = 0;
    for (auto& [suffix, materialIndices] : groups) {
        // Resolve suffix to a manifest permutation
        std::string resolvedSuffix = suffix;

        // Build full path for this permutation
        std::string permBase = m_pipelineBase + resolvedSuffix;
        std::string vertPath = permBase + ".vert.spv";
        std::string fragPath = permBase + ".frag.spv";

        if (!fs::exists(vertPath) || !fs::exists(fragPath)) {
            std::cerr << "[warn] Missing shader for permutation '" << resolvedSuffix
                      << "', falling back to base" << std::endl;
            permBase = m_pipelineBase;
            vertPath = permBase + ".vert.spv";
            fragPath = permBase + ".frag.spv";
            resolvedSuffix = "";
        }

        PermutationPipeline perm;
        perm.suffix = resolvedSuffix;
        perm.basePath = permBase;
        perm.materialIndices = materialIndices;

        // Load shaders
        auto vertCode = SpvLoader::loadSPIRV(vertPath);
        perm.vertModule = SpvLoader::createShaderModule(ctx.device, vertCode);
        auto fragCode = SpvLoader::loadSPIRV(fragPath);
        perm.fragModule = SpvLoader::createShaderModule(ctx.device, fragCode);

        // Load reflection
        std::string vertJsonPath = permBase + ".vert.json";
        std::string fragJsonPath = permBase + ".frag.json";
        if (fs::exists(vertJsonPath)) perm.vertRefl = parseReflectionJson(vertJsonPath);
        if (fs::exists(fragJsonPath)) perm.fragRefl = parseReflectionJson(fragJsonPath);

        // Map materials to this permutation
        for (int mi : materialIndices) {
            m_materialToPermutation[mi] = permIdx;
        }

        m_permutations.push_back(std::move(perm));
        permIdx++;

        std::cout << "[info] Loaded permutation '" << resolvedSuffix << "' from " << permBase
                  << " (" << materialIndices.size() << " material(s))" << std::endl;
    }

    // Create shared set 0 layout (MVP + Light) from first permutation's vertex reflection
    // All permutations share the same set 0 layout
    {
        std::vector<ReflectionData> set0Stages = {m_permutations[0].vertRefl};
        // Only include set 0 bindings from vertex stage
        auto sharedLayouts = createDescriptorSetLayoutsMultiStage(ctx.device, set0Stages);
        auto it = sharedLayouts.find(0);
        if (it != sharedLayouts.end()) {
            m_sharedSet0Layout = it->second;
        }
        // Clean up other layouts (we only want set 0)
        for (auto& [idx, layout] : sharedLayouts) {
            if (idx != 0) vkDestroyDescriptorSetLayout(ctx.device, layout, nullptr);
        }
    }

    // Count total descriptor pool requirements
    std::unordered_map<VkDescriptorType, uint32_t> typeCounts;
    uint32_t maxSets = 1; // set 0

    // Set 0 bindings (shared)
    for (auto& [setIdx, bindings] : m_permutations[0].vertRefl.descriptor_sets) {
        if (setIdx != 0) continue;
        for (auto& b : bindings) {
            typeCounts[bindingTypeToVkDescriptorType(b.type)]++;
        }
    }

    // Per-permutation set 1 bindings
    for (auto& perm : m_permutations) {
        std::vector<ReflectionData> stages = {perm.vertRefl, perm.fragRefl};
        auto mergedBindings = getMergedBindings(stages);
        int fragSetIdx = 1;
        for (auto& b : mergedBindings) {
            if (b.name == "Material") fragSetIdx = b.set;
        }

        // Create set 1 layout for this permutation
        auto permLayouts = createDescriptorSetLayoutsMultiStage(ctx.device, stages);
        auto it = permLayouts.find(fragSetIdx);
        if (it != permLayouts.end()) {
            perm.materialSetLayout = it->second;
        }
        // Clean up other layouts
        for (auto& [idx, layout] : permLayouts) {
            if (idx != fragSetIdx) vkDestroyDescriptorSetLayout(ctx.device, layout, nullptr);
        }

        for (auto& b : mergedBindings) {
            if (b.set != fragSetIdx) continue;
            typeCounts[bindingTypeToVkDescriptorType(b.type)] += static_cast<uint32_t>(perm.materialIndices.size());
        }
        maxSets += static_cast<uint32_t>(perm.materialIndices.size());
    }

    // Create descriptor pool
    std::vector<VkDescriptorPoolSize> poolSizes;
    for (auto& [type, count] : typeCounts) {
        poolSizes.push_back({type, count});
    }

    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = maxSets;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr, &descriptorPool);

    // Allocate and write shared set 0 (MVP + Light)
    if (m_sharedSet0Layout != VK_NULL_HANDLE) {
        VkDescriptorSetAllocateInfo allocSetInfo = {};
        allocSetInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocSetInfo.descriptorPool = descriptorPool;
        allocSetInfo.descriptorSetCount = 1;
        allocSetInfo.pSetLayouts = &m_sharedSet0Layout;
        vkAllocateDescriptorSets(ctx.device, &allocSetInfo, &m_vertexDescSet);

        // Write MVP + Light to set 0
        auto mergedBindings = getMergedBindings({m_permutations[0].vertRefl});
        struct DescWriteInfo { VkDescriptorBufferInfo bufferInfo; VkDescriptorImageInfo imageInfo; };
        std::vector<DescWriteInfo> writeInfos;
        writeInfos.reserve(mergedBindings.size());
        std::vector<VkWriteDescriptorSet> writes;
        writes.reserve(mergedBindings.size());

        for (auto& b : mergedBindings) {
            if (b.set != 0) continue;
            DescWriteInfo info = {};
            VkWriteDescriptorSet w = {};
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = m_vertexDescSet;
            w.dstBinding = static_cast<uint32_t>(b.binding);
            w.descriptorCount = 1;

            if (b.type == "uniform_buffer") {
                w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                if (b.name == "MVP") {
                    info.bufferInfo = {mvpBuffer, 0, static_cast<VkDeviceSize>(b.size > 0 ? b.size : 192)};
                } else if (b.name == "Light") {
                    info.bufferInfo = {lightBuffer, 0, static_cast<VkDeviceSize>(b.size > 0 ? b.size : 32)};
                } else continue;
                writeInfos.push_back(info);
                w.pBufferInfo = &writeInfos.back().bufferInfo;
            } else continue;
            writes.push_back(w);
        }
        if (!writes.empty()) {
            vkUpdateDescriptorSets(ctx.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        }
    }

    // For each permutation: allocate per-material descriptor sets and create pipeline
    for (auto& perm : m_permutations) {
        std::vector<ReflectionData> stages = {perm.vertRefl, perm.fragRefl};
        auto mergedBindings = getMergedBindings(stages);
        int fragSetIdx = 1;
        for (auto& b : mergedBindings) {
            if (b.name == "Material") fragSetIdx = b.set;
        }

        perm.perMaterialDescSets.resize(perm.materialIndices.size());
        perm.perMaterialUBOs.resize(perm.materialIndices.size());
        perm.perMaterialAllocations.resize(perm.materialIndices.size());

        for (size_t i = 0; i < perm.materialIndices.size(); i++) {
            int mi = perm.materialIndices[i];

            // Create Material UBO
            MaterialUBOData materialData{};
            if (mi < static_cast<int>(m_scene->getGltfScene().materials.size())) {
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
                            &perm.perMaterialUBOs[i], &perm.perMaterialAllocations[i], nullptr);

            void* mapped;
            vmaMapMemory(ctx.allocator, perm.perMaterialAllocations[i], &mapped);
            memcpy(mapped, &materialData, sizeof(MaterialUBOData));
            vmaUnmapMemory(ctx.allocator, perm.perMaterialAllocations[i]);

            // Allocate descriptor set
            VkDescriptorSetAllocateInfo allocSetInfo = {};
            allocSetInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            allocSetInfo.descriptorPool = descriptorPool;
            allocSetInfo.descriptorSetCount = 1;
            allocSetInfo.pSetLayouts = &perm.materialSetLayout;
            vkAllocateDescriptorSets(ctx.device, &allocSetInfo, &perm.perMaterialDescSets[i]);

            // Write descriptors
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
                w.dstSet = perm.perMaterialDescSets[i];
                w.dstBinding = static_cast<uint32_t>(b.binding);
                w.descriptorCount = 1;

                if (b.type == "uniform_buffer") {
                    w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                    if (b.name == "Material") {
                        info.bufferInfo = {perm.perMaterialUBOs[i], 0, sizeof(MaterialUBOData)};
                    } else if (b.name == "Light" && lightBuffer != VK_NULL_HANDLE) {
                        info.bufferInfo = {lightBuffer, 0, static_cast<VkDeviceSize>(b.size > 0 ? b.size : 32)};
                    } else continue;
                    writeInfos.push_back(info);
                    w.pBufferInfo = &writeInfos.back().bufferInfo;
                } else if (b.type == "sampler") {
                    w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
                    auto& iblTextures = m_scene->getIBLTextures();
                    auto iblIt = iblTextures.find(b.name);
                    if (iblIt != iblTextures.end()) {
                        info.imageInfo = {}; info.imageInfo.sampler = iblIt->second.sampler;
                    } else {
                        auto& tex = m_scene->getTextureForBinding(b.name);
                        info.imageInfo = {}; info.imageInfo.sampler = tex.sampler;
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
                        if (mi < static_cast<int>(perMatTextures.size())) {
                            auto it = perMatTextures[mi].find(b.name);
                            if (it != perMatTextures[mi].end())
                                texPtr = const_cast<GPUTexture*>(&it->second);
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
                } else continue;
                writes.push_back(w);
            }

            if (!writes.empty()) {
                vkUpdateDescriptorSets(ctx.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
            }
        }

        // Create pipeline layout: set 0 (shared) + set 1 (per-permutation)
        std::vector<VkDescriptorSetLayout> layouts;
        if (m_sharedSet0Layout != VK_NULL_HANDLE) layouts.push_back(m_sharedSet0Layout);
        if (perm.materialSetLayout != VK_NULL_HANDLE) layouts.push_back(perm.materialSetLayout);

        // Collect push constant ranges
        std::vector<VkPushConstantRange> pushRanges;
        for (auto& pc : perm.vertRefl.push_constants) {
            VkPushConstantRange range = {};
            range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
            range.offset = 0;
            range.size = static_cast<uint32_t>(pc.size);
            pushRanges.push_back(range);
        }
        for (auto& pc : perm.fragRefl.push_constants) {
            VkPushConstantRange range = {};
            range.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            range.offset = 0;
            range.size = static_cast<uint32_t>(pc.size);
            pushRanges.push_back(range);
        }

        VkPipelineLayoutCreateInfo layoutInfo = {};
        layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutInfo.setLayoutCount = static_cast<uint32_t>(layouts.size());
        layoutInfo.pSetLayouts = layouts.data();
        layoutInfo.pushConstantRangeCount = static_cast<uint32_t>(pushRanges.size());
        layoutInfo.pPushConstantRanges = pushRanges.data();
        vkCreatePipelineLayout(ctx.device, &layoutInfo, nullptr, &perm.pipelineLayout);

        // Create graphics pipeline (same as createPipelinePBR but per-permutation)
        VkPipelineShaderStageCreateInfo shaderStages[2] = {};
        shaderStages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        shaderStages[0].module = perm.vertModule;
        shaderStages[0].pName = "main";
        shaderStages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        shaderStages[1].module = perm.fragModule;
        shaderStages[1].pName = "main";

        int bufferStride = (m_scene && m_scene->hasGltfScene()) ? 48 : 0;
        auto reflectedInput = createReflectedVertexInput(perm.vertRefl, bufferStride);

        VkPipelineVertexInputStateCreateInfo vertexInput = {};
        vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInput.vertexBindingDescriptionCount = 1;
        vertexInput.pVertexBindingDescriptions = &reflectedInput.binding;
        vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(reflectedInput.attributes.size());
        vertexInput.pVertexAttributeDescriptions = reflectedInput.attributes.data();

        VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkViewport viewport = {0, 0, (float)renderWidth, (float)renderHeight, 0, 1};
        VkRect2D scissor = {{0, 0}, {renderWidth, renderHeight}};
        VkPipelineViewportStateCreateInfo viewportState = {};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1; viewportState.pViewports = &viewport;
        viewportState.scissorCount = 1; viewportState.pScissors = &scissor;

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

        VkPipelineColorBlendAttachmentState colorBlendAtt = {};
        colorBlendAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                       VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo colorBlending = {};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAtt;

        VkGraphicsPipelineCreateInfo pipeInfo = {};
        pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipeInfo.stageCount = 2;
        pipeInfo.pStages = shaderStages;
        pipeInfo.pVertexInputState = &vertexInput;
        pipeInfo.pInputAssemblyState = &inputAssembly;
        pipeInfo.pViewportState = &viewportState;
        pipeInfo.pRasterizationState = &rasterizer;
        pipeInfo.pMultisampleState = &multisampling;
        pipeInfo.pDepthStencilState = &depthStencil;
        pipeInfo.pColorBlendState = &colorBlending;
        pipeInfo.layout = perm.pipelineLayout;
        pipeInfo.renderPass = renderPass;

        if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &pipeInfo,
                                      nullptr, &perm.pipeline) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create permutation pipeline: " + perm.suffix);
        }
    }

    std::cout << "[info] Multi-pipeline setup complete: " << m_permutations.size()
              << " permutation(s) for " << totalMaterials << " material(s)" << std::endl;
}

// --------------------------------------------------------------------------
// Record draw commands (shared between render() and renderToSwapchain())
// --------------------------------------------------------------------------

void RasterRenderer::recordDrawCommands(VkCommandBuffer cmd) {
    const std::string& sceneSource = m_scene->getSceneSource();

    if (sceneSource == "triangle") {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        if (pushConstantSize > 0 && pushConstantStageFlags != 0) {
            vkCmdPushConstants(cmd, pipelineLayout, pushConstantStageFlags,
                               0, pushConstantSize, pushConstantData.data());
        }
        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &triangleVB, &offset);
        vkCmdDraw(cmd, 3, 1, 0, 0);
    } else if (sceneSource == "fullscreen" || m_renderPath == "fullscreen") {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        if (pushConstantSize > 0 && pushConstantStageFlags != 0) {
            vkCmdPushConstants(cmd, pipelineLayout, pushConstantStageFlags,
                               0, pushConstantSize, pushConstantData.data());
        }
        vkCmdDraw(cmd, 3, 1, 0, 0);
    } else if (m_bindlessMode) {
        // Bindless mode: single pipeline, single set of descriptors, push material_index per draw
        const auto& mesh = m_scene->getMesh();
        VkDeviceSize vbOffset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &mesh.vertexBuffer, &vbOffset);
        vkCmdBindIndexBuffer(cmd, mesh.indexBuffer, 0, VK_INDEX_TYPE_UINT32);

        // Bind pipeline once
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

        // Bind all descriptor sets (0, 1, 2) once
        int maxSet = -1;
        for (auto& [idx, _] : reflectedDescSets) { maxSet = std::max(maxSet, idx); }

        std::vector<VkDescriptorSet> allSets;
        for (int i = 0; i <= maxSet; i++) {
            auto it = reflectedDescSets.find(i);
            if (it != reflectedDescSets.end()) {
                allSets.push_back(it->second);
            } else {
                // Should not happen with proper reflection, but guard against it
                allSets.push_back(VK_NULL_HANDLE);
            }
        }
        if (!allSets.empty()) {
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    pipelineLayout, 0,
                                    static_cast<uint32_t>(allSets.size()),
                                    allSets.data(), 0, nullptr);
        }

        // Draw each range with push constant { model_matrix, material_index }
        const auto& drawRanges = m_scene->getDrawRanges();
        // Prepare push constant buffer (model mat4 at offset 0, material_index uint at offset 64)
        std::vector<uint8_t> pcData(pushConstantSize, 0);
        glm::mat4 modelMatrix(1.0f);  // identity - transforms already baked into vertices
        if (pushConstantSize >= 64) {
            memcpy(pcData.data(), &modelMatrix, sizeof(glm::mat4));
        }

        for (auto& range : drawRanges) {
            // Set material_index at offset 64
            if (pushConstantSize >= 68) {
                uint32_t matIdx = static_cast<uint32_t>(range.materialIndex);
                memcpy(pcData.data() + 64, &matIdx, sizeof(uint32_t));
            }

            vkCmdPushConstants(cmd, pipelineLayout, pushConstantStageFlags,
                               0, pushConstantSize, pcData.data());
            vkCmdDrawIndexed(cmd, range.indexCount, 1, range.indexOffset, 0, 0);
        }
    } else if (m_multiPipeline) {
        // Multi-pipeline mode: switch pipeline per permutation
        const auto& mesh = m_scene->getMesh();
        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &mesh.vertexBuffer, &offset);
        vkCmdBindIndexBuffer(cmd, mesh.indexBuffer, 0, VK_INDEX_TYPE_UINT32);

        // Bind shared set 0 (MVP + Light)
        if (m_vertexDescSet != VK_NULL_HANDLE && !m_permutations.empty()) {
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    m_permutations[0].pipelineLayout, 0, 1,
                                    &m_vertexDescSet, 0, nullptr);
        }

        const auto& drawRanges = m_scene->getDrawRanges();
        int currentPermIdx = -1;

        for (auto& range : drawRanges) {
            int matIdx = range.materialIndex;
            int permIdx = (matIdx < static_cast<int>(m_materialToPermutation.size()))
                ? m_materialToPermutation[matIdx] : 0;
            auto& perm = m_permutations[permIdx];

            // Switch pipeline if permutation changed
            if (permIdx != currentPermIdx) {
                currentPermIdx = permIdx;
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, perm.pipeline);
                if (pushConstantSize > 0 && pushConstantStageFlags != 0) {
                    vkCmdPushConstants(cmd, perm.pipelineLayout, pushConstantStageFlags,
                                       0, pushConstantSize, pushConstantData.data());
                }
                // Re-bind set 0 after pipeline change
                if (m_vertexDescSet != VK_NULL_HANDLE) {
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                            perm.pipelineLayout, 0, 1,
                                            &m_vertexDescSet, 0, nullptr);
                }
            }

            // Find per-material descriptor set index within this permutation
            int localMatIdx = -1;
            for (size_t j = 0; j < perm.materialIndices.size(); j++) {
                if (perm.materialIndices[j] == matIdx) { localMatIdx = static_cast<int>(j); break; }
            }
            if (localMatIdx >= 0) {
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                        perm.pipelineLayout, 1, 1,
                                        &perm.perMaterialDescSets[localMatIdx], 0, nullptr);
            }

            vkCmdDrawIndexed(cmd, range.indexCount, 1, range.indexOffset, 0, 0);
        }
    } else {
        // Single-pipeline mode (original behavior)
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        if (pushConstantSize > 0 && pushConstantStageFlags != 0) {
            vkCmdPushConstants(cmd, pipelineLayout, pushConstantStageFlags,
                               0, pushConstantSize, pushConstantData.data());
        }

        const auto& mesh = m_scene->getMesh();
        if (m_vertexDescSet != VK_NULL_HANDLE) {
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    pipelineLayout, 0, 1, &m_vertexDescSet, 0, nullptr);
        }

        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &mesh.vertexBuffer, &offset);
        vkCmdBindIndexBuffer(cmd, mesh.indexBuffer, 0, VK_INDEX_TYPE_UINT32);

        const auto& drawRanges = m_scene->getDrawRanges();
        if (!drawRanges.empty() && !m_perMaterialDescSets.empty()) {
            int currentMat = -1;
            for (auto& range : drawRanges) {
                if (range.materialIndex != currentMat) {
                    currentMat = range.materialIndex;
                    int matIdx = std::min(currentMat, static_cast<int>(m_perMaterialDescSets.size()) - 1);
                    if (matIdx >= 0) {
                        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                                pipelineLayout, 1, 1, &m_perMaterialDescSets[matIdx], 0, nullptr);
                    }
                }
                vkCmdDrawIndexed(cmd, range.indexCount, 1, range.indexOffset, 0, 0);
            }
        } else {
            int maxSet = -1;
            for (auto& [idx, _] : reflectedDescSets) { maxSet = std::max(maxSet, idx); }
            for (int i = 0; i <= maxSet; i++) {
                auto it = reflectedDescSets.find(i);
                if (it != reflectedDescSets.end()) {
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                            pipelineLayout, static_cast<uint32_t>(i),
                                            1, &it->second, 0, nullptr);
                }
            }
            vkCmdDrawIndexed(cmd, mesh.indexCount, 1, 0, 0, 0);
        }
    }
}

// --------------------------------------------------------------------------
// Rendering
// --------------------------------------------------------------------------

void RasterRenderer::render(VulkanContext& ctx) {
    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();

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

    ctx.cmdBeginLabel(cmd, "Raster Pass", 0.2f, 0.8f, 0.2f, 1.0f);
    vkCmdBeginRenderPass(cmd, &rpBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
    recordDrawCommands(cmd);
    vkCmdEndRenderPass(cmd);
    ctx.cmdEndLabel(cmd);
    ctx.endSingleTimeCommands(cmd);
}

// --------------------------------------------------------------------------
// Blit to swapchain
// --------------------------------------------------------------------------

void RasterRenderer::blitToSwapchain(VulkanContext& ctx, VkCommandBuffer cmd,
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
}

// --------------------------------------------------------------------------
// Interactive mode: render to swapchain
// --------------------------------------------------------------------------

void RasterRenderer::renderToSwapchain(VulkanContext& ctx,
                                       VkImage swapImage, VkImageView swapView,
                                       VkFormat swapFormat, VkExtent2D extent,
                                       VkSemaphore waitSem, VkSemaphore signalSem,
                                       VkFence fence) {
    // Free previous frame's command buffer (fence was already waited on by caller)
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

    ctx.cmdBeginLabel(cmd, "Raster Pass", 0.2f, 0.8f, 0.2f, 1.0f);
    vkCmdBeginRenderPass(cmd, &rpBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
    recordDrawCommands(cmd);
    vkCmdEndRenderPass(cmd);
    ctx.cmdEndLabel(cmd);

    // --- Blit offscreen -> swapchain ---
    blitToSwapchain(ctx, cmd, swapImage, extent);

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
// Orbit camera: update MVP + Light uniform buffers per frame
// --------------------------------------------------------------------------

void RasterRenderer::updateCamera(VulkanContext& ctx, glm::vec3 eye, glm::vec3 target,
                                  glm::vec3 up, float fovY, float aspect,
                                  float nearPlane, float farPlane) {
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

    // Cleanup multi-pipeline resources
    for (auto& perm : m_permutations) {
        if (perm.pipeline) vkDestroyPipeline(ctx.device, perm.pipeline, nullptr);
        if (perm.pipelineLayout) vkDestroyPipelineLayout(ctx.device, perm.pipelineLayout, nullptr);
        if (perm.vertModule) vkDestroyShaderModule(ctx.device, perm.vertModule, nullptr);
        if (perm.fragModule) vkDestroyShaderModule(ctx.device, perm.fragModule, nullptr);
        if (perm.materialSetLayout) vkDestroyDescriptorSetLayout(ctx.device, perm.materialSetLayout, nullptr);
        for (size_t i = 0; i < perm.perMaterialUBOs.size(); i++) {
            if (perm.perMaterialUBOs[i]) {
                vmaDestroyBuffer(ctx.allocator, perm.perMaterialUBOs[i], perm.perMaterialAllocations[i]);
            }
        }
    }
    m_permutations.clear();
    if (m_sharedSet0Layout) {
        vkDestroyDescriptorSetLayout(ctx.device, m_sharedSet0Layout, nullptr);
        m_sharedSet0Layout = VK_NULL_HANDLE;
    }

    // Cleanup single-pipeline resources
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
    if (m_materialBuffer) vmaDestroyBuffer(ctx.allocator, m_materialBuffer, m_materialAllocation);

    // Cleanup bindless resources
    if (m_bindlessMaterials.buffer) {
        vmaDestroyBuffer(ctx.allocator, m_bindlessMaterials.buffer, m_bindlessMaterials.allocation);
        m_bindlessMaterials.buffer = VK_NULL_HANDLE;
    }
    m_bindlessTextures = {};  // imageViews/samplers are owned by SceneManager
    m_bindlessMode = false;

    for (size_t i = 0; i < m_perMaterialBuffers.size(); i++) {
        if (m_perMaterialBuffers[i]) {
            vmaDestroyBuffer(ctx.allocator, m_perMaterialBuffers[i], m_perMaterialAllocations[i]);
        }
    }
    m_perMaterialBuffers.clear();
    m_perMaterialAllocations.clear();
    m_perMaterialDescSets.clear();

    if (descriptorPool) vkDestroyDescriptorPool(ctx.device, descriptorPool, nullptr);
    for (auto& [_, layout] : reflectedSetLayouts) {
        vkDestroyDescriptorSetLayout(ctx.device, layout, nullptr);
    }
    reflectedSetLayouts.clear();
    reflectedDescSets.clear();
}
