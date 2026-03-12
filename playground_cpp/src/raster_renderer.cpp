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
// Embedded shadow depth vertex shader SPIR-V
// --------------------------------------------------------------------------
// Minimal vertex shader:
//   #version 450
//   layout(push_constant) uniform PC { mat4 mvp; };
//   layout(location = 0) in vec3 inPosition;
//   void main() { gl_Position = mvp * vec4(inPosition, 1.0); }
//
// Compiled with: glslangValidator -V -e main shadow.vert -o shadow.vert.spv
// Then xxd -i to get C array. This is the pre-compiled SPIR-V binary.
static const uint32_t shadow_vert_spv[] = {
    0x07230203, 0x00010000, 0x0008000b, 0x00000023,
    0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e,
    0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x0007000f, 0x00000000, 0x00000004, 0x6e69616d,
    0x00000000, 0x0000000d, 0x00000019, 0x00030003,
    0x00000002, 0x000001c2, 0x00040005, 0x00000004,
    0x6e69616d, 0x00000000, 0x00060005, 0x0000000b,
    0x505f6c67, 0x65567265, 0x78657472, 0x00000000,
    0x00060006, 0x0000000b, 0x00000000, 0x505f6c67,
    0x7469736f, 0x006e6f69, 0x00070006, 0x0000000b,
    0x00000001, 0x505f6c67, 0x746e696f, 0x657a6953,
    0x00000000, 0x00070006, 0x0000000b, 0x00000002,
    0x435f6c67, 0x4470696c, 0x61747369, 0x0065636e,
    0x00070006, 0x0000000b, 0x00000003, 0x435f6c67,
    0x446c6c75, 0x61747369, 0x0065636e, 0x00030005,
    0x0000000d, 0x00000000, 0x00030005, 0x00000011,
    0x00004350, 0x00040006, 0x00000011, 0x00000000,
    0x0070766d, 0x00030005, 0x00000013, 0x00000000,
    0x00050005, 0x00000019, 0x6f506e69, 0x69746973,
    0x00006e6f, 0x00030047, 0x0000000b, 0x00000002,
    0x00050048, 0x0000000b, 0x00000000, 0x0000000b,
    0x00000000, 0x00050048, 0x0000000b, 0x00000001,
    0x0000000b, 0x00000001, 0x00050048, 0x0000000b,
    0x00000002, 0x0000000b, 0x00000003, 0x00050048,
    0x0000000b, 0x00000003, 0x0000000b, 0x00000004,
    0x00030047, 0x00000011, 0x00000002, 0x00040048,
    0x00000011, 0x00000000, 0x00000005, 0x00050048,
    0x00000011, 0x00000000, 0x00000007, 0x00000010,
    0x00050048, 0x00000011, 0x00000000, 0x00000023,
    0x00000000, 0x00040047, 0x00000019, 0x0000001e,
    0x00000000, 0x00020013, 0x00000002, 0x00030021,
    0x00000003, 0x00000002, 0x00030016, 0x00000006,
    0x00000020, 0x00040017, 0x00000007, 0x00000006,
    0x00000004, 0x00040015, 0x00000008, 0x00000020,
    0x00000000, 0x0004002b, 0x00000008, 0x00000009,
    0x00000001, 0x0004001c, 0x0000000a, 0x00000006,
    0x00000009, 0x0006001e, 0x0000000b, 0x00000007,
    0x00000006, 0x0000000a, 0x0000000a, 0x00040020,
    0x0000000c, 0x00000003, 0x0000000b, 0x0004003b,
    0x0000000c, 0x0000000d, 0x00000003, 0x00040015,
    0x0000000e, 0x00000020, 0x00000001, 0x0004002b,
    0x0000000e, 0x0000000f, 0x00000000, 0x00040018,
    0x00000010, 0x00000007, 0x00000004, 0x0003001e,
    0x00000011, 0x00000010, 0x00040020, 0x00000012,
    0x00000009, 0x00000011, 0x0004003b, 0x00000012,
    0x00000013, 0x00000009, 0x00040020, 0x00000014,
    0x00000009, 0x00000010, 0x00040017, 0x00000017,
    0x00000006, 0x00000003, 0x00040020, 0x00000018,
    0x00000001, 0x00000017, 0x0004003b, 0x00000018,
    0x00000019, 0x00000001, 0x0004002b, 0x00000006,
    0x0000001b, 0x3f800000, 0x00040020, 0x00000021,
    0x00000003, 0x00000007, 0x00050036, 0x00000002,
    0x00000004, 0x00000000, 0x00000003, 0x000200f8,
    0x00000005, 0x00050041, 0x00000014, 0x00000015,
    0x00000013, 0x0000000f, 0x0004003d, 0x00000010,
    0x00000016, 0x00000015, 0x0004003d, 0x00000017,
    0x0000001a, 0x00000019, 0x00050051, 0x00000006,
    0x0000001c, 0x0000001a, 0x00000000, 0x00050051,
    0x00000006, 0x0000001d, 0x0000001a, 0x00000001,
    0x00050051, 0x00000006, 0x0000001e, 0x0000001a,
    0x00000002, 0x00070050, 0x00000007, 0x0000001f,
    0x0000001c, 0x0000001d, 0x0000001e, 0x0000001b,
    0x00050091, 0x00000007, 0x00000020, 0x00000016,
    0x0000001f, 0x00050041, 0x00000021, 0x00000022,
    0x0000000d, 0x0000000f, 0x0003003e, 0x00000022,
    0x00000020, 0x000100fd, 0x00010038
};

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
        depthInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

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
        depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
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
// Multi-light resource setup (called AFTER fragReflection is loaded)
// --------------------------------------------------------------------------

void RasterRenderer::setupMultiLightResources(VulkanContext& ctx) {
    // Detect from fragment reflection whether the shader uses a "lights" storage buffer
    m_hasMultiLight = false;
    for (auto& [setIdx, bindings] : fragReflection.descriptor_sets) {
        for (auto& b : bindings) {
            if (b.name == "lights" && b.type == "storage_buffer") {
                m_hasMultiLight = true;
                break;
            }
        }
        if (m_hasMultiLight) break;
    }

    if (m_hasMultiLight && m_scene) {
        // Create lights SSBO from packed scene light data
        auto lightBuf = m_scene->packLightsBuffer();
        if (lightBuf.empty()) {
            // Add a dummy light to avoid empty buffer
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

        // Create SceneLight UBO: vec3 view_pos at offset 0, int light_count at offset 12
        // Must match SPIR-V layout (16 bytes total, no padding between fields)
        struct SceneLightUBO {
            glm::vec3 viewPos;      // offset 0, size 12
            int32_t lightCount;     // offset 12, size 4
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
// Shadow map infrastructure setup
// --------------------------------------------------------------------------

void RasterRenderer::setupShadowMaps(VulkanContext& ctx) {
    // 1. Create 2D array depth image (MAX_SHADOW_MAPS layers)
    VkImageCreateInfo imageInfo = {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = VK_FORMAT_D32_SFLOAT;
    imageInfo.extent = {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = MAX_SHADOW_MAPS;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    VkResult res = vmaCreateImage(ctx.allocator, &imageInfo, &allocInfo,
                                   &m_shadowImage, &m_shadowImageAlloc, nullptr);
    if (res != VK_SUCCESS) {
        std::cerr << "[warn] Failed to create shadow map image, disabling shadows" << std::endl;
        m_hasShadows = false;
        return;
    }

    // 2. Create array image view (for binding as sampler2DArray in main pass)
    VkImageViewCreateInfo arrayViewInfo = {};
    arrayViewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    arrayViewInfo.image = m_shadowImage;
    arrayViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    arrayViewInfo.format = VK_FORMAT_D32_SFLOAT;
    arrayViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    arrayViewInfo.subresourceRange.baseMipLevel = 0;
    arrayViewInfo.subresourceRange.levelCount = 1;
    arrayViewInfo.subresourceRange.baseArrayLayer = 0;
    arrayViewInfo.subresourceRange.layerCount = MAX_SHADOW_MAPS;
    vkCreateImageView(ctx.device, &arrayViewInfo, nullptr, &m_shadowImageView);

    // 3. Create per-layer image views (for framebuffer attachments)
    m_shadowLayerViews.resize(MAX_SHADOW_MAPS);
    for (uint32_t i = 0; i < MAX_SHADOW_MAPS; i++) {
        VkImageViewCreateInfo layerViewInfo = {};
        layerViewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        layerViewInfo.image = m_shadowImage;
        layerViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        layerViewInfo.format = VK_FORMAT_D32_SFLOAT;
        layerViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        layerViewInfo.subresourceRange.baseMipLevel = 0;
        layerViewInfo.subresourceRange.levelCount = 1;
        layerViewInfo.subresourceRange.baseArrayLayer = i;
        layerViewInfo.subresourceRange.layerCount = 1;
        vkCreateImageView(ctx.device, &layerViewInfo, nullptr, &m_shadowLayerViews[i]);
    }

    // 4. Create depth-only render pass
    VkAttachmentDescription depthAtt = {};
    depthAtt.format = VK_FORMAT_D32_SFLOAT;
    depthAtt.samples = VK_SAMPLE_COUNT_1_BIT;
    depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAtt.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

    VkAttachmentReference depthRef = {};
    depthRef.attachment = 0;
    depthRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 0;
    subpass.pColorAttachments = nullptr;
    subpass.pDepthStencilAttachment = &depthRef;

    VkSubpassDependency dependencies[2] = {};
    // EXTERNAL -> 0: ensure prior reads complete before depth writes
    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0;
    dependencies[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dependencies[0].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependencies[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    dependencies[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    // 0 -> EXTERNAL: ensure depth writes complete before shader reads
    dependencies[1].srcSubpass = 0;
    dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[1].srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dependencies[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    VkRenderPassCreateInfo rpInfo = {};
    rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpInfo.attachmentCount = 1;
    rpInfo.pAttachments = &depthAtt;
    rpInfo.subpassCount = 1;
    rpInfo.pSubpasses = &subpass;
    rpInfo.dependencyCount = 2;
    rpInfo.pDependencies = dependencies;

    if (vkCreateRenderPass(ctx.device, &rpInfo, nullptr, &m_shadowRenderPass) != VK_SUCCESS) {
        std::cerr << "[warn] Failed to create shadow render pass" << std::endl;
        return;
    }

    // 5. Create per-layer framebuffers
    m_shadowFramebuffers.resize(MAX_SHADOW_MAPS);
    for (uint32_t i = 0; i < MAX_SHADOW_MAPS; i++) {
        VkFramebufferCreateInfo fbInfo = {};
        fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fbInfo.renderPass = m_shadowRenderPass;
        fbInfo.attachmentCount = 1;
        fbInfo.pAttachments = &m_shadowLayerViews[i];
        fbInfo.width = SHADOW_MAP_SIZE;
        fbInfo.height = SHADOW_MAP_SIZE;
        fbInfo.layers = 1;
        vkCreateFramebuffer(ctx.device, &fbInfo, nullptr, &m_shadowFramebuffers[i]);
    }

    // 6. Create comparison sampler (for shadow sampling with PCF)
    VkSamplerCreateInfo samplerInfo = {};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    samplerInfo.compareEnable = VK_TRUE;
    samplerInfo.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 1.0f;
    vkCreateSampler(ctx.device, &samplerInfo, nullptr, &m_shadowSampler);

    // 7. Create shadow pipeline layout (push constant: mat4 lightVP = 64 bytes)
    VkPushConstantRange pushRange = {};
    pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(glm::mat4);  // 64 bytes for the MVP

    VkPipelineLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 0;
    layoutInfo.pSetLayouts = nullptr;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushRange;
    vkCreatePipelineLayout(ctx.device, &layoutInfo, nullptr, &m_shadowPipelineLayout);

    // Create shadow vertex shader module from embedded SPIR-V
    VkShaderModuleCreateInfo shaderInfo = {};
    shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderInfo.codeSize = sizeof(shadow_vert_spv);
    shaderInfo.pCode = shadow_vert_spv;
    vkCreateShaderModule(ctx.device, &shaderInfo, nullptr, &m_shadowVertModule);

    // Create shadow pipeline (depth-only, vertex-only for shadow pass)
    VkPipelineShaderStageCreateInfo shaderStage = {};
    shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    shaderStage.module = m_shadowVertModule;
    shaderStage.pName = "main";

    // Vertex input: only position at location 0 (vec3, offset 0)
    // The vertex buffer stride depends on the scene format (32 or 48 bytes)
    int vbStride = (m_scene && m_scene->hasGltfScene()) ? 48 : 32;

    VkVertexInputBindingDescription binding = {};
    binding.binding = 0;
    binding.stride = static_cast<uint32_t>(vbStride);
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription posAttr = {};
    posAttr.location = 0;
    posAttr.binding = 0;
    posAttr.format = VK_FORMAT_R32G32B32_SFLOAT;
    posAttr.offset = 0;

    VkPipelineVertexInputStateCreateInfo vertexInput = {};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount = 1;
    vertexInput.pVertexBindingDescriptions = &binding;
    vertexInput.vertexAttributeDescriptionCount = 1;
    vertexInput.pVertexAttributeDescriptions = &posAttr;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport viewport = {0, 0, (float)SHADOW_MAP_SIZE, (float)SHADOW_MAP_SIZE, 0, 1};
    VkRect2D scissor = {{0, 0}, {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE}};
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
    rasterizer.depthBiasEnable = VK_TRUE;
    rasterizer.depthBiasConstantFactor = 1.25f;
    rasterizer.depthBiasSlopeFactor = 1.75f;

    VkPipelineMultisampleStateCreateInfo multisampling = {};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil = {};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

    // No color blend state needed (depth-only)
    VkPipelineColorBlendStateCreateInfo colorBlending = {};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 0;

    VkGraphicsPipelineCreateInfo pipeInfo = {};
    pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeInfo.stageCount = 1;  // vertex only
    pipeInfo.pStages = &shaderStage;
    pipeInfo.pVertexInputState = &vertexInput;
    pipeInfo.pInputAssemblyState = &inputAssembly;
    pipeInfo.pViewportState = &viewportState;
    pipeInfo.pRasterizationState = &rasterizer;
    pipeInfo.pMultisampleState = &multisampling;
    pipeInfo.pDepthStencilState = &depthStencil;
    pipeInfo.pColorBlendState = &colorBlending;
    pipeInfo.layout = m_shadowPipelineLayout;
    pipeInfo.renderPass = m_shadowRenderPass;
    pipeInfo.subpass = 0;

    if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &pipeInfo,
                                  nullptr, &m_shadowPipeline) != VK_SUCCESS) {
        std::cerr << "[warn] Failed to create shadow pipeline" << std::endl;
        return;
    }

    // 8. Allocate shadow matrices SSBO (80 bytes per entry * MAX_SHADOW_MAPS)
    VkBufferCreateInfo ssboBufInfo = {};
    ssboBufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    ssboBufInfo.size = MAX_SHADOW_MAPS * 80;  // 80 bytes per ShadowEntry
    ssboBufInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    VmaAllocationCreateInfo ssboAllocInfo = {};
    ssboAllocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;

    vmaCreateBuffer(ctx.allocator, &ssboBufInfo, &ssboAllocInfo,
                    &m_shadowMatricesSSBO, &m_shadowMatricesSSBOAlloc, nullptr);

    // Transition shadow image to shader read initially (all layers)
    VkCommandBuffer initCmd = ctx.beginSingleTimeCommands();
    VkImageMemoryBarrier initBarrier = {};
    initBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    initBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    initBarrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    initBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    initBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    initBarrier.image = m_shadowImage;
    initBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    initBarrier.subresourceRange.baseMipLevel = 0;
    initBarrier.subresourceRange.levelCount = 1;
    initBarrier.subresourceRange.baseArrayLayer = 0;
    initBarrier.subresourceRange.layerCount = MAX_SHADOW_MAPS;
    initBarrier.srcAccessMask = 0;
    initBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(initCmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &initBarrier);
    ctx.endSingleTimeCommands(initCmd);

    m_hasShadows = true;
    std::cout << "[info] Shadow map infrastructure created: " << MAX_SHADOW_MAPS
              << " layers at " << SHADOW_MAP_SIZE << "x" << SHADOW_MAP_SIZE << std::endl;
}

// --------------------------------------------------------------------------
// Render shadow maps for all shadow-casting lights
// --------------------------------------------------------------------------

void RasterRenderer::renderShadowMaps(VulkanContext& ctx, VkCommandBuffer cmd) {
    if (!m_hasShadows || !m_scene) return;

    // Compute shadow data using current camera parameters
    // Read back current MVP to extract view/proj
    struct MVPData { glm::mat4 model, view, projection; };
    MVPData mvp;
    void* mapped;
    vmaMapMemory(ctx.allocator, mvpAllocation, &mapped);
    memcpy(&mvp, mapped, sizeof(MVPData));
    vmaUnmapMemory(ctx.allocator, mvpAllocation);

    float nearClip = 0.1f;
    float farClip = m_scene->hasSceneBounds() ? m_scene->getAutoFar() : 100.0f;
    m_scene->computeShadowData(mvp.view, mvp.projection, nearClip, farClip);

    m_numShadowMaps = m_scene->getShadowMapCount();
    if (m_numShadowMaps == 0) return;

    const auto& shadowEntries = m_scene->getShadowEntries();
    const auto& mesh = m_scene->getMesh();
    if (mesh.vertexBuffer == VK_NULL_HANDLE) return;

    ctx.cmdBeginLabel(cmd, "Shadow Pass", 0.5f, 0.3f, 0.1f, 1.0f);

    for (int i = 0; i < m_numShadowMaps && i < static_cast<int>(MAX_SHADOW_MAPS); i++) {
        // Begin shadow render pass for this layer
        VkClearValue depthClear = {};
        depthClear.depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo rpBeginInfo = {};
        rpBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpBeginInfo.renderPass = m_shadowRenderPass;
        rpBeginInfo.framebuffer = m_shadowFramebuffers[i];
        rpBeginInfo.renderArea.offset = {0, 0};
        rpBeginInfo.renderArea.extent = {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE};
        rpBeginInfo.clearValueCount = 1;
        rpBeginInfo.pClearValues = &depthClear;

        vkCmdBeginRenderPass(cmd, &rpBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

        // Bind shadow pipeline
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_shadowPipeline);

        // Bind vertex/index buffers
        VkDeviceSize vbOffset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &mesh.vertexBuffer, &vbOffset);
        vkCmdBindIndexBuffer(cmd, mesh.indexBuffer, 0, VK_INDEX_TYPE_UINT32);

        // Draw all meshes with lightVP as push constant
        // lightVP = shadow_entry.viewProjection (model is identity since transforms are baked)
        glm::mat4 lightMVP = shadowEntries[i].viewProjection;

        const auto& drawRanges = m_scene->getDrawRanges();
        int totalIndices = 0;
        if (!drawRanges.empty()) {
            for (auto& range : drawRanges) {
                vkCmdPushConstants(cmd, m_shadowPipelineLayout,
                                   VK_SHADER_STAGE_VERTEX_BIT, 0,
                                   sizeof(glm::mat4), &lightMVP);
                vkCmdDrawIndexed(cmd, range.indexCount, 1, range.indexOffset, 0, 0);
                totalIndices += range.indexCount;
            }
        } else {
            vkCmdPushConstants(cmd, m_shadowPipelineLayout,
                               VK_SHADER_STAGE_VERTEX_BIT, 0,
                               sizeof(glm::mat4), &lightMVP);
            vkCmdDrawIndexed(cmd, mesh.indexCount, 1, 0, 0, 0);
            totalIndices = mesh.indexCount;
        }

        vkCmdEndRenderPass(cmd);
    }

    ctx.cmdEndLabel(cmd);

    // Update shadow matrices SSBO
    auto shadowBuf = m_scene->packShadowBuffer();
    if (!shadowBuf.empty()) {
        vmaMapMemory(ctx.allocator, m_shadowMatricesSSBOAlloc, &mapped);
        size_t copySize = std::min(shadowBuf.size() * sizeof(float),
                                    static_cast<size_t>(MAX_SHADOW_MAPS * 80));
        memcpy(mapped, shadowBuf.data(), copySize);
        vmaUnmapMemory(ctx.allocator, m_shadowMatricesSSBOAlloc);
    }

    // Update light buffer with shadow indices (repack lights with updated shadow indices)
    if (m_hasMultiLight && m_lightsSSBO != VK_NULL_HANDLE) {
        auto lightBuf = m_scene->packLightsBuffer();
        if (!lightBuf.empty()) {
            vmaMapMemory(ctx.allocator, m_lightsSSBOAlloc, &mapped);
            memcpy(mapped, lightBuf.data(), lightBuf.size() * sizeof(float));
            vmaUnmapMemory(ctx.allocator, m_lightsSSBOAlloc);
        }
    }
}

// --------------------------------------------------------------------------
// Shadow map cleanup
// --------------------------------------------------------------------------

void RasterRenderer::cleanupShadowMaps(VulkanContext& ctx) {
    if (m_shadowPipeline) vkDestroyPipeline(ctx.device, m_shadowPipeline, nullptr);
    m_shadowPipeline = VK_NULL_HANDLE;

    if (m_shadowPipelineLayout) vkDestroyPipelineLayout(ctx.device, m_shadowPipelineLayout, nullptr);
    m_shadowPipelineLayout = VK_NULL_HANDLE;

    if (m_shadowVertModule) vkDestroyShaderModule(ctx.device, m_shadowVertModule, nullptr);
    m_shadowVertModule = VK_NULL_HANDLE;

    for (auto fb : m_shadowFramebuffers) {
        if (fb) vkDestroyFramebuffer(ctx.device, fb, nullptr);
    }
    m_shadowFramebuffers.clear();

    if (m_shadowRenderPass) vkDestroyRenderPass(ctx.device, m_shadowRenderPass, nullptr);
    m_shadowRenderPass = VK_NULL_HANDLE;

    if (m_shadowSampler) vkDestroySampler(ctx.device, m_shadowSampler, nullptr);
    m_shadowSampler = VK_NULL_HANDLE;

    for (auto view : m_shadowLayerViews) {
        if (view) vkDestroyImageView(ctx.device, view, nullptr);
    }
    m_shadowLayerViews.clear();

    if (m_shadowImageView) vkDestroyImageView(ctx.device, m_shadowImageView, nullptr);
    m_shadowImageView = VK_NULL_HANDLE;

    if (m_shadowImage) vmaDestroyImage(ctx.allocator, m_shadowImage, m_shadowImageAlloc);
    m_shadowImage = VK_NULL_HANDLE;
    m_shadowImageAlloc = VK_NULL_HANDLE;

    if (m_shadowMatricesSSBO) vmaDestroyBuffer(ctx.allocator, m_shadowMatricesSSBO, m_shadowMatricesSSBOAlloc);
    m_shadowMatricesSSBO = VK_NULL_HANDLE;
    m_shadowMatricesSSBOAlloc = VK_NULL_HANDLE;

    m_hasShadows = false;
    m_numShadowMaps = 0;
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
                } else if (b.name == "SceneLight" && m_hasMultiLight && m_sceneLightUBO != VK_NULL_HANDLE) {
                    info.bufferInfo = {m_sceneLightUBO, 0, static_cast<VkDeviceSize>(b.size > 0 ? b.size : 32)};
                } else {
                    continue;
                }
                writeInfos.push_back(info);
                w.pBufferInfo = &writeInfos.back().bufferInfo;
                writes.push_back(w);
            } else if (b.type == "storage_buffer") {
                // Multi-light SSBO
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
                // Materials SSBO
                else if (b.name == "materials" && m_bindlessMaterials.buffer != VK_NULL_HANDLE) {
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
                // Shadow matrices SSBO
                else if (b.name == "shadow_matrices" && m_hasShadows && m_shadowMatricesSSBO != VK_NULL_HANDLE) {
                    DescWriteInfo info = {};
                    info.bufferInfo = {m_shadowMatricesSSBO, 0, VK_WHOLE_SIZE};
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
                // Shadow sampler
                if (b.name == "shadow_maps" && m_hasShadows && m_shadowSampler != VK_NULL_HANDLE) {
                    info.imageInfo.sampler = m_shadowSampler;
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
                // Shadow maps array texture
                if (b.name == "shadow_maps" && m_hasShadows && m_shadowImageView != VK_NULL_HANDLE) {
                    info.imageInfo.imageView = m_shadowImageView;
                    info.imageInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
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

    // Write vertex set (set 0): MVP + Light (or SceneLight for multi-light)
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
                } else if (b.name == "shadow_matrices" && m_hasShadows && m_shadowMatricesSSBO != VK_NULL_HANDLE) {
                    w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    info.bufferInfo = {m_shadowMatricesSSBO, 0, VK_WHOLE_SIZE};
                    writeInfos.push_back(info);
                    w.pBufferInfo = &writeInfos.back().bufferInfo;
                } else {
                    continue;
                }
            } else if (b.type == "sampled_image" && b.name == "shadow_maps" && m_hasShadows && m_shadowImageView != VK_NULL_HANDLE) {
                w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                info.imageInfo = {};
                info.imageInfo.imageView = m_shadowImageView;
                info.imageInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
                writeInfos.push_back(info);
                w.pImageInfo = &writeInfos.back().imageInfo;
            } else if (b.type == "sampler" && b.name == "shadow_maps" && m_hasShadows && m_shadowSampler != VK_NULL_HANDLE) {
                w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
                info.imageInfo = {};
                info.imageInfo.sampler = m_shadowSampler;
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
                    } else if (b.name == "shadow_matrices" && m_hasShadows && m_shadowMatricesSSBO != VK_NULL_HANDLE) {
                        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                        info.bufferInfo = {m_shadowMatricesSSBO, 0, VK_WHOLE_SIZE};
                        writeInfos.push_back(info);
                        w.pBufferInfo = &writeInfos.back().bufferInfo;
                    } else {
                        continue;
                    }
                } else if (b.type == "sampler") {
                    w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
                    if (b.name == "shadow_maps" && m_hasShadows && m_shadowSampler != VK_NULL_HANDLE) {
                        info.imageInfo = {};
                        info.imageInfo.sampler = m_shadowSampler;
                    } else {
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
                    }
                    writeInfos.push_back(info);
                    w.pImageInfo = &writeInfos.back().imageInfo;
                } else if (b.type == "sampled_image" || b.type == "sampled_cube_image") {
                    w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                    // Shadow maps array texture
                    if (b.name == "shadow_maps" && m_hasShadows && m_shadowImageView != VK_NULL_HANDLE) {
                        info.imageInfo = {};
                        info.imageInfo.imageView = m_shadowImageView;
                        info.imageInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
                    } else {
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
                    } // end shadow_maps else
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
    m_needsDepth = (sceneSource == "sphere" || sceneSource == "lighttest" || SceneManager::isGltfFile(sceneSource));

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

            // Detect multi-light from fragment reflection (must run before shadow setup
            // because shadow re-pack depends on m_hasMultiLight and m_lightsSSBO)
            setupMultiLightResources(ctx);

            // Detect shadow support from reflection and set up shadow maps
            {
                bool hasShadowMaps = false, hasShadowMatrices = false;
                for (auto& [setIdx, bindings] : fragReflection.descriptor_sets) {
                    for (auto& b : bindings) {
                        if (b.name == "shadow_maps") hasShadowMaps = true;
                        if (b.name == "shadow_matrices" && b.type == "storage_buffer") hasShadowMatrices = true;
                    }
                }
                if (hasShadowMaps && hasShadowMatrices) {
                    std::cout << "[info] Shader requests shadow maps, setting up shadow infrastructure" << std::endl;
                    auto& lights = const_cast<std::vector<SceneLight>&>(m_scene->getLights());
                    for (auto& l : lights) {
                        if (l.type == SceneLight::Directional || l.type == SceneLight::Spot)
                            l.castsShadow = true;
                    }
                    setupShadowMaps(ctx);
                    // Re-pack lights SSBO with updated shadow indices
                    if (m_hasMultiLight && m_lightsSSBO != VK_NULL_HANDLE && m_hasShadows) {
                        float nearClip = 0.1f;
                        float farClip = m_scene->hasSceneBounds() ? m_scene->getAutoFar() : 100.0f;
                        float aspect = static_cast<float>(renderWidth) / static_cast<float>(renderHeight);
                        glm::mat4 view = m_scene->hasSceneBounds()
                            ? glm::lookAt(m_scene->getAutoEye(), m_scene->getAutoTarget(), m_scene->getAutoUp())
                            : Camera::lookAt(Camera::DEFAULT_EYE, Camera::DEFAULT_TARGET, Camera::DEFAULT_UP);
                        glm::mat4 proj = m_scene->hasSceneBounds()
                            ? glm::perspective(glm::radians(45.0f), aspect, nearClip, farClip)
                            : Camera::perspective(Camera::DEFAULT_FOV, aspect, nearClip, farClip);
                        m_scene->computeShadowData(view, proj, nearClip, farClip);
                        auto lightBuf = m_scene->packLightsBuffer();
                        if (!lightBuf.empty()) {
                            void* mapped;
                            vmaMapMemory(ctx.allocator, m_lightsSSBOAlloc, &mapped);
                            memcpy(mapped, lightBuf.data(), lightBuf.size() * sizeof(float));
                            vmaUnmapMemory(ctx.allocator, m_lightsSSBOAlloc);
                        }
                        auto shadowBuf = m_scene->packShadowBuffer();
                        if (!shadowBuf.empty()) {
                            void* mapped;
                            vmaMapMemory(ctx.allocator, m_shadowMatricesSSBOAlloc, &mapped);
                            memcpy(mapped, shadowBuf.data(), shadowBuf.size() * sizeof(float));
                            vmaUnmapMemory(ctx.allocator, m_shadowMatricesSSBOAlloc);
                        }
                    }
                }
            }

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
                // Detect shadow support from reflection and set up shadow maps
                {
                    bool hasShadowMaps = false, hasShadowMatrices = false;
                    for (auto& [setIdx, bindings] : fragReflection.descriptor_sets) {
                        for (auto& b : bindings) {
                            if (b.name == "shadow_maps") hasShadowMaps = true;
                            if (b.name == "shadow_matrices" && b.type == "storage_buffer") hasShadowMatrices = true;
                        }
                    }
                    if (hasShadowMaps && hasShadowMatrices) {
                        std::cout << "[info] Shader requests shadow maps, setting up shadow infrastructure" << std::endl;
                        auto& lights = const_cast<std::vector<SceneLight>&>(m_scene->getLights());
                        for (auto& l : lights) {
                            if (l.type == SceneLight::Directional || l.type == SceneLight::Spot)
                                l.castsShadow = true;
                        }
                        setupShadowMaps(ctx);
                    }
                }
                setupMultiLightResources(ctx);
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

    // Pre-enable castsShadow on directional/spot lights if the shader manifest
    // supports shadows. This must happen BEFORE groupMaterialsByFeatures() so
    // that detectMaterialFeatures() includes "has_shadows" in each material's
    // feature set, selecting the correct +shadows permutation.
    {
        bool manifestHasShadows = false;
        for (const auto& fname : manifest.featureNames) {
            if (fname == "has_shadows") { manifestHasShadows = true; break; }
        }
        if (manifestHasShadows && m_scene) {
            auto& lights = const_cast<std::vector<SceneLight>&>(m_scene->getLights());
            for (auto& l : lights) {
                if (l.type == SceneLight::Directional || l.type == SceneLight::Spot)
                    l.castsShadow = true;
            }
        }
    }

    // Group materials by feature set (now includes has_shadows if applicable)
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

    // Detect multi-light from first permutation's fragment reflection and create buffers
    if (!m_permutations.empty()) {
        fragReflection = m_permutations[0].fragRefl;
        setupMultiLightResources(ctx);
    }

    // Detect shadow support from first permutation's fragment reflection
    if (!m_permutations.empty()) {
        bool hasShadowMaps = false, hasShadowMatrices = false;
        for (auto& [setIdx, bindings] : m_permutations[0].fragRefl.descriptor_sets) {
            for (auto& b : bindings) {
                if (b.name == "shadow_maps") hasShadowMaps = true;
                if (b.name == "shadow_matrices" && b.type == "storage_buffer") hasShadowMatrices = true;
            }
        }
        if (hasShadowMaps && hasShadowMatrices) {
            std::cout << "[info] Shader requests shadow maps, setting up shadow infrastructure" << std::endl;
            auto& lights = const_cast<std::vector<SceneLight>&>(m_scene->getLights());
            for (auto& l : lights) {
                if (l.type == SceneLight::Directional || l.type == SceneLight::Spot)
                    l.castsShadow = true;
            }
            setupShadowMaps(ctx);
        }
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
                } else if (b.name == "SceneLight" && m_hasMultiLight && m_sceneLightUBO != VK_NULL_HANDLE) {
                    info.bufferInfo = {m_sceneLightUBO, 0, static_cast<VkDeviceSize>(b.size > 0 ? b.size : 32)};
                } else continue;
                writeInfos.push_back(info);
                w.pBufferInfo = &writeInfos.back().bufferInfo;
            } else if (b.type == "storage_buffer") {
                if (b.name == "lights" && m_hasMultiLight && m_lightsSSBO != VK_NULL_HANDLE) {
                    w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    info.bufferInfo = {m_lightsSSBO, 0, VK_WHOLE_SIZE};
                    writeInfos.push_back(info);
                    w.pBufferInfo = &writeInfos.back().bufferInfo;
                } else if (b.name == "shadow_matrices" && m_hasShadows && m_shadowMatricesSSBO != VK_NULL_HANDLE) {
                    w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    info.bufferInfo = {m_shadowMatricesSSBO, 0, VK_WHOLE_SIZE};
                    writeInfos.push_back(info);
                    w.pBufferInfo = &writeInfos.back().bufferInfo;
                } else continue;
            } else if (b.type == "sampled_image" && b.name == "shadow_maps" && m_hasShadows && m_shadowImageView != VK_NULL_HANDLE) {
                w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                info.imageInfo = {};
                info.imageInfo.imageView = m_shadowImageView;
                info.imageInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
                writeInfos.push_back(info);
                w.pImageInfo = &writeInfos.back().imageInfo;
            } else if (b.type == "sampler" && b.name == "shadow_maps" && m_hasShadows && m_shadowSampler != VK_NULL_HANDLE) {
                w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
                info.imageInfo = {};
                info.imageInfo.sampler = m_shadowSampler;
                writeInfos.push_back(info);
                w.pImageInfo = &writeInfos.back().imageInfo;
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
                    } else if (b.name == "SceneLight" && m_hasMultiLight && m_sceneLightUBO != VK_NULL_HANDLE) {
                        info.bufferInfo = {m_sceneLightUBO, 0, static_cast<VkDeviceSize>(b.size > 0 ? b.size : 32)};
                    } else continue;
                    writeInfos.push_back(info);
                    w.pBufferInfo = &writeInfos.back().bufferInfo;
                } else if (b.type == "storage_buffer") {
                    if (b.name == "lights" && m_hasMultiLight && m_lightsSSBO != VK_NULL_HANDLE) {
                        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                        info.bufferInfo = {m_lightsSSBO, 0, VK_WHOLE_SIZE};
                        writeInfos.push_back(info);
                        w.pBufferInfo = &writeInfos.back().bufferInfo;
                    } else if (b.name == "shadow_matrices" && m_hasShadows && m_shadowMatricesSSBO != VK_NULL_HANDLE) {
                        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                        info.bufferInfo = {m_shadowMatricesSSBO, 0, VK_WHOLE_SIZE};
                        writeInfos.push_back(info);
                        w.pBufferInfo = &writeInfos.back().bufferInfo;
                    } else continue;
                } else if (b.type == "sampler") {
                    w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
                    if (b.name == "shadow_maps" && m_hasShadows && m_shadowSampler != VK_NULL_HANDLE) {
                        info.imageInfo = {}; info.imageInfo.sampler = m_shadowSampler;
                    } else {
                    auto& iblTextures = m_scene->getIBLTextures();
                    auto iblIt = iblTextures.find(b.name);
                    if (iblIt != iblTextures.end()) {
                        info.imageInfo = {}; info.imageInfo.sampler = iblIt->second.sampler;
                    } else {
                        auto& tex = m_scene->getTextureForBinding(b.name);
                        info.imageInfo = {}; info.imageInfo.sampler = tex.sampler;
                    }
                    }
                    writeInfos.push_back(info);
                    w.pImageInfo = &writeInfos.back().imageInfo;
                } else if (b.type == "sampled_image" || b.type == "sampled_cube_image") {
                    w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                    if (b.name == "shadow_maps" && m_hasShadows && m_shadowImageView != VK_NULL_HANDLE) {
                        info.imageInfo = {};
                        info.imageInfo.imageView = m_shadowImageView;
                        info.imageInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
                    } else {
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
                    } // end shadow_maps else
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
// Update a material UBO at runtime (editor property editing)
// --------------------------------------------------------------------------

void RasterRenderer::updateMaterialUBO(VulkanContext& ctx, int materialIndex, const MaterialUBOData& data) {
    void* mapped = nullptr;

    // Single-pipeline mode: material index 0 uses m_materialBuffer
    if (!m_multiPipeline && materialIndex == 0 && m_materialBuffer != VK_NULL_HANDLE) {
        vmaMapMemory(ctx.allocator, m_materialAllocation, &mapped);
        memcpy(mapped, &data, sizeof(MaterialUBOData));
        vmaUnmapMemory(ctx.allocator, m_materialAllocation);
        return;
    }

    // Multi-pipeline mode: per-material buffers
    if (materialIndex >= 0 && materialIndex < static_cast<int>(m_perMaterialBuffers.size())) {
        vmaMapMemory(ctx.allocator, m_perMaterialAllocations[materialIndex], &mapped);
        memcpy(mapped, &data, sizeof(MaterialUBOData));
        vmaUnmapMemory(ctx.allocator, m_perMaterialAllocations[materialIndex]);
        return;
    }

    // Also check permutation per-material UBOs
    for (auto& perm : m_permutations) {
        for (size_t i = 0; i < perm.materialIndices.size(); i++) {
            if (perm.materialIndices[i] == materialIndex && i < perm.perMaterialUBOs.size()) {
                vmaMapMemory(ctx.allocator, perm.perMaterialAllocations[i], &mapped);
                memcpy(mapped, &data, sizeof(MaterialUBOData));
                vmaUnmapMemory(ctx.allocator, perm.perMaterialAllocations[i]);
                return;
            }
        }
    }
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

    // Shadow pass: render depth maps before main color pass
    if (m_hasShadows) {
        renderShadowMaps(ctx, cmd);
    }

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

    // Shadow pass: render depth maps before main color pass
    if (m_hasShadows) {
        renderShadowMaps(ctx, cmd);
    }

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

    // Update SceneLight UBO viewPos for multi-light mode
    if (m_hasMultiLight && m_sceneLightUBO != VK_NULL_HANDLE) {
        struct SceneLightUBO {
            glm::vec3 viewPos;      // offset 0, size 12
            int32_t lightCount;     // offset 12, size 4
        };
        vmaMapMemory(ctx.allocator, m_sceneLightUBOAlloc, &mapped);
        SceneLightUBO* sl = static_cast<SceneLightUBO*>(mapped);
        sl->viewPos = eye;
        vmaUnmapMemory(ctx.allocator, m_sceneLightUBOAlloc);
    }
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

    // Cleanup multi-light buffers
    if (m_lightsSSBO) vmaDestroyBuffer(ctx.allocator, m_lightsSSBO, m_lightsSSBOAlloc);
    if (m_sceneLightUBO) vmaDestroyBuffer(ctx.allocator, m_sceneLightUBO, m_sceneLightUBOAlloc);
    m_lightsSSBO = VK_NULL_HANDLE;
    m_sceneLightUBO = VK_NULL_HANDLE;
    m_hasMultiLight = false;

    // Cleanup shadow map resources
    cleanupShadowMaps(ctx);

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
