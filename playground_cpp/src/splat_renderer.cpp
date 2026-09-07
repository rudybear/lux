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
#include <cstdlib>
#include <cmath>

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

// Reads a `"gaussian_splatting": { ..., "<key>": true|false, ... }` boolean
// flag from the preprocess stage's reflection JSON (same lightweight
// substring-scan approach as readShaderShDegree -- luxc's reflection writer
// always emits `"key": true`/`"key": false` with that exact spacing, so this
// avoids pulling in a general JSON parser for one boolean lookup).
static bool readShaderBoolFlag(const std::string& shaderBase, const std::string& key) {
    std::string jsonPath = shaderBase + ".comp.json";
    if (!fs::exists(jsonPath)) return false;
    std::ifstream f(jsonPath);
    if (!f.is_open()) return false;
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    auto gsPos = content.find("\"gaussian_splatting\"");
    if (gsPos == std::string::npos) return false;
    std::string needleTrue = "\"" + key + "\": true";
    auto keyPos = content.find("\"" + key + "\"", gsPos);
    if (keyPos == std::string::npos) return false;
    return content.compare(keyPos, needleTrue.size(), needleTrue) == 0;
}

// Applies a sub-pixel jitter (in pixels) to a projection matrix by directly
// shifting the post-divide NDC x/y coordinates: for any projection matrix P,
// adding `d * row3(P)` to `row_k(P)` adds exactly `d` to `clip[k]/clip.w`
// after the perspective divide, regardless of the matrix's internal sign
// convention (Vulkan Y-flip already baked in via proj[1][1] *= -1 or not).
// Positive jx shifts rendered content right; positive jy shifts it down
// (matches the OpenCV convention `K[1,2] += jy`, see docs/lux-4d-spec.md
// section 3 -- lux's NDC-to-pixel mapping in the vertex stage,
// pixel = (ndc*0.5+0.5)*screen_size, is already a direct increasing map from
// ndc.y to pixel row, so a positive ndc.y delta moves content down).
static glm::mat4 applySplatJitter(glm::mat4 proj, float jitterXPixels, float jitterYPixels,
                                   uint32_t width, uint32_t height) {
    if (jitterXPixels == 0.0f && jitterYPixels == 0.0f) return proj;
    float dx = 2.0f * jitterXPixels / static_cast<float>(width);
    float dy = 2.0f * jitterYPixels / static_cast<float>(height);
    for (int c = 0; c < 4; ++c) {
        proj[c][0] += dx * proj[c][3];
        proj[c][1] += dy * proj[c][3];
    }
    return proj;
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
// GPU timestamp-query profiling (mobile-DLSS Android live demo, Stage 1)
// --------------------------------------------------------------------------

void SplatRenderer::setGpuTimingEnabled(VulkanContext& ctx, bool enabled) {
    if (enabled == gpuTimingEnabled_) return;
    if (enabled) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(ctx.physicalDevice, &props);
        timestampPeriodNs_ = props.limits.timestampPeriod;
        if (timestampPeriodNs_ <= 0.0) {
            // Device reports no usable timestamp period (spec allows 0 to
            // mean "unsupported" on some implementations) -- leave disabled
            // rather than divide by zero below.
            return;
        }
        VkQueryPoolCreateInfo qpci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpci.queryCount = 4;
        if (vkCreateQueryPool(ctx.device, &qpci, nullptr, &timestampPool_) != VK_SUCCESS) {
            return;
        }
        gpuTimingEnabled_ = true;
    } else {
        if (timestampPool_ != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(ctx.device);
            vkDestroyQueryPool(ctx.device, timestampPool_, nullptr);
            timestampPool_ = VK_NULL_HANDLE;
        }
        gpuTimingEnabled_ = false;
    }
}

// --------------------------------------------------------------------------
// Offscreen render target (VMA)
// --------------------------------------------------------------------------

void SplatRenderer::createOffscreenTarget(VulkanContext& ctx) {
    // Color image
    VkImageCreateInfo imageInfo = {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
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
    viewInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
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

    // --- DLSS input-contract outputs (docs/lux-4d-spec.md section 3) ---
    // Single packed RGBA32F `out_aux` attachment (mv.x*a, mv.y*a, depth*a,
    // a) -- replaces the earlier two-attachment (RGBA32F each)
    // out_motion/out_depth design with zero wasted lanes (still a real 2x
    // bandwidth win when foreground_coverage is off; NOT downgraded to
    // RGBA16F -- see splat_renderer.h's getAuxImage() comment for the
    // measured precision failure that ruled that out) and
    // luxc/expansion/splat_expander.py's out_aux comment.
    if (hasMotionVectors_ || hasExpectedDepth_) {
        VkImageCreateInfo auxInfo = imageInfo;
        auxInfo.format = VK_FORMAT_R32G32B32A32_SFLOAT;
        vmaCreateImage(ctx.allocator, &auxInfo, &allocInfo, &auxImage_, &auxAlloc_, nullptr);

        VkImageViewCreateInfo auxViewInfo = viewInfo;
        auxViewInfo.image = auxImage_;
        auxViewInfo.format = VK_FORMAT_R32G32B32A32_SFLOAT;
        vkCreateImageView(ctx.device, &auxViewInfo, nullptr, &auxView_);
    }
    // Second, smaller RGBA16F `out_fg` attachment (fg*a, 0, 0, a) -- see
    // splat_renderer.h's getFgImage() comment for why this can't share
    // out_aux's lanes.
    if (hasForegroundCoverage_) {
        VkImageCreateInfo fgInfo = imageInfo;
        fgInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        vmaCreateImage(ctx.allocator, &fgInfo, &allocInfo, &fgImage_, &fgAlloc_, nullptr);

        VkImageViewCreateInfo fgViewInfo = viewInfo;
        fgViewInfo.image = fgImage_;
        fgViewInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        vkCreateImageView(ctx.device, &fgViewInfo, nullptr, &fgView_);
    }
}

// --------------------------------------------------------------------------
// Render pass (color + depth, finalLayout = TRANSFER_SRC for blit/screenshot)
// --------------------------------------------------------------------------

void SplatRenderer::createRenderPass(VkDevice device) {
    // Color attachments: out_color always present; the single packed
    // out_aux attachment (docs/lux-4d-spec.md section 3) appended when
    // motion_vectors or expected_depth is enabled, with the SAME
    // premultiplied-alpha blend as color (set up in createPipelines) so
    // overlapping splats blend correctly. The depth-test attachment is
    // always last.
    std::vector<VkAttachmentDescription> attachments;
    std::vector<VkAttachmentReference> colorRefs;

    VkAttachmentDescription colorAttach = {};
    colorAttach.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    colorAttach.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttach.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttach.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttach.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttach.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttach.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttach.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    attachments.push_back(colorAttach);
    colorRefs.push_back({static_cast<uint32_t>(attachments.size() - 1), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});

    if (hasMotionVectors_ || hasExpectedDepth_) {
        VkAttachmentDescription aux = colorAttach;
        aux.format = VK_FORMAT_R32G32B32A32_SFLOAT;
        attachments.push_back(aux);
        colorRefs.push_back({static_cast<uint32_t>(attachments.size() - 1), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
    }
    if (hasForegroundCoverage_) {
        VkAttachmentDescription fg = colorAttach;
        fg.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        attachments.push_back(fg);
        colorRefs.push_back({static_cast<uint32_t>(attachments.size() - 1), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
    }

    // Depth (test)
    VkAttachmentDescription depthAttach = {};
    depthAttach.format = VK_FORMAT_D32_SFLOAT;
    depthAttach.samples = VK_SAMPLE_COUNT_1_BIT;
    depthAttach.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttach.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttach.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttach.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttach.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttach.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    attachments.push_back(depthAttach);
    VkAttachmentReference depthRef = {static_cast<uint32_t>(attachments.size() - 1),
                                       VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = static_cast<uint32_t>(colorRefs.size());
    subpass.pColorAttachments = colorRefs.data();
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
    rpInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    rpInfo.pAttachments = attachments.data();
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
    attachments[0].format = VK_FORMAT_R16G16B16A16_SFLOAT;
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
    attachments[0].format = VK_FORMAT_R16G16B16A16_SFLOAT;
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
    std::vector<VkImageView> fbViews;
    fbViews.push_back(colorView_);
    if (hasMotionVectors_ || hasExpectedDepth_) fbViews.push_back(auxView_);
    if (hasForegroundCoverage_) fbViews.push_back(fgView_);
    fbViews.push_back(depthView_);

    VkFramebufferCreateInfo fbInfo = {};
    fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fbInfo.renderPass = renderPass_;
    fbInfo.attachmentCount = static_cast<uint32_t>(fbViews.size());
    fbInfo.pAttachments = fbViews.data();
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

void SplatRenderer::createPipelines(VulkanContext& ctx, const std::string& shaderBase) {
    VkDevice device = ctx.device;
    // --- Descriptor set layouts ---

    // Compute: 4 input + N SH coefficients + 7 output SSBOs
    // Output order: proj_center, proj_axes, proj_conic, proj_color, sort_keys, sorted_indices, visible_count
    // + (motion_vectors) splat_prev_pos, projected_mv, prev_camera_mats
    // + (expected_depth) projected_depth
    // + (foreground_coverage) splat_foreground, projected_foreground,
    // appended in that order (matches splat_expander._build_preprocess_stage
    // exactly).
    uint32_t numShCoeffs = numShCoeffsForDegree(shaderShDegree_);
    uint32_t numComputeBindings = 4 + numShCoeffs + 7
        + (hasMotionVectors_ ? 3 : 0) + (hasExpectedDepth_ ? 1 : 0)
        + (hasForegroundCoverage_ ? 2 : 0);
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

    // Render: 5 SSBOs (projected_centers, axes, conics, colors, sorted_indices)
    // + (motion_vectors) projected_mv + (expected_depth) projected_depth
    // + (foreground_coverage) projected_foreground.
    uint32_t numRenderBindings = 5 + (hasMotionVectors_ ? 1 : 0) + (hasExpectedDepth_ ? 1 : 0)
        + (hasForegroundCoverage_ ? 1 : 0);
    std::vector<VkDescriptorSetLayoutBinding> renderBindings(numRenderBindings);
    for (uint32_t i = 0; i < numRenderBindings; ++i) {
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

    // Compute push constants: view(64) + proj(64) + camPos(12) + pad(4) + focal(8) + screen(8) + numSplats(4) + pad(12) = 176 bytes.
    // proj_matrix_unjittered/prev_view_proj_unjittered (motion_vectors) are
    // NOT pushed here -- they live in prevCameraBuffer_ (a tiny storage
    // buffer, see splat_renderer.h) instead. They used to be pushed as two
    // extra mat4 push-constant fields, putting this block at 304 bytes;
    // that's within what desktop/iOS GPUs report (MoltenVK: 4096 on Apple
    // Silicon, checked via `vulkaninfo`) but OVER the 256-byte
    // maxPushConstantsSize Mali-G715 actually reports, which is a Vulkan
    // spec violation (VUID-VkPushConstantRange-size-00298) -- Mali's driver
    // accepted the oversized range without validation layers active, then
    // silently served undefined/recycled data for the out-of-range tail
    // (bytes 256-303, i.e. most of prev_view_proj_unjittered) at dispatch
    // time, corrupting every motion vector on Android by 3-6 orders of
    // magnitude while leaving color/depth (which only need bytes 0-175/
    // 0-239) correct -- see docs/rendering-engines.md's Android
    // live-rendering demo MV investigation and the startup check below.
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

    // --- Guardrail against this exact class of bug recurring (any future
    // push-constant field added to any of this renderer's shaders) ---
    // Vulkan allows vkCreatePipelineLayout to silently accept a push-
    // constant range larger than the device supports (no validation layers
    // required to catch it), so check explicitly and fail loudly instead
    // of quietly corrupting whatever field lands past the limit.
    {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(ctx.physicalDevice, &props);
        uint32_t maxPc = props.limits.maxPushConstantsSize;
        if (computePush.size > maxPc || renderPush.size > maxPc) {
            throw std::runtime_error(
                "SplatRenderer::createPipelines: a push-constant range exceeds this "
                "device's maxPushConstantsSize (" + std::to_string(maxPc) + " bytes on \"" +
                std::string(props.deviceName) + "\") -- compute push=" +
                std::to_string(computePush.size) + "B, render push=" +
                std::to_string(renderPush.size) + "B. This is a Vulkan spec violation "
                "(VUID-VkPushConstantRange-size-00298) some drivers accept without "
                "validation layers active, then serve undefined data for the "
                "out-of-range bytes at dispatch time -- see the Android MV corruption "
                "this exact bug caused (docs/rendering-engines.md). Move the offending "
                "field(s) into a storage buffer (see prev_camera_mats in "
                "luxc/expansion/splat_expander.py) instead of adding push-constant bytes.");
        }
    }

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

    // out_aux (docs/lux-4d-spec.md section 3, packed mv/depth/fg) uses the
    // SAME premultiplied-alpha blend equation as out_color, so overlapping
    // splats blend to the correct visibility-weighted average for free.
    uint32_t numColorAttachments = 1 + ((hasMotionVectors_ || hasExpectedDepth_) ? 1 : 0)
        + (hasForegroundCoverage_ ? 1 : 0);
    std::vector<VkPipelineColorBlendAttachmentState> blendAttachments(numColorAttachments, blendAttachment);

    VkPipelineColorBlendStateCreateInfo colorBlending = {};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = numColorAttachments;
    colorBlending.pAttachments = blendAttachments.data();

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
    // + 1 morph-apply set (13 bindings, see _build_morph_apply_stage) for dynamic splats.
    static constexpr uint32_t kMorphBindingCount = 13;
    VkDescriptorPoolSize poolSize = {};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = numComputeBindings + numRenderBindings + 16 + kMorphBindingCount + 4; // + morph + margin

    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 2 + 5 + 1;  // compute, render + 2 histogram + 1 prefix_sum + 2 scatter + morph
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

    // TRANSFER_SRC: posBuffer_ (the first buffer created with this flag set,
    // below) is used as a vkCmdCopyBuffer source both by the motion-vector
    // first-frame prevPosBuffer_ seed and by seedPreviousMorphTime()
    // (docs/lux-4d-spec.md section 3).
    VkBufferUsageFlags ssbo = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

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
                    VMA_MEMORY_USAGE_GPU_ONLY, projAxesBuffer_, projAxesAlloc_);
    createVmaBuffer(ctx.allocator, numSplats_ * 4 * sizeof(float), ssbo,
                    VMA_MEMORY_USAGE_GPU_ONLY, projConicBuffer_, projConicAlloc_);
    createVmaBuffer(ctx.allocator, numSplats_ * 4 * sizeof(float), ssbo,
                    VMA_MEMORY_USAGE_GPU_ONLY, projColorBuffer_, projColorAlloc_);

    // --- DLSS input-contract outputs (docs/lux-4d-spec.md section 3) ---
    if (hasMotionVectors_) {
        createVmaBuffer(ctx.allocator, numSplats_ * 2 * sizeof(float), ssbo,
                        VMA_MEMORY_USAGE_GPU_ONLY, projMvBuffer_, projMvAlloc_);
        createPrevPosBuffer(ctx, data);
        // 2x mat4 = 128 bytes; needs TRANSFER_DST for the per-frame
        // vkCmdUpdateBuffer write in render() (see splat_renderer.h).
        createVmaBuffer(ctx.allocator, 2 * sizeof(glm::mat4),
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        VMA_MEMORY_USAGE_GPU_ONLY, prevCameraBuffer_, prevCameraAlloc_);
    }
    if (hasExpectedDepth_) {
        createVmaBuffer(ctx.allocator, numSplats_ * sizeof(float), ssbo,
                        VMA_MEMORY_USAGE_GPU_ONLY, projDepthBuffer_, projDepthAlloc_);
    }
    if (hasForegroundCoverage_) {
        // Input: one 0.0/1.0 value per splat, uploaded once at load time
        // (gltf_loader.cpp already resolved the _FOREGROUND-attribute /
        // morph-delta-fallback / all-zero precedence into data.foreground).
        VkDeviceSize fgSize = numSplats_ * sizeof(float);
        createVmaBuffer(ctx.allocator, fgSize, ssbo,
                        VMA_MEMORY_USAGE_CPU_TO_GPU, foregroundBuffer_, foregroundAlloc_);
        if (data.foreground.size() == numSplats_) {
            uploadVmaBuffer(ctx.allocator, foregroundAlloc_, data.foreground.data(), fgSize);
        } else {
            // Should not happen (gltf_loader.cpp always sizes `foreground`
            // to num_splats), but never leave uninitialized GPU memory.
            void* mapped = nullptr;
            vmaMapMemory(ctx.allocator, foregroundAlloc_, &mapped);
            std::memset(mapped, 0, static_cast<size_t>(fgSize));
            vmaUnmapMemory(ctx.allocator, foregroundAlloc_);
        }
        createVmaBuffer(ctx.allocator, numSplats_ * sizeof(float), ssbo,
                        VMA_MEMORY_USAGE_GPU_ONLY, projForegroundBuffer_, projForegroundAlloc_);
    }

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
    writeSSBO(computeDescSet_, outputBase + 1, projAxesBuffer_, numSplats_ * 4 * sizeof(float));
    writeSSBO(computeDescSet_, outputBase + 2, projConicBuffer_, numSplats_ * 4 * sizeof(float));
    writeSSBO(computeDescSet_, outputBase + 3, projColorBuffer_, numSplats_ * 4 * sizeof(float));
    writeSSBO(computeDescSet_, outputBase + 4, sortKeysBuffer_, numSplats_ * sizeof(uint32_t));
    writeSSBO(computeDescSet_, outputBase + 5, sortedIndicesBuffer_, numSplats_ * sizeof(uint32_t));
    writeSSBO(computeDescSet_, outputBase + 6, visibleCountBuffer_, sizeof(uint32_t));

    // --- DLSS input-contract outputs (docs/lux-4d-spec.md section 3) ---
    // Appended after visible_count, matching splat_expander._build_preprocess_stage's
    // storage_buffers order exactly: [motion_vectors: splat_prev_pos, projected_mv,
    // prev_camera_mats] then [expected_depth: projected_depth].
    uint32_t computeNextBinding = outputBase + 7;
    if (hasMotionVectors_) {
        writeSSBO(computeDescSet_, computeNextBinding++, prevPosBuffer_, numSplats_ * 4 * sizeof(float));
        writeSSBO(computeDescSet_, computeNextBinding++, projMvBuffer_, numSplats_ * 2 * sizeof(float));
        writeSSBO(computeDescSet_, computeNextBinding++, prevCameraBuffer_, 2 * sizeof(glm::mat4));
    }
    if (hasExpectedDepth_) {
        writeSSBO(computeDescSet_, computeNextBinding++, projDepthBuffer_, numSplats_ * sizeof(float));
    }
    if (hasForegroundCoverage_) {
        writeSSBO(computeDescSet_, computeNextBinding++, foregroundBuffer_, numSplats_ * sizeof(float));
        writeSSBO(computeDescSet_, computeNextBinding++, projForegroundBuffer_, numSplats_ * sizeof(float));
    }

    // Render set: projected_centers(0), axes(1), conics(2), colors(3), sorted_indices(4)
    // + (motion_vectors) projected_mv + (expected_depth) projected_depth
    // + (foreground_coverage) projected_foreground.
    writeSSBO(renderDescSet_, 0, projCenterBuffer_, numSplats_ * 4 * sizeof(float));
    writeSSBO(renderDescSet_, 1, projAxesBuffer_, numSplats_ * 4 * sizeof(float));
    writeSSBO(renderDescSet_, 2, projConicBuffer_, numSplats_ * 4 * sizeof(float));
    writeSSBO(renderDescSet_, 3, projColorBuffer_, numSplats_ * 4 * sizeof(float));
    writeSSBO(renderDescSet_, 4, sortedIndicesBuffer_, numSplats_ * sizeof(uint32_t));
    uint32_t renderNextBinding = 5;
    if (hasMotionVectors_) {
        writeSSBO(renderDescSet_, renderNextBinding++, projMvBuffer_, numSplats_ * 2 * sizeof(float));
    }
    if (hasExpectedDepth_) {
        writeSSBO(renderDescSet_, renderNextBinding++, projDepthBuffer_, numSplats_ * sizeof(float));
    }
    if (hasForegroundCoverage_) {
        writeSSBO(renderDescSet_, renderNextBinding++, projForegroundBuffer_, numSplats_ * sizeof(float));
    }

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
// Dynamic (4D) splats: morph-apply pipeline (luxc-emitted <base>.morph.comp.spv)
// --------------------------------------------------------------------------
//
// Buffer bindings match splat_expander._build_morph_apply_stage's storage_buffers
// declaration order exactly (auto-assigned bindings 0..12 by luxc):
//   0 splat_base_pos, 1 splat_base_rot, 2 splat_base_sh0,
//   3 morph_index,
//   4 morph_delta_pos_lo, 5 morph_delta_rot_lo, 6 morph_delta_sh0_lo,
//   7 morph_delta_pos_hi, 8 morph_delta_rot_hi, 9 morph_delta_sh0_hi,
//   10 splat_pos, 11 splat_rot, 12 splat_sh0  (== preprocess's own input buffers)
void SplatRenderer::createMorphPipeline(VkDevice device, const std::string& shaderBase) {
    std::string morphSpvPath = shaderBase + ".morph.comp.spv";
    if (!fs::exists(morphSpvPath)) {
        // Not a `motion: keyframes` pipeline -- static splats render exactly
        // as before, no morph stage.
        return;
    }

    static constexpr uint32_t kMorphBindingCount = 13;
    std::vector<VkDescriptorSetLayoutBinding> bindings(kMorphBindingCount);
    for (uint32_t i = 0; i < kMorphBindingCount; ++i) {
        bindings[i] = {};
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = kMorphBindingCount;
    layoutInfo.pBindings = bindings.data();
    vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &morphSetLayout_);

    // Push constants: segment_offset(u32) + segment_count(u32) + weight_lo(f32) + weight_hi(f32) = 16B
    VkPushConstantRange pushRange = {};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = 16;

    VkPipelineLayoutCreateInfo pipeLayoutInfo = {};
    pipeLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeLayoutInfo.setLayoutCount = 1;
    pipeLayoutInfo.pSetLayouts = &morphSetLayout_;
    pipeLayoutInfo.pushConstantRangeCount = 1;
    pipeLayoutInfo.pPushConstantRanges = &pushRange;
    vkCreatePipelineLayout(device, &pipeLayoutInfo, nullptr, &morphLayout_);

    auto code = SpvLoader::loadSPIRV(morphSpvPath);
    VkShaderModule module = SpvLoader::createShaderModule(device, code);

    VkComputePipelineCreateInfo pipeInfo = {};
    pipeInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeInfo.stage.module = module;
    pipeInfo.stage.pName = "main";
    pipeInfo.layout = morphLayout_;
    vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &morphPipeline_);
    vkDestroyShaderModule(device, module, nullptr);

    std::cout << "[info] Loaded dynamic-splat morph-apply shader: " << morphSpvPath << std::endl;
}

void SplatRenderer::createMorphBuffers(VulkanContext& ctx, const GaussianSplatData& data) {
    dynamics_ = data.dynamics;
    if (!dynamics_.has_motion || morphPipeline_ == VK_NULL_HANDLE) {
        if (dynamics_.has_motion && morphPipeline_ == VK_NULL_HANDLE) {
            std::cerr << "[warn] Scene has morph-target animation but the shader base '"
                      << "' has no .morph.comp.spv -- compile with `motion: keyframes` "
                      << "(e.g. examples/gaussian_splat_dynamic.lux) to animate it. "
                      << "Rendering the static base frame." << std::endl;
        }
        return;
    }

    morphSegments_ = buildSplatMorphSegments(dynamics_);
    segmentOffsets_.resize(morphSegments_.size());
    segmentCounts_.resize(morphSegments_.size());

    std::vector<uint32_t> catIndex;
    std::vector<float> catPosLo, catPosHi, catRotLo, catRotHi, catSh0Lo, catSh0Hi;
    for (size_t seg = 0; seg < morphSegments_.size(); seg++) {
        const auto& s = morphSegments_[seg];
        segmentOffsets_[seg] = static_cast<uint32_t>(catIndex.size());
        segmentCounts_[seg] = static_cast<uint32_t>(s.index.size());
        catIndex.insert(catIndex.end(), s.index.begin(), s.index.end());
        catPosLo.insert(catPosLo.end(), s.dposLo.begin(), s.dposLo.end());
        catPosHi.insert(catPosHi.end(), s.dposHi.begin(), s.dposHi.end());
        catRotLo.insert(catRotLo.end(), s.drotLo.begin(), s.drotLo.end());
        catRotHi.insert(catRotHi.end(), s.drotHi.begin(), s.drotHi.end());
        catSh0Lo.insert(catSh0Lo.end(), s.dsh0Lo.begin(), s.dsh0Lo.end());
        catSh0Hi.insert(catSh0Hi.end(), s.dsh0Hi.begin(), s.dsh0Hi.end());
    }
    morphTotalEntries_ = static_cast<uint32_t>(catIndex.size());

    // Pad vec3 delta arrays to vec4 (matches StorageBufferDecl("...", "vec4") in luxc).
    auto pad3to4 = [](const std::vector<float>& src) {
        std::vector<float> out(src.size() / 3 * 4, 0.0f);
        for (size_t i = 0; i < src.size() / 3; i++) {
            out[i * 4 + 0] = src[i * 3 + 0];
            out[i * 4 + 1] = src[i * 3 + 1];
            out[i * 4 + 2] = src[i * 3 + 2];
        }
        return out;
    };
    std::vector<float> catPosLo4 = pad3to4(catPosLo), catPosHi4 = pad3to4(catPosHi);
    std::vector<float> catSh0Lo4 = pad3to4(catSh0Lo), catSh0Hi4 = pad3to4(catSh0Hi);

    VkBufferUsageFlags ssbo = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    auto uploadNew = [&](const void* data_, VkDeviceSize size, VkBuffer& buf, VmaAllocation& alloc) {
        createVmaBuffer(ctx.allocator, size, ssbo, VMA_MEMORY_USAGE_CPU_TO_GPU, buf, alloc);
        uploadVmaBuffer(ctx.allocator, alloc, data_, size);
    };

    // Immutable base copies (posBuffer_/rotBuffer_/shBuffers_[0] are the mutable
    // "working" buffers preprocess reads -- these are separate, untouched originals).
    uploadNew(data.positions.data(), data.positions.size() * sizeof(float), baseposBuffer_, baseposAlloc_);
    uploadNew(data.rotations.data(), numSplats_ * 4 * sizeof(float), baserotBuffer_, baserotAlloc_);
    {
        std::vector<float> sh0base(numSplats_ * 4, 0.0f);
        if (!data.sh_coefficients.empty()) {
            const auto& c = data.sh_coefficients[0];
            uint32_t fps = c.empty() ? 0 : static_cast<uint32_t>(c.size() / numSplats_);
            if (fps == 3) {
                for (uint32_t i = 0; i < numSplats_; i++)
                    for (int k = 0; k < 3; k++) sh0base[i * 4 + k] = c[i * 3 + k];
            } else if (fps == 4) {
                sh0base = c;
            }
        }
        uploadNew(sh0base.data(), sh0base.size() * sizeof(float), basesh0Buffer_, basesh0Alloc_);
    }

    uploadNew(catIndex.data(), std::max<size_t>(catIndex.size(), 1) * sizeof(uint32_t), morphIndexBuffer_, morphIndexAlloc_);
    uploadNew(catPosLo4.data(), std::max<size_t>(catPosLo4.size(), 4) * sizeof(float), morphPosLoBuffer_, morphPosLoAlloc_);
    uploadNew(catRotLo.data(), std::max<size_t>(catRotLo.size(), 4) * sizeof(float), morphRotLoBuffer_, morphRotLoAlloc_);
    uploadNew(catSh0Lo4.data(), std::max<size_t>(catSh0Lo4.size(), 4) * sizeof(float), morphSh0LoBuffer_, morphSh0LoAlloc_);
    uploadNew(catPosHi4.data(), std::max<size_t>(catPosHi4.size(), 4) * sizeof(float), morphPosHiBuffer_, morphPosHiAlloc_);
    uploadNew(catRotHi.data(), std::max<size_t>(catRotHi.size(), 4) * sizeof(float), morphRotHiBuffer_, morphRotHiAlloc_);
    uploadNew(catSh0Hi4.data(), std::max<size_t>(catSh0Hi4.size(), 4) * sizeof(float), morphSh0HiBuffer_, morphSh0HiAlloc_);

    // --- Allocate + write the morph descriptor set ---
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool_;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &morphSetLayout_;
    vkAllocateDescriptorSets(ctx.device, &allocInfo, &morphDescSet_);

    auto writeSSBO = [&](uint32_t binding, VkBuffer buffer, VkDeviceSize size) {
        VkDescriptorBufferInfo bufInfo = {};
        bufInfo.buffer = buffer;
        bufInfo.offset = 0;
        bufInfo.range = std::max(size, VkDeviceSize(4));
        VkWriteDescriptorSet write = {};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = morphDescSet_;
        write.dstBinding = binding;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &bufInfo;
        vkUpdateDescriptorSets(ctx.device, 1, &write, 0, nullptr);
    };
    VkDeviceSize vec4Total = morphTotalEntries_ * 4 * sizeof(float);
    writeSSBO(0, baseposBuffer_, numSplats_ * 4 * sizeof(float));
    writeSSBO(1, baserotBuffer_, numSplats_ * 4 * sizeof(float));
    writeSSBO(2, basesh0Buffer_, numSplats_ * 4 * sizeof(float));
    writeSSBO(3, morphIndexBuffer_, morphTotalEntries_ * sizeof(uint32_t));
    writeSSBO(4, morphPosLoBuffer_, vec4Total);
    writeSSBO(5, morphRotLoBuffer_, vec4Total);
    writeSSBO(6, morphSh0LoBuffer_, vec4Total);
    writeSSBO(7, morphPosHiBuffer_, vec4Total);
    writeSSBO(8, morphRotHiBuffer_, vec4Total);
    writeSSBO(9, morphSh0HiBuffer_, vec4Total);
    // The morph stage's OUTPUT buffers are exactly preprocess's own INPUT
    // buffers -- no copy, no separate storage. This is what makes
    // preprocess itself require zero changes for dynamic splats.
    writeSSBO(10, posBuffer_, numSplats_ * 4 * sizeof(float));
    writeSSBO(11, rotBuffer_, numSplats_ * 4 * sizeof(float));
    writeSSBO(12, shBuffers_.empty() ? posBuffer_ : shBuffers_[0], numSplats_ * 4 * sizeof(float));

    std::cout << "[info] Dynamic splats: " << morphSegments_.size() << " segments, "
              << morphTotalEntries_ << " total sparse entries" << std::endl;
}

// --------------------------------------------------------------------------
// DLSS input-contract outputs: previous-frame position buffer
// --------------------------------------------------------------------------
//
// For static splats (no `motion: keyframes`), positions never change frame
// to frame, so splat_prev_pos can simply alias posBuffer_ -- motion vectors
// then reduce to the pure camera-motion term, with zero extra memory/copies.
// For dynamic splats, a distinct buffer is required: it's seeded to the
// scene's base positions here, and render() copies posBuffer_ into it once
// per frame (after the morph-apply stage has written the current frame's
// positions), so it always holds the PREVIOUS frame's animated position by
// the time the NEXT frame's preprocess dispatch reads it.
void SplatRenderer::createPrevPosBuffer(VulkanContext& ctx, const GaussianSplatData& data) {
    if (!data.dynamics.has_motion) {
        prevPosBuffer_ = posBuffer_;
        prevPosOwned_ = false;
        return;
    }
    VkBufferUsageFlags ssbo = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    createVmaBuffer(ctx.allocator, data.positions.size() * sizeof(float), ssbo,
                    VMA_MEMORY_USAGE_GPU_ONLY, prevPosBuffer_, prevPosAlloc_);
    prevPosOwned_ = true;
    // Seed with the base pose; render()'s firstMvFrame_ handling additionally
    // copies the first frame's own (post-morph) position into this buffer
    // before preprocess runs, so mv == 0 exactly on frame 1 regardless of
    // the initial animation time.
    VkDeviceSize sz = data.positions.size() * sizeof(float);
    VkBuffer staging = VK_NULL_HANDLE;
    VmaAllocation stagingAlloc = VK_NULL_HANDLE;
    createVmaBuffer(ctx.allocator, sz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                    VMA_MEMORY_USAGE_CPU_TO_GPU, staging, stagingAlloc);
    uploadVmaBuffer(ctx.allocator, stagingAlloc, data.positions.data(), sz);
    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
    VkBufferCopy copy = {0, 0, sz};
    vkCmdCopyBuffer(cmd, staging, prevPosBuffer_, 1, &copy);
    ctx.endSingleTimeCommands(cmd);
    vmaDestroyBuffer(ctx.allocator, staging, stagingAlloc);
}

float SplatRenderer::animationDuration() const {
    if (!dynamics_.has_motion || dynamics_.keyframes.empty()) return 0.0f;
    return dynamics_.keyframes.back().time;
}

float SplatRenderer::frameToTime(int frame) const {
    return splatFrameToTime(dynamics_, frame);
}

void SplatRenderer::setMorphTime(float seconds) {
    currentMorphTime_ = seconds;
}

void SplatRenderer::stepKeyframe(int direction) {
    if (!dynamics_.has_motion || dynamics_.keyframes.empty()) return;
    const auto& kf = dynamics_.keyframes;
    size_t nearest = 0;
    float best = std::fabs(kf[0].time - currentMorphTime_);
    for (size_t i = 1; i < kf.size(); i++) {
        float d = std::fabs(kf[i].time - currentMorphTime_);
        if (d < best) { best = d; nearest = i; }
    }
    long stepped = static_cast<long>(nearest) + direction;
    stepped = std::max<long>(0, std::min<long>(stepped, static_cast<long>(kf.size()) - 1));
    currentMorphTime_ = kf[static_cast<size_t>(stepped)].time;
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

    // DLSS input-contract outputs (docs/lux-4d-spec.md section 3): detected
    // once from the preprocess stage's reflection JSON.
    hasMotionVectors_ = readShaderBoolFlag(shaderBase, "motion_vectors");
    hasExpectedDepth_ = readShaderBoolFlag(shaderBase, "expected_depth");
    hasForegroundCoverage_ = readShaderBoolFlag(shaderBase, "foreground_coverage");
    if (hasMotionVectors_ || hasExpectedDepth_) {
        std::cout << "[info] DLSS outputs enabled: motion_vectors=" << hasMotionVectors_
                  << " expected_depth=" << hasExpectedDepth_
                  << " foreground_coverage=" << hasForegroundCoverage_ << std::endl;
    }

    createOffscreenTarget(ctx);
    createRenderPass(ctx.device);
    createRenderPassLoad(ctx.device);
    createRenderPassLoadDepth(ctx.device);
    createFramebuffer(ctx.device);
    createFramebufferLoad(ctx.device);
    createFramebufferLoadDepth(ctx.device);
    createPipelines(ctx, shaderBase);
    createSortPipelines(ctx.device);
    createBuffers(ctx, data);
    createMorphPipeline(ctx.device, shaderBase);
    createMorphBuffers(ctx, data);

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
            projMatrixUnjittered_ = projMatrix_;
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
            projMatrixUnjittered_ = projMatrix_;
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
    glm::mat4 proj = glm::perspective(fovY, aspect, nearPlane, farPlane);
    proj[1][1] *= -1.0f; // Vulkan Y-flip
    projMatrixUnjittered_ = proj;
    projMatrix_ = applySplatJitter(proj, jitterX_, jitterY_, width_, height_);

    // Focal lengths: fy = h/(2*tan(fov_y/2)), fx = fy for square pixels
    focalY_ = 0.5f * static_cast<float>(height_) / tanf(fovY * 0.5f);
    focalX_ = focalY_;
}

void SplatRenderer::updateCameraExplicit(glm::vec3 eye, glm::mat4 viewMatrix, glm::mat4 projMatrix,
                                          float focalX, float focalY) {
    camPos_ = eye;
    viewMatrix_ = viewMatrix;
    projMatrixUnjittered_ = projMatrix;
    projMatrix_ = applySplatJitter(projMatrix, jitterX_, jitterY_, width_, height_);
    focalX_ = focalX;
    focalY_ = focalY;
}

void SplatRenderer::setJitter(float jitterXPixels, float jitterYPixels) {
    jitterX_ = jitterXPixels;
    jitterY_ = jitterYPixels;
    // Re-derive the jittered matrix from the last-known unjittered one so
    // setJitter() can be called either before or after updateCamera().
    projMatrix_ = applySplatJitter(projMatrixUnjittered_, jitterX_, jitterY_, width_, height_);
}

// --------------------------------------------------------------------------
// Dynamic splats: morph-apply dispatch (shared by render() and
// seedPreviousMorphTime())
// --------------------------------------------------------------------------
//
// Two dispatches so arbitrary time scrubbing (not just monotonic playback)
// stays correct: (1) reset every gaussian that EVER moves back to base
// (weight 0/0 == base + 0*lo + 0*hi == base, exactly); (2) apply the
// currently active segment's weighted deltas. Both dispatch sizes are
// proportional to sparse counts, not numSplats_. See docs/lux-4d-spec.md
// and SPECIFICATION.md 12.8.
// Perf note (mobile-DLSS Android live-demo Stage 1 timing breakdown,
// docs/rendering-engines.md): this used to unconditionally "reset" the
// FULL concatenated morph_index array (every keyframe segment's sparse
// entries, summed across the whole clip -- num_keyframes * segment_size,
// e.g. 38 * 65,641 = 2.49M entries for the pruned juggle clip) on every
// single call, then "apply" just the current segment (65,641 entries) --
// i.e. ~38x more reset work than apply work, every frame, regardless of
// which keyframe was active. Measured cost on Pixel 9 Pro XL / Mali-G715
// (playground_android): ~13ms of the ~18ms "preprocess" GPU-timestamp
// window (which spans dispatchMorph() + the actual per-splat projection
// compute) was this reset pass -- the per-splat projection itself is only
// ~0.3-0.5ms for 119,826 splats.
//
// The full-array reset is unnecessary: the "apply" shader (splat_expander.
// py's _build_morph_apply_body) is NOT incremental -- it always computes
// `splat_pos[idx] = splat_base_pos[idx] + weight_lo*delta_lo + weight_hi*
// delta_hi` from the immutable base buffers, so any index the CURRENT
// apply touches is fully overwritten regardless of what reset did to it.
// Reset only matters for indices that are NOT touched by the current
// segment but hold a stale value some EARLIER apply call wrote. By
// induction, any such residue is always fully bounded to the single
// segment applied by the immediately preceding dispatchMorph() call
// (whether from render()'s per-frame playback, seedPreviousMorphTime()'s
// one-shot seed, or stepKeyframe()'s UI jump) -- every prior call already
// cleaned up ITS predecessor's residue the same way, so residue never
// accumulates past one step back. Resetting exactly that one segment
// (segmentCounts_[lastAppliedSegment_], not the whole concatenated array)
// is therefore both correct and ~num_keyframes times cheaper; when the
// segment doesn't change between calls (typical -- keyframes are sparser
// than the render frame rate) it's skipped entirely, since apply already
// overwrites the same indices a moment later.
void SplatRenderer::dispatchMorph(VkCommandBuffer cmd, float timeSeconds) {
    if (!hasMotion()) return;

    struct MorphPush {
        uint32_t segmentOffset;
        uint32_t segmentCount;
        float weightLow;
        float weightHigh;
    };

    VkMemoryBarrier morphBarrier = {};
    morphBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    morphBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    morphBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, morphPipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, morphLayout_,
                            0, 1, &morphDescSet_, 0, nullptr);

    SplatMorphState state = evaluateSplatMorphState(dynamics_, timeSeconds);
    bool hasActive = (state.weightLow != 0.0f || state.weightHigh != 0.0f) &&
                      state.highTargetIndex >= 0 &&
                      static_cast<size_t>(state.highTargetIndex) < segmentCounts_.size() &&
                      segmentCounts_[state.highTargetIndex] > 0;
    int curSeg = hasActive ? state.highTargetIndex : -1;

    if (lastAppliedSegment_ >= 0 && lastAppliedSegment_ != curSeg) {
        uint32_t seg = static_cast<uint32_t>(lastAppliedSegment_);
        if (segmentCounts_[seg] > 0) {
            MorphPush resetPush = {segmentOffsets_[seg], segmentCounts_[seg], 0.0f, 0.0f};
            vkCmdPushConstants(cmd, morphLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(resetPush), &resetPush);
            uint32_t resetGroups = (segmentCounts_[seg] + 255) / 256;
            vkCmdDispatch(cmd, resetGroups, 1, 1);
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 1, &morphBarrier, 0, nullptr, 0, nullptr);
        }
    }

    if (hasActive) {
        uint32_t seg = static_cast<uint32_t>(curSeg);
        MorphPush applyPush = {segmentOffsets_[seg], segmentCounts_[seg], state.weightLow, state.weightHigh};
        vkCmdPushConstants(cmd, morphLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(applyPush), &applyPush);
        uint32_t applyGroups = (segmentCounts_[seg] + 255) / 256;
        vkCmdDispatch(cmd, applyGroups, 1, 1);
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &morphBarrier, 0, nullptr, 0, nullptr);
    }

    lastAppliedSegment_ = curSeg;
}

// --------------------------------------------------------------------------
// Motion vectors: seed splat_prev_pos from a real previous-time morph
// evaluation (docs/lux-4d-spec.md section 3's --time-prev/--frame-prev
// follow-up)
// --------------------------------------------------------------------------
//
// Needed ONLY on the very first frame (no render() call has happened yet
// to establish "prev = last frame's current" via its own GPU ping-pong
// copy -- see render()'s "carry history forward for the NEXT frame"
// posBuffer_->prevPosBuffer_ copy below) or for an explicit one-shot
// reseed, such as the headless CLI's --time-prev/--frame-prev flags
// (called exactly once, before the first render()). Both of those cases
// are characterized by firstMvFrame_ still being true when this is
// called, which the guard below keys off. After the first render() call,
// prevPosBuffer_ is ALREADY correctly refreshed every frame by render()'s
// own copy (no extra command buffer) -- so a continuous per-frame caller
// that (redundantly) calls this every frame alongside render()'s own
// per-frame dispatchMorph(cmd, currentMorphTime_) now pays nothing beyond
// the one-time first-frame cost, instead of an extra full blocking
// beginSingleTimeCommands/endSingleTimeCommands GPU round trip on every
// frame (the analogous fix to metal_splat_luxc_renderer.h's
// seedPreviousMorphTime -- see its comment for the measured M1 iPad cost
// of this same redundant-call pattern on that backend).
void SplatRenderer::seedPreviousMorphTime(VulkanContext& ctx, float prevTimeSeconds) {
    if (!hasMotion() || !hasMotionVectors_ || !prevPosOwned_ || !firstMvFrame_) return;

    // Previous camera hasn't been explicitly seeded yet (i.e. no
    // setPreviousCameraExplicit()/--camera-json-prev call happened before
    // this one): default it to the CURRENT camera -- same "prev == curr"
    // convention render() itself uses on frame 1 -- so mv reflects pure
    // actor motion, not a spurious jump from an unset (identity) previous
    // camera. Call setPreviousCameraExplicit() *before* this one to get
    // real previous-camera motion too.
    prevViewMatrix_ = viewMatrix_;
    prevProjMatrixUnjittered_ = projMatrixUnjittered_;

    // Evaluate the morph at prevTimeSeconds into the working splat_pos
    // buffer (scratch space here -- render() unconditionally re-dispatches
    // the morph for the *current* time on every call, so whatever this
    // leaves behind gets overwritten immediately after), then copy the
    // result into prevPosBuffer_.
    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
    dispatchMorph(cmd, prevTimeSeconds);

    VkMemoryBarrier toTransfer = {};
    toTransfer.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    toTransfer.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    toTransfer.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 1, &toTransfer, 0, nullptr, 0, nullptr);

    VkBufferCopy copy = {0, 0, static_cast<VkDeviceSize>(numSplats_) * 4 * sizeof(float)};
    vkCmdCopyBuffer(cmd, posBuffer_, prevPosBuffer_, 1, &copy);
    ctx.endSingleTimeCommands(cmd);

    // A real previous position has now been seeded; don't let render()'s
    // first-frame auto-seed (prev == curr) clobber it.
    firstMvFrame_ = false;
}

// --------------------------------------------------------------------------
// Render
// --------------------------------------------------------------------------

void SplatRenderer::render(VulkanContext& ctx) {
    if (numSplats_ == 0) return;

    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();

    if (gpuTimingEnabled_) {
        vkCmdResetQueryPool(cmd, timestampPool_, 0, 4);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timestampPool_, 0);
    }

    // --- Motion vectors: seed history on the very first frame so mv == 0 ---
    // (docs/lux-4d-spec.md section 3). Seeding here (not at construction)
    // means it doesn't matter how many updateCamera()/setMorphTime() calls
    // happened before the first render() -- "current" always equals "prev"
    // for that first call, regardless of the starting camera or animation
    // time.
    if (hasMotionVectors_ && firstMvFrame_) {
        prevViewMatrix_ = viewMatrix_;
        prevProjMatrixUnjittered_ = projMatrixUnjittered_;
    }

    // --- Dynamic splats: morph-apply compute pass (runs before preprocess) ---
    dispatchMorph(cmd, currentMorphTime_);

    // Debug: dump the first N post-morph splat attributes (position,
    // rotation xyzw, scale log-space, opacity logit-space, sh0) as fed to
    // the preprocess compute stage -- for diffing against the Metal
    // backends' equivalent dump (LUX_DEBUG_SPLAT_DUMP env var on all three
    // backends). Forces an early command-buffer submit+wait here (like the
    // existing hasMotionVectors_ one below, just earlier) purely for this
    // diagnostic readback -- posBuffer_/rotBuffer_/shBuffers_[0] are
    // VMA_MEMORY_USAGE_CPU_TO_GPU (host-visible), and dispatchMorph()
    // writes them in place.
    if (const char* dumpPath = std::getenv("LUX_DEBUG_SPLAT_DUMP")) {
        ctx.endSingleTimeCommands(cmd);
        cmd = ctx.beginSingleTimeCommands();

        uint32_t n = std::min<uint32_t>(16, numSplats_);
        std::ofstream out(dumpPath);
        out << "# backend=vulkan n=" << n << " time=" << currentMorphTime_ << "\n";
        void* mapped = nullptr;

        vmaMapMemory(ctx.allocator, posAlloc_, &mapped);
        auto* pos = static_cast<float*>(mapped);
        for (uint32_t i = 0; i < n; ++i)
            out << "pos " << i << " " << pos[i*4] << " " << pos[i*4+1] << " " << pos[i*4+2] << " " << pos[i*4+3] << "\n";
        vmaUnmapMemory(ctx.allocator, posAlloc_);

        vmaMapMemory(ctx.allocator, rotAlloc_, &mapped);
        auto* rot = static_cast<float*>(mapped);
        for (uint32_t i = 0; i < n; ++i)
            out << "rot_xyzw " << i << " " << rot[i*4] << " " << rot[i*4+1] << " " << rot[i*4+2] << " " << rot[i*4+3] << "\n";
        vmaUnmapMemory(ctx.allocator, rotAlloc_);

        vmaMapMemory(ctx.allocator, scaleAlloc_, &mapped);
        auto* scl = static_cast<float*>(mapped);
        for (uint32_t i = 0; i < n; ++i)
            out << "scale_log " << i << " " << scl[i*4] << " " << scl[i*4+1] << " " << scl[i*4+2] << "\n";
        vmaUnmapMemory(ctx.allocator, scaleAlloc_);

        vmaMapMemory(ctx.allocator, opacityAlloc_, &mapped);
        auto* op = static_cast<float*>(mapped);
        for (uint32_t i = 0; i < n; ++i)
            out << "opacity_logit " << i << " " << op[i] << "\n";
        vmaUnmapMemory(ctx.allocator, opacityAlloc_);

        if (!shAllocs_.empty()) {
            vmaMapMemory(ctx.allocator, shAllocs_[0], &mapped);
            auto* sh0 = static_cast<float*>(mapped);
            for (uint32_t i = 0; i < n; ++i)
                out << "sh0 " << i << " " << sh0[i*4] << " " << sh0[i*4+1] << " " << sh0[i*4+2] << "\n";
            vmaUnmapMemory(ctx.allocator, shAllocs_[0]);
        }
        std::cout << "[debug-splat] dumped " << n << " splats to " << dumpPath << std::endl;
    }

    // --- Motion vectors: seed splat_prev_pos on the very first frame ---
    // (dynamic splats only -- static splats' prevPosBuffer_ already aliases
    // posBuffer_, so curr == prev trivially). Copies THIS frame's own
    // (post-morph) position into prevPosBuffer_ before preprocess reads it,
    // so the position term of mv is also exactly 0 on frame 1.
    if (hasMotionVectors_ && firstMvFrame_ && prevPosOwned_) {
        // dispatchMorph()'s own trailing barrier above only covers
        // COMPUTE_SHADER_BIT -> COMPUTE_SHADER_BIT (for the reset->apply
        // sequencing within dispatchMorph itself); the copy below reads
        // posBuffer_ via the TRANSFER stage, which that barrier's dstStage
        // does NOT include. Missing this meant the copy's read of
        // posBuffer_ was not guaranteed to observe the morph compute
        // shader's writes, an actual (confirmed, reproducible) source of
        // intermittent wrong/garbage splat_prev_pos data on the very first
        // rendered frame of a dynamic-splat DLSS pipeline.
        VkMemoryBarrier toTransfer = {};
        toTransfer.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        toTransfer.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        toTransfer.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 1, &toTransfer, 0, nullptr, 0, nullptr);

        VkBufferCopy copy = {0, 0, static_cast<VkDeviceSize>(numSplats_) * 4 * sizeof(float)};
        vkCmdCopyBuffer(cmd, posBuffer_, prevPosBuffer_, 1, &copy);
        VkMemoryBarrier copyBarrier = {};
        copyBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        copyBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        copyBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &copyBarrier, 0, nullptr, 0, nullptr);
    }

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
        float _pad1[2];       // offset 168 (pad to 176, portable across
                               // every backend's maxPushConstantsSize --
                               // see the guardrail check in createPipelines
                               // and prevCameraBuffer_'s comment in the
                               // header for why proj_matrix_unjittered/
                               // prev_view_proj_unjittered do NOT live here)
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

    // prev_camera_mats[0]=proj_matrix_unjittered, [1]=prev_view_proj_unjittered
    // (splat_expander.py) -- written via vkCmdUpdateBuffer (a small inline
    // transfer recorded directly in this command buffer, <=65536 bytes per
    // Vulkan's limit; ours is 128) followed by a barrier gating the
    // preprocess dispatch below on that transfer completing, exactly
    // mirroring how this data used to reach the shader via push constants.
    if (hasMotionVectors_) {
        glm::mat4 prevCameraMats[2];
        prevCameraMats[0] = projMatrixUnjittered_;
        prevCameraMats[1] = prevProjMatrixUnjittered_ * prevViewMatrix_;
        vkCmdUpdateBuffer(cmd, prevCameraBuffer_, 0, sizeof(prevCameraMats), prevCameraMats);

        VkMemoryBarrier updateBarrier = {};
        updateBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        updateBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        updateBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &updateBarrier, 0, nullptr, 0, nullptr);
    }

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

    if (gpuTimingEnabled_) {
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, timestampPool_, 1);
    }

    // Fully drain the queue here when motion vectors are enabled.
    //
    // This preprocess compute dispatch writes projected_mv (among other
    // projected_* buffers) which the vertex shader reads much later in this
    // same command buffer, after the 4-pass GPU radix sort. In-buffer
    // pipeline barriers (including, empirically, one broadened to
    // VK_PIPELINE_STAGE_ALL_COMMANDS_BIT / VK_ACCESS_MEMORY_WRITE_BIT right
    // before the render pass) were NOT sufficient to make this reliable on
    // MoltenVK/Apple Silicon: projected_center/conic/color were always
    // read correctly (verified bit-identical output across dozens of
    // repeated runs), but projected_mv was intermittently stale/wrong in a
    // small number of runs despite the preprocess dispatch's own output
    // being independently verified deterministic and correct at this exact
    // point (docs/lux-4d-spec.md section 3's --time-prev follow-up
    // debugging). A full queue drain right here -- ending and restarting
    // the command buffer, forcing MoltenVK to emit a real, separate Metal
    // command buffer boundary rather than a mid-buffer compute-encoder
    // fence -- reliably eliminated the flakiness (repeated stress runs
    // showed 0 failures after this change vs. a consistent ~30-50% failure
    // rate before it). This costs an extra CPU-GPU round trip per frame,
    // paid only when motion vectors are enabled.
    if (hasMotionVectors_) {
        ctx.endSingleTimeCommands(cmd);
        cmd = ctx.beginSingleTimeCommands();
    }

    // --- Motion vectors: carry history forward for the NEXT frame ---
    // Preprocess has now consumed this frame's prevPosBuffer_/prev camera
    // matrices; safe to overwrite. Each render() call is fully
    // GPU-synchronous (endSingleTimeCommands waits for completion below),
    // so no cross-frame synchronization beyond this command buffer's own
    // barriers is needed.
    if (hasMotionVectors_) {
        if (prevPosOwned_) {
            VkBufferCopy copy = {0, 0, static_cast<VkDeviceSize>(numSplats_) * 4 * sizeof(float)};
            vkCmdCopyBuffer(cmd, posBuffer_, prevPosBuffer_, 1, &copy);
        }
        prevViewMatrix_ = viewMatrix_;
        prevProjMatrixUnjittered_ = projMatrixUnjittered_;
        firstMvFrame_ = false;
    }

    // --- Sort scheduling decision (see setSortSchedule()'s header comment) ---
    bool needsSort = true;
    if (sortEveryNFrames_ > 1 || sortViewThresholdDeg_ > 0.0f) {
        glm::vec3 viewDir = glm::normalize(
            -glm::vec3(viewMatrix_[0][2], viewMatrix_[1][2], viewMatrix_[2][2]));
        bool viewChanged = true;
        if (hasLastSortedView_ && sortViewThresholdDeg_ > 0.0f) {
            float cosAngle = glm::clamp(glm::dot(viewDir, lastSortedViewDir_), -1.0f, 1.0f);
            float angleDeg = glm::degrees(std::acos(cosAngle));
            viewChanged = angleDeg >= sortViewThresholdDeg_;
        }
        bool budgetElapsed = framesSinceSort_ >= sortEveryNFrames_;
        needsSort = !hasLastSortedView_ || budgetElapsed ||
            (sortViewThresholdDeg_ > 0.0f && viewChanged);
        if (needsSort) {
            framesSinceSort_ = 0;
            lastSortedViewDir_ = viewDir;
            hasLastSortedView_ = true;
        } else {
            framesSinceSort_++;
        }
    }

    // --- GPU Radix Sort (4 passes, 8 bits per pass = 32-bit keys) ---
    // Skipped entirely on frames the schedule above decides don't need a
    // fresh order -- sortedIndicesBuffer_/sortKeysBuffer_ simply keep
    // whatever the last real sort left in them (GPU_ONLY memory, never
    // implicitly cleared). Splat VISIBILITY is unaffected either way (a
    // per-splat decision made in preprocess/fragment, not by sort
    // position) -- only back-to-front BLEND ORDER can be briefly stale.
    if (needsSort) {
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

    if (gpuTimingEnabled_) {
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, timestampPool_, 2);
    }

    // Barrier: sort compute -> vertex/fragment shader reads.
    //
    // srcStageMask=ALL_COMMANDS (not just COMPUTE_SHADER_BIT) is deliberate:
    // this needs to cover not only the radix sort's own writes but also the
    // *original* preprocess dispatch's writes to projected_center/conic/
    // color/mv/depth from much earlier in this same command buffer. Those
    // are already nominally covered by the compute->compute barriers chained
    // through the sort passes above, but MoltenVK has been observed to not
    // reliably propagate that chained dependency all the way through when a
    // LATER barrier changes destination stage from COMPUTE_SHADER_BIT to
    // VERTEX_SHADER_BIT|FRAGMENT_SHADER_BIT -- in practice this showed up as
    // an intermittent (run-to-run nondeterministic) stale/garbage read of
    // projected_mv specifically by the vertex shader, even though the
    // preprocess compute's OWN output was verified bit-identical across
    // runs (docs/lux-4d-spec.md section 3's --time-prev follow-up
    // debugging). Using ALL_COMMANDS_BIT as the source stage is the
    // conservative fix: it makes this barrier depend on literally
    // everything recorded earlier in the command buffer, not just the
    // stage-matching subset, closing the gap regardless of its exact
    // driver-level root cause.
    barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         0, 1, &barrier, 0, nullptr, 0, nullptr);

    // --- Render pass ---
    // Use LOAD render pass when a background has been blitted in (hybrid
    // compositing) -- never true together with the DLSS outputs, see
    // preloadBackground()/preloadDepth().
    VkRenderPassBeginInfo rpBegin = {};
    rpBegin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    std::vector<VkClearValue> clearValues;
    if (hasBackgroundDepth_) {
        // Full hybrid: both color and depth loaded from raster pass
        rpBegin.renderPass = renderPassLoadDepth_;
        rpBegin.framebuffer = framebufferLoadDepth_;
        clearValues = {VkClearValue{}, VkClearValue{}};
        clearValues[1].depthStencil = {1.0f, 0};
    } else if (hasBackground_) {
        // Color-only compositing: color loaded, depth cleared
        rpBegin.renderPass = renderPassLoad_;
        rpBegin.framebuffer = framebufferLoad_;
        clearValues = {VkClearValue{}, VkClearValue{}};
        clearValues[1].depthStencil = {1.0f, 0};
    } else {
        rpBegin.renderPass = renderPass_;
        rpBegin.framebuffer = framebuffer_;
        // Order matches createFramebuffer(): color, [out_aux], [out_fg], depth_test.
        // out_aux/out_fg clear to 0 -- "0 where nothing was drawn" per
        // docs/lux-4d-spec.md section 3. Color's alpha is cleared to 0 (not
        // the usual opaque-black 1.0) whenever the DLSS outputs are enabled:
        // the host needs color.a to be a genuine "how much splat coverage
        // landed in this pixel" signal (0..1, accumulated via the same
        // premultiplied-alpha blend as the aux attachments) to correctly
        // un-premultiply motion/depth -- an opaque background alpha would
        // saturate it to 1 everywhere regardless of actual splat coverage.
        // Non-DLSS pipelines are unaffected (alpha stays 1, unchanged).
        clearValues.push_back(VkClearValue{});
        clearValues.back().color = {{0.0f, 0.0f, 0.0f, (hasMotionVectors_ || hasExpectedDepth_) ? 0.0f : 1.0f}};
        if (hasMotionVectors_ || hasExpectedDepth_) {
            clearValues.push_back(VkClearValue{});
            clearValues.back().color = {{0.0f, 0.0f, 0.0f, 0.0f}};
        }
        if (hasForegroundCoverage_) {
            clearValues.push_back(VkClearValue{});
            clearValues.back().color = {{0.0f, 0.0f, 0.0f, 0.0f}};
        }
        clearValues.push_back(VkClearValue{});
        clearValues.back().depthStencil = {1.0f, 0};
    }
    rpBegin.renderArea.extent = {width_, height_};
    rpBegin.clearValueCount = static_cast<uint32_t>(clearValues.size());
    rpBegin.pClearValues = clearValues.data();

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
    // alpha_min default (3DGS/gsplat convention, SPECIFICATION.md 12.8): 1/255.
    // NOTE: like the pre-existing alpha_cutoff this replaces, this value is
    // NOT read back from the compiled splat's `alpha_min`/`alpha_cutoff`
    // config (that field only feeds reflection metadata) -- it's a fixed
    // host-side default, unchanged from that prior architecture.
    renderPush.alphaCutoff = 1.0f / 255.0f;
    vkCmdPushConstants(cmd, renderLayout_,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(renderPush), &renderPush);

    // Instanced draw: 6 vertices (quad) x numSplats instances
    vkCmdDraw(cmd, 6, numSplats_, 0, 0);

    vkCmdEndRenderPass(cmd);

    if (gpuTimingEnabled_) {
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timestampPool_, 3);
    }

    // Submit and wait
    ctx.endSingleTimeCommands(cmd);

    if (gpuTimingEnabled_) {
        uint64_t ts[4] = {};
        VkResult qr = vkGetQueryPoolResults(ctx.device, timestampPool_, 0, 4, sizeof(ts), ts, sizeof(uint64_t),
                                            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
        if (qr == VK_SUCCESS) {
            lastGpuTimings_.preprocessMs = static_cast<double>(ts[1] - ts[0]) * timestampPeriodNs_ * 1e-6;
            lastGpuTimings_.sortMs = static_cast<double>(ts[2] - ts[1]) * timestampPeriodNs_ * 1e-6;
            lastGpuTimings_.drawMs = static_cast<double>(ts[3] - ts[2]) * timestampPeriodNs_ * 1e-6;
            lastGpuTimings_.valid = true;
        } else {
            lastGpuTimings_.valid = false;
        }
    }
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
    if (hasMotionVectors_ || hasExpectedDepth_) {
        // Hybrid mesh+splat compositing (LOAD render pass, 2 attachments)
        // isn't supported together with the extra DLSS output attachments
        // in this iteration -- render() always uses the primary N-attachment
        // render pass when either flag is set. See docs/lux-4d-spec.md
        // section 3 and splat_renderer.h's createRenderPass comment.
        std::cout << "[warn] preloadBackground() ignored: motion_vectors/"
                     "expected_depth splats don't support background compositing yet."
                  << std::endl;
        return;
    }
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
    if (hasMotionVectors_ || hasExpectedDepth_) {
        std::cout << "[warn] preloadDepth() ignored: motion_vectors/"
                     "expected_depth splats don't support background compositing yet."
                  << std::endl;
        return;
    }
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

    if (timestampPool_ != VK_NULL_HANDLE) {
        vkDestroyQueryPool(ctx.device, timestampPool_, nullptr);
        timestampPool_ = VK_NULL_HANDLE;
    }

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
    if (auxView_ != VK_NULL_HANDLE) vkDestroyImageView(ctx.device, auxView_, nullptr);
    if (fgView_ != VK_NULL_HANDLE) vkDestroyImageView(ctx.device, fgView_, nullptr);

    // Images (VMA)
    if (colorImage_ != VK_NULL_HANDLE) vmaDestroyImage(ctx.allocator, colorImage_, colorAlloc_);
    if (depthImage_ != VK_NULL_HANDLE) vmaDestroyImage(ctx.allocator, depthImage_, depthAlloc_);
    if (auxImage_ != VK_NULL_HANDLE) vmaDestroyImage(ctx.allocator, auxImage_, auxAlloc_);
    if (fgImage_ != VK_NULL_HANDLE) vmaDestroyImage(ctx.allocator, fgImage_, fgAlloc_);

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
    destroyVmaBuffer(ctx.allocator, projAxesBuffer_, projAxesAlloc_);
    destroyVmaBuffer(ctx.allocator, projConicBuffer_, projConicAlloc_);
    destroyVmaBuffer(ctx.allocator, projColorBuffer_, projColorAlloc_);
    destroyVmaBuffer(ctx.allocator, projMvBuffer_, projMvAlloc_);
    destroyVmaBuffer(ctx.allocator, prevCameraBuffer_, prevCameraAlloc_);
    destroyVmaBuffer(ctx.allocator, projDepthBuffer_, projDepthAlloc_);
    destroyVmaBuffer(ctx.allocator, foregroundBuffer_, foregroundAlloc_);
    destroyVmaBuffer(ctx.allocator, projForegroundBuffer_, projForegroundAlloc_);
    if (prevPosOwned_) {
        destroyVmaBuffer(ctx.allocator, prevPosBuffer_, prevPosAlloc_);
    }
    prevPosBuffer_ = VK_NULL_HANDLE;
    prevPosOwned_ = false;
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

    // Dynamic splats: morph-apply pipeline + buffers
    if (morphPipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(ctx.device, morphPipeline_, nullptr);
    if (morphLayout_ != VK_NULL_HANDLE)   vkDestroyPipelineLayout(ctx.device, morphLayout_, nullptr);
    if (morphSetLayout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(ctx.device, morphSetLayout_, nullptr);
    morphPipeline_ = VK_NULL_HANDLE;
    morphLayout_ = VK_NULL_HANDLE;
    morphSetLayout_ = VK_NULL_HANDLE;
    morphDescSet_ = VK_NULL_HANDLE;
    destroyVmaBuffer(ctx.allocator, baseposBuffer_, baseposAlloc_);
    destroyVmaBuffer(ctx.allocator, baserotBuffer_, baserotAlloc_);
    destroyVmaBuffer(ctx.allocator, basesh0Buffer_, basesh0Alloc_);
    destroyVmaBuffer(ctx.allocator, morphIndexBuffer_, morphIndexAlloc_);
    destroyVmaBuffer(ctx.allocator, morphPosLoBuffer_, morphPosLoAlloc_);
    destroyVmaBuffer(ctx.allocator, morphRotLoBuffer_, morphRotLoAlloc_);
    destroyVmaBuffer(ctx.allocator, morphSh0LoBuffer_, morphSh0LoAlloc_);
    destroyVmaBuffer(ctx.allocator, morphPosHiBuffer_, morphPosHiAlloc_);
    destroyVmaBuffer(ctx.allocator, morphRotHiBuffer_, morphRotHiAlloc_);
    destroyVmaBuffer(ctx.allocator, morphSh0HiBuffer_, morphSh0HiAlloc_);

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
    auxView_ = VK_NULL_HANDLE;
    auxImage_ = VK_NULL_HANDLE;
    fgView_ = VK_NULL_HANDLE;
    fgImage_ = VK_NULL_HANDLE;
    hasMotionVectors_ = false;
    hasExpectedDepth_ = false;
    hasForegroundCoverage_ = false;
    firstMvFrame_ = true;
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
