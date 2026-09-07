#include "input_assembly.h"
#include "vulkan_context.h"
#include "spv_loader.h"
#include "dlss_io.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {

VkBuffer createBuffer(VmaAllocator allocator, VkDeviceSize sizeBytes, VkBufferUsageFlags usage,
                       VmaAllocation& allocOut) {
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = std::max<VkDeviceSize>(sizeBytes, 16);
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;  // host-visible + device-accessible (UMA)
    VkBuffer buf = VK_NULL_HANDLE;
    if (vmaCreateBuffer(allocator, &info, &allocInfo, &buf, &allocOut, nullptr) != VK_SUCCESS) {
        throw std::runtime_error("input_assembly: failed to create buffer");
    }
    return buf;
}

void uploadFloats(VmaAllocator allocator, VmaAllocation alloc, const std::vector<float>& data) {
    void* mapped = nullptr;
    vmaMapMemory(allocator, alloc, &mapped);
    if (!data.empty()) std::memcpy(mapped, data.data(), data.size() * sizeof(float));
    vmaUnmapMemory(allocator, alloc);
}

void zeroBuffer(VmaAllocator allocator, VmaAllocation alloc, size_t numBytes) {
    void* mapped = nullptr;
    vmaMapMemory(allocator, alloc, &mapped);
    std::memset(mapped, 0, numBytes);
    vmaUnmapMemory(allocator, alloc);
}

VkDescriptorSetLayout makeSetLayout(VkDevice device, uint32_t numStorageBuffers) {
    std::vector<VkDescriptorSetLayoutBinding> bindings(numStorageBuffers);
    for (uint32_t i = 0; i < numStorageBuffers; ++i) {
        bindings[i] = {};
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    info.bindingCount = numStorageBuffers;
    info.pBindings = bindings.data();
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    vkCreateDescriptorSetLayout(device, &info, nullptr, &layout);
    return layout;
}

VkPipelineLayout makePipelineLayout(VkDevice device, VkDescriptorSetLayout setLayout, uint32_t pushConstantSize) {
    VkPushConstantRange range{};
    range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    range.offset = 0;
    range.size = pushConstantSize;
    VkPipelineLayoutCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    info.setLayoutCount = 1;
    info.pSetLayouts = &setLayout;
    info.pushConstantRangeCount = 1;
    info.pPushConstantRanges = &range;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    vkCreatePipelineLayout(device, &info, nullptr, &layout);
    return layout;
}

VkPipeline makeComputePipeline(VkDevice device, VkPipelineLayout layout, const std::string& spvPath) {
    auto code = SpvLoader::loadSPIRV(spvPath);
    VkShaderModule module = SpvLoader::createShaderModule(device, code);
    VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    info.stage.module = module;
    info.stage.pName = "main";
    info.layout = layout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline) != VK_SUCCESS) {
        throw std::runtime_error("input_assembly: failed to create compute pipeline for " + spvPath);
    }
    vkDestroyShaderModule(device, module, nullptr);
    return pipeline;
}

void writeDescriptorSet(VkDevice device, VkDescriptorSet set, const std::vector<VkBuffer>& buffers) {
    std::vector<VkDescriptorBufferInfo> infos(buffers.size());
    std::vector<VkWriteDescriptorSet> writes(buffers.size());
    for (size_t i = 0; i < buffers.size(); ++i) {
        infos[i] = {buffers[i], 0, VK_WHOLE_SIZE};
        writes[i] = {};
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = static_cast<uint32_t>(i);
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

// Pulls raw texel bytes off an already-TRANSFER_SRC_OPTIMAL image straight
// into a persistent storage buffer (no separate staging buffer / host
// round-trip -- the destination buffer IS host-visible/coherent already,
// see createBuffer's VMA_MEMORY_USAGE_CPU_TO_GPU).
void copyImageToBuffer(VulkanContext& ctx, VkImage image, VkBuffer buffer,
                        uint32_t width, uint32_t height) {
    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {width, height, 1};
    vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, 1, &region);
    ctx.endSingleTimeCommands(cmd);
}

void dispatchOne(VulkanContext& ctx, VkPipeline pipeline, VkPipelineLayout layout, VkDescriptorSet set,
                  const void* pushData, uint32_t pushSize, uint32_t groupsX, uint32_t groupsY) {
    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
    if (pushSize > 0) vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushSize, pushData);
    vkCmdDispatch(cmd, groupsX, groupsY, 1);
    ctx.endSingleTimeCommands(cmd);
}

// Must byte-match the GLSL push_constant block in
// shaders_glsl/input_assembly_assemble.comp exactly (10 x vec4/uvec4 = 160B,
// comfortably under the Mali-confirmed 256B push-constant limit --
// docs/rendering-engines.md's "root-cause the MV corruption" commit).
struct AssemblePush {
    float eye[4];
    float rAxis[4];
    float uAxis[4];
    float fAxis[4];
    float bgCenter[4];
    float camParams[4];  // fx, fy, cx, cy
    float extra[4];      // bgRadius, mvScale, jitterX, jitterY
    uint32_t dims1[4];   // proxyW, proxyH, netW, netH
    uint32_t dims2[4];   // paramStride, hiddenChannels, texW, texH
    uint32_t flags[4];   // firstFrame, texChannels, fgSource, pad
};
static_assert(sizeof(AssemblePush) == 160, "AssemblePush must match input_assembly_assemble.comp's push_constant block");

struct UnpremulPush {
    uint32_t width, height;
};

}  // namespace

struct InputAssembly::Impl {
    VkDescriptorSetLayout unpremulLayout = VK_NULL_HANDLE, assembleLayout = VK_NULL_HANDLE;
    VkPipelineLayout unpremulPL = VK_NULL_HANDLE, assemblePL = VK_NULL_HANDLE;
    VkPipeline unpremulPipe = VK_NULL_HANDLE, assemblePipe = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet unpremulSet = VK_NULL_HANDLE, assembleSet = VK_NULL_HANDLE;

    VkBuffer bColorRaw = VK_NULL_HANDLE, bMotionRaw = VK_NULL_HANDLE, bDepthRaw = VK_NULL_HANDLE;
    VmaAllocation aColorRaw{}, aMotionRaw{}, aDepthRaw{};

    VkBuffer bDepthPing[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VmaAllocation aDepthPing[2]{};
    int depthPingIndex = 0;

    // Current frame's un-premultiplied foreground coverage (out_depth's .g
    // channel) -- no ping-pong needed, build_input pools only this frame's
    // fg (see input_assembly_unpremul_depth.comp's header comment).
    VkBuffer bFgCur = VK_NULL_HANDLE;
    VmaAllocation aFgCur{};

    VkBuffer bTexture = VK_NULL_HANDLE;
    VmaAllocation aTexture{};

    VkBuffer bZeroHidden = VK_NULL_HANDLE;
    VmaAllocation aZeroHidden{};

    VkBuffer bOutput = VK_NULL_HANDLE;
    VmaAllocation aOutput{};

    VulkanContext* ctx = nullptr;
    void* mappedOutput = nullptr;
    void* mappedDepthPing[2] = {nullptr, nullptr};
};

InputAssembly::~InputAssembly() {
    if (!impl_) return;
    VkDevice device = impl_->ctx ? impl_->ctx->device : VK_NULL_HANDLE;
    if (device) {
        vkDeviceWaitIdle(device);
        if (impl_->unpremulPipe) vkDestroyPipeline(device, impl_->unpremulPipe, nullptr);
        if (impl_->assemblePipe) vkDestroyPipeline(device, impl_->assemblePipe, nullptr);
        if (impl_->unpremulPL) vkDestroyPipelineLayout(device, impl_->unpremulPL, nullptr);
        if (impl_->assemblePL) vkDestroyPipelineLayout(device, impl_->assemblePL, nullptr);
        if (impl_->unpremulLayout) vkDestroyDescriptorSetLayout(device, impl_->unpremulLayout, nullptr);
        if (impl_->assembleLayout) vkDestroyDescriptorSetLayout(device, impl_->assembleLayout, nullptr);
        if (impl_->pool) vkDestroyDescriptorPool(device, impl_->pool, nullptr);
        auto& alloc = impl_->ctx->allocator;
        if (impl_->mappedOutput) vmaUnmapMemory(alloc, impl_->aOutput);
        if (impl_->mappedDepthPing[0]) vmaUnmapMemory(alloc, impl_->aDepthPing[0]);
        if (impl_->mappedDepthPing[1]) vmaUnmapMemory(alloc, impl_->aDepthPing[1]);
        if (impl_->bColorRaw) vmaDestroyBuffer(alloc, impl_->bColorRaw, impl_->aColorRaw);
        if (impl_->bMotionRaw) vmaDestroyBuffer(alloc, impl_->bMotionRaw, impl_->aMotionRaw);
        if (impl_->bDepthRaw) vmaDestroyBuffer(alloc, impl_->bDepthRaw, impl_->aDepthRaw);
        if (impl_->bDepthPing[0]) vmaDestroyBuffer(alloc, impl_->bDepthPing[0], impl_->aDepthPing[0]);
        if (impl_->bDepthPing[1]) vmaDestroyBuffer(alloc, impl_->bDepthPing[1], impl_->aDepthPing[1]);
        if (impl_->bFgCur) vmaDestroyBuffer(alloc, impl_->bFgCur, impl_->aFgCur);
        if (impl_->bTexture) vmaDestroyBuffer(alloc, impl_->bTexture, impl_->aTexture);
        if (impl_->bZeroHidden) vmaDestroyBuffer(alloc, impl_->bZeroHidden, impl_->aZeroHidden);
        if (impl_->bOutput) vmaDestroyBuffer(alloc, impl_->bOutput, impl_->aOutput);
    }
    delete impl_;
}

void InputAssembly::init(VulkanContext& ctx, const std::string& textureNpyPath,
                          const std::string& bgSphereNpyPath, uint32_t proxyW, uint32_t proxyH,
                          uint32_t paramStride, uint32_t hiddenChannels,
                          const std::string& shaderDir) {
    impl_ = new Impl();
    impl_->ctx = &ctx;
    proxyW_ = proxyW;
    proxyH_ = proxyH;
    paramStride_ = paramStride;
    hiddenChannels_ = hiddenChannels;

    auto roundUp = [](uint32_t x, uint32_t m) { return ((x + m - 1) / m) * m; };
    uint32_t multiple = 8 * paramStride;
    uint32_t paddedProxyW = roundUp(proxyW, multiple);
    uint32_t paddedProxyH = roundUp(proxyH, multiple);
    netW_ = paddedProxyW / paramStride;
    netH_ = paddedProxyH / paramStride;

    DlssIO::NpyArray texArr = DlssIO::readNpyFloat32(textureNpyPath);
    if (texArr.shape.size() != 3) throw std::runtime_error("texture.npy: expected 3D [C,H,W]");
    texChannels_ = static_cast<uint32_t>(texArr.shape[0]);
    texH_ = static_cast<uint32_t>(texArr.shape[1]);
    texW_ = static_cast<uint32_t>(texArr.shape[2]);
    if (texChannels_ > 8) throw std::runtime_error("texture.npy: >8 channels not supported by the .comp kernel");

    DlssIO::NpyArray sphereArr = DlssIO::readNpyFloat32(bgSphereNpyPath);
    if (sphereArr.data.size() != 4) throw std::runtime_error("bg_sphere.npy: expected 4 floats");
    bgSphere_.assign(sphereArr.data.begin(), sphereArr.data.end());

    VmaAllocator alloc = ctx.allocator;
    impl_->bTexture = createBuffer(alloc, texArr.data.size() * sizeof(float),
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, impl_->aTexture);
    uploadFloats(alloc, impl_->aTexture, texArr.data);

    const size_t proxyN = static_cast<size_t>(proxyW) * proxyH;
    impl_->bColorRaw = createBuffer(alloc, proxyN * 8,  // RGBA16_SFLOAT = 8B/texel
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                     impl_->aColorRaw);
    impl_->bMotionRaw = createBuffer(alloc, proxyN * 16,  // RGBA32_SFLOAT = 16B/texel
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                      impl_->aMotionRaw);
    impl_->bDepthRaw = createBuffer(alloc, proxyN * 16,
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                     impl_->aDepthRaw);
    for (int i = 0; i < 2; i++) {
        impl_->bDepthPing[i] = createBuffer(alloc, proxyN * sizeof(float),
                                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, impl_->aDepthPing[i]);
        zeroBuffer(alloc, impl_->aDepthPing[i], proxyN * sizeof(float));
        vmaMapMemory(alloc, impl_->aDepthPing[i], &impl_->mappedDepthPing[i]);
    }
    impl_->depthPingIndex = 0;
    firstFrame_ = true;

    impl_->bFgCur = createBuffer(alloc, proxyN * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, impl_->aFgCur);
    zeroBuffer(alloc, impl_->aFgCur, proxyN * sizeof(float));

    uint32_t channels = getChannels();
    const size_t hiddenN = static_cast<size_t>(netW_) * netH_ * hiddenChannels;
    impl_->bZeroHidden = createBuffer(alloc, std::max<size_t>(hiddenN, 1) * sizeof(float),
                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, impl_->aZeroHidden);
    zeroBuffer(alloc, impl_->aZeroHidden, std::max<size_t>(hiddenN, 1) * sizeof(float));

    const size_t outN = static_cast<size_t>(netW_) * netH_ * channels;
    impl_->bOutput = createBuffer(alloc, outN * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, impl_->aOutput);
    vmaMapMemory(alloc, impl_->aOutput, &impl_->mappedOutput);

    // --- pipelines ---
    impl_->unpremulLayout = makeSetLayout(ctx.device, 3);
    impl_->assembleLayout = makeSetLayout(ctx.device, 8);
    impl_->unpremulPL = makePipelineLayout(ctx.device, impl_->unpremulLayout, sizeof(UnpremulPush));
    impl_->assemblePL = makePipelineLayout(ctx.device, impl_->assembleLayout, sizeof(AssemblePush));
    impl_->unpremulPipe = makeComputePipeline(ctx.device, impl_->unpremulPL,
                                               shaderDir + "/input_assembly_unpremul_depth.comp.spv");
    impl_->assemblePipe = makeComputePipeline(ctx.device, impl_->assemblePL,
                                               shaderDir + "/input_assembly_assemble.comp.spv");

    VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 + 8};
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = 2;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr, &impl_->pool);

    VkDescriptorSetLayout layouts[2] = {impl_->unpremulLayout, impl_->assembleLayout};
    VkDescriptorSet sets[2];
    VkDescriptorSetAllocateInfo dsAlloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsAlloc.descriptorPool = impl_->pool;
    dsAlloc.descriptorSetCount = 2;
    dsAlloc.pSetLayouts = layouts;
    vkAllocateDescriptorSets(ctx.device, &dsAlloc, sets);
    impl_->unpremulSet = sets[0];
    impl_->assembleSet = sets[1];

    writeDescriptorSet(ctx.device, impl_->unpremulSet, {impl_->bDepthRaw, impl_->bDepthPing[0], impl_->bFgCur});
    // assembleSet bindings 2/3 (curDepth/prevDepth) are rewritten per-frame
    // in run() (ping-pong swap) -- initial wiring here just needs valid
    // buffers bound so the descriptor set is complete from the start.
    writeDescriptorSet(ctx.device, impl_->assembleSet,
                        {impl_->bColorRaw, impl_->bMotionRaw, impl_->bDepthPing[0], impl_->bDepthPing[1],
                         impl_->bTexture, impl_->bZeroHidden, impl_->bFgCur, impl_->bOutput});
}

void InputAssembly::run(VulkanContext& ctx, VkImage colorImage, VkImage depthImage, VkImage motionImage,
                         uint32_t proxyW, uint32_t proxyH, const float* hiddenIn,
                         float eyeX, float eyeY, float eyeZ,
                         float rX, float rY, float rZ, float uX, float uY, float uZ,
                         float fX, float fY, float fZ,
                         float fx, float fy, float cx, float cy,
                         float jitterProxyX, float jitterProxyY) {
    copyImageToBuffer(ctx, colorImage, impl_->bColorRaw, proxyW, proxyH);
    copyImageToBuffer(ctx, depthImage, impl_->bDepthRaw, proxyW, proxyH);
    copyImageToBuffer(ctx, motionImage, impl_->bMotionRaw, proxyW, proxyH);

    int curIdx = impl_->depthPingIndex;
    int prevIdx = 1 - curIdx;

    // Pass 1: unpremultiply this frame's depth into the "current" ping side,
    // and this frame's foreground coverage (out_depth's .g channel) into
    // bFgCur (no ping-pong -- see its declaration comment).
    writeDescriptorSet(ctx.device, impl_->unpremulSet,
                        {impl_->bDepthRaw, impl_->bDepthPing[curIdx], impl_->bFgCur});
    UnpremulPush upPush{proxyW, proxyH};
    uint32_t gx = (proxyW + 15) / 16, gy = (proxyH + 15) / 16;
    dispatchOne(ctx, impl_->unpremulPipe, impl_->unpremulPL, impl_->unpremulSet, &upPush, sizeof(upPush), gx, gy);

    // Pass 2: assemble. bindings 2/3 = curDepth (just written), prevDepth (last frame's).
    VkBuffer hiddenBuf = impl_->bZeroHidden;
    if (hiddenIn != nullptr) {
        const size_t hiddenN = static_cast<size_t>(netW_) * netH_ * hiddenChannels_;
        uploadFloats(ctx.allocator, impl_->aZeroHidden, std::vector<float>(hiddenIn, hiddenIn + hiddenN));
    }
    writeDescriptorSet(ctx.device, impl_->assembleSet,
                        {impl_->bColorRaw, impl_->bMotionRaw, impl_->bDepthPing[curIdx], impl_->bDepthPing[prevIdx],
                         impl_->bTexture, hiddenBuf, impl_->bFgCur, impl_->bOutput});

    AssemblePush push{};
    push.eye[0] = eyeX; push.eye[1] = eyeY; push.eye[2] = eyeZ; push.eye[3] = 0;
    push.rAxis[0] = rX; push.rAxis[1] = rY; push.rAxis[2] = rZ; push.rAxis[3] = 0;
    push.uAxis[0] = uX; push.uAxis[1] = uY; push.uAxis[2] = uZ; push.uAxis[3] = 0;
    push.fAxis[0] = fX; push.fAxis[1] = fY; push.fAxis[2] = fZ; push.fAxis[3] = 0;
    push.bgCenter[0] = bgSphere_[0]; push.bgCenter[1] = bgSphere_[1]; push.bgCenter[2] = bgSphere_[2]; push.bgCenter[3] = 0;
    push.camParams[0] = fx; push.camParams[1] = fy; push.camParams[2] = cx; push.camParams[3] = cy;
    push.extra[0] = bgSphere_[3]; push.extra[1] = 1.0f / 16.0f; push.extra[2] = jitterProxyX; push.extra[3] = jitterProxyY;
    push.dims1[0] = proxyW; push.dims1[1] = proxyH; push.dims1[2] = netW_; push.dims1[3] = netH_;
    push.dims2[0] = paramStride_; push.dims2[1] = hiddenChannels_; push.dims2[2] = texW_; push.dims2[3] = texH_;
    push.flags[0] = firstFrame_ ? 1u : 0u; push.flags[1] = texChannels_; push.flags[2] = kFgSourceExpectedDepthG; push.flags[3] = 0;

    uint32_t ngx = (netW_ + 15) / 16, ngy = (netH_ + 15) / 16;
    dispatchOne(ctx, impl_->assemblePipe, impl_->assemblePL, impl_->assembleSet, &push, sizeof(push), ngx, ngy);

    wasFirstFrame_ = firstFrame_;
    impl_->depthPingIndex = prevIdx;
    firstFrame_ = false;
}

const float* InputAssembly::getOutputHostPtr() const {
    return static_cast<const float*>(impl_->mappedOutput);
}

void InputAssembly::dumpToNpy(const std::string& path) const {
    size_t n = static_cast<size_t>(netW_) * netH_ * getChannels();
    std::vector<float> data(getOutputHostPtr(), getOutputHostPtr() + n);
    DlssIO::writeNpyFloat32(path, data, {netH_, netW_, getChannels()});
}

const float* InputAssembly::getDepthWrittenThisFrame() const {
    // run() already advanced depthPingIndex to point at what is now "prev";
    // "written this frame" is therefore the OTHER side.
    return static_cast<const float*>(impl_->mappedDepthPing[1 - impl_->depthPingIndex]);
}
const float* InputAssembly::getDepthFromPreviousFrame() const {
    return static_cast<const float*>(impl_->mappedDepthPing[impl_->depthPingIndex]);
}
