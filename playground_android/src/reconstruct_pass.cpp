// Copied from playground_cpp/src/reconstruct_runner.cpp (context-taking
// overloads only) -- see reconstruct_pass.h's header comment for why.
#include "reconstruct_pass.h"
#include "vulkan_context.h"
#include "spv_loader.h"
#include "dlss_io.h"

#include <glm/glm.hpp>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <array>
#include <chrono>

namespace fs = std::filesystem;

namespace {

inline double msSince(std::chrono::high_resolution_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
}

// Task 2: a tiny 2-slot GPU timestamp-query bracket, written via its own
// single-time command buffers immediately before the first and immediately
// after the last dispatch of one frame's compute chain -- see
// ReconstructTimingsMs::dispatchGpuMs's comment for what this does and
// doesn't capture. `beginSingleTimeCommands()`/`endSingleTimeCommands()`
// already fully drain the queue (per dispatchOne's own comment), so the
// result is available to vkGetQueryPoolResults immediately after the second
// call returns -- no extra sync needed.
struct GpuTimestampBracket {
    VkQueryPool pool = VK_NULL_HANDLE;
    double periodNs = 1.0;

    void init(VulkanContext& ctx) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(ctx.physicalDevice, &props);
        periodNs = props.limits.timestampPeriod > 0.0 ? props.limits.timestampPeriod : 1.0;
        VkQueryPoolCreateInfo qpci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpci.queryCount = 2;
        vkCreateQueryPool(ctx.device, &qpci, nullptr, &pool);
    }
    void destroy(VkDevice device) {
        if (pool != VK_NULL_HANDLE) vkDestroyQueryPool(device, pool, nullptr);
    }
    void writeStart(VulkanContext& ctx) {
        VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
        vkCmdResetQueryPool(cmd, pool, 0, 2);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, pool, 0);
        ctx.endSingleTimeCommands(cmd);
    }
    void writeEnd(VulkanContext& ctx) {
        VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool, 1);
        ctx.endSingleTimeCommands(cmd);
    }
    double readDeltaMs(VkDevice device) {
        uint64_t ts[2] = {0, 0};
        VkResult qr = vkGetQueryPoolResults(device, pool, 0, 2, sizeof(ts), ts, sizeof(uint64_t),
                                             VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
        if (qr != VK_SUCCESS) return 0.0;
        return static_cast<double>(ts[1] - ts[0]) * periodNs * 1e-6;
    }
};

// --- Small local helpers (same conventions as splat_renderer.cpp) ---------

VkBuffer createHostVisibleBuffer(VmaAllocator allocator, VkDeviceSize sizeBytes,
                                  VmaAllocation& allocOut) {
    VkBufferCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = std::max<VkDeviceSize>(sizeBytes, 16);
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;  // host-visible + device-accessible (UMA)

    VkBuffer buf = VK_NULL_HANDLE;
    if (vmaCreateBuffer(allocator, &info, &allocInfo, &buf, &allocOut, nullptr) != VK_SUCCESS) {
        throw std::runtime_error("reconstruct_runner: failed to create buffer");
    }
    return buf;
}

void uploadFloats(VmaAllocator allocator, VmaAllocation alloc, const std::vector<float>& data) {
    void* mapped = nullptr;
    vmaMapMemory(allocator, alloc, &mapped);
    if (!data.empty()) std::memcpy(mapped, data.data(), data.size() * sizeof(float));
    vmaUnmapMemory(allocator, alloc);
}

// Task 4 (docs/rendering-engines.md, remove the readback stalls/copies where
// cheap): raw-pointer overload -- ReconstructLive::run()'s FrameInputs are
// already plain `const float*` (caller-owned, valid for the call's duration
// per the header comment), so the std::vector<float>(ptr, ptr+n) the
// original call sites built here was a second, whole-buffer CPU copy paid
// on EVERY frame for no reason (measured worst on I.aPacked, the packed
// net-output upload -- netW*netH*outCh floats, tens of MB at this model's
// shapes) before this function's own memcpy into mapped GPU memory did the
// real work a second time. This overload maps+memcpys directly from the
// caller's pointer, same single copy uploadFloats(vector) already does
// internally, just without manufacturing a throwaway vector first.
void uploadFloats(VmaAllocator allocator, VmaAllocation alloc, const float* data, size_t n) {
    void* mapped = nullptr;
    vmaMapMemory(allocator, alloc, &mapped);
    if (n > 0 && data != nullptr) std::memcpy(mapped, data, n * sizeof(float));
    vmaUnmapMemory(allocator, alloc);
}

void zeroBuffer(VmaAllocator allocator, VmaAllocation alloc, size_t numFloats) {
    void* mapped = nullptr;
    vmaMapMemory(allocator, alloc, &mapped);
    std::memset(mapped, 0, numFloats * sizeof(float));
    vmaUnmapMemory(allocator, alloc);
}

std::vector<float> downloadFloats(VmaAllocator allocator, VmaAllocation alloc, size_t numFloats) {
    std::vector<float> out(numFloats);
    void* mapped = nullptr;
    vmaMapMemory(allocator, alloc, &mapped);
    std::memcpy(out.data(), mapped, numFloats * sizeof(float));
    vmaUnmapMemory(allocator, alloc);
    return out;
}

void copyBufferHost(VmaAllocator allocator, VmaAllocation dst, VmaAllocation src, size_t numFloats) {
    void* dstPtr = nullptr;
    void* srcPtr = nullptr;
    vmaMapMemory(allocator, dst, &dstPtr);
    vmaMapMemory(allocator, src, &srcPtr);
    std::memcpy(dstPtr, srcPtr, numFloats * sizeof(float));
    vmaUnmapMemory(allocator, src);
    vmaUnmapMemory(allocator, dst);
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
    VkDescriptorSetLayoutCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    info.bindingCount = numStorageBuffers;
    info.pBindings = bindings.data();
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    vkCreateDescriptorSetLayout(device, &info, nullptr, &layout);
    return layout;
}

VkPipelineLayout makePipelineLayout(VkDevice device, VkDescriptorSetLayout setLayout,
                                     uint32_t pushConstantSize) {
    VkPushConstantRange range = {};
    range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    range.offset = 0;
    range.size = pushConstantSize;

    VkPipelineLayoutCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
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
    VkComputePipelineCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    info.stage.module = module;
    info.stage.pName = "main";
    info.layout = layout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline) != VK_SUCCESS) {
        throw std::runtime_error("reconstruct_runner: failed to create compute pipeline for " + spvPath);
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

void dispatchOne(VulkanContext& ctx, VkPipeline pipeline, VkPipelineLayout layout,
                  VkDescriptorSet set, const void* pushData, uint32_t pushSize,
                  uint32_t totalThreads) {
    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushSize, pushData);
    uint32_t groups = (totalThreads + 255) / 256;
    vkCmdDispatch(cmd, groups, 1, 1);
    // Each dispatch is its own fully-synchronous submission (full queue
    // drain via endSingleTimeCommands' vkQueueWaitIdle) -- deliberately
    // conservative: docs/lux-4d-spec.md section 3's MV-race debugging
    // found that a MoltenVK/Apple-Silicon compute-to-compute dependency
    // chained through several later barriers was NOT always reliably
    // observed by a later stage in the same command buffer, while a full
    // drain between stages always was. Exactness matters far more than
    // per-frame latency for this offline validation tool.
    ctx.endSingleTimeCommands(cmd);
}

// Task B (docs/rendering-engines.md, "reconstruct-pass queue waits"):
// records one dispatch into an ALREADY-open command buffer -- no begin/end/
// submit of its own -- followed by a compute-to-compute pipeline barrier,
// instead of dispatchOne's fully-synchronous single-dispatch submission.
// Used ONLY by ReconstructLive::run() below: the live, on-device
// Reconstruction-mode path, which only ever runs against the Mali-G715's
// conformant Vulkan driver on Android, not dispatchOne's MoltenVK-flaky
// desktop/offline validation path above (runReconstructDump), which keeps
// the old per-dispatch vkQueueWaitIdle unchanged.
void recordDispatchBarriered(VkCommandBuffer cmd, VkPipeline pipeline, VkPipelineLayout layout,
                              VkDescriptorSet set, const void* pushData, uint32_t pushSize,
                              uint32_t totalThreads, bool barrierAfter) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushSize, pushData);
    uint32_t groups = (totalThreads + 255) / 256;
    vkCmdDispatch(cmd, groups, 1, 1);
    if (!barrierAfter) return;
    // One VkMemoryBarrier (all buffers), not a per-buffer scoped barrier:
    // ReconstructLive::init()'s descriptor sets chain bguv -> memory -> warp
    // -> apply -> blend, each stage consuming a mix of earlier stages'
    // outputs plus its own dedicated inputs (see the writeDescriptorSet
    // calls there), so naming every producer/consumer buffer pair by hand
    // would not be meaningfully tighter than one global barrier at this
    // buffer count/size, and a global barrier is trivially provable correct
    // (every write before this point is visible to every read after it).
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          0, 1, &barrier, 0, nullptr, 0, nullptr);
}

// numpy's row-major (flat[i*4+j] == M[row i, col j]) -> GLM's column-major
// storage (mat[col][row]) -- same convention as DlssIO::cvViewToGl's own
// row-major-input handling. SPIR-V push-constant mat4 fields are
// column-major (standard GLSL/std430 convention), matching GLM's raw
// &mat[0][0] layout directly, so no further transform is needed once built
// this way (see splat_renderer.cpp's own `std::memcpy(push.view,
// &viewMatrix_[0][0], 64)` for the same pattern).
glm::mat4 rowMajorToGlm4x4(const std::vector<float>& flat16) {
    glm::mat4 m(1.0f);
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            m[c][r] = flat16[r * 4 + c];
    return m;
}

struct Meta {
    int s = 2, k = 4, param_stride = 1, hidden = 8;
    int proxy_w = 0, proxy_h = 0, target_w = 0, target_h = 0, net_w = 0, net_h = 0;
    int num_frames = 0;
    // Scene-memory path (SPECIFICATION.md 12.9's `memory: { channels,
    // hidden }` sub-block) -- memory_channels == 0 means the compiled
    // pipeline (and this dump) has no memory path, exactly the stage-1
    // 2-way blend. Mirrors metal_reconstruct_runner.cpp's identical Meta.
    int memory_channels = 0, memory_hidden = 0, tex_w = 0, tex_h = 0;
};

Meta loadMeta(const std::string& path) {
    Meta m;
    auto get = [&](const char* key) -> long { return DlssIO::readJsonIntField(path, key); };
    m.s = static_cast<int>(get("s"));
    m.k = static_cast<int>(get("k"));
    m.param_stride = static_cast<int>(get("param_stride"));
    m.hidden = static_cast<int>(get("hidden"));
    m.proxy_w = static_cast<int>(get("proxy_w"));
    m.proxy_h = static_cast<int>(get("proxy_h"));
    m.target_w = static_cast<int>(get("target_w"));
    m.target_h = static_cast<int>(get("target_h"));
    m.net_w = static_cast<int>(get("net_w"));
    m.net_h = static_cast<int>(get("net_h"));
    m.num_frames = static_cast<int>(get("num_frames"));
    if (m.proxy_w <= 0 || m.proxy_h <= 0 || m.target_w <= 0 || m.target_h <= 0 ||
        m.net_w <= 0 || m.net_h <= 0 || m.num_frames <= 0) {
        throw std::runtime_error("reconstruct_runner: invalid/missing field in " + path);
    }
    long mc = get("memory_channels");
    long mh = get("memory_hidden");
    long tw = get("tex_w");
    long th = get("tex_h");
    if (mc > 0 && mh > 0 && tw > 0 && th > 0) {
        m.memory_channels = static_cast<int>(mc);
        m.memory_hidden = static_cast<int>(mh);
        m.tex_w = static_cast<int>(tw);
        m.tex_h = static_cast<int>(th);
    }
    return m;
}

struct BguvPush {
    uint32_t width, height;
    float _pad0[2];
    float k_params[4];       // fx, fy, cx, cy
    float cam_to_world[16];  // column-major (GLM convention)
    float bg_sphere[4];      // centre.xyz, radius
};
static_assert(sizeof(BguvPush) == 112, "BguvPush must match reconstruct_expander.py's bguv push layout");

struct MemoryPush {
    uint32_t width, height, tex_w, tex_h;
};
static_assert(sizeof(MemoryPush) == 16, "MemoryPush must match reconstruct_expander.py's memory push layout");

} // namespace

// NOTE: the desktop file's no-ctx overload (constructs its own headless
// VulkanContext via GLFW) is deliberately NOT copied here -- see
// reconstruct_pass.h's header comment.

int runReconstructDump(VulkanContext& ctx, const std::string& dumpDir, const std::string& outDirIn,
                        const std::string& pipelineBase, ReconstructTimingsMs* outTimings) {
    auto tSetup = std::chrono::high_resolution_clock::now();
    std::string outDir = outDirIn.empty() ? dumpDir : outDirIn;
    fs::create_directories(outDir);

    Meta meta;
    try {
        meta = loadMeta(dumpDir + "/meta.json");
    } catch (const std::exception& e) {
        std::cerr << "[error] " << e.what() << std::endl;
        return 1;
    }
    const int S = meta.s, K = meta.k, PS = meta.param_stride, HIDDEN = meta.hidden;
    const int SP = S * PS;
    const bool hasMemory = meta.memory_channels > 0;
    const int NBLEND = hasMemory ? 3 : 1;
    const int totalCh = SP * SP * K * K + SP * SP * NBLEND + HIDDEN;
    const int MC = meta.memory_channels, MH = meta.memory_hidden, TEXW = meta.tex_w, TEXH = meta.tex_h;
    std::cout << "[reconstruct] s=" << S << " k=" << K << " param_stride=" << PS
              << " hidden=" << HIDDEN << " proxy=" << meta.proxy_w << "x" << meta.proxy_h
              << " target=" << meta.target_w << "x" << meta.target_h
              << " net=" << meta.net_w << "x" << meta.net_h
              << " frames=" << meta.num_frames
              << (hasMemory ? (" memory(channels=" + std::to_string(MC) + " hidden=" + std::to_string(MH) +
                                " tex=" + std::to_string(TEXW) + "x" + std::to_string(TEXH) + ")") : "")
              << std::endl;

    // --- Pipelines (3 independent compute stages, each its own set layout
    // matching the StorageBufferDecl order reconstruct_expander.py emits;
    // +bguv/+memory when the compiled pipeline has the scene-memory path) ---
    VkDescriptorSetLayout warpLayout = makeSetLayout(ctx.device, 3);
    VkDescriptorSetLayout applyLayout = makeSetLayout(ctx.device, 5);
    VkDescriptorSetLayout blendLayout = makeSetLayout(ctx.device, hasMemory ? 6 : 5);
    VkDescriptorSetLayout bguvLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout memoryLayout = VK_NULL_HANDLE;

    VkPipelineLayout warpPL = makePipelineLayout(ctx.device, warpLayout, 16);   // 4 uint
    VkPipelineLayout applyPL = makePipelineLayout(ctx.device, applyLayout, 32); // 6 uint + 2 scalar
    VkPipelineLayout blendPL = makePipelineLayout(ctx.device, blendLayout, 16); // 2 uint (padded)
    VkPipelineLayout bguvPL = VK_NULL_HANDLE;
    VkPipelineLayout memoryPL = VK_NULL_HANDLE;

    VkPipeline warpPipe, applyPipe, blendPipe;
    VkPipeline bguvPipe = VK_NULL_HANDLE, memoryPipe = VK_NULL_HANDLE;
    try {
        warpPipe = makeComputePipeline(ctx.device, warpPL, pipelineBase + ".warp.comp.spv");
        applyPipe = makeComputePipeline(ctx.device, applyPL, pipelineBase + ".apply.comp.spv");
        blendPipe = makeComputePipeline(ctx.device, blendPL, pipelineBase + ".blend.comp.spv");
        if (hasMemory) {
            bguvLayout = makeSetLayout(ctx.device, 1);
            memoryLayout = makeSetLayout(ctx.device, 8);
            bguvPL = makePipelineLayout(ctx.device, bguvLayout, sizeof(BguvPush));
            memoryPL = makePipelineLayout(ctx.device, memoryLayout, sizeof(MemoryPush));
            bguvPipe = makeComputePipeline(ctx.device, bguvPL, pipelineBase + ".bguv.comp.spv");
            memoryPipe = makeComputePipeline(ctx.device, memoryPL, pipelineBase + ".memory.comp.spv");
        }
    } catch (const std::exception& e) {
        std::cerr << "[error] " << e.what() << std::endl;
        return 1;
    }

    uint32_t totalDescriptors = 3 + 5 + (hasMemory ? 6u : 5u) + (hasMemory ? (1u + 8u) : 0u);
    uint32_t totalSets = hasMemory ? 5 : 3;
    VkDescriptorPoolSize poolSize = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, totalDescriptors};
    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = totalSets;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr, &pool);

    std::vector<VkDescriptorSetLayout> layouts = {warpLayout, applyLayout, blendLayout};
    if (hasMemory) { layouts.push_back(bguvLayout); layouts.push_back(memoryLayout); }
    std::vector<VkDescriptorSet> sets(layouts.size());
    VkDescriptorSetAllocateInfo dsAlloc = {};
    dsAlloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsAlloc.descriptorPool = pool;
    dsAlloc.descriptorSetCount = static_cast<uint32_t>(layouts.size());
    dsAlloc.pSetLayouts = layouts.data();
    vkAllocateDescriptorSets(ctx.device, &dsAlloc, sets.data());
    VkDescriptorSet warpSet = sets[0], applySet = sets[1], blendSet = sets[2];
    VkDescriptorSet bguvSet = hasMemory ? sets[3] : VK_NULL_HANDLE;
    VkDescriptorSet memorySet = hasMemory ? sets[4] : VK_NULL_HANDLE;

    // --- Buffers ---
    const size_t proxyColorN = static_cast<size_t>(meta.proxy_w) * meta.proxy_h * 3;
    const size_t mvProxyN = static_cast<size_t>(meta.proxy_w) * meta.proxy_h * 2;
    const size_t targetColorN = static_cast<size_t>(meta.target_w) * meta.target_h * 3;
    const size_t targetScalarN = static_cast<size_t>(meta.target_w) * meta.target_h;
    const size_t hiddenN = targetScalarN * HIDDEN;
    const size_t packedN = static_cast<size_t>(meta.net_w) * meta.net_h * totalCh;
    const size_t blendN = targetScalarN * NBLEND;  // alpha_out (N=1) or blend_out (N=3)

    VmaAllocation aProxyColor, aMvProxy, aWarped, aPrevColor;
    VmaAllocation aPacked, aSpatial, aBlend, aHidden;
    VmaAllocation aDisocc, aOutColor;

    VkBuffer bProxyColor = createHostVisibleBuffer(ctx.allocator, proxyColorN * 4, aProxyColor);
    VkBuffer bMvProxy = createHostVisibleBuffer(ctx.allocator, mvProxyN * 4, aMvProxy);
    VkBuffer bWarped = createHostVisibleBuffer(ctx.allocator, targetColorN * 4, aWarped);
    VkBuffer bPrevColor = createHostVisibleBuffer(ctx.allocator, targetColorN * 4, aPrevColor);
    VkBuffer bPacked = createHostVisibleBuffer(ctx.allocator, packedN * 4, aPacked);
    VkBuffer bSpatial = createHostVisibleBuffer(ctx.allocator, targetColorN * 4, aSpatial);
    VkBuffer bBlend = createHostVisibleBuffer(ctx.allocator, blendN * 4, aBlend);
    VkBuffer bHidden = createHostVisibleBuffer(ctx.allocator, hiddenN * 4, aHidden);
    VkBuffer bDisocc = createHostVisibleBuffer(ctx.allocator, targetScalarN * 4, aDisocc);
    VkBuffer bOutColor = createHostVisibleBuffer(ctx.allocator, targetColorN * 4, aOutColor);

    // --- Scene-memory buffers (loaded once) ---
    VmaAllocation aBgTexture{}, aFc1W{}, aFc1B{}, aFc2W{}, aFc2B{}, aBguvTarget{}, aBgFeaturesDummy{}, aMemoryColor{};
    VkBuffer bBgTexture = VK_NULL_HANDLE, bFc1W = VK_NULL_HANDLE, bFc1B = VK_NULL_HANDLE;
    VkBuffer bFc2W = VK_NULL_HANDLE, bFc2B = VK_NULL_HANDLE, bBguvTarget = VK_NULL_HANDLE;
    VkBuffer bBgFeaturesDummy = VK_NULL_HANDLE, bMemoryColor = VK_NULL_HANDLE;
    std::array<float, 4> bgSphere{};

    if (hasMemory) {
        try {
            auto tex = DlssIO::readNpyFloat32(dumpDir + "/texture.npy");
            auto sph = DlssIO::readNpyFloat32(dumpDir + "/bg_sphere.npy");
            if (tex.data.size() != static_cast<size_t>(MC) * TEXH * TEXW || sph.data.size() != 4) {
                std::cerr << "[error] texture.npy/bg_sphere.npy size mismatch against meta.json" << std::endl;
                return 1;
            }
            std::copy(sph.data.begin(), sph.data.end(), bgSphere.begin());

            auto fc1w = DlssIO::readNpzMemberFloat32(dumpDir + "/memory_head.npz", "fc1_w");
            auto fc1b = DlssIO::readNpzMemberFloat32(dumpDir + "/memory_head.npz", "fc1_b");
            auto fc2w = DlssIO::readNpzMemberFloat32(dumpDir + "/memory_head.npz", "fc2_w");
            auto fc2b = DlssIO::readNpzMemberFloat32(dumpDir + "/memory_head.npz", "fc2_b");
            if (fc1w.data.size() != static_cast<size_t>(MH) * MC || fc1b.data.size() != static_cast<size_t>(MH) ||
                fc2w.data.size() != static_cast<size_t>(3) * MH || fc2b.data.size() != 3) {
                std::cerr << "[error] memory_head.npz array size mismatch against meta.json" << std::endl;
                return 1;
            }

            bBgTexture = createHostVisibleBuffer(ctx.allocator, tex.data.size() * 4, aBgTexture);
            uploadFloats(ctx.allocator, aBgTexture, tex.data);
            bFc1W = createHostVisibleBuffer(ctx.allocator, fc1w.data.size() * 4, aFc1W);
            uploadFloats(ctx.allocator, aFc1W, fc1w.data);
            bFc1B = createHostVisibleBuffer(ctx.allocator, fc1b.data.size() * 4, aFc1B);
            uploadFloats(ctx.allocator, aFc1B, fc1b.data);
            bFc2W = createHostVisibleBuffer(ctx.allocator, fc2w.data.size() * 4, aFc2W);
            uploadFloats(ctx.allocator, aFc2W, fc2w.data);
            bFc2B = createHostVisibleBuffer(ctx.allocator, fc2b.data.size() * 4, aFc2B);
            uploadFloats(ctx.allocator, aFc2B, fc2b.data);
        } catch (const std::exception& e) {
            std::cerr << "[error] " << e.what() << std::endl;
            return 1;
        }

        bBguvTarget = createHostVisibleBuffer(ctx.allocator, targetScalarN * 2 * 4, aBguvTarget);
        bBgFeaturesDummy = createHostVisibleBuffer(ctx.allocator, targetScalarN * static_cast<size_t>(MC) * 4, aBgFeaturesDummy);
        bMemoryColor = createHostVisibleBuffer(ctx.allocator, targetColorN * 4, aMemoryColor);
    }

    writeDescriptorSet(ctx.device, warpSet, {bPrevColor, bMvProxy, bWarped});
    writeDescriptorSet(ctx.device, applySet, {bPacked, bProxyColor, bSpatial, bBlend, bHidden});
    if (hasMemory) {
        writeDescriptorSet(ctx.device, blendSet, {bSpatial, bWarped, bBlend, bDisocc, bMemoryColor, bOutColor});
        writeDescriptorSet(ctx.device, bguvSet, {bBguvTarget});
        writeDescriptorSet(ctx.device, memorySet,
                            {bBguvTarget, bBgTexture, bFc1W, bFc1B, bFc2W, bFc2B, bBgFeaturesDummy, bMemoryColor});
    } else {
        writeDescriptorSet(ctx.device, blendSet, {bSpatial, bWarped, bBlend, bDisocc, bOutColor});
    }

    // prev_color starts at zero, matching mobiledlss.train.train.rollout's
    // `prev_color = torch.zeros(B, 3, H, W, ...)` initial state.
    zeroBuffer(ctx.allocator, aPrevColor, targetColorN);

    struct WarpPush { uint32_t target_w, target_h, proxy_w, proxy_h; };
    struct ApplyPush {
        uint32_t target_w, target_h, proxy_w, proxy_h, net_w, net_h;
        float jitter_x, jitter_y;
    };
    struct BlendPush { uint32_t target_w, target_h, _pad0, _pad1; };

    WarpPush warpPush = {static_cast<uint32_t>(meta.target_w), static_cast<uint32_t>(meta.target_h),
                          static_cast<uint32_t>(meta.proxy_w), static_cast<uint32_t>(meta.proxy_h)};
    BlendPush blendPush = {static_cast<uint32_t>(meta.target_w), static_cast<uint32_t>(meta.target_h), 0, 0};
    uint32_t totalTargetThreads = static_cast<uint32_t>(targetScalarN);

    GpuTimestampBracket gpuBracket;
    if (outTimings != nullptr) gpuBracket.init(ctx);
    if (outTimings != nullptr) outTimings->setupMs += msSince(tSetup);

    for (int t = 0; t < meta.num_frames; ++t) {
        std::string suf = "_f" + std::to_string(t) + ".npy";
        auto tRead = std::chrono::high_resolution_clock::now();
        auto proxyColor = DlssIO::readNpyFloat32(dumpDir + "/proxy_color" + suf);
        auto mvProxy = DlssIO::readNpyFloat32(dumpDir + "/mv_proxy" + suf);
        auto jitter = DlssIO::readNpyFloat32(dumpDir + "/jitter" + suf);
        auto packed = DlssIO::readNpyFloat32(dumpDir + "/packed_params" + suf);
        auto disocc = DlssIO::readNpyFloat32(dumpDir + "/disocc" + suf);

        if (proxyColor.data.size() != proxyColorN || mvProxy.data.size() != mvProxyN ||
            jitter.data.size() != 2 || packed.data.size() != packedN ||
            disocc.data.size() != targetScalarN) {
            std::cerr << "[error] frame " << t << ": dump array size mismatch against meta.json"
                       << " (proxy_color=" << proxyColor.data.size() << "/" << proxyColorN
                       << " mv_proxy=" << mvProxy.data.size() << "/" << mvProxyN
                       << " packed_params=" << packed.data.size() << "/" << packedN
                       << " disocc=" << disocc.data.size() << "/" << targetScalarN << ")" << std::endl;
            return 1;
        }

        std::vector<float> kParams, camToWorld;
        if (hasMemory) {
            kParams = DlssIO::readNpyFloat32(dumpDir + "/k_params" + suf).data;
            camToWorld = DlssIO::readNpyFloat32(dumpDir + "/cam_to_world" + suf).data;
            if (kParams.size() != 4 || camToWorld.size() != 16) {
                std::cerr << "[error] frame " << t << ": k_params/cam_to_world size mismatch" << std::endl;
                return 1;
            }
        }
        if (outTimings != nullptr) outTimings->fileReadMs += msSince(tRead);

        auto tUpload = std::chrono::high_resolution_clock::now();
        uploadFloats(ctx.allocator, aProxyColor, proxyColor.data);
        uploadFloats(ctx.allocator, aMvProxy, mvProxy.data);
        uploadFloats(ctx.allocator, aPacked, packed.data);
        uploadFloats(ctx.allocator, aDisocc, disocc.data);
        if (outTimings != nullptr) outTimings->uploadMs += msSince(tUpload);

        auto tDispatch = std::chrono::high_resolution_clock::now();
        if (outTimings != nullptr) gpuBracket.writeStart(ctx);

        if (hasMemory) {
            BguvPush bguvPush{};
            bguvPush.width = static_cast<uint32_t>(meta.target_w);
            bguvPush.height = static_cast<uint32_t>(meta.target_h);
            std::copy(kParams.begin(), kParams.end(), bguvPush.k_params);
            glm::mat4 camToWorldGlm = rowMajorToGlm4x4(camToWorld);
            std::memcpy(bguvPush.cam_to_world, &camToWorldGlm[0][0], 64);
            std::copy(bgSphere.begin(), bgSphere.end(), bguvPush.bg_sphere);

            dispatchOne(ctx, bguvPipe, bguvPL, bguvSet, &bguvPush, sizeof(bguvPush), totalTargetThreads);

            MemoryPush memPush = {static_cast<uint32_t>(meta.target_w), static_cast<uint32_t>(meta.target_h),
                                   static_cast<uint32_t>(TEXW), static_cast<uint32_t>(TEXH)};
            dispatchOne(ctx, memoryPipe, memoryPL, memorySet, &memPush, sizeof(memPush), totalTargetThreads);
        }

        dispatchOne(ctx, warpPipe, warpPL, warpSet, &warpPush, sizeof(warpPush), totalTargetThreads);

        ApplyPush applyPush = {static_cast<uint32_t>(meta.target_w), static_cast<uint32_t>(meta.target_h),
                                static_cast<uint32_t>(meta.proxy_w), static_cast<uint32_t>(meta.proxy_h),
                                static_cast<uint32_t>(meta.net_w), static_cast<uint32_t>(meta.net_h),
                                jitter.data[0], jitter.data[1]};
        dispatchOne(ctx, applyPipe, applyPL, applySet, &applyPush, sizeof(applyPush), totalTargetThreads);

        dispatchOne(ctx, blendPipe, blendPL, blendSet, &blendPush, sizeof(blendPush), totalTargetThreads);

        if (outTimings != nullptr) {
            gpuBracket.writeEnd(ctx);
            outTimings->dispatchGpuMs += gpuBracket.readDeltaMs(ctx.device);
            outTimings->dispatchCpuMs += msSince(tDispatch);
        }

        auto tDownload = std::chrono::high_resolution_clock::now();
        auto outColor = downloadFloats(ctx.allocator, aOutColor, targetColorN);
        auto hidden = downloadFloats(ctx.allocator, aHidden, hiddenN);
        std::vector<float> bguvOut;
        if (hasMemory) bguvOut = downloadFloats(ctx.allocator, aBguvTarget, targetScalarN * 2);
        if (outTimings != nullptr) outTimings->downloadMs += msSince(tDownload);

        auto tWrite = std::chrono::high_resolution_clock::now();
        DlssIO::writeNpyFloat32(outDir + "/out_f" + std::to_string(t) + ".npy", outColor,
                                 {meta.target_h, meta.target_w, 3});
        DlssIO::writeNpyFloat32(outDir + "/hidden_f" + std::to_string(t) + ".npy", hidden,
                                 {meta.target_h, meta.target_w, HIDDEN});
        if (hasMemory) {
            DlssIO::writeNpyFloat32(outDir + "/bguv_target_f" + std::to_string(t) + ".npy", bguvOut,
                                     {meta.target_h, meta.target_w, 2});
        }
        if (outTimings != nullptr) outTimings->fileWriteMs += msSince(tWrite);

        // Carry `out` forward as next frame's `prev_color` (host-side copy
        // -- all buffers are host-visible/coherent VMA_MEMORY_USAGE_CPU_TO_GPU,
        // so no GPU copy command or extra barrier is needed here; the prior
        // dispatch's own endSingleTimeCommands() already fully drained the
        // queue, guaranteeing the CPU sees the GPU's writes).
        copyBufferHost(ctx.allocator, aPrevColor, aOutColor, targetColorN);

        std::cout << "[reconstruct] frame " << t << " done" << std::endl;
    }

    auto tTeardown = std::chrono::high_resolution_clock::now();
    gpuBracket.destroy(ctx.device);
    vkDestroyPipeline(ctx.device, warpPipe, nullptr);
    vkDestroyPipeline(ctx.device, applyPipe, nullptr);
    vkDestroyPipeline(ctx.device, blendPipe, nullptr);
    vkDestroyPipelineLayout(ctx.device, warpPL, nullptr);
    vkDestroyPipelineLayout(ctx.device, applyPL, nullptr);
    vkDestroyPipelineLayout(ctx.device, blendPL, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, warpLayout, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, applyLayout, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, blendLayout, nullptr);
    if (hasMemory) {
        vkDestroyPipeline(ctx.device, bguvPipe, nullptr);
        vkDestroyPipeline(ctx.device, memoryPipe, nullptr);
        vkDestroyPipelineLayout(ctx.device, bguvPL, nullptr);
        vkDestroyPipelineLayout(ctx.device, memoryPL, nullptr);
        vkDestroyDescriptorSetLayout(ctx.device, bguvLayout, nullptr);
        vkDestroyDescriptorSetLayout(ctx.device, memoryLayout, nullptr);
        vmaDestroyBuffer(ctx.allocator, bBgTexture, aBgTexture);
        vmaDestroyBuffer(ctx.allocator, bFc1W, aFc1W);
        vmaDestroyBuffer(ctx.allocator, bFc1B, aFc1B);
        vmaDestroyBuffer(ctx.allocator, bFc2W, aFc2W);
        vmaDestroyBuffer(ctx.allocator, bFc2B, aFc2B);
        vmaDestroyBuffer(ctx.allocator, bBguvTarget, aBguvTarget);
        vmaDestroyBuffer(ctx.allocator, bBgFeaturesDummy, aBgFeaturesDummy);
        vmaDestroyBuffer(ctx.allocator, bMemoryColor, aMemoryColor);
    }
    vkDestroyDescriptorPool(ctx.device, pool, nullptr);
    vmaDestroyBuffer(ctx.allocator, bProxyColor, aProxyColor);
    vmaDestroyBuffer(ctx.allocator, bMvProxy, aMvProxy);
    vmaDestroyBuffer(ctx.allocator, bWarped, aWarped);
    vmaDestroyBuffer(ctx.allocator, bPrevColor, aPrevColor);
    vmaDestroyBuffer(ctx.allocator, bPacked, aPacked);
    vmaDestroyBuffer(ctx.allocator, bSpatial, aSpatial);
    vmaDestroyBuffer(ctx.allocator, bBlend, aBlend);
    vmaDestroyBuffer(ctx.allocator, bHidden, aHidden);
    vmaDestroyBuffer(ctx.allocator, bDisocc, aDisocc);
    vmaDestroyBuffer(ctx.allocator, bOutColor, aOutColor);
    if (outTimings != nullptr) outTimings->teardownMs += msSince(tTeardown);

    std::cout << "[reconstruct] done: " << meta.num_frames << " frames written to " << outDir << std::endl;
    return 0;
}

// NOTE: the desktop file's no-ctx overload is deliberately not copied here
// either -- same reason as runReconstructDump's above.

int runDumpBgFeatures(VulkanContext& ctx, const std::string& dumpDir, const std::string& outDirIn,
                       const std::string& pipelineBase) {
    std::string outDir = outDirIn.empty() ? dumpDir : outDirIn;
    fs::create_directories(outDir);

    Meta meta;
    try {
        meta = loadMeta(dumpDir + "/meta.json");
    } catch (const std::exception& e) {
        std::cerr << "[error] " << e.what() << std::endl;
        return 1;
    }
    if (meta.memory_channels <= 0) {
        std::cerr << "[error] --dump-bg-features: dump directory's meta.json has no memory_channels "
                     "(this dump/pipeline has no scene-memory path)" << std::endl;
        return 1;
    }
    const int MC = meta.memory_channels, MH = meta.memory_hidden, TEXW = meta.tex_w, TEXH = meta.tex_h;
    std::cout << "[dump-bg-features] proxy=" << meta.proxy_w << "x" << meta.proxy_h
              << " channels=" << MC << " tex=" << TEXW << "x" << TEXH
              << " frames=" << meta.num_frames << std::endl;

    VkDescriptorSetLayout bguvLayout = makeSetLayout(ctx.device, 1);
    VkDescriptorSetLayout memoryLayout = makeSetLayout(ctx.device, 8);
    VkPipelineLayout bguvPL = makePipelineLayout(ctx.device, bguvLayout, sizeof(BguvPush));
    VkPipelineLayout memoryPL = makePipelineLayout(ctx.device, memoryLayout, sizeof(MemoryPush));
    VkPipeline bguvPipe, memoryPipe;
    try {
        bguvPipe = makeComputePipeline(ctx.device, bguvPL, pipelineBase + ".bguv.comp.spv");
        memoryPipe = makeComputePipeline(ctx.device, memoryPL, pipelineBase + ".memory.comp.spv");
    } catch (const std::exception& e) {
        std::cerr << "[error] " << e.what() << std::endl;
        return 1;
    }

    VkDescriptorPoolSize poolSize = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 + 8};
    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 2;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr, &pool);

    VkDescriptorSetLayout layouts[2] = {bguvLayout, memoryLayout};
    VkDescriptorSetAllocateInfo dsAlloc = {};
    dsAlloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsAlloc.descriptorPool = pool;
    dsAlloc.descriptorSetCount = 2;
    dsAlloc.pSetLayouts = layouts;
    VkDescriptorSet sets[2];
    vkAllocateDescriptorSets(ctx.device, &dsAlloc, sets);
    VkDescriptorSet bguvSet = sets[0], memorySet = sets[1];

    auto tex = DlssIO::readNpyFloat32(dumpDir + "/texture.npy");
    auto sph = DlssIO::readNpyFloat32(dumpDir + "/bg_sphere.npy");
    if (tex.data.size() != static_cast<size_t>(MC) * TEXH * TEXW || sph.data.size() != 4) {
        std::cerr << "[error] texture.npy/bg_sphere.npy size mismatch against meta.json" << std::endl;
        return 1;
    }
    std::array<float, 4> bgSphere{};
    std::copy(sph.data.begin(), sph.data.end(), bgSphere.begin());

    const size_t proxyScalarN = static_cast<size_t>(meta.proxy_w) * meta.proxy_h;
    VmaAllocation aBgTexture, aFc1W, aFc1B, aFc2W, aFc2B, aBguvProxy, aBgFeatures, aMemoryColorDummy;
    VkBuffer bBgTexture = createHostVisibleBuffer(ctx.allocator, tex.data.size() * 4, aBgTexture);
    uploadFloats(ctx.allocator, aBgTexture, tex.data);
    // Decoder weights aren't needed for this proxy-feature-only dump (the
    // bg_features_out write happens before the decoder runs), but every
    // declared Vulkan descriptor binding must point at a validly-sized
    // buffer regardless (unlike Metal/SPIRV-Cross, which drops genuinely
    // unused ones) -- zero-filled real-sized buffers are fine since the
    // resulting memory_color_out is discarded.
    VkBuffer bFc1W = createHostVisibleBuffer(ctx.allocator, static_cast<size_t>(MH) * MC * 4, aFc1W);
    zeroBuffer(ctx.allocator, aFc1W, static_cast<size_t>(MH) * MC);
    VkBuffer bFc1B = createHostVisibleBuffer(ctx.allocator, static_cast<size_t>(MH) * 4, aFc1B);
    zeroBuffer(ctx.allocator, aFc1B, static_cast<size_t>(MH));
    VkBuffer bFc2W = createHostVisibleBuffer(ctx.allocator, static_cast<size_t>(3) * MH * 4, aFc2W);
    zeroBuffer(ctx.allocator, aFc2W, static_cast<size_t>(3) * MH);
    VkBuffer bFc2B = createHostVisibleBuffer(ctx.allocator, 3 * 4, aFc2B);
    zeroBuffer(ctx.allocator, aFc2B, 3);
    VkBuffer bBguvProxy = createHostVisibleBuffer(ctx.allocator, proxyScalarN * 2 * 4, aBguvProxy);
    VkBuffer bBgFeatures = createHostVisibleBuffer(ctx.allocator, proxyScalarN * static_cast<size_t>(MC) * 4, aBgFeatures);
    VkBuffer bMemoryColorDummy = createHostVisibleBuffer(ctx.allocator, proxyScalarN * 3 * 4, aMemoryColorDummy);

    writeDescriptorSet(ctx.device, bguvSet, {bBguvProxy});
    writeDescriptorSet(ctx.device, memorySet,
                        {bBguvProxy, bBgTexture, bFc1W, bFc1B, bFc2W, bFc2B, bBgFeatures, bMemoryColorDummy});

    uint32_t totalProxyThreads = static_cast<uint32_t>(proxyScalarN);

    for (int t = 0; t < meta.num_frames; ++t) {
        std::string suf = "_f" + std::to_string(t) + ".npy";
        auto kParams = DlssIO::readNpyFloat32(dumpDir + "/k_params_proxy" + suf);
        auto camToWorld = DlssIO::readNpyFloat32(dumpDir + "/cam_to_world" + suf);
        if (kParams.data.size() != 4 || camToWorld.data.size() != 16) {
            std::cerr << "[error] frame " << t << ": k_params_proxy/cam_to_world size mismatch" << std::endl;
            return 1;
        }
        BguvPush bguvPush{};
        bguvPush.width = static_cast<uint32_t>(meta.proxy_w);
        bguvPush.height = static_cast<uint32_t>(meta.proxy_h);
        std::copy(kParams.data.begin(), kParams.data.end(), bguvPush.k_params);
        glm::mat4 camToWorldGlm = rowMajorToGlm4x4(camToWorld.data);
        std::memcpy(bguvPush.cam_to_world, &camToWorldGlm[0][0], 64);
        std::copy(bgSphere.begin(), bgSphere.end(), bguvPush.bg_sphere);

        dispatchOne(ctx, bguvPipe, bguvPL, bguvSet, &bguvPush, sizeof(bguvPush), totalProxyThreads);

        MemoryPush memPush = {static_cast<uint32_t>(meta.proxy_w), static_cast<uint32_t>(meta.proxy_h),
                               static_cast<uint32_t>(TEXW), static_cast<uint32_t>(TEXH)};
        dispatchOne(ctx, memoryPipe, memoryPL, memorySet, &memPush, sizeof(memPush), totalProxyThreads);

        auto feat = downloadFloats(ctx.allocator, aBgFeatures, proxyScalarN * static_cast<size_t>(MC));
        DlssIO::writeNpyFloat32(outDir + "/bg_features_f" + std::to_string(t) + ".npy", feat,
                                 {meta.proxy_h, meta.proxy_w, MC});
        std::cout << "[dump-bg-features] frame " << t << " done" << std::endl;
    }

    vkDestroyPipeline(ctx.device, bguvPipe, nullptr);
    vkDestroyPipeline(ctx.device, memoryPipe, nullptr);
    vkDestroyPipelineLayout(ctx.device, bguvPL, nullptr);
    vkDestroyPipelineLayout(ctx.device, memoryPL, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, bguvLayout, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, memoryLayout, nullptr);
    vkDestroyDescriptorPool(ctx.device, pool, nullptr);
    vmaDestroyBuffer(ctx.allocator, bBgTexture, aBgTexture);
    vmaDestroyBuffer(ctx.allocator, bFc1W, aFc1W);
    vmaDestroyBuffer(ctx.allocator, bFc1B, aFc1B);
    vmaDestroyBuffer(ctx.allocator, bFc2W, aFc2W);
    vmaDestroyBuffer(ctx.allocator, bFc2B, aFc2B);
    vmaDestroyBuffer(ctx.allocator, bBguvProxy, aBguvProxy);
    vmaDestroyBuffer(ctx.allocator, bBgFeatures, aBgFeatures);
    vmaDestroyBuffer(ctx.allocator, bMemoryColorDummy, aMemoryColorDummy);

    std::cout << "[dump-bg-features] done: " << meta.num_frames << " frames written to " << outDir << std::endl;
    return 0;
}

// --- ReconstructLive (task 4) --------------------------------------------
// Same pipelines/buffers/descriptor sets as runReconstructDump's per-call
// body, but created once in init() and reused by every run() call instead
// of being recreated (and torn down) every time -- see reconstruct_pass.h's
// class comment for the measured savings.

struct ReconstructLive::Impl {
    Meta meta;
    bool hasMemory = false;
    int totalCh = 0, NBLEND = 0, SP = 0;
    size_t proxyColorN = 0, mvProxyN = 0, targetColorN = 0, targetScalarN = 0, hiddenN = 0, packedN = 0, blendN = 0;

    VkDescriptorSetLayout warpLayout = VK_NULL_HANDLE, applyLayout = VK_NULL_HANDLE, blendLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout bguvLayout = VK_NULL_HANDLE, memoryLayout = VK_NULL_HANDLE;
    VkPipelineLayout warpPL = VK_NULL_HANDLE, applyPL = VK_NULL_HANDLE, blendPL = VK_NULL_HANDLE;
    VkPipelineLayout bguvPL = VK_NULL_HANDLE, memoryPL = VK_NULL_HANDLE;
    VkPipeline warpPipe = VK_NULL_HANDLE, applyPipe = VK_NULL_HANDLE, blendPipe = VK_NULL_HANDLE;
    VkPipeline bguvPipe = VK_NULL_HANDLE, memoryPipe = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet warpSet = VK_NULL_HANDLE, applySet = VK_NULL_HANDLE, blendSet = VK_NULL_HANDLE;
    VkDescriptorSet bguvSet = VK_NULL_HANDLE, memorySet = VK_NULL_HANDLE;

    VkBuffer bProxyColor = VK_NULL_HANDLE, bMvProxy = VK_NULL_HANDLE, bWarped = VK_NULL_HANDLE, bPrevColor = VK_NULL_HANDLE;
    VkBuffer bPacked = VK_NULL_HANDLE, bSpatial = VK_NULL_HANDLE, bBlend = VK_NULL_HANDLE, bHidden = VK_NULL_HANDLE;
    VkBuffer bDisocc = VK_NULL_HANDLE, bOutColor = VK_NULL_HANDLE;
    VmaAllocation aProxyColor{}, aMvProxy{}, aWarped{}, aPrevColor{};
    VmaAllocation aPacked{}, aSpatial{}, aBlend{}, aHidden{}, aDisocc{}, aOutColor{};

    VkBuffer bBgTexture = VK_NULL_HANDLE, bFc1W = VK_NULL_HANDLE, bFc1B = VK_NULL_HANDLE;
    VkBuffer bFc2W = VK_NULL_HANDLE, bFc2B = VK_NULL_HANDLE, bBguvTarget = VK_NULL_HANDLE;
    VkBuffer bBgFeaturesDummy = VK_NULL_HANDLE, bMemoryColor = VK_NULL_HANDLE;
    VmaAllocation aBgTexture{}, aFc1W{}, aFc1B{}, aFc2W{}, aFc2B{}, aBguvTarget{}, aBgFeaturesDummy{}, aMemoryColor{};
    std::array<float, 4> bgSphere{};

    GpuTimestampBracket gpuBracket;
    std::vector<float> outColorHost, hiddenHost, bguvHost;
    VulkanContext* ctx = nullptr;

    // Task B: one persistent command buffer + one fence for the whole
    // 5-dispatch reconstruct chain, allocated/created once in init() and
    // re-recorded every run() call (vkBeginCommandBuffer implicitly resets
    // it -- ctx.commandPool is created with
    // VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT on both the Android
    // and desktop VulkanContext, see android_vulkan_context.cpp/
    // vulkan_context.cpp) instead of dispatchOne's per-call
    // allocate+free+fully-synchronous-submit.
    VkCommandBuffer reconCmd = VK_NULL_HANDLE;
    VkFence reconFence = VK_NULL_HANDLE;
};

ReconstructLive::~ReconstructLive() {
    if (!impl_) return;
    VkDevice device = impl_->ctx ? impl_->ctx->device : VK_NULL_HANDLE;
    if (device) {
        vkDeviceWaitIdle(device);
        if (impl_->reconFence != VK_NULL_HANDLE) vkDestroyFence(device, impl_->reconFence, nullptr);
        if (impl_->reconCmd != VK_NULL_HANDLE) {
            vkFreeCommandBuffers(device, impl_->ctx->commandPool, 1, &impl_->reconCmd);
        }
        impl_->gpuBracket.destroy(device);
        vkDestroyPipeline(device, impl_->warpPipe, nullptr);
        vkDestroyPipeline(device, impl_->applyPipe, nullptr);
        vkDestroyPipeline(device, impl_->blendPipe, nullptr);
        vkDestroyPipelineLayout(device, impl_->warpPL, nullptr);
        vkDestroyPipelineLayout(device, impl_->applyPL, nullptr);
        vkDestroyPipelineLayout(device, impl_->blendPL, nullptr);
        vkDestroyDescriptorSetLayout(device, impl_->warpLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, impl_->applyLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, impl_->blendLayout, nullptr);
        if (impl_->hasMemory) {
            vkDestroyPipeline(device, impl_->bguvPipe, nullptr);
            vkDestroyPipeline(device, impl_->memoryPipe, nullptr);
            vkDestroyPipelineLayout(device, impl_->bguvPL, nullptr);
            vkDestroyPipelineLayout(device, impl_->memoryPL, nullptr);
            vkDestroyDescriptorSetLayout(device, impl_->bguvLayout, nullptr);
            vkDestroyDescriptorSetLayout(device, impl_->memoryLayout, nullptr);
            auto& alloc = impl_->ctx->allocator;
            vmaDestroyBuffer(alloc, impl_->bBgTexture, impl_->aBgTexture);
            vmaDestroyBuffer(alloc, impl_->bFc1W, impl_->aFc1W);
            vmaDestroyBuffer(alloc, impl_->bFc1B, impl_->aFc1B);
            vmaDestroyBuffer(alloc, impl_->bFc2W, impl_->aFc2W);
            vmaDestroyBuffer(alloc, impl_->bFc2B, impl_->aFc2B);
            vmaDestroyBuffer(alloc, impl_->bBguvTarget, impl_->aBguvTarget);
            vmaDestroyBuffer(alloc, impl_->bBgFeaturesDummy, impl_->aBgFeaturesDummy);
            vmaDestroyBuffer(alloc, impl_->bMemoryColor, impl_->aMemoryColor);
        }
        vkDestroyDescriptorPool(device, impl_->pool, nullptr);
        auto& alloc = impl_->ctx->allocator;
        vmaDestroyBuffer(alloc, impl_->bProxyColor, impl_->aProxyColor);
        vmaDestroyBuffer(alloc, impl_->bMvProxy, impl_->aMvProxy);
        vmaDestroyBuffer(alloc, impl_->bWarped, impl_->aWarped);
        vmaDestroyBuffer(alloc, impl_->bPrevColor, impl_->aPrevColor);
        vmaDestroyBuffer(alloc, impl_->bPacked, impl_->aPacked);
        vmaDestroyBuffer(alloc, impl_->bSpatial, impl_->aSpatial);
        vmaDestroyBuffer(alloc, impl_->bBlend, impl_->aBlend);
        vmaDestroyBuffer(alloc, impl_->bHidden, impl_->aHidden);
        vmaDestroyBuffer(alloc, impl_->bDisocc, impl_->aDisocc);
        vmaDestroyBuffer(alloc, impl_->bOutColor, impl_->aOutColor);
    }
    delete impl_;
}

void ReconstructLive::init(VulkanContext& ctx, const std::string& pipelineBase,
                            int s, int k, int paramStride, int hidden,
                            int proxyW, int proxyH, int targetW, int targetH, int netW, int netH,
                            int memoryChannels, int memoryHidden, int texW, int texH,
                            const std::string& textureNpyPath, const std::string& bgSphereNpyPath,
                            const std::string& memoryHeadNpzPath) {
    impl_ = new Impl();
    impl_->ctx = &ctx;
    Impl& I = *impl_;
    I.meta.s = s; I.meta.k = k; I.meta.param_stride = paramStride; I.meta.hidden = hidden;
    I.meta.proxy_w = proxyW; I.meta.proxy_h = proxyH; I.meta.target_w = targetW; I.meta.target_h = targetH;
    I.meta.net_w = netW; I.meta.net_h = netH; I.meta.num_frames = 1;
    I.meta.memory_channels = memoryChannels; I.meta.memory_hidden = memoryHidden;
    I.meta.tex_w = texW; I.meta.tex_h = texH;
    I.hasMemory = memoryChannels > 0;
    const int S = s, K = k, PS = paramStride, HIDDEN = hidden;
    I.SP = S * PS;
    I.NBLEND = I.hasMemory ? 3 : 1;
    I.totalCh = I.SP * I.SP * K * K + I.SP * I.SP * I.NBLEND + HIDDEN;
    const int MC = memoryChannels, MH = memoryHidden, TEXW = texW, TEXH = texH;

    I.warpLayout = makeSetLayout(ctx.device, 3);
    I.applyLayout = makeSetLayout(ctx.device, 5);
    I.blendLayout = makeSetLayout(ctx.device, I.hasMemory ? 6 : 5);
    I.warpPL = makePipelineLayout(ctx.device, I.warpLayout, 16);
    I.applyPL = makePipelineLayout(ctx.device, I.applyLayout, 32);
    I.blendPL = makePipelineLayout(ctx.device, I.blendLayout, 16);
    I.warpPipe = makeComputePipeline(ctx.device, I.warpPL, pipelineBase + ".warp.comp.spv");
    I.applyPipe = makeComputePipeline(ctx.device, I.applyPL, pipelineBase + ".apply.comp.spv");
    I.blendPipe = makeComputePipeline(ctx.device, I.blendPL, pipelineBase + ".blend.comp.spv");
    if (I.hasMemory) {
        I.bguvLayout = makeSetLayout(ctx.device, 1);
        I.memoryLayout = makeSetLayout(ctx.device, 8);
        I.bguvPL = makePipelineLayout(ctx.device, I.bguvLayout, sizeof(BguvPush));
        I.memoryPL = makePipelineLayout(ctx.device, I.memoryLayout, sizeof(MemoryPush));
        I.bguvPipe = makeComputePipeline(ctx.device, I.bguvPL, pipelineBase + ".bguv.comp.spv");
        I.memoryPipe = makeComputePipeline(ctx.device, I.memoryPL, pipelineBase + ".memory.comp.spv");
    }

    uint32_t totalDescriptors = 3 + 5 + (I.hasMemory ? 6u : 5u) + (I.hasMemory ? (1u + 8u) : 0u);
    uint32_t totalSets = I.hasMemory ? 5 : 3;
    VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, totalDescriptors};
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = totalSets;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr, &I.pool);

    std::vector<VkDescriptorSetLayout> layouts = {I.warpLayout, I.applyLayout, I.blendLayout};
    if (I.hasMemory) { layouts.push_back(I.bguvLayout); layouts.push_back(I.memoryLayout); }
    std::vector<VkDescriptorSet> sets(layouts.size());
    VkDescriptorSetAllocateInfo dsAlloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsAlloc.descriptorPool = I.pool;
    dsAlloc.descriptorSetCount = static_cast<uint32_t>(layouts.size());
    dsAlloc.pSetLayouts = layouts.data();
    vkAllocateDescriptorSets(ctx.device, &dsAlloc, sets.data());
    I.warpSet = sets[0]; I.applySet = sets[1]; I.blendSet = sets[2];
    if (I.hasMemory) { I.bguvSet = sets[3]; I.memorySet = sets[4]; }

    I.proxyColorN = static_cast<size_t>(proxyW) * proxyH * 3;
    I.mvProxyN = static_cast<size_t>(proxyW) * proxyH * 2;
    I.targetColorN = static_cast<size_t>(targetW) * targetH * 3;
    I.targetScalarN = static_cast<size_t>(targetW) * targetH;
    I.hiddenN = I.targetScalarN * HIDDEN;
    I.packedN = static_cast<size_t>(netW) * netH * I.totalCh;
    I.blendN = I.targetScalarN * I.NBLEND;

    I.bProxyColor = createHostVisibleBuffer(ctx.allocator, I.proxyColorN * 4, I.aProxyColor);
    I.bMvProxy = createHostVisibleBuffer(ctx.allocator, I.mvProxyN * 4, I.aMvProxy);
    I.bWarped = createHostVisibleBuffer(ctx.allocator, I.targetColorN * 4, I.aWarped);
    I.bPrevColor = createHostVisibleBuffer(ctx.allocator, I.targetColorN * 4, I.aPrevColor);
    I.bPacked = createHostVisibleBuffer(ctx.allocator, I.packedN * 4, I.aPacked);
    I.bSpatial = createHostVisibleBuffer(ctx.allocator, I.targetColorN * 4, I.aSpatial);
    I.bBlend = createHostVisibleBuffer(ctx.allocator, I.blendN * 4, I.aBlend);
    I.bHidden = createHostVisibleBuffer(ctx.allocator, I.hiddenN * 4, I.aHidden);
    I.bDisocc = createHostVisibleBuffer(ctx.allocator, I.targetScalarN * 4, I.aDisocc);
    I.bOutColor = createHostVisibleBuffer(ctx.allocator, I.targetColorN * 4, I.aOutColor);
    zeroBuffer(ctx.allocator, I.aPrevColor, I.targetColorN);

    if (I.hasMemory) {
        auto tex = DlssIO::readNpyFloat32(textureNpyPath);
        auto sph = DlssIO::readNpyFloat32(bgSphereNpyPath);
        std::copy(sph.data.begin(), sph.data.end(), I.bgSphere.begin());
        auto fc1w = DlssIO::readNpzMemberFloat32(memoryHeadNpzPath, "fc1_w");
        auto fc1b = DlssIO::readNpzMemberFloat32(memoryHeadNpzPath, "fc1_b");
        auto fc2w = DlssIO::readNpzMemberFloat32(memoryHeadNpzPath, "fc2_w");
        auto fc2b = DlssIO::readNpzMemberFloat32(memoryHeadNpzPath, "fc2_b");
        I.bBgTexture = createHostVisibleBuffer(ctx.allocator, tex.data.size() * 4, I.aBgTexture);
        uploadFloats(ctx.allocator, I.aBgTexture, tex.data);
        I.bFc1W = createHostVisibleBuffer(ctx.allocator, fc1w.data.size() * 4, I.aFc1W);
        uploadFloats(ctx.allocator, I.aFc1W, fc1w.data);
        I.bFc1B = createHostVisibleBuffer(ctx.allocator, fc1b.data.size() * 4, I.aFc1B);
        uploadFloats(ctx.allocator, I.aFc1B, fc1b.data);
        I.bFc2W = createHostVisibleBuffer(ctx.allocator, fc2w.data.size() * 4, I.aFc2W);
        uploadFloats(ctx.allocator, I.aFc2W, fc2w.data);
        I.bFc2B = createHostVisibleBuffer(ctx.allocator, fc2b.data.size() * 4, I.aFc2B);
        uploadFloats(ctx.allocator, I.aFc2B, fc2b.data);
        I.bBguvTarget = createHostVisibleBuffer(ctx.allocator, I.targetScalarN * 2 * 4, I.aBguvTarget);
        I.bBgFeaturesDummy = createHostVisibleBuffer(ctx.allocator, I.targetScalarN * static_cast<size_t>(MC) * 4,
                                                       I.aBgFeaturesDummy);
        I.bMemoryColor = createHostVisibleBuffer(ctx.allocator, I.targetColorN * 4, I.aMemoryColor);
    }

    writeDescriptorSet(ctx.device, I.warpSet, {I.bPrevColor, I.bMvProxy, I.bWarped});
    writeDescriptorSet(ctx.device, I.applySet, {I.bPacked, I.bProxyColor, I.bSpatial, I.bBlend, I.bHidden});
    if (I.hasMemory) {
        writeDescriptorSet(ctx.device, I.blendSet, {I.bSpatial, I.bWarped, I.bBlend, I.bDisocc, I.bMemoryColor, I.bOutColor});
        writeDescriptorSet(ctx.device, I.bguvSet, {I.bBguvTarget});
        writeDescriptorSet(ctx.device, I.memorySet,
                            {I.bBguvTarget, I.bBgTexture, I.bFc1W, I.bFc1B, I.bFc2W, I.bFc2B, I.bBgFeaturesDummy, I.bMemoryColor});
    } else {
        writeDescriptorSet(ctx.device, I.blendSet, {I.bSpatial, I.bWarped, I.bBlend, I.bDisocc, I.bOutColor});
    }

    I.gpuBracket.init(ctx);
    I.outColorHost.resize(I.targetColorN);
    I.hiddenHost.resize(I.hiddenN);
    if (I.hasMemory) I.bguvHost.resize(I.targetScalarN * 2);

    // Task B: persistent command buffer + fence for the batched dispatch
    // chain in run() (see the Impl field comment above).
    VkCommandBufferAllocateInfo cbAlloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbAlloc.commandPool = ctx.commandPool;
    cbAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbAlloc.commandBufferCount = 1;
    vkAllocateCommandBuffers(ctx.device, &cbAlloc, &I.reconCmd);
    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    vkCreateFence(ctx.device, &fenceInfo, nullptr, &I.reconFence);

    (void)TEXW; (void)TEXH; (void)MC; (void)MH;
}

void ReconstructLive::reset(VulkanContext& ctx) {
    if (!impl_) return;
    zeroBuffer(ctx.allocator, impl_->aPrevColor, impl_->targetColorN);
}

ReconstructLive::FrameOutputs ReconstructLive::run(VulkanContext& ctx, const FrameInputs& in,
                                                     ReconstructTimingsMs* outTimings, bool downloadDebugOutputs) {
    Impl& I = *impl_;
    auto tUpload = std::chrono::high_resolution_clock::now();
    // Raw-pointer overload (see its own comment above) -- was
    // std::vector<float>(in.X, in.X+N) per call, a full extra CPU copy of
    // every input (worst: I.aPacked, tens of MB) before uploadFloats' own
    // memcpy did the real work again.
    uploadFloats(ctx.allocator, I.aProxyColor, in.proxyColor, I.proxyColorN);
    uploadFloats(ctx.allocator, I.aMvProxy, in.mvProxy, I.mvProxyN);
    uploadFloats(ctx.allocator, I.aPacked, in.packed, I.packedN);
    uploadFloats(ctx.allocator, I.aDisocc, in.disocc, I.targetScalarN);
    if (outTimings != nullptr) outTimings->uploadMs += msSince(tUpload);

    auto tDispatch = std::chrono::high_resolution_clock::now();

    // Task B (docs/rendering-engines.md, "reconstruct-pass queue waits"):
    // one command buffer for the whole bguv->memory->warp->apply->blend
    // chain, pipeline barriers between dispatches (recordDispatchBarriered,
    // above), one fence submit+wait for the frame -- replaces what was up
    // to 5 fully-synchronous dispatchOne submissions (5 vkQueueWaitIdle
    // round trips) plus 2 more from GpuTimestampBracket's own
    // writeStart()/writeEnd() single-time commands whenever timing was
    // requested (7 total CPU-GPU stalls/frame). The timestamp queries are
    // folded into this same command buffer instead. Same dispatches, same
    // descriptor sets, same push constants, same buffers, same ordering
    // (each barrier makes every earlier write visible to every later
    // read/write before the next dispatch starts) -- only the submission
    // granularity changed, so output is bit-identical to the old
    // per-dispatch-drain path (verified: see the task's PSNR/diff report).
    vkResetCommandBuffer(I.reconCmd, 0);
    VkCommandBufferBeginInfo cmdBeginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    cmdBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(I.reconCmd, &cmdBeginInfo);

    if (outTimings != nullptr) {
        vkCmdResetQueryPool(I.reconCmd, I.gpuBracket.pool, 0, 2);
        vkCmdWriteTimestamp(I.reconCmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, I.gpuBracket.pool, 0);
    }

    if (I.hasMemory) {
        BguvPush bguvPush{};
        bguvPush.width = static_cast<uint32_t>(I.meta.target_w);
        bguvPush.height = static_cast<uint32_t>(I.meta.target_h);
        std::copy(in.kParams, in.kParams + 4, bguvPush.k_params);
        std::vector<float> camToWorldVec(in.camToWorld, in.camToWorld + 16);
        glm::mat4 camToWorldGlm = rowMajorToGlm4x4(camToWorldVec);
        std::memcpy(bguvPush.cam_to_world, &camToWorldGlm[0][0], 64);
        std::copy(I.bgSphere.begin(), I.bgSphere.end(), bguvPush.bg_sphere);
        recordDispatchBarriered(I.reconCmd, I.bguvPipe, I.bguvPL, I.bguvSet, &bguvPush, sizeof(bguvPush),
                                 static_cast<uint32_t>(I.targetScalarN), /*barrierAfter=*/true);

        MemoryPush memPush = {static_cast<uint32_t>(I.meta.target_w), static_cast<uint32_t>(I.meta.target_h),
                               static_cast<uint32_t>(I.meta.tex_w), static_cast<uint32_t>(I.meta.tex_h)};
        recordDispatchBarriered(I.reconCmd, I.memoryPipe, I.memoryPL, I.memorySet, &memPush, sizeof(memPush),
                                 static_cast<uint32_t>(I.targetScalarN), /*barrierAfter=*/true);
    }

    struct WarpPush { uint32_t target_w, target_h, proxy_w, proxy_h; };
    WarpPush warpPush = {static_cast<uint32_t>(I.meta.target_w), static_cast<uint32_t>(I.meta.target_h),
                          static_cast<uint32_t>(I.meta.proxy_w), static_cast<uint32_t>(I.meta.proxy_h)};
    recordDispatchBarriered(I.reconCmd, I.warpPipe, I.warpPL, I.warpSet, &warpPush, sizeof(warpPush),
                             static_cast<uint32_t>(I.targetScalarN), /*barrierAfter=*/true);

    struct ApplyPush {
        uint32_t target_w, target_h, proxy_w, proxy_h, net_w, net_h;
        float jitter_x, jitter_y;
    };
    ApplyPush applyPush = {static_cast<uint32_t>(I.meta.target_w), static_cast<uint32_t>(I.meta.target_h),
                            static_cast<uint32_t>(I.meta.proxy_w), static_cast<uint32_t>(I.meta.proxy_h),
                            static_cast<uint32_t>(I.meta.net_w), static_cast<uint32_t>(I.meta.net_h),
                            in.jitterX, in.jitterY};
    recordDispatchBarriered(I.reconCmd, I.applyPipe, I.applyPL, I.applySet, &applyPush, sizeof(applyPush),
                             static_cast<uint32_t>(I.targetScalarN), /*barrierAfter=*/true);

    struct BlendPush { uint32_t target_w, target_h, _pad0, _pad1; };
    BlendPush blendPush = {static_cast<uint32_t>(I.meta.target_w), static_cast<uint32_t>(I.meta.target_h), 0, 0};
    // No barrier after the last dispatch: the fence wait below already
    // guarantees all of this command buffer's work (including blend's
    // writes to I.bOutColor) is complete -- and, since these are the same
    // HOST_COHERENT VMA buffers downloadFloats()/copyBufferHost() already
    // read right after the old per-dispatch vkQueueWaitIdle without an
    // extra barrier, complete-and-host-visible -- before the CPU download
    // below runs.
    recordDispatchBarriered(I.reconCmd, I.blendPipe, I.blendPL, I.blendSet, &blendPush, sizeof(blendPush),
                             static_cast<uint32_t>(I.targetScalarN), /*barrierAfter=*/false);

    if (outTimings != nullptr) {
        vkCmdWriteTimestamp(I.reconCmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, I.gpuBracket.pool, 1);
    }
    vkEndCommandBuffer(I.reconCmd);

    vkResetFences(ctx.device, 1, &I.reconFence);
    VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &I.reconCmd;
    vkQueueSubmit(ctx.graphicsQueue, 1, &submitInfo, I.reconFence);
    vkWaitForFences(ctx.device, 1, &I.reconFence, VK_TRUE, UINT64_MAX);

    if (outTimings != nullptr) {
        outTimings->dispatchGpuMs += I.gpuBracket.readDeltaMs(ctx.device);
        outTimings->dispatchCpuMs += msSince(tDispatch);
    }

    // Task 4 (docs/rendering-engines.md, remove the readback stalls/copies
    // where cheap -- "debug dumps off by default"): I.aHidden/I.aBguvTarget
    // are write-only scratch from the "apply"/"bguv" dispatches above --
    // NEITHER feeds back into any later dispatch in this same run() (only
    // aPrevColor does, via the copyBufferHost below) NOR does the live
    // Reconstruction-mode display caller (runReconstructionModeFrame) ever
    // read FrameOutputs::hidden/bguvTarget -- they only exist for the
    // offline/dump tooling's hidden_f{t}.npy/bguv_target_f{t}.npy artifacts
    // (runReconstructDump's OWN, separate download path, unaffected by this
    // flag). Downloading them here on every live-displayed frame anyway was
    // reading back ~21MB/frame (hiddenN=targetW*targetH*hidden, tens of MB
    // at this model's shapes, plus bguvTarget when scene-memory is on) this
    // caller then immediately discarded -- measured as the majority of
    // RECON_TIMING's recon_download. `downloadDebugOutputs` (opt-in, off by
    // default) restores the old always-download behavior for anyone who
    // does want to inspect them live.
    auto tDownload = std::chrono::high_resolution_clock::now();
    I.outColorHost = downloadFloats(ctx.allocator, I.aOutColor, I.targetColorN);
    if (downloadDebugOutputs) {
        I.hiddenHost = downloadFloats(ctx.allocator, I.aHidden, I.hiddenN);
        if (I.hasMemory) I.bguvHost = downloadFloats(ctx.allocator, I.aBguvTarget, I.targetScalarN * 2);
    }
    if (outTimings != nullptr) outTimings->downloadMs += msSince(tDownload);

    // Carry forward as next call's prev_color -- see class comment re: this
    // now being genuine cross-frame continuity, unlike runReconstructDump's
    // per-call-fresh-zeroed buffer.
    copyBufferHost(ctx.allocator, I.aPrevColor, I.aOutColor, I.targetColorN);

    FrameOutputs out;
    out.outColor = I.outColorHost.data();
    out.hidden = downloadDebugOutputs ? I.hiddenHost.data() : nullptr;
    out.bguvTarget = (downloadDebugOutputs && I.hasMemory) ? I.bguvHost.data() : nullptr;
    return out;
}
