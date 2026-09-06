#include "reconstruct_runner.h"
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

namespace fs = std::filesystem;

namespace {

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

int runReconstructDump(const std::string& dumpDir, const std::string& outDir,
                        const std::string& pipelineBase) {
    VulkanContext ctx;
    try {
        ctx.init(false, true, nullptr, false);
    } catch (const std::exception& e) {
        std::cerr << "[error] Failed to initialize Vulkan: " << e.what() << std::endl;
        return 1;
    }
    return runReconstructDump(ctx, dumpDir, outDir, pipelineBase);
}

int runReconstructDump(VulkanContext& ctx, const std::string& dumpDir, const std::string& outDirIn,
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

    for (int t = 0; t < meta.num_frames; ++t) {
        std::string suf = "_f" + std::to_string(t) + ".npy";
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

        uploadFloats(ctx.allocator, aProxyColor, proxyColor.data);
        uploadFloats(ctx.allocator, aMvProxy, mvProxy.data);
        uploadFloats(ctx.allocator, aPacked, packed.data);
        uploadFloats(ctx.allocator, aDisocc, disocc.data);

        if (hasMemory) {
            auto kParams = DlssIO::readNpyFloat32(dumpDir + "/k_params" + suf);
            auto camToWorld = DlssIO::readNpyFloat32(dumpDir + "/cam_to_world" + suf);
            if (kParams.data.size() != 4 || camToWorld.data.size() != 16) {
                std::cerr << "[error] frame " << t << ": k_params/cam_to_world size mismatch" << std::endl;
                return 1;
            }
            BguvPush bguvPush{};
            bguvPush.width = static_cast<uint32_t>(meta.target_w);
            bguvPush.height = static_cast<uint32_t>(meta.target_h);
            std::copy(kParams.data.begin(), kParams.data.end(), bguvPush.k_params);
            glm::mat4 camToWorldGlm = rowMajorToGlm4x4(camToWorld.data);
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

        auto outColor = downloadFloats(ctx.allocator, aOutColor, targetColorN);
        auto hidden = downloadFloats(ctx.allocator, aHidden, hiddenN);
        DlssIO::writeNpyFloat32(outDir + "/out_f" + std::to_string(t) + ".npy", outColor,
                                 {meta.target_h, meta.target_w, 3});
        DlssIO::writeNpyFloat32(outDir + "/hidden_f" + std::to_string(t) + ".npy", hidden,
                                 {meta.target_h, meta.target_w, HIDDEN});
        if (hasMemory) {
            auto bguvOut = downloadFloats(ctx.allocator, aBguvTarget, targetScalarN * 2);
            DlssIO::writeNpyFloat32(outDir + "/bguv_target_f" + std::to_string(t) + ".npy", bguvOut,
                                     {meta.target_h, meta.target_w, 2});
        }

        // Carry `out` forward as next frame's `prev_color` (host-side copy
        // -- all buffers are host-visible/coherent VMA_MEMORY_USAGE_CPU_TO_GPU,
        // so no GPU copy command or extra barrier is needed here; the prior
        // dispatch's own endSingleTimeCommands() already fully drained the
        // queue, guaranteeing the CPU sees the GPU's writes).
        copyBufferHost(ctx.allocator, aPrevColor, aOutColor, targetColorN);

        std::cout << "[reconstruct] frame " << t << " done" << std::endl;
    }

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

    std::cout << "[reconstruct] done: " << meta.num_frames << " frames written to " << outDir << std::endl;
    return 0;
}

int runDumpBgFeatures(const std::string& dumpDir, const std::string& outDir,
                       const std::string& pipelineBase) {
    VulkanContext ctx;
    try {
        ctx.init(false, true, nullptr, false);
    } catch (const std::exception& e) {
        std::cerr << "[error] Failed to initialize Vulkan: " << e.what() << std::endl;
        return 1;
    }
    return runDumpBgFeatures(ctx, dumpDir, outDir, pipelineBase);
}

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
