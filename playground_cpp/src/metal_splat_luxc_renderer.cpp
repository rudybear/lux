#include "metal_splat_luxc_renderer.h"
#include "gltf_loader.h"
#include <algorithm>
#include <numeric>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <cmath>

namespace fs = std::filesystem;

namespace {

// Same lightweight substring-scan JSON reader as splat_renderer.cpp's
// readShaderShDegree/readShaderBoolFlag (luxc's reflection writer always
// emits `"key": true`/`"key": false`/`"key": N` with exact spacing).
std::string readFile(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return {};
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

bool readGsBoolFlag(const std::string& content, const std::string& key) {
    auto gsPos = content.find("\"gaussian_splatting\"");
    if (gsPos == std::string::npos) return false;
    std::string needleTrue = "\"" + key + "\": true";
    auto keyPos = content.find("\"" + key + "\"", gsPos);
    if (keyPos == std::string::npos) return false;
    return content.compare(keyPos, needleTrue.size(), needleTrue) == 0;
}

// Same substring-scan approach, for a `"key": "value"` string field (used
// for "aux_precision": "half"/"float" -- bench/lux_perf_ablation.md task 2
// follow-up).
std::string readGsStringFlag(const std::string& content, const std::string& key,
                              const std::string& fallback) {
    auto gsPos = content.find("\"gaussian_splatting\"");
    if (gsPos == std::string::npos) return fallback;
    auto keyPos = content.find("\"" + key + "\"", gsPos);
    if (keyPos == std::string::npos) return fallback;
    auto colonPos = content.find(':', keyPos);
    if (colonPos == std::string::npos) return fallback;
    auto openQuote = content.find('"', colonPos);
    if (openQuote == std::string::npos) return fallback;
    auto closeQuote = content.find('"', openQuote + 1);
    if (closeQuote == std::string::npos) return fallback;
    return content.substr(openQuote + 1, closeQuote - openQuote - 1);
}

// SPIRV-Cross's MSL backend drops resource bindings that are declared but
// never actually read/written in the shader body (e.g. splat_expander.py's
// `visible_count` -- described as "atomic counter (incremented per visible
// splat)" but not yet wired to an actual atomic_add anywhere; it's declared
// solely for a future compaction pass) -- unlike Vulkan, which keeps every
// declared descriptor-set binding regardless of whether the shader body
// touches it, since VkDescriptorSetLayout is declaration-based, not
// usage-based. `trySetBuffer` treats a missing binding as "this shader
// build doesn't need it" rather than an error.
void trySetBuffer(MTL::ComputeCommandEncoder* enc, const TranspiledShader& s,
                   MTL::Buffer* buf, uint32_t binding) {
    uint32_t idx = s.findBufferIndex(0, binding);
    if (idx != UINT32_MAX) enc->setBuffer(buf, 0, idx);
}

void trySetVertexBuffer(MTL::RenderCommandEncoder* enc, const TranspiledShader& s,
                         MTL::Buffer* buf, uint32_t binding) {
    uint32_t idx = s.findBufferIndex(0, binding);
    if (idx != UINT32_MAX) enc->setVertexBuffer(buf, 0, idx);
}

// Same jitter math as the Vulkan splat renderer's applySplatJitter (NOT
// MetalSplatRenderer's negated Y version): this backend runs the actual
// Vulkan-authored, unmodified compiled shader, and (since the createPipelines
// negative-height viewport flip) now reproduces Vulkan's rendering
// convention exactly end to end -- confirmed empirically (83.7 dB colour
// match, 1.15e-7 depth error, vs. Vulkan on the juggle DLSS scene) -- so
// the jitter's Y sign must follow Vulkan's convention too, not Metal's.
glm::mat4 applyLuxcSplatJitter(glm::mat4 proj, float jitterXPixels, float jitterYPixels,
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

} // namespace

MetalSplatLuxcRenderer::~MetalSplatLuxcRenderer() {}

// --------------------------------------------------------------------------
// Render targets (identical formats to MetalSplatRenderer)
// --------------------------------------------------------------------------

void MetalSplatLuxcRenderer::createRenderTargets(MetalContext& ctx) {
    auto* colorDesc = MTL::TextureDescriptor::texture2DDescriptor(
        getColorFormat(), width_, height_, false);
    colorDesc->setUsage(MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead);
    colorDesc->setStorageMode(MTL::StorageModePrivate);
    colorTarget_ = ctx.newTexture(colorDesc);
    colorTarget_->setLabel(NS::String::string("SplatColorLuxc", NS::UTF8StringEncoding));

    auto* depthDesc = MTL::TextureDescriptor::texture2DDescriptor(
        MTL::PixelFormatDepth32Float, width_, height_, false);
    depthDesc->setUsage(MTL::TextureUsageRenderTarget);
    depthDesc->setStorageMode(MTL::StorageModePrivate);
    depthTarget_ = ctx.newTexture(depthDesc);
    depthTarget_->setLabel(NS::String::string("SplatDepthLuxc", NS::UTF8StringEncoding));

    // Packed RGBA32Float `out_aux` attachment (bench/lux_perf_ablation.md
    // task 2): (mv.x*alpha, mv.y*alpha, depth*alpha, alpha) -- replaces the
    // earlier two-texture (RGBA32Float each, each with an always-0 `.z`)
    // motionTarget_/expectedDepthTarget_ design with one that packs 3 real
    // values with ZERO wasted lanes; see getAuxTexture()'s header comment
    // and luxc/expansion/splat_expander.py's out_aux comment. `.w` MUST
    // stay genuine alpha (required for correct hardware blend
    // accumulation). Format is a compile-time host hint (aux_precision:
    // float/half) -- see getAuxFormat()'s comment.
    if (hasMotionVectors_ || hasExpectedDepth_) {
        auto* auxDesc = MTL::TextureDescriptor::texture2DDescriptor(
            getAuxFormat(), width_, height_, false);
        auxDesc->setUsage(MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead);
        auxDesc->setStorageMode(MTL::StorageModePrivate);
        auxTarget_ = ctx.newTexture(auxDesc);
        auxTarget_->setLabel(NS::String::string("SplatAuxLuxc", NS::UTF8StringEncoding));
    }
    // Second, smaller `out_fg` attachment (fg*alpha, ..., alpha) -- see
    // getFgTexture()'s comment for why this can't share out_aux's lanes.
    if (hasForegroundCoverage_) {
        auto* fgDesc = MTL::TextureDescriptor::texture2DDescriptor(
            getFgFormat(), width_, height_, false);
        fgDesc->setUsage(MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead);
        fgDesc->setStorageMode(MTL::StorageModePrivate);
        fgTarget_ = ctx.newTexture(fgDesc);
        fgTarget_->setLabel(NS::String::string("SplatFgLuxc", NS::UTF8StringEncoding));
    }
}

// --------------------------------------------------------------------------
// Pipeline creation: transpile the luxc-compiled SPIR-V to MSL
// --------------------------------------------------------------------------

void MetalSplatLuxcRenderer::createPipelines(MetalContext& ctx) {
    NS::Error* error = nullptr;

    std::string compJson = readFile(shaderBase_ + ".comp.json");
    hasMotionVectors_ = readGsBoolFlag(compJson, "motion_vectors");
    hasExpectedDepth_ = readGsBoolFlag(compJson, "expected_depth");
    hasForegroundCoverage_ = readGsBoolFlag(compJson, "foreground_coverage");
    auxPrecisionHalf_ = readGsStringFlag(compJson, "aux_precision", "float") == "half";

    transpiler_.transpileInto(compShader_, shaderBase_ + ".comp.spv", SpvExecModel::GLCompute);
    computePipeline_ = ctx.device->newComputePipelineState(compShader_.function, &error);
    if (!computePipeline_) {
        std::string msg = "luxc splat: failed to create compute pipeline";
        if (error) msg += std::string(": ") + error->localizedDescription()->utf8String();
        throw std::runtime_error(msg);
    }

    transpiler_.transpileInto(vertShader_, shaderBase_ + ".vert.spv", SpvExecModel::Vertex);
    transpiler_.transpileInto(fragShader_, shaderBase_ + ".frag.spv", SpvExecModel::Fragment);

    auto* pipeDesc = MTL::RenderPipelineDescriptor::alloc()->init();
    pipeDesc->setVertexFunction(vertShader_.function);
    pipeDesc->setFragmentFunction(fragShader_.function);

    auto* colorAtt = pipeDesc->colorAttachments()->object(0);
    colorAtt->setPixelFormat(getColorFormat());
    colorAtt->setBlendingEnabled(true);
    colorAtt->setSourceRGBBlendFactor(MTL::BlendFactorOne);
    colorAtt->setDestinationRGBBlendFactor(MTL::BlendFactorOneMinusSourceAlpha);
    colorAtt->setRgbBlendOperation(MTL::BlendOperationAdd);
    colorAtt->setSourceAlphaBlendFactor(MTL::BlendFactorOne);
    colorAtt->setDestinationAlphaBlendFactor(MTL::BlendFactorOneMinusSourceAlpha);
    colorAtt->setAlphaBlendOperation(MTL::BlendOperationAdd);

    // Unlike MetalSplatRenderer's hand-written splat_fragment (which blends
    // motion/depth manually via framebuffer fetch because 32-bit float
    // attachments were found not to support hardware blending on earlier
    // Apple GPUs), this backend runs the luxc-compiled fragment shader
    // unmodified -- it was authored assuming Vulkan's fixed-function
    // ONE/ONE_MINUS_SRC_ALPHA hardware blend (splat_expander.py's
    // _build_fragment_stage has no manual accumulation logic at all), so
    // hardware blending must actually be enabled here for these attachments
    // to composite correctly across overlapping splats. Empirically, Apple
    // Silicon (M4 Max) DOES support blending on RGBA32Float/RG32Float
    // render targets (verified by this backend's own MV/depth parity
    // numbers) -- MetalSplatRenderer's comment may reflect older hardware
    // or a since-resolved driver limitation.
    // out_aux (packed mv.x/mv.y/depth/alpha, bench/lux_perf_ablation.md
    // task 2) and out_fg (fg/alpha, second attachment) use the SAME
    // hardware premultiplied-alpha blend as out_color -- each attachment's
    // own `.w` independently drives its own blend decay, so each needs
    // this blend state set up on it directly.
    uint32_t nextColorSlot = 1;
    if (hasMotionVectors_ || hasExpectedDepth_) {
        auto* auxAtt = pipeDesc->colorAttachments()->object(nextColorSlot++);
        auxAtt->setPixelFormat(getAuxFormat());
        auxAtt->setBlendingEnabled(true);
        auxAtt->setSourceRGBBlendFactor(MTL::BlendFactorOne);
        auxAtt->setDestinationRGBBlendFactor(MTL::BlendFactorOneMinusSourceAlpha);
        auxAtt->setRgbBlendOperation(MTL::BlendOperationAdd);
        auxAtt->setSourceAlphaBlendFactor(MTL::BlendFactorOne);
        auxAtt->setDestinationAlphaBlendFactor(MTL::BlendFactorOneMinusSourceAlpha);
        auxAtt->setAlphaBlendOperation(MTL::BlendOperationAdd);
    }
    if (hasForegroundCoverage_) {
        auto* fgAtt = pipeDesc->colorAttachments()->object(nextColorSlot++);
        fgAtt->setPixelFormat(getFgFormat());
        fgAtt->setBlendingEnabled(true);
        fgAtt->setSourceRGBBlendFactor(MTL::BlendFactorOne);
        fgAtt->setDestinationRGBBlendFactor(MTL::BlendFactorOneMinusSourceAlpha);
        fgAtt->setRgbBlendOperation(MTL::BlendOperationAdd);
        fgAtt->setSourceAlphaBlendFactor(MTL::BlendFactorOne);
        fgAtt->setDestinationAlphaBlendFactor(MTL::BlendFactorOneMinusSourceAlpha);
        fgAtt->setAlphaBlendOperation(MTL::BlendOperationAdd);
    }

    pipeDesc->setDepthAttachmentPixelFormat(MTL::PixelFormatDepth32Float);

    renderPipeline_ = ctx.device->newRenderPipelineState(pipeDesc, &error);
    pipeDesc->release();
    if (!renderPipeline_) {
        std::string msg = "luxc splat: failed to create render pipeline";
        if (error) msg += std::string(": ") + error->localizedDescription()->utf8String();
        throw std::runtime_error(msg);
    }

    auto* dsDesc = MTL::DepthStencilDescriptor::alloc()->init();
    dsDesc->setDepthCompareFunction(MTL::CompareFunctionLessEqual);
    dsDesc->setDepthWriteEnabled(false);
    depthStencilState_ = ctx.device->newDepthStencilState(dsDesc);
    dsDesc->release();
}

void MetalSplatLuxcRenderer::createSortPipelines(MetalContext& ctx) {
    NS::Error* error = nullptr;
    transpiler_.transpileInto(sortHistogramShader_, "shaders/radix_sort/histogram.comp.spv", SpvExecModel::GLCompute);
    transpiler_.transpileInto(sortPrefixSumShader_, "shaders/radix_sort/prefix_sum.comp.spv", SpvExecModel::GLCompute);
    transpiler_.transpileInto(sortScatterShader_, "shaders/radix_sort/scatter.comp.spv", SpvExecModel::GLCompute);

    sortHistogramPipeline_ = ctx.device->newComputePipelineState(sortHistogramShader_.function, &error);
    if (!sortHistogramPipeline_) throw std::runtime_error("luxc splat: failed to create sort-histogram pipeline");
    sortPrefixSumPipeline_ = ctx.device->newComputePipelineState(sortPrefixSumShader_.function, &error);
    if (!sortPrefixSumPipeline_) throw std::runtime_error("luxc splat: failed to create sort-prefix-sum pipeline");
    sortScatterPipeline_ = ctx.device->newComputePipelineState(sortScatterShader_.function, &error);
    if (!sortScatterPipeline_) throw std::runtime_error("luxc splat: failed to create sort-scatter pipeline");

    transpiler_.transpileInto(sortReduceRangeShader_, "shaders/radix_sort/reduce_range.comp.spv", SpvExecModel::GLCompute);
    transpiler_.transpileInto(sortQuantizeShader_, "shaders/radix_sort/quantize.comp.spv", SpvExecModel::GLCompute);
    sortReduceRangePipeline_ = ctx.device->newComputePipelineState(sortReduceRangeShader_.function, &error);
    if (!sortReduceRangePipeline_) throw std::runtime_error("luxc splat: failed to create sort-reduce-range pipeline");
    sortQuantizePipeline_ = ctx.device->newComputePipelineState(sortQuantizeShader_.function, &error);
    if (!sortQuantizePipeline_) throw std::runtime_error("luxc splat: failed to create sort-quantize pipeline");
}

// --------------------------------------------------------------------------
// Buffer creation and data upload (identical layout to MetalSplatRenderer)
// --------------------------------------------------------------------------

void MetalSplatLuxcRenderer::createBuffers(MetalContext& ctx, const GaussianSplatData& data) {
    numSplats_ = data.num_splats;
    shDegree_ = data.sh_degree;
    if (numSplats_ == 0) return;

    hostPositions_ = data.positions;
    posBuffer_ = ctx.newBuffer(hostPositions_.data(), hostPositions_.size() * sizeof(float),
                                MTL::ResourceStorageModeShared);
    posBuffer_->setLabel(NS::String::string("SplatPositionsLuxc", NS::UTF8StringEncoding));

    rotBuffer_ = ctx.newBuffer(data.rotations.data(), numSplats_ * 4 * sizeof(float),
                                MTL::ResourceStorageModeShared);
    rotBuffer_->setLabel(NS::String::string("SplatRotationsLuxc", NS::UTF8StringEncoding));

    std::vector<float> scale4(numSplats_ * 4);
    for (uint32_t i = 0; i < numSplats_; ++i) {
        scale4[i * 4 + 0] = data.scales[i * 3 + 0];
        scale4[i * 4 + 1] = data.scales[i * 3 + 1];
        scale4[i * 4 + 2] = data.scales[i * 3 + 2];
        scale4[i * 4 + 3] = 0.0f;
    }
    scaleBuffer_ = ctx.newBuffer(scale4.data(), scale4.size() * sizeof(float),
                                  MTL::ResourceStorageModeShared);
    scaleBuffer_->setLabel(NS::String::string("SplatScalesLuxc", NS::UTF8StringEncoding));

    opacityBuffer_ = ctx.newBuffer(data.opacities.data(), numSplats_ * sizeof(float),
                                    MTL::ResourceStorageModeShared);
    opacityBuffer_->setLabel(NS::String::string("SplatOpacitiesLuxc", NS::UTF8StringEncoding));

    if (!data.sh_coefficients.empty() && !data.sh_coefficients[0].empty()) {
        const auto& coeffs = data.sh_coefficients[0];
        uint32_t srcFloatsPerSplat = static_cast<uint32_t>(coeffs.size() / numSplats_);
        if (srcFloatsPerSplat == 3) {
            std::vector<float> padded(numSplats_ * 4, 0.0f);
            for (uint32_t i = 0; i < numSplats_; ++i) {
                padded[i * 4 + 0] = coeffs[i * 3 + 0];
                padded[i * 4 + 1] = coeffs[i * 3 + 1];
                padded[i * 4 + 2] = coeffs[i * 3 + 2];
            }
            shBuffer_ = ctx.newBuffer(padded.data(), padded.size() * sizeof(float), MTL::ResourceStorageModeShared);
        } else {
            shBuffer_ = ctx.newBuffer(coeffs.data(), coeffs.size() * sizeof(float), MTL::ResourceStorageModeShared);
        }
    } else {
        std::vector<float> dummy(numSplats_ * 4, 0.0f);
        shBuffer_ = ctx.newBuffer(dummy.data(), dummy.size() * sizeof(float), MTL::ResourceStorageModeShared);
    }
    shBuffer_->setLabel(NS::String::string("SplatSH0Luxc", NS::UTF8StringEncoding));

    // Quad index buffer (perf; see metal_splat_luxc_renderer.h's
    // quadIndexBuffer_ comment): constant content, independent of
    // numSplats_ -- created once here regardless of scene data.
    {
        static const uint16_t kQuadIndices[6] = {0, 1, 2, 2, 1, 3};
        quadIndexBuffer_ = ctx.newBuffer(kQuadIndices, sizeof(kQuadIndices), MTL::ResourceStorageModeShared);
        quadIndexBuffer_->setLabel(NS::String::string("SplatQuadIndicesLuxc", NS::UTF8StringEncoding));
    }

    size_t vec4Size = numSplats_ * 4 * sizeof(float);
    projCenterBuffer_ = ctx.newBuffer(vec4Size, MTL::ResourceStorageModeShared);
    projAxesBuffer_ = ctx.newBuffer(vec4Size, MTL::ResourceStorageModeShared);
    projExtentBuffer_ = ctx.newBuffer(vec4Size, MTL::ResourceStorageModeShared);
    projColorBuffer_ = ctx.newBuffer(vec4Size, MTL::ResourceStorageModeShared);
    visibleCountBuffer_ = ctx.newBuffer(std::max<size_t>(4, sizeof(uint32_t)), MTL::ResourceStorageModeShared);

    if (hasMotionVectors_) {
        projMvBuffer_ = ctx.newBuffer(numSplats_ * 2 * sizeof(float), MTL::ResourceStorageModeShared);
        prevPosBuffer_ = ctx.newBuffer(hostPositions_.data(), hostPositions_.size() * sizeof(float),
                                        MTL::ResourceStorageModeShared);
        prevCameraBuffer_ = ctx.newBuffer(2 * sizeof(glm::mat4), MTL::ResourceStorageModeShared);
    } else {
        // Dummy 1-element buffers -- the compute stage only declares these
        // storage buffers when motion_vectors/expected_depth are enabled
        // (splat_expander.py), so they're never actually bound in that case,
        // but keep pointers non-null for uniform code below.
        projMvBuffer_ = ctx.newBuffer(8, MTL::ResourceStorageModeShared);
        prevPosBuffer_ = ctx.newBuffer(16, MTL::ResourceStorageModeShared);
        prevCameraBuffer_ = ctx.newBuffer(16, MTL::ResourceStorageModeShared);
    }
    if (hasExpectedDepth_) {
        projDepthBuffer_ = ctx.newBuffer(numSplats_ * sizeof(float), MTL::ResourceStorageModeShared);
    } else {
        projDepthBuffer_ = ctx.newBuffer(4, MTL::ResourceStorageModeShared);
    }
    if (hasForegroundCoverage_) {
        // gltf_loader.cpp already resolved the _FOREGROUND-attribute /
        // morph-delta-fallback / all-zero precedence into data.foreground.
        if (data.foreground.size() == numSplats_) {
            foregroundBuffer_ = ctx.newBuffer(data.foreground.data(), numSplats_ * sizeof(float),
                                               MTL::ResourceStorageModeShared);
        } else {
            std::vector<float> zeroFg(numSplats_, 0.0f);
            foregroundBuffer_ = ctx.newBuffer(zeroFg.data(), numSplats_ * sizeof(float),
                                               MTL::ResourceStorageModeShared);
        }
        projForegroundBuffer_ = ctx.newBuffer(numSplats_ * sizeof(float), MTL::ResourceStorageModeShared);
    } else {
        foregroundBuffer_ = ctx.newBuffer(4, MTL::ResourceStorageModeShared);
        projForegroundBuffer_ = ctx.newBuffer(4, MTL::ResourceStorageModeShared);
    }

    // --- Sort buffers (GPU radix sort, ping-pong A/B) ---
    sortKeysBuffer_ = ctx.newBuffer(numSplats_ * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    sortKeysBBuffer_ = ctx.newBuffer(numSplats_ * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    sortedIndicesBuffer_ = ctx.newBuffer(numSplats_ * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    sortValsBBuffer_ = ctx.newBuffer(numSplats_ * sizeof(uint32_t), MTL::ResourceStorageModeShared);

    static const uint32_t SORT_TILE_SIZE = 3840;
    static const uint32_t PREFIX_SUM_BLOCK_SIZE = 2048;
    sortNumWg_ = (numSplats_ + SORT_TILE_SIZE - 1) / SORT_TILE_SIZE;
    uint32_t histogramSize = std::max(256u * sortNumWg_ * 4u, 4u);
    histogramBuffer_ = ctx.newBuffer(histogramSize, MTL::ResourceStorageModeShared);
    uint32_t totalHistEntries = 256 * sortNumWg_;
    uint32_t numPartitions = (totalHistEntries + PREFIX_SUM_BLOCK_SIZE - 1) / PREFIX_SUM_BLOCK_SIZE;
    uint32_t partitionSumsSize = std::max(numPartitions * 4u, 4u);
    partitionSumsBuffer_ = ctx.newBuffer(partitionSumsSize, MTL::ResourceStorageModeShared);

    // 16-bit key quantization range (2 uints: min, max).
    keyRangeBuffer_ = ctx.newBuffer(2 * sizeof(uint32_t), MTL::ResourceStorageModeShared);

    std::cout << "[metal-luxc] GPU radix sort: " << numSplats_ << " splats, "
              << sortNumWg_ << " workgroups, " << totalHistEntries << " histogram entries, "
              << numPartitions << " partitions" << std::endl;

    // --- Dynamic splats: cache base attributes + precompute segments ---
    dynamics_ = data.dynamics;
    if (dynamics_.has_motion) {
        basePositions_ = hostPositions_;
        baseRotations_.assign(data.rotations.begin(), data.rotations.end());
        baseSH0_.assign(numSplats_ * 4, 0.0f);
        if (!dynamics_.targets.empty()) {
            auto* shPtr = static_cast<const float*>(shBuffer_->contents());
            std::memcpy(baseSH0_.data(), shPtr, numSplats_ * 4 * sizeof(float));
        }

        morphSegments_ = buildSplatMorphSegments(dynamics_);

        std::vector<uint32_t> allIdx;
        for (auto& t : dynamics_.targets) allIdx.insert(allIdx.end(), t.indices.begin(), t.indices.end());
        std::sort(allIdx.begin(), allIdx.end());
        allIdx.erase(std::unique(allIdx.begin(), allIdx.end()), allIdx.end());
        everMovingIndices_ = std::move(allIdx);

        std::cout << "[metal-luxc] Dynamic splats: " << morphSegments_.size() << " segments, "
                  << everMovingIndices_.size() << " gaussians ever move" << std::endl;

        createMorphPipeline(ctx);
        createMorphBuffers(ctx);
    }
}

// --------------------------------------------------------------------------
// Dynamic splats: GPU morph-apply (transpiled <shaderBase>.morph.comp.spv)
// --------------------------------------------------------------------------

void MetalSplatLuxcRenderer::createMorphPipeline(MetalContext& ctx) {
    NS::Error* error = nullptr;
    transpiler_.transpileInto(morphShader_, shaderBase_ + ".morph.comp.spv", SpvExecModel::GLCompute);
    morphPipeline_ = ctx.device->newComputePipelineState(morphShader_.function, &error);
    if (!morphPipeline_) {
        std::string msg = "luxc splat: failed to create morph compute pipeline";
        if (error) msg += std::string(": ") + error->localizedDescription()->utf8String();
        throw std::runtime_error(msg);
    }
}

void MetalSplatLuxcRenderer::createMorphBuffers(MetalContext& ctx) {
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

    auto ensureNonEmpty = [](std::vector<float>& v, size_t floatsPerEntry) {
        if (v.empty()) v.resize(floatsPerEntry, 0.0f);
    };
    ensureNonEmpty(catPosLo4, 4); ensureNonEmpty(catPosHi4, 4);
    ensureNonEmpty(catRotLo, 4);  ensureNonEmpty(catRotHi, 4);
    ensureNonEmpty(catSh0Lo4, 4); ensureNonEmpty(catSh0Hi4, 4);
    if (catIndex.empty()) catIndex.push_back(0);

    baseGpuPosBuffer_ = ctx.newBuffer(basePositions_.data(), basePositions_.size() * sizeof(float),
                                       MTL::ResourceStorageModeShared);
    baseGpuRotBuffer_ = ctx.newBuffer(baseRotations_.data(), baseRotations_.size() * sizeof(float),
                                       MTL::ResourceStorageModeShared);
    baseGpuSh0Buffer_ = ctx.newBuffer(baseSH0_.data(), baseSH0_.size() * sizeof(float),
                                       MTL::ResourceStorageModeShared);

    morphIndexBuffer_ = ctx.newBuffer(catIndex.data(), catIndex.size() * sizeof(uint32_t),
                                       MTL::ResourceStorageModeShared);
    morphPosLoBuffer_ = ctx.newBuffer(catPosLo4.data(), catPosLo4.size() * sizeof(float), MTL::ResourceStorageModeShared);
    morphRotLoBuffer_ = ctx.newBuffer(catRotLo.data(), catRotLo.size() * sizeof(float), MTL::ResourceStorageModeShared);
    morphSh0LoBuffer_ = ctx.newBuffer(catSh0Lo4.data(), catSh0Lo4.size() * sizeof(float), MTL::ResourceStorageModeShared);
    morphPosHiBuffer_ = ctx.newBuffer(catPosHi4.data(), catPosHi4.size() * sizeof(float), MTL::ResourceStorageModeShared);
    morphRotHiBuffer_ = ctx.newBuffer(catRotHi.data(), catRotHi.size() * sizeof(float), MTL::ResourceStorageModeShared);
    morphSh0HiBuffer_ = ctx.newBuffer(catSh0Hi4.data(), catSh0Hi4.size() * sizeof(float), MTL::ResourceStorageModeShared);
}

float MetalSplatLuxcRenderer::animationDuration() const {
    if (!dynamics_.has_motion || dynamics_.keyframes.empty()) return 0.0f;
    return dynamics_.keyframes.back().time;
}

float MetalSplatLuxcRenderer::frameToTime(int frame) const {
    return splatFrameToTime(dynamics_, frame);
}

void MetalSplatLuxcRenderer::stepKeyframe(int direction) {
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
    setMorphTime(kf[static_cast<size_t>(stepped)].time);
}

void MetalSplatLuxcRenderer::setMorphTime(float seconds, MTL::CommandBuffer* sharedCmdBuf) {
    if (!dynamics_.has_motion) return;
    currentMorphTime_ = seconds;
    SplatMorphState state = evaluateSplatMorphState(dynamics_, seconds);

    struct MorphPush {
        uint32_t segmentOffset;
        uint32_t segmentCount;
        float weightLow;
        float weightHigh;
    };

    MTL::CommandBuffer* cmdBuf = sharedCmdBuf ? sharedCmdBuf : ctx_->beginCommandBuffer();
    auto* enc = cmdBuf->computeCommandEncoder();
    enc->setComputePipelineState(morphPipeline_);
    trySetBuffer(enc, morphShader_, baseGpuPosBuffer_, 0);
    trySetBuffer(enc, morphShader_, baseGpuRotBuffer_, 1);
    trySetBuffer(enc, morphShader_, baseGpuSh0Buffer_, 2);
    trySetBuffer(enc, morphShader_, morphIndexBuffer_, 3);
    trySetBuffer(enc, morphShader_, morphPosLoBuffer_, 4);
    trySetBuffer(enc, morphShader_, morphRotLoBuffer_, 5);
    trySetBuffer(enc, morphShader_, morphSh0LoBuffer_, 6);
    trySetBuffer(enc, morphShader_, morphPosHiBuffer_, 7);
    trySetBuffer(enc, morphShader_, morphRotHiBuffer_, 8);
    trySetBuffer(enc, morphShader_, morphSh0HiBuffer_, 9);
    trySetBuffer(enc, morphShader_, posBuffer_, 10);
    trySetBuffer(enc, morphShader_, rotBuffer_, 11);
    trySetBuffer(enc, morphShader_, shBuffer_, 12);

    uint32_t threadGroupSize = static_cast<uint32_t>(morphPipeline_->maxTotalThreadsPerThreadgroup());
    if (threadGroupSize > 256) threadGroupSize = 256;

    // Morph-reset scope fix (bench/lux_perf_ablation.md "Root cause #1",
    // ported from splat_renderer.cpp's dispatchMorph() -- see that
    // function's header comment for the full correctness argument). The
    // "apply" dispatch below is non-incremental (splat_pos[idx] =
    // splat_base_pos[idx] + weight_lo*delta_lo + weight_hi*delta_hi,
    // always read from the immutable base buffers), so any index the
    // CURRENT apply touches is fully overwritten regardless of what an
    // earlier call left there. A "reset" is only needed for indices left
    // stale by the *immediately preceding* setMorphTime() call's segment
    // (by induction, residue never accumulates past one step back -- every
    // prior call already cleaned up its own predecessor's residue the same
    // way) -- so only that one segment (segmentCounts_[lastAppliedSegment_],
    // tracked below) needs resetting, not the whole concatenated
    // morphTotalEntries_ array (was ~38x more reset work than apply work
    // on the juggle scene, ~13ms of an ~18ms Mali preprocess window; same
    // shader/buffer layout here, so the same win applies to Apple Silicon).
    // -1 (no prior segment, e.g. the very first call) or an unchanged
    // segment both skip the reset dispatch entirely.
    bool hasActive = (state.weightLow != 0.0f || state.weightHigh != 0.0f) &&
                      state.highTargetIndex >= 0 &&
                      static_cast<size_t>(state.highTargetIndex) < segmentCounts_.size() &&
                      segmentCounts_[state.highTargetIndex] > 0;
    int curSeg = hasActive ? state.highTargetIndex : -1;

    if (lastAppliedSegment_ >= 0 && lastAppliedSegment_ != curSeg) {
        uint32_t seg = static_cast<uint32_t>(lastAppliedSegment_);
        if (segmentCounts_[seg] > 0) {
            MorphPush resetPush = {segmentOffsets_[seg], segmentCounts_[seg], 0.0f, 0.0f};
            enc->setBytes(&resetPush, sizeof(resetPush), morphShader_.pushConstantBufferIndex);
            uint32_t groups = (segmentCounts_[seg] + threadGroupSize - 1) / threadGroupSize;
            enc->dispatchThreadgroups(MTL::Size(groups, 1, 1), MTL::Size(threadGroupSize, 1, 1));
        }
    }

    if (hasActive) {
        uint32_t seg = static_cast<uint32_t>(curSeg);
        MorphPush applyPush = {segmentOffsets_[seg], segmentCounts_[seg], state.weightLow, state.weightHigh};
        enc->setBytes(&applyPush, sizeof(applyPush), morphShader_.pushConstantBufferIndex);
        uint32_t groups = (segmentCounts_[seg] + threadGroupSize - 1) / threadGroupSize;
        enc->dispatchThreadgroups(MTL::Size(groups, 1, 1), MTL::Size(threadGroupSize, 1, 1));
    }

    lastAppliedSegment_ = curSeg;

    enc->endEncoding();

    if (!sharedCmdBuf) {
        // Self-contained mode (default): commit + block here, then refresh
        // the CPU-side mirror -- needed by callers like stepKeyframe() (UI
        // feedback prints currentMorphTimeSeconds() but reads no buffer
        // contents), seedPreviousMorphTime() (memcpy's posBuffer_'s result
        // into prevPosBuffer_ right after this call), and the headless
        // one-shot LUX_DEBUG_SPLAT_DUMP/--dump-splat-buffers paths.
        cmdBuf->commit();
        cmdBuf->waitUntilCompleted();
        std::memcpy(hostPositions_.data(), posBuffer_->contents(), hostPositions_.size() * sizeof(float));
    }
    // Shared-command-buffer mode: no commit/wait/readback here -- the caller
    // owns the buffer's lifecycle (commit + waitUntilCompleted) and,
    // because hostPositions_ is not refreshed in this mode, must not rely
    // on it or on posBuffer_->contents() being valid until its own wait
    // completes. This is the fold-into-the-frame's-own-command-buffer path
    // (see this method's declaration comment in the header).
}

// --------------------------------------------------------------------------
// Camera
// --------------------------------------------------------------------------

void MetalSplatLuxcRenderer::updateCamera(glm::vec3 eye, glm::vec3 target, glm::vec3 up,
                                           float fovY, float aspect, float nearPlane, float farPlane) {
    camPos_ = eye;
    viewMatrix_ = glm::lookAt(eye, target, up);
    projMatrixUnjittered_ = glm::perspective(fovY, aspect, nearPlane, farPlane);
    projMatrix_ = applyLuxcSplatJitter(projMatrixUnjittered_, jitterX_, jitterY_, width_, height_);
    focalY_ = 0.5f * static_cast<float>(height_) / tanf(fovY * 0.5f);
    focalX_ = focalY_;
}

void MetalSplatLuxcRenderer::updateCameraExplicit(glm::vec3 eye, glm::mat4 viewMatrix, glm::mat4 projMatrix,
                                                   float focalX, float focalY) {
    camPos_ = eye;
    viewMatrix_ = viewMatrix;
    projMatrixUnjittered_ = projMatrix;
    projMatrix_ = applyLuxcSplatJitter(projMatrix, jitterX_, jitterY_, width_, height_);
    focalX_ = focalX;
    focalY_ = focalY;
}

void MetalSplatLuxcRenderer::setJitter(float jitterXPixels, float jitterYPixels) {
    jitterX_ = jitterXPixels;
    jitterY_ = jitterYPixels;
    projMatrix_ = applyLuxcSplatJitter(projMatrixUnjittered_, jitterX_, jitterY_, width_, height_);
}

// --------------------------------------------------------------------------
// init
// --------------------------------------------------------------------------

void MetalSplatLuxcRenderer::init(MetalContext& ctx, const GaussianSplatData& data,
                                   const std::string& shaderBase, uint32_t width, uint32_t height) {
    ctx_ = &ctx;
    shaderBase_ = shaderBase;
    width_ = width;
    height_ = height;
    transpiler_.init(ctx.device);

    // hasMotionVectors_/hasExpectedDepth_ are set inside createPipelines()
    // (reads the compute stage's reflection JSON) -- createBuffers() and
    // createRenderTargets() both depend on them, so pipelines must be
    // created first here (unlike MetalSplatRenderer, where both flags are
    // hardcoded `true` and buffer/render-target creation order doesn't
    // matter).
    createPipelines(ctx);
    createSortPipelines(ctx);
    createBuffers(ctx, data);
    createRenderTargets(ctx);

    std::cout << "[metal-luxc] MetalSplatLuxcRenderer initialized: " << numSplats_ << " splats, "
              << width << "x" << height
              << " (motion_vectors=" << hasMotionVectors_ << " expected_depth=" << hasExpectedDepth_
              << " foreground_coverage=" << hasForegroundCoverage_ << ")"
              << std::endl;
}

// --------------------------------------------------------------------------
// render()
// --------------------------------------------------------------------------

// encodeFrame()/render() split (playground_cpp/src/metal_live_reconstruct.* --
// see encodeFrame()'s header comment): everything below through the render
// pass's endEncoding()/rpDesc->release() is the exact body render() used to
// run inline, now factored into encodeFrame() so a continuous per-frame
// caller can fuse it into its own command buffer. render() itself (below
// encodeFrame()) is just begin+encodeFrame+commit+wait+GPU-timing-readback.
void MetalSplatLuxcRenderer::encodeFrame(MetalContext& ctx, MTL::CommandBuffer* cmdBuf) {
    if (numSplats_ == 0) return;

    if (hasMotionVectors_ && firstMvFrame_) {
        prevViewMatrix_ = viewMatrix_;
        prevProjMatrixUnjittered_ = projMatrixUnjittered_;
        std::memcpy(prevPosBuffer_->contents(), hostPositions_.data(), hostPositions_.size() * sizeof(float));
    }

    // --- Preprocess compute: projection, covariance, SH, sort-key gen ---
    // proj_matrix_unjittered/prev_view_proj_unjittered do NOT live in this
    // push-constant struct (unlike the pre-fix version) -- they're written
    // into prevCameraBuffer_ below instead, matching splat_renderer.cpp's
    // Vulkan path exactly (see that file's ComputePush comment / the
    // guardrail check in its createPipelines for why: a 304-byte compute
    // push-constant block was fine on MoltenVK's reported 4096-byte budget,
    // never an issue here, but silently corrupted every motion vector on
    // Android's Mali-G715, which only supports 256).
    struct ComputePush {
        float view[16];
        float proj[16];
        float camPos[3];
        float _pad0;
        float screenW, screenH;
        uint32_t numSplats;
        float focalX;
        float focalY;
        int32_t shDegree;
        float _pad1[2];
    } push = {};
    std::memcpy(push.view, &viewMatrix_[0][0], 64);
    std::memcpy(push.proj, &projMatrix_[0][0], 64);
    push.camPos[0] = camPos_.x; push.camPos[1] = camPos_.y; push.camPos[2] = camPos_.z;
    push.screenW = static_cast<float>(width_);
    push.screenH = static_cast<float>(height_);
    push.numSplats = numSplats_;
    push.focalX = focalX_;
    push.focalY = focalY_;
    push.shDegree = static_cast<int32_t>(shDegree_);
    if (hasMotionVectors_) {
        // prev_camera_mats[0]=proj_matrix_unjittered, [1]=prev_view_proj_unjittered.
        // Direct memcpy into the shared-storage buffer's contents (like
        // prevPosBuffer_'s firstMvFrame_ seed above) -- no explicit barrier
        // needed, Metal auto-tracks buffer hazards within a command buffer
        // in submission order, and this write happens on the CPU well
        // before the compute encoder below is even created.
        glm::mat4 prevCameraMats[2];
        prevCameraMats[0] = projMatrixUnjittered_;
        prevCameraMats[1] = prevProjMatrixUnjittered_ * prevViewMatrix_;
        std::memcpy(prevCameraBuffer_->contents(), prevCameraMats, sizeof(prevCameraMats));
    }

    auto t0 = std::chrono::steady_clock::now();
    // Single command buffer for the *entire* frame (preprocess + all radix-sort
    // range-reduction/quantization/passes + render) -- `cmdBuf` is the caller's own (render() below passes a
    // freshly-begun one it will commit+wait on itself right after this call
    // returns; metal_live_reconstruct.* passes its own per-frame buffer and
    // keeps encoding into it afterwards). Previously each of the ~22 individual
    // dispatches (1 preprocess + 4 passes x 5 sub-dispatches + 1 render) used
    // its own beginCommandBuffer()+commit()+waitUntilCompleted(), which is
    // resolution-independent CPU<->GPU synchronization overhead (Metal already
    // auto-tracks buffer/texture hazards *within* one command buffer across
    // encoder boundaries in submission order, so no manual fences are needed to
    // merge these safely). The inner `{ }` scopes below are kept only to let
    // each stage redeclare its own `enc` without a name clash.
    {
        auto* enc = cmdBuf->computeCommandEncoder();
        enc->setComputePipelineState(computePipeline_);
        trySetBuffer(enc, compShader_, posBuffer_, 0);
        trySetBuffer(enc, compShader_, rotBuffer_, 1);
        trySetBuffer(enc, compShader_, scaleBuffer_, 2);
        trySetBuffer(enc, compShader_, opacityBuffer_, 3);
        trySetBuffer(enc, compShader_, shBuffer_, 4);
        trySetBuffer(enc, compShader_, projCenterBuffer_, 5);
        trySetBuffer(enc, compShader_, projAxesBuffer_, 6);
        trySetBuffer(enc, compShader_, projExtentBuffer_, 7);
        trySetBuffer(enc, compShader_, projColorBuffer_, 8);
        trySetBuffer(enc, compShader_, sortKeysBuffer_, 9);
        trySetBuffer(enc, compShader_, sortedIndicesBuffer_, 10);
        trySetBuffer(enc, compShader_, visibleCountBuffer_, 11);
        uint32_t nextBinding = 12;
        if (hasMotionVectors_) {
            trySetBuffer(enc, compShader_, prevPosBuffer_, nextBinding++);
            trySetBuffer(enc, compShader_, projMvBuffer_, nextBinding++);
            trySetBuffer(enc, compShader_, prevCameraBuffer_, nextBinding++);
        }
        if (hasExpectedDepth_) {
            trySetBuffer(enc, compShader_, projDepthBuffer_, nextBinding++);
        }
        if (hasForegroundCoverage_) {
            trySetBuffer(enc, compShader_, foregroundBuffer_, nextBinding++);
            trySetBuffer(enc, compShader_, projForegroundBuffer_, nextBinding++);
        }
        enc->setBytes(&push, sizeof(push), compShader_.pushConstantBufferIndex);
        uint32_t groups = (numSplats_ + 255) / 256;
        enc->dispatchThreadgroups(MTL::Size(groups, 1, 1), MTL::Size(256, 1, 1));
        enc->endEncoding();
    }
    lastPreprocessMs_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    if (hasMotionVectors_) {
        // GPU-side copy (was a CPU memcpy before the single-command-buffer
        // refactor above): the preprocess compute encoder just encoded
        // above reads the OLD prevPosBuffer_ to compute this frame's MV,
        // but hasn't actually *run* yet (commit() is deferred to the very
        // end of this function now) -- a CPU memcpy here would race ahead
        // and clobber prevPosBuffer_ with this frame's positions before the
        // GPU ever reads the old ones. A blit encoded into the same shared
        // cmdBuf, right after the preprocess encoder, keeps the ordering
        // correct (Metal executes a command buffer's encoders in
        // submission order) with no CPU/GPU sync needed.
        auto* posCopyBlit = cmdBuf->blitCommandEncoder();
        posCopyBlit->copyFromBuffer(posBuffer_, 0, prevPosBuffer_, 0, hostPositions_.size() * sizeof(float));
        posCopyBlit->endEncoding();
        prevViewMatrix_ = viewMatrix_;
        prevProjMatrixUnjittered_ = projMatrixUnjittered_;
        firstMvFrame_ = false;
    }

    // --- Sort scheduling decision (see setSortSchedule()'s header comment;
    // ported verbatim from splat_renderer.cpp's identical Vulkan logic).
    // Default (sortEveryNFrames_==1, sortViewThresholdDeg_==0) always
    // re-sorts (needsSort stays true unconditionally below), so behavior is
    // unchanged unless a caller opts in.
    bool needsSort = true;
    if (sortEveryNFrames_ > 1 || sortViewThresholdDeg_ > 0.0f || sortTranslateThreshold_ > 0.0f) {
        glm::vec3 viewDir = glm::normalize(
            -glm::vec3(viewMatrix_[0][2], viewMatrix_[1][2], viewMatrix_[2][2]));
        bool viewChanged = true;
        if (hasLastSortedView_ && sortViewThresholdDeg_ > 0.0f) {
            float cosAngle = glm::clamp(glm::dot(viewDir, lastSortedViewDir_), -1.0f, 1.0f);
            float angleDeg = glm::degrees(std::acos(cosAngle));
            viewChanged = angleDeg >= sortViewThresholdDeg_;
        }
        bool translated = true;
        if (hasLastSortedView_ && sortTranslateThreshold_ > 0.0f) {
            translated = glm::length(camPos_ - lastSortedCamPos_) >= sortTranslateThreshold_;
        }
        // Any change at all (not thresholded) -- see setSortSchedule()'s
        // header comment: this is what makes the schedule safe for
        // dynamic/morphing scenes.
        bool morphChanged = hasLastSortedView_ && (currentMorphTime_ != lastSortedMorphTime_);
        bool budgetElapsed = framesSinceSort_ >= sortEveryNFrames_;
        needsSort = !hasLastSortedView_ || budgetElapsed ||
            (sortViewThresholdDeg_ > 0.0f && viewChanged) ||
            (sortTranslateThreshold_ > 0.0f && translated) ||
            morphChanged;
        if (needsSort) {
            framesSinceSort_ = 0;
            lastSortedViewDir_ = viewDir;
            lastSortedCamPos_ = camPos_;
            lastSortedMorphTime_ = currentMorphTime_;
            hasLastSortedView_ = true;
        } else {
            framesSinceSort_++;
        }
    }

    // --- GPU radix sort (range reduction + quantization, then 2 passes,
    // 8 bits/pass = 16-bit keys -- perf; bench/lux_perf_ablation.md's "Key
    // width / passes"; previously 4 passes over the full 32-bit key) ---
    // Skipped on frames the schedule above decides don't need a fresh
    // order -- sortedIndicesBuffer_/sortKeysBuffer_ simply keep whatever
    // the last real sort left in them (GPU-only buffers, never implicitly
    // cleared; always land back in buffer A after an even pass count, and
    // this always runs 0 or 2 passes, never a partial/odd count, so that
    // invariant holds across skipped frames too). Splat VISIBILITY is
    // unaffected either way (a per-splat decision made in
    // preprocess/fragment, not by sort position) -- only back-to-front
    // BLEND ORDER can be briefly stale.
    auto t1 = std::chrono::steady_clock::now();
    if (needsSort) {
        static const uint32_t PREFIX_SUM_BLOCK_SIZE = 2048;
        uint32_t numElements = numSplats_;
        uint32_t numWg = sortNumWg_;
        uint32_t totalHistogram = 256 * numWg;
        uint32_t numParts = (totalHistogram + PREFIX_SUM_BLOCK_SIZE - 1) / PREFIX_SUM_BLOCK_SIZE;

        struct SortPush { uint32_t numElements; uint32_t bitOffset; };

        // --- Range reduction + quantization (buffer A, before the
        // ping-pong pass loop). No explicit barriers needed between these
        // compute-encoder-per-dispatch blocks -- Metal's default hazard
        // tracking (MTLResourceHazardTrackingModeTracked, the default for
        // these MTL::ResourceStorageModeShared buffers) already serializes
        // dependent encoders within one command buffer, matching how the
        // existing pass loop below relies on it between histogram/prefix_
        // sum/scatter.
        {
            // Clear key_range to (min=UINT_MAX, max=0) -- both sentinel
            // values are byte-uniform, so a single-byte fillBuffer value
            // (0xFF / 0x00) is exact, matching Vulkan's vkCmdFillBuffer
            // 32-bit-word fill of the same two values.
            auto* clearBlit = cmdBuf->blitCommandEncoder();
            clearBlit->fillBuffer(keyRangeBuffer_, NS::Range(0, sizeof(uint32_t)), 0xFF);
            clearBlit->fillBuffer(keyRangeBuffer_, NS::Range(sizeof(uint32_t), sizeof(uint32_t)), 0x00);
            clearBlit->endEncoding();

            {
                auto* enc = cmdBuf->computeCommandEncoder();
                enc->setComputePipelineState(sortReduceRangePipeline_);
                trySetBuffer(enc, sortReduceRangeShader_, sortKeysBuffer_, 0);
                trySetBuffer(enc, sortReduceRangeShader_, keyRangeBuffer_, 1);
                SortPush rp = {numElements, 0};
                enc->setBytes(&rp, sizeof(rp), sortReduceRangeShader_.pushConstantBufferIndex);
                enc->dispatchThreadgroups(MTL::Size(numWg, 1, 1), MTL::Size(256, 1, 1));
                enc->endEncoding();
            }
            {
                auto* enc = cmdBuf->computeCommandEncoder();
                enc->setComputePipelineState(sortQuantizePipeline_);
                trySetBuffer(enc, sortQuantizeShader_, sortKeysBuffer_, 0);
                trySetBuffer(enc, sortQuantizeShader_, keyRangeBuffer_, 1);
                SortPush qp = {numElements, 0};
                enc->setBytes(&qp, sizeof(qp), sortQuantizeShader_.pushConstantBufferIndex);
                enc->dispatchThreadgroups(MTL::Size(numWg, 1, 1), MTL::Size(256, 1, 1));
                enc->endEncoding();
            }
        }

        for (uint32_t pass = 0; pass < 2; ++pass) {
            uint32_t bitOffset = pass * 8;
            uint32_t ping = pass % 2;  // 0 = A->B, 1 = B->A
            MTL::Buffer* keysIn = ping == 0 ? sortKeysBuffer_ : sortKeysBBuffer_;
            MTL::Buffer* keysOut = ping == 0 ? sortKeysBBuffer_ : sortKeysBuffer_;
            MTL::Buffer* valsIn = ping == 0 ? sortedIndicesBuffer_ : sortValsBBuffer_;
            MTL::Buffer* valsOut = ping == 0 ? sortValsBBuffer_ : sortedIndicesBuffer_;

            // Histogram
            {
                auto* enc = cmdBuf->computeCommandEncoder();
                enc->setComputePipelineState(sortHistogramPipeline_);
                trySetBuffer(enc, sortHistogramShader_, keysIn, 0);
                trySetBuffer(enc, sortHistogramShader_, histogramBuffer_, 1);
                SortPush hp = {numElements, bitOffset};
                enc->setBytes(&hp, sizeof(hp), sortHistogramShader_.pushConstantBufferIndex);
                enc->dispatchThreadgroups(MTL::Size(numWg, 1, 1), MTL::Size(256, 1, 1));
                enc->endEncoding();
            }
            // Prefix sum (3 sub-passes)
            {
                auto* enc = cmdBuf->computeCommandEncoder();
                enc->setComputePipelineState(sortPrefixSumPipeline_);
                trySetBuffer(enc, sortPrefixSumShader_, histogramBuffer_, 0);
                trySetBuffer(enc, sortPrefixSumShader_, partitionSumsBuffer_, 1);
                SortPush ps0 = {totalHistogram, 0};
                enc->setBytes(&ps0, sizeof(ps0), sortPrefixSumShader_.pushConstantBufferIndex);
                enc->dispatchThreadgroups(MTL::Size(numParts, 1, 1), MTL::Size(1024, 1, 1));
                enc->endEncoding();
            }
            {
                auto* enc = cmdBuf->computeCommandEncoder();
                enc->setComputePipelineState(sortPrefixSumPipeline_);
                trySetBuffer(enc, sortPrefixSumShader_, histogramBuffer_, 0);
                trySetBuffer(enc, sortPrefixSumShader_, partitionSumsBuffer_, 1);
                SortPush ps1 = {numParts, 1};
                enc->setBytes(&ps1, sizeof(ps1), sortPrefixSumShader_.pushConstantBufferIndex);
                enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(1024, 1, 1));
                enc->endEncoding();
            }
            {
                auto* enc = cmdBuf->computeCommandEncoder();
                enc->setComputePipelineState(sortPrefixSumPipeline_);
                trySetBuffer(enc, sortPrefixSumShader_, histogramBuffer_, 0);
                trySetBuffer(enc, sortPrefixSumShader_, partitionSumsBuffer_, 1);
                SortPush ps2 = {totalHistogram, 2};
                enc->setBytes(&ps2, sizeof(ps2), sortPrefixSumShader_.pushConstantBufferIndex);
                enc->dispatchThreadgroups(MTL::Size(numParts, 1, 1), MTL::Size(1024, 1, 1));
                enc->endEncoding();
            }
            // Scatter
            {
                auto* enc = cmdBuf->computeCommandEncoder();
                enc->setComputePipelineState(sortScatterPipeline_);
                trySetBuffer(enc, sortScatterShader_, keysIn, 0);
                trySetBuffer(enc, sortScatterShader_, keysOut, 1);
                trySetBuffer(enc, sortScatterShader_, valsIn, 2);
                trySetBuffer(enc, sortScatterShader_, valsOut, 3);
                trySetBuffer(enc, sortScatterShader_, histogramBuffer_, 4);
                SortPush sp = {numElements, bitOffset};
                enc->setBytes(&sp, sizeof(sp), sortScatterShader_.pushConstantBufferIndex);
                enc->dispatchThreadgroups(MTL::Size(numWg, 1, 1), MTL::Size(256, 1, 1));
                enc->endEncoding();
            }
        }
        // After 2 (even) passes, sorted result is back in buffer A
        // (sortKeysBuffer_ / sortedIndicesBuffer_), matching Vulkan.
    }
    lastSortMs_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();

    if (std::getenv("LUX_DEBUG_SORT_DUMP")) {
        std::string prefix = std::getenv("LUX_DEBUG_SORT_DUMP");
        std::vector<uint32_t> keys(numSplats_), idx(numSplats_);
        std::memcpy(keys.data(), sortKeysBuffer_->contents(), numSplats_ * sizeof(uint32_t));
        std::memcpy(idx.data(), sortedIndicesBuffer_->contents(), numSplats_ * sizeof(uint32_t));
        uint32_t numOutOfOrder = 0;
        for (uint32_t i = 1; i < numSplats_; ++i) if (keys[i] < keys[i - 1]) numOutOfOrder++;
        std::cout << "[debug-sort] numSplats=" << numSplats_ << " numOutOfOrder=" << numOutOfOrder
                  << " first10keys=";
        for (int i = 0; i < 10 && i < (int)numSplats_; ++i) std::cout << keys[i] << " ";
        std::cout << " last10keys=";
        for (uint32_t i = numSplats_ > 10 ? numSplats_ - 10 : 0; i < numSplats_; ++i) std::cout << keys[i] << " ";
        std::cout << std::endl;
    }

    // --- Render pass ---
    auto t2 = std::chrono::steady_clock::now();
    auto* rpDesc = MTL::RenderPassDescriptor::alloc()->init();
    auto* colorAtt = rpDesc->colorAttachments()->object(0);
    colorAtt->setTexture(colorTarget_);
    colorAtt->setLoadAction(MTL::LoadActionClear);
    colorAtt->setStoreAction(MTL::StoreActionStore);
    // Alpha clears to 0 (not opaque) whenever DLSS outputs are enabled, so
    // color.a is a genuine coverage signal for host-side un-premultiply --
    // same rationale as the Vulkan splat renderer.
    float clearA = (hasMotionVectors_ || hasExpectedDepth_) ? 0.0f : 1.0f;
    colorAtt->setClearColor(MTL::ClearColor(0.0, 0.0, 0.0, clearA));

    uint32_t nextRpSlot = 1;
    if (hasMotionVectors_ || hasExpectedDepth_) {
        auto* auxAtt = rpDesc->colorAttachments()->object(nextRpSlot++);
        auxAtt->setTexture(auxTarget_);
        auxAtt->setLoadAction(MTL::LoadActionClear);
        auxAtt->setStoreAction(MTL::StoreActionStore);
        auxAtt->setClearColor(MTL::ClearColor(0.0, 0.0, 0.0, 0.0));
    }
    if (hasForegroundCoverage_) {
        auto* fgAtt = rpDesc->colorAttachments()->object(nextRpSlot++);
        fgAtt->setTexture(fgTarget_);
        fgAtt->setLoadAction(MTL::LoadActionClear);
        fgAtt->setStoreAction(MTL::StoreActionStore);
        fgAtt->setClearColor(MTL::ClearColor(0.0, 0.0, 0.0, 0.0));
    }

    auto* depthAtt = rpDesc->depthAttachment();
    depthAtt->setTexture(depthTarget_);
    depthAtt->setLoadAction(MTL::LoadActionClear);
    depthAtt->setStoreAction(MTL::StoreActionDontCare);
    depthAtt->setClearDepth(1.0);

    auto* enc = cmdBuf->renderCommandEncoder(rpDesc);
    // Negative-height viewport flip (the standard MoltenVK/Vulkan-on-Metal
    // trick): Vulkan's NDC has +Y pointing down; Metal's native NDC has +Y
    // pointing up. This compiled vertex shader (splat_expander.py) computes
    // gl_Position assuming a Vulkan-convention rasterizer -- confirmed via
    // LUX_DUMP_MSL_DIR: it builds screen pixel coordinates via the plain
    // Vulkan `pixel = (ndc*0.5+0.5)*screen_size` mapping with no shader-side
    // correction -- and this SPIRV-Cross version's MSL backend has no
    // automatic gl_Position Y-flip option (checked: no flip_vert_y or
    // similar in spirv_msl.hpp), so without this the whole render is
    // vertically mirrored relative to Vulkan's output.
    enc->setViewport(MTL::Viewport{0.0, static_cast<double>(height_),
                                    static_cast<double>(width_), -static_cast<double>(height_), 0.0, 1.0});
    enc->setScissorRect(MTL::ScissorRect{0, 0, width_, height_});
    enc->setRenderPipelineState(renderPipeline_);
    enc->setDepthStencilState(depthStencilState_);

    // Fragment stage in this pipeline has no storage-buffer bindings
    // (splat_expander.py's fragment reflection is empty) -- only the
    // vertex stage pulls from buffers.
    trySetVertexBuffer(enc, vertShader_, projCenterBuffer_, 0);
    trySetVertexBuffer(enc, vertShader_, projAxesBuffer_, 1);
    trySetVertexBuffer(enc, vertShader_, projExtentBuffer_, 2);
    trySetVertexBuffer(enc, vertShader_, projColorBuffer_, 3);
    trySetVertexBuffer(enc, vertShader_, sortedIndicesBuffer_, 4);
    uint32_t vNext = 5;
    if (hasMotionVectors_) trySetVertexBuffer(enc, vertShader_, projMvBuffer_, vNext++);
    if (hasExpectedDepth_) trySetVertexBuffer(enc, vertShader_, projDepthBuffer_, vNext++);
    if (hasForegroundCoverage_) trySetVertexBuffer(enc, vertShader_, projForegroundBuffer_, vNext++);

    struct RenderPush { float screenW, screenH; uint32_t visibleCount; float alphaMin; } renderPush = {};
    renderPush.screenW = static_cast<float>(width_);
    renderPush.screenH = static_cast<float>(height_);
    renderPush.visibleCount = numSplats_;
    renderPush.alphaMin = 1.0f / 255.0f;
    if (vertShader_.pushConstantBufferIndex != UINT32_MAX)
        enc->setVertexBytes(&renderPush, sizeof(renderPush), vertShader_.pushConstantBufferIndex);
    if (fragShader_.pushConstantBufferIndex != UINT32_MAX)
        enc->setFragmentBytes(&renderPush, sizeof(renderPush), fragShader_.pushConstantBufferIndex);

    // Indexed instanced draw: 6 indices over 4 unique vertices (quad) x
    // numSplats instances (perf; see metal_splat_luxc_renderer.h's
    // quadIndexBuffer_ comment / splat_expander.py's quad-corner comment)
    // -- 4 vertex-shader invocations/splat instead of the old non-indexed 6.
    enc->drawIndexedPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(6), MTL::IndexTypeUInt16,
                                quadIndexBuffer_, NS::UInteger(0), NS::UInteger(numSplats_));
    enc->endEncoding();
    rpDesc->release();
    lastRenderMs_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t2).count();
}

void MetalSplatLuxcRenderer::render(MetalContext& ctx) {
    if (numSplats_ == 0) return;
    MTL::CommandBuffer* cmdBuf = ctx.beginCommandBuffer();
    encodeFrame(ctx, cmdBuf);
    cmdBuf->commit();
    cmdBuf->waitUntilCompleted();
    // Real GPU busy time for the WHOLE fused command buffer (preprocess +
    // sort + draw) -- directly comparable to MetalSplatter's own
    // commandBuffer.gpuStartTime/gpuEndTime methodology (bench/
    // lux_perf_ablation.md in mobiledlss). Unlike lastPreprocessMs_/
    // lastSortMs_/lastRenderMs_ (CPU wall-clock deltas taken WHILE encoding,
    // before this single commit()/waitUntilCompleted() -- i.e. they mostly
    // measure CPU encode time, not GPU execution time, in this
    // fused-command-buffer design), this is a true GPU timestamp,
    // unaffected by CPU-side encode/dispatch overhead.
    lastGpuTotalMs_ = (cmdBuf->GPUEndTime() - cmdBuf->GPUStartTime()) * 1000.0;

    std::cout << "[metal-luxc] render(): preprocess=" << lastPreprocessMs_ << "ms sort="
              << lastSortMs_ << "ms render=" << lastRenderMs_ << "ms gpuTotal="
              << lastGpuTotalMs_ << "ms" << std::endl;
}

// --------------------------------------------------------------------------
// renderProfiled() -- bench-only stage-split GPU timing (see header
// comment). Deliberate near-duplicate of render() above rather than a
// shared refactor: render() is parity-critical (byte-for-byte pixel output
// validated against Vulkan/gsplat by tests/mobiledlss's verify_lux_vs_
// gsplat.py), so touching its exact command-buffer/encoder sequence for a
// diagnostic feature was judged too risky. This function encodes the SAME
// three stages but each into its OWN command buffer, waiting on each
// before starting the next -- so GPUStartTime()/GPUEndTime() of each
// buffer is a real, isolated per-stage GPU timestamp, at the cost of two
// extra CPU<->GPU round trips per frame that the fused render() doesn't
// pay. Honors the same setSortSchedule() state as render() (so a --bench
// run profiling with scheduled sort shows the same amortized sort cost the
// production path would see).
// --------------------------------------------------------------------------
void MetalSplatLuxcRenderer::renderProfiled(MetalContext& ctx, double* preprocessGpuMs,
                                             double* sortGpuMs, double* drawGpuMs,
                                             double* totalGpuMs) {
    *preprocessGpuMs = *sortGpuMs = *drawGpuMs = *totalGpuMs = 0.0;
    if (numSplats_ == 0) return;

    if (hasMotionVectors_ && firstMvFrame_) {
        prevViewMatrix_ = viewMatrix_;
        prevProjMatrixUnjittered_ = projMatrixUnjittered_;
        std::memcpy(prevPosBuffer_->contents(), hostPositions_.data(), hostPositions_.size() * sizeof(float));
    }

    struct ComputePush {
        float view[16];
        float proj[16];
        float camPos[3];
        float _pad0;
        float screenW, screenH;
        uint32_t numSplats;
        float focalX;
        float focalY;
        int32_t shDegree;
        float _pad1[2];
    } push = {};
    std::memcpy(push.view, &viewMatrix_[0][0], 64);
    std::memcpy(push.proj, &projMatrix_[0][0], 64);
    push.camPos[0] = camPos_.x; push.camPos[1] = camPos_.y; push.camPos[2] = camPos_.z;
    push.screenW = static_cast<float>(width_);
    push.screenH = static_cast<float>(height_);
    push.numSplats = numSplats_;
    push.focalX = focalX_;
    push.focalY = focalY_;
    push.shDegree = static_cast<int32_t>(shDegree_);
    if (hasMotionVectors_) {
        glm::mat4 prevCameraMats[2];
        prevCameraMats[0] = projMatrixUnjittered_;
        prevCameraMats[1] = prevProjMatrixUnjittered_ * prevViewMatrix_;
        std::memcpy(prevCameraBuffer_->contents(), prevCameraMats, sizeof(prevCameraMats));
    }

    // --- Stage 1: preprocess (own command buffer) ---
    MTL::CommandBuffer* cmdBuf1 = ctx.beginCommandBuffer();
    {
        auto* enc = cmdBuf1->computeCommandEncoder();
        enc->setComputePipelineState(computePipeline_);
        trySetBuffer(enc, compShader_, posBuffer_, 0);
        trySetBuffer(enc, compShader_, rotBuffer_, 1);
        trySetBuffer(enc, compShader_, scaleBuffer_, 2);
        trySetBuffer(enc, compShader_, opacityBuffer_, 3);
        trySetBuffer(enc, compShader_, shBuffer_, 4);
        trySetBuffer(enc, compShader_, projCenterBuffer_, 5);
        trySetBuffer(enc, compShader_, projAxesBuffer_, 6);
        trySetBuffer(enc, compShader_, projExtentBuffer_, 7);
        trySetBuffer(enc, compShader_, projColorBuffer_, 8);
        trySetBuffer(enc, compShader_, sortKeysBuffer_, 9);
        trySetBuffer(enc, compShader_, sortedIndicesBuffer_, 10);
        trySetBuffer(enc, compShader_, visibleCountBuffer_, 11);
        uint32_t nextBinding = 12;
        if (hasMotionVectors_) {
            trySetBuffer(enc, compShader_, prevPosBuffer_, nextBinding++);
            trySetBuffer(enc, compShader_, projMvBuffer_, nextBinding++);
            trySetBuffer(enc, compShader_, prevCameraBuffer_, nextBinding++);
        }
        if (hasExpectedDepth_) {
            trySetBuffer(enc, compShader_, projDepthBuffer_, nextBinding++);
        }
        if (hasForegroundCoverage_) {
            trySetBuffer(enc, compShader_, foregroundBuffer_, nextBinding++);
            trySetBuffer(enc, compShader_, projForegroundBuffer_, nextBinding++);
        }
        enc->setBytes(&push, sizeof(push), compShader_.pushConstantBufferIndex);
        uint32_t groups = (numSplats_ + 255) / 256;
        enc->dispatchThreadgroups(MTL::Size(groups, 1, 1), MTL::Size(256, 1, 1));
        enc->endEncoding();
    }
    if (hasMotionVectors_) {
        auto* posCopyBlit = cmdBuf1->blitCommandEncoder();
        posCopyBlit->copyFromBuffer(posBuffer_, 0, prevPosBuffer_, 0, hostPositions_.size() * sizeof(float));
        posCopyBlit->endEncoding();
        prevViewMatrix_ = viewMatrix_;
        prevProjMatrixUnjittered_ = projMatrixUnjittered_;
        firstMvFrame_ = false;
    }
    cmdBuf1->commit();
    cmdBuf1->waitUntilCompleted();
    *preprocessGpuMs = (cmdBuf1->GPUEndTime() - cmdBuf1->GPUStartTime()) * 1000.0;

    // --- Sort scheduling decision (identical to render()'s) ---
    bool needsSort = true;
    if (sortEveryNFrames_ > 1 || sortViewThresholdDeg_ > 0.0f || sortTranslateThreshold_ > 0.0f) {
        glm::vec3 viewDir = glm::normalize(
            -glm::vec3(viewMatrix_[0][2], viewMatrix_[1][2], viewMatrix_[2][2]));
        bool viewChanged = true;
        if (hasLastSortedView_ && sortViewThresholdDeg_ > 0.0f) {
            float cosAngle = glm::clamp(glm::dot(viewDir, lastSortedViewDir_), -1.0f, 1.0f);
            float angleDeg = glm::degrees(std::acos(cosAngle));
            viewChanged = angleDeg >= sortViewThresholdDeg_;
        }
        bool translated = true;
        if (hasLastSortedView_ && sortTranslateThreshold_ > 0.0f) {
            translated = glm::length(camPos_ - lastSortedCamPos_) >= sortTranslateThreshold_;
        }
        // Any change at all (not thresholded) -- see setSortSchedule()'s
        // header comment: this is what makes the schedule safe for
        // dynamic/morphing scenes.
        bool morphChanged = hasLastSortedView_ && (currentMorphTime_ != lastSortedMorphTime_);
        bool budgetElapsed = framesSinceSort_ >= sortEveryNFrames_;
        needsSort = !hasLastSortedView_ || budgetElapsed ||
            (sortViewThresholdDeg_ > 0.0f && viewChanged) ||
            (sortTranslateThreshold_ > 0.0f && translated) ||
            morphChanged;
        if (needsSort) {
            framesSinceSort_ = 0;
            lastSortedViewDir_ = viewDir;
            lastSortedCamPos_ = camPos_;
            lastSortedMorphTime_ = currentMorphTime_;
            hasLastSortedView_ = true;
        } else {
            framesSinceSort_++;
        }
    }

    // --- Stage 2: sort (own command buffer, skipped per schedule) ---
    if (needsSort) {
        MTL::CommandBuffer* cmdBuf2 = ctx.beginCommandBuffer();
        static const uint32_t PREFIX_SUM_BLOCK_SIZE = 2048;
        uint32_t numElements = numSplats_;
        uint32_t numWg = sortNumWg_;
        uint32_t totalHistogram = 256 * numWg;
        uint32_t numParts = (totalHistogram + PREFIX_SUM_BLOCK_SIZE - 1) / PREFIX_SUM_BLOCK_SIZE;
        struct SortPush { uint32_t numElements; uint32_t bitOffset; };

        // --- Range reduction + quantization (see render()'s identical,
        // more heavily commented block) ---
        {
            auto* clearBlit = cmdBuf2->blitCommandEncoder();
            clearBlit->fillBuffer(keyRangeBuffer_, NS::Range(0, sizeof(uint32_t)), 0xFF);
            clearBlit->fillBuffer(keyRangeBuffer_, NS::Range(sizeof(uint32_t), sizeof(uint32_t)), 0x00);
            clearBlit->endEncoding();

            {
                auto* enc = cmdBuf2->computeCommandEncoder();
                enc->setComputePipelineState(sortReduceRangePipeline_);
                trySetBuffer(enc, sortReduceRangeShader_, sortKeysBuffer_, 0);
                trySetBuffer(enc, sortReduceRangeShader_, keyRangeBuffer_, 1);
                SortPush rp = {numElements, 0};
                enc->setBytes(&rp, sizeof(rp), sortReduceRangeShader_.pushConstantBufferIndex);
                enc->dispatchThreadgroups(MTL::Size(numWg, 1, 1), MTL::Size(256, 1, 1));
                enc->endEncoding();
            }
            {
                auto* enc = cmdBuf2->computeCommandEncoder();
                enc->setComputePipelineState(sortQuantizePipeline_);
                trySetBuffer(enc, sortQuantizeShader_, sortKeysBuffer_, 0);
                trySetBuffer(enc, sortQuantizeShader_, keyRangeBuffer_, 1);
                SortPush qp = {numElements, 0};
                enc->setBytes(&qp, sizeof(qp), sortQuantizeShader_.pushConstantBufferIndex);
                enc->dispatchThreadgroups(MTL::Size(numWg, 1, 1), MTL::Size(256, 1, 1));
                enc->endEncoding();
            }
        }

        for (uint32_t pass = 0; pass < 2; ++pass) {
            uint32_t bitOffset = pass * 8;
            uint32_t ping = pass % 2;
            MTL::Buffer* keysIn = ping == 0 ? sortKeysBuffer_ : sortKeysBBuffer_;
            MTL::Buffer* keysOut = ping == 0 ? sortKeysBBuffer_ : sortKeysBuffer_;
            MTL::Buffer* valsIn = ping == 0 ? sortedIndicesBuffer_ : sortValsBBuffer_;
            MTL::Buffer* valsOut = ping == 0 ? sortValsBBuffer_ : sortedIndicesBuffer_;
            {
                auto* enc = cmdBuf2->computeCommandEncoder();
                enc->setComputePipelineState(sortHistogramPipeline_);
                trySetBuffer(enc, sortHistogramShader_, keysIn, 0);
                trySetBuffer(enc, sortHistogramShader_, histogramBuffer_, 1);
                SortPush hp = {numElements, bitOffset};
                enc->setBytes(&hp, sizeof(hp), sortHistogramShader_.pushConstantBufferIndex);
                enc->dispatchThreadgroups(MTL::Size(numWg, 1, 1), MTL::Size(256, 1, 1));
                enc->endEncoding();
            }
            {
                auto* enc = cmdBuf2->computeCommandEncoder();
                enc->setComputePipelineState(sortPrefixSumPipeline_);
                trySetBuffer(enc, sortPrefixSumShader_, histogramBuffer_, 0);
                trySetBuffer(enc, sortPrefixSumShader_, partitionSumsBuffer_, 1);
                SortPush ps0 = {totalHistogram, 0};
                enc->setBytes(&ps0, sizeof(ps0), sortPrefixSumShader_.pushConstantBufferIndex);
                enc->dispatchThreadgroups(MTL::Size(numParts, 1, 1), MTL::Size(1024, 1, 1));
                enc->endEncoding();
            }
            {
                auto* enc = cmdBuf2->computeCommandEncoder();
                enc->setComputePipelineState(sortPrefixSumPipeline_);
                trySetBuffer(enc, sortPrefixSumShader_, histogramBuffer_, 0);
                trySetBuffer(enc, sortPrefixSumShader_, partitionSumsBuffer_, 1);
                SortPush ps1 = {numParts, 1};
                enc->setBytes(&ps1, sizeof(ps1), sortPrefixSumShader_.pushConstantBufferIndex);
                enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(1024, 1, 1));
                enc->endEncoding();
            }
            {
                auto* enc = cmdBuf2->computeCommandEncoder();
                enc->setComputePipelineState(sortPrefixSumPipeline_);
                trySetBuffer(enc, sortPrefixSumShader_, histogramBuffer_, 0);
                trySetBuffer(enc, sortPrefixSumShader_, partitionSumsBuffer_, 1);
                SortPush ps2 = {totalHistogram, 2};
                enc->setBytes(&ps2, sizeof(ps2), sortPrefixSumShader_.pushConstantBufferIndex);
                enc->dispatchThreadgroups(MTL::Size(numParts, 1, 1), MTL::Size(1024, 1, 1));
                enc->endEncoding();
            }
            {
                auto* enc = cmdBuf2->computeCommandEncoder();
                enc->setComputePipelineState(sortScatterPipeline_);
                trySetBuffer(enc, sortScatterShader_, keysIn, 0);
                trySetBuffer(enc, sortScatterShader_, keysOut, 1);
                trySetBuffer(enc, sortScatterShader_, valsIn, 2);
                trySetBuffer(enc, sortScatterShader_, valsOut, 3);
                trySetBuffer(enc, sortScatterShader_, histogramBuffer_, 4);
                SortPush sp = {numElements, bitOffset};
                enc->setBytes(&sp, sizeof(sp), sortScatterShader_.pushConstantBufferIndex);
                enc->dispatchThreadgroups(MTL::Size(numWg, 1, 1), MTL::Size(256, 1, 1));
                enc->endEncoding();
            }
        }
        cmdBuf2->commit();
        cmdBuf2->waitUntilCompleted();
        *sortGpuMs = (cmdBuf2->GPUEndTime() - cmdBuf2->GPUStartTime()) * 1000.0;
    }

    // --- Stage 3: draw (own command buffer) ---
    MTL::CommandBuffer* cmdBuf3 = ctx.beginCommandBuffer();
    auto* rpDesc = MTL::RenderPassDescriptor::alloc()->init();
    auto* colorAtt = rpDesc->colorAttachments()->object(0);
    colorAtt->setTexture(colorTarget_);
    colorAtt->setLoadAction(MTL::LoadActionClear);
    colorAtt->setStoreAction(MTL::StoreActionStore);
    float clearA = (hasMotionVectors_ || hasExpectedDepth_) ? 0.0f : 1.0f;
    colorAtt->setClearColor(MTL::ClearColor(0.0, 0.0, 0.0, clearA));
    uint32_t nextRpSlot = 1;
    if (hasMotionVectors_ || hasExpectedDepth_) {
        auto* auxAtt = rpDesc->colorAttachments()->object(nextRpSlot++);
        auxAtt->setTexture(auxTarget_);
        auxAtt->setLoadAction(MTL::LoadActionClear);
        auxAtt->setStoreAction(MTL::StoreActionStore);
        auxAtt->setClearColor(MTL::ClearColor(0.0, 0.0, 0.0, 0.0));
    }
    if (hasForegroundCoverage_) {
        auto* fgAtt = rpDesc->colorAttachments()->object(nextRpSlot++);
        fgAtt->setTexture(fgTarget_);
        fgAtt->setLoadAction(MTL::LoadActionClear);
        fgAtt->setStoreAction(MTL::StoreActionStore);
        fgAtt->setClearColor(MTL::ClearColor(0.0, 0.0, 0.0, 0.0));
    }
    auto* depthAtt = rpDesc->depthAttachment();
    depthAtt->setTexture(depthTarget_);
    depthAtt->setLoadAction(MTL::LoadActionClear);
    depthAtt->setStoreAction(MTL::StoreActionDontCare);
    depthAtt->setClearDepth(1.0);

    auto* enc = cmdBuf3->renderCommandEncoder(rpDesc);
    enc->setViewport(MTL::Viewport{0.0, static_cast<double>(height_),
                                    static_cast<double>(width_), -static_cast<double>(height_), 0.0, 1.0});
    enc->setScissorRect(MTL::ScissorRect{0, 0, width_, height_});
    enc->setRenderPipelineState(renderPipeline_);
    enc->setDepthStencilState(depthStencilState_);
    trySetVertexBuffer(enc, vertShader_, projCenterBuffer_, 0);
    trySetVertexBuffer(enc, vertShader_, projAxesBuffer_, 1);
    trySetVertexBuffer(enc, vertShader_, projExtentBuffer_, 2);
    trySetVertexBuffer(enc, vertShader_, projColorBuffer_, 3);
    trySetVertexBuffer(enc, vertShader_, sortedIndicesBuffer_, 4);
    uint32_t vNext = 5;
    if (hasMotionVectors_) trySetVertexBuffer(enc, vertShader_, projMvBuffer_, vNext++);
    if (hasExpectedDepth_) trySetVertexBuffer(enc, vertShader_, projDepthBuffer_, vNext++);
    if (hasForegroundCoverage_) trySetVertexBuffer(enc, vertShader_, projForegroundBuffer_, vNext++);
    struct RenderPush { float screenW, screenH; uint32_t visibleCount; float alphaMin; } renderPush = {};
    renderPush.screenW = static_cast<float>(width_);
    renderPush.screenH = static_cast<float>(height_);
    renderPush.visibleCount = numSplats_;
    renderPush.alphaMin = 1.0f / 255.0f;
    if (vertShader_.pushConstantBufferIndex != UINT32_MAX)
        enc->setVertexBytes(&renderPush, sizeof(renderPush), vertShader_.pushConstantBufferIndex);
    if (fragShader_.pushConstantBufferIndex != UINT32_MAX)
        enc->setFragmentBytes(&renderPush, sizeof(renderPush), fragShader_.pushConstantBufferIndex);
    // Indexed instanced draw: 6 indices over 4 unique vertices (quad) x
    // numSplats instances (perf; see metal_splat_luxc_renderer.h's
    // quadIndexBuffer_ comment / splat_expander.py's quad-corner comment)
    // -- 4 vertex-shader invocations/splat instead of the old non-indexed 6.
    enc->drawIndexedPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(6), MTL::IndexTypeUInt16,
                                quadIndexBuffer_, NS::UInteger(0), NS::UInteger(numSplats_));
    enc->endEncoding();
    rpDesc->release();
    cmdBuf3->commit();
    cmdBuf3->waitUntilCompleted();
    *drawGpuMs = (cmdBuf3->GPUEndTime() - cmdBuf3->GPUStartTime()) * 1000.0;

    *totalGpuMs = *preprocessGpuMs + *sortGpuMs + *drawGpuMs;
}

void MetalSplatLuxcRenderer::renderToDrawable(MetalContext&, CA::MetalDrawable*) {
    throw std::runtime_error("MetalSplatLuxcRenderer::renderToDrawable not implemented (headless-only backend)");
}

void MetalSplatLuxcRenderer::cleanup() {
    // Intentionally leaks MTL::Buffer*/Texture*/PipelineState* -- matches
    // MetalSplatRenderer::cleanup()'s own scope (process-lifetime headless
    // tool; Metal objects are released when the process exits).
}
