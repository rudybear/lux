#pragma once

#include <vulkan/vulkan.h>
#include "vk_mem_alloc.h"
#include "gltf_loader.h"  // SplatDynamics / SplatMorphSegment (dynamic splats)
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <string>
#include <cstdint>
#include <algorithm>

struct VulkanContext;

class SplatRenderer {
public:
    SplatRenderer() = default;
    ~SplatRenderer();

    void init(VulkanContext& ctx, const GaussianSplatData& data,
              const std::string& shaderBase, uint32_t width, uint32_t height);

    // --- Dynamic (4D) splats: morph-target keyframe animation ---
    // Dispatches luxc's compiled <shaderBase>.morph.comp.spv (see
    // splat_expander._build_morph_apply_stage / SPECIFICATION.md 12.8)
    // before the existing preprocess dispatch each frame -- the playground
    // only binds buffers to what luxc emitted, per docs/lux-4d-spec.md.
    bool hasMotion() const { return dynamics_.has_motion && morphPipeline_ != VK_NULL_HANDLE; }
    float animationDuration() const;
    float frameToTime(int frame) const;
    void setMorphTime(float seconds);
    float currentMorphTimeSeconds() const { return currentMorphTime_; }
    void stepKeyframe(int direction);

    void updateCamera(glm::vec3 eye, glm::vec3 target, glm::vec3 up,
                      float fovY, float aspect, float nearPlane, float farPlane);

    // --- Camera bridge (docs/lux-4d-spec.md section 4) ---
    // Sets the camera directly from an already-built view matrix + a
    // (possibly off-axis, e.g. from OpenCV intrinsics via
    // DlssIO::buildIntrinsicsProjection) projection matrix, bypassing the
    // fovY/aspect-based construction in updateCamera(). `eye` is the
    // world-space camera position (for the SH view-dependent color term).
    void updateCameraExplicit(glm::vec3 eye, glm::mat4 viewMatrix, glm::mat4 projMatrix,
                               float focalX, float focalY);

    // Explicitly seeds the motion-vector camera history (docs/lux-4d-spec.md
    // section 3) instead of letting the first render() call auto-seed it to
    // "prev == curr" (mv == 0). Lets a single headless process produce a
    // deterministic non-zero motion vector from one render() call, driven
    // by two distinct --camera-json files (current + previous) -- used by
    // the MV-under-camera-motion test.
    void setPreviousCameraExplicit(glm::mat4 prevViewMatrix, glm::mat4 prevProjMatrixUnjittered) {
        prevViewMatrix_ = prevViewMatrix;
        prevProjMatrixUnjittered_ = prevProjMatrixUnjittered;
        firstMvFrame_ = false;
    }

    // Explicitly seeds splat_prev_pos (docs/lux-4d-spec.md section 3) by
    // evaluating the morph at `prevTimeSeconds` -- for `motion: keyframes`
    // splats only (no-op otherwise, since static splats' prevPosBuffer_
    // already aliases posBuffer_). Restores the working splat_pos/rot/sh0
    // buffers to whatever setMorphTime() last set afterward (they're
    // scratch space here; render() re-dispatches the morph for the current
    // time on every call anyway, so this doesn't need to leave them in any
    // particular state). Without this, a single headless render's mv only
    // ever reflects camera motion (prevPos defaults to the current frame's
    // own position) -- call this (mirroring setPreviousCameraExplicit) to
    // additionally validate mv against a real *positional* delta from one
    // process invocation.
    void seedPreviousMorphTime(VulkanContext& ctx, float prevTimeSeconds);

    // --- DLSS input-contract outputs (docs/lux-4d-spec.md section 3) ---
    // True when the compiled shader base was built with `motion_vectors: true`
    // / `expected_depth: true` (splat_expander.py); detected once at init()
    // time by scanning the preprocess stage's reflection JSON.
    bool hasMotionVectors() const { return hasMotionVectors_; }
    bool hasExpectedDepth() const { return hasExpectedDepth_; }
    // True when the compiled shader was built with `foreground_coverage: true`
    // (implies hasExpectedDepth_ -- the flag adds a SECOND small attachment,
    // getFgImage() below, not a channel of getAuxImage()).
    bool hasForegroundCoverage() const { return hasForegroundCoverage_; }
    // "aux_precision: half" was set on the compiled splat block -- see
    // getAuxFormat() (out_fg's format is independent of this flag -- see
    // getFgFormat()).
    bool auxPrecisionHalf() const { return auxPrecisionHalf_; }

    // Sub-pixel jitter in PIXELS, applied to the projection matrix used for
    // rasterization only -- motion vectors always use the unjittered
    // current/previous view-projections (see docs/lux-4d-spec.md section 3).
    // Positive jx shifts rendered content right, positive jy shifts it down.
    void setJitter(float jitterXPixels, float jitterYPixels);

    // Packed DLSS aux attachment (bench/lux_perf_ablation.md task 2):
    // `out_aux` is ONE attachment carrying
    // (mv.x*alpha, mv.y*alpha, depth*alpha, alpha) -- replaces the earlier
    // TWO-attachment (out_motion RGBA32F, out_depth RGBA32F, each with an
    // always-0 `.z`) design with one attachment that packs 3 real values
    // with ZERO wasted lanes. `.w` MUST be the genuine per-fragment alpha,
    // not a repurposed data channel -- see
    // luxc/expansion/splat_expander.py's out_aux comment for why: Vulkan's
    // fixed-function SRC_ALPHA/ONE_MINUS_SRC_ALPHA blend factors read
    // "source alpha" from THIS attachment's own 4th component specifically,
    // so anything else there corrupts the blend accumulation itself (a
    // real, measured, 10-50x MV/depth error in an earlier version of this
    // design that tried to pack `fg` into `.w` instead and un-premultiply
    // everything via out_color's alpha -- NOT merely a precision tradeoff,
    // a GPU-side correctness bug). Format is a compile-time host hint
    // (`aux_precision: float|half` on the splat block, read from the
    // reflection JSON, NOT baked into the shader -- same vec4 output
    // either way): "float" (default) is VK_FORMAT_R32G32B32A32_SFLOAT,
    // the SAME full-precision accumulation the old two-attachment design
    // had -- 2->1 attachment, 32B/px->16B/px whenever foreground_coverage
    // is off. "half" is VK_FORMAT_R16G16B16A16_SFLOAT, 8B/px -- MEASURED
    // (with the `.w`-alpha bug already fixed, isolating pure half-float
    // accumulation precision) to FAIL the parity gate: MV median error
    // 0.012-0.016px (gate <=0.01px), depth relative median error ~5.4e-3
    // (gate <=1e-3) -- premultiplied mv/depth values are not
    // magnitude-bounded like color's own [0,1] channels, so half-float
    // blend rounding over many overlapping splats still exceeds this
    // task's tight gates. See bench/lux_perf_ablation.md for the full
    // table. Kept opt-in only (examples/gaussian_splat_dlss_half.lux),
    // default stays "float".
    VkImage getAuxImage() const { return auxImage_; }
    VkFormat getAuxFormat() const {
        return auxPrecisionHalf_ ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R32G32B32A32_SFLOAT;
    }

    // Second, smaller attachment for foreground_coverage (only allocated
    // when hasForegroundCoverage()): `out_fg` = (fg*alpha, 0, 0, alpha).
    // Can't share `out_aux` -- that attachment's 3 non-alpha lanes are
    // already spoken for by mv.xy/depth, and `.w` must stay genuine alpha
    // (see getAuxImage()'s comment) in EVERY blended attachment
    // independently, not just one. ALWAYS RGBA16F, regardless of
    // aux_precision -- fg (like alpha itself) is bounded to [0,1], the
    // same magnitude-boundedness that keeps out_color's own RGBA16F blend
    // accumulation safe applies here too (measured: no precision
    // regression vs. RGBA32F), unlike out_aux's mv/depth values which are
    // NOT magnitude-bounded. Do NOT drop this to RG16F under
    // "aux_precision: half": that was tried and MEASURED to fail the
    // parity gate hard (a full 0-to-1 flip on ~1% of pixels, p99/max abs
    // diff both exactly 1.0) -- Vulkan/Metal's fixed-function
    // SRC_ALPHA/ONE_MINUS_SRC_ALPHA blend reads "source alpha" from THIS
    // attachment's own 4th component specifically, and a 2-channel format
    // has no 4th component for it to read, so the blend silently treats
    // source alpha as a constant instead -- a real correctness bug, not a
    // precision tradeoff (see getAuxFormat()'s comment for why out_aux
    // itself is still safe to shrink: it keeps its `.w` alpha lane in
    // both "float" and "half"). "aux_precision: half" therefore only
    // changes getAuxFormat(); out_fg's format is independent of it.
    VkImage getFgImage() const { return fgImage_; }
    VkFormat getFgFormat() const { return VK_FORMAT_R16G16B16A16_SFLOAT; }

    void render(VulkanContext& ctx);

    // --- Optional GPU timestamp-query profiling (mobile-DLSS Android live
    // demo, Stage 1 "timing breakdown" -- docs/rendering-engines.md). Off by
    // default (zero query-pool overhead when disabled, so desktop CLI /
    // existing callers are unaffected). When enabled, render() writes 4
    // timestamps into an internal VkQueryPool: [0] frame start (top of
    // pipe), [1] end of the preprocess compute dispatch (before the radix
    // sort), [2] end of the radix sort (before the render pass), [3] end of
    // the draw (after vkCmdEndRenderPass) -- and lastGpuTimingsMs() converts
    // the deltas to milliseconds once render() returns (already
    // GPU-synchronous via endSingleTimeCommands, so results are always
    // ready by then). Call setGpuTimingEnabled() once after init().
    struct GpuTimingsMs {
        double preprocessMs = 0.0;  // [0]->[1]
        double sortMs = 0.0;        // [1]->[2] (includes the MoltenVK-only MV
                                     // queue-drain round trip when hasMotionVectors_)
        double drawMs = 0.0;        // [2]->[3]
        bool valid = false;
    };
    void setGpuTimingEnabled(VulkanContext& ctx, bool enabled);
    GpuTimingsMs lastGpuTimingsMs() const { return lastGpuTimings_; }

    // --- Sort scheduling (perf; bench/lux_perf_ablation.md in mobiledlss:
    // the GPU radix sort is a flat ~5.7-7.1ms/frame cost regardless of
    // scene, independent of whether the view actually changed enough to
    // need a fresh back-to-front order). Default (everyNFrames=1,
    // viewChangeThresholdDeg=0) is UNCHANGED, exactness-preserving
    // behavior -- every render() call re-sorts, exactly as before this was
    // added; this is what tests/mobiledlss parity checks/headless one-shot
    // renders all still get, since a single render() call always has
    // framesSinceSort_ == 0 (never skips) regardless of the schedule.
    // Opt in via setSortSchedule() for a continuous multi-frame caller
    // (interactive/live rendering) willing to trade briefly-stale
    // back-to-front blend order (draw() still runs every frame against
    // whatever sortedIndicesBuffer_ last held -- correctness of WHICH
    // splats are visible is unaffected, since that's decided per-splat in
    // preprocess/fragment, not by sort position) for amortized sort cost:
    // a fresh sort runs when EITHER `framesSinceSort_ >= everyNFrames` OR
    // the camera's view direction has rotated more than
    // `viewChangeThresholdDeg` since the last real sort (whichever comes
    // first), so a fast-panning camera still re-sorts promptly even under
    // a large everyNFrames budget.
    void setSortSchedule(uint32_t everyNFrames, float viewChangeThresholdDeg) {
        sortEveryNFrames_ = std::max<uint32_t>(1, everyNFrames);
        sortViewThresholdDeg_ = viewChangeThresholdDeg;
    }

    void blitToSwapchain(VulkanContext& ctx, VkCommandBuffer cmd,
                         VkImage swapImage, VkExtent2D extent);

    // Blit to swapchain in compositing mode: transitions from PRESENT_SRC
    // instead of UNDEFINED (for drawing on top of a previously-rendered frame).
    void blitToSwapchainComposite(VulkanContext& ctx, VkCommandBuffer cmd,
                                   VkImage swapImage, VkExtent2D extent);

    // Preload a background image into the splat color buffer.
    // Subsequent render() call will use LOAD instead of CLEAR so splats
    // are composited on top of the background.
    void preloadBackground(VulkanContext& ctx, VkImage srcImage, VkFormat srcFormat,
                           uint32_t srcWidth, uint32_t srcHeight);

    // Preload depth from raster pass into splat depth buffer.
    // Splats will depth-test against mesh geometry so occluded splats are hidden.
    void preloadDepth(VulkanContext& ctx, VkImage srcDepthImage,
                      uint32_t srcWidth, uint32_t srcHeight);

    void cleanup(VulkanContext& ctx);

    VkImage getOutputImage() const { return colorImage_; }
    // RGBA16_SFLOAT (docs/lux-4d-spec.md section 3's PSNR follow-up): the
    // splat color target used to be R8G8B8A8_UNORM, which -- with hundreds
    // of overlapping low-alpha fragments per pixel -- accumulates
    // per-blend rounding to 1/255 and caps effective precision; 16-bit
    // float supports hardware blending on Apple GPUs (unlike 32-bit, see
    // the Metal splat/motion/depth attachments) and removes that rounding
    // entirely. Screenshot::saveImageToPNG converts to 8-bit only at the
    // very end, by rounding (not truncating).
    VkFormat getOutputFormat() const { return VK_FORMAT_R16G16B16A16_SFLOAT; }
    uint32_t getWidth() const { return width_; }
    uint32_t getHeight() const { return height_; }

private:
    uint32_t width_ = 0, height_ = 0;
    uint32_t numSplats_ = 0;

    // Offscreen render targets
    VkImage colorImage_ = VK_NULL_HANDLE;
    VmaAllocation colorAlloc_ = VK_NULL_HANDLE;
    VkImageView colorView_ = VK_NULL_HANDLE;

    VkImage depthImage_ = VK_NULL_HANDLE;
    VmaAllocation depthAlloc_ = VK_NULL_HANDLE;
    VkImageView depthView_ = VK_NULL_HANDLE;

    // --- DLSS input-contract outputs (only allocated when motion_vectors
    // or expected_depth is detected on the compiled shader) ---
    bool hasMotionVectors_ = false;
    bool hasExpectedDepth_ = false;
    bool hasForegroundCoverage_ = false;
    // "aux_precision: half" (bench/lux_perf_ablation.md task 2 follow-up)
    // -- see getAuxFormat()/getFgFormat() for what this selects.
    bool auxPrecisionHalf_ = false;
    // Packed RGBA32F attachment (mv.x*a, mv.y*a, depth*a, a) -- see
    // getAuxImage()'s comment (NOT RGBA16F -- measured precision failure).
    VkImage auxImage_ = VK_NULL_HANDLE;
    VmaAllocation auxAlloc_ = VK_NULL_HANDLE;
    VkImageView auxView_ = VK_NULL_HANDLE;
    // Second, smaller RGBA16F attachment (fg*a, 0, 0, a) -- only allocated
    // when hasForegroundCoverage_ -- see getFgImage()'s comment.
    VkImage fgImage_ = VK_NULL_HANDLE;
    VmaAllocation fgAlloc_ = VK_NULL_HANDLE;
    VkImageView fgView_ = VK_NULL_HANDLE;

    VkRenderPass renderPass_ = VK_NULL_HANDLE;
    VkFramebuffer framebuffer_ = VK_NULL_HANDLE;

    // Second render pass/framebuffer pair with LOAD_OP_LOAD for color compositing
    VkRenderPass renderPassLoad_ = VK_NULL_HANDLE;
    VkFramebuffer framebufferLoad_ = VK_NULL_HANDLE;
    bool hasBackground_ = false;

    // Third render pass/framebuffer pair: LOAD both color AND depth (full hybrid compositing)
    VkRenderPass renderPassLoadDepth_ = VK_NULL_HANDLE;
    VkFramebuffer framebufferLoadDepth_ = VK_NULL_HANDLE;
    bool hasBackgroundDepth_ = false;

    // Pipelines
    VkPipeline computePipeline_ = VK_NULL_HANDLE;
    VkPipeline renderPipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout computeLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout renderLayout_ = VK_NULL_HANDLE;

    // Descriptor sets
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout computeSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout renderSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSet computeDescSet_ = VK_NULL_HANDLE;
    VkDescriptorSet renderDescSet_ = VK_NULL_HANDLE;

    // Splat GPU buffers (VMA)
    VkBuffer posBuffer_ = VK_NULL_HANDLE;       VmaAllocation posAlloc_ = VK_NULL_HANDLE;
    VkBuffer rotBuffer_ = VK_NULL_HANDLE;       VmaAllocation rotAlloc_ = VK_NULL_HANDLE;
    VkBuffer scaleBuffer_ = VK_NULL_HANDLE;     VmaAllocation scaleAlloc_ = VK_NULL_HANDLE;
    VkBuffer opacityBuffer_ = VK_NULL_HANDLE;   VmaAllocation opacityAlloc_ = VK_NULL_HANDLE;
    std::vector<VkBuffer> shBuffers_;
    std::vector<VmaAllocation> shAllocs_;

    // Projected output buffers
    VkBuffer projCenterBuffer_ = VK_NULL_HANDLE;  VmaAllocation projCenterAlloc_ = VK_NULL_HANDLE;
    // Oriented-quad half-axis vectors (major.xy, minor.xy), one vec4 per
    // splat -- see splat_expander.py's "Oriented quads" note. Read by the
    // vertex stage instead of the old scalar radius*(quad_x,quad_y) offset.
    VkBuffer projAxesBuffer_ = VK_NULL_HANDLE;    VmaAllocation projAxesAlloc_ = VK_NULL_HANDLE;
    VkBuffer projConicBuffer_ = VK_NULL_HANDLE;   VmaAllocation projConicAlloc_ = VK_NULL_HANDLE;
    VkBuffer projColorBuffer_ = VK_NULL_HANDLE;   VmaAllocation projColorAlloc_ = VK_NULL_HANDLE;
    VkBuffer projMvBuffer_ = VK_NULL_HANDLE;      VmaAllocation projMvAlloc_ = VK_NULL_HANDLE;
    VkBuffer projDepthBuffer_ = VK_NULL_HANDLE;   VmaAllocation projDepthAlloc_ = VK_NULL_HANDLE;
    // Per-frame camera data for the (jitter-free) motion-vector projection:
    // [0]=proj_matrix_unjittered, [1]=prev_view_proj_unjittered (2x mat4 =
    // 128 bytes) -- a tiny storage buffer instead of push constants (see
    // splat_expander.py's "Why not push constants" comment: pushing these
    // as push-constant fields put the compute stage's push-constant block
    // at 304 bytes, over Mali-G715's 256-byte maxPushConstantsSize, which
    // silently corrupted every motion vector on Android). Written via
    // vkCmdUpdateBuffer each render() call, matching the old memcpy-into-
    // push-constants call site exactly.
    VkBuffer prevCameraBuffer_ = VK_NULL_HANDLE;  VmaAllocation prevCameraAlloc_ = VK_NULL_HANDLE;
    // foreground_coverage: splat_foreground (input, uploaded once from
    // GaussianSplatData::foreground) and projected_foreground (output,
    // per-splat passthrough written by the preprocess compute stage).
    VkBuffer foregroundBuffer_ = VK_NULL_HANDLE;      VmaAllocation foregroundAlloc_ = VK_NULL_HANDLE;
    VkBuffer projForegroundBuffer_ = VK_NULL_HANDLE;  VmaAllocation projForegroundAlloc_ = VK_NULL_HANDLE;

    // Sort buffers (buffer A = primary, written by compute shader)
    VkBuffer sortKeysBuffer_ = VK_NULL_HANDLE;      VmaAllocation sortKeysAlloc_ = VK_NULL_HANDLE;
    VkBuffer sortedIndicesBuffer_ = VK_NULL_HANDLE;  VmaAllocation sortedIndicesAlloc_ = VK_NULL_HANDLE;

    // Visible count (atomic counter)
    VkBuffer visibleCountBuffer_ = VK_NULL_HANDLE;  VmaAllocation visibleCountAlloc_ = VK_NULL_HANDLE;

    // GPU radix sort resources
    VkBuffer sortKeysBBuffer_ = VK_NULL_HANDLE;      VmaAllocation sortKeysBAlloc_ = VK_NULL_HANDLE;
    VkBuffer sortValsBBuffer_ = VK_NULL_HANDLE;      VmaAllocation sortValsBAlloc_ = VK_NULL_HANDLE;
    VkBuffer histogramBuffer_ = VK_NULL_HANDLE;      VmaAllocation histogramAlloc_ = VK_NULL_HANDLE;
    VkBuffer partitionSumsBuffer_ = VK_NULL_HANDLE;  VmaAllocation partitionSumsAlloc_ = VK_NULL_HANDLE;

    // Sort pipelines (3 compute stages)
    VkPipeline sortHistogramPipeline_ = VK_NULL_HANDLE;
    VkPipeline sortPrefixSumPipeline_ = VK_NULL_HANDLE;
    VkPipeline sortScatterPipeline_ = VK_NULL_HANDLE;

    // Sort pipeline layouts
    VkPipelineLayout sortHistogramLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout sortPrefixSumLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout sortScatterLayout_ = VK_NULL_HANDLE;

    // Sort descriptor set layouts
    VkDescriptorSetLayout sortHistogramSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout sortPrefixSumSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout sortScatterSetLayout_ = VK_NULL_HANDLE;

    // Sort descriptor sets: [0]=A->B, [1]=B->A for histogram and scatter; single for prefix_sum
    VkDescriptorSet sortHistogramDescSets_[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkDescriptorSet sortPrefixSumDescSet_ = VK_NULL_HANDLE;
    VkDescriptorSet sortScatterDescSets_[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};

    // Precomputed sort workgroup count
    uint32_t sortNumWg_ = 0;

    // Camera state
    glm::mat4 viewMatrix_{1.0f};
    glm::mat4 projMatrix_{1.0f};             // may carry --jitter offset (rasterization only)
    glm::mat4 projMatrixUnjittered_{1.0f};   // always jitter-free (used for motion vectors)
    glm::vec3 camPos_{0.0f, 0.0f, 3.0f};
    float focalX_ = 256.0f;
    float focalY_ = 256.0f;
    float jitterX_ = 0.0f, jitterY_ = 0.0f;  // pixels

    // Sort scheduling state (see setSortSchedule() above).
    uint32_t sortEveryNFrames_ = 1;       // 1 = always sort (default, exactness-preserving)
    float sortViewThresholdDeg_ = 0.0f;   // 0 = never trigger early on view change alone
    uint32_t framesSinceSort_ = 0;
    glm::vec3 lastSortedViewDir_{0.0f, 0.0f, -1.0f};
    bool hasLastSortedView_ = false;

    // Previous-frame camera history for motion vectors (docs/lux-4d-spec.md
    // section 3). Seeded to equal the current frame's matrices on the first
    // render() call so mv == 0 on frame 1, then carried forward each frame.
    glm::mat4 prevViewMatrix_{1.0f};
    glm::mat4 prevProjMatrixUnjittered_{1.0f};
    bool firstMvFrame_ = true;

    // Previous-frame animated world position, for motion_vectors. For
    // static splats (!hasMotion()) this simply aliases posBuffer_ (positions
    // never change); for `motion: keyframes` splats it's a distinct buffer
    // the host copies posBuffer_ into once per frame, after the morph-apply
    // stage has written the current frame's positions.
    VkBuffer prevPosBuffer_ = VK_NULL_HANDLE;
    VmaAllocation prevPosAlloc_ = VK_NULL_HANDLE;
    bool prevPosOwned_ = false;
    uint32_t shDegree_ = 0;        // scene's actual SH degree (for push constant)
    uint32_t shaderShDegree_ = 0;  // shader's compiled SH degree (for descriptor layout)

    // --- GPU timestamp-query profiling (see setGpuTimingEnabled() above) ---
    VkQueryPool timestampPool_ = VK_NULL_HANDLE;
    bool gpuTimingEnabled_ = false;
    double timestampPeriodNs_ = 1.0;
    GpuTimingsMs lastGpuTimings_;

    // Helpers
    void createOffscreenTarget(VulkanContext& ctx);
    void createRenderPass(VkDevice device);
    void createRenderPassLoad(VkDevice device);
    void createRenderPassLoadDepth(VkDevice device);
    void createFramebuffer(VkDevice device);
    void createFramebufferLoad(VkDevice device);
    void createFramebufferLoadDepth(VkDevice device);
    void createPipelines(VulkanContext& ctx, const std::string& shaderBase);
    void createSortPipelines(VkDevice device);
    void createBuffers(VulkanContext& ctx, const GaussianSplatData& data);

    // --- Dynamic splats ---
    SplatDynamics dynamics_;
    std::vector<SplatMorphSegment> morphSegments_;
    std::vector<uint32_t> segmentOffsets_;  // CPU-side, into the concatenated morph_* buffers
    std::vector<uint32_t> segmentCounts_;
    uint32_t morphTotalEntries_ = 0;
    float currentMorphTime_ = 0.0f;
    // Perf: dispatchMorph() only needs to reset the segment it PREVIOUSLY
    // applied (not the full concatenated multi-keyframe morph_index array)
    // before applying the new one -- see dispatchMorph()'s comment for the
    // correctness argument. -1 == no segment applied yet (fresh buffers,
    // already == base attrs, nothing to reset).
    int lastAppliedSegment_ = -1;

    VkPipeline morphPipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout morphLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout morphSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSet morphDescSet_ = VK_NULL_HANDLE;

    VkBuffer baseposBuffer_ = VK_NULL_HANDLE;   VmaAllocation baseposAlloc_ = VK_NULL_HANDLE;
    VkBuffer baserotBuffer_ = VK_NULL_HANDLE;   VmaAllocation baserotAlloc_ = VK_NULL_HANDLE;
    VkBuffer basesh0Buffer_ = VK_NULL_HANDLE;   VmaAllocation basesh0Alloc_ = VK_NULL_HANDLE;
    VkBuffer morphIndexBuffer_ = VK_NULL_HANDLE;  VmaAllocation morphIndexAlloc_ = VK_NULL_HANDLE;
    VkBuffer morphPosLoBuffer_ = VK_NULL_HANDLE;  VmaAllocation morphPosLoAlloc_ = VK_NULL_HANDLE;
    VkBuffer morphRotLoBuffer_ = VK_NULL_HANDLE;  VmaAllocation morphRotLoAlloc_ = VK_NULL_HANDLE;
    VkBuffer morphSh0LoBuffer_ = VK_NULL_HANDLE;  VmaAllocation morphSh0LoAlloc_ = VK_NULL_HANDLE;
    VkBuffer morphPosHiBuffer_ = VK_NULL_HANDLE;  VmaAllocation morphPosHiAlloc_ = VK_NULL_HANDLE;
    VkBuffer morphRotHiBuffer_ = VK_NULL_HANDLE;  VmaAllocation morphRotHiAlloc_ = VK_NULL_HANDLE;
    VkBuffer morphSh0HiBuffer_ = VK_NULL_HANDLE;  VmaAllocation morphSh0HiAlloc_ = VK_NULL_HANDLE;

    void createMorphPipeline(VkDevice device, const std::string& shaderBase);
    void createMorphBuffers(VulkanContext& ctx, const GaussianSplatData& data);
    // Records the two-dispatch morph-apply sequence (reset then weighted
    // apply) for `timeSeconds` into `cmd`, writing splat_pos/rot/sh0 (the
    // working buffers). Shared by render() and seedPreviousMorphTime().
    void dispatchMorph(VkCommandBuffer cmd, float timeSeconds);

    // --- DLSS input-contract outputs ---
    void createPrevPosBuffer(VulkanContext& ctx, const GaussianSplatData& data);
};
