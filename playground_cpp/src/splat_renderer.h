#pragma once

#include <vulkan/vulkan.h>
#include "vk_mem_alloc.h"
#include "gltf_loader.h"  // SplatDynamics / SplatMorphSegment (dynamic splats)
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <string>
#include <cstdint>

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

    // Sub-pixel jitter in PIXELS, applied to the projection matrix used for
    // rasterization only -- motion vectors always use the unjittered
    // current/previous view-projections (see docs/lux-4d-spec.md section 3).
    // Positive jx shifts rendered content right, positive jy shifts it down.
    void setJitter(float jitterXPixels, float jitterYPixels);

    VkImage getMotionImage() const { return motionImage_; }
    VkFormat getMotionFormat() const { return VK_FORMAT_R32G32B32A32_SFLOAT; }  // xy=mv*alpha, z=0, w=alpha
    VkImage getExpectedDepthImage() const { return expectedDepthImage_; }
    // vec4 (x=depth*alpha, y/z unused, w=alpha), not vec2 -- Vulkan's
    // fixed-function alpha blend reads "source alpha" from the 4th
    // component of the fragment output for THIS attachment; a vec2 output
    // has none, and this was empirically found to blend as if src alpha
    // were 0 (undecayed running sum instead of the correct back-to-front
    // "over" composite) on MoltenVK/Apple GPUs. See
    // luxc/expansion/splat_expander.py's out_depth comment and
    // docs/lux-4d-spec.md section 3's depth-regression follow-up.
    VkFormat getExpectedDepthFormat() const { return VK_FORMAT_R32G32B32A32_SFLOAT; }

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

    // --- DLSS input-contract outputs (only allocated when the respective
    // flag is detected on the compiled shader) ---
    bool hasMotionVectors_ = false;
    bool hasExpectedDepth_ = false;
    VkImage motionImage_ = VK_NULL_HANDLE;
    VmaAllocation motionAlloc_ = VK_NULL_HANDLE;
    VkImageView motionView_ = VK_NULL_HANDLE;
    VkImage expectedDepthImage_ = VK_NULL_HANDLE;
    VmaAllocation expectedDepthAlloc_ = VK_NULL_HANDLE;
    VkImageView expectedDepthView_ = VK_NULL_HANDLE;

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
