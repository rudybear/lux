#pragma once

#include "metal_renderer_interface.h"
#include "metal_context.h"
#include "metal_scene_manager.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <string>
#include <cstdint>

struct GaussianSplatData;

class MetalSplatRenderer : public IMetalRenderer {
public:
    MetalSplatRenderer() = default;
    ~MetalSplatRenderer();

    void init(MetalContext& ctx, const GaussianSplatData& data,
              uint32_t width, uint32_t height);

    // --- Dynamic (4D) splats: morph-target keyframe animation ---
    // See docs/lux-4d-spec.md. Metal's splat pipeline already hand-writes
    // its compute/render kernels in MSL rather than transpiling luxc's
    // compiled SPIR-V (unlike the raster/mesh Metal paths, and unlike the
    // Vulkan splat renderer, which loads compiled .comp.spv directly) --
    // this applies the identical morph-apply algorithm luxc emits in
    // splat_expander._build_morph_apply_stage, but evaluated on the CPU and
    // written directly into the shared-storage-mode Metal buffers (no GPU
    // dispatch needed: MTL::ResourceStorageModeShared buffers are already
    // CPU-writable unified memory). Cost is proportional to the number of
    // gaussians that ever move in the whole animation, not the splat count.
    bool hasMotion() const { return dynamics_.has_motion; }
    float animationDuration() const;
    // Map a video-frame index to seconds (extras.fps if present, else the
    // animation's own keyframe times).
    float frameToTime(int frame) const;
    // Evaluate the animation at `seconds` and write the result directly into
    // posBuffer_/rotBuffer_/shBuffer_. Call before render()/renderToDrawable().
    void setMorphTime(float seconds);
    float currentMorphTimeSeconds() const { return currentMorphTime_; }
    // Snap to the next/previous keyframe time (direction = +1 or -1), for
    // "[" / "]" step-one-keyframe interactive playback.
    void stepKeyframe(int direction);

    void render(MetalContext& ctx) override;
    void renderToDrawable(MetalContext& ctx, CA::MetalDrawable* drawable) override;
    void updateCamera(glm::vec3 eye, glm::vec3 target, glm::vec3 up,
                      float fovY, float aspect,
                      float nearPlane, float farPlane) override;

    // --- Camera bridge (docs/lux-4d-spec.md section 4) ---
    void updateCameraExplicit(glm::vec3 eye, glm::mat4 viewMatrix, glm::mat4 projMatrix,
                               float focalX, float focalY);
    // Explicitly seed the motion-vector camera history instead of the
    // default prev==curr (mv=0) auto-seed on the first render(). See the
    // Vulkan SplatRenderer's identical method for rationale.
    void setPreviousCameraExplicit(glm::mat4 prevViewMatrix, glm::mat4 prevProjMatrixUnjittered) {
        prevViewMatrix_ = prevViewMatrix;
        prevProjMatrixUnjittered_ = prevProjMatrixUnjittered;
        firstMvFrame_ = false;
    }

    // --- DLSS input-contract outputs (docs/lux-4d-spec.md section 3) ---
    // Unlike the Vulkan splat renderer (which detects these from luxc's
    // compiled reflection JSON), Metal's splat pipeline is fully
    // hand-written MSL with no compiled-shader source of truth -- motion
    // vectors + expected depth are always computed; the host only pays for
    // the extra textures/readback when it actually asks for them.
    bool hasMotionVectors() const { return true; }
    bool hasExpectedDepth() const { return true; }
    void setJitter(float jitterXPixels, float jitterYPixels);

    MTL::Texture* getMotionTexture() const { return motionTarget_; }
    MTL::Texture* getExpectedDepthTexture() const { return expectedDepthTarget_; }

    MTL::Texture* getOutputTexture() const override { return colorTarget_; }
    uint32_t getWidth() const override { return width_; }
    uint32_t getHeight() const override { return height_; }

    void cleanup() override;

private:
    uint32_t width_ = 0, height_ = 0;
    uint32_t numSplats_ = 0;
    uint32_t shDegree_ = 0;

    // Offscreen render targets
    MTL::Texture* colorTarget_ = nullptr;
    MTL::Texture* depthTarget_ = nullptr;
    // DLSS input-contract outputs (docs/lux-4d-spec.md section 3): xy=mv*alpha,
    // z=0, w=alpha (motion); x=depth*alpha, y=alpha (expected depth) --
    // alpha duplicated at full float32 precision for exact host-side
    // un-premultiply, same rationale as the Vulkan splat renderer.
    MTL::Texture* motionTarget_ = nullptr;
    MTL::Texture* expectedDepthTarget_ = nullptr;

    // Compute pipeline (projection)
    MTL::ComputePipelineState* computePipeline_ = nullptr;

    // Render pipeline (alpha-blended quad drawing)
    MTL::RenderPipelineState* renderPipeline_ = nullptr;
    MTL::DepthStencilState* depthStencilState_ = nullptr;

    // Splat input buffers
    MTL::Buffer* posBuffer_ = nullptr;
    MTL::Buffer* rotBuffer_ = nullptr;
    MTL::Buffer* scaleBuffer_ = nullptr;
    MTL::Buffer* opacityBuffer_ = nullptr;
    MTL::Buffer* shBuffer_ = nullptr;

    // Projected output buffers (written by compute, read by render)
    MTL::Buffer* projCenterBuffer_ = nullptr;
    MTL::Buffer* projConicBuffer_ = nullptr;
    MTL::Buffer* projColorBuffer_ = nullptr;
    MTL::Buffer* projMvBuffer_ = nullptr;      // float2 per splat, raw pixel-space mv
    MTL::Buffer* projDepthBuffer_ = nullptr;   // float per splat, raw camera-space z

    // Sort index buffer (CPU-sorted, uploaded each frame)
    MTL::Buffer* sortedIndicesBuffer_ = nullptr;

    // Previous-frame animated world position (docs/lux-4d-spec.md section 3).
    // Shared-storage-mode Metal buffers are already CPU-writable unified
    // memory, so "double buffering" is just a memcpy from posBuffer_'s
    // current contents -- no GPU copy/barrier needed, unlike Vulkan.
    MTL::Buffer* prevPosBuffer_ = nullptr;

    // Camera state
    glm::mat4 viewMatrix_{1.0f};
    glm::mat4 projMatrix_{1.0f};              // may carry --jitter (rasterization only)
    glm::mat4 projMatrixUnjittered_{1.0f};    // always jitter-free (motion vectors)
    glm::vec3 camPos_{0.0f, 0.0f, 3.0f};
    float focalX_ = 256.0f;
    float focalY_ = 256.0f;
    float jitterX_ = 0.0f, jitterY_ = 0.0f;

    glm::mat4 prevViewMatrix_{1.0f};
    glm::mat4 prevProjMatrixUnjittered_{1.0f};
    bool firstMvFrame_ = true;

    // Cached positions for CPU sort (kept in sync with posBuffer_'s current,
    // possibly-morphed, contents)
    std::vector<float> hostPositions_;

    // --- Dynamic splats ---
    SplatDynamics dynamics_;
    std::vector<SplatMorphSegment> morphSegments_;   // one per target/segment, see buildSplatMorphSegments
    std::vector<uint32_t> everMovingIndices_;        // union of all targets' indices (for reset-to-base)
    std::vector<float> basePositions_;               // vec4 per splat, immutable
    std::vector<float> baseRotations_;               // vec4 per splat, immutable
    std::vector<float> baseSH0_;                     // vec4 per splat, immutable
    float currentMorphTime_ = 0.0f;

    // Helpers
    void createRenderTargets(MetalContext& ctx);
    void createPipelines(MetalContext& ctx);
    void createBuffers(MetalContext& ctx, const GaussianSplatData& data);
    void cpuSort();

    // Internal render to a specific render pass descriptor
    void renderToTarget(MetalContext& ctx, MTL::Texture* colorTex,
                        MTL::Texture* depthTex, uint32_t drawW, uint32_t drawH);
};
