#pragma once

// Metal splat renderer, luxc-compiled-pipeline backend (--splat-backend luxc;
// NOT YET the default -- see metal_main.cpp's splatBackend comment. Status:
// builds and runs end-to-end (compute preprocess, 4-pass GPU radix sort,
// vertex/fragment render, morph, DLSS motion/depth outputs all execute
// without error) and two real bugs found+fixed during bring-up (expected-
// depth attachment needed RGBA32Float + hardware alpha blending, not
// MetalSplatRenderer's RG32Float/manual-blend convention -- Apple Silicon
// DOES support float32 attachment blending, unlike the comment that
// motivated the hand-written path's manual approach) got depth to 0.36%
// median rel. error vs. Vulkan on the juggle DLSS scene (frame 40, sort:
// view_depth) -- close but short of the <1e-3 acceptance bar. Colour is NOT
// yet at parity: ~28 dB vs. gsplat and ~28 dB vs. Vulkan's own output on the
// same scene (need >=45 dB), with GPU radix sort verified correctly ordered
// (0 out-of-order keys) and per-splat SH colour clamp already matching
// Vulkan's, so the remaining gap's root cause is still open -- possibly a
// numerical-precision difference between MoltenVK's own internal SPIR-V ->
// MSL transpile (used for the "Vulkan" backend on this Mac) and this
// backend's direct ShaderTranspiler pass, given hundreds of overlapping
// low-alpha premultiplied blends per pixel are extremely sensitive to
// exactly this kind of divergence. Runs the SAME compiled SPIR-V
// (examples/<shaderBase>.{comp,vert,frag,morph.comp}.spv, emitted by
// luxc/expansion/splat_expander.py) and the shared GPU radix sort
// (shaders/radix_sort/{histogram,prefix_sum,scatter}.comp.spv) that the
// Vulkan splat renderer (splat_renderer.cpp) already uses, transpiled to MSL
// via the existing ShaderTranspiler (SPIR-V -> MSL, the same mechanism
// metal_mesh_renderer.cpp/metal_reconstruct_runner.cpp/metal_unet_runner.cpp
// already use) -- unlike MetalSplatRenderer (metal_splat_renderer.{h,cpp},
// "--splat-backend hand"), which hand-writes its own MSL and therefore has
// no compiled-.lux source of truth (no `sort:`, and a hand-maintained
// covariance/SH/compositing implementation that can silently drift from the
// Vulkan/gsplat-verified one -- see the SH-colour-clamp gap this backend was
// created to close).
//
// Buffer names, push-constant field layouts and dispatch order are 1:1 with
// splat_renderer.cpp (Vulkan) -- see that file's render()/dispatchMorph()
// for the reference orchestration this ports; CPU-side data prep (vec4
// padding, morph segment building) is verbatim-identical to
// MetalSplatRenderer's own (backend-agnostic: it just fills MTL::Buffer*
// contents), since Metal's shared-storage buffers are the same "already
// laid out for the GPU" data regardless of which pipeline reads them.

#include "metal_renderer_interface.h"
#include "metal_context.h"
#include "metal_scene_manager.h"
#include "metal_shader_transpiler.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <string>
#include <cstdint>
#include <cstring>

struct GaussianSplatData;

class MetalSplatLuxcRenderer : public IMetalRenderer {
public:
    MetalSplatLuxcRenderer() = default;
    ~MetalSplatLuxcRenderer();

    // `shaderBase` (e.g. "examples/gaussian_splat_dlss") selects the
    // compiled pipeline -- unlike MetalSplatRenderer::init(), which has no
    // such parameter (it never reads any compiled `.lux` output). `sort:`,
    // `motion_vectors:`, `expected_depth:` are read from the compute
    // stage's own reflection JSON, exactly like the Vulkan SplatRenderer.
    void init(MetalContext& ctx, const GaussianSplatData& data, const std::string& shaderBase,
              uint32_t width, uint32_t height);

    // --- Dynamic (4D) splats: morph-target keyframe animation ---
    bool hasMotion() const { return dynamics_.has_motion; }
    float animationDuration() const;
    float frameToTime(int frame) const;
    void setMorphTime(float seconds);
    float currentMorphTimeSeconds() const { return currentMorphTime_; }
    void stepKeyframe(int direction);

    void render(MetalContext& ctx) override;
    void renderToDrawable(MetalContext& ctx, CA::MetalDrawable* drawable) override;
    void updateCamera(glm::vec3 eye, glm::vec3 target, glm::vec3 up,
                      float fovY, float aspect,
                      float nearPlane, float farPlane) override;

    // --- Camera bridge (docs/lux-4d-spec.md section 4) ---
    void updateCameraExplicit(glm::vec3 eye, glm::mat4 viewMatrix, glm::mat4 projMatrix,
                               float focalX, float focalY);
    void setPreviousCameraExplicit(glm::mat4 prevViewMatrix, glm::mat4 prevProjMatrixUnjittered) {
        prevViewMatrix_ = prevViewMatrix;
        prevProjMatrixUnjittered_ = prevProjMatrixUnjittered;
        firstMvFrame_ = false;
    }
    void seedPreviousMorphTime(float prevTimeSeconds) {
        if (!dynamics_.has_motion) return;
        if (firstMvFrame_) {
            prevViewMatrix_ = viewMatrix_;
            prevProjMatrixUnjittered_ = projMatrixUnjittered_;
        }
        float savedTime = currentMorphTime_;
        setMorphTime(prevTimeSeconds);
        std::memcpy(prevPosBuffer_->contents(), hostPositions_.data(),
                    hostPositions_.size() * sizeof(float));
        setMorphTime(savedTime);
        firstMvFrame_ = false;
    }

    // --- DLSS input-contract outputs -- read from the compiled reflection
    // (unlike MetalSplatRenderer, which always computes both). ---
    bool hasMotionVectors() const { return hasMotionVectors_; }
    bool hasExpectedDepth() const { return hasExpectedDepth_; }
    void setJitter(float jitterXPixels, float jitterYPixels);

    MTL::Texture* getMotionTexture() const { return motionTarget_; }
    MTL::Texture* getExpectedDepthTexture() const { return expectedDepthTarget_; }
    // RGBA32Float here (4 floats/pixel: x=depth*alpha, y/z unused, w=alpha)
    // vs. MetalSplatRenderer's RG32Float (2 floats/pixel) -- this backend's
    // fragment shader needs the real 4th-component alpha for hardware
    // blending (commit 1445ec3's out_depth fix); the hand-written path
    // blends manually and never needed it.
    static constexpr uint32_t kExpectedDepthChannels = 4;

    // `sort:` is compile-time on this backend (baked into the compiled
    // pipeline, exactly like Vulkan) -- this is a no-op kept only so
    // metal_main.cpp's shared call site doesn't need a backend branch for
    // it; use --splat-backend hand + --sort to override sort at the CLI.
    void setSortByViewDepth(bool) {}

    MTL::Texture* getOutputTexture() const override { return colorTarget_; }
    uint32_t getWidth() const override { return width_; }
    uint32_t getHeight() const override { return height_; }

    const float* debugPosBufferPtr() const { return static_cast<const float*>(posBuffer_->contents()); }
    const float* debugRotBufferPtr() const { return static_cast<const float*>(rotBuffer_->contents()); }
    const float* debugSh0BufferPtr() const { return static_cast<const float*>(shBuffer_->contents()); }

    void cleanup() override;

private:
    uint32_t width_ = 0, height_ = 0;
    uint32_t numSplats_ = 0;
    uint32_t shDegree_ = 0;
    std::string shaderBase_;

    MetalContext* ctx_ = nullptr;
    ShaderTranspiler transpiler_;

    // Offscreen render targets (same formats as MetalSplatRenderer).
    MTL::Texture* colorTarget_ = nullptr;
    MTL::Texture* depthTarget_ = nullptr;
    MTL::Texture* motionTarget_ = nullptr;
    MTL::Texture* expectedDepthTarget_ = nullptr;

    // Transpiled compute (preprocess) + render (vert/frag) + morph shaders.
    TranspiledShader compShader_;
    TranspiledShader vertShader_;
    TranspiledShader fragShader_;
    TranspiledShader morphShader_;
    MTL::ComputePipelineState* computePipeline_ = nullptr;
    MTL::RenderPipelineState* renderPipeline_ = nullptr;
    MTL::DepthStencilState* depthStencilState_ = nullptr;
    MTL::ComputePipelineState* morphPipeline_ = nullptr;
    bool hasMotionVectors_ = false;
    bool hasExpectedDepth_ = false;

    // GPU radix sort (shaders/radix_sort/*.comp.spv, transpiled) -- same
    // 4-pass histogram/prefix_sum/scatter ping-pong scheme as
    // splat_renderer.cpp. One workgroup-size-256 histogram/scatter pipeline,
    // one workgroup-size-1024 prefix_sum pipeline (matching each shader's
    // own compiled --define workgroup_size).
    TranspiledShader sortHistogramShader_;
    TranspiledShader sortPrefixSumShader_;
    TranspiledShader sortScatterShader_;
    MTL::ComputePipelineState* sortHistogramPipeline_ = nullptr;
    MTL::ComputePipelineState* sortPrefixSumPipeline_ = nullptr;
    MTL::ComputePipelineState* sortScatterPipeline_ = nullptr;

    // Splat input buffers (identical layout/prep to MetalSplatRenderer's).
    MTL::Buffer* posBuffer_ = nullptr;
    MTL::Buffer* rotBuffer_ = nullptr;
    MTL::Buffer* scaleBuffer_ = nullptr;
    MTL::Buffer* opacityBuffer_ = nullptr;
    MTL::Buffer* shBuffer_ = nullptr;

    // Projected output buffers (written by preprocess compute, read by sort/render).
    MTL::Buffer* projCenterBuffer_ = nullptr;
    MTL::Buffer* projConicBuffer_ = nullptr;
    MTL::Buffer* projColorBuffer_ = nullptr;
    MTL::Buffer* projMvBuffer_ = nullptr;
    MTL::Buffer* projDepthBuffer_ = nullptr;
    MTL::Buffer* visibleCountBuffer_ = nullptr;

    // Sort keys/values, ping-pong buffer B + radix-sort scratch (histogram,
    // partition sums) -- see splat_renderer.cpp's identical buffer set.
    MTL::Buffer* sortKeysBuffer_ = nullptr;       // A
    MTL::Buffer* sortKeysBBuffer_ = nullptr;      // B
    MTL::Buffer* sortedIndicesBuffer_ = nullptr;  // vals A
    MTL::Buffer* sortValsBBuffer_ = nullptr;      // vals B
    MTL::Buffer* histogramBuffer_ = nullptr;
    MTL::Buffer* partitionSumsBuffer_ = nullptr;
    uint32_t sortNumWg_ = 0;

    MTL::Buffer* prevPosBuffer_ = nullptr;

    // Camera state (identical to MetalSplatRenderer).
    glm::mat4 viewMatrix_{1.0f};
    glm::mat4 projMatrix_{1.0f};
    glm::mat4 projMatrixUnjittered_{1.0f};
    glm::vec3 camPos_{0.0f, 0.0f, 3.0f};
    float focalX_ = 256.0f;
    float focalY_ = 256.0f;
    float jitterX_ = 0.0f, jitterY_ = 0.0f;

    glm::mat4 prevViewMatrix_{1.0f};
    glm::mat4 prevProjMatrixUnjittered_{1.0f};
    bool firstMvFrame_ = true;

    std::vector<float> hostPositions_;

    // --- Dynamic splats: GPU morph-apply (same data as MetalSplatRenderer) ---
    SplatDynamics dynamics_;
    std::vector<SplatMorphSegment> morphSegments_;
    std::vector<uint32_t> everMovingIndices_;
    std::vector<float> basePositions_;
    std::vector<float> baseRotations_;
    std::vector<float> baseSH0_;
    float currentMorphTime_ = 0.0f;

    MTL::Buffer* baseGpuPosBuffer_ = nullptr;
    MTL::Buffer* baseGpuRotBuffer_ = nullptr;
    MTL::Buffer* baseGpuSh0Buffer_ = nullptr;
    MTL::Buffer* morphIndexBuffer_ = nullptr;
    MTL::Buffer* morphPosLoBuffer_ = nullptr;
    MTL::Buffer* morphRotLoBuffer_ = nullptr;
    MTL::Buffer* morphSh0LoBuffer_ = nullptr;
    MTL::Buffer* morphPosHiBuffer_ = nullptr;
    MTL::Buffer* morphRotHiBuffer_ = nullptr;
    MTL::Buffer* morphSh0HiBuffer_ = nullptr;
    std::vector<uint32_t> segmentOffsets_;
    std::vector<uint32_t> segmentCounts_;
    uint32_t morphTotalEntries_ = 0;

    // Per-stage GPU timing (mean-of-1, this render() call) -- for parity
    // reporting against MetalSplatRenderer's hand-written path.
    double lastPreprocessMs_ = 0.0;
    double lastSortMs_ = 0.0;
    double lastRenderMs_ = 0.0;

    // Helpers
    void createRenderTargets(MetalContext& ctx);
    void createPipelines(MetalContext& ctx);
    void createMorphPipeline(MetalContext& ctx);
    void createMorphBuffers(MetalContext& ctx);
    void createBuffers(MetalContext& ctx, const GaussianSplatData& data);
    void createSortPipelines(MetalContext& ctx);
    void createSortBuffers(MetalContext& ctx);
};
