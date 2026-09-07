#pragma once

// Metal splat renderer, luxc-compiled-pipeline backend (--splat-backend luxc,
// now the DEFAULT -- see metal_main.cpp's splatBackend comment). Status: AT
// PARITY with Vulkan on the juggle DLSS scene (frame 40, sort: view_depth,
// --camera-json): 51.3 dB colour PSNR vs. the gsplat reference (>=45 dB
// bar met), 83.7 dB colour PSNR vs. Vulkan's own output (i.e. visually
// identical), depth median rel. error 1.15e-7 vs. Vulkan (<1e-3 bar met by
// four orders of magnitude), motion vectors median abs diff ~4e-6 px vs.
// Vulkan (a handful -- 10 of 518400 -- boundary/tie-breaking pixels reach
// up to 0.04 px, just outside the <1e-3 px *max* bar, but the typical/
// median case is essentially exact).
//
// Root cause of the (much larger, ~28 dB) gap this backend originally
// shipped with: a missing Metal viewport Y-flip. Vulkan's NDC has +Y
// pointing down; Metal's native NDC has +Y pointing up; this SPIRV-Cross
// version has no automatic gl_Position-flip option (checked: no
// flip_vert_y or equivalent in spirv_msl.hpp), and the compiled vertex
// shader (splat_expander.py, confirmed via dumping its transpiled MSL with
// LUX_DUMP_MSL_DIR) builds gl_Position via the plain Vulkan
// `pixel = (ndc*0.5+0.5)*screen_size` convention with no shader-side
// correction of its own -- so the whole render was vertically mirrored
// relative to Vulkan until createPipelines's negative-height viewport flip
// was added (the standard MoltenVK trick), paired with
// kMetalYConvention=false (DlssIO::buildIntrinsicsProjection's
// metalYConvention flag exists specifically to compensate for
// MetalSplatRenderer's *own*, different, shader-side Y handling -- see its
// kMetalYConvention=true -- and must NOT be applied here) and a matching
// (non-negated) jitter Y sign. Two smaller bugs fixed earlier during
// bring-up, both real but not the dominant cause: expected-depth needed an
// RGBA32Float attachment with real hardware alpha blending (Apple Silicon
// DOES support float32 attachment blending, contrary to the hand-written
// path's own comment -- see kExpectedDepthChannels), and per-gaussian
// inputs (position/quaternion-xyzw/scale-log/opacity-logit/sh0) were
// verified byte-identical across all three backends via LUX_DEBUG_SPLAT_DUMP
// (ruling out any data-layout mismatch). RelaxedPrecision/`half` and
// MTLCompileOptions fast math were also checked and ruled out (see
// metal_shader_transpiler.cpp's safeMathCompileOptions and LUX_DUMP_MSL_DIR)
// -- neither was present/mattered; the Y-flip was the entire remaining gap.
//
// Runs the SAME compiled SPIR-V
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
    // `sharedCmdBuf` (default nullptr): when non-null, encodes the morph
    // compute dispatch into THAT command buffer instead of beginning,
    // committing, and blocking on a dedicated one of its own -- the caller
    // is then responsible for committing/waiting and must NOT rely on
    // hostPositions_/posBuffer_->contents() being valid until it does (no
    // CPU readback happens in this mode). This is how render() itself
    // could fold "evaluate this frame's morph" into its own single
    // per-frame command buffer (mirroring splat_renderer.cpp's Vulkan
    // render(), which already dispatches its morph this way) instead of
    // paying a separate CPU<->GPU round trip -- available for a continuous
    // per-frame caller (e.g. playground_ios/SplatView.mm) to adopt; not
    // used internally by this class yet since render() currently expects
    // posBuffer_/rotBuffer_/shBuffer_ to already hold the current frame's
    // morphed data on entry (the caller calls plain setMorphTime(seconds)
    // first). With sharedCmdBuf null (the default), behavior is unchanged:
    // a fully self-contained, blocking call.
    void setMorphTime(float seconds, MTL::CommandBuffer* sharedCmdBuf = nullptr);
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
    // Seeds prevPosBuffer_ with a real (non-current-time) morph evaluation.
    // Needed ONLY on the very first frame (no render() call has happened
    // yet to establish "prev = last frame's current" via its own GPU
    // ping-pong copy -- see render()'s posBuffer_->prevPosBuffer_ blit) or
    // for an explicit one-shot reseed, such as the headless CLI's
    // --time-prev/--frame-prev flags (called exactly once, before the
    // first render()). Both of those cases are characterized by
    // firstMvFrame_ still being true when this is called, which is what
    // the guard below keys off. After the first render() call,
    // prevPosBuffer_ is ALREADY correctly refreshed every frame by
    // render()'s own blit (no CPU readback, no extra command buffer) --
    // so a continuous per-frame caller that (redundantly, and previously
    // unconditionally) calls this alongside its own per-frame
    // setMorphTime(tCur) now pays nothing beyond the one-time first-frame
    // cost. Before this guard, an unconditional per-frame call here cost
    // TWO extra full blocking setMorphTime() GPU round trips (dispatch +
    // waitUntilCompleted, once for prevTimeSeconds, once to restore
    // currentTime) on top of the caller's own per-frame setMorphTime(tCur)
    // -- three morph evaluations per frame instead of one, ~30ms of a
    // 58ms frame measured on M1 iPad (playground_ios/SplatView.mm's
    // continuous rendering loop).
    void seedPreviousMorphTime(float prevTimeSeconds) {
        if (!dynamics_.has_motion || !firstMvFrame_) return;
        prevViewMatrix_ = viewMatrix_;
        prevProjMatrixUnjittered_ = projMatrixUnjittered_;
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
    // Packed into getExpectedDepthTexture()'s .g channel -- no separate
    // texture/attachment (see SPECIFICATION.md 12.8's foreground_coverage entry).
    bool hasForegroundCoverage() const { return hasForegroundCoverage_; }
    void setJitter(float jitterXPixels, float jitterYPixels);

    MTL::Texture* getMotionTexture() const { return motionTarget_; }
    MTL::Texture* getExpectedDepthTexture() const { return expectedDepthTarget_; }
    // RGBA32Float here (4 floats/pixel: x=depth*alpha, y/z unused, w=alpha)
    // vs. MetalSplatRenderer's RG32Float (2 floats/pixel) -- this backend's
    // fragment shader needs the real 4th-component alpha for hardware
    // blending (commit 1445ec3's out_depth fix); the hand-written path
    // blends manually and never needed it.
    static constexpr uint32_t kExpectedDepthChannels = 4;

    // The luxc-compiled vertex shader (splat_expander.py's
    // _build_vertex_body) maps screen.y = (ndc.y*0.5+0.5)*H directly --
    // Vulkan's own convention, no extra flip outside the projection matrix
    // (confirmed by dumping the transpiled MSL: LUX_DUMP_MSL_DIR=<dir>,
    // see the vertex stage's `_119` computation) -- so DlssIO::
    // buildIntrinsicsProjection must be called with metalYConvention=false
    // here, NOT MetalSplatRenderer's true (that flag exists specifically to
    // compensate for the *hand-written* shader's own extra Y flip, which
    // this transpiled-from-Vulkan shader doesn't have). Using true here was
    // a real, confirmed bug: it negated the whole projection Y row,
    // vertically mis-projecting every splat and (found empirically) fully
    // explaining this backend's colour-parity gap vs. Vulkan on the juggle
    // DLSS scene -- see the class comment's revised numbers.
    static constexpr bool kMetalYConvention = false;

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
    const float* debugScaleBufferPtr() const { return static_cast<const float*>(scaleBuffer_->contents()); }
    const float* debugOpacityBufferPtr() const { return static_cast<const float*>(opacityBuffer_->contents()); }

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
    bool hasForegroundCoverage_ = false;

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
    // foreground_coverage: splat_foreground (input, uploaded once from
    // GaussianSplatData::foreground) and projected_foreground (output).
    MTL::Buffer* foregroundBuffer_ = nullptr;
    MTL::Buffer* projForegroundBuffer_ = nullptr;

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
    // prev_camera_mats[0]=proj_matrix_unjittered, [1]=prev_view_proj_unjittered
    // (128 bytes) -- moved out of push constants into a storage buffer;
    // see splat_renderer.h's identical field and splat_expander.py's "Why
    // not push constants" comment (Mali-G715's 256-byte
    // maxPushConstantsSize vs. the 304-byte block these two mat4 fields
    // used to make; MoltenVK on Apple Silicon reports 4096, so this bug
    // never manifested on the Metal backend, but the fix is shared since
    // both backends transpile/consume the same compiled shader).
    MTL::Buffer* prevCameraBuffer_ = nullptr;

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
