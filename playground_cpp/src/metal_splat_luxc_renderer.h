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
#include <algorithm>

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
    // Encodes the SAME preprocess+sort+draw sequence render() does, but into
    // the CALLER's own `cmdBuf` (no beginCommandBuffer()/commit()/
    // waitUntilCompleted() -- the caller owns the buffer's lifetime and
    // hazard tracking across the rest of its own frame). render() itself is
    // just `cmdBuf = ctx.beginCommandBuffer(); encodeFrame(ctx, cmdBuf);
    // cmdBuf->commit(); cmdBuf->waitUntilCompleted();` plus its own GPU-timing
    // readback -- a verbatim factor-out, not a behavioral change, so
    // render()'s own parity guarantees (tests/mobiledlss's Vulkan/gsplat
    // comparison) are untouched. Added for a continuous per-frame caller
    // (playground_cpp/src/metal_live_reconstruct.*) that fuses this splat
    // pass into the SAME command buffer as its input-assembly/UNet/
    // reconstruct passes, eliminating the extra CPU<->GPU round trip a
    // dedicated render() call would otherwise force between stages every
    // frame. lastPreprocessMs_/lastSortMs_/lastRenderMs_ are still updated
    // (CPU encode-time estimates, as documented on getLastGpuTotalMs());
    // lastGpuTotalMs_ is NOT (it needs cmdBuf's post-commit GPUStartTime/
    // GPUEndTime, which only the buffer's actual committer can read) -- read
    // `cmdBuf->GPUStartTime()/GPUEndTime()` after your own commit instead.
    void encodeFrame(MetalContext& ctx, MTL::CommandBuffer* cmdBuf);
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
    // True when compiled with foreground_coverage: true -- a SEPARATE
    // texture/attachment, getFgTexture() below (see its comment for why it
    // can't share getAuxTexture()'s lanes).
    bool hasForegroundCoverage() const { return hasForegroundCoverage_; }

    // Colour attachment format experiment (port of splat_renderer.h's
    // Vulkan getColorFormat()/setColorFormat8BitExperiment() -- see that
    // header's comment for the full history/rationale: RGBA16Float was
    // adopted over 8-bit specifically because hundreds of overlapping
    // low-alpha blended fragments accumulate 1/255 rounding at 8 bits;
    // bench/lux_perf_ablation.md's draw-stage ablation re-tested it
    // scoped to colour-only (no DLSS aux) pipelines and measured a real
    // ~8.6% draw win on Mali but a PSNR estimate below this task's >=50dB
    // gate, so it stayed opt-in there too). RGBA16Float always for any
    // DLSS-aux pipeline (motion_vectors/expected_depth/
    // foreground_coverage); RGBA8Unorm ONLY for colour-only, and ONLY
    // when explicitly enabled via setColorFormat8BitExperiment(). Must be
    // set before init() -- createRenderTargets()/createPipelines() both
    // read it once at construction time.
    MTL::PixelFormat getColorFormat() const {
        if (hasMotionVectors_ || hasExpectedDepth_ || hasForegroundCoverage_) {
            return MTL::PixelFormatRGBA16Float;
        }
        return colorFormat8Bit_ ? MTL::PixelFormatRGBA8Unorm : MTL::PixelFormatRGBA16Float;
    }
    // Opt-in switch (default false = unchanged RGBA16Float colour-only
    // behavior).
    void setColorFormat8BitExperiment(bool enabled) { colorFormat8Bit_ = enabled; }

    // "aux_precision: half" was set on the compiled splat block -- see
    // getAuxFormat() (out_fg's format is independent of this flag -- see
    // getFgFormat()).
    bool auxPrecisionHalf() const { return auxPrecisionHalf_; }
    void setJitter(float jitterXPixels, float jitterYPixels);

    // Packed DLSS aux texture (bench/lux_perf_ablation.md task 2): ONE
    // texture carrying (mv.x*alpha, mv.y*alpha, depth*alpha, alpha) --
    // replaces the earlier two-texture (motionTarget_ RGBA32Float,
    // expectedDepthTarget_ RGBA32Float, each with an always-0 `.z`) design
    // with one that packs 3 real values with ZERO wasted lanes. `.w` MUST
    // be the genuine per-fragment alpha, not a repurposed data channel --
    // see luxc/expansion/splat_expander.py's out_aux comment for why:
    // Metal's fixed-function alpha blend (same semantics as Vulkan's
    // SRC_ALPHA/ONE_MINUS_SRC_ALPHA here) reads "source alpha" from THIS
    // texture's own 4th output component specifically, so anything else
    // there corrupts the GPU-side blend accumulation itself (measured: a
    // real 10-50x MV/depth error on the juggle DLSS scene from an earlier
    // version of this design that packed `fg*alpha` into `.w` instead and
    // tried to un-premultiply via getOutputTexture()'s alpha afterward --
    // NOT a valid precision tradeoff, a genuine correctness bug). Format
    // ("float", default, RGBA32Float vs "half", RGBA16Float) is a
    // compile-time host hint (`aux_precision:`) -- same vec4 shader output
    // either way, always 4 channels regardless of precision (unlike an
    // earlier version of this option, which also shrank getFgTexture() to
    // 2 channels under "half" -- see getFgTexture()'s comment for why that
    // was reverted). Valid whenever hasMotionVectors() ||
    // hasExpectedDepth(). "half" MEASURED (bench/
    // lux_perf_ablation.md): MV median error 0.012-0.016px (gate
    // <=0.01px) and depth relative median error ~5.4e-3 (gate <=1e-3),
    // both a real, milder-than-the-alpha-bug FAIL -- premultiplied mv/depth
    // values are not magnitude-bounded like color's own [0,1] channels, so
    // half-float blend-accumulation rounding over many overlapping splats
    // still exceeds this task's tight gates. Kept opt-in only, default
    // stays "float".
    MTL::Texture* getAuxTexture() const { return auxTarget_; }
    static constexpr uint32_t kAuxChannels = 4;
    MTL::PixelFormat getAuxFormat() const {
        return auxPrecisionHalf_ ? MTL::PixelFormatRGBA16Float : MTL::PixelFormatRGBA32Float;
    }

    // Second, smaller texture for foreground_coverage (only allocated when
    // hasForegroundCoverage()): (fg*alpha, 0, 0, alpha). Can't share
    // getAuxTexture()'s lanes -- those 3 non-alpha lanes are already
    // mv.xy/depth, and `.w` must independently stay genuine alpha in
    // EVERY blended texture, not just one (see getAuxTexture()'s comment).
    // ALWAYS RGBA16Float, 4 channels, regardless of aux_precision -- fg
    // (like alpha itself) is bounded to [0,1], the same
    // magnitude-boundedness that keeps color's own RGBA16Float blend
    // accumulation safe applies here too. Do NOT drop this to RG16Float
    // (2 channels) under "aux_precision: half": that was tried and
    // MEASURED to FAIL the parity gate hard (see bench/lux_perf_ablation.md):
    // a full 0-to-1 flip on ~1% of pixels, consistent with a 2-component
    // format not carrying a real stored alpha component for the
    // fixed-function blend to read (the blend factor falls back to a
    // constant instead of reading this texture's own `.w`, breaking the
    // decay term for overlapping fragments) -- a real correctness bug,
    // not a precision tradeoff, unlike getAuxFormat()'s "half" (still 4
    // channels, still a real `.w`, only ordinary half-float rounding
    // error). "aux_precision: half" therefore only changes
    // getAuxFormat(); out_fg's format/channel count is independent of it.
    MTL::Texture* getFgTexture() const { return fgTarget_; }
    uint32_t getFgChannels() const { return 4; }
    MTL::PixelFormat getFgFormat() const { return MTL::PixelFormatRGBA16Float; }

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

    // --- Sort scheduling (perf; port of splat_renderer.h's Vulkan
    // setSortSchedule() -- see that header's comment for the full
    // rationale). Metal previously had NO equivalent: render() always ran
    // the full 4-pass GPU radix sort every frame regardless of whether the
    // camera view had actually changed enough to need a fresh
    // back-to-front order. Default (everyNFrames=1, viewChangeThresholdDeg
    // =0) is UNCHANGED, exactness-preserving behavior -- a single render()
    // call always has framesSinceSort_==0 (never skips), so headless
    // one-shot renders, tests, and the mobiledlss parity checks are
    // unaffected. Opt in via setSortSchedule() for a continuous multi-frame
    // caller (interactive/live rendering, or --bench below) willing to
    // trade briefly-stale blend order for amortized sort cost.
    void setSortSchedule(uint32_t everyNFrames, float viewChangeThresholdDeg) {
        sortEveryNFrames_ = std::max<uint32_t>(1, everyNFrames);
        sortViewThresholdDeg_ = viewChangeThresholdDeg;
    }

    // Real GPU-side timing of the most recent render() call, from
    // MTLCommandBuffer::GPUStartTime()/GPUEndTime() (actual GPU busy time,
    // not the CPU-side encode-and-wait wall clock that lastPreprocessMs_/
    // lastSortMs_/lastRenderMs_ below approximate) -- see render()'s own
    // comment for why those three are CPU encode-time estimates, not real
    // per-stage GPU numbers, in the (default) fused-single-command-buffer
    // path. This is the number directly comparable to MetalSplatter's own
    // `commandBuffer.gpuStartTime/gpuEndTime` methodology
    // (bench/lux_perf_ablation.md in mobiledlss).
    double getLastGpuTotalMs() const { return lastGpuTotalMs_; }

    // --bench-only stage-split GPU timing: encodes preprocess/sort/draw
    // into THREE separate command buffers (instead of render()'s one
    // fused buffer), waiting on each before starting the next, so
    // GPUStartTime()/GPUEndTime() of each buffer gives a REAL per-stage GPU
    // number instead of the fused path's CPU-encode-time estimate. This
    // adds real CPU<->GPU round-trip sync overhead between stages that the
    // fused production render() path does not pay -- use only for
    // diagnostic stage-breakdown reporting, never for the headline
    // total-vs-MetalSplatter comparison (use getLastGpuTotalMs() after a
    // plain render() call for that).
    void renderProfiled(MetalContext& ctx, double* preprocessGpuMs, double* sortGpuMs,
                         double* drawGpuMs, double* totalGpuMs);

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
    // Packed RGBA32Float attachment (mv.x*a, mv.y*a, depth*a, a) -- see
    // getAuxTexture()'s comment (NOT RGBA16Float -- measured precision
    // failure).
    MTL::Texture* auxTarget_ = nullptr;
    // Second, smaller RGBA16Float attachment (fg*a, 0, 0, a) -- only
    // allocated when hasForegroundCoverage_ -- see getFgTexture()'s comment.
    MTL::Texture* fgTarget_ = nullptr;

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
    bool auxPrecisionHalf_ = false;
    // Colour attachment format experiment -- see getColorFormat()/
    // setColorFormat8BitExperiment() above. Default false = unchanged
    // RGBA16Float colour-only behavior.
    bool colorFormat8Bit_ = false;

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
    // Oriented-quad half-axis vectors (major.xy, minor.xy) -- see
    // splat_expander.py's "Oriented quads" note.
    MTL::Buffer* projAxesBuffer_ = nullptr;
    // (t_major, t_minor, 0, 0) per splat -- replaces the old inverse-2D-
    // covariance "conic" this buffer held; see splat_expander.py's
    // "Isotropic fragment evaluation" note.
    MTL::Buffer* projExtentBuffer_ = nullptr;
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
    // Real total GPU time (GPUEndTime-GPUStartTime) of the fused command
    // buffer -- see getLastGpuTotalMs()'s header comment.
    double lastGpuTotalMs_ = 0.0;

    // Sort scheduling state (see setSortSchedule() above) -- mirrors
    // splat_renderer.h's identical Vulkan fields exactly.
    uint32_t sortEveryNFrames_ = 1;
    float sortViewThresholdDeg_ = 0.0f;
    uint32_t framesSinceSort_ = 0;
    glm::vec3 lastSortedViewDir_{0.0f, 0.0f, -1.0f};
    bool hasLastSortedView_ = false;

    // Morph-reset scope (bench/lux_perf_ablation.md "Root cause #1", ported
    // from splat_renderer.h/.cpp's dispatchMorph() -- see setMorphTime()'s
    // updated comment for the full rationale). -1 = no segment applied yet
    // (skip the reset entirely on the very first call: posBuffer_/
    // rotBuffer_/shBuffer_ already hold base values from createBuffers()).
    int lastAppliedSegment_ = -1;

    // Helpers
    void createRenderTargets(MetalContext& ctx);
    void createPipelines(MetalContext& ctx);
    void createMorphPipeline(MetalContext& ctx);
    void createMorphBuffers(MetalContext& ctx);
    void createBuffers(MetalContext& ctx, const GaussianSplatData& data);
    void createSortPipelines(MetalContext& ctx);
    void createSortBuffers(MetalContext& ctx);
};
