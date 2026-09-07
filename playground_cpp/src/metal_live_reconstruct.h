#pragma once

// Shared live-inference driver: owns the whole validated proxy-render ->
// net-input-assembly -> MPSGraph-UNet -> reconstruct-with-memory chain
// (formerly playground_ios-only: SplatView.mm's Reconstruction-mode branch
// directly owned a second MetalSplatLuxcRenderer + NetInputAssembly +
// MPSGraphUNet + ReconstructPass) and encodes one frame's worth of work into
// a SINGLE caller-supplied MTL::CommandBuffer* with no host waits between
// stages -- see encodeFrame()'s comment.
//
// Shared between playground_ios/Source/SplatView.mm (the live iOS demo,
// which now only owns the display link/layer/mode switch/HUD/dumps) and
// metal_main.cpp's `--live-bench` CLI mode (headless, Mac) -- both hosts
// point this at the same mobiledlss/demo/ios_assets/exported/ asset bundle
// and the same examples/gaussian_splat_dlss compiled pipeline, so the Mac
// benchmark measures exactly what the iPad runs.

#include <Metal/Metal.hpp>
#include <glm/glm.hpp>
#include <cstdint>
#include <memory>
#include <string>

#include "metal_splat_luxc_renderer.h"
#include "net_input_assembly.h"
#include "mps_graph_unet.h"
#include "live_reconstruct_pass.h"

class MetalContext;
struct GaussianSplatData;

class MetalLiveReconstruct {
public:
    ~MetalLiveReconstruct();

    struct InitParams {
        std::string shaderBase;          // e.g. "examples/gaussian_splat_dlss"
        uint32_t proxyW = 0, proxyH = 0;
        uint32_t targetW = 0, targetH = 0;
        uint32_t paramStride = 2;
        uint32_t hiddenChannels = 8;
        std::string textureNpyPath;      // exported/texture.npy
        std::string bgSphereNpyPath;     // exported/bg_sphere.npy
        std::string unetWeightsBinPath;  // exported/unet_weights.fp16.bin
        std::string unetLayersTxtPath;   // exported/unet_weights.layers.txt
        std::string memoryHeadNpzPath;   // exported/memory_head.npz
        // Sort scheduling (bench/lux_perf_ablation.md) -- see
        // MetalSplatLuxcRenderer::setSortSchedule()'s own comment. Default
        // (1, 0) is exactness-preserving (every-frame sort, matches a
        // from-scratch parity/PSNR run); live callers (SplatView.mm,
        // --live-bench) opt into (4, 2.0f).
        uint32_t sortEveryNFrames = 1;
        float sortViewThresholdDeg = 0.0f;
    };

    void init(MetalContext& ctx, const GaussianSplatData& proxySceneData, const InitParams& params);

    uint32_t getNetW() const { return netW_; }
    uint32_t getNetH() const { return netH_; }
    uint32_t getProxyW() const { return proxyW_; }
    uint32_t getProxyH() const { return proxyH_; }
    uint32_t getTargetW() const { return targetW_; }
    uint32_t getTargetH() const { return targetH_; }

    MetalSplatLuxcRenderer& proxyRenderer() { return *splatRProxy_; }
    NetInputAssembly& netInput() { return netInput_; }
    MPSGraphUNet& unet() { return unet_; }
    LiveReconstructPass& reconstruct() { return reconstruct_; }
    MTL::Buffer* getUnetOutputBuffer() const { return unetOutputBuffer_; }
    MTL::Texture* getReconOutputTexture() const { return reconOutputTex_; }

    // Re-entry after skipping frames (mirrors NetInputAssembly::reset()/
    // LiveReconstructPass::reset()): forces the next encodeFrame() call to
    // be treated as the very first one again (full disocclusion, zero
    // hidden/warp/blend history).
    void resetHistory() {
        netInput_.reset();
        reconstruct_.reset();
    }

    // One camera basis -- eye + right/up/forward axes, GL view/proj
    // matrices (unjittered; jitter is applied separately, see
    // FrameInputs::jitter*), and focal lengths. Shape matches
    // LiveOrbitCamera::Result (orbit_camera.h) but is kept independent of it so
    // a non-orbit camera source could drive this class too.
    struct CameraFrame {
        glm::vec3 eye{0.0f}, r{1.0f, 0.0f, 0.0f}, u{0.0f, 1.0f, 0.0f}, f{0.0f, 0.0f, 1.0f};
        glm::mat4 viewGl{1.0f};
        glm::mat4 proj{1.0f};
        float fx = 0.0f, fy = 0.0f;
    };

    struct FrameInputs {
        CameraFrame proxyCur;    // this frame's proxy-res camera
        CameraFrame proxyPrev;   // previous frame's proxy-res camera (motion vectors)
        CameraFrame targetCur;   // this frame's target-res camera (reconstruct's bg-sphere UV)
        float morphTimeCur = 0.0f;
        float morphTimePrev = 0.0f;
        float jitterProxyX = 0.0f, jitterProxyY = 0.0f;    // proxy-pixel units
        float jitterTargetX = 0.0f, jitterTargetY = 0.0f;  // target-pixel units
    };

    // Encodes proxy render (incl. morph) + net input assembly + UNet +
    // reconstruct into `cmdBuf`, writing the final image into
    // getReconOutputTexture() (target res, RGBA16Float). NO commit()/
    // waitUntilCompleted() of its own -- resource hazards between these
    // encoders are tracked by Metal automatically within one command buffer
    // (encoder submission order), the same guarantee metal_splat_luxc_
    // renderer.cpp's own single-command-buffer render()/encodeFrame()
    // already relies on. Returns the command buffer the caller must
    // actually commit: usually the same `cmdBuf` passed in, but see
    // MPSGraphUNet::encode()'s comment -- MPSGraph may internally swap the
    // live MTLCommandBuffer out from under it. Callers must use the
    // RETURNED pointer, not their own `cmdBuf`, for anything encoded after
    // this call (e.g. a display blit of getReconOutputTexture()) and for the
    // final commit/wait (or commit + completion-handler semaphore signal).
    //
    // Ownership: the returned pointer carries an EXTRA retain (see
    // MPSGraphUNet::encode()'s doc comment) that the CALLER now owns and
    // must balance with `->release()` once fully done with the frame's
    // command buffer (after commit()+waitUntilCompleted(), or after the
    // completion handler fires in a multiple-frames-in-flight setup).
    // Skipping this leaks one command buffer object per frame; getting it
    // backwards (releasing before commit) is a use-after-free.
    MTL::CommandBuffer* encodeFrame(MetalContext& ctx, MTL::CommandBuffer* cmdBuf, const FrameInputs& in);

    // Real total GPU busy time (ms) across a possible MPSGraph
    // commitAndContinue split: `before` is the command buffer the caller
    // originally passed to encodeFrame()/MPSGraphUNet::encode(), `after` is
    // the pointer that call returned. When MPSGraph's UNet is large enough
    // to trigger an internal commitAndContinue, `after != before` -- the
    // REAL heavy compute may have been submitted (and already committed,
    // by MPSGraph itself) on `before` before the swap, so `after`'s own
    // GPUStartTime()/GPUEndTime() alone can radically under-report the
    // frame's true GPU cost (measured: ~0.7ms fused-total vs. a ~3ms
    // proxy+input+recon stage-split sum, i.e. the ENTIRE UNet cost missing
    // -- exactly this bug, caught via --live-bench on an M4 Max). Call this
    // AFTER `after`'s own commit()+waitUntilCompleted() -- `before`, if
    // different, was already committed internally by MPSGraph and is safe
    // to wait on and read at that point (command buffers on one queue
    // complete in commit order, so waiting on the later one guarantees the
    // earlier one is done too).
    static double gpuMsAcrossPossibleSplit(MTL::CommandBuffer* before, MTL::CommandBuffer* after);

    // --- Diagnostic per-stage GPU timing -- mirrors MetalSplatLuxcRenderer::
    // renderProfiled()'s own precedent: a deliberate near-duplicate of
    // encodeFrame() rather than a shared refactor, since splitting stages
    // into their own command buffers means real GPUStartTime()/GPUEndTime()
    // readbacks (and the CPU<->GPU round trips those force) in between --
    // the fused production path (encodeFrame()) never pays these. Runs
    // proxy/input-assembly/UNet/reconstruct each in their own command
    // buffer, waiting on each, and reports real per-stage GPU ms plus the
    // sum. Debug/bench use only (metal_main.cpp's `--live-bench`, and an
    // on-device HUD toggle) -- mutates the exact same history state a normal
    // encodeFrame() call would (same ping-pong indices advance), so don't
    // interleave the two calls without an explicit resetHistory() in
    // between if exact reproducibility matters.
    struct ProfiledResult {
        double proxyGpuMs = 0.0;
        double inputGpuMs = 0.0;
        double netGpuMs = 0.0;
        double reconGpuMs = 0.0;
        double totalGpuMs = 0.0;
        int numCommandBuffers = 0;
    };
    ProfiledResult encodeFrameProfiled(MetalContext& ctx, const FrameInputs& in);

private:
    uint32_t proxyW_ = 0, proxyH_ = 0, targetW_ = 0, targetH_ = 0, netW_ = 0, netH_ = 0;
    uint32_t nBlend_ = 3;

    std::unique_ptr<MetalSplatLuxcRenderer> splatRProxy_;
    NetInputAssembly netInput_;
    MPSGraphUNet unet_;
    LiveReconstructPass reconstruct_;

    MTL::Buffer* unetOutputBuffer_ = nullptr;
    MTL::Texture* reconOutputTex_ = nullptr;

    void applyCameraAndJitter(const FrameInputs& in);
};
