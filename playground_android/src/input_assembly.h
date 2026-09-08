#pragma once

#include <vulkan/vulkan.h>
#include <cstdint>
#include <string>
#include <vector>

struct VulkanContext;

// Stage 3 of the mobile-DLSS Android live demo (docs/rendering-engines.md):
// assembles ParamPredUNet's 26-channel NHWC input tensor
// (mobiledlss/train/model.py::build_input) on-GPU from the proxy-res
// (480x270) DLSS attachments the proxy SplatRenderer produces, as two
// hand-written GLSL compute shaders (playground_android/shaders_glsl/
// input_assembly_{unpremul_depth,assemble}.comp, compiled with the NDK's
// glslc -- NOT luxc; this is new, not-yet-declarative code per the task
// brief, mirroring how docs/rendering-engines.md's fused-GPU-compute UNet
// kernels started as hand-written `compute{}` blocks before any `network`
// syntax existed).
//
// Deliberately does NOT bind SplatRenderer's colour/motion/expected-depth
// VkImages directly as sampled/storage images in the compute descriptor
// sets -- those images are only created with COLOR_ATTACHMENT_BIT |
// TRANSFER_SRC_BIT | TRANSFER_DST_BIT usage (splat_renderer.cpp, a reused-
// by-reference file this task must not modify), not SAMPLED_BIT/STORAGE_BIT.
// Instead, each frame's raw texel bytes are pulled off the GPU with a plain
// vkCmdCopyImageToBuffer (the images are already left in
// TRANSFER_SRC_OPTIMAL after SplatRenderer::render() -- see
// createRenderPass's finalLayout -- so no extra barrier is needed) into
// persistent storage buffers, and both compute passes below operate on
// buffers only -- the exact same "everything is a storage buffer" style
// playground_cpp/src/reconstruct_runner.cpp already uses for the
// reconstruct passes. Sampling that the Metal reference (NetInputAssembly.mm)
// did with real hardware texture units (bg-scene-texture bilinear,
// zero-padded previous-depth bilinear) is done as plain manual buffer
// arithmetic in GLSL instead (see the .comp source for the derivation of
// each).
class InputAssembly {
public:
    ~InputAssembly();

    // netW/netH: mobiledlss.train.export._PaddedExportModel's convention --
    // proxy h/w replicate-padded up to a multiple of 8*paramStride before
    // box-pooling, so the network always runs at a multiple-of-8 resolution
    // (proxy 480x270, paramStride=2 -> net 240x136, not "true" 240x135).
    // auxFormat/fgFormat: query these from the SAME SplatRenderer whose
    // images run() will be given (SplatRenderer::getAuxFormat()/
    // getFgFormat()) -- NOT hardcoded here, since lux 5d5630c made
    // `aux_precision: half` (RGBA16F out_aux) the default, superseding the
    // RGBA32F this class originally assumed (getAuxFormat() is itself a
    // runtime host query on the SplatRenderer side, not a compile-time
    // constant -- see its comment). Only RGBA32F/RGBA16F are supported
    // (bytesPerTexel() throws otherwise); the GLSL side
    // (shaders_glsl/input_assembly_*.comp's readAux()) resolves the actual
    // byte layout per-dispatch from the auxIsHalf push-constant flag run()
    // derives from auxFormat here, so passing either supported format is
    // genuinely handled, not just sized correctly and silently
    // misinterpreted. fgFormat is effectively always RGBA16F (see
    // getFgFormat()'s comment for why that's independent of aux_precision)
    // but still queried, not assumed, for the same reason.
    void init(VulkanContext& ctx, const std::string& textureNpyPath,
              const std::string& bgSphereNpyPath, uint32_t proxyW, uint32_t proxyH,
              uint32_t paramStride, uint32_t hiddenChannels,
              const std::string& shaderDir, VkFormat auxFormat, VkFormat fgFormat);

    uint32_t getNetW() const { return netW_; }
    uint32_t getNetH() const { return netH_; }
    static constexpr uint32_t kNonHiddenChannels = 18;
    uint32_t getChannels() const { return kNonHiddenChannels + hiddenChannels_; }

    // fg channel source (see NetInputAssembly.h's identical enum/comment):
    // NEVER proxy alpha -- measured on iOS to collapse the model
    // 36.4->24.3dB (the net reads "alpha coverage" as "moving actor here,
    // distrust history/memory" everywhere alpha>0, not just where the
    // actor truly is). run() always requests kFgSourceExpectedDepthG now
    // that gaussian_splat_dlss's foreground_coverage output is wired
    // (unpremul_depth.comp's CurFg buffer -- originally read via out_depth's
    // .g channel, now via the separate out_fg attachment lux commit 6ed0334
    // introduced; the enum name predates that repacking and is kept
    // unchanged since its MEANING -- "real per-pixel fg, not constant 0" --
    // hasn't -- see input_assembly.cpp/the .comp sources); kFgSourceConstantZero
    // is kept only as the pre-foreground-coverage fallback value.
    enum FgSource : uint32_t { kFgSourceConstantZero = 0, kFgSourceExpectedDepthG = 1 };

    // Task 2 (Reconstruction-mode time budget) timing breakdown for one
    // run() call -- CPU wall-clock (readbackMs/computeMs) plus a real GPU
    // timestamp span (gpuMs). GPU-pipelining task (docs/rendering-engines.md,
    // "restructure the Reconstruction frame into a GPU-pipelined chain"):
    // run() used to be 5 separate beginSingleTimeCommands()/
    // endSingleTimeCommands() round trips (3x copyImageToBuffer + 2x
    // dispatchOne, i.e. 5 full queue drains) -- readbackMs/computeMs's own
    // header comments describe that old shape. It is now ONE command buffer
    // (3 copies -> barrier -> unpremul dispatch -> barrier -> assemble
    // dispatch, same pattern as reconstruct_pass.cpp's
    // recordDispatchBarriered/GpuTimestampBracket) submitted once, so
    // readbackMs/computeMs are now just CPU-side sub-splits of that ONE
    // wall-clock span (recorded via std::chrono either side of the
    // copy-record calls vs. the dispatch-record calls, not two separate
    // drains any more) and gpuMs is the real GPU-timestamp delta
    // (VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT..BOTTOM_OF_PIPE_BIT) bracketing the
    // whole merged command buffer -- mirrors MetalLiveReconstruct::
    // ProfiledResult::inputGpuMs on the Mac/iOS reference.
    struct Timings {
        double readbackMs = 0.0;  // CPU-side record time for the 3 copyImageToBuffer calls (no longer a separate drain)
        double computeMs = 0.0;   // CPU-side record time for the unpremul + assemble dispatches, plus the one shared submit+wait
        double gpuMs = 0.0;       // real GPU timestamp delta across the whole merged command buffer (copies + both dispatches)
    };

    // Runs both passes for one frame: (1) un-premultiply this frame's proxy
    // depth+fg into the ping-pong/current buffers, (2) assemble the packed
    // fp32 NHWC tensor into getOutputHostPtr(). `colorImage`/`auxImage`/
    // `fgImage` are the proxy SplatRenderer's raw attachments (already in
    // TRANSFER_SRC_OPTIMAL after render()) -- lux commit 6ed0334 packed the
    // old separate motion/depth attachments into one `auxImage`
    // (SplatRenderer::getAuxImage(), RGBA32F: x=mv.x*a, y=mv.y*a, z=depth*a,
    // w=a) plus a separate `fgImage` (getFgImage(), RGBA16F: x=fg*a, w=a);
    // `hiddenIn` may be null (zero hidden state -- Stage 3/4 validation
    // config; Stage 5 feeds the real recurrent state). eye/rAxis/uAxis/fAxis
    // are world-space camera position/right/up(down)/forward axes (see
    // android_main.cpp's buildCvViewRowMajor -- its r/u/f ARE these axes
    // directly, no camera-to-world matrix construction needed). `outTimings`
    // (optional) receives this call's breakdown for task 2's profiling.
    void run(VulkanContext& ctx, VkImage colorImage, VkImage auxImage, VkImage fgImage,
             uint32_t proxyW, uint32_t proxyH, const float* hiddenIn,
             float eyeX, float eyeY, float eyeZ,
             float rX, float rY, float rZ, float uX, float uY, float uZ,
             float fX, float fY, float fZ,
             float fx, float fy, float cx, float cy,
             float jitterProxyX, float jitterProxyY, Timings* outTimings = nullptr);

    // GPU-pipelining task (docs/rendering-engines.md, "restructure the
    // Reconstruction frame into a GPU-pipelined chain", goal 2): same work
    // as run() (3 copies -> barrier -> unpremul -> barrier -> assemble),
    // recorded into a command buffer the CALLER already began and owns --
    // no vkQueueSubmit/vkQueueWaitIdle inside. Lets a caller fuse this with
    // the proxy SplatRenderer's own encodeFrame() (whose colorImage/
    // auxImage/fgImage this reads) into ONE submit with one fence, instead
    // of each doing its own full queue drain. `outTimings->readbackMs`/
    // `computeMs` (CPU record time for their respective sections) are
    // filled in as usual; `gpuMs` is left at 0 -- call
    // fetchGpuTimingsAfterFence() once the caller has waited for this
    // command buffer's completion to fill it in.
    void encode(VulkanContext& ctx, VkCommandBuffer cmd, VkImage colorImage, VkImage auxImage, VkImage fgImage,
                uint32_t proxyW, uint32_t proxyH, const float* hiddenIn,
                float eyeX, float eyeY, float eyeZ,
                float rX, float rY, float rZ, float uX, float uY, float uZ,
                float fX, float fY, float fZ,
                float fx, float fy, float cx, float cy,
                float jitterProxyX, float jitterProxyY, Timings* outTimings = nullptr);

    // Reads back the GPU timestamp-query results written by the most
    // recent encode() call and writes the elapsed ms into *outGpuMs (left
    // untouched if outGpuMs is null). Caller must have already waited
    // (fence/vkQueueWaitIdle) for that command buffer's completion. run()
    // does this internally (it's already GPU-synchronous via its own
    // endSingleTimeCommands); encode() callers must call this themselves.
    void fetchGpuTimingsAfterFence(VulkanContext& ctx, double* outGpuMs);

    // Host-visible pointer to this frame's packed NHWC fp32 output
    // (netW*netH*getChannels() floats) -- valid immediately after run()
    // returns (all buffers are VMA_MEMORY_USAGE_CPU_TO_GPU / coherent, and
    // run() fully drains the queue via ctx.endSingleTimeCommands()).
    const float* getOutputHostPtr() const;
    void dumpToNpy(const std::string& path) const;

    // GPU-pipelining task: raw copied bytes off SplatRenderer::
    // getOutputImage()/getAuxImage() this frame's run() call just pulled
    // into bColorRaw/bAuxRaw (persistently host-mapped, same
    // VMA_MEMORY_USAGE_CPU_TO_GPU/coherent buffers run()'s own compute
    // passes read) -- valid immediately after run() returns, same
    // convention as getOutputHostPtr(). Exists so a caller that ALSO needs
    // these exact same image bytes for something else (android_main.cpp's
    // Reconstruction-mode FrameInputs::proxyColor/mvProxy, previously
    // built by its own separate readColorAndMvProxyCombined()
    // copyImageToBuffer + vkQueueWaitIdle round trip) can read them
    // straight out of run()'s own already-completed readback instead of
    // paying for a second GPU copy of the identical image -- see
    // android_main.cpp's readPremulColorAndUnpremulMvFromInputAssembly().
    // getRawColorHostPtr(): RGBA16_SFLOAT, 8 bytes/texel (SplatRenderer's
    // color attachment format is always RGBA16F). getRawAuxHostPtr():
    // RGBA32F (16B/texel) or RGBA16F (8B/texel) per isAuxHalf() -- matches
    // whatever VkFormat init()'s auxFormat param was given (query it from
    // the SAME SplatRenderer, e.g. getAuxFormat(), for these bytes to mean
    // anything).
    const void* getRawColorHostPtr() const;
    const void* getRawAuxHostPtr() const;
    bool isAuxHalf() const { return auxIsHalf_; }

    // Proxy-res unpremultiplied depth this frame just wrote / the frame
    // before it -- for Stage 5's target-resolution disocclusion (its own
    // separate computation from these, per NetInputAssembly.h's
    // getDepthWrittenThisFrame/getDepthFromPreviousFrame comment: "CALL
    // THESE AFTER run() returns" -- run() flips the ping-pong index at the
    // very end of each call, so these two already account for that).
    const float* getDepthWrittenThisFrame() const;
    const float* getDepthFromPreviousFrame() const;
    bool wasFirstFrame() const { return wasFirstFrame_; }

    uint32_t getTexChannels() const { return texChannels_; }

private:
    // Shared body of run()/encode() -- see input_assembly.cpp's comment on
    // the definition. Records the 3 copies + barrier + unpremul + barrier +
    // assemble into `cmd` (no begin/end/submit); does the host-side
    // descriptor rewrites/hiddenIn upload first (independent of `cmd`).
    void encodeCore(VulkanContext& ctx, VkCommandBuffer cmd, VkImage colorImage, VkImage auxImage, VkImage fgImage,
                     uint32_t proxyW, uint32_t proxyH, const float* hiddenIn,
                     float eyeX, float eyeY, float eyeZ,
                     float rX, float rY, float rZ, float uX, float uY, float uZ,
                     float fX, float fY, float fZ,
                     float fx, float fy, float cx, float cy,
                     float jitterProxyX, float jitterProxyY, Timings* outTimings);

    struct Impl;
    Impl* impl_ = nullptr;

    uint32_t proxyW_ = 0, proxyH_ = 0, paramStride_ = 1, hiddenChannels_ = 0;
    uint32_t netW_ = 0, netH_ = 0;
    uint32_t texChannels_ = 0, texW_ = 0, texH_ = 0;
    bool firstFrame_ = true;
    bool wasFirstFrame_ = true;
    bool auxIsHalf_ = false;  // from init()'s auxFormat param -- see its comment
    std::vector<float> bgSphere_;  // cx, cy, cz, r
};
