#pragma once

#include <string>

struct VulkanContext;

// Android-side copy of playground_cpp/src/reconstruct_runner.{h,cpp}'s
// context-taking overloads ONLY (per the task brief's explicit "reuse ... by
// reference or by copying the relevant code into
// playground_android/src/reconstruct_pass.{h,cpp}" option) -- the desktop
// file's OTHER, no-ctx overloads construct their own headless VulkanContext
// via `ctx.init(false, true, nullptr, false)`, which resolves to
// vulkan_context.cpp's GLFW-based implementation (the one playground_cpp
// source file this whole app deliberately excludes, see CMakeLists.txt's
// comment) -- linking the original file wholesale into this Android target
// pulls in an undefined `VulkanContext::init(bool, bool, GLFWwindow*,
// bool)` reference even though nothing here calls that overload, since
// --gc-sections only strips dead code AFTER all undefined symbols in
// retained translation units are resolved. Dropping the two GLFW-reaching
// functions (kept, unchanged, in the desktop file) avoids the problem
// entirely; everything else below is copied byte-for-byte from
// reconstruct_runner.cpp/.h (same docs/lux-reconstruct-spec.md contract),
// not reimplemented.
//
// Runs the compiled reconstruct.{warp,apply,blend}.comp.spv compute stages
// (plus {bguv,memory}.comp.spv when the compiled pipeline has
// SPECIFICATION.md 12.9's scene-memory `memory: { channels, hidden }`
// sub-block -- see meta.json's memory_channels field) against a directory
// of dumped .npy inputs (produced offline by mobiledlss's
// reconstruct_reference_dump.py -- network outputs are computed offline, no
// splat rendering or network execution happens here), writing
// out_f{t}.npy / hidden_f{t}.npy per frame. No window, no scene. Runs
// against an ALREADY-CONSTRUCTED VulkanContext (device/queue/command-pool/
// allocator only -- no window, no GLFW calls anywhere in this file).
//
// Task 2 (Reconstruction-mode time budget) breakdown for one runReconstructDump()
// call, CPU wall-clock plus one GPU timestamp-query pair bracketing the
// per-frame compute dispatch chain (bguv+memory+warp+apply+blend -- each its
// own fully-synchronous submission per dispatchOne()'s comment, so
// dispatchCpuMs already includes GPU execution + per-submission overhead;
// dispatchGpuMs isolates just the GPU-visible span between the first and
// last dispatch of the chain, i.e. GPU busy + GPU-side idle between
// submissions, but NOT the CPU-side command-buffer recording/fence-wait
// overhead dispatchCpuMs also carries). setupMs/teardownMs are almost
// entirely per-CALL Vulkan object churn (pipelines/descriptor
// sets/buffers created fresh and destroyed every call -- see runReconstructDump's
// header comment) plus, on the FIRST call only, the one-time scene-memory
// weights load (texture.npy/bg_sphere.npy/memory_head.npz). All *Ms fields
// are summed across every frame in `meta.num_frames` if the call processes
// more than one (the live 1-frame-per-displayed-frame caller only ever
// passes num_frames=1, so for that caller these are already per-frame).
struct ReconstructTimingsMs {
    double setupMs = 0.0;      // pipeline/descriptor-set/buffer creation + static scene-memory load
    double fileReadMs = 0.0;   // per-frame npy reads (proxy_color/mv_proxy/jitter/packed_params/disocc[/k_params/cam_to_world])
    double uploadMs = 0.0;     // per-frame uploadFloats() calls (host->device-visible memcpy)
    double dispatchCpuMs = 0.0;  // wall time of the bguv/memory/warp/apply/blend dispatchOne() calls
    double dispatchGpuMs = 0.0;  // GPU timestamp delta across that same span
    double downloadMs = 0.0;   // per-frame downloadFloats() calls (out_color/hidden[/bguv_target])
    double fileWriteMs = 0.0;  // per-frame npy writes (out_f{t}/hidden_f{t}[/bguv_target_f{t}])
    double teardownMs = 0.0;   // pipeline/descriptor-set/buffer destruction
};

// Returns 0 on success, non-zero on error (message on stderr). `outTimings`
// (optional) receives this call's task-2 profiling breakdown.
int runReconstructDump(VulkanContext& ctx, const std::string& dumpDir, const std::string& outDir,
                        const std::string& pipelineBase, ReconstructTimingsMs* outTimings = nullptr);

// Scene-memory path only (SPECIFICATION.md 12.9): runs just the bguv+memory
// stages at *proxy* resolution, for the network host to consume the sampled
// scene-texture features (the network itself runs outside this pass). Reads
// the same dump-directory files (bg_sphere.npy, texture.npy,
// k_params_proxy_f{t}.npy, cam_to_world_f{t}.npy, meta.json), writes the
// same bg_features_f{t}.npy per frame.
int runDumpBgFeatures(VulkanContext& ctx, const std::string& dumpDir, const std::string& outDir,
                       const std::string& pipelineBase);

// Task 4 (docs/rendering-engines.md, remove the readback stalls/copies where
// cheap): a PERSISTENT equivalent of runReconstructDump's per-frame body --
// pipelines/descriptor sets/buffers + the static scene-memory weights
// (texture/bg_sphere/memory_head) are created/loaded exactly ONCE in init(),
// not on every call, and run() takes/returns plain in-memory float
// pointers instead of reading/writing dump-directory .npy files. Measured
// (task 2's RECON_TIMING) to eliminate ~150ms/frame of pure per-call Vulkan
// object churn + disk round-trip out of Reconstruction mode's ~400ms/frame
// (setup ~61ms + file read ~27ms + file write ~17ms + teardown ~0.1ms, plus
// the caller's own dump-write/output-read npy round trip) that
// runReconstructDump necessarily pays every call since it's also used by
// the one-shot Stage 5/offline dump tools this class deliberately does NOT
// replace (those still call runReconstructDump -- see android_main.cpp's
// runReconstructDump call sites).
//
// Same warp/apply/blend[/bguv/memory] compute chain, same
// ReconstructTimingsMs breakdown for run() (setupMs/teardownMs stay ~0 after
// init(); fileReadMs/fileWriteMs stay 0 always -- there are no files).
// prev_color starts at zero in init() and is carried forward by run() across
// calls exactly like runReconstructDump's own per-call loop does across
// FRAMES within one call -- i.e. genuine temporal continuity across
// displayed frames now, which the old per-call-fresh-zeroed-buffer
// design never had (each runReconstructionModeFrame() call previously
// created its own fresh all-zero prev_color, discarded at the end of that
// same call -- a pre-existing behavioral quirk, not something this class
// was asked to fix, but worth flagging: call reset() if a discontinuous
// jump (e.g. a mode switch) should NOT be blended against stale history).
class ReconstructLive {
public:
    ~ReconstructLive();

    // meta fields mirror runReconstructDump's Meta struct exactly (same
    // s/k/param_stride/hidden/proxy_w/proxy_h/target_w/target_h/net_w/net_h
    // contract); memoryChannels<=0 means no scene-memory path (2-way blend
    // only, matching loadMeta's convention). textureNpyPath/bgSphereNpyPath/
    // memoryHeadNpzPath are only read (once) when memoryChannels>0.
    void init(VulkanContext& ctx, const std::string& pipelineBase,
              int s, int k, int paramStride, int hidden,
              int proxyW, int proxyH, int targetW, int targetH, int netW, int netH,
              int memoryChannels, int memoryHidden, int texW, int texH,
              const std::string& textureNpyPath, const std::string& bgSphereNpyPath,
              const std::string& memoryHeadNpzPath);

    // Resets prev_color to zero (see class comment) without tearing down any
    // GPU objects -- cheap, call on a Reconstruction-mode re-entry after a
    // mode switch skipped frames.
    void reset(VulkanContext& ctx);

    struct FrameInputs {
        const float* proxyColor;  // [proxyH,proxyW,3]
        const float* mvProxy;     // [proxyH,proxyW,2]
        float jitterX = 0.0f, jitterY = 0.0f;
        const float* packed;      // [netH,netW, s*s*K*K + s*s*(hasMemory?3:1) + hidden]
        const float* disocc;      // [targetH,targetW]
        const float* kParams = nullptr;      // [4]: fx,fy,cx,cy -- only used if memoryChannels>0
        const float* camToWorld = nullptr;   // [16] row-major -- only used if memoryChannels>0
    };
    // Pointers valid until the next run() call (owned by this object, same
    // convention as InputAssembly::getOutputHostPtr()/NetRunner::run()).
    struct FrameOutputs {
        const float* outColor = nullptr;  // [targetH,targetW,3]
        const float* hidden = nullptr;    // [targetH,targetW,hidden]
        const float* bguvTarget = nullptr;  // [targetH,targetW,2], null if memoryChannels<=0
    };
    FrameOutputs run(VulkanContext& ctx, const FrameInputs& in, ReconstructTimingsMs* outTimings = nullptr);

private:
    struct Impl;
    Impl* impl_ = nullptr;
};
