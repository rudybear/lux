#pragma once

#include <string>

struct VulkanContext;

// Standalone reconstruction-pass runner (docs/lux-reconstruct-spec.md): runs
// the compiled reconstruct.{warp,apply,blend}.comp.spv compute stages (plus
// {bguv,memory}.comp.spv when the compiled pipeline has SPECIFICATION.md
// 12.9's scene-memory `memory: { channels, hidden }` sub-block -- see
// meta.json's memory_channels field) against a directory of dumped .npy
// inputs (produced offline by mobiledlss's reconstruct_reference_dump.py --
// network outputs are computed offline, no splat rendering or network
// execution happens here), writing out_f{t}.npy / hidden_f{t}.npy per
// frame. No window, no scene.
//
// Two overloads: the plain one constructs and owns its own headless
// VulkanContext (existing desktop-CLI behaviour, unchanged); the
// context-taking one runs against an ALREADY-CONSTRUCTED VulkanContext
// (device/queue/command-pool/allocator only -- no window, no GLFW calls
// anywhere in this file), so a caller that builds its VulkanContext some
// other way (e.g. a future Android NativeActivity path constructing one
// from an ANativeWindow, or headless with no surface at all) can reuse the
// exact same stage-dispatch logic. The plain overload is implemented in
// terms of the context-taking one.
//
// Returns 0 on success, non-zero on error (message on stderr).
int runReconstructDump(const std::string& dumpDir, const std::string& outDir,
                        const std::string& pipelineBase);
int runReconstructDump(VulkanContext& ctx, const std::string& dumpDir, const std::string& outDir,
                        const std::string& pipelineBase);

// Scene-memory path only (SPECIFICATION.md 12.9): runs just the bguv+memory
// stages at *proxy* resolution, for the network host to consume the sampled
// scene-texture features (the network itself runs outside this pass). See
// metal_reconstruct_runner.h's runDumpBgFeaturesMetal for the identical
// Metal-side contract; reads the same dump-directory files (bg_sphere.npy,
// texture.npy, k_params_proxy_f{t}.npy, cam_to_world_f{t}.npy, meta.json),
// writes the same bg_features_f{t}.npy per frame.
int runDumpBgFeatures(const std::string& dumpDir, const std::string& outDir,
                       const std::string& pipelineBase);
int runDumpBgFeatures(VulkanContext& ctx, const std::string& dumpDir, const std::string& outDir,
                       const std::string& pipelineBase);
