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
// Returns 0 on success, non-zero on error (message on stderr).
int runReconstructDump(VulkanContext& ctx, const std::string& dumpDir, const std::string& outDir,
                        const std::string& pipelineBase);

// Scene-memory path only (SPECIFICATION.md 12.9): runs just the bguv+memory
// stages at *proxy* resolution, for the network host to consume the sampled
// scene-texture features (the network itself runs outside this pass). Reads
// the same dump-directory files (bg_sphere.npy, texture.npy,
// k_params_proxy_f{t}.npy, cam_to_world_f{t}.npy, meta.json), writes the
// same bg_features_f{t}.npy per frame.
int runDumpBgFeatures(VulkanContext& ctx, const std::string& dumpDir, const std::string& outDir,
                       const std::string& pipelineBase);
