#pragma once

#include <string>

// Metal counterpart of reconstruct_runner.h (docs/lux-reconstruct-spec.md):
// transpiles the same luxc-compiled reconstruct.{warp,apply,blend}.comp.spv
// to MSL via ShaderTranspiler (SPIR-V -> MSL, same mechanism
// metal_mesh_renderer.cpp already uses for vertex/fragment/mesh stages;
// GLCompute is just another SPIR-V execution model) and runs them as
// MTL::ComputePipelineState dispatches. No window, no scene.
//
// If the compiled pipeline also has `<pipelineBase>.bguv.comp.spv` /
// `.memory.comp.spv` (a `memory: { channels, hidden }` sub-block was
// declared, SPECIFICATION.md 12.9's scene-memory path), the memory path
// runs too -- `apply`/`blend` use the 3-way blend, and the dump directory
// must additionally carry `bg_sphere.npy`, `texture.npy`, `memory_head.npz`,
// and per-frame `k_params_f{t}.npy`/`cam_to_world_f{t}.npy` (see
// docs/rendering-engines.md's `--reconstruct-dump` directory layout).
int runReconstructDumpMetal(const std::string& dumpDir, const std::string& outDir,
                             const std::string& pipelineBase);

// Runs only the `bguv`+`memory` stages (SPECIFICATION.md 12.9's scene-memory
// path) at *proxy* resolution, for the network host to consume the sampled
// scene-texture features as an extra input-channel block (the network
// itself runs outside this pass). Reads `bg_sphere.npy`, `texture.npy`,
// `k_params_proxy_f{t}.npy`/`cam_to_world_f{t}.npy`, `meta.json`
// (`memory_channels`, `proxy_w/h`, `num_frames`) from `dumpDir`; writes
// `bg_features_f{t}.npy` (`[proxy_h, proxy_w, memory_channels]`) per frame
// into `outDir`.
int runDumpBgFeaturesMetal(const std::string& dumpDir, const std::string& outDir,
                            const std::string& pipelineBase);
