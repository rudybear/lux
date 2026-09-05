#pragma once

#include <string>

// Metal counterpart of reconstruct_runner.h (docs/lux-reconstruct-spec.md):
// transpiles the same luxc-compiled reconstruct.{warp,apply,blend}.comp.spv
// to MSL via ShaderTranspiler (SPIR-V -> MSL, same mechanism
// metal_mesh_renderer.cpp already uses for vertex/fragment/mesh stages;
// GLCompute is just another SPIR-V execution model) and runs them as
// MTL::ComputePipelineState dispatches. No window, no scene.
int runReconstructDumpMetal(const std::string& dumpDir, const std::string& outDir,
                             const std::string& pipelineBase);
