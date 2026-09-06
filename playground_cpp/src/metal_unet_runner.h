#pragma once

#include <string>

// Metal counterpart of unet_runner.h (docs/lux-unet-spec.md): transpiles the
// same luxc-compiled unet_*.comp.spv kernels to MSL via ShaderTranspiler.
int runUnetDumpMetal(const std::string& inputNpyPath, const std::string& weightsBlobPath,
                      const std::string& manifestJsonPath, const std::string& outputNpyPath,
                      const std::string& kernelPipelineDir);
