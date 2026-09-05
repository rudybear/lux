#pragma once

#include <string>

// Standalone reconstruction-pass runner (docs/lux-reconstruct-spec.md): runs
// the compiled reconstruct.{warp,apply,blend}.comp.spv compute stages
// against a directory of dumped .npy inputs (produced offline by
// mobiledlss's reconstruct_reference_dump.py -- network outputs are
// computed offline, no splat rendering or network execution happens here),
// writing out_f{t}.npy / hidden_f{t}.npy per frame. No window, no scene.
//
// Returns 0 on success, non-zero on error (message on stderr).
int runReconstructDump(const std::string& dumpDir, const std::string& outDir,
                        const std::string& pipelineBase);
