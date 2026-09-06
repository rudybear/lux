#pragma once

#include <string>

// Fused-GPU-compute ParamPredUNet runner (docs/lux-unet-spec.md, stage 1-3:
// hand-written generic conv kernels chained by host code for this specific
// network's fixed layer sequence -- not yet a data-driven lux `network`
// block, see the spec's step 4). Runs the network on one dumped
// `build_input` tensor (mobiledlss's tools/reconstruct_reference_dump.py)
// and writes the packed `forward_packed`-equivalent output.
//
// `weightsBlobPath` is export_lux_weights.py's `--out-blob`;
// `manifestJsonPath` its `--out-manifest` (the JSON file -- this reads the
// `.layers.txt` sidecar written alongside it, same stem, for simpler C++
// parsing; see export_lux_weights.py's own comment on that file).
int runUnetDump(const std::string& inputNpyPath, const std::string& weightsBlobPath,
                const std::string& manifestJsonPath, const std::string& outputNpyPath,
                const std::string& kernelPipelineDir);
