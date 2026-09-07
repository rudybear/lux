#pragma once

#include <Metal/Metal.hpp>
#include <string>
#include <vector>
#include <cstdint>

// Forward declarations
namespace spirv_cross { class CompilerMSL; }

struct TranspiledShader {
    std::string mslSource;
    MTL::Library* library = nullptr;
    MTL::Function* function = nullptr;

    // Resource binding map: SPIR-V (set, binding) → Metal index
    struct BindingMap {
        uint32_t spvSet;
        uint32_t spvBinding;
        std::string name;
        uint32_t mslBuffer = UINT32_MAX;
        uint32_t mslTexture = UINT32_MAX;
        uint32_t mslSampler = UINT32_MAX;
    };
    std::vector<BindingMap> bindings;
    uint32_t pushConstantBufferIndex = UINT32_MAX;

    // Find Metal buffer index for a SPIR-V (set, binding)
    uint32_t findBufferIndex(uint32_t set, uint32_t binding) const;
    uint32_t findTextureIndex(uint32_t set, uint32_t binding) const;
    uint32_t findSamplerIndex(uint32_t set, uint32_t binding) const;

    void cleanup();
};

class ShaderTranspiler {
public:
    void init(MTL::Device* device);

    // Transpile a .spv file to MSL and compile to MTLLibrary
    // stage: 0=vertex, 3=fragment, 5268=mesh (spv::ExecutionModel values)
    TranspiledShader transpile(const std::string& spvPath, uint32_t executionModel);

    // Transpile raw SPIR-V data
    TranspiledShader transpileData(const std::vector<uint32_t>& spirvData, uint32_t executionModel);

    // Transpile directly into an existing TranspiledShader (avoids return-by-value issues)
    void transpileInto(TranspiledShader& result, const std::string& spvPath, uint32_t executionModel);

    void cleanup();

private:
    MTL::Device* m_device = nullptr;

    // Internal transpile implementation
    void transpileInto(TranspiledShader& result, const std::vector<uint32_t>& spirvData, uint32_t executionModel);

    // Load SPIR-V binary from file
    std::vector<uint32_t> loadSPIRV(const std::string& path);

    // Compile MSL source to Metal library
    MTL::Library* compileMSL(const std::string& source);

    // Metal-only, fragment-stage-only: runs `spirv-opt
    // --convert-relaxed-to-half` on the given SPIR-V (in place) via
    // subprocess (mirrors luxc/codegen/spv_assembler.py's run_spirv_opt
    // pattern). See bench/lux_perf_ablation.md in mobiledlss ("precision:
    // relaxed" follow-up): SPIRV-Cross's MSL backend does NOT
    // automatically lower `RelaxedPrecision`-decorated fp32 ops to real
    // MSL `half` -- that requires this actual SPIR-V type-rewriting pass
    // (part of SPIRV-Tools, distinct from SPIRV-Cross) BEFORE the
    // transpile step. Applying it to a shader with no RelaxedPrecision
    // decorations (i.e. every non-`_relaxed` pipeline) is a verified
    // byte-identical no-op (spirv-opt has nothing to convert), so this is
    // safe to call unconditionally on every fragment-stage SPIR-V module
    // without needing a separate runtime flag -- it only ever changes
    // anything for pipelines actually compiled with the `precision:
    // relaxed` splat option (the "existing option" this is gated behind).
    // Silently leaves spirvData unmodified if spirv-opt is unavailable or
    // fails (matches the Python helper's fail-open behavior) -- Vulkan is
    // completely unaffected either way (this only touches the in-memory
    // copy loaded for the Metal transpile, never the .spv file on disk).
    void convertRelaxedToHalfIfFragment(std::vector<uint32_t>& spirvData, uint32_t executionModel);
};

// SPIR-V execution model constants (matches spirv.hpp values)
namespace SpvExecModel {
    constexpr uint32_t Vertex = 0;
    constexpr uint32_t Fragment = 4;
    constexpr uint32_t GLCompute = 5;    // compute shaders (docs/lux-reconstruct-spec.md)
    constexpr uint32_t TaskEXT = 5267;   // SPV_EXT_mesh_shader task/object stage
    constexpr uint32_t MeshEXT = 5268;   // SPV_EXT_mesh_shader mesh stage
}
