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
};

// SPIR-V execution model constants (matches spirv.hpp values)
namespace SpvExecModel {
    constexpr uint32_t Vertex = 0;
    constexpr uint32_t Fragment = 4;
    constexpr uint32_t TaskEXT = 5267;   // SPV_EXT_mesh_shader task/object stage
    constexpr uint32_t MeshEXT = 5268;   // SPV_EXT_mesh_shader mesh stage
}
