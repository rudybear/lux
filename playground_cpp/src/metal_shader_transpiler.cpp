#include "metal_shader_transpiler.h"
#include "reflected_pipeline.h"
#if __has_include(<spirv_cross/spirv_msl.hpp>)
#include <spirv_cross/spirv_msl.hpp>
#else
#include <spirv_msl.hpp>
#endif
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <memory>

// ---------------------------------------------------------------------------
// TranspiledShader helpers
// ---------------------------------------------------------------------------

uint32_t TranspiledShader::findBufferIndex(uint32_t set, uint32_t binding) const {
    for (auto& b : bindings) {
        if (b.spvSet == set && b.spvBinding == binding && b.mslBuffer != UINT32_MAX)
            return b.mslBuffer;
    }
    return UINT32_MAX;
}

uint32_t TranspiledShader::findTextureIndex(uint32_t set, uint32_t binding) const {
    for (auto& b : bindings) {
        if (b.spvSet == set && b.spvBinding == binding && b.mslTexture != UINT32_MAX)
            return b.mslTexture;
    }
    return UINT32_MAX;
}

uint32_t TranspiledShader::findSamplerIndex(uint32_t set, uint32_t binding) const {
    for (auto& b : bindings) {
        if (b.spvSet == set && b.spvBinding == binding && b.mslSampler != UINT32_MAX)
            return b.mslSampler;
    }
    return UINT32_MAX;
}

void TranspiledShader::cleanup() {
    if (function) { function->release(); function = nullptr; }
    if (library) { library->release(); library = nullptr; }
}

// ---------------------------------------------------------------------------
// ShaderTranspiler
// ---------------------------------------------------------------------------

void ShaderTranspiler::init(MTL::Device* device) {
    m_device = device;
}

void ShaderTranspiler::cleanup() {
    m_device = nullptr;
}

std::vector<uint32_t> ShaderTranspiler::loadSPIRV(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open SPIR-V file: " + path);
    }
    size_t size = static_cast<size_t>(file.tellg());
    file.seekg(0);

    if (size % 4 != 0) {
        throw std::runtime_error("Invalid SPIR-V file size (not aligned to 4 bytes): " + path);
    }

    std::vector<uint32_t> data(size / 4);
    file.read(reinterpret_cast<char*>(data.data()), size);

    // Verify SPIR-V magic
    if (data.empty() || data[0] != 0x07230203) {
        throw std::runtime_error("Invalid SPIR-V magic number in: " + path);
    }

    return data;
}

MTL::Library* ShaderTranspiler::compileMSL(const std::string& source) {
    NS::Error* error = nullptr;
    auto* nsSource = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
    auto* options = MTL::CompileOptions::alloc()->init();

    MTL::Library* library = m_device->newLibrary(nsSource, options, &error);
    options->release();

    if (!library) {
        std::string errorMsg = "Failed to compile MSL";
        if (error) {
            errorMsg += ": ";
            errorMsg += error->localizedDescription()->utf8String();
        }
        std::cerr << "[metal] MSL compilation error. Source:\n" << source.substr(0, 2000) << std::endl;
        throw std::runtime_error(errorMsg);
    }

    return library;
}

TranspiledShader ShaderTranspiler::transpile(const std::string& spvPath, uint32_t executionModel) {
    auto spirvData = loadSPIRV(spvPath);
    TranspiledShader result;
    transpileInto(result, spirvData, executionModel);
    return result;
}

void ShaderTranspiler::transpileInto(TranspiledShader& result, const std::string& spvPath, uint32_t executionModel) {
    auto spirvData = loadSPIRV(spvPath);
    transpileInto(result, spirvData, executionModel);
}

TranspiledShader ShaderTranspiler::transpileData(const std::vector<uint32_t>& spirvData, uint32_t executionModel) {
    TranspiledShader result;
    transpileInto(result, spirvData, executionModel);
    return result;
}

void ShaderTranspiler::transpileInto(TranspiledShader& result, const std::vector<uint32_t>& spirvData, uint32_t executionModel) {

    // Use heap allocation for the compiler to control destruction order
    // and work around potential stack/destructor issues
    auto compiler = std::make_unique<spirv_cross::CompilerMSL>(spirvData);

    // Configure MSL options
    auto opts = compiler->get_msl_options();
    opts.platform = spirv_cross::CompilerMSL::Options::macOS;
    opts.msl_version = spirv_cross::CompilerMSL::Options::make_msl_version(3, 0);
    opts.argument_buffers = false; // Start with discrete bindings
    opts.pad_fragment_output_components = true;
    opts.texture_buffer_native = true;
    opts.force_native_arrays = true;
    compiler->set_msl_options(opts);

    // Get shader resources for reflection
    auto resources = compiler->get_shader_resources();

    // CompilerMSL may not report push_constant_buffers in get_shader_resources()
    // for fragment shaders. Detect push constants by scanning the SPIR-V binary
    // directly for OpVariable (opcode 59) with StorageClass::PushConstant (9).
    bool hasPushConstants = !resources.push_constant_buffers.empty();
    if (!hasPushConstants) {
        // SPIR-V header is 5 words; instructions follow
        const uint32_t SpvOpVariable = 59;
        const uint32_t SpvStorageClassPushConstant = 9;
        size_t i = 5; // skip header
        while (i < spirvData.size()) {
            uint32_t word0 = spirvData[i];
            uint16_t opcode = word0 & 0xFFFF;
            uint16_t wordCount = (word0 >> 16) & 0xFFFF;
            if (wordCount == 0) break; // malformed
            // OpVariable: [opcode_word, result_type, result_id, storage_class, ...]
            if (opcode == SpvOpVariable && wordCount >= 4) {
                uint32_t storageClass = spirvData[i + 3];
                if (storageClass == SpvStorageClassPushConstant) {
                    hasPushConstants = true;
                    break;
                }
            }
            i += wordCount;
        }
    }

    // SPIRV-Cross appends .__sampler / .__image when combining separate
    // samplers+images into MSL combined textures. Strip these suffixes so
    // binding names match the scene texture map keys (e.g. "base_color_tex").
    auto cleanName = [](const std::string& name) -> std::string {
        static const std::string suffixes[] = {".__sampler", ".__image"};
        for (auto& s : suffixes) {
            if (name.size() > s.size() && name.compare(name.size() - s.size(), s.size(), s) == 0)
                return name.substr(0, name.size() - s.size());
        }
        return name;
    };

    // Assign explicit buffer indices for all stages.
    // For vertex shaders, reserve buffer(0) for vertex data (stage_in).
    // For other stages, start buffer assignments at 0.
    uint32_t explicitPushConstantIdx = UINT32_MAX;
    {
        uint32_t bufIdx = (executionModel == SpvExecModel::Vertex) ? 1 : 0;
        for (auto& ub : resources.uniform_buffers) {
            uint32_t set = compiler->get_decoration(ub.id, spv::DecorationDescriptorSet);
            uint32_t binding = compiler->get_decoration(ub.id, spv::DecorationBinding);
            spirv_cross::MSLResourceBinding mslBinding;
            mslBinding.stage = static_cast<spv::ExecutionModel>(executionModel);
            mslBinding.desc_set = set;
            mslBinding.binding = binding;
            mslBinding.msl_buffer = bufIdx++;
            mslBinding.msl_texture = 0;
            mslBinding.msl_sampler = 0;
            compiler->add_msl_resource_binding(mslBinding);
        }
        for (auto& sb : resources.storage_buffers) {
            uint32_t set = compiler->get_decoration(sb.id, spv::DecorationDescriptorSet);
            uint32_t binding = compiler->get_decoration(sb.id, spv::DecorationBinding);
            spirv_cross::MSLResourceBinding mslBinding;
            mslBinding.stage = static_cast<spv::ExecutionModel>(executionModel);
            mslBinding.desc_set = set;
            mslBinding.binding = binding;
            mslBinding.msl_buffer = bufIdx++;
            mslBinding.msl_texture = 0;
            mslBinding.msl_sampler = 0;
            compiler->add_msl_resource_binding(mslBinding);
        }
        if (hasPushConstants) {
            explicitPushConstantIdx = bufIdx;
            spirv_cross::MSLResourceBinding mslBinding;
            mslBinding.stage = static_cast<spv::ExecutionModel>(executionModel);
            mslBinding.desc_set = spirv_cross::kPushConstDescSet;
            mslBinding.binding = spirv_cross::kPushConstBinding;
            mslBinding.msl_buffer = bufIdx++;
            mslBinding.msl_texture = 0;
            mslBinding.msl_sampler = 0;
            compiler->add_msl_resource_binding(mslBinding);
        }
    }

    // Track resource IDs and their names for post-compilation binding query
    struct ResourceInfo {
        uint32_t id;
        uint32_t spvSet;
        uint32_t spvBinding;
        std::string name;
        enum Kind { Buffer, Texture, Sampler } kind;
    };
    std::vector<ResourceInfo> trackedResources;

    for (auto& ub : resources.uniform_buffers) {
        uint32_t set = compiler->get_decoration(ub.id, spv::DecorationDescriptorSet);
        uint32_t binding = compiler->get_decoration(ub.id, spv::DecorationBinding);
        trackedResources.push_back({ub.id, set, binding, cleanName(ub.name), ResourceInfo::Buffer});
    }
    for (auto& sb : resources.storage_buffers) {
        uint32_t set = compiler->get_decoration(sb.id, spv::DecorationDescriptorSet);
        uint32_t binding = compiler->get_decoration(sb.id, spv::DecorationBinding);
        trackedResources.push_back({sb.id, set, binding, cleanName(sb.name), ResourceInfo::Buffer});
    }
    for (auto& si : resources.sampled_images) {
        uint32_t set = compiler->get_decoration(si.id, spv::DecorationDescriptorSet);
        uint32_t binding = compiler->get_decoration(si.id, spv::DecorationBinding);
        trackedResources.push_back({si.id, set, binding, cleanName(si.name), ResourceInfo::Texture});
    }
    for (auto& img : resources.separate_images) {
        uint32_t set = compiler->get_decoration(img.id, spv::DecorationDescriptorSet);
        uint32_t binding = compiler->get_decoration(img.id, spv::DecorationBinding);
        trackedResources.push_back({img.id, set, binding, cleanName(img.name), ResourceInfo::Texture});
    }
    for (auto& samp : resources.separate_samplers) {
        uint32_t set = compiler->get_decoration(samp.id, spv::DecorationDescriptorSet);
        uint32_t binding = compiler->get_decoration(samp.id, spv::DecorationBinding);
        trackedResources.push_back({samp.id, set, binding, cleanName(samp.name), ResourceInfo::Sampler});
    }

    // Compile to MSL — let SPIRV-Cross auto-assign texture/sampler indices
    // (except for vertex buffers where we reserved index 0)
    result.mslSource = compiler->compile();

    // After compilation, query the ACTUAL Metal binding indices.
    // This correctly handles dead-code elimination and index compaction.
    for (auto& ri : trackedResources) {
        uint32_t actualBinding = compiler->get_automatic_msl_resource_binding(ri.id);
        uint32_t actualSecondary = compiler->get_automatic_msl_resource_binding_secondary(ri.id);

        // Skip resources that were dead-code eliminated
        if (actualBinding == UINT32_MAX && actualSecondary == UINT32_MAX) continue;

        TranspiledShader::BindingMap bm;
        bm.spvSet = ri.spvSet;
        bm.spvBinding = ri.spvBinding;
        bm.name = ri.name;

        if (ri.kind == ResourceInfo::Buffer) {
            bm.mslBuffer = actualBinding;
        } else if (ri.kind == ResourceInfo::Texture) {
            bm.mslTexture = actualBinding;
            // For combined image-samplers, SPIRV-Cross auto-assigns matching
            // texture and sampler indices. get_automatic_msl_resource_binding_secondary
            // may not populate the sampler index for auto-assigned bindings,
            // so fall back to using the texture index (the MSL default).
            bm.mslSampler = (actualSecondary != UINT32_MAX) ? actualSecondary : actualBinding;
        } else if (ri.kind == ResourceInfo::Sampler) {
            bm.mslSampler = actualBinding;
        }
        result.bindings.push_back(bm);
    }

    // Push constant binding: use our explicitly assigned index.
    // get_automatic_msl_resource_binding() returns UINT32_MAX for explicit bindings,
    // so we use the index we tracked during assignment.
    if (hasPushConstants && explicitPushConstantIdx != UINT32_MAX) {
        result.pushConstantBufferIndex = explicitPushConstantIdx;
    }

    std::cout << "[metal] Transpiled MSL (" << result.mslSource.size() << " chars), "
              << result.bindings.size() << " bindings, "
              << (result.pushConstantBufferIndex != UINT32_MAX ? "has push constants" : "no push constants")
              << std::endl;


    // Destroy the compiler before creating Metal objects
    compiler.reset();

    // Compile MSL to Metal library
    result.library = compileMSL(result.mslSource);

    // Get the main entry point function
    auto* funcName = NS::String::string("main0", NS::UTF8StringEncoding);
    result.function = result.library->newFunction(funcName);
    if (!result.function) {
        throw std::runtime_error("Failed to find 'main0' function in compiled MSL");
    }
}
