#include "metal_raster_renderer.h"
#include "camera.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

// --------------------------------------------------------------------------
// Fullscreen vertex shader (built-in MSL)
// --------------------------------------------------------------------------

static const char* kFullscreenVertMSL = R"(
#include <metal_stdlib>
using namespace metal;

struct VertexOut {
    float4 position [[position]];
    float2 uv [[user(locn0)]];
};

vertex VertexOut fullscreen_vert(uint vid [[vertex_id]]) {
    VertexOut out;
    // Generate fullscreen triangle: 3 vertices covering [-1,1] NDC
    float2 positions[3] = {float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0)};
    float2 uvs[3] = {float2(0.0, 1.0), float2(2.0, 1.0), float2(0.0, -1.0)};
    out.position = float4(positions[vid], 0.0, 1.0);
    out.uv = uvs[vid];
    return out;
}
)";

// --------------------------------------------------------------------------
// Render target creation
// --------------------------------------------------------------------------

void MetalRasterRenderer::createRenderTargets(MetalContext& ctx) {
    // Color target
    auto* colorDesc = MTL::TextureDescriptor::texture2DDescriptor(
        MTL::PixelFormatRGBA8Unorm, m_width, m_height, false);
    colorDesc->setUsage(MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead);
    colorDesc->setStorageMode(MTL::StorageModePrivate);
    m_colorTarget = ctx.newTexture(colorDesc);
    m_colorTarget->setLabel(NS::String::string("OffscreenColor", NS::UTF8StringEncoding));

    if (m_needsDepth) {
        auto* depthDesc = MTL::TextureDescriptor::texture2DDescriptor(
            MTL::PixelFormatDepth32Float, m_width, m_height, false);
        depthDesc->setUsage(MTL::TextureUsageRenderTarget);
        depthDesc->setStorageMode(MTL::StorageModePrivate);
        m_depthTarget = ctx.newTexture(depthDesc);
        m_depthTarget->setLabel(NS::String::string("OffscreenDepth", NS::UTF8StringEncoding));
    }
}

void MetalRasterRenderer::createDepthStencilState(MetalContext& ctx) {
    auto* dsDesc = MTL::DepthStencilDescriptor::alloc()->init();
    dsDesc->setDepthCompareFunction(MTL::CompareFunctionLess);
    dsDesc->setDepthWriteEnabled(true);
    m_depthStencilState = ctx.device->newDepthStencilState(dsDesc);
    dsDesc->release();
}

// --------------------------------------------------------------------------
// Pipeline creation helpers
// --------------------------------------------------------------------------

MTL::RenderPipelineState* MetalRasterRenderer::createRenderPipelineState(
    MetalContext& ctx, MTL::Function* vertFunc, MTL::Function* fragFunc,
    const ReflectionData* vertRefl, int overrideStride) {

    auto* pipeDesc = MTL::RenderPipelineDescriptor::alloc()->init();
    pipeDesc->setVertexFunction(vertFunc);
    pipeDesc->setFragmentFunction(fragFunc);
    pipeDesc->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatRGBA8Unorm);

    if (m_needsDepth) {
        pipeDesc->setDepthAttachmentPixelFormat(MTL::PixelFormatDepth32Float);
    }

    // Configure vertex descriptor from reflection if available
    if (vertRefl && !vertRefl->vertex_attributes.empty()) {
        auto* vertDesc = MTL::VertexDescriptor::alloc()->init();
        auto* layouts = vertDesc->layouts();
        auto* attrs = vertDesc->attributes();

        int stride = (overrideStride > 0) ? overrideStride : vertRefl->vertex_stride;
        layouts->object(0)->setStride(stride);
        layouts->object(0)->setStepFunction(MTL::VertexStepFunctionPerVertex);

        for (auto& attr : vertRefl->vertex_attributes) {
            attrs->object(attr.location)->setFormat(reflectionFormatToMTLVertexFormat(attr.format));
            attrs->object(attr.location)->setOffset(attr.offset);
            attrs->object(attr.location)->setBufferIndex(0);
        }

        pipeDesc->setVertexDescriptor(vertDesc);
        vertDesc->release();
    }

    NS::Error* error = nullptr;
    auto* pipeState = ctx.device->newRenderPipelineState(pipeDesc, &error);
    pipeDesc->release();

    if (!pipeState) {
        std::string errorMsg = "Failed to create Metal render pipeline state";
        if (error) {
            errorMsg += ": ";
            errorMsg += error->localizedDescription()->utf8String();
        }
        throw std::runtime_error(errorMsg);
    }

    return pipeState;
}

// --------------------------------------------------------------------------
// Triangle pipeline
// --------------------------------------------------------------------------

void MetalRasterRenderer::createPipelineTriangle(MetalContext& ctx) {
    // Compile simple triangle shaders from SPIR-V
    m_transpiler.transpileInto(m_vertexShader, m_pipelineBase + ".vert.spv", SpvExecModel::Vertex);
    m_transpiler.transpileInto(m_fragmentShader, m_pipelineBase + ".frag.spv", SpvExecModel::Fragment);

    // Load reflection for vertex descriptor
    std::string vertJsonPath = m_pipelineBase + ".vert.lux.json";
    if (fs::exists(vertJsonPath)) m_vertReflection = parseReflectionJson(vertJsonPath);

    int stride = 24; // 2x vec3 (position + color) = 24 bytes
    m_pipelineState = createRenderPipelineState(ctx, m_vertexShader.function,
                                                m_fragmentShader.function,
                                                &m_vertReflection, stride);

    // Upload triangle vertices
    struct TriVert { float px, py, pz, cr, cg, cb; };
    TriVert triVerts[3] = {
        { 0.0f, -0.5f, 0.0f,  1.0f, 0.0f, 0.0f},
        { 0.5f,  0.5f, 0.0f,  0.0f, 1.0f, 0.0f},
        {-0.5f,  0.5f, 0.0f,  0.0f, 0.0f, 1.0f}
    };
    m_triangleVB = ctx.newBuffer(triVerts, sizeof(triVerts), MTL::ResourceStorageModeShared);
}

// --------------------------------------------------------------------------
// Fullscreen pipeline
// --------------------------------------------------------------------------

void MetalRasterRenderer::createPipelineFullscreen(MetalContext& ctx) {
    // Compile built-in fullscreen vertex shader from MSL
    NS::Error* error = nullptr;
    auto* mslSource = NS::String::string(kFullscreenVertMSL, NS::UTF8StringEncoding);
    auto* compileOpts = MTL::CompileOptions::alloc()->init();
    m_fullscreenLib = ctx.device->newLibrary(mslSource, compileOpts, &error);
    compileOpts->release();

    if (!m_fullscreenLib) {
        throw std::runtime_error("Failed to compile fullscreen vertex shader");
    }

    auto* funcName = NS::String::string("fullscreen_vert", NS::UTF8StringEncoding);
    m_fullscreenVertFunc = m_fullscreenLib->newFunction(funcName);

    // Fragment shader from SPIR-V
    m_transpiler.transpileInto(m_fragmentShader, m_pipelineBase + ".frag.spv", SpvExecModel::Fragment);

    // No vertex descriptor — fullscreen generates vertices procedurally
    m_pipelineState = createRenderPipelineState(ctx, m_fullscreenVertFunc,
                                                m_fragmentShader.function, nullptr);
}

// --------------------------------------------------------------------------
// PBR pipeline
// --------------------------------------------------------------------------

void MetalRasterRenderer::createPipelinePBR(MetalContext& ctx) {
    // Load reflection data
    std::string vertJsonPath = m_pipelineBase + ".vert.lux.json";
    std::string fragJsonPath = m_pipelineBase + ".frag.lux.json";
    m_vertReflection = parseReflectionJson(vertJsonPath);
    m_fragReflection = parseReflectionJson(fragJsonPath);

    // Transpile shaders
    m_transpiler.transpileInto(m_vertexShader, m_pipelineBase + ".vert.spv", SpvExecModel::Vertex);
    m_transpiler.transpileInto(m_fragmentShader, m_pipelineBase + ".frag.spv", SpvExecModel::Fragment);

    // Determine vertex stride (48 for glTF with tangents, 32 for basic)
    int vertexStride = m_scene->hasGltfScene() ? 48 : 32;

    m_pipelineState = createRenderPipelineState(ctx, m_vertexShader.function,
                                                m_fragmentShader.function,
                                                &m_vertReflection, vertexStride);

    // Create MVP uniform buffer (3 mat4 = 192 bytes)
    m_mvpBuffer = ctx.newBuffer(192, MTL::ResourceStorageModeShared);

    // Create light uniform buffer (we'll size it from reflection)
    uint32_t lightBufSize = 64; // default
    for (auto& [setIdx, bindings] : m_fragReflection.descriptor_sets) {
        for (auto& b : bindings) {
            if (b.name.find("Light") != std::string::npos || b.name.find("light") != std::string::npos) {
                lightBufSize = std::max(lightBufSize, static_cast<uint32_t>(b.size));
            }
        }
    }
    m_lightBuffer = ctx.newBuffer(lightBufSize, MTL::ResourceStorageModeShared);

    // Light UBO layout: vec3 light_dir (padded) + vec3 view_pos (padded)
    // Must match the shader's Light struct: { float3 light_dir; float3 view_pos; }
    struct LightData {
        glm::vec3 lightDir;
        float _pad0;
        glm::vec3 viewPos;
        float _pad1;
    } lightData;
    lightData.lightDir = glm::normalize(glm::vec3(1.0f, 0.8f, 0.6f));
    lightData._pad0 = 0.0f;
    lightData.viewPos = m_scene->hasSceneBounds() ? m_scene->getAutoEye() : Camera::DEFAULT_EYE;
    lightData._pad1 = 0.0f;
    memcpy(m_lightBuffer->contents(), &lightData, sizeof(lightData));

    // Create material UBOs
    if (m_scene->hasGltfScene() && !m_scene->getGltfScene().materials.empty()) {
        auto& materials = m_scene->getGltfScene().materials;
        m_perMaterialUBOs.resize(materials.size());
        for (size_t i = 0; i < materials.size(); i++) {
            m_perMaterialUBOs[i] = ctx.newBuffer(sizeof(MaterialUBOData), MTL::ResourceStorageModeShared);
            fillMaterialUBO(m_perMaterialUBOs[i], static_cast<int>(i));
        }
    } else {
        // Single default material
        m_materialBuffer = ctx.newBuffer(sizeof(MaterialUBOData), MTL::ResourceStorageModeShared);
        MaterialUBOData defaultMat;
        memcpy(m_materialBuffer->contents(), &defaultMat, sizeof(defaultMat));
    }

    // Set up push constants from reflection and populate with initial values
    auto fillPushConstants = [&](const ReflectionData& refl) {
        for (auto& pc : refl.push_constants) {
            if (pc.size <= 0) continue;
            m_pushConstantSize = std::max(m_pushConstantSize, static_cast<uint32_t>(pc.size));
            m_pushConstantData.resize(m_pushConstantSize, 0);
            for (auto& f : pc.fields) {
                if (f.name == "light_dir" && static_cast<uint32_t>(f.offset) + 12 <= m_pushConstantSize) {
                    glm::vec3 lightDir = glm::normalize(glm::vec3(1.0f, 0.8f, 0.6f));
                    memcpy(m_pushConstantData.data() + f.offset, &lightDir, sizeof(glm::vec3));
                } else if (f.name == "view_pos" && static_cast<uint32_t>(f.offset) + 12 <= m_pushConstantSize) {
                    glm::vec3 eye = m_scene->hasSceneBounds() ? m_scene->getAutoEye() : Camera::DEFAULT_EYE;
                    memcpy(m_pushConstantData.data() + f.offset, &eye, sizeof(glm::vec3));
                }
            }
        }
    };
    fillPushConstants(m_vertReflection);
    fillPushConstants(m_fragReflection);

    // Initialize MVP
    float aspect = static_cast<float>(m_width) / static_cast<float>(m_height);
    auto cam = Camera::getDefaultMatrices(aspect);
    if (m_scene->hasSceneBounds()) {
        cam.view = Camera::lookAt(m_scene->getAutoEye(), m_scene->getAutoTarget(), m_scene->getAutoUp());
        cam.projection = Camera::perspective(glm::radians(45.0f), aspect, 0.1f, m_scene->getAutoFar());
    }
    struct { glm::mat4 model, view, proj; } mvpData = {cam.model, cam.view, cam.projection};
    memcpy(m_mvpBuffer->contents(), &mvpData, sizeof(mvpData));

    // Check for permutation selection
    ShaderManifest manifest = tryLoadManifest(m_pipelineBase);
    bool hasMultipleMaterials = m_scene->hasGltfScene() &&
                                 m_scene->getGltfScene().materials.size() > 1;
    if (!manifest.permutations.empty() && hasMultipleMaterials) {
        setupMultiPipeline(ctx, manifest);
    } else if (!manifest.permutations.empty() && m_scene->hasGltfScene()) {
        // Single-material scene: resolve best-matching permutation
        auto sceneFeatures = m_scene->detectSceneFeatures();
        std::string suffix = findPermutationSuffix(manifest, sceneFeatures);
        if (!suffix.empty()) {
            std::string candidateBase = m_pipelineBase + suffix;
            if (fs::exists(candidateBase + ".vert.spv") && fs::exists(candidateBase + ".frag.spv")) {
                std::cout << "[metal] Resolved single-material permutation: " << suffix << std::endl;

                // Re-transpile with permutation shaders (fully reset old state)
                m_vertexShader.cleanup();
                m_vertexShader.bindings.clear();
                m_vertexShader.pushConstantBufferIndex = UINT32_MAX;
                m_fragmentShader.cleanup();
                m_fragmentShader.bindings.clear();
                m_fragmentShader.pushConstantBufferIndex = UINT32_MAX;
                if (m_pipelineState) { m_pipelineState->release(); m_pipelineState = nullptr; }

                // Load permutation reflection data
                std::string vertJson = candidateBase + ".vert.lux.json";
                std::string fragJson = candidateBase + ".frag.lux.json";
                if (fs::exists(vertJson)) m_vertReflection = parseReflectionJson(vertJson);
                if (fs::exists(fragJson)) m_fragReflection = parseReflectionJson(fragJson);

                m_transpiler.transpileInto(m_vertexShader, candidateBase + ".vert.spv", SpvExecModel::Vertex);
                m_transpiler.transpileInto(m_fragmentShader, candidateBase + ".frag.spv", SpvExecModel::Fragment);

                int vertexStride = m_scene->hasGltfScene() ? 48 : 32;
                m_pipelineState = createRenderPipelineState(ctx, m_vertexShader.function,
                                                            m_fragmentShader.function,
                                                            &m_vertReflection, vertexStride);

                // Re-setup push constants from permutation reflection
                m_pushConstantSize = 0;
                m_pushConstantData.clear();
                fillPushConstants(m_vertReflection);
                fillPushConstants(m_fragReflection);
            }
        }
    }
}

// --------------------------------------------------------------------------
// Multi-pipeline setup
// --------------------------------------------------------------------------

void MetalRasterRenderer::setupMultiPipeline(MetalContext& ctx, const ShaderManifest& manifest) {
    m_multiPipeline = true;
    auto& materials = m_scene->getGltfScene().materials;

    // Group materials by features
    std::map<std::string, std::vector<int>> groups;
    for (size_t i = 0; i < materials.size(); i++) {
        auto features = m_scene->detectMaterialFeatures(static_cast<int>(i));
        std::string suffix = findPermutationSuffix(manifest, features);
        groups[suffix].push_back(static_cast<int>(i));
    }

    m_materialToPermutation.resize(materials.size(), 0);

    for (auto& [suffix, matIndices] : groups) {
        MetalPermutation perm;
        perm.suffix = suffix;

        std::string permBase = m_pipelineBase + suffix;
        std::string vertSpv = permBase + ".vert.spv";
        std::string fragSpv = permBase + ".frag.spv";

        // Fall back to base shaders if permutation doesn't exist
        if (!fs::exists(vertSpv)) vertSpv = m_pipelineBase + ".vert.spv";
        if (!fs::exists(fragSpv)) fragSpv = m_pipelineBase + ".frag.spv";

        m_transpiler.transpileInto(perm.vertShader, vertSpv, SpvExecModel::Vertex);
        m_transpiler.transpileInto(perm.fragShader, fragSpv, SpvExecModel::Fragment);

        // Load reflection for permutation
        std::string vertJson = permBase + ".vert.lux.json";
        std::string fragJson = permBase + ".frag.lux.json";
        if (!fs::exists(vertJson)) vertJson = m_pipelineBase + ".vert.lux.json";
        if (!fs::exists(fragJson)) fragJson = m_pipelineBase + ".frag.lux.json";

        ReflectionData vertRefl = parseReflectionJson(vertJson);
        int vertexStride = m_scene->hasGltfScene() ? 48 : 32;

        perm.pipelineState = createRenderPipelineState(ctx, perm.vertShader.function,
                                                        perm.fragShader.function,
                                                        &vertRefl, vertexStride);

        // Create per-material UBOs for this permutation
        perm.materialIndices = matIndices;
        perm.perMaterialUBOs.resize(matIndices.size());
        for (size_t i = 0; i < matIndices.size(); i++) {
            perm.perMaterialUBOs[i] = ctx.newBuffer(sizeof(MaterialUBOData), MTL::ResourceStorageModeShared);
            fillMaterialUBO(perm.perMaterialUBOs[i], matIndices[i]);
        }

        int permIdx = static_cast<int>(m_permutations.size());
        for (int mi : matIndices) {
            m_materialToPermutation[mi] = permIdx;
        }

        std::cout << "[metal] Permutation \"" << suffix << "\" with " << matIndices.size()
                  << " materials" << std::endl;

        m_permutations.push_back(std::move(perm));
    }
}

// --------------------------------------------------------------------------
// Fill material UBO
// --------------------------------------------------------------------------

void MetalRasterRenderer::fillMaterialUBO(MTL::Buffer* buffer, int materialIndex) {
    MaterialUBOData ubo;
    if (m_scene->hasGltfScene() && materialIndex >= 0 &&
        materialIndex < static_cast<int>(m_scene->getGltfScene().materials.size())) {

        auto& mat = m_scene->getGltfScene().materials[materialIndex];
        ubo.baseColorFactor = mat.baseColor;
        ubo.metallicFactor = mat.metallic;
        ubo.roughnessFactor = mat.roughness;
        ubo.emissiveFactor = mat.emissive;
        ubo.emissiveStrength = mat.emissiveStrength;
        ubo.ior = mat.ior;
        ubo.clearcoatFactor = mat.clearcoatFactor;
        ubo.clearcoatRoughnessFactor = mat.clearcoatRoughnessFactor;
        ubo.sheenColorFactor = mat.sheenColorFactor;
        ubo.sheenRoughnessFactor = mat.sheenRoughnessFactor;
        ubo.transmissionFactor = mat.transmissionFactor;

        // UV transforms
        {
            auto& t = mat.base_color_uv_xform;
            ubo.baseColorUvSt = glm::vec4(t.offset.x, t.offset.y, t.scale.x, t.scale.y);
            ubo.baseColorUvRot = t.rotation;
        }
        {
            auto& t = mat.normal_uv_xform;
            ubo.normalUvSt = glm::vec4(t.offset.x, t.offset.y, t.scale.x, t.scale.y);
            ubo.normalUvRot = t.rotation;
        }
        {
            auto& t = mat.metallic_roughness_uv_xform;
            ubo.mrUvSt = glm::vec4(t.offset.x, t.offset.y, t.scale.x, t.scale.y);
            ubo.mrUvRot = t.rotation;
        }
    }
    memcpy(buffer->contents(), &ubo, sizeof(ubo));
}

// --------------------------------------------------------------------------
// Update MVP
// --------------------------------------------------------------------------

void MetalRasterRenderer::updateMVPBuffer() {
    // Already written to m_mvpBuffer->contents() directly
}

void MetalRasterRenderer::updateCamera(glm::vec3 eye, glm::vec3 target,
                                        glm::vec3 up, float fovY, float aspect,
                                        float nearPlane, float farPlane) {
    if (!m_mvpBuffer) return;

    glm::mat4 model(1.0f);
    glm::mat4 view = Camera::lookAt(eye, target, up);
    glm::mat4 proj = Camera::perspective(fovY, aspect, nearPlane, farPlane);

    struct { glm::mat4 model, view, proj; } mvpData = {model, view, proj};
    memcpy(m_mvpBuffer->contents(), &mvpData, sizeof(mvpData));

    // Update Light uniform (preserve light_dir, update view_pos)
    if (m_lightBuffer) {
        struct LightData {
            glm::vec3 lightDir; float _pad0;
            glm::vec3 viewPos; float _pad1;
        };
        LightData light;
        memcpy(&light, m_lightBuffer->contents(), sizeof(LightData));
        light.viewPos = eye;
        memcpy(m_lightBuffer->contents(), &light, sizeof(LightData));
    }

    // Update push constant view_pos (for shaders using push constants instead of UBO)
    if (m_pushConstantSize > 0 && !m_pushConstantData.empty()) {
        auto updatePC = [&](const ReflectionData& refl) {
            for (auto& pc : refl.push_constants) {
                for (auto& f : pc.fields) {
                    if (f.name == "view_pos" && static_cast<uint32_t>(f.offset) + 12 <= m_pushConstantSize) {
                        memcpy(m_pushConstantData.data() + f.offset, &eye, sizeof(glm::vec3));
                    }
                }
            }
        };
        updatePC(m_vertReflection);
        updatePC(m_fragReflection);
    }
}

// --------------------------------------------------------------------------
// Resource binding + draw
// --------------------------------------------------------------------------

void MetalRasterRenderer::bindResourcesAndDraw(MTL::RenderCommandEncoder* enc,
                                                const TranspiledShader& vertShader,
                                                const TranspiledShader& fragShader,
                                                int materialIndex,
                                                MTL::Buffer* materialUBO) {
    // Bind vertex buffer at index 0 (stage_in)
    auto& mesh = m_scene->getMesh();
    enc->setVertexBuffer(mesh.vertexBuffer, 0, 0);

    // Bind uniform buffers by looking up SPIR-V bindings in transpiler maps
    // MVP buffer is typically (set=0, binding=0) in vertex stage
    for (auto& b : vertShader.bindings) {
        if (b.mslBuffer != UINT32_MAX) {
            if (b.name.find("MVP") != std::string::npos || b.name.find("mvp") != std::string::npos ||
                b.name.find("Matrices") != std::string::npos || b.name.find("matrices") != std::string::npos) {
                if (m_mvpBuffer) enc->setVertexBuffer(m_mvpBuffer, 0, b.mslBuffer);
            }
        }
    }

    // Bind fragment stage resources
    for (auto& b : fragShader.bindings) {
        if (b.mslBuffer != UINT32_MAX) {
            // Light buffer
            if (b.name.find("Light") != std::string::npos || b.name.find("light") != std::string::npos) {
                if (m_lightBuffer) enc->setFragmentBuffer(m_lightBuffer, 0, b.mslBuffer);
            }
            // Material buffer
            else if (b.name.find("Material") != std::string::npos || b.name.find("material") != std::string::npos ||
                     b.name.find("Properties") != std::string::npos || b.name.find("properties") != std::string::npos) {
                if (materialUBO) {
                    enc->setFragmentBuffer(materialUBO, 0, b.mslBuffer);
                } else if (materialIndex >= 0 && materialIndex < static_cast<int>(m_perMaterialUBOs.size())) {
                    enc->setFragmentBuffer(m_perMaterialUBOs[materialIndex], 0, b.mslBuffer);
                } else if (m_materialBuffer) {
                    enc->setFragmentBuffer(m_materialBuffer, 0, b.mslBuffer);
                }
            }
        }

        // Bind textures
        if (b.mslTexture != UINT32_MAX) {
            auto& tex = m_scene->getTextureForBinding(b.name, materialIndex);
            if (tex.texture) {
                enc->setFragmentTexture(tex.texture, b.mslTexture);
            }
            if (b.mslSampler != UINT32_MAX && tex.sampler) {
                enc->setFragmentSamplerState(tex.sampler, b.mslSampler);
            }
        }
    }

    // Push constants
    if (m_pushConstantSize > 0 && !m_pushConstantData.empty()) {
        if (vertShader.pushConstantBufferIndex != UINT32_MAX) {
            enc->setVertexBytes(m_pushConstantData.data(), m_pushConstantData.size(),
                               vertShader.pushConstantBufferIndex);
        }
        if (fragShader.pushConstantBufferIndex != UINT32_MAX) {
            enc->setFragmentBytes(m_pushConstantData.data(), m_pushConstantData.size(),
                                 fragShader.pushConstantBufferIndex);
        }
    }

    // Draw
    enc->drawIndexedPrimitives(MTL::PrimitiveTypeTriangle,
                               mesh.indexCount, MTL::IndexTypeUInt32,
                               mesh.indexBuffer, 0);
}

// --------------------------------------------------------------------------
// Main init
// --------------------------------------------------------------------------

void MetalRasterRenderer::init(MetalContext& ctx, MetalSceneManager& scene,
                                const std::string& pipelineBase,
                                const std::string& renderPath,
                                uint32_t width, uint32_t height) {
    m_scene = &scene;
    m_pipelineBase = pipelineBase;
    m_renderPath = renderPath;
    m_width = width;
    m_height = height;
    m_needsDepth = (renderPath != "fullscreen" && renderPath != "triangle");

    m_transpiler.init(ctx.device);
    createRenderTargets(ctx);

    if (m_needsDepth) {
        createDepthStencilState(ctx);
    }

    if (renderPath == "triangle") {
        createPipelineTriangle(ctx);
    } else if (renderPath == "fullscreen") {
        createPipelineFullscreen(ctx);
    } else {
        createPipelinePBR(ctx);
    }

    std::cout << "[metal] Raster renderer initialized: " << renderPath
              << " (" << width << "x" << height << ")" << std::endl;
}

// --------------------------------------------------------------------------
// Render to internal target
// --------------------------------------------------------------------------

void MetalRasterRenderer::renderToTarget(MetalContext& ctx, MTL::RenderPassDescriptor* rpDesc) {
    auto* cmdBuf = ctx.beginCommandBuffer();
    auto* enc = cmdBuf->renderCommandEncoder(rpDesc);

    enc->setViewport({0.0, 0.0, static_cast<double>(m_width), static_cast<double>(m_height), 0.0, 1.0});
    enc->setScissorRect({0, 0, m_width, m_height});

    if (m_pipelineState) {
        enc->setRenderPipelineState(m_pipelineState);
    }

    if (m_depthStencilState) {
        enc->setDepthStencilState(m_depthStencilState);
    }

    if (m_renderPath != "triangle" && m_renderPath != "fullscreen") {
        // Flip winding because Vulkan Y-flip in projection inverts winding
        enc->setFrontFacingWinding(MTL::WindingClockwise);
        enc->setCullMode(MTL::CullModeBack);
    }

    if (m_renderPath == "triangle") {
        enc->setVertexBuffer(m_triangleVB, 0, 0);
        enc->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0), NS::UInteger(3));
    } else if (m_renderPath == "fullscreen") {
        // Bind fragment resources
        for (auto& b : m_fragmentShader.bindings) {
            if (b.mslTexture != UINT32_MAX) {
                auto& tex = m_scene->getTextureForBinding(b.name);
                if (tex.texture) enc->setFragmentTexture(tex.texture, b.mslTexture);
                if (b.mslSampler != UINT32_MAX && tex.sampler) {
                    enc->setFragmentSamplerState(tex.sampler, b.mslSampler);
                }
            }
        }
        enc->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0), NS::UInteger(3));
    } else if (m_multiPipeline) {
        // Multi-material: draw per permutation, per material
        auto& drawRanges = m_scene->getDrawRanges();
        for (auto& perm : m_permutations) {
            enc->setRenderPipelineState(perm.pipelineState);

            for (size_t pi = 0; pi < perm.materialIndices.size(); pi++) {
                int matIdx = perm.materialIndices[pi];
                MTL::Buffer* matUBO = perm.perMaterialUBOs[pi];

                // Find draw ranges for this material
                for (auto& dr : drawRanges) {
                    if (dr.materialIndex == matIdx) {
                        // Bind vertex buffer
                        enc->setVertexBuffer(m_scene->getMesh().vertexBuffer, 0, 0);

                        // Bind MVP to vertex stage
                        for (auto& b : perm.vertShader.bindings) {
                            if (b.mslBuffer != UINT32_MAX &&
                                (b.name.find("MVP") != std::string::npos ||
                                 b.name.find("Matrices") != std::string::npos)) {
                                enc->setVertexBuffer(m_mvpBuffer, 0, b.mslBuffer);
                            }
                        }

                        // Bind fragment resources
                        for (auto& b : perm.fragShader.bindings) {
                            if (b.mslBuffer != UINT32_MAX) {
                                if (b.name.find("Light") != std::string::npos) {
                                    enc->setFragmentBuffer(m_lightBuffer, 0, b.mslBuffer);
                                } else if (b.name.find("Material") != std::string::npos ||
                                           b.name.find("Properties") != std::string::npos) {
                                    enc->setFragmentBuffer(matUBO, 0, b.mslBuffer);
                                }
                            }
                            if (b.mslTexture != UINT32_MAX) {
                                auto& tex = m_scene->getTextureForBinding(b.name, matIdx);
                                if (tex.texture) enc->setFragmentTexture(tex.texture, b.mslTexture);
                                if (b.mslSampler != UINT32_MAX && tex.sampler) {
                                    enc->setFragmentSamplerState(tex.sampler, b.mslSampler);
                                }
                            }
                        }

                        // Push constants
                        if (m_pushConstantSize > 0) {
                            if (perm.vertShader.pushConstantBufferIndex != UINT32_MAX) {
                                enc->setVertexBytes(m_pushConstantData.data(), m_pushConstantData.size(),
                                                   perm.vertShader.pushConstantBufferIndex);
                            }
                            if (perm.fragShader.pushConstantBufferIndex != UINT32_MAX) {
                                enc->setFragmentBytes(m_pushConstantData.data(), m_pushConstantData.size(),
                                                     perm.fragShader.pushConstantBufferIndex);
                            }
                        }

                        enc->drawIndexedPrimitives(MTL::PrimitiveTypeTriangle,
                                                   dr.indexCount, MTL::IndexTypeUInt32,
                                                   m_scene->getMesh().indexBuffer,
                                                   dr.indexOffset * sizeof(uint32_t));
                    }
                }
            }
        }
    } else {
        // Single pipeline: draw per material draw range or single draw
        auto& drawRanges = m_scene->getDrawRanges();
        if (!drawRanges.empty()) {
            auto& mesh = m_scene->getMesh();
            enc->setVertexBuffer(mesh.vertexBuffer, 0, 0);

            for (auto& dr : drawRanges) {
                // Bind uniform buffers per material
                for (auto& b : m_vertexShader.bindings) {
                    if (b.mslBuffer != UINT32_MAX &&
                        (b.name.find("MVP") != std::string::npos ||
                         b.name.find("Matrices") != std::string::npos)) {
                        if (m_mvpBuffer) enc->setVertexBuffer(m_mvpBuffer, 0, b.mslBuffer);
                    }
                }
                for (auto& b : m_fragmentShader.bindings) {
                    if (b.mslBuffer != UINT32_MAX) {
                        if (b.name.find("Light") != std::string::npos) {
                            if (m_lightBuffer) enc->setFragmentBuffer(m_lightBuffer, 0, b.mslBuffer);
                        } else if (b.name.find("Material") != std::string::npos ||
                                   b.name.find("Properties") != std::string::npos) {
                            if (dr.materialIndex >= 0 && dr.materialIndex < static_cast<int>(m_perMaterialUBOs.size())) {
                                enc->setFragmentBuffer(m_perMaterialUBOs[dr.materialIndex], 0, b.mslBuffer);
                            } else if (m_materialBuffer) {
                                enc->setFragmentBuffer(m_materialBuffer, 0, b.mslBuffer);
                            }
                        }
                    }
                    if (b.mslTexture != UINT32_MAX) {
                        auto& tex = m_scene->getTextureForBinding(b.name, dr.materialIndex);
                        if (tex.texture) enc->setFragmentTexture(tex.texture, b.mslTexture);
                        if (b.mslSampler != UINT32_MAX && tex.sampler) {
                            enc->setFragmentSamplerState(tex.sampler, b.mslSampler);
                        }
                    }
                }
                // Push constants
                if (m_pushConstantSize > 0 && !m_pushConstantData.empty()) {
                    if (m_vertexShader.pushConstantBufferIndex != UINT32_MAX) {
                        enc->setVertexBytes(m_pushConstantData.data(), m_pushConstantData.size(),
                                           m_vertexShader.pushConstantBufferIndex);
                    }
                    if (m_fragmentShader.pushConstantBufferIndex != UINT32_MAX) {
                        enc->setFragmentBytes(m_pushConstantData.data(), m_pushConstantData.size(),
                                             m_fragmentShader.pushConstantBufferIndex);
                    }
                }
                // Draw only this range
                enc->drawIndexedPrimitives(MTL::PrimitiveTypeTriangle,
                                           dr.indexCount, MTL::IndexTypeUInt32,
                                           mesh.indexBuffer,
                                           dr.indexOffset * sizeof(uint32_t));
            }
        } else {
            bindResourcesAndDraw(enc, m_vertexShader, m_fragmentShader);
        }
    }

    enc->endEncoding();
    ctx.submitAndWait(cmdBuf);
}

void MetalRasterRenderer::render(MetalContext& ctx) {
    auto* rpDesc = MTL::RenderPassDescriptor::alloc()->init();

    auto* colorAtt = rpDesc->colorAttachments()->object(0);
    colorAtt->setTexture(m_colorTarget);
    colorAtt->setLoadAction(MTL::LoadActionClear);
    colorAtt->setStoreAction(MTL::StoreActionStore);
    colorAtt->setClearColor(MTL::ClearColor(0.1, 0.1, 0.15, 1.0));

    if (m_needsDepth && m_depthTarget) {
        auto* depthAtt = rpDesc->depthAttachment();
        depthAtt->setTexture(m_depthTarget);
        depthAtt->setLoadAction(MTL::LoadActionClear);
        depthAtt->setStoreAction(MTL::StoreActionDontCare);
        depthAtt->setClearDepth(1.0);
    }

    renderToTarget(ctx, rpDesc);
    rpDesc->release();
}

void MetalRasterRenderer::renderToDrawable(MetalContext& ctx, CA::MetalDrawable* drawable) {
    // Use drawable texture dimensions for viewport (handles resize)
    auto* drawableTex = drawable->texture();
    uint32_t drawW = static_cast<uint32_t>(drawableTex->width());
    uint32_t drawH = static_cast<uint32_t>(drawableTex->height());

    // Rebuild depth buffer if drawable size changed
    if (m_needsDepth && m_depthTarget &&
        (m_depthTarget->width() != drawW || m_depthTarget->height() != drawH)) {
        m_depthTarget->release();
        auto* depthDesc = MTL::TextureDescriptor::texture2DDescriptor(
            MTL::PixelFormatDepth32Float, drawW, drawH, false);
        depthDesc->setUsage(MTL::TextureUsageRenderTarget);
        depthDesc->setStorageMode(MTL::StorageModePrivate);
        m_depthTarget = ctx.newTexture(depthDesc);
    }

    auto* rpDesc = MTL::RenderPassDescriptor::alloc()->init();

    auto* colorAtt = rpDesc->colorAttachments()->object(0);
    colorAtt->setTexture(drawableTex);
    colorAtt->setLoadAction(MTL::LoadActionClear);
    colorAtt->setStoreAction(MTL::StoreActionStore);
    colorAtt->setClearColor(MTL::ClearColor(0.1, 0.1, 0.15, 1.0));

    if (m_needsDepth && m_depthTarget) {
        auto* depthAtt = rpDesc->depthAttachment();
        depthAtt->setTexture(m_depthTarget);
        depthAtt->setLoadAction(MTL::LoadActionClear);
        depthAtt->setStoreAction(MTL::StoreActionDontCare);
        depthAtt->setClearDepth(1.0);
    }

    auto* cmdBuf = ctx.beginCommandBuffer();
    auto* enc = cmdBuf->renderCommandEncoder(rpDesc);

    enc->setViewport({0.0, 0.0, static_cast<double>(drawW), static_cast<double>(drawH), 0.0, 1.0});
    enc->setScissorRect({0, 0, drawW, drawH});

    if (m_depthStencilState) enc->setDepthStencilState(m_depthStencilState);

    if (m_renderPath != "triangle" && m_renderPath != "fullscreen") {
        enc->setFrontFacingWinding(MTL::WindingClockwise);
        enc->setCullMode(MTL::CullModeBack);
    }

    if (m_renderPath == "triangle") {
        enc->setRenderPipelineState(m_pipelineState);
        enc->setVertexBuffer(m_triangleVB, 0, 0);
        enc->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0), NS::UInteger(3));
    } else if (m_renderPath == "fullscreen") {
        enc->setRenderPipelineState(m_pipelineState);
        for (auto& b : m_fragmentShader.bindings) {
            if (b.mslTexture != UINT32_MAX) {
                auto& tex = m_scene->getTextureForBinding(b.name);
                if (tex.texture) enc->setFragmentTexture(tex.texture, b.mslTexture);
                if (b.mslSampler != UINT32_MAX && tex.sampler) {
                    enc->setFragmentSamplerState(tex.sampler, b.mslSampler);
                }
            }
        }
        enc->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0), NS::UInteger(3));
    } else if (m_multiPipeline) {
        // Multi-material: draw per permutation, per material
        auto& drawRanges = m_scene->getDrawRanges();
        for (auto& perm : m_permutations) {
            enc->setRenderPipelineState(perm.pipelineState);

            for (size_t pi = 0; pi < perm.materialIndices.size(); pi++) {
                int matIdx = perm.materialIndices[pi];
                MTL::Buffer* matUBO = perm.perMaterialUBOs[pi];

                for (auto& dr : drawRanges) {
                    if (dr.materialIndex == matIdx) {
                        enc->setVertexBuffer(m_scene->getMesh().vertexBuffer, 0, 0);
                        for (auto& b : perm.vertShader.bindings) {
                            if (b.mslBuffer != UINT32_MAX &&
                                (b.name.find("MVP") != std::string::npos ||
                                 b.name.find("Matrices") != std::string::npos)) {
                                enc->setVertexBuffer(m_mvpBuffer, 0, b.mslBuffer);
                            }
                        }
                        for (auto& b : perm.fragShader.bindings) {
                            if (b.mslBuffer != UINT32_MAX) {
                                if (b.name.find("Light") != std::string::npos) {
                                    enc->setFragmentBuffer(m_lightBuffer, 0, b.mslBuffer);
                                } else if (b.name.find("Material") != std::string::npos ||
                                           b.name.find("Properties") != std::string::npos) {
                                    enc->setFragmentBuffer(matUBO, 0, b.mslBuffer);
                                }
                            }
                            if (b.mslTexture != UINT32_MAX) {
                                auto& tex = m_scene->getTextureForBinding(b.name, matIdx);
                                if (tex.texture) enc->setFragmentTexture(tex.texture, b.mslTexture);
                                if (b.mslSampler != UINT32_MAX && tex.sampler) {
                                    enc->setFragmentSamplerState(tex.sampler, b.mslSampler);
                                }
                            }
                        }
                        if (m_pushConstantSize > 0) {
                            if (perm.vertShader.pushConstantBufferIndex != UINT32_MAX)
                                enc->setVertexBytes(m_pushConstantData.data(), m_pushConstantData.size(),
                                                   perm.vertShader.pushConstantBufferIndex);
                            if (perm.fragShader.pushConstantBufferIndex != UINT32_MAX)
                                enc->setFragmentBytes(m_pushConstantData.data(), m_pushConstantData.size(),
                                                     perm.fragShader.pushConstantBufferIndex);
                        }
                        enc->drawIndexedPrimitives(MTL::PrimitiveTypeTriangle,
                                                   dr.indexCount, MTL::IndexTypeUInt32,
                                                   m_scene->getMesh().indexBuffer,
                                                   dr.indexOffset * sizeof(uint32_t));
                    }
                }
            }
        }
    } else {
        // Single pipeline with per-range draws
        enc->setRenderPipelineState(m_pipelineState);
        auto& drawRanges = m_scene->getDrawRanges();
        if (!drawRanges.empty()) {
            auto& mesh = m_scene->getMesh();
            enc->setVertexBuffer(mesh.vertexBuffer, 0, 0);
            for (auto& dr : drawRanges) {
                for (auto& b : m_vertexShader.bindings) {
                    if (b.mslBuffer != UINT32_MAX &&
                        (b.name.find("MVP") != std::string::npos ||
                         b.name.find("Matrices") != std::string::npos)) {
                        if (m_mvpBuffer) enc->setVertexBuffer(m_mvpBuffer, 0, b.mslBuffer);
                    }
                }
                for (auto& b : m_fragmentShader.bindings) {
                    if (b.mslBuffer != UINT32_MAX) {
                        if (b.name.find("Light") != std::string::npos) {
                            if (m_lightBuffer) enc->setFragmentBuffer(m_lightBuffer, 0, b.mslBuffer);
                        } else if (b.name.find("Material") != std::string::npos ||
                                   b.name.find("Properties") != std::string::npos) {
                            if (dr.materialIndex >= 0 && dr.materialIndex < static_cast<int>(m_perMaterialUBOs.size()))
                                enc->setFragmentBuffer(m_perMaterialUBOs[dr.materialIndex], 0, b.mslBuffer);
                            else if (m_materialBuffer)
                                enc->setFragmentBuffer(m_materialBuffer, 0, b.mslBuffer);
                        }
                    }
                    if (b.mslTexture != UINT32_MAX) {
                        auto& tex = m_scene->getTextureForBinding(b.name, dr.materialIndex);
                        if (tex.texture) enc->setFragmentTexture(tex.texture, b.mslTexture);
                        if (b.mslSampler != UINT32_MAX && tex.sampler)
                            enc->setFragmentSamplerState(tex.sampler, b.mslSampler);
                    }
                }
                if (m_pushConstantSize > 0 && !m_pushConstantData.empty()) {
                    if (m_vertexShader.pushConstantBufferIndex != UINT32_MAX)
                        enc->setVertexBytes(m_pushConstantData.data(), m_pushConstantData.size(),
                                           m_vertexShader.pushConstantBufferIndex);
                    if (m_fragmentShader.pushConstantBufferIndex != UINT32_MAX)
                        enc->setFragmentBytes(m_pushConstantData.data(), m_pushConstantData.size(),
                                             m_fragmentShader.pushConstantBufferIndex);
                }
                enc->drawIndexedPrimitives(MTL::PrimitiveTypeTriangle,
                                           dr.indexCount, MTL::IndexTypeUInt32,
                                           mesh.indexBuffer,
                                           dr.indexOffset * sizeof(uint32_t));
            }
        } else {
            bindResourcesAndDraw(enc, m_vertexShader, m_fragmentShader);
        }
    }

    enc->endEncoding();
    cmdBuf->presentDrawable(drawable);
    cmdBuf->commit();

    rpDesc->release();
}

// --------------------------------------------------------------------------
// Cleanup
// --------------------------------------------------------------------------

void MetalRasterRenderer::cleanup() {
    m_vertexShader.cleanup();
    m_fragmentShader.cleanup();

    if (m_pipelineState) { m_pipelineState->release(); m_pipelineState = nullptr; }
    if (m_depthStencilState) { m_depthStencilState->release(); m_depthStencilState = nullptr; }

    if (m_colorTarget) { m_colorTarget->release(); m_colorTarget = nullptr; }
    if (m_depthTarget) { m_depthTarget->release(); m_depthTarget = nullptr; }

    if (m_mvpBuffer) { m_mvpBuffer->release(); m_mvpBuffer = nullptr; }
    if (m_lightBuffer) { m_lightBuffer->release(); m_lightBuffer = nullptr; }
    if (m_materialBuffer) { m_materialBuffer->release(); m_materialBuffer = nullptr; }
    if (m_triangleVB) { m_triangleVB->release(); m_triangleVB = nullptr; }

    for (auto* buf : m_perMaterialUBOs) { if (buf) buf->release(); }
    m_perMaterialUBOs.clear();

    for (auto& perm : m_permutations) {
        perm.vertShader.cleanup();
        perm.fragShader.cleanup();
        if (perm.pipelineState) perm.pipelineState->release();
        for (auto* buf : perm.perMaterialUBOs) { if (buf) buf->release(); }
    }
    m_permutations.clear();

    if (m_fullscreenVertFunc) { m_fullscreenVertFunc->release(); m_fullscreenVertFunc = nullptr; }
    if (m_fullscreenLib) { m_fullscreenLib->release(); m_fullscreenLib = nullptr; }

    m_transpiler.cleanup();
}
