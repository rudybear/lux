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
    std::string vertJsonPath = m_pipelineBase + ".vert.json";
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
    std::string vertJsonPath = m_pipelineBase + ".vert.json";
    std::string fragJsonPath = m_pipelineBase + ".frag.json";
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
                std::string vertJson = candidateBase + ".vert.json";
                std::string fragJson = candidateBase + ".frag.json";
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

    // Detect multi-light support from fragment shader bindings
    setupMultiLightResources(ctx);

    // Detect shadow support from fragment shader bindings and set up shadow maps
    {
        bool hasShadowMaps = false, hasShadowMatrices = false;
        auto checkBindings = [&](const TranspiledShader& shader) {
            for (auto& b : shader.bindings) {
                if (b.name == "shadow_maps") hasShadowMaps = true;
                if (b.name == "shadow_matrices" && b.mslBuffer != UINT32_MAX) hasShadowMatrices = true;
            }
        };
        checkBindings(m_fragmentShader);
        // Also check first permutation if multi-pipeline
        if (!m_permutations.empty()) {
            checkBindings(m_permutations[0].fragShader);
        }

        if (hasShadowMaps && hasShadowMatrices) {
            std::cout << "[metal] Shader requests shadow maps, setting up shadow infrastructure" << std::endl;
            auto& lights = m_scene->getLightsMutable();
            for (auto& l : lights) {
                if (l.type == SceneLight::Directional || l.type == SceneLight::Spot)
                    l.castsShadow = true;
            }
            setupShadowMaps(ctx);

            // Re-pack lights SSBO with updated shadow indices
            if (m_hasMultiLight && m_lightsSSBO && m_hasShadows) {
                float nearClip = 0.1f;
                float farClip = m_scene->hasSceneBounds() ? m_scene->getAutoFar() : 100.0f;
                float aspect = static_cast<float>(m_width) / static_cast<float>(m_height);
                glm::mat4 view = m_scene->hasSceneBounds()
                    ? glm::lookAt(m_scene->getAutoEye(), m_scene->getAutoTarget(), m_scene->getAutoUp())
                    : Camera::lookAt(Camera::DEFAULT_EYE, Camera::DEFAULT_TARGET, Camera::DEFAULT_UP);
                glm::mat4 proj = m_scene->hasSceneBounds()
                    ? glm::perspective(glm::radians(45.0f), aspect, nearClip, farClip)
                    : Camera::perspective(Camera::DEFAULT_FOV, aspect, nearClip, farClip);
                m_scene->computeShadowData(view, proj, nearClip, farClip);
                auto lightBuf = m_scene->packLightsBuffer();
                if (!lightBuf.empty()) {
                    memcpy(m_lightsSSBO->contents(), lightBuf.data(), lightBuf.size() * sizeof(float));
                }
                auto shadowBuf = m_scene->packShadowBuffer();
                if (!shadowBuf.empty()) {
                    memcpy(m_shadowMatricesBuffer->contents(), shadowBuf.data(),
                           shadowBuf.size() * sizeof(float));
                }
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
        std::string vertJson = permBase + ".vert.json";
        std::string fragJson = permBase + ".frag.json";
        if (!fs::exists(vertJson)) vertJson = m_pipelineBase + ".vert.json";
        if (!fs::exists(fragJson)) fragJson = m_pipelineBase + ".frag.json";

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
// Embedded shadow depth vertex shader (MSL)
// --------------------------------------------------------------------------
// Minimal vertex shader for shadow pass:
//   Takes a mat4 MVP via buffer and position via vertex input, outputs gl_Position.

static const char* kShadowVertMSL = R"(
#include <metal_stdlib>
using namespace metal;

struct ShadowPC {
    float4x4 mvp;
};

struct VertexIn {
    float3 position [[attribute(0)]];
};

struct VertexOut {
    float4 position [[position]];
};

vertex VertexOut shadow_vert(VertexIn in [[stage_in]],
                             constant ShadowPC& pc [[buffer(1)]]) {
    VertexOut out;
    out.position = pc.mvp * float4(in.position, 1.0);
    return out;
}
)";

// --------------------------------------------------------------------------
// Multi-light resource setup
// --------------------------------------------------------------------------

void MetalRasterRenderer::setupMultiLightResources(MetalContext& ctx) {
    // Detect from fragment reflection whether the shader uses a "lights" storage buffer
    m_hasMultiLight = false;
    for (auto& b : m_fragmentShader.bindings) {
        if (b.name == "lights" && b.mslBuffer != UINT32_MAX) {
            m_hasMultiLight = true;
            break;
        }
    }

    // Also check permutation shaders
    if (!m_hasMultiLight && !m_permutations.empty()) {
        for (auto& b : m_permutations[0].fragShader.bindings) {
            if (b.name == "lights" && b.mslBuffer != UINT32_MAX) {
                m_hasMultiLight = true;
                break;
            }
        }
    }

    if (m_hasMultiLight && m_scene) {
        // Create lights SSBO from packed scene light data
        auto lightBuf = m_scene->packLightsBuffer();
        if (lightBuf.empty()) {
            // Add a dummy light to avoid empty buffer
            lightBuf.resize(16, 0.0f);
        }
        m_lightsSSBO = ctx.newBuffer(lightBuf.data(), lightBuf.size() * sizeof(float),
                                      MTL::ResourceStorageModeShared);

        // Create SceneLight UBO: vec3 view_pos at offset 0, int light_count at offset 12
        struct SceneLightUBO {
            glm::vec3 viewPos;
            int32_t lightCount;
        };
        glm::vec3 viewPos = m_scene->hasSceneBounds() ? m_scene->getAutoEye() : Camera::DEFAULT_EYE;
        SceneLightUBO sceneLightData;
        sceneLightData.viewPos = viewPos;
        sceneLightData.lightCount = m_scene->getLightCount();
        m_sceneLightUBO = ctx.newBuffer(&sceneLightData, sizeof(SceneLightUBO),
                                         MTL::ResourceStorageModeShared);

        std::cout << "[metal] Multi-light SSBO created: " << m_scene->getLightCount()
                  << " light(s), " << lightBuf.size() * sizeof(float) << " bytes" << std::endl;
    }
}

// --------------------------------------------------------------------------
// Shadow map infrastructure setup
// --------------------------------------------------------------------------

void MetalRasterRenderer::setupShadowMaps(MetalContext& ctx) {
    // 1. Create 2D array depth texture (MAX_SHADOW_MAPS layers)
    auto* texDesc = MTL::TextureDescriptor::alloc()->init();
    texDesc->setTextureType(MTL::TextureType2DArray);
    texDesc->setPixelFormat(MTL::PixelFormatDepth32Float);
    texDesc->setWidth(SHADOW_MAP_SIZE);
    texDesc->setHeight(SHADOW_MAP_SIZE);
    texDesc->setArrayLength(MAX_SHADOW_MAPS);
    texDesc->setMipmapLevelCount(1);
    texDesc->setUsage(MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead);
    texDesc->setStorageMode(MTL::StorageModePrivate);

    m_shadowDepthArray = ctx.newTexture(texDesc);
    texDesc->release();
    if (!m_shadowDepthArray) {
        std::cerr << "[metal][warn] Failed to create shadow map depth array, disabling shadows" << std::endl;
        m_hasShadows = false;
        return;
    }
    m_shadowDepthArray->setLabel(NS::String::string("ShadowDepthArray", NS::UTF8StringEncoding));

    // 2. Create comparison sampler (for shadow sampling with PCF)
    auto* samplerDesc = MTL::SamplerDescriptor::alloc()->init();
    samplerDesc->setMinFilter(MTL::SamplerMinMagFilterLinear);
    samplerDesc->setMagFilter(MTL::SamplerMinMagFilterLinear);
    samplerDesc->setSAddressMode(MTL::SamplerAddressModeClampToBorderColor);
    samplerDesc->setTAddressMode(MTL::SamplerAddressModeClampToBorderColor);
    samplerDesc->setRAddressMode(MTL::SamplerAddressModeClampToBorderColor);
    samplerDesc->setBorderColor(MTL::SamplerBorderColorOpaqueWhite);
    samplerDesc->setCompareFunction(MTL::CompareFunctionLessEqual);
    m_shadowSampler = ctx.device->newSamplerState(samplerDesc);
    samplerDesc->release();

    // 3. Compile shadow vertex shader from embedded MSL
    NS::Error* error = nullptr;
    auto* mslSource = NS::String::string(kShadowVertMSL, NS::UTF8StringEncoding);
    auto* compileOpts = MTL::CompileOptions::alloc()->init();
    m_shadowShaderLib = ctx.device->newLibrary(mslSource, compileOpts, &error);
    compileOpts->release();

    if (!m_shadowShaderLib) {
        std::cerr << "[metal][warn] Failed to compile shadow vertex shader";
        if (error) std::cerr << ": " << error->localizedDescription()->utf8String();
        std::cerr << std::endl;
        m_hasShadows = false;
        return;
    }

    auto* funcName = NS::String::string("shadow_vert", NS::UTF8StringEncoding);
    m_shadowVertFunc = m_shadowShaderLib->newFunction(funcName);

    // 4. Create depth-only render pipeline state (vertex only, no fragment output)
    auto* pipeDesc = MTL::RenderPipelineDescriptor::alloc()->init();
    pipeDesc->setVertexFunction(m_shadowVertFunc);
    pipeDesc->setFragmentFunction(nullptr);  // No fragment shader for depth-only
    pipeDesc->setDepthAttachmentPixelFormat(MTL::PixelFormatDepth32Float);
    // No color attachments for shadow pass

    // Vertex descriptor: only position at location 0
    int vbStride = (m_scene && m_scene->hasGltfScene()) ? 48 : 32;
    auto* vertDesc = MTL::VertexDescriptor::alloc()->init();
    vertDesc->layouts()->object(0)->setStride(vbStride);
    vertDesc->layouts()->object(0)->setStepFunction(MTL::VertexStepFunctionPerVertex);
    vertDesc->attributes()->object(0)->setFormat(MTL::VertexFormatFloat3);
    vertDesc->attributes()->object(0)->setOffset(0);
    vertDesc->attributes()->object(0)->setBufferIndex(0);
    pipeDesc->setVertexDescriptor(vertDesc);
    vertDesc->release();

    m_shadowPipeline = ctx.device->newRenderPipelineState(pipeDesc, &error);
    pipeDesc->release();

    if (!m_shadowPipeline) {
        std::cerr << "[metal][warn] Failed to create shadow pipeline";
        if (error) std::cerr << ": " << error->localizedDescription()->utf8String();
        std::cerr << std::endl;
        m_hasShadows = false;
        return;
    }

    // 5. Create depth stencil state for shadow pass (depth write + test enabled)
    auto* dsDesc = MTL::DepthStencilDescriptor::alloc()->init();
    dsDesc->setDepthCompareFunction(MTL::CompareFunctionLessEqual);
    dsDesc->setDepthWriteEnabled(true);
    m_shadowDepthState = ctx.device->newDepthStencilState(dsDesc);
    dsDesc->release();

    // 6. Allocate shadow matrices buffer (80 bytes per entry * MAX_SHADOW_MAPS)
    m_shadowMatricesBuffer = ctx.newBuffer(MAX_SHADOW_MAPS * 80,
                                            MTL::ResourceStorageModeShared);

    m_hasShadows = true;
    std::cout << "[metal] Shadow map infrastructure created: " << MAX_SHADOW_MAPS
              << " layers at " << SHADOW_MAP_SIZE << "x" << SHADOW_MAP_SIZE << std::endl;
}

// --------------------------------------------------------------------------
// Render shadow maps for all shadow-casting lights
// --------------------------------------------------------------------------

void MetalRasterRenderer::renderShadowMaps(MetalContext& ctx, MTL::CommandBuffer* cmdBuf) {
    if (!m_hasShadows || !m_scene) return;

    // Compute shadow data using current camera parameters
    struct MVPData { glm::mat4 model, view, projection; };
    MVPData mvp;
    memcpy(&mvp, m_mvpBuffer->contents(), sizeof(MVPData));

    float nearClip = 0.1f;
    float farClip = m_scene->hasSceneBounds() ? m_scene->getAutoFar() : 100.0f;
    m_scene->computeShadowData(mvp.view, mvp.projection, nearClip, farClip);

    m_numShadowMaps = m_scene->getShadowMapCount();
    if (m_numShadowMaps == 0) return;

    const auto& shadowEntries = m_scene->getShadowEntries();
    auto& mesh = m_scene->getMesh();
    if (!mesh.vertexBuffer) return;

    for (int i = 0; i < m_numShadowMaps && i < static_cast<int>(MAX_SHADOW_MAPS); i++) {
        // Create render pass descriptor targeting the i-th layer of the depth array
        auto* rpDesc = MTL::RenderPassDescriptor::alloc()->init();
        auto* depthAtt = rpDesc->depthAttachment();
        depthAtt->setTexture(m_shadowDepthArray);
        depthAtt->setSlice(i);
        depthAtt->setLoadAction(MTL::LoadActionClear);
        depthAtt->setStoreAction(MTL::StoreActionStore);
        depthAtt->setClearDepth(1.0);

        auto* enc = cmdBuf->renderCommandEncoder(rpDesc);

        enc->setViewport({0.0, 0.0, static_cast<double>(SHADOW_MAP_SIZE),
                          static_cast<double>(SHADOW_MAP_SIZE), 0.0, 1.0});
        enc->setScissorRect({0, 0, SHADOW_MAP_SIZE, SHADOW_MAP_SIZE});

        enc->setRenderPipelineState(m_shadowPipeline);
        enc->setDepthStencilState(m_shadowDepthState);
        enc->setCullMode(MTL::CullModeNone);

        // Depth bias to reduce shadow acne
        enc->setDepthBias(1.25f, 1.75f, 0.0f);

        // Bind vertex buffer at index 0
        enc->setVertexBuffer(mesh.vertexBuffer, 0, 0);

        // lightVP push constant at buffer index 1
        glm::mat4 lightMVP = shadowEntries[i].viewProjection;

        const auto& drawRanges = m_scene->getDrawRanges();
        if (!drawRanges.empty()) {
            for (auto& range : drawRanges) {
                enc->setVertexBytes(&lightMVP, sizeof(glm::mat4), 1);
                enc->drawIndexedPrimitives(MTL::PrimitiveTypeTriangle,
                                            range.indexCount, MTL::IndexTypeUInt32,
                                            mesh.indexBuffer,
                                            range.indexOffset * sizeof(uint32_t));
            }
        } else {
            enc->setVertexBytes(&lightMVP, sizeof(glm::mat4), 1);
            enc->drawIndexedPrimitives(MTL::PrimitiveTypeTriangle,
                                        mesh.indexCount, MTL::IndexTypeUInt32,
                                        mesh.indexBuffer, 0);
        }

        enc->endEncoding();
        rpDesc->release();
    }

    // Update shadow matrices buffer
    auto shadowBuf = m_scene->packShadowBuffer();
    if (!shadowBuf.empty()) {
        size_t copySize = std::min(shadowBuf.size() * sizeof(float),
                                    static_cast<size_t>(MAX_SHADOW_MAPS * 80));
        memcpy(m_shadowMatricesBuffer->contents(), shadowBuf.data(), copySize);
    }

    // Update light buffer with shadow indices (repack lights with updated shadow indices)
    if (m_hasMultiLight && m_lightsSSBO) {
        auto lightBuf = m_scene->packLightsBuffer();
        if (!lightBuf.empty()) {
            memcpy(m_lightsSSBO->contents(), lightBuf.data(), lightBuf.size() * sizeof(float));
        }
    }
}

// --------------------------------------------------------------------------
// Bind shadow resources to main pass encoder
// --------------------------------------------------------------------------

void MetalRasterRenderer::bindShadowResources(MTL::RenderCommandEncoder* enc,
                                                const TranspiledShader& fragShader) {
    if (!m_hasShadows) return;

    for (auto& b : fragShader.bindings) {
        // Bind shadow_matrices storage buffer
        if (b.mslBuffer != UINT32_MAX && b.name == "shadow_matrices" && m_shadowMatricesBuffer) {
            enc->setFragmentBuffer(m_shadowMatricesBuffer, 0, b.mslBuffer);
        }

        // Bind shadow_maps texture (depth2d_array)
        if (b.mslTexture != UINT32_MAX && b.name == "shadow_maps" && m_shadowDepthArray) {
            enc->setFragmentTexture(m_shadowDepthArray, b.mslTexture);
        }

        // Bind shadow_maps comparison sampler
        if (b.mslSampler != UINT32_MAX && b.name == "shadow_maps" && m_shadowSampler) {
            enc->setFragmentSamplerState(m_shadowSampler, b.mslSampler);
        }

        // Bind multi-light resources
        if (b.mslBuffer != UINT32_MAX && b.name == "lights" && m_lightsSSBO) {
            enc->setFragmentBuffer(m_lightsSSBO, 0, b.mslBuffer);
        }
        if (b.mslBuffer != UINT32_MAX && b.name == "SceneLight" && m_sceneLightUBO) {
            enc->setFragmentBuffer(m_sceneLightUBO, 0, b.mslBuffer);
        }
    }
}

// --------------------------------------------------------------------------
// Shadow map cleanup
// --------------------------------------------------------------------------

void MetalRasterRenderer::cleanupShadowMaps() {
    if (m_shadowPipeline) { m_shadowPipeline->release(); m_shadowPipeline = nullptr; }
    if (m_shadowDepthState) { m_shadowDepthState->release(); m_shadowDepthState = nullptr; }
    if (m_shadowVertFunc) { m_shadowVertFunc->release(); m_shadowVertFunc = nullptr; }
    if (m_shadowShaderLib) { m_shadowShaderLib->release(); m_shadowShaderLib = nullptr; }
    if (m_shadowDepthArray) { m_shadowDepthArray->release(); m_shadowDepthArray = nullptr; }
    if (m_shadowSampler) { m_shadowSampler->release(); m_shadowSampler = nullptr; }
    if (m_shadowMatricesBuffer) { m_shadowMatricesBuffer->release(); m_shadowMatricesBuffer = nullptr; }
    m_hasShadows = false;
    m_numShadowMaps = 0;
}

// --------------------------------------------------------------------------
// Update material UBO at runtime (editor property editing)
// --------------------------------------------------------------------------

void MetalRasterRenderer::updateMaterialUBO(int materialIndex, const MaterialUBOData& data) {
    // Single-pipeline mode: check per-material UBOs first, then fallback buffer
    if (!m_multiPipeline) {
        if (materialIndex >= 0 && materialIndex < static_cast<int>(m_perMaterialUBOs.size())) {
            memcpy(m_perMaterialUBOs[materialIndex]->contents(), &data, sizeof(data));
            return;
        }
        // Fallback: single material buffer
        if (materialIndex == 0 && m_materialBuffer) {
            memcpy(m_materialBuffer->contents(), &data, sizeof(data));
            return;
        }
    }

    // Multi-pipeline mode: find the permutation containing this material
    if (materialIndex >= 0 && materialIndex < static_cast<int>(m_materialToPermutation.size())) {
        int permIdx = m_materialToPermutation[materialIndex];
        if (permIdx >= 0 && permIdx < static_cast<int>(m_permutations.size())) {
            auto& perm = m_permutations[permIdx];
            for (size_t i = 0; i < perm.materialIndices.size(); i++) {
                if (perm.materialIndices[i] == materialIndex) {
                    memcpy(perm.perMaterialUBOs[i]->contents(), &data, sizeof(data));
                    return;
                }
            }
        }
    }
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

    // Update SceneLight UBO view_pos (for multi-light shaders)
    if (m_sceneLightUBO) {
        struct SceneLightUBO {
            glm::vec3 viewPos;
            int32_t lightCount;
        };
        SceneLightUBO slData;
        memcpy(&slData, m_sceneLightUBO->contents(), sizeof(SceneLightUBO));
        slData.viewPos = eye;
        memcpy(m_sceneLightUBO->contents(), &slData, sizeof(SceneLightUBO));
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

    // Bind shadow/multi-light resources
    bindShadowResources(enc, fragShader);

    // Bind fragment stage resources
    for (auto& b : fragShader.bindings) {
        if (b.mslBuffer != UINT32_MAX) {
            // Light buffer (skip names handled by bindShadowResources)
            if ((b.name.find("Light") != std::string::npos || b.name.find("light") != std::string::npos) &&
                b.name != "lights" && b.name != "SceneLight") {
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

        // Bind textures (skip shadow_maps, handled by bindShadowResources)
        if (b.mslTexture != UINT32_MAX && b.name != "shadow_maps") {
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

    // Render shadow maps before main pass (each layer gets its own render encoder)
    if (m_hasShadows) {
        renderShadowMaps(ctx, cmdBuf);
    }

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

            // Bind shadow/multi-light resources once per permutation
            bindShadowResources(enc, perm.fragShader);

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
                                if (b.name.find("Light") != std::string::npos &&
                                    b.name != "lights" && b.name != "SceneLight") {
                                    enc->setFragmentBuffer(m_lightBuffer, 0, b.mslBuffer);
                                } else if (b.name.find("Material") != std::string::npos ||
                                           b.name.find("Properties") != std::string::npos) {
                                    enc->setFragmentBuffer(matUBO, 0, b.mslBuffer);
                                }
                            }
                            if (b.mslTexture != UINT32_MAX && b.name != "shadow_maps") {
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
        // Bind shadow/multi-light resources once for the pipeline
        bindShadowResources(enc, m_fragmentShader);

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
                        if (b.name.find("Light") != std::string::npos &&
                            b.name != "lights" && b.name != "SceneLight") {
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
                    if (b.mslTexture != UINT32_MAX && b.name != "shadow_maps") {
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

    // Render shadow maps before main pass
    if (m_hasShadows) {
        renderShadowMaps(ctx, cmdBuf);
    }

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

            // Bind shadow/multi-light resources once per permutation
            bindShadowResources(enc, perm.fragShader);

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
                                if (b.name.find("Light") != std::string::npos &&
                                    b.name != "lights" && b.name != "SceneLight") {
                                    enc->setFragmentBuffer(m_lightBuffer, 0, b.mslBuffer);
                                } else if (b.name.find("Material") != std::string::npos ||
                                           b.name.find("Properties") != std::string::npos) {
                                    enc->setFragmentBuffer(matUBO, 0, b.mslBuffer);
                                }
                            }
                            if (b.mslTexture != UINT32_MAX && b.name != "shadow_maps") {
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

        // Bind shadow/multi-light resources once for the pipeline
        bindShadowResources(enc, m_fragmentShader);

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
                        if (b.name.find("Light") != std::string::npos &&
                            b.name != "lights" && b.name != "SceneLight") {
                            if (m_lightBuffer) enc->setFragmentBuffer(m_lightBuffer, 0, b.mslBuffer);
                        } else if (b.name.find("Material") != std::string::npos ||
                                   b.name.find("Properties") != std::string::npos) {
                            if (dr.materialIndex >= 0 && dr.materialIndex < static_cast<int>(m_perMaterialUBOs.size()))
                                enc->setFragmentBuffer(m_perMaterialUBOs[dr.materialIndex], 0, b.mslBuffer);
                            else if (m_materialBuffer)
                                enc->setFragmentBuffer(m_materialBuffer, 0, b.mslBuffer);
                        }
                    }
                    if (b.mslTexture != UINT32_MAX && b.name != "shadow_maps") {
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

void MetalRasterRenderer::renderToDrawableNoPresent(MetalContext& ctx, CA::MetalDrawable* drawable) {
    // Same as renderToDrawable but without presentDrawable (for editor overlay).
    // The caller will present after adding the ImGui overlay pass.
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

    // Use renderToTarget which creates command buffer, encodes, commits (no present)
    renderToTarget(ctx, rpDesc);
    rpDesc->release();
}

// --------------------------------------------------------------------------
// Cleanup
// --------------------------------------------------------------------------

void MetalRasterRenderer::cleanup() {
    // Shadow map cleanup
    cleanupShadowMaps();

    // Multi-light cleanup
    if (m_lightsSSBO) { m_lightsSSBO->release(); m_lightsSSBO = nullptr; }
    if (m_sceneLightUBO) { m_sceneLightUBO->release(); m_sceneLightUBO = nullptr; }

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
