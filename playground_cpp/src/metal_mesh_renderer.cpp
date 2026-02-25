#include "metal_mesh_renderer.h"
#include "camera.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <cstring>
#include <filesystem>

namespace fs = std::filesystem;

// --------------------------------------------------------------------------
// Render targets
// --------------------------------------------------------------------------

void MetalMeshRenderer::createRenderTargets(MetalContext& ctx) {
    auto* colorDesc = MTL::TextureDescriptor::texture2DDescriptor(
        MTL::PixelFormatRGBA8Unorm, m_width, m_height, false);
    colorDesc->setUsage(MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead);
    colorDesc->setStorageMode(MTL::StorageModePrivate);
    m_colorTarget = ctx.newTexture(colorDesc);

    auto* depthDesc = MTL::TextureDescriptor::texture2DDescriptor(
        MTL::PixelFormatDepth32Float, m_width, m_height, false);
    depthDesc->setUsage(MTL::TextureUsageRenderTarget);
    depthDesc->setStorageMode(MTL::StorageModePrivate);
    m_depthTarget = ctx.newTexture(depthDesc);
}

void MetalMeshRenderer::createDepthStencilState(MetalContext& ctx) {
    auto* dsDesc = MTL::DepthStencilDescriptor::alloc()->init();
    dsDesc->setDepthCompareFunction(MTL::CompareFunctionLess);
    dsDesc->setDepthWriteEnabled(true);
    m_depthStencilState = ctx.device->newDepthStencilState(dsDesc);
    dsDesc->release();
}

// --------------------------------------------------------------------------
// Upload meshlet data
// --------------------------------------------------------------------------

void MetalMeshRenderer::uploadMeshletData(MetalContext& ctx) {
    auto& vertices = m_scene->getVertices();
    auto& indices = m_scene->getIndices();

    // Build meshlets from scene geometry
    auto result = buildMeshlets(indices.data(),
                                static_cast<uint32_t>(indices.size()),
                                static_cast<uint32_t>(vertices.size()));

    m_totalMeshlets = static_cast<uint32_t>(result.meshlets.size());

    // Upload to Metal buffers
    m_meshletDescBuffer = ctx.newBuffer(result.meshlets.data(),
        result.meshlets.size() * sizeof(MeshletDescriptor), MTL::ResourceStorageModeShared);
    m_meshletVertBuffer = ctx.newBuffer(result.meshletVertices.data(),
        result.meshletVertices.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    m_meshletTriBuffer = ctx.newBuffer(result.meshletTriangles.data(),
        result.meshletTriangles.size() * sizeof(uint32_t), MTL::ResourceStorageModeShared);

    std::cout << "[metal] Meshlets: " << m_totalMeshlets << " meshlets, "
              << result.meshletVertices.size() << " vertices, "
              << result.meshletTriangles.size() << " triangles" << std::endl;
}

// --------------------------------------------------------------------------
// Upload SoA vertex data
// --------------------------------------------------------------------------

void MetalMeshRenderer::uploadVertexData(MetalContext& ctx) {
    auto& vertices = m_scene->getVertices();

    // Extract SoA from interleaved vertices
    std::vector<glm::vec3> positions(vertices.size());
    std::vector<glm::vec3> normals(vertices.size());
    std::vector<glm::vec2> texCoords(vertices.size());

    for (size_t i = 0; i < vertices.size(); i++) {
        positions[i] = vertices[i].position;
        normals[i] = vertices[i].normal;
        texCoords[i] = vertices[i].uv;
    }

    m_positionsBuffer = ctx.newBuffer(positions.data(),
        positions.size() * sizeof(glm::vec3), MTL::ResourceStorageModeShared);
    m_normalsBuffer = ctx.newBuffer(normals.data(),
        normals.size() * sizeof(glm::vec3), MTL::ResourceStorageModeShared);
    m_texCoordsBuffer = ctx.newBuffer(texCoords.data(),
        texCoords.size() * sizeof(glm::vec2), MTL::ResourceStorageModeShared);

    // Extract tangents from glTF if available
    if (m_scene->hasGltfScene()) {
        auto& gltfScene = m_scene->getGltfScene();
        std::vector<glm::vec4> tangents(vertices.size(), glm::vec4(1, 0, 0, 1));

        auto drawItems = flattenScene(const_cast<GltfScene&>(gltfScene));
        size_t vertOff = 0;
        for (auto& item : drawItems) {
            auto& gmesh = gltfScene.meshes[item.meshIndex];
            for (size_t i = 0; i < gmesh.vertices.size(); i++) {
                if (gmesh.hasTangents) {
                    tangents[vertOff + i] = gmesh.vertices[i].tangent;
                }
            }
            vertOff += gmesh.vertices.size();
        }

        m_tangentsBuffer = ctx.newBuffer(tangents.data(),
            tangents.size() * sizeof(glm::vec4), MTL::ResourceStorageModeShared);
    } else {
        std::vector<glm::vec4> tangents(vertices.size(), glm::vec4(1, 0, 0, 1));
        m_tangentsBuffer = ctx.newBuffer(tangents.data(),
            tangents.size() * sizeof(glm::vec4), MTL::ResourceStorageModeShared);
    }
}

// --------------------------------------------------------------------------
// Create mesh pipeline
// --------------------------------------------------------------------------

void MetalMeshRenderer::createPipeline(MetalContext& ctx) {
    // Transpile shaders
    std::string meshSpv = m_pipelineBase + ".mesh.spv";
    std::string fragSpv = m_pipelineBase + ".frag.spv";

    m_transpiler.transpileInto(m_meshShader, meshSpv, SpvExecModel::MeshEXT);
    m_transpiler.transpileInto(m_fragShader, fragSpv, SpvExecModel::Fragment);

    // Check for task/object shader
    std::string taskSpv = m_pipelineBase + ".task.spv";
    if (fs::exists(taskSpv)) {
        m_transpiler.transpileInto(m_objectShader, taskSpv, SpvExecModel::TaskEXT);
    }

    // Create mesh render pipeline descriptor
    auto* meshDesc = MTL::MeshRenderPipelineDescriptor::alloc()->init();

    if (m_objectShader.function) {
        meshDesc->setObjectFunction(m_objectShader.function);
    }
    meshDesc->setMeshFunction(m_meshShader.function);
    meshDesc->setFragmentFunction(m_fragShader.function);

    meshDesc->colorAttachments()->object(0)->setPixelFormat(MTL::PixelFormatRGBA8Unorm);
    meshDesc->setDepthAttachmentPixelFormat(MTL::PixelFormatDepth32Float);

    // Set thread group sizes
    meshDesc->setMaxTotalThreadsPerMeshThreadgroup(32);
    if (m_objectShader.function) {
        meshDesc->setMaxTotalThreadsPerObjectThreadgroup(32);
    }

    NS::Error* error = nullptr;
    MTL::RenderPipelineReflection* reflection = nullptr;
    m_pipelineState = ctx.device->newRenderPipelineState(meshDesc, 0, &reflection, &error);
    meshDesc->release();

    if (!m_pipelineState) {
        std::string errorMsg = "Failed to create mesh render pipeline";
        if (error) {
            errorMsg += ": ";
            errorMsg += error->localizedDescription()->utf8String();
        }
        throw std::runtime_error(errorMsg);
    }

    // Load reflection data for uniform binding
    std::string meshJson = m_pipelineBase + ".mesh.lux.json";
    std::string fragJson = m_pipelineBase + ".frag.lux.json";
    ReflectionData meshRefl, fragRefl;
    if (fs::exists(meshJson)) meshRefl = parseReflectionJson(meshJson);
    if (fs::exists(fragJson)) fragRefl = parseReflectionJson(fragJson);

    // Create uniform buffers
    m_mvpBuffer = ctx.newBuffer(192, MTL::ResourceStorageModeShared);
    m_lightBuffer = ctx.newBuffer(64, MTL::ResourceStorageModeShared);

    // Light UBO layout: vec3 light_dir (padded) + vec3 view_pos (padded)
    struct LightData {
        glm::vec3 lightDir; float _pad0;
        glm::vec3 viewPos; float _pad1;
    } lightData;
    lightData.lightDir = glm::normalize(glm::vec3(1.0f, 0.8f, 0.6f));
    lightData._pad0 = 0;
    lightData.viewPos = m_scene->hasSceneBounds() ? m_scene->getAutoEye() : Camera::DEFAULT_EYE;
    lightData._pad1 = 0;
    memcpy(m_lightBuffer->contents(), &lightData, sizeof(lightData));

    // Initialize MVP
    float aspect = static_cast<float>(m_width) / static_cast<float>(m_height);
    auto cam = Camera::getDefaultMatrices(aspect);
    if (m_scene->hasSceneBounds()) {
        cam.view = Camera::lookAt(m_scene->getAutoEye(), m_scene->getAutoTarget(), m_scene->getAutoUp());
        cam.projection = Camera::perspective(glm::radians(45.0f), aspect, 0.1f, m_scene->getAutoFar());
    }
    struct { glm::mat4 model, view, proj; } mvpData = {cam.model, cam.view, cam.projection};
    memcpy(m_mvpBuffer->contents(), &mvpData, sizeof(mvpData));

    // Create material buffer
    if (m_scene->hasGltfScene() && !m_scene->getGltfScene().materials.empty()) {
        auto& mat = m_scene->getGltfScene().materials[0];
        MaterialUBOData ubo;
        ubo.baseColorFactor = mat.baseColor;
        ubo.metallicFactor = mat.metallic;
        ubo.roughnessFactor = mat.roughness;
        m_materialBuffer = ctx.newBuffer(sizeof(MaterialUBOData), MTL::ResourceStorageModeShared);
        memcpy(m_materialBuffer->contents(), &ubo, sizeof(ubo));
    } else {
        MaterialUBOData ubo;
        m_materialBuffer = ctx.newBuffer(sizeof(MaterialUBOData), MTL::ResourceStorageModeShared);
        memcpy(m_materialBuffer->contents(), &ubo, sizeof(ubo));
    }

    // Push constants
    for (auto& pc : meshRefl.push_constants) {
        m_pushConstantSize = std::max(m_pushConstantSize, static_cast<uint32_t>(pc.size));
    }
    for (auto& pc : fragRefl.push_constants) {
        m_pushConstantSize = std::max(m_pushConstantSize, static_cast<uint32_t>(pc.size));
    }
    if (m_pushConstantSize > 0) {
        m_pushConstantData.resize(m_pushConstantSize, 0);
    }
}

// --------------------------------------------------------------------------
// Multi-pipeline setup
// --------------------------------------------------------------------------

void MetalMeshRenderer::setupMultiPipeline(MetalContext& ctx, const ShaderManifest& manifest) {
    m_multiPipeline = true;
    // Similar to raster renderer multi-pipeline, but shared mesh shader
    // Implementation deferred — single pipeline suffices for Phase 11a
    std::cout << "[metal] Mesh multi-pipeline: " << manifest.permutations.size()
              << " permutations available (using single pipeline for now)" << std::endl;
}

// --------------------------------------------------------------------------
// Init
// --------------------------------------------------------------------------

void MetalMeshRenderer::init(MetalContext& ctx, MetalSceneManager& scene,
                              const std::string& pipelineBase,
                              uint32_t width, uint32_t height) {
    m_scene = &scene;
    m_pipelineBase = pipelineBase;
    m_width = width;
    m_height = height;

    if (!ctx.meshShaderSupported) {
        throw std::runtime_error("Metal mesh shaders require Apple GPU family Apple7+ (Metal 3)");
    }

    m_transpiler.init(ctx.device);
    createRenderTargets(ctx);
    createDepthStencilState(ctx);
    uploadMeshletData(ctx);
    uploadVertexData(ctx);
    createPipeline(ctx);

    std::cout << "[metal] Mesh renderer initialized (" << width << "x" << height << ")" << std::endl;
}

// --------------------------------------------------------------------------
// Render
// --------------------------------------------------------------------------

void MetalMeshRenderer::renderToTarget(MetalContext& ctx, MTL::RenderPassDescriptor* rpDesc) {
    auto* cmdBuf = ctx.beginCommandBuffer();
    auto* enc = cmdBuf->renderCommandEncoder(rpDesc);

    enc->setViewport({0.0, 0.0, static_cast<double>(m_width), static_cast<double>(m_height), 0.0, 1.0});
    enc->setScissorRect({0, 0, m_width, m_height});
    enc->setRenderPipelineState(m_pipelineState);
    enc->setDepthStencilState(m_depthStencilState);
    enc->setFrontFacingWinding(MTL::WindingClockwise);
    enc->setCullMode(MTL::CullModeBack);

    // Bind object (task) shader resources if present
    if (m_objectShader.function) {
        for (auto& b : m_objectShader.bindings) {
            if (b.mslBuffer != UINT32_MAX) {
                if (b.name.find("meshlet") != std::string::npos || b.name.find("Meshlet") != std::string::npos) {
                    if (b.name.find("Desc") != std::string::npos || b.name.find("desc") != std::string::npos)
                        enc->setObjectBuffer(m_meshletDescBuffer, 0, b.mslBuffer);
                    else if (b.name.find("Vert") != std::string::npos || b.name.find("vert") != std::string::npos)
                        enc->setObjectBuffer(m_meshletVertBuffer, 0, b.mslBuffer);
                    else if (b.name.find("Tri") != std::string::npos || b.name.find("tri") != std::string::npos)
                        enc->setObjectBuffer(m_meshletTriBuffer, 0, b.mslBuffer);
                }
                else if (b.name.find("MVP") != std::string::npos || b.name.find("Matrices") != std::string::npos)
                    enc->setObjectBuffer(m_mvpBuffer, 0, b.mslBuffer);
            }
        }
        if (m_pushConstantSize > 0 && m_objectShader.pushConstantBufferIndex != UINT32_MAX) {
            enc->setObjectBytes(m_pushConstantData.data(), m_pushConstantData.size(),
                               m_objectShader.pushConstantBufferIndex);
        }
    }

    // Bind mesh shader resources
    // Meshlet data
    for (auto& b : m_meshShader.bindings) {
        if (b.mslBuffer != UINT32_MAX) {
            if (b.name.find("meshlet") != std::string::npos || b.name.find("Meshlet") != std::string::npos) {
                if (b.name.find("Desc") != std::string::npos || b.name.find("desc") != std::string::npos) {
                    enc->setMeshBuffer(m_meshletDescBuffer, 0, b.mslBuffer);
                } else if (b.name.find("Vert") != std::string::npos || b.name.find("vert") != std::string::npos) {
                    enc->setMeshBuffer(m_meshletVertBuffer, 0, b.mslBuffer);
                } else if (b.name.find("Tri") != std::string::npos || b.name.find("tri") != std::string::npos) {
                    enc->setMeshBuffer(m_meshletTriBuffer, 0, b.mslBuffer);
                }
            }
            // SoA vertex buffers
            else if (b.name.find("position") != std::string::npos || b.name.find("Position") != std::string::npos) {
                enc->setMeshBuffer(m_positionsBuffer, 0, b.mslBuffer);
            }
            else if (b.name.find("normal") != std::string::npos || b.name.find("Normal") != std::string::npos) {
                enc->setMeshBuffer(m_normalsBuffer, 0, b.mslBuffer);
            }
            else if (b.name.find("texcoord") != std::string::npos || b.name.find("TexCoord") != std::string::npos ||
                     b.name.find("uv") != std::string::npos || b.name.find("UV") != std::string::npos) {
                enc->setMeshBuffer(m_texCoordsBuffer, 0, b.mslBuffer);
            }
            else if (b.name.find("tangent") != std::string::npos || b.name.find("Tangent") != std::string::npos) {
                if (m_tangentsBuffer) enc->setMeshBuffer(m_tangentsBuffer, 0, b.mslBuffer);
            }
            // MVP in mesh stage
            else if (b.name.find("MVP") != std::string::npos || b.name.find("Matrices") != std::string::npos) {
                enc->setMeshBuffer(m_mvpBuffer, 0, b.mslBuffer);
            }
        }
    }

    // Bind fragment resources
    for (auto& b : m_fragShader.bindings) {
        if (b.mslBuffer != UINT32_MAX) {
            if (b.name.find("Light") != std::string::npos || b.name.find("light") != std::string::npos) {
                enc->setFragmentBuffer(m_lightBuffer, 0, b.mslBuffer);
            }
            else if (b.name.find("Material") != std::string::npos || b.name.find("Properties") != std::string::npos) {
                if (m_materialBuffer) enc->setFragmentBuffer(m_materialBuffer, 0, b.mslBuffer);
            }
        }
        if (b.mslTexture != UINT32_MAX) {
            auto& tex = m_scene->getTextureForBinding(b.name);
            if (tex.texture) enc->setFragmentTexture(tex.texture, b.mslTexture);
            if (b.mslSampler != UINT32_MAX && tex.sampler) {
                enc->setFragmentSamplerState(tex.sampler, b.mslSampler);
            }
        }
    }

    // Push constants
    if (m_pushConstantSize > 0) {
        if (m_meshShader.pushConstantBufferIndex != UINT32_MAX) {
            enc->setMeshBytes(m_pushConstantData.data(), m_pushConstantData.size(),
                             m_meshShader.pushConstantBufferIndex);
        }
        if (m_fragShader.pushConstantBufferIndex != UINT32_MAX) {
            enc->setFragmentBytes(m_pushConstantData.data(), m_pushConstantData.size(),
                                 m_fragShader.pushConstantBufferIndex);
        }
    }

    // Dispatch mesh threadgroups
    uint32_t numGroups = (m_totalMeshlets + 31) / 32;
    enc->drawMeshThreadgroups(MTL::Size(numGroups, 1, 1),
                              MTL::Size(32, 1, 1),
                              MTL::Size(32, 1, 1));

    enc->endEncoding();
    ctx.submitAndWait(cmdBuf);
}

void MetalMeshRenderer::render(MetalContext& ctx) {
    auto* rpDesc = MTL::RenderPassDescriptor::alloc()->init();

    auto* colorAtt = rpDesc->colorAttachments()->object(0);
    colorAtt->setTexture(m_colorTarget);
    colorAtt->setLoadAction(MTL::LoadActionClear);
    colorAtt->setStoreAction(MTL::StoreActionStore);
    colorAtt->setClearColor(MTL::ClearColor(0.1, 0.1, 0.15, 1.0));

    auto* depthAtt = rpDesc->depthAttachment();
    depthAtt->setTexture(m_depthTarget);
    depthAtt->setLoadAction(MTL::LoadActionClear);
    depthAtt->setStoreAction(MTL::StoreActionDontCare);
    depthAtt->setClearDepth(1.0);

    renderToTarget(ctx, rpDesc);
    rpDesc->release();
}

void MetalMeshRenderer::renderToDrawable(MetalContext& ctx, CA::MetalDrawable* drawable) {
    // Use drawable texture dimensions for viewport (handles resize)
    auto* drawableTex = drawable->texture();
    uint32_t drawW = static_cast<uint32_t>(drawableTex->width());
    uint32_t drawH = static_cast<uint32_t>(drawableTex->height());

    // Rebuild depth buffer if drawable size changed
    if (m_depthTarget &&
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

    auto* depthAtt = rpDesc->depthAttachment();
    depthAtt->setTexture(m_depthTarget);
    depthAtt->setLoadAction(MTL::LoadActionClear);
    depthAtt->setStoreAction(MTL::StoreActionDontCare);
    depthAtt->setClearDepth(1.0);

    auto* cmdBuf = ctx.beginCommandBuffer();
    auto* enc = cmdBuf->renderCommandEncoder(rpDesc);

    enc->setViewport({0.0, 0.0, static_cast<double>(drawW), static_cast<double>(drawH), 0.0, 1.0});
    enc->setScissorRect({0, 0, drawW, drawH});
    enc->setRenderPipelineState(m_pipelineState);
    enc->setDepthStencilState(m_depthStencilState);
    enc->setFrontFacingWinding(MTL::WindingClockwise);
    enc->setCullMode(MTL::CullModeBack);

    // Bind object (task) shader resources if present
    if (m_objectShader.function) {
        for (auto& b : m_objectShader.bindings) {
            if (b.mslBuffer != UINT32_MAX) {
                if (b.name.find("meshlet") != std::string::npos || b.name.find("Meshlet") != std::string::npos) {
                    if (b.name.find("Desc") != std::string::npos) enc->setObjectBuffer(m_meshletDescBuffer, 0, b.mslBuffer);
                    else if (b.name.find("Vert") != std::string::npos) enc->setObjectBuffer(m_meshletVertBuffer, 0, b.mslBuffer);
                    else if (b.name.find("Tri") != std::string::npos) enc->setObjectBuffer(m_meshletTriBuffer, 0, b.mslBuffer);
                }
                else if (b.name.find("MVP") != std::string::npos || b.name.find("Matrices") != std::string::npos)
                    enc->setObjectBuffer(m_mvpBuffer, 0, b.mslBuffer);
            }
        }
        if (m_pushConstantSize > 0 && m_objectShader.pushConstantBufferIndex != UINT32_MAX)
            enc->setObjectBytes(m_pushConstantData.data(), m_pushConstantData.size(),
                               m_objectShader.pushConstantBufferIndex);
    }

    // Bind mesh shader resources
    for (auto& b : m_meshShader.bindings) {
        if (b.mslBuffer != UINT32_MAX) {
            if (b.name.find("meshlet") != std::string::npos || b.name.find("Meshlet") != std::string::npos) {
                if (b.name.find("Desc") != std::string::npos) enc->setMeshBuffer(m_meshletDescBuffer, 0, b.mslBuffer);
                else if (b.name.find("Vert") != std::string::npos) enc->setMeshBuffer(m_meshletVertBuffer, 0, b.mslBuffer);
                else if (b.name.find("Tri") != std::string::npos) enc->setMeshBuffer(m_meshletTriBuffer, 0, b.mslBuffer);
            }
            else if (b.name.find("position") != std::string::npos) enc->setMeshBuffer(m_positionsBuffer, 0, b.mslBuffer);
            else if (b.name.find("normal") != std::string::npos) enc->setMeshBuffer(m_normalsBuffer, 0, b.mslBuffer);
            else if (b.name.find("texcoord") != std::string::npos || b.name.find("uv") != std::string::npos) enc->setMeshBuffer(m_texCoordsBuffer, 0, b.mslBuffer);
            else if (b.name.find("tangent") != std::string::npos && m_tangentsBuffer) enc->setMeshBuffer(m_tangentsBuffer, 0, b.mslBuffer);
            else if (b.name.find("MVP") != std::string::npos || b.name.find("Matrices") != std::string::npos) enc->setMeshBuffer(m_mvpBuffer, 0, b.mslBuffer);
        }
    }
    for (auto& b : m_fragShader.bindings) {
        if (b.mslBuffer != UINT32_MAX) {
            if (b.name.find("Light") != std::string::npos) enc->setFragmentBuffer(m_lightBuffer, 0, b.mslBuffer);
            else if (b.name.find("Material") != std::string::npos) {
                if (m_materialBuffer) enc->setFragmentBuffer(m_materialBuffer, 0, b.mslBuffer);
            }
        }
        if (b.mslTexture != UINT32_MAX) {
            auto& tex = m_scene->getTextureForBinding(b.name);
            if (tex.texture) enc->setFragmentTexture(tex.texture, b.mslTexture);
            if (b.mslSampler != UINT32_MAX && tex.sampler) enc->setFragmentSamplerState(tex.sampler, b.mslSampler);
        }
    }

    // Push constants
    if (m_pushConstantSize > 0) {
        if (m_meshShader.pushConstantBufferIndex != UINT32_MAX)
            enc->setMeshBytes(m_pushConstantData.data(), m_pushConstantData.size(),
                             m_meshShader.pushConstantBufferIndex);
        if (m_fragShader.pushConstantBufferIndex != UINT32_MAX)
            enc->setFragmentBytes(m_pushConstantData.data(), m_pushConstantData.size(),
                                 m_fragShader.pushConstantBufferIndex);
    }

    uint32_t numGroups = (m_totalMeshlets + 31) / 32;
    enc->drawMeshThreadgroups(MTL::Size(numGroups, 1, 1), MTL::Size(32, 1, 1), MTL::Size(32, 1, 1));

    enc->endEncoding();
    cmdBuf->presentDrawable(drawable);
    cmdBuf->commit();
    rpDesc->release();
}

void MetalMeshRenderer::updateCamera(glm::vec3 eye, glm::vec3 target,
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
}

// --------------------------------------------------------------------------
// Cleanup
// --------------------------------------------------------------------------

void MetalMeshRenderer::cleanup() {
    m_objectShader.cleanup();
    m_meshShader.cleanup();
    m_fragShader.cleanup();

    if (m_pipelineState) { m_pipelineState->release(); m_pipelineState = nullptr; }
    if (m_depthStencilState) { m_depthStencilState->release(); m_depthStencilState = nullptr; }

    if (m_colorTarget) { m_colorTarget->release(); m_colorTarget = nullptr; }
    if (m_depthTarget) { m_depthTarget->release(); m_depthTarget = nullptr; }

    if (m_meshletDescBuffer) { m_meshletDescBuffer->release(); m_meshletDescBuffer = nullptr; }
    if (m_meshletVertBuffer) { m_meshletVertBuffer->release(); m_meshletVertBuffer = nullptr; }
    if (m_meshletTriBuffer) { m_meshletTriBuffer->release(); m_meshletTriBuffer = nullptr; }

    if (m_positionsBuffer) { m_positionsBuffer->release(); m_positionsBuffer = nullptr; }
    if (m_normalsBuffer) { m_normalsBuffer->release(); m_normalsBuffer = nullptr; }
    if (m_texCoordsBuffer) { m_texCoordsBuffer->release(); m_texCoordsBuffer = nullptr; }
    if (m_tangentsBuffer) { m_tangentsBuffer->release(); m_tangentsBuffer = nullptr; }

    if (m_mvpBuffer) { m_mvpBuffer->release(); m_mvpBuffer = nullptr; }
    if (m_lightBuffer) { m_lightBuffer->release(); m_lightBuffer = nullptr; }
    if (m_materialBuffer) { m_materialBuffer->release(); m_materialBuffer = nullptr; }

    for (auto& perm : m_meshPermutations) {
        perm.fragShader.cleanup();
        if (perm.pipelineState) perm.pipelineState->release();
        for (auto* buf : perm.perMaterialUBOs) { if (buf) buf->release(); }
    }
    m_meshPermutations.clear();

    m_transpiler.cleanup();
}
