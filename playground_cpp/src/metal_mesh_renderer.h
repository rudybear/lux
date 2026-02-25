#pragma once

#include "metal_renderer_interface.h"
#include "metal_context.h"
#include "metal_scene_manager.h"
#include "metal_shader_transpiler.h"
#include "reflected_pipeline.h"
#include "material_ubo.h"
#include "meshlet.h"
#include <string>
#include <vector>
#include <unordered_map>

// A group of meshlets sharing a common material
struct MetalMeshletGroup {
    uint32_t firstMeshlet;
    uint32_t meshletCount;
    int materialIndex;
    int permutationIndex;
};

// Per-permutation mesh pipeline data
struct MetalMeshPermutation {
    std::string suffix;
    TranspiledShader fragShader;
    MTL::RenderPipelineState* pipelineState = nullptr;
    std::vector<MTL::Buffer*> perMaterialUBOs;
    std::vector<int> materialIndices;
};

class MetalMeshRenderer : public IMetalRenderer {
public:
    void init(MetalContext& ctx, MetalSceneManager& scene,
              const std::string& pipelineBase,
              uint32_t width, uint32_t height);

    void render(MetalContext& ctx) override;
    void renderToDrawable(MetalContext& ctx, CA::MetalDrawable* drawable) override;
    void updateCamera(glm::vec3 eye, glm::vec3 target,
                      glm::vec3 up, float fovY, float aspect,
                      float nearPlane, float farPlane) override;

    MTL::Texture* getOutputTexture() const override { return m_colorTarget; }
    uint32_t getWidth() const override { return m_width; }
    uint32_t getHeight() const override { return m_height; }

    void cleanup() override;

private:
    uint32_t m_width = 0, m_height = 0;
    uint32_t m_totalMeshlets = 0;
    std::string m_pipelineBase;

    MetalSceneManager* m_scene = nullptr;
    ShaderTranspiler m_transpiler;

    // Render targets
    MTL::Texture* m_colorTarget = nullptr;
    MTL::Texture* m_depthTarget = nullptr;

    // Depth stencil
    MTL::DepthStencilState* m_depthStencilState = nullptr;

    // Single-pipeline mode
    MTL::RenderPipelineState* m_pipelineState = nullptr;
    TranspiledShader m_objectShader;   // task/amplification → [[object]]
    TranspiledShader m_meshShader;     // mesh → [[mesh]]
    TranspiledShader m_fragShader;     // fragment

    // Meshlet data buffers
    MTL::Buffer* m_meshletDescBuffer = nullptr;
    MTL::Buffer* m_meshletVertBuffer = nullptr;
    MTL::Buffer* m_meshletTriBuffer = nullptr;

    // SoA vertex data
    MTL::Buffer* m_positionsBuffer = nullptr;
    MTL::Buffer* m_normalsBuffer = nullptr;
    MTL::Buffer* m_texCoordsBuffer = nullptr;
    MTL::Buffer* m_tangentsBuffer = nullptr;

    // Uniform buffers
    MTL::Buffer* m_mvpBuffer = nullptr;
    MTL::Buffer* m_lightBuffer = nullptr;
    MTL::Buffer* m_materialBuffer = nullptr;

    // Push constant data
    std::vector<uint8_t> m_pushConstantData;
    uint32_t m_pushConstantSize = 0;

    // Multi-pipeline mode
    bool m_multiPipeline = false;
    std::vector<MetalMeshPermutation> m_meshPermutations;
    std::vector<int> m_materialToPermutation;
    std::vector<MetalMeshletGroup> m_meshletGroups;

    void createRenderTargets(MetalContext& ctx);
    void createDepthStencilState(MetalContext& ctx);
    void uploadMeshletData(MetalContext& ctx);
    void uploadVertexData(MetalContext& ctx);
    void createPipeline(MetalContext& ctx);
    void setupMultiPipeline(MetalContext& ctx, const ShaderManifest& manifest);

    void renderToTarget(MetalContext& ctx, MTL::RenderPassDescriptor* rpDesc);
};
