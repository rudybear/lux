#pragma once

#include "metal_renderer_interface.h"
#include "metal_context.h"
#include "metal_scene_manager.h"
#include "metal_shader_transpiler.h"
#include "reflected_pipeline.h"
#include "material_ubo.h"
#include <string>
#include <vector>
#include <unordered_map>

// Per-permutation pipeline data for Metal multi-material rendering
struct MetalPermutation {
    std::string suffix;
    TranspiledShader vertShader;
    TranspiledShader fragShader;
    MTL::RenderPipelineState* pipelineState = nullptr;
    std::vector<MTL::Buffer*> perMaterialUBOs;
    std::vector<int> materialIndices;
};

class MetalRasterRenderer : public IMetalRenderer {
public:
    void init(MetalContext& ctx, MetalSceneManager& scene,
              const std::string& pipelineBase,
              const std::string& renderPath,
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
    std::string m_renderPath;
    std::string m_pipelineBase;
    uint32_t m_width = 512;
    uint32_t m_height = 512;

    MetalSceneManager* m_scene = nullptr;
    ShaderTranspiler m_transpiler;

    // Render targets (offscreen)
    MTL::Texture* m_colorTarget = nullptr;
    MTL::Texture* m_depthTarget = nullptr;
    bool m_needsDepth = false;

    // Single-pipeline mode
    MTL::RenderPipelineState* m_pipelineState = nullptr;
    MTL::DepthStencilState* m_depthStencilState = nullptr;
    TranspiledShader m_vertexShader;
    TranspiledShader m_fragmentShader;
    ReflectionData m_vertReflection;
    ReflectionData m_fragReflection;

    // Uniform buffers
    MTL::Buffer* m_mvpBuffer = nullptr;
    MTL::Buffer* m_lightBuffer = nullptr;
    MTL::Buffer* m_materialBuffer = nullptr;

    // Per-material UBOs (single-pipeline mode)
    std::vector<MTL::Buffer*> m_perMaterialUBOs;

    // Push constant data
    std::vector<uint8_t> m_pushConstantData;
    uint32_t m_pushConstantSize = 0;

    // Multi-pipeline mode (permutations)
    bool m_multiPipeline = false;
    std::vector<MetalPermutation> m_permutations;
    std::vector<int> m_materialToPermutation;

    // Triangle vertex buffer (for simple triangle mode)
    MTL::Buffer* m_triangleVB = nullptr;

    // Fullscreen quad shader (built-in MSL)
    MTL::Library* m_fullscreenLib = nullptr;
    MTL::Function* m_fullscreenVertFunc = nullptr;

    void createRenderTargets(MetalContext& ctx);
    void createDepthStencilState(MetalContext& ctx);

    // Pipeline creation per render path
    void createPipelineTriangle(MetalContext& ctx);
    void createPipelineFullscreen(MetalContext& ctx);
    void createPipelinePBR(MetalContext& ctx);

    // Multi-pipeline setup
    void setupMultiPipeline(MetalContext& ctx, const ShaderManifest& manifest);

    // Resource binding + draw
    void bindResourcesAndDraw(MTL::RenderCommandEncoder* enc,
                              const TranspiledShader& vertShader,
                              const TranspiledShader& fragShader,
                              int materialIndex = -1,
                              MTL::Buffer* materialUBO = nullptr);

    // Create render pipeline from transpiled shaders
    MTL::RenderPipelineState* createRenderPipelineState(
        MetalContext& ctx,
        MTL::Function* vertFunc, MTL::Function* fragFunc,
        const ReflectionData* vertRefl, int overrideStride = 0);

    // Fill material UBO from scene data
    void fillMaterialUBO(MTL::Buffer* buffer, int materialIndex);

    // Update MVP buffer
    void updateMVPBuffer();

    // Internal render to a specific render pass descriptor
    void renderToTarget(MetalContext& ctx, MTL::RenderPassDescriptor* rpDesc);
};
