#pragma once

#include "metal_renderer_interface.h"
#include "metal_context.h"
#include "metal_scene_manager.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <string>
#include <cstdint>

struct GaussianSplatData;

class MetalSplatRenderer : public IMetalRenderer {
public:
    MetalSplatRenderer() = default;
    ~MetalSplatRenderer();

    void init(MetalContext& ctx, const GaussianSplatData& data,
              uint32_t width, uint32_t height);

    void render(MetalContext& ctx) override;
    void renderToDrawable(MetalContext& ctx, CA::MetalDrawable* drawable) override;
    void updateCamera(glm::vec3 eye, glm::vec3 target, glm::vec3 up,
                      float fovY, float aspect,
                      float nearPlane, float farPlane) override;

    MTL::Texture* getOutputTexture() const override { return colorTarget_; }
    uint32_t getWidth() const override { return width_; }
    uint32_t getHeight() const override { return height_; }

    void cleanup() override;

private:
    uint32_t width_ = 0, height_ = 0;
    uint32_t numSplats_ = 0;
    uint32_t shDegree_ = 0;

    // Offscreen render targets
    MTL::Texture* colorTarget_ = nullptr;
    MTL::Texture* depthTarget_ = nullptr;

    // Compute pipeline (projection)
    MTL::ComputePipelineState* computePipeline_ = nullptr;

    // Render pipeline (alpha-blended quad drawing)
    MTL::RenderPipelineState* renderPipeline_ = nullptr;
    MTL::DepthStencilState* depthStencilState_ = nullptr;

    // Splat input buffers
    MTL::Buffer* posBuffer_ = nullptr;
    MTL::Buffer* rotBuffer_ = nullptr;
    MTL::Buffer* scaleBuffer_ = nullptr;
    MTL::Buffer* opacityBuffer_ = nullptr;
    MTL::Buffer* shBuffer_ = nullptr;

    // Projected output buffers (written by compute, read by render)
    MTL::Buffer* projCenterBuffer_ = nullptr;
    MTL::Buffer* projConicBuffer_ = nullptr;
    MTL::Buffer* projColorBuffer_ = nullptr;

    // Sort index buffer (CPU-sorted, uploaded each frame)
    MTL::Buffer* sortedIndicesBuffer_ = nullptr;

    // Camera state
    glm::mat4 viewMatrix_{1.0f};
    glm::mat4 projMatrix_{1.0f};
    glm::vec3 camPos_{0.0f, 0.0f, 3.0f};
    float focalX_ = 256.0f;
    float focalY_ = 256.0f;

    // Cached positions for CPU sort
    std::vector<float> hostPositions_;

    // Helpers
    void createRenderTargets(MetalContext& ctx);
    void createPipelines(MetalContext& ctx);
    void createBuffers(MetalContext& ctx, const GaussianSplatData& data);
    void cpuSort();

    // Internal render to a specific render pass descriptor
    void renderToTarget(MetalContext& ctx, MTL::Texture* colorTex,
                        MTL::Texture* depthTex, uint32_t drawW, uint32_t drawH);
};
