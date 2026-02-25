#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

struct GLFWwindow;

class MetalContext {
public:
    MTL::Device* device = nullptr;
    MTL::CommandQueue* commandQueue = nullptr;
    CA::MetalLayer* metalLayer = nullptr;

    // Capabilities
    bool meshShaderSupported = false;
    bool argumentBuffersTier2 = false;

    void init(GLFWwindow* window);
    void initHeadless();
    void cleanup();

    // Frame helpers
    MTL::CommandBuffer* beginCommandBuffer();

    // Resource creation helpers
    MTL::Buffer* newBuffer(const void* data, size_t size, MTL::ResourceOptions opts = MTL::ResourceStorageModeShared);
    MTL::Buffer* newBuffer(size_t size, MTL::ResourceOptions opts = MTL::ResourceStorageModeShared);
    MTL::Texture* newTexture(MTL::TextureDescriptor* desc);

    // Single-shot command buffer for uploads
    void submitAndWait(MTL::CommandBuffer* cmdBuf);

private:
    void detectCapabilities();
};
