// NS_PRIVATE_IMPLEMENTATION, CA_PRIVATE_IMPLEMENTATION, MTL_PRIVATE_IMPLEMENTATION
// are defined via CMake's set_source_files_properties() for this translation unit.

#include "metal_context.h"
#include "metal_bridge.h"
#include <iostream>
#include <stdexcept>

void MetalContext::init(GLFWwindow* window) {
    device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        throw std::runtime_error("Failed to create Metal device");
    }

    std::cout << "[metal] Device: " << device->name()->utf8String() << std::endl;

    commandQueue = device->newCommandQueue();
    if (!commandQueue) {
        throw std::runtime_error("Failed to create Metal command queue");
    }

    metalLayer = createMetalLayer(window, device);
    if (!metalLayer) {
        throw std::runtime_error("Failed to create CAMetalLayer");
    }

    detectCapabilities();
}

void MetalContext::initHeadless() {
    device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        throw std::runtime_error("Failed to create Metal device");
    }

    std::cout << "[metal] Device: " << device->name()->utf8String() << std::endl;

    commandQueue = device->newCommandQueue();
    if (!commandQueue) {
        throw std::runtime_error("Failed to create Metal command queue");
    }

    metalLayer = nullptr; // No layer in headless mode

    detectCapabilities();
}

void MetalContext::detectCapabilities() {
    // Metal 3 mesh shaders require Apple GPU family Apple7+
    meshShaderSupported = device->supportsFamily(MTL::GPUFamilyApple7);

    // Argument buffers Tier 2
    argumentBuffersTier2 = (device->argumentBuffersSupport() == MTL::ArgumentBuffersTier2);

    std::cout << "[metal] Mesh shaders: " << (meshShaderSupported ? "supported" : "not supported") << std::endl;
    std::cout << "[metal] Argument buffers Tier 2: " << (argumentBuffersTier2 ? "yes" : "no") << std::endl;
}

void MetalContext::cleanup() {
    if (metalLayer) {
        metalLayer->release();
        metalLayer = nullptr;
    }
    if (commandQueue) {
        commandQueue->release();
        commandQueue = nullptr;
    }
    if (device) {
        device->release();
        device = nullptr;
    }
}

MTL::CommandBuffer* MetalContext::beginCommandBuffer() {
    return commandQueue->commandBuffer();
}

MTL::Buffer* MetalContext::newBuffer(const void* data, size_t size, MTL::ResourceOptions opts) {
    return device->newBuffer(data, size, opts);
}

MTL::Buffer* MetalContext::newBuffer(size_t size, MTL::ResourceOptions opts) {
    return device->newBuffer(size, opts);
}

MTL::Texture* MetalContext::newTexture(MTL::TextureDescriptor* desc) {
    return device->newTexture(desc);
}

void MetalContext::submitAndWait(MTL::CommandBuffer* cmdBuf) {
    cmdBuf->commit();
    cmdBuf->waitUntilCompleted();
}
