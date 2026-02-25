#import <Cocoa/Cocoa.h>
#import <QuartzCore/CAMetalLayer.h>

#define GLFW_INCLUDE_NONE
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

CA::MetalLayer* createMetalLayer(GLFWwindow* window, MTL::Device* device) {
    NSWindow* nsWindow = glfwGetCocoaWindow(window);
    if (!nsWindow) return nullptr;

    // Create CAMetalLayer
    CAMetalLayer* layer = [CAMetalLayer layer];
    layer.device = (__bridge id<MTLDevice>)device;
    layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
    layer.displaySyncEnabled = YES;

    // Set drawable size from framebuffer
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    layer.drawableSize = CGSizeMake(width, height);

    // Attach to window's content view
    nsWindow.contentView.layer = layer;
    nsWindow.contentView.wantsLayer = YES;

    return (__bridge CA::MetalLayer*)layer;
}

void updateDrawableSize(CA::MetalLayer* layer, GLFWwindow* window) {
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    CAMetalLayer* caLayer = (__bridge CAMetalLayer*)layer;
    caLayer.drawableSize = CGSizeMake(width, height);
}
