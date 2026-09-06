// playground_ios only: link-time stub for the macOS/GLFW-only half of
// playground_cpp/src/metal_bridge.h's API.
//
// metal_context.cpp (reused as-is, see playground_ios/README) defines
// MetalContext::init(GLFWwindow*), which calls createMetalLayer()/
// updateDrawableSize() (declared in metal_bridge.h, implemented for macOS
// in metal_bridge.mm via Cocoa/GLFW -- neither of which exist on iOS). This
// app never calls MetalContext::init(GLFWwindow*) -- it uses initHeadless()
// and then points ctx.metalLayer directly at a CAMetalLayer-backed UIView
// (see SplatView.mm) -- but metal_context.cpp still needs these two symbols
// to link. GLFWwindow is only ever used here as an opaque pointer type (per
// metal_bridge.h's forward declaration), so no GLFW headers are needed.
#include "metal_bridge.h"

CA::MetalLayer* createMetalLayer(GLFWwindow* /*window*/, MTL::Device* /*device*/) {
    return nullptr;
}

void updateDrawableSize(CA::MetalLayer* /*layer*/, GLFWwindow* /*window*/) {
    // no-op: SplatView manages its own CAMetalLayer's drawableSize.
}
