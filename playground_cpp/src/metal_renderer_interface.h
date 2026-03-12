#pragma once

#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <glm/glm.hpp>

class MetalContext;

class IMetalRenderer {
public:
    virtual void render(MetalContext& ctx) = 0;
    virtual void updateCamera(glm::vec3 eye, glm::vec3 target,
                              glm::vec3 up, float fovY, float aspect,
                              float nearPlane, float farPlane) = 0;

    // Render to a CAMetalDrawable (interactive mode)
    virtual void renderToDrawable(MetalContext& ctx, CA::MetalDrawable* drawable) = 0;

    // Render scene to a drawable's texture without presenting (for editor overlay).
    // Default implementation calls renderToDrawable (subclasses can override).
    virtual void renderToDrawableNoPresent(MetalContext& ctx, CA::MetalDrawable* drawable) {
        renderToDrawable(ctx, drawable);
    }

    // Offscreen output access (for screenshot)
    virtual MTL::Texture* getOutputTexture() const = 0;
    virtual uint32_t getWidth() const = 0;
    virtual uint32_t getHeight() const = 0;

    virtual void cleanup() = 0;
    virtual ~IMetalRenderer() = default;
};
