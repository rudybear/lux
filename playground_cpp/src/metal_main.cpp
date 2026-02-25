#include "metal_context.h"
#include "metal_bridge.h"
#include "metal_raster_renderer.h"
#include "metal_mesh_renderer.h"
#include "metal_scene_manager.h"
#include "metal_renderer_interface.h"
#include "metal_screenshot.h"
#include "reflected_pipeline.h"
#include "camera.h"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <iostream>
#include <string>
#include <memory>
#include <filesystem>
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace fs = std::filesystem;

// --------------------------------------------------------------------------
// Orbit camera for interactive mode
// --------------------------------------------------------------------------

struct OrbitCamera {
    float yaw = 0.0f;
    float pitch = 0.15f;
    float distance = 3.0f;
    glm::vec3 target{0.0f};
    glm::vec3 up{0.0f, 1.0f, 0.0f};
    float fovY = glm::radians(45.0f);
    float nearPlane = 0.1f;
    float farPlane = 100.0f;

    bool dragging = false;
    double lastX = 0, lastY = 0;

    glm::vec3 getEye() const {
        float x = distance * cosf(pitch) * sinf(yaw);
        float y = distance * sinf(pitch);
        float z = distance * cosf(pitch) * cosf(yaw);
        return target + glm::vec3(x, y, z);
    }

    void initFromAutoCamera(const glm::vec3& eye, const glm::vec3& tgt,
                            const glm::vec3& u, float far_) {
        target = tgt;
        up = u;
        farPlane = far_;
        glm::vec3 dir = eye - tgt;
        distance = glm::length(dir);
        if (distance > 0.001f) {
            dir /= distance;
            pitch = asinf(glm::clamp(dir.y, -1.0f, 1.0f));
            yaw = atan2f(dir.x, dir.z);
        }
    }
};

static OrbitCamera g_orbit;

static void mouseButtonCallback(GLFWwindow* window, int button, int action, int /*mods*/) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        g_orbit.dragging = (action == GLFW_PRESS);
        if (g_orbit.dragging) {
            glfwGetCursorPos(window, &g_orbit.lastX, &g_orbit.lastY);
        }
    }
}

static void cursorPosCallback(GLFWwindow* /*window*/, double xpos, double ypos) {
    if (!g_orbit.dragging) return;
    float dx = static_cast<float>(xpos - g_orbit.lastX);
    float dy = static_cast<float>(ypos - g_orbit.lastY);
    g_orbit.lastX = xpos;
    g_orbit.lastY = ypos;
    g_orbit.yaw += dx * 0.005f;
    g_orbit.pitch += dy * 0.005f;
    g_orbit.pitch = glm::clamp(g_orbit.pitch, -1.5f, 1.5f);
}

static void scrollCallback(GLFWwindow* /*window*/, double /*xoff*/, double yoff) {
    g_orbit.distance *= (1.0f - 0.1f * static_cast<float>(yoff));
    g_orbit.distance = glm::clamp(g_orbit.distance, 0.01f, 1000.0f);
}

// --------------------------------------------------------------------------
// CLI argument parsing
// --------------------------------------------------------------------------

struct CLIOptions {
    std::string shaderBase;
    std::string sceneSource;
    std::string iblName;
    std::string forceMode;
    uint32_t width = 512;
    uint32_t height = 512;
    std::string output = "output.png";
    bool interactive = false;
    bool headless = true;
};

static void printUsage(const char* program) {
    std::cout << "Usage: " << program << " [OPTIONS]\n"
              << "\nOptions:\n"
              << "  --scene <SOURCE>       Scene: sphere, fullscreen, triangle, or .glb/.gltf path\n"
              << "  --pipeline <BASE>      Compiled shader base path\n"
              << "  --ibl <NAME>           IBL environment name\n"
              << "  --mode <MODE>          Rendering mode: mesh\n"
              << "  --width <N>            Output width (default: 512)\n"
              << "  --height <N>           Output height (default: 512)\n"
              << "  --output <PATH>        Output PNG path (default: output.png)\n"
              << "  --interactive          Open GLFW preview window\n"
              << "  --headless             Offscreen render only (default)\n"
              << "  --help                 Show this help message\n"
              << std::endl;
}

static CLIOptions parseArgs(int argc, char* argv[]) {
    CLIOptions opts;

    if (argc < 2) {
        printUsage(argv[0]);
        std::exit(1);
    }

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            std::exit(0);
        } else if (arg == "--scene" && i + 1 < argc) {
            opts.sceneSource = argv[++i];
        } else if (arg == "--pipeline" && i + 1 < argc) {
            opts.shaderBase = argv[++i];
        } else if (arg == "--ibl" && i + 1 < argc) {
            opts.iblName = argv[++i];
        } else if (arg == "--mode" && i + 1 < argc) {
            opts.forceMode = argv[++i];
        } else if (arg == "--width" && i + 1 < argc) {
            opts.width = static_cast<uint32_t>(std::stoi(argv[++i]));
        } else if (arg == "--height" && i + 1 < argc) {
            opts.height = static_cast<uint32_t>(std::stoi(argv[++i]));
        } else if (arg == "--output" && i + 1 < argc) {
            opts.output = argv[++i];
        } else if (arg == "--interactive") {
            opts.interactive = true;
            opts.headless = false;
        } else if (arg == "--headless") {
            opts.headless = true;
            opts.interactive = false;
        } else if (arg[0] != '-') {
            opts.shaderBase = arg;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            printUsage(argv[0]);
            std::exit(1);
        }
    }

    if (opts.sceneSource.empty()) {
        if (!opts.shaderBase.empty()) {
            opts.sceneSource = "sphere";
        } else {
            std::cerr << "Error: --scene is required\n";
            printUsage(argv[0]);
            std::exit(1);
        }
    }

    return opts;
}

// --------------------------------------------------------------------------
// Pipeline resolution
// --------------------------------------------------------------------------

static std::string resolveDefaultPipeline(const std::string& scene) {
    namespace fs = std::filesystem;

    // glTF scenes: prefer layered permutation pipeline, fall back to basic gltf_pbr
    if (MetalSceneManager::isGltfFile(scene)) {
        if (fs::exists("shadercache/gltf_pbr_layered.manifest.json"))
            return "shadercache/gltf_pbr_layered";
        if (fs::exists("shadercache/gltf_pbr.frag.spv"))
            return "shadercache/gltf_pbr";
        return "examples/gltf_pbr";
    }
    if (scene == "fullscreen")
        throw std::runtime_error("--pipeline required for fullscreen scenes");
    if (scene == "triangle") {
        if (fs::exists("shadercache/hello_triangle.vert.spv"))
            return "shadercache/hello_triangle";
        return "examples/hello_triangle";
    }
    // PBR sphere
    if (fs::exists("shadercache/pbr_basic.vert.spv"))
        return "shadercache/pbr_basic";
    return "examples/pbr_basic";
}

static std::string detectRenderPath(const std::string& base, const std::string& forceMode = "",
                                     const std::string& scene = "") {
    // Scene type overrides shader-based detection for special scenes
    if (scene == "triangle") return "triangle";
    if (scene == "fullscreen") return "fullscreen";

    if (forceMode == "mesh" && fs::exists(base + ".mesh.spv") && fs::exists(base + ".frag.spv")) return "mesh";
    if (fs::exists(base + ".vert.spv") && fs::exists(base + ".frag.spv")) return "raster";
    if (fs::exists(base + ".mesh.spv") && fs::exists(base + ".frag.spv")) return "mesh";
    if (fs::exists(base + ".frag.spv")) return "fullscreen";
    throw std::runtime_error("No shader files found for: " + base);
}

// --------------------------------------------------------------------------
// Headless rendering
// --------------------------------------------------------------------------

static int runHeadless(const CLIOptions& opts) {
    std::string renderPath = detectRenderPath(opts.shaderBase, opts.forceMode, opts.sceneSource);
    bool needMesh = (renderPath == "mesh");

    std::cout << "[metal] Headless render: scene=\"" << opts.sceneSource
              << "\" pipeline=\"" << opts.shaderBase
              << "\" path=" << renderPath
              << " (" << opts.width << "x" << opts.height << ")" << std::endl;

    // Metal-cpp requires an autorelease pool for temporary Objective-C objects
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

    MetalContext ctx;
    try {
        ctx.initHeadless();
    } catch (const std::exception& e) {
        std::cerr << "[error] Failed to initialize Metal: " << e.what() << std::endl;
        pool->release();
        return 1;
    }

    if (needMesh && !ctx.meshShaderSupported) {
        std::cerr << "[error] Mesh shader pipeline requested but Metal 3 mesh shaders not supported." << std::endl;
        ctx.cleanup();
        pool->release();
        return 1;
    }

    try {
        MetalSceneManager scene;
        scene.loadScene(opts.sceneSource);

        int vertexStride = scene.hasGltfScene() ? 48 : 32;
        scene.uploadToGPU(ctx, vertexStride);
        scene.uploadTextures(ctx);
        scene.loadIBLAssets(ctx, opts.iblName);

        // Resolve permutation
        std::string resolvedBase = opts.shaderBase;
        if (scene.hasGltfScene()) {
            ShaderManifest manifest = tryLoadManifest(opts.shaderBase);
            bool hasMultiMat = scene.getGltfScene().materials.size() > 1;
            if (!manifest.permutations.empty() && !hasMultiMat) {
                auto features = scene.detectSceneFeatures();
                std::string suffix = findPermutationSuffix(manifest, features);
                if (!suffix.empty()) {
                    std::string ext = needMesh ? ".mesh.spv" : ".vert.spv";
                    if (fs::exists(opts.shaderBase + suffix + ext)) {
                        resolvedBase = opts.shaderBase + suffix;
                        std::cout << "[metal] Resolved permutation: " << suffix << std::endl;
                    }
                }
            }
        }

        std::unique_ptr<IMetalRenderer> renderer;
        if (needMesh) {
            auto meshR = std::make_unique<MetalMeshRenderer>();
            meshR->init(ctx, scene, resolvedBase, opts.width, opts.height);
            renderer = std::move(meshR);
        } else {
            auto raster = std::make_unique<MetalRasterRenderer>();
            raster->init(ctx, scene, opts.shaderBase, renderPath, opts.width, opts.height);
            renderer = std::move(raster);
        }

        renderer->render(ctx);

        MetalScreenshot::saveTextureToPNG(ctx, renderer->getOutputTexture(),
                                           renderer->getWidth(), renderer->getHeight(),
                                           opts.output);

        renderer->cleanup();
        scene.cleanup();
    } catch (const std::exception& e) {
        std::cerr << "[error] Rendering failed: " << e.what() << std::endl;
        ctx.cleanup();
        pool->release();
        return 1;
    }

    ctx.cleanup();
    pool->release();
    std::cout << "[metal] Done." << std::endl;
    return 0;
}

// --------------------------------------------------------------------------
// Interactive rendering
// --------------------------------------------------------------------------

static int runInteractive(CLIOptions opts) {
    if (opts.width == 512 && opts.height == 512) {
        opts.width = 1024;
        opts.height = 768;
    }

    std::string renderPath = detectRenderPath(opts.shaderBase, opts.forceMode, opts.sceneSource);
    bool needMesh = (renderPath == "mesh");

    std::cout << "[metal] Interactive render: scene=\"" << opts.sceneSource
              << "\" pipeline=\"" << opts.shaderBase
              << "\" path=" << renderPath
              << " (" << opts.width << "x" << opts.height << ")" << std::endl;

    if (!glfwInit()) {
        std::cerr << "[error] Failed to initialize GLFW" << std::endl;
        return 1;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    GLFWwindow* window = glfwCreateWindow(
        static_cast<int>(opts.width), static_cast<int>(opts.height),
        "Lux Playground (Metal)", nullptr, nullptr);
    if (!window) {
        std::cerr << "[error] Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return 1;
    }

    MetalContext ctx;
    try {
        ctx.init(window);
    } catch (const std::exception& e) {
        std::cerr << "[error] Failed to initialize Metal: " << e.what() << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    if (needMesh && !ctx.meshShaderSupported) {
        std::cerr << "[error] Mesh shader pipeline requested but Metal 3 mesh shaders not supported." << std::endl;
        ctx.cleanup();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    MetalSceneManager scene;
    std::unique_ptr<IMetalRenderer> renderer;

    try {
        scene.loadScene(opts.sceneSource);
        int vertexStride = scene.hasGltfScene() ? 48 : 32;
        scene.uploadToGPU(ctx, vertexStride);
        scene.uploadTextures(ctx);
        scene.loadIBLAssets(ctx, opts.iblName);

        std::string resolvedBase = opts.shaderBase;
        if (scene.hasGltfScene()) {
            ShaderManifest manifest = tryLoadManifest(opts.shaderBase);
            bool hasMultiMat = scene.getGltfScene().materials.size() > 1;
            if (!manifest.permutations.empty() && !hasMultiMat) {
                auto features = scene.detectSceneFeatures();
                std::string suffix = findPermutationSuffix(manifest, features);
                if (!suffix.empty()) {
                    std::string ext = needMesh ? ".mesh.spv" : ".vert.spv";
                    if (fs::exists(opts.shaderBase + suffix + ext)) {
                        resolvedBase = opts.shaderBase + suffix;
                    }
                }
            }
        }

        if (needMesh) {
            auto meshR = std::make_unique<MetalMeshRenderer>();
            meshR->init(ctx, scene, resolvedBase, opts.width, opts.height);
            renderer = std::move(meshR);
        } else {
            auto raster = std::make_unique<MetalRasterRenderer>();
            raster->init(ctx, scene, opts.shaderBase, renderPath, opts.width, opts.height);
            renderer = std::move(raster);
        }
    } catch (const std::exception& e) {
        std::cerr << "[error] Failed to initialize renderer: " << e.what() << std::endl;
        ctx.cleanup();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    // Initialize orbit camera
    if (scene.hasSceneBounds()) {
        g_orbit.initFromAutoCamera(scene.getAutoEye(), scene.getAutoTarget(),
                                   scene.getAutoUp(), scene.getAutoFar());
    }

    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);
    glfwSetScrollCallback(window, scrollCallback);

    std::cout << "[metal] Starting render loop. Press ESC or close the window to quit." << std::endl;
    std::cout << "[metal] Mouse: drag to orbit, scroll to zoom." << std::endl;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
            continue;
        }

        int fbWidth, fbHeight;
        glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
        if (fbWidth == 0 || fbHeight == 0) {
            glfwWaitEvents();
            continue;
        }

        // Update drawable size
        updateDrawableSize(ctx.metalLayer, window);

        // Update camera
        float aspect = static_cast<float>(fbWidth) / static_cast<float>(fbHeight);
        renderer->updateCamera(g_orbit.getEye(), g_orbit.target, g_orbit.up,
                              g_orbit.fovY, aspect, g_orbit.nearPlane, g_orbit.farPlane);

        // Get next drawable and render
        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
        CA::MetalDrawable* drawable = ctx.metalLayer->nextDrawable();
        if (drawable) {
            renderer->renderToDrawable(ctx, drawable);
        }
        pool->release();
    }

    // Cleanup
    renderer->cleanup();
    scene.cleanup();
    ctx.cleanup();

    glfwDestroyWindow(window);
    glfwTerminate();

    std::cout << "[metal] Done." << std::endl;
    return 0;
}

// --------------------------------------------------------------------------
// Entry point
// --------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    CLIOptions opts = parseArgs(argc, argv);

    if (opts.shaderBase.empty()) {
        try {
            opts.shaderBase = resolveDefaultPipeline(opts.sceneSource);
            std::cout << "[metal] Resolved pipeline: " << opts.shaderBase << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[error] " << e.what() << std::endl;
            return 1;
        }
    }

    try {
        if (opts.interactive) {
            return runInteractive(opts);
        } else {
            return runHeadless(opts);
        }
    } catch (const std::exception& e) {
        std::cerr << "[fatal] Unhandled exception: " << e.what() << std::endl;
        return 1;
    }
}
