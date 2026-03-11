#include "vulkan_context.h"
#include "spv_loader.h"
#include "raster_renderer.h"
#include "rt_renderer.h"
#include "mesh_renderer.h"
#include "scene_manager.h"
#include "renderer_interface.h"
#include "reflected_pipeline.h"
#include "scene.h"
#include "camera.h"
#include "screenshot.h"
#include "editor_ui.h"

#include <GLFW/glfw3.h>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <filesystem>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <chrono>
#include <fstream>

namespace fs = std::filesystem;

// --------------------------------------------------------------------------
// Orbit camera for interactive mode
// --------------------------------------------------------------------------

struct OrbitCamera {
    float yaw = 0.0f;        // horizontal angle (radians)
    float pitch = 0.15f;     // vertical angle (radians), slight default elevation
    float distance = 3.0f;   // distance from target
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
static EditorUI* g_editorUI = nullptr; // non-null when editor mode is active

static void mouseButtonCallback(GLFWwindow* window, int button, int action, int /*mods*/) {
    // Let ImGui handle input when it wants to capture the mouse
    if (g_editorUI && g_editorUI->wantCaptureMouse()) return;

    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        g_orbit.dragging = (action == GLFW_PRESS);
        if (g_orbit.dragging) {
            glfwGetCursorPos(window, &g_orbit.lastX, &g_orbit.lastY);
        }
    }
}

static void cursorPosCallback(GLFWwindow* /*window*/, double xpos, double ypos) {
    if (g_editorUI && g_editorUI->wantCaptureMouse()) return;
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
    if (g_editorUI && g_editorUI->wantCaptureMouse()) return;
    g_orbit.distance *= (1.0f - 0.1f * static_cast<float>(yoff));
    g_orbit.distance = glm::clamp(g_orbit.distance, 0.01f, 1000.0f);
}

// --------------------------------------------------------------------------
// CLI argument parsing (scene/pipeline architecture)
// --------------------------------------------------------------------------

struct CLIOptions {
    std::string shaderBase;    // --pipeline value (resolved from scene if not given)
    std::string sceneSource;   // --scene value (required)
    std::string iblName;       // --ibl value (optional, auto-detects if empty)
    std::string forceMode;     // --mode rt forces RT render path
    uint32_t width = 512;
    uint32_t height = 512;
    std::string output = "output.png";
    bool interactive = false;
    bool headless = true;
    bool editor = false;       // --editor: enable ImGui scene editor overlay
    bool forceValidation = false;
    bool demoLights = false;
    bool sponzaLights = false;
    bool noIBL = false;
};

static void printUsage(const char* program) {
    std::cout << "Usage: " << program << " [OPTIONS] [shader_base]\n"
              << "\n"
              << "Arguments:\n"
              << "  [shader_base]          Base path for .spv shader files (positional, legacy)\n"
              << "\n"
              << "Options:\n"
              << "  --scene <SOURCE>       Scene source: sphere, fullscreen, triangle, or path to .glb/.gltf\n"
              << "  --pipeline <BASE>      Compiled shader base path (auto-resolved from scene if omitted)\n"
              << "  --ibl <NAME>           IBL environment name (default: auto-detect pisa/neutral)\n"
              << "  --mode <MODE>          (Legacy) Rendering mode: triangle|fullscreen|pbr|rt\n"
              << "  --width <N>            Output width in pixels (default: 512)\n"
              << "  --height <N>           Output height in pixels (default: 512)\n"
              << "  --output <PATH>        Output PNG path (default: output.png)\n"
              << "  --interactive          Open GLFW preview window\n"
              << "  --editor               Interactive mode with ImGui scene editor\n"
              << "  --headless             Offscreen render only (default)\n"
              << "  --validation           Enable Vulkan validation layers (even in release)\n"
              << "  --demo-lights          Add 3 demo lights (directional + point + spot) with shadows\n"
              << "  --sponza-lights        Sponza courtyard lights (sun + orbiting torch + accent)\n"
              << "  --no-ibl               Disable IBL environment loading\n"
              << "  --help                 Show this help message\n"
              << std::endl;
}

static CLIOptions parseArgs(int argc, char* argv[]) {
    CLIOptions opts;
    std::string legacyMode;

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
            // Legacy --mode support: map to --scene
            i++;
            legacyMode = argv[i];
            std::transform(legacyMode.begin(), legacyMode.end(), legacyMode.begin(), ::tolower);
        } else if (arg == "--width" && i + 1 < argc) {
            i++;
            opts.width = static_cast<uint32_t>(std::stoi(argv[i]));
        } else if (arg == "--height" && i + 1 < argc) {
            i++;
            opts.height = static_cast<uint32_t>(std::stoi(argv[i]));
        } else if (arg == "--output" && i + 1 < argc) {
            i++;
            opts.output = argv[i];
        } else if (arg == "--interactive") {
            opts.interactive = true;
            opts.headless = false;
        } else if (arg == "--editor") {
            opts.editor = true;
            opts.interactive = true;
            opts.headless = false;
        } else if (arg == "--headless") {
            opts.headless = true;
            opts.interactive = false;
        } else if (arg == "--validation") {
            opts.forceValidation = true;
        } else if (arg == "--demo-lights") {
            opts.demoLights = true;
        } else if (arg == "--sponza-lights") {
            opts.sponzaLights = true;
        } else if (arg == "--no-ibl") {
            opts.noIBL = true;
        } else if (arg[0] != '-') {
            opts.shaderBase = arg;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            printUsage(argv[0]);
            std::exit(1);
        }
    }

    // Legacy --mode backwards compatibility: map mode to scene source
    if (!legacyMode.empty()) {
        if (legacyMode == "rt" || legacyMode == "mesh") {
            opts.forceMode = legacyMode;
        }
        if (opts.sceneSource.empty()) {
            if (legacyMode == "triangle")        opts.sceneSource = "triangle";
            else if (legacyMode == "fullscreen") opts.sceneSource = "fullscreen";
            else if (legacyMode == "pbr")        opts.sceneSource = "sphere";
            else if (legacyMode == "rt")         opts.sceneSource = "sphere";
            else {
                std::cerr << "Unknown legacy mode: " << legacyMode << std::endl;
                std::exit(1);
            }
            std::cout << "[info] Legacy --mode \"" << legacyMode
                      << "\" mapped to --scene \"" << opts.sceneSource << "\"" << std::endl;
        }
    }

    // Backwards compat: if no --scene, treat positional arg as pipeline base and use "sphere" as scene
    if (opts.sceneSource.empty()) {
        if (!opts.shaderBase.empty()) {
            opts.sceneSource = "sphere";  // default scene
        } else {
            std::cerr << "Error: --scene is required\n";
            printUsage(argv[0]);
            std::exit(1);
        }
    }

    return opts;
}

// --------------------------------------------------------------------------
// Demo lights setup
// --------------------------------------------------------------------------

static void setupDemoLights(SceneManager& scene) {
    // Clear any existing lights
    scene.clearLights();

    // Light 1: warm directional (sun-like), casts shadow
    SceneLight sun;
    sun.type = SceneLight::Directional;
    sun.direction = glm::normalize(glm::vec3(0.6f, -0.8f, 0.4f));
    sun.color = glm::vec3(1.0f, 0.95f, 0.85f);
    sun.intensity = 1.2f;
    sun.castsShadow = true;
    scene.addLight(sun);

    // Light 2: blue point light (left side)
    SceneLight blue;
    blue.type = SceneLight::Point;
    blue.position = glm::vec3(-2.0f, 1.0f, 1.0f);
    blue.color = glm::vec3(0.3f, 0.5f, 1.0f);
    blue.intensity = 3.0f;
    blue.range = 10.0f;
    scene.addLight(blue);

    // Light 3: red spot light (right side), casts shadow
    SceneLight spot;
    spot.type = SceneLight::Spot;
    spot.position = glm::vec3(2.5f, 2.0f, 1.5f);
    spot.direction = glm::normalize(glm::vec3(-1.0f, -1.0f, -0.5f));
    spot.color = glm::vec3(1.0f, 0.3f, 0.2f);
    spot.intensity = 5.0f;
    spot.range = 15.0f;
    spot.innerConeAngle = 0.2f;
    spot.outerConeAngle = 0.5f;
    spot.castsShadow = true;
    scene.addLight(spot);

    std::cout << "[info] Demo lights: 3 lights (directional+shadow, point, spot+shadow)"
              << std::endl;
}

// --------------------------------------------------------------------------
// Sponza courtyard lights: sun + orbiting torch + blue accent
// --------------------------------------------------------------------------

static void setupSponzaLights(SceneManager& scene) {
    scene.clearLights();

    // Sun: warm directional light casting shadows through arches
    SceneLight sun;
    sun.type = SceneLight::Directional;
    sun.direction = glm::normalize(glm::vec3(0.5f, -0.7f, 0.3f));
    sun.color = glm::vec3(1.0f, 0.95f, 0.85f);
    sun.intensity = 8.0f;
    sun.castsShadow = true;
    scene.addLight(sun);

    // Torch: orange spot light that orbits inside the courtyard (animated per frame)
    SceneLight torch;
    torch.type = SceneLight::Spot;
    torch.position = glm::vec3(0.0f, 600.0f, 0.0f);
    torch.direction = glm::normalize(glm::vec3(0.0f, -200.0f, 0.0f) - torch.position);
    torch.color = glm::vec3(1.0f, 0.7f, 0.3f);
    torch.intensity = 500000.0f;
    torch.range = 3000.0f;
    torch.innerConeAngle = 0.3f;
    torch.outerConeAngle = 0.7f;
    torch.castsShadow = true;
    scene.addLight(torch);

    // Accent: blue point light (no shadow)
    SceneLight accent;
    accent.type = SceneLight::Point;
    accent.position = glm::vec3(-500.0f, 400.0f, -300.0f);
    accent.color = glm::vec3(0.3f, 0.5f, 1.0f);
    accent.intensity = 100000.0f;
    accent.range = 2000.0f;
    scene.addLight(accent);

    std::cout << "[info] Sponza lights: 3 lights (sun+shadow, orbiting torch+shadow, blue accent)"
              << std::endl;
}

// --------------------------------------------------------------------------
// Lighttest scene lights: sun + red point + green point
// --------------------------------------------------------------------------

static void setupTestLights(SceneManager& scene) {
    scene.clearLights();

    // Directional sun (warm white, casts shadow) — angled to cast cube shadow on plane
    // direction = toward the light (convention used by lighting.lux evaluate_directional_light)
    SceneLight sun;
    sun.type = SceneLight::Directional;
    sun.direction = glm::normalize(glm::vec3(1.0f, 2.0f, 0.5f));
    sun.color = glm::vec3(1.0f, 0.98f, 0.95f);
    sun.intensity = 2.0f;
    sun.castsShadow = true;
    scene.addLight(sun);

    // Red point light (left side, elevated)
    SceneLight red;
    red.type = SceneLight::Point;
    red.position = glm::vec3(-3.0f, 2.5f, -2.0f);
    red.color = glm::vec3(1.0f, 0.1f, 0.1f);
    red.intensity = 15.0f;
    red.range = 20.0f;
    scene.addLight(red);

    // Green point light (right side, elevated)
    SceneLight green;
    green.type = SceneLight::Point;
    green.position = glm::vec3(3.0f, 2.5f, 2.0f);
    green.color = glm::vec3(0.1f, 1.0f, 0.1f);
    green.intensity = 15.0f;
    green.range = 20.0f;
    scene.addLight(green);

    std::cout << "[info] Test lights: 3 lights (directional+shadow, red point, green point)"
              << std::endl;
}

// --------------------------------------------------------------------------
// Pipeline resolution and render path detection
// --------------------------------------------------------------------------

static std::string resolveDefaultPipeline(const std::string& scene) {
    if (scene.size() > 4 && (scene.substr(scene.size()-4) == ".glb" || scene.substr(scene.size()-5) == ".gltf"))
        return "examples/gltf_pbr";
    if (scene == "lighttest")
        return "shadercache/gltf_pbr_layered";
    if (scene == "fullscreen")
        throw std::runtime_error("--pipeline required for fullscreen scenes");
    if (scene == "triangle")
        return "examples/hello_triangle";
    return "examples/pbr_basic";
}

static bool hasGaussianSplattingKey(const std::string& base) {
    // Check for compute shader presence (.comp.spv) as splat render indicator
    if (fs::exists(base + ".comp.spv")) return true;
    // Check reflection JSON for gaussian_splatting key
    std::string jsonPath = base + ".reflect.json";
    if (fs::exists(jsonPath)) {
        std::ifstream f(jsonPath);
        if (f.is_open()) {
            std::string content((std::istreambuf_iterator<char>(f)),
                                 std::istreambuf_iterator<char>());
            if (content.find("gaussian_splatting") != std::string::npos) return true;
        }
    }
    return false;
}

static std::string detectRenderPath(const std::string& base, const std::string& forceMode = "") {
    if (forceMode == "rt" && fs::exists(base + ".rgen.spv")) return "rt";
    if (forceMode == "mesh" && fs::exists(base + ".mesh.spv") && fs::exists(base + ".frag.spv")) return "mesh";
    // Detect gaussian splatting compute pipeline
    if (hasGaussianSplattingKey(base)) return "splat";
    // Prefer raster over RT when both exist (raster supports per-material draw calls)
    if (fs::exists(base + ".vert.spv") && fs::exists(base + ".frag.spv")) return "raster";
    if (fs::exists(base + ".mesh.spv") && fs::exists(base + ".frag.spv")) return "mesh";
    if (fs::exists(base + ".rgen.spv")) return "rt";
    if (fs::exists(base + ".frag.spv")) return "fullscreen";
    throw std::runtime_error("No shader files found for: " + base);
}

// --------------------------------------------------------------------------
// Headless rendering
// --------------------------------------------------------------------------

static int runHeadless(const CLIOptions& opts) {
    std::string renderPath = detectRenderPath(opts.shaderBase, opts.forceMode);
    bool needRT = (renderPath == "rt");
    bool needMesh = (renderPath == "mesh");
    bool needSplat = (renderPath == "splat");
    VkPipelineStageFlags dstStage = needRT
        ? VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR
        : (needSplat ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
                     : VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

    std::cout << "[info] Headless render: scene=\"" << opts.sceneSource
              << "\" pipeline=\"" << opts.shaderBase
              << "\" path=" << renderPath
              << " (" << opts.width << "x" << opts.height << ")" << std::endl;

    // Initialize Vulkan context
    VulkanContext ctx;
    try {
        ctx.init(needRT, true, nullptr, opts.forceValidation);
    } catch (const std::exception& e) {
        std::cerr << "[error] Failed to initialize Vulkan: " << e.what() << std::endl;
        return 1;
    }

    if (needRT && !ctx.supportsRT()) {
        std::cerr << "[error] RT mode requested but ray tracing is not supported on this device." << std::endl;
        ctx.cleanup();
        return 1;
    }

    if (needMesh && !ctx.supportsMeshShader()) {
        std::cerr << "[error] Mesh shader pipeline requested but VK_EXT_mesh_shader is not supported." << std::endl;
        ctx.cleanup();
        return 1;
    }

    try {
        // Shared scene setup
        SceneManager scene;
        bool isLightTest = (opts.sceneSource == "lighttest");
        if (isLightTest) {
            scene.loadProceduralTestScene(ctx);
            setupTestLights(scene);
            scene.overrideAutoCamera(
                glm::vec3(6.0f, 4.0f, 6.0f),
                glm::vec3(0.0f, 1.0f, 0.0f),
                glm::vec3(0.0f, 1.0f, 0.0f),
                50.0f);
        } else {
            scene.loadScene(ctx, opts.sceneSource);
        }

        // Set up demo/sponza lights if requested (before GPU upload so light count is known)
        // Auto-detect Sponza scene by filename
        bool useSponzaLights = opts.sponzaLights;
        if (!isLightTest && !useSponzaLights && !opts.demoLights) {
            std::string sceneLower = opts.sceneSource;
            std::transform(sceneLower.begin(), sceneLower.end(), sceneLower.begin(), ::tolower);
            if (sceneLower.find("ponza") != std::string::npos) {
                useSponzaLights = true;
                std::cout << "[info] Auto-detected Sponza scene, enabling sponza lights" << std::endl;
            }
        }
        if (useSponzaLights) {
            setupSponzaLights(scene);
            // Interior courtyard camera for headless renders
            scene.overrideAutoCamera(
                glm::vec3(-1200.0f, 200.0f, 0.0f),
                glm::vec3(200.0f, 200.0f, 0.0f),
                glm::vec3(0.0f, 1.0f, 0.0f),
                5000.0f);
        } else if (opts.demoLights) {
            setupDemoLights(scene);
        }

        // Always use 48-byte stride for glTF scenes so that flattenScene()
        // is called (which generates draw ranges and applies world transforms).
        // RT/mesh renderers create their own SoA buffers but still need draw
        // ranges for per-material rendering. Raster shaders that don't use
        // tangent (stride=32) simply ignore the extra bytes.
        int vertexStride = scene.hasGltfScene() ? 48 : 32;

        scene.uploadToGPU(ctx, vertexStride);
        scene.uploadTextures(ctx);
        bool skipIBL = opts.noIBL || isLightTest;
        if (!skipIBL) {
            scene.loadIBLAssets(ctx, dstStage, opts.iblName);
        }

        // Resolve permutation for RT/mesh paths from manifest + scene features.
        // Renderers with multi-material support (raster, mesh, RT) handle
        // permutation selection internally when a manifest exists.
        // Only fall back to single-permutation resolution when the renderer
        // doesn't support multi-material (e.g., legacy code paths).
        std::string resolvedBase = opts.shaderBase;
        bool hasManifest = false;
        bool hasMultipleMaterials = scene.hasGltfScene() && scene.getGltfScene().materials.size() > 1;
        if (scene.hasGltfScene()) {
            ShaderManifest manifest = tryLoadManifest(opts.shaderBase);
            hasManifest = !manifest.permutations.empty();
            if (hasManifest && !hasMultipleMaterials) {
                // Single material: resolve to best permutation for the whole scene
                auto features = scene.detectSceneFeatures();
                std::string suffix = findPermutationSuffix(manifest, features);
                if (!suffix.empty()) {
                    std::string ext = needRT ? ".rgen.spv" : (needMesh ? ".mesh.spv" : ".vert.spv");
                    if (fs::exists(opts.shaderBase + suffix + ext)) {
                        resolvedBase = opts.shaderBase + suffix;
                        std::cout << "[info] Resolved permutation: " << suffix << std::endl;
                    }
                }
            }
            // Multi-material with manifest: pass unresolved base to renderer
            // (renderers handle per-material permutation selection internally)
        }

        // Initialize splat renderer if scene has gaussian splat data
        if (needSplat && scene.hasSplatData()) {
            scene.initSplatRenderer(ctx, resolvedBase, opts.width, opts.height);
        }

        std::unique_ptr<IRenderer> renderer;
        if (needSplat && scene.getSplatRenderer()) {
            // Gaussian splatting compute path: use SplatRenderer directly
            auto* splatR = scene.getSplatRenderer();
            splatR->render(ctx);

            Screenshot::saveImageToPNG(ctx, splatR->getOutputImage(), splatR->getOutputFormat(),
                                        splatR->getWidth(), splatR->getHeight(),
                                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, opts.output);
            scene.cleanup(ctx);
        } else {
            if (needRT) {
                auto rt = std::make_unique<RTRenderer>();
                rt->init(ctx, resolvedBase + ".rgen.spv", resolvedBase + ".rmiss.spv",
                         resolvedBase + ".rchit.spv", opts.width, opts.height, scene);
                renderer = std::move(rt);
            } else if (needMesh) {
                auto meshR = std::make_unique<MeshRenderer>();
                meshR->init(ctx, scene, resolvedBase, opts.width, opts.height);
                renderer = std::move(meshR);
            } else {
                auto raster = std::make_unique<RasterRenderer>();
                raster->init(ctx, scene, opts.shaderBase, renderPath, opts.width, opts.height);
                renderer = std::move(raster);
            }

            renderer->render(ctx);

            Screenshot::saveImageToPNG(ctx, renderer->getOutputImage(), renderer->getOutputFormat(),
                                        renderer->getWidth(), renderer->getHeight(),
                                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, opts.output);

            renderer->cleanup(ctx);
            scene.cleanup(ctx);
        }
    } catch (const std::exception& e) {
        std::cerr << "[error] Rendering failed: " << e.what() << std::endl;
        ctx.cleanup();
        return 1;
    }

    ctx.cleanup();
    std::cout << "[info] Done." << std::endl;
    return 0;
}

// --------------------------------------------------------------------------
// Interactive rendering
// --------------------------------------------------------------------------

static int runInteractive(CLIOptions opts) {
    // Use larger window for interactive mode if user didn't specify
    if (opts.width == 512 && opts.height == 512) {
        opts.width = opts.editor ? 1280 : 1024;
        opts.height = opts.editor ? 800 : 768;
    }
    std::string renderPath = detectRenderPath(opts.shaderBase, opts.forceMode);
    bool needRT = (renderPath == "rt");
    bool needMesh = (renderPath == "mesh");
    bool needSplat = (renderPath == "splat");
    VkPipelineStageFlags dstStage = needRT
        ? VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR
        : (needSplat ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
                     : VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

    std::cout << "[info] Interactive render: scene=\"" << opts.sceneSource
              << "\" pipeline=\"" << opts.shaderBase
              << "\" path=" << renderPath
              << (opts.editor ? " (editor)" : "")
              << " (" << opts.width << "x" << opts.height << ")" << std::endl;

    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "[error] Failed to initialize GLFW" << std::endl;
        return 1;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    const char* windowTitle = opts.editor ? "Lux Editor" : "Lux Playground";
    GLFWwindow* window = glfwCreateWindow(
        static_cast<int>(opts.width), static_cast<int>(opts.height),
        windowTitle, nullptr, nullptr);
    if (!window) {
        std::cerr << "[error] Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return 1;
    }

    // Initialize Vulkan context with surface
    VulkanContext ctx;
    try {
        ctx.init(needRT, false, window, opts.forceValidation);
    } catch (const std::exception& e) {
        std::cerr << "[error] Failed to initialize Vulkan: " << e.what() << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    if (needRT && !ctx.supportsRT()) {
        std::cerr << "[error] RT mode requested but ray tracing is not supported on this device." << std::endl;
        ctx.cleanup();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    if (needMesh && !ctx.supportsMeshShader()) {
        std::cerr << "[error] Mesh shader pipeline requested but VK_EXT_mesh_shader is not supported." << std::endl;
        ctx.cleanup();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    // Create swapchain
    try {
        ctx.createSwapchain(opts.width, opts.height);
    } catch (const std::exception& e) {
        std::cerr << "[error] Failed to create swapchain: " << e.what() << std::endl;
        ctx.cleanup();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    // Create synchronization objects
    VkSemaphore imageAvailableSem = VK_NULL_HANDLE;
    VkSemaphore renderFinishedSem = VK_NULL_HANDLE;
    VkSemaphore sceneRenderSem = VK_NULL_HANDLE;  // intermediate: scene done, before imgui
    VkFence inFlightFence = VK_NULL_HANDLE;

    VkSemaphoreCreateInfo semInfo = {};
    semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    vkCreateSemaphore(ctx.device, &semInfo, nullptr, &imageAvailableSem);
    vkCreateSemaphore(ctx.device, &semInfo, nullptr, &renderFinishedSem);
    vkCreateSemaphore(ctx.device, &semInfo, nullptr, &sceneRenderSem);

    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    vkCreateFence(ctx.device, &fenceInfo, nullptr, &inFlightFence);

    // Initialize editor UI (if --editor)
    EditorUI editorUI;
    bool editorActive = opts.editor;
    bool editorNeedsReinit = false;  // set after swapchain recreation
    if (editorActive) {
        try {
            editorUI.init(ctx, window,
                          static_cast<uint32_t>(ctx.swapchainImages.size()));
            editorUI.getState().currentPipelineBase = opts.shaderBase;
            // Scan for available pipelines
            editorUI.scanPipelines("shadercache");
            editorUI.scanPipelines("examples");
            g_editorUI = &editorUI;
        } catch (const std::exception& e) {
            std::cerr << "[error] Failed to initialize editor: " << e.what() << std::endl;
            editorActive = false;
        }
    }

    // Initialize shared scene
    SceneManager scene;
    std::unique_ptr<IRenderer> renderer;
    bool useRT = needRT;
    bool useMesh = needMesh;
    bool useSplat = needSplat;

    // Track whether this is a Sponza scene for torch animation
    bool useSponzaLights = opts.sponzaLights;

    bool isLightTest = (opts.sceneSource == "lighttest");

    try {
        if (isLightTest) {
            scene.loadProceduralTestScene(ctx);
            setupTestLights(scene);
            scene.overrideAutoCamera(
                glm::vec3(6.0f, 4.0f, 6.0f),
                glm::vec3(0.0f, 1.0f, 0.0f),
                glm::vec3(0.0f, 1.0f, 0.0f),
                50.0f);
        } else {
            scene.loadScene(ctx, opts.sceneSource);
        }

        // Set up demo/sponza lights if requested
        // Auto-detect Sponza scene by filename
        if (!isLightTest && !useSponzaLights && !opts.demoLights) {
            std::string sceneLower = opts.sceneSource;
            std::transform(sceneLower.begin(), sceneLower.end(), sceneLower.begin(), ::tolower);
            if (sceneLower.find("ponza") != std::string::npos) {
                useSponzaLights = true;
                std::cout << "[info] Auto-detected Sponza scene, enabling sponza lights" << std::endl;
            }
        }
        if (useSponzaLights) {
            setupSponzaLights(scene);
            scene.overrideAutoCamera(
                glm::vec3(-1000.0f, 100.0f, -200.0f),
                glm::vec3(500.0f, 50.0f, 0.0f),
                glm::vec3(0.0f, 1.0f, 0.0f),
                3000.0f);
        } else if (opts.demoLights) {
            setupDemoLights(scene);
        }

        // Always use 48-byte stride for glTF raster scenes so that flattenScene()
        // is called (which generates draw ranges and applies world transforms).
        int vertexStride = scene.hasGltfScene() ? 48 : 32;

        scene.uploadToGPU(ctx, vertexStride);
        scene.uploadTextures(ctx);
        bool skipIBL = opts.noIBL || isLightTest;
        if (!skipIBL) {
            scene.loadIBLAssets(ctx, dstStage, opts.iblName);
        }

        // Resolve permutation for RT/mesh paths.
        // Multi-material scenes: pass unresolved base so renderers handle
        // per-material permutation selection internally.
        // Single-material scenes: resolve to best permutation for the whole scene.
        std::string resolvedBase = opts.shaderBase;
        bool hasMultiMat = scene.hasGltfScene() && scene.getGltfScene().materials.size() > 1;
        if (scene.hasGltfScene()) {
            ShaderManifest manifest = tryLoadManifest(opts.shaderBase);
            if (!manifest.permutations.empty() && !hasMultiMat) {
                auto features = scene.detectSceneFeatures();
                std::string suffix = findPermutationSuffix(manifest, features);
                if (!suffix.empty()) {
                    std::string ext = useRT ? ".rgen.spv" : (useMesh ? ".mesh.spv" : ".vert.spv");
                    if (fs::exists(opts.shaderBase + suffix + ext)) {
                        resolvedBase = opts.shaderBase + suffix;
                        std::cout << "[info] Resolved permutation: " << suffix << std::endl;
                    }
                }
            }
        }

        // Initialize splat renderer if scene has gaussian splat data
        if (useSplat && scene.hasSplatData()) {
            scene.initSplatRenderer(ctx, resolvedBase, opts.width, opts.height);
        }

        if (useRT) {
            auto rt = std::make_unique<RTRenderer>();
            std::string rgenPath = resolvedBase + ".rgen.spv";
            std::string rmissPath = resolvedBase + ".rmiss.spv";
            std::string rchitPath = resolvedBase + ".rchit.spv";
            rt->init(ctx, rgenPath, rmissPath, rchitPath,
                     opts.width, opts.height, scene);
            renderer = std::move(rt);
        } else if (useMesh) {
            auto meshR = std::make_unique<MeshRenderer>();
            meshR->init(ctx, scene, resolvedBase,
                        opts.width, opts.height);
            renderer = std::move(meshR);
        } else if (!useSplat) {
            auto raster = std::make_unique<RasterRenderer>();
            raster->init(ctx, scene, opts.shaderBase, renderPath,
                         opts.width, opts.height);
            renderer = std::move(raster);
        }
    } catch (const std::exception& e) {
        std::cerr << "[error] Failed to initialize renderer: " << e.what() << std::endl;
        if (editorActive) { editorUI.cleanup(ctx); g_editorUI = nullptr; }
        vkDestroySemaphore(ctx.device, imageAvailableSem, nullptr);
        vkDestroySemaphore(ctx.device, renderFinishedSem, nullptr);
        vkDestroySemaphore(ctx.device, sceneRenderSem, nullptr);
        vkDestroyFence(ctx.device, inFlightFence, nullptr);
        ctx.cleanup();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    // Initialize orbit camera from scene auto-camera
    if (scene.hasSceneBounds()) {
        g_orbit.initFromAutoCamera(
            scene.getAutoEye(),
            scene.getAutoTarget(),
            scene.getAutoUp(),
            scene.getAutoFar());
    }

    // Register GLFW input callbacks
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);
    glfwSetScrollCallback(window, scrollCallback);

    std::cout << "[info] Starting render loop. Press ESC or close the window to quit." << std::endl;
    std::cout << "[info] Mouse: drag to orbit, scroll to zoom." << std::endl;

    VkCommandBuffer rtBlitCmd = VK_NULL_HANDLE; // Track RT blit cmd for deferred free
    VkCommandBuffer editorCmd = VK_NULL_HANDLE; // Track editor cmd for deferred free
    auto startTime = std::chrono::high_resolution_clock::now();

    // Render loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // Check for ESC key (only when editor is not capturing keyboard)
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            if (!editorActive || !editorUI.wantCaptureKeyboard()) {
                glfwSetWindowShouldClose(window, GLFW_TRUE);
                continue;
            }
        }

        // Check for minimized window
        int fbWidth, fbHeight;
        glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
        if (fbWidth == 0 || fbHeight == 0) {
            glfwWaitEvents();
            continue;
        }

        // Wait for previous frame to finish
        vkWaitForFences(ctx.device, 1, &inFlightFence, VK_TRUE, UINT64_MAX);

        // Free previous RT blit command buffer (now safe after fence)
        if (rtBlitCmd != VK_NULL_HANDLE) {
            vkFreeCommandBuffers(ctx.device, ctx.commandPool, 1, &rtBlitCmd);
            rtBlitCmd = VK_NULL_HANDLE;
        }
        // Free previous editor command buffer
        if (editorCmd != VK_NULL_HANDLE) {
            vkFreeCommandBuffers(ctx.device, ctx.commandPool, 1, &editorCmd);
            editorCmd = VK_NULL_HANDLE;
        }

        // Acquire swapchain image
        uint32_t imageIndex;
        VkResult acquireResult = vkAcquireNextImageKHR(
            ctx.device, ctx.swapchain, UINT64_MAX,
            imageAvailableSem, VK_NULL_HANDLE, &imageIndex);

        if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
            // Recreate swapchain
            vkDeviceWaitIdle(ctx.device);
            for (auto& iv : ctx.swapchainImageViews) {
                vkDestroyImageView(ctx.device, iv, nullptr);
            }
            ctx.swapchainImageViews.clear();
            vkDestroySwapchainKHR(ctx.device, ctx.swapchain, nullptr);
            ctx.swapchain = VK_NULL_HANDLE;
            ctx.createSwapchain(static_cast<uint32_t>(fbWidth), static_cast<uint32_t>(fbHeight));
            if (editorActive) editorNeedsReinit = true;
            continue;
        }
        if (acquireResult != VK_SUCCESS && acquireResult != VK_SUBOPTIMAL_KHR) {
            std::cerr << "[error] Failed to acquire swapchain image" << std::endl;
            break;
        }

        // Reinitialize editor framebuffers after swapchain recreation
        if (editorActive && editorNeedsReinit) {
            editorUI.cleanup(ctx);
            editorUI.init(ctx, window,
                          static_cast<uint32_t>(ctx.swapchainImages.size()));
            editorUI.getState().currentPipelineBase = opts.shaderBase;
            editorNeedsReinit = false;
        }

        vkResetFences(ctx.device, 1, &inFlightFence);

        // Begin editor frame (before any ImGui calls this frame)
        if (editorActive) {
            editorUI.beginFrame();
            editorUI.drawPanels(scene);
            editorUI.updateFPS();
        }

        // Animate Sponza torch light (light[1] orbits inside the courtyard)
        if (useSponzaLights && scene.getLightCount() >= 2) {
            auto now = std::chrono::high_resolution_clock::now();
            float elapsed = std::chrono::duration<float>(now - startTime).count();
            float angle = elapsed * 0.3f; // orbit speed
            glm::vec3 pos(800.0f * sinf(angle), 600.0f, 400.0f * cosf(angle));
            auto& lights = scene.getLightsMutable();
            lights[1].position = pos;
            lights[1].direction = glm::normalize(glm::vec3(0.0f, -200.0f, 0.0f) - pos);
        }

        // Update orbit camera matrices each frame
        {
            float aspect = static_cast<float>(ctx.swapchainExtent.width) /
                           static_cast<float>(ctx.swapchainExtent.height);
            if (renderer) {
                renderer->updateCamera(ctx, g_orbit.getEye(), g_orbit.target,
                                       g_orbit.up, g_orbit.fovY, aspect,
                                       g_orbit.nearPlane, g_orbit.farPlane);
            }
            if (useSplat && scene.getSplatRenderer()) {
                scene.getSplatRenderer()->updateCamera(g_orbit.getEye(), g_orbit.target,
                                                        g_orbit.up, g_orbit.fovY, aspect,
                                                        g_orbit.nearPlane, g_orbit.farPlane);
            }
        }

        // Determine which semaphore the scene render signals:
        // if editor is active, signal sceneRenderSem (intermediate); else signal renderFinishedSem
        VkSemaphore sceneSignalSem = editorActive ? sceneRenderSem : renderFinishedSem;
        // The fence is only signaled by the last submit (editor if active, scene otherwise)
        VkFence sceneSubmitFence = editorActive ? VK_NULL_HANDLE : inFlightFence;

        if (useSplat && scene.getSplatRenderer()) {
            // Gaussian splatting compute path: render to storage image and blit to swapchain
            scene.getSplatRenderer()->render(ctx);

            rtBlitCmd = ctx.beginSingleTimeCommands();
            VkCommandBuffer cmd = rtBlitCmd;

            scene.getSplatRenderer()->blitToSwapchain(ctx, cmd,
                ctx.swapchainImages[imageIndex], ctx.swapchainExtent);

            vkEndCommandBuffer(cmd);

            VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            VkSubmitInfo submitInfo = {};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.waitSemaphoreCount = 1;
            submitInfo.pWaitSemaphores = &imageAvailableSem;
            submitInfo.pWaitDstStageMask = &waitStage;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &cmd;
            submitInfo.signalSemaphoreCount = 1;
            submitInfo.pSignalSemaphores = &sceneSignalSem;

            vkQueueSubmit(ctx.graphicsQueue, 1, &submitInfo, sceneSubmitFence);
        } else if (useRT) {
            // RT rendering: render to storage image (synchronous)
            renderer->render(ctx);

            // Blit RT output to swapchain with proper semaphore synchronization
            rtBlitCmd = ctx.beginSingleTimeCommands();
            VkCommandBuffer cmd = rtBlitCmd;

            renderer->blitToSwapchain(ctx, cmd, ctx.swapchainImages[imageIndex], ctx.swapchainExtent);

            vkEndCommandBuffer(cmd);

            // Submit with semaphore sync: wait on imageAvailable, signal sceneSignalSem + fence
            VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            VkSubmitInfo submitInfo = {};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.waitSemaphoreCount = 1;
            submitInfo.pWaitSemaphores = &imageAvailableSem;
            submitInfo.pWaitDstStageMask = &waitStage;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &cmd;
            submitInfo.signalSemaphoreCount = 1;
            submitInfo.pSignalSemaphores = &sceneSignalSem;

            vkQueueSubmit(ctx.graphicsQueue, 1, &submitInfo, sceneSubmitFence);
        } else if (useMesh) {
            // Mesh shader rendering to swapchain
            auto* meshPtr = static_cast<MeshRenderer*>(renderer.get());
            meshPtr->renderToSwapchain(ctx,
                ctx.swapchainImages[imageIndex],
                ctx.swapchainImageViews[imageIndex],
                ctx.swapchainFormat, ctx.swapchainExtent,
                imageAvailableSem, sceneSignalSem, sceneSubmitFence);
        } else {
            // Raster rendering to swapchain
            // Cast to RasterRenderer for renderToSwapchain (uses internal command buffer)
            auto* rasterPtr = static_cast<RasterRenderer*>(renderer.get());
            rasterPtr->renderToSwapchain(ctx,
                ctx.swapchainImages[imageIndex],
                ctx.swapchainImageViews[imageIndex],
                ctx.swapchainFormat, ctx.swapchainExtent,
                imageAvailableSem, sceneSignalSem, sceneSubmitFence);
        }

        // Editor overlay pass: record ImGui draw commands and submit
        if (editorActive) {
            editorCmd = ctx.beginSingleTimeCommands();

            // Transition swapchain image to color attachment (scene left it in PRESENT_SRC)
            VkImageMemoryBarrier barrier = {};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            barrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = ctx.swapchainImages[imageIndex];
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            barrier.subresourceRange.baseMipLevel = 0;
            barrier.subresourceRange.levelCount = 1;
            barrier.subresourceRange.baseArrayLayer = 0;
            barrier.subresourceRange.layerCount = 1;
            barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;

            vkCmdPipelineBarrier(editorCmd,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                0, 0, nullptr, 0, nullptr, 1, &barrier);

            // Begin ImGui render pass (loads existing content, ends with PRESENT_SRC)
            VkRenderPassBeginInfo rpBegin = {};
            rpBegin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            rpBegin.renderPass = editorUI.getImGuiRenderPass();
            rpBegin.framebuffer = editorUI.getImGuiFramebuffer(imageIndex);
            rpBegin.renderArea.offset = { 0, 0 };
            rpBegin.renderArea.extent = ctx.swapchainExtent;

            vkCmdBeginRenderPass(editorCmd, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);

            // Render ImGui draw data
            editorUI.endFrame(editorCmd);

            vkCmdEndRenderPass(editorCmd);
            vkEndCommandBuffer(editorCmd);

            // Submit editor pass: wait on sceneRenderSem, signal renderFinishedSem + fence
            VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            VkSubmitInfo submitInfo = {};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.waitSemaphoreCount = 1;
            submitInfo.pWaitSemaphores = &sceneRenderSem;
            submitInfo.pWaitDstStageMask = &waitStage;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &editorCmd;
            submitInfo.signalSemaphoreCount = 1;
            submitInfo.pSignalSemaphores = &renderFinishedSem;

            vkQueueSubmit(ctx.graphicsQueue, 1, &submitInfo, inFlightFence);
        }

        // Present (wait for rendering to finish before presenting)
        VkPresentInfoKHR presentInfo = {};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &renderFinishedSem;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &ctx.swapchain;
        presentInfo.pImageIndices = &imageIndex;

        VkResult presentResult = vkQueuePresentKHR(ctx.graphicsQueue, &presentInfo);
        if (presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR) {
            // Recreate swapchain on next frame
            vkDeviceWaitIdle(ctx.device);
            for (auto& iv : ctx.swapchainImageViews) {
                vkDestroyImageView(ctx.device, iv, nullptr);
            }
            ctx.swapchainImageViews.clear();
            vkDestroySwapchainKHR(ctx.device, ctx.swapchain, nullptr);
            ctx.swapchain = VK_NULL_HANDLE;
            glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
            ctx.createSwapchain(static_cast<uint32_t>(fbWidth), static_cast<uint32_t>(fbHeight));
            if (editorActive) editorNeedsReinit = true;
        }

        // Handle editor pipeline reload requests
        if (editorActive && editorUI.getState().pipelineReloadRequested) {
            editorUI.getState().pipelineReloadRequested = false;
            std::string newBase = editorUI.getState().pendingPipelinePath;
            std::cout << "[editor] Hot-swap pipeline to: " << newBase << std::endl;

            vkDeviceWaitIdle(ctx.device);

            // Destroy old renderer
            if (renderer) {
                renderer->cleanup(ctx);
                renderer.reset();
            }

            // Create new raster renderer with the new pipeline
            try {
                std::string newPath = detectRenderPath(newBase, "");
                auto raster = std::make_unique<RasterRenderer>();
                raster->init(ctx, scene, newBase, newPath,
                             ctx.swapchainExtent.width, ctx.swapchainExtent.height);
                renderer = std::move(raster);
                useRT = false;
                useMesh = false;
                useSplat = false;

                editorUI.getState().currentPipelineBase = newBase;
                opts.shaderBase = newBase;
                std::cout << "[editor] Pipeline hot-swap successful" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "[editor] Pipeline hot-swap failed: " << e.what() << std::endl;
                // Try to restore old renderer
                try {
                    std::string oldPath = detectRenderPath(opts.shaderBase, "");
                    auto raster = std::make_unique<RasterRenderer>();
                    raster->init(ctx, scene, opts.shaderBase, oldPath,
                                 ctx.swapchainExtent.width, ctx.swapchainExtent.height);
                    renderer = std::move(raster);
                } catch (...) {
                    std::cerr << "[editor] Failed to restore previous pipeline!" << std::endl;
                }
            }
        }
    }

    // Cleanup
    vkDeviceWaitIdle(ctx.device);

    g_editorUI = nullptr;
    if (editorActive) {
        editorUI.cleanup(ctx);
    }

    if (renderer) {
        renderer->cleanup(ctx);
    }
    scene.cleanup(ctx);

    vkDestroySemaphore(ctx.device, imageAvailableSem, nullptr);
    vkDestroySemaphore(ctx.device, renderFinishedSem, nullptr);
    vkDestroySemaphore(ctx.device, sceneRenderSem, nullptr);
    vkDestroyFence(ctx.device, inFlightFence, nullptr);

    ctx.cleanup();
    glfwDestroyWindow(window);
    glfwTerminate();

    std::cout << "[info] Done." << std::endl;
    return 0;
}

// --------------------------------------------------------------------------
// Entry point
// --------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    CLIOptions opts = parseArgs(argc, argv);

    // Resolve pipeline from scene if not explicitly given
    if (opts.shaderBase.empty()) {
        try {
            opts.shaderBase = resolveDefaultPipeline(opts.sceneSource);
            std::cout << "[info] Resolved pipeline: " << opts.shaderBase << std::endl;
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
