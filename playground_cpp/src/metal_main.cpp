#include "metal_context.h"
#include "metal_bridge.h"
#include "metal_raster_renderer.h"
#include "metal_mesh_renderer.h"
#include "metal_splat_renderer.h"
#include "metal_scene_manager.h"
#include "metal_renderer_interface.h"
#include "metal_screenshot.h"
#include "reflected_pipeline.h"
#include "scene_light.h"
#include "camera.h"
#include "editor_panels.h"
#include "material_ubo.h"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_metal.h>

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
#include <chrono>

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
static EditorPanels* g_editorPanels = nullptr;

static void mouseButtonCallback(GLFWwindow* window, int button, int action, int /*mods*/) {
    // Let ImGui handle input first when editor is active
    if (g_editorPanels && g_editorPanels->wantCaptureMouse()) return;

    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        g_orbit.dragging = (action == GLFW_PRESS);
        if (g_orbit.dragging) {
            glfwGetCursorPos(window, &g_orbit.lastX, &g_orbit.lastY);
        }
    }
}

static void cursorPosCallback(GLFWwindow* /*window*/, double xpos, double ypos) {
    if (g_editorPanels && g_editorPanels->wantCaptureMouse()) return;

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
    if (g_editorPanels && g_editorPanels->wantCaptureMouse()) return;

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
    bool demoLights = false;
    bool sponzaLights = false;
    bool editor = false;
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
              << "  --editor               Open interactive editor with ImGui overlay\n"
              << "  --headless             Offscreen render only (default)\n"
              << "  --demo-lights          Add 3 demo lights (directional + point + spot) with shadows\n"
              << "  --sponza-lights        Sponza courtyard lights (sun + torch + accent)\n"
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
        } else if (arg == "--editor") {
            opts.editor = true;
            opts.interactive = true;
            opts.headless = false;
        } else if (arg == "--headless") {
            opts.headless = true;
            opts.interactive = false;
        } else if (arg == "--demo-lights") {
            opts.demoLights = true;
        } else if (arg == "--sponza-lights") {
            opts.sponzaLights = true;
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
// Demo lights setup
// --------------------------------------------------------------------------

static void setupDemoLights(MetalSceneManager& scene) {
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

    std::cout << "[metal] Demo lights: 3 lights (directional+shadow, point, spot+shadow)"
              << std::endl;
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
// Sponza courtyard lights: sun + torch + blue accent
// --------------------------------------------------------------------------

static void setupSponzaLights(MetalSceneManager& scene) {
    scene.clearLights();

    // Sun: warm directional light, casts shadow
    SceneLight sun;
    sun.type = SceneLight::Directional;
    sun.direction = glm::normalize(glm::vec3(0.5f, -0.7f, 0.3f));
    sun.color = glm::vec3(1.0f, 0.95f, 0.85f);
    sun.intensity = 8.0f;
    sun.castsShadow = true;
    scene.addLight(sun);

    // Torch: orange spot light inside the courtyard, casts shadow
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

    std::cout << "[metal] Sponza lights: 3 lights (sun+shadow, torch+shadow, blue accent)" << std::endl;
}

// --------------------------------------------------------------------------
// Headless rendering
// --------------------------------------------------------------------------

static int runHeadless(const CLIOptions& opts) {
    std::cout << "[metal] Headless render: scene=\"" << opts.sceneSource
              << "\" (" << opts.width << "x" << opts.height << ")" << std::endl;

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

    try {
        MetalSceneManager scene;
        scene.loadScene(opts.sceneSource);

        // Check for gaussian splat data early — splat scenes use embedded MSL,
        // no external shader pipeline needed
        if (scene.hasSplatData()) {
            std::cout << "[metal] Detected gaussian splat data, using MetalSplatRenderer" << std::endl;
            auto splatR = std::make_unique<MetalSplatRenderer>();
            splatR->init(ctx, scene.getSplatData(), opts.width, opts.height);

            splatR->render(ctx);

            MetalScreenshot::saveTextureToPNG(ctx, splatR->getOutputTexture(),
                                               splatR->getWidth(), splatR->getHeight(),
                                               opts.output);

            splatR->cleanup();
            scene.cleanup();
            ctx.cleanup();
            pool->release();
            std::cout << "[metal] Done." << std::endl;
            return 0;
        }

        // Non-splat path: populate lights from glTF scene data
        if (scene.hasGltfScene()) {
            scene.populateLightsFromGltf(scene.getGltfScene());
        }

        // Set up demo/sponza lights if requested (overrides glTF lights)
        bool useSponzaLights = opts.sponzaLights;
        if (!useSponzaLights && !opts.demoLights) {
            std::string sceneLower = opts.sceneSource;
            std::transform(sceneLower.begin(), sceneLower.end(), sceneLower.begin(), ::tolower);
            if (sceneLower.find("ponza") != std::string::npos) {
                useSponzaLights = true;
                std::cout << "[metal] Auto-detected Sponza scene, enabling sponza lights" << std::endl;
            }
        }
        if (useSponzaLights) {
            setupSponzaLights(scene);
            scene.overrideAutoCamera(
                glm::vec3(-1200.0f, 200.0f, 0.0f),
                glm::vec3(200.0f, 200.0f, 0.0f),
                glm::vec3(0.0f, 1.0f, 0.0f),
                5000.0f);
        } else if (opts.demoLights) {
            setupDemoLights(scene);
        }

        // Non-splat path: resolve shader pipeline
        std::string renderPath = detectRenderPath(opts.shaderBase, opts.forceMode, opts.sceneSource);
        bool needMesh = (renderPath == "mesh");

        std::cout << "[metal] pipeline=\"" << opts.shaderBase
                  << "\" path=" << renderPath << std::endl;

        if (needMesh && !ctx.meshShaderSupported) {
            std::cerr << "[error] Mesh shader pipeline requested but Metal 3 mesh shaders not supported." << std::endl;
            scene.cleanup();
            ctx.cleanup();
            pool->release();
            return 1;
        }


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
        opts.width = opts.editor ? 1280 : 1024;
        opts.height = opts.editor ? 800 : 768;
    }

    std::cout << "[metal] Interactive render: scene=\"" << opts.sceneSource
              << (opts.editor ? " (editor)" : "")
              << " (" << opts.width << "x" << opts.height << ")" << std::endl;

    if (!glfwInit()) {
        std::cerr << "[error] Failed to initialize GLFW" << std::endl;
        return 1;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    const char* windowTitle = opts.editor ? "Lux Editor (Metal)" : "Lux Playground (Metal)";
    GLFWwindow* window = glfwCreateWindow(
        static_cast<int>(opts.width), static_cast<int>(opts.height),
        windowTitle, nullptr, nullptr);
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

    // Initialize editor UI (if --editor)
    EditorPanels editorPanels;
    bool editorActive = opts.editor;
    if (editorActive) {
        try {
            IMGUI_CHECKVERSION();
            ImGui::CreateContext();

            ImGuiIO& io = ImGui::GetIO();
            io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

            ImGui::StyleColorsDark();
            ImGuiStyle& style = ImGui::GetStyle();
            style.WindowRounding = 4.0f;
            style.FrameRounding = 2.0f;
            style.GrabRounding = 2.0f;
            style.WindowBorderSize = 1.0f;
            style.FrameBorderSize = 0.0f;
            style.Alpha = 0.95f;

            ImVec4* colors = style.Colors;
            colors[ImGuiCol_WindowBg] = ImVec4(0.08f, 0.08f, 0.12f, 0.94f);
            colors[ImGuiCol_TitleBg] = ImVec4(0.10f, 0.10f, 0.16f, 1.0f);
            colors[ImGuiCol_TitleBgActive] = ImVec4(0.15f, 0.15f, 0.25f, 1.0f);
            colors[ImGuiCol_Header] = ImVec4(0.20f, 0.20f, 0.35f, 0.50f);
            colors[ImGuiCol_HeaderHovered] = ImVec4(0.30f, 0.30f, 0.50f, 0.70f);
            colors[ImGuiCol_HeaderActive] = ImVec4(0.35f, 0.35f, 0.55f, 0.90f);
            colors[ImGuiCol_Button] = ImVec4(0.20f, 0.22f, 0.35f, 0.80f);
            colors[ImGuiCol_ButtonHovered] = ImVec4(0.30f, 0.32f, 0.50f, 1.0f);
            colors[ImGuiCol_ButtonActive] = ImVec4(0.35f, 0.38f, 0.60f, 1.0f);
            colors[ImGuiCol_FrameBg] = ImVec4(0.12f, 0.12f, 0.20f, 0.80f);
            colors[ImGuiCol_FrameBgHovered] = ImVec4(0.18f, 0.18f, 0.30f, 0.80f);
            colors[ImGuiCol_FrameBgActive] = ImVec4(0.22f, 0.22f, 0.38f, 0.80f);
            colors[ImGuiCol_SliderGrab] = ImVec4(0.40f, 0.45f, 0.70f, 1.0f);
            colors[ImGuiCol_SliderGrabActive] = ImVec4(0.50f, 0.55f, 0.80f, 1.0f);
            colors[ImGuiCol_CheckMark] = ImVec4(0.50f, 0.60f, 1.0f, 1.0f);
            colors[ImGuiCol_Separator] = ImVec4(0.25f, 0.25f, 0.40f, 0.50f);
            colors[ImGuiCol_Tab] = ImVec4(0.15f, 0.15f, 0.25f, 0.90f);
            colors[ImGuiCol_TabHovered] = ImVec4(0.30f, 0.30f, 0.50f, 1.0f);
            colors[ImGuiCol_TabActive] = ImVec4(0.25f, 0.25f, 0.45f, 1.0f);
            colors[ImGuiCol_MenuBarBg] = ImVec4(0.10f, 0.10f, 0.16f, 1.0f);

            ImGui_ImplGlfw_InitForOther(window, true);
            ImGui_ImplMetal_Init(ctx.device);

            editorPanels.getState().currentPipelineBase = opts.shaderBase;
            editorPanels.scanPipelines("shadercache");
            editorPanels.scanPipelines("examples");
            g_editorPanels = &editorPanels;

            std::cout << "[editor] ImGui Metal backend initialized successfully" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[error] Failed to initialize editor: " << e.what() << std::endl;
            editorActive = false;
        }
    }


    MetalSceneManager scene;
    std::unique_ptr<IMetalRenderer> renderer;

    try {
        scene.loadScene(opts.sceneSource);

        // Check for gaussian splat data early — splat scenes use embedded MSL
        if (scene.hasSplatData()) {
            std::cout << "[metal] Detected gaussian splat data, using MetalSplatRenderer" << std::endl;
            auto splatR = std::make_unique<MetalSplatRenderer>();
            splatR->init(ctx, scene.getSplatData(), opts.width, opts.height);
            renderer = std::move(splatR);
        } else {
            // Non-splat: populate lights from glTF
            if (scene.hasGltfScene()) {
                scene.populateLightsFromGltf(scene.getGltfScene());
            }

            // Set up demo/sponza lights if requested
            bool useSponzaLightsI = opts.sponzaLights;
            if (!useSponzaLightsI && !opts.demoLights) {
                std::string sceneLower = opts.sceneSource;
                std::transform(sceneLower.begin(), sceneLower.end(), sceneLower.begin(), ::tolower);
                if (sceneLower.find("ponza") != std::string::npos) {
                    useSponzaLightsI = true;
                    std::cout << "[metal] Auto-detected Sponza scene, enabling sponza lights" << std::endl;
                }
            }
            if (useSponzaLightsI) {
                setupSponzaLights(scene);
                scene.overrideAutoCamera(
                    glm::vec3(-1200.0f, 200.0f, 0.0f),
                    glm::vec3(200.0f, 200.0f, 0.0f),
                    glm::vec3(0.0f, 1.0f, 0.0f),
                    5000.0f);
            } else if (opts.demoLights) {
                setupDemoLights(scene);
            }

            // Resolve pipeline
            std::string renderPath = detectRenderPath(opts.shaderBase, opts.forceMode, opts.sceneSource);
            bool needMesh = (renderPath == "mesh");

            std::cout << "[metal] pipeline=\"" << opts.shaderBase
                      << "\" path=" << renderPath << std::endl;

            if (needMesh && !ctx.meshShaderSupported) {
                throw std::runtime_error("Mesh shader pipeline requested but Metal 3 mesh shaders not supported");
            }

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
        }
    } catch (const std::exception& e) {
        std::cerr << "[error] Failed to initialize renderer: " << e.what() << std::endl;
        if (editorActive) {
            ImGui_ImplMetal_Shutdown();
            ImGui_ImplGlfw_Shutdown();
            ImGui::DestroyContext();
        }
        ctx.cleanup();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    // Initialize orbit camera from splat bounding box if no scene bounds
    if (scene.hasSplatData() && !scene.hasSceneBounds()) {
        const auto& sd = scene.getSplatData();
        glm::vec3 minB(1e9f), maxB(-1e9f);
        for (uint32_t i = 0; i < sd.num_splats; ++i) {
            float x = sd.positions[i * 4 + 0];
            float y = sd.positions[i * 4 + 1];
            float z = sd.positions[i * 4 + 2];
            minB = glm::min(minB, glm::vec3(x, y, z));
            maxB = glm::max(maxB, glm::vec3(x, y, z));
        }
        glm::vec3 center = (minB + maxB) * 0.5f;
        float radius = glm::length(maxB - minB) * 0.5f;
        if (radius < 0.001f) radius = 1.0f;
        glm::vec3 eye = center + glm::vec3(0.0f, radius * 0.5f, radius * 2.5f);
        g_orbit.initFromAutoCamera(eye, center, glm::vec3(0, 1, 0), radius * 10.0f);
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

        // Check for ESC key (only when editor is not capturing keyboard)
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            if (!editorActive || !editorPanels.wantCaptureKeyboard()) {
                glfwSetWindowShouldClose(window, GLFW_TRUE);
                continue;
            }
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
            if (editorActive) {
                // ---- Editor mode: render scene, then overlay ImGui ----
                auto* drawableTex = drawable->texture();

                // Build ImGui render pass descriptor first (needed for NewFrame)
                auto* imguiRPDesc = MTL::RenderPassDescriptor::alloc()->init();
                auto* imguiColorAtt = imguiRPDesc->colorAttachments()->object(0);
                imguiColorAtt->setTexture(drawableTex);
                imguiColorAtt->setLoadAction(MTL::LoadActionLoad);   // preserve scene
                imguiColorAtt->setStoreAction(MTL::StoreActionStore);

                // Begin ImGui frame (Metal backend needs the render pass descriptor)
                ImGui_ImplMetal_NewFrame(imguiRPDesc);
                ImGui_ImplGlfw_NewFrame();
                ImGui::NewFrame();

                // Draw editor panels
                editorPanels.drawPanels(scene);
                editorPanels.updateFPS();

                // Finalize ImGui frame (builds draw data)
                ImGui::Render();

                // Step 1: Render the 3D scene to drawable (no present, synchronous)
                renderer->renderToDrawableNoPresent(ctx, drawable);

                // Step 2: ImGui overlay pass on top of the scene
                auto* imguiCmdBuf = ctx.beginCommandBuffer();
                auto* imguiEnc = imguiCmdBuf->renderCommandEncoder(imguiRPDesc);

                ImGui_ImplMetal_RenderDrawData(ImGui::GetDrawData(), imguiCmdBuf, imguiEnc);

                imguiEnc->endEncoding();
                imguiCmdBuf->presentDrawable(drawable);
                imguiCmdBuf->commit();

                imguiRPDesc->release();
            } else {
                // Non-editor mode: just render the scene directly
                renderer->renderToDrawable(ctx, drawable);
            }
        }
        pool->release();

        // Handle editor glTF load requests
        if (editorActive && editorPanels.getState().gltfLoadRequested) {
            editorPanels.getState().gltfLoadRequested = false;
            std::string newGltf = editorPanels.getState().pendingGltfPath;
            std::cout << "[editor] Loading glTF: " << newGltf << std::endl;

            // Destroy old renderer
            if (renderer) {
                renderer->cleanup();
                renderer.reset();
            }

            // Reload scene
            try {
                scene.cleanup();
                scene.loadScene(newGltf);

                int vertexStride = scene.hasGltfScene() ? 48 : 32;
                scene.uploadToGPU(ctx, vertexStride);
                scene.uploadTextures(ctx);
                scene.loadIBLAssets(ctx, opts.iblName);

                // Rebuild renderer with current pipeline
                std::string rp = detectRenderPath(opts.shaderBase, "", newGltf);
                auto raster = std::make_unique<MetalRasterRenderer>();
                raster->init(ctx, scene, opts.shaderBase, rp, opts.width, opts.height);
                renderer = std::move(raster);

                // Reset camera to new scene bounds
                if (scene.hasSceneBounds()) {
                    g_orbit.initFromAutoCamera(
                        scene.getAutoEye(), scene.getAutoTarget(),
                        scene.getAutoUp(), scene.getAutoFar());
                }

                // Re-init editor state from new scene
                editorPanels.getState().nodeTransforms.clear();
                editorPanels.getState().materialOverrides.clear();
                editorPanels.getState().selectedNodeIndex = -1;
                editorPanels.getState().selectedMaterialIndex = -1;
                editorPanels.resetSceneState();

                opts.sceneSource = newGltf;
                std::cout << "[editor] glTF load successful" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "[editor] glTF load failed: " << e.what() << std::endl;
            }
        }

        // Handle editor material property changes
        if (editorActive && renderer && scene.hasGltfScene()) {
            auto& state = editorPanels.getState();
            auto* rasterR = dynamic_cast<MetalRasterRenderer*>(renderer.get());
            if (rasterR) {
                for (int mi = 0; mi < static_cast<int>(state.materialOverrides.size()); mi++) {
                    auto& ov = state.materialOverrides[mi];
                    if (ov.modified) {
                        ov.modified = false;
                        MaterialUBOData data{};
                        data.baseColorFactor = glm::vec4(ov.baseColor[0], ov.baseColor[1],
                                                         ov.baseColor[2], ov.baseColor[3]);
                        data.metallicFactor = ov.metallic;
                        data.roughnessFactor = ov.roughness;
                        data.emissiveFactor = glm::vec3(ov.emissive[0], ov.emissive[1], ov.emissive[2]);
                        data.emissiveStrength = 1.0f;
                        // Preserve other fields from original material
                        if (mi < static_cast<int>(scene.getGltfScene().materials.size())) {
                            const auto& mat = scene.getGltfScene().materials[mi];
                            data.ior = mat.ior;
                            data.clearcoatFactor = mat.clearcoatFactor;
                            data.clearcoatRoughnessFactor = mat.clearcoatRoughnessFactor;
                            data.sheenColorFactor = mat.sheenColorFactor;
                            data.sheenRoughnessFactor = mat.sheenRoughnessFactor;
                            data.transmissionFactor = mat.transmissionFactor;
                        }
                        rasterR->updateMaterialUBO(mi, data);
                    }
                }
            }
        }

        // Handle editor pipeline reload requests
        if (editorActive && editorPanels.getState().pipelineReloadRequested) {
            editorPanels.getState().pipelineReloadRequested = false;
            std::string newBase = editorPanels.getState().pendingPipelinePath;
            std::cout << "[editor] Hot-swap pipeline to: " << newBase << std::endl;

            // Destroy old renderer
            if (renderer) {
                renderer->cleanup();
                renderer.reset();
            }

            // Create new raster renderer with the new pipeline
            try {
                std::string newPath = detectRenderPath(newBase, "", opts.sceneSource);
                auto raster = std::make_unique<MetalRasterRenderer>();
                raster->init(ctx, scene, newBase, newPath, opts.width, opts.height);
                renderer = std::move(raster);

                opts.shaderBase = newBase;
                editorPanels.getState().currentPipelineBase = newBase;
                std::cout << "[editor] Pipeline hot-swap successful" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "[editor] Pipeline hot-swap failed: " << e.what() << std::endl;
            }
        }
    }

    // Cleanup
    if (editorActive) {
        ImGui_ImplMetal_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        g_editorPanels = nullptr;
    }

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
            // Pipeline resolution failure is OK for splat-only glTF scenes —
            // they use embedded MSL compute shaders, no external pipeline needed.
            if (MetalSceneManager::isGltfFile(opts.sceneSource)) {
                std::cout << "[metal] No shader pipeline found (may be splat-only scene)" << std::endl;
            } else {
                std::cerr << "[error] " << e.what() << std::endl;
                return 1;
            }
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
