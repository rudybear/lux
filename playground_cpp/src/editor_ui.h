#pragma once

/**
 * EditorUI - Dear ImGui-based scene editor overlay for the Lux playground.
 *
 * Provides: scene tree panel, properties panel, pipeline selector,
 * viewport controls (camera mode, wireframe, background color, FPS counter).
 *
 * The editor is activated with --editor and renders as an ImGui overlay
 * on top of the 3D viewport.
 */

#include <vulkan/vulkan.h>
#include "vulkan_context.h"
#include "scene_manager.h"
#include "gltf_loader.h"

#include <imgui.h>

#include <string>
#include <vector>
#include <functional>
#include <chrono>

struct GLFWwindow;

// Forward declarations
class RasterRenderer;

// Callback type for pipeline hot-swap
using PipelineReloadCallback = std::function<void(const std::string& newPipelineBase)>;

// Camera mode for viewport controls
enum class CameraMode {
    Orbit = 0,
    Fly = 1
};

// Editor state shared between UI and main loop
struct EditorState {
    // Selection
    int selectedNodeIndex = -1;
    int selectedMaterialIndex = -1;

    // Viewport controls
    CameraMode cameraMode = CameraMode::Orbit;
    bool wireframe = false;
    bool wireframeChanged = false;
    float backgroundColor[3] = { 0.1f, 0.1f, 0.15f };
    bool backgroundChanged = false;

    // Pipeline selection
    std::string currentPipelineBase;
    std::vector<std::string> availablePipelines;
    int selectedPipelineIndex = 0;
    bool pipelineReloadRequested = false;
    std::string pendingPipelinePath;

    // Pipeline scan directory
    std::string pipelineScanDir = "shadercache";

    // File load
    std::string pendingGltfPath;
    bool gltfLoadRequested = false;
    char gltfPathInput[512] = {};

    // FPS tracking
    float fps = 0.0f;
    float frameTimeMs = 0.0f;
    int frameCount = 0;
    std::chrono::high_resolution_clock::time_point lastFpsUpdate;

    // Node transforms (editable copies from glTF scene)
    struct NodeTransform {
        float position[3] = { 0.0f, 0.0f, 0.0f };
        float rotation[3] = { 0.0f, 0.0f, 0.0f }; // euler degrees
        float scale[3]    = { 1.0f, 1.0f, 1.0f };
    };
    std::vector<NodeTransform> nodeTransforms;

    // Material overrides (editable copies)
    struct MaterialOverride {
        float baseColor[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
        float metallic = 0.0f;
        float roughness = 1.0f;
        float emissive[3] = { 0.0f, 0.0f, 0.0f };
        bool modified = false;
    };
    std::vector<MaterialOverride> materialOverrides;
};

class EditorUI {
public:
    EditorUI() = default;
    ~EditorUI();

    /**
     * Initialize ImGui with Vulkan and GLFW backends.
     * Must be called after VulkanContext::init() and swapchain creation.
     */
    void init(VulkanContext& ctx, GLFWwindow* window, uint32_t imageCount);

    /**
     * Cleanup all ImGui resources.
     */
    void cleanup(VulkanContext& ctx);

    /**
     * Begin a new ImGui frame. Call once per frame before drawing panels.
     */
    void beginFrame();

    /**
     * Draw all editor panels. Call between beginFrame() and endFrame().
     */
    void drawPanels(SceneManager& scene);

    /**
     * Finalize the ImGui frame and record draw commands into the given command buffer.
     * The command buffer must be in a recording state with an active render pass
     * (the ImGui render pass).
     */
    void endFrame(VkCommandBuffer cmd);

    /**
     * Record ImGui rendering into a command buffer that renders to the given
     * swapchain image. This creates a simple render pass, records ImGui draw
     * data, and ends the render pass.
     */
    void renderToSwapchain(VulkanContext& ctx, VkCommandBuffer cmd,
                           VkImage swapImage, VkImageView swapView,
                           VkFormat swapFormat, VkExtent2D extent);

    /**
     * Update FPS counter. Call once per frame.
     */
    void updateFPS();

    /**
     * Scan a directory for available .lux.json reflection files (pipelines).
     */
    void scanPipelines(const std::string& directory);

    /**
     * Check if ImGui wants to capture mouse input (hovering over a panel).
     */
    bool wantCaptureMouse() const;

    /**
     * Check if ImGui wants to capture keyboard input.
     */
    bool wantCaptureKeyboard() const;

    /**
     * Access the shared editor state.
     */
    EditorState& getState() { return m_state; }
    const EditorState& getState() const { return m_state; }

    /**
     * Access the ImGui render pass (needed by main loop for command recording).
     */
    VkRenderPass getImGuiRenderPass() const { return m_imguiRenderPass; }

    /**
     * Access ImGui framebuffer for a given swapchain image index.
     */
    VkFramebuffer getImGuiFramebuffer(uint32_t imageIndex) const {
        if (imageIndex < m_imguiFramebuffers.size())
            return m_imguiFramebuffers[imageIndex];
        return VK_NULL_HANDLE;
    }

    /**
     * Set the pipeline reload callback.
     */
    void setPipelineReloadCallback(PipelineReloadCallback cb) { m_reloadCallback = std::move(cb); }

private:
    bool m_initialized = false;
    EditorState m_state;
    PipelineReloadCallback m_reloadCallback;

    // ImGui Vulkan resources
    VkDescriptorPool m_imguiDescriptorPool = VK_NULL_HANDLE;
    VkRenderPass m_imguiRenderPass = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> m_imguiFramebuffers;

    // Panel drawing methods
    void drawMenuBar();
    void drawSceneTreePanel(SceneManager& scene);
    void drawPropertiesPanel(SceneManager& scene);
    void drawPipelineSelector();
    void drawViewportControls();

    // Scene tree helpers
    void drawNodeTree(const GltfScene& gltfScene, int nodeIndex);

    // Initialize editable state from scene data
    void initFromScene(SceneManager& scene);
    bool m_sceneInitialized = false;

public:
    // Reset scene state (call after loading a new glTF)
    void resetSceneState() { m_sceneInitialized = false; }
};
