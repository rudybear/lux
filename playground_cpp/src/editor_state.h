#pragma once

/**
 * EditorState - Shared editor state used by both Vulkan and Metal editor UIs.
 *
 * Contains selection, viewport controls, pipeline selection, FPS tracking,
 * node transforms, material overrides, etc.
 *
 * This header has NO backend-specific dependencies (no Vulkan, no Metal).
 */

#include <string>
#include <vector>
#include <functional>
#include <chrono>

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
