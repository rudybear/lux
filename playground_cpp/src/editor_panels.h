#pragma once

/**
 * EditorPanels - Backend-agnostic ImGui panel drawing for the Lux editor.
 *
 * Provides: scene tree panel, properties panel, pipeline selector,
 * viewport controls (camera mode, wireframe, background color, FPS counter).
 *
 * This class uses only ImGui APIs and has NO backend-specific dependencies.
 * Both Vulkan and Metal editor UIs use this to draw their panels.
 */

#include "editor_state.h"
#include "gltf_loader.h"

#include <imgui.h>
#include <string>

// Forward declarations - use MetalSceneManager on Metal, SceneManager on Vulkan.
// EditorPanels uses a duck-typing approach: it accesses scene data through
// methods that both SceneManager and MetalSceneManager provide.

class EditorPanels {
public:
    EditorPanels() = default;

    /**
     * Draw all editor panels. Call between ImGui::NewFrame() and ImGui::Render().
     * Template parameter allows both SceneManager and MetalSceneManager.
     */
    template<typename SceneType>
    void drawPanels(SceneType& scene) {
        initFromScene(scene);
        drawMenuBar();
        drawViewportControls();
        drawSceneTreePanel(scene);
        drawPropertiesPanel(scene);
        drawPipelineSelector();
    }

    /**
     * Update FPS counter. Call once per frame.
     */
    void updateFPS();

    /**
     * Scan a directory for available .vert.spv/.frag.spv shader pipelines.
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
     * Set the pipeline reload callback.
     */
    void setPipelineReloadCallback(PipelineReloadCallback cb) { m_reloadCallback = std::move(cb); }

    /**
     * Reset scene state (call after loading a new glTF).
     */
    void resetSceneState() { m_sceneInitialized = false; }

private:
    EditorState m_state;
    PipelineReloadCallback m_reloadCallback;
    bool m_sceneInitialized = false;

    // Panel drawing methods
    void drawMenuBar();
    void drawPipelineSelector();
    void drawViewportControls();

    // Scene tree helpers
    void drawNodeTree(const GltfScene& gltfScene, int nodeIndex);

    // Templated panel methods (need scene access)
    template<typename SceneType>
    void drawSceneTreePanel(SceneType& scene);

    template<typename SceneType>
    void drawPropertiesPanel(SceneType& scene);

    // Initialize editable state from scene data
    template<typename SceneType>
    void initFromScene(SceneType& scene);
};

// ==========================================================================
// Template implementations (must be in header)
// ==========================================================================

template<typename SceneType>
void EditorPanels::initFromScene(SceneType& scene) {
    if (m_sceneInitialized) return;

    if (scene.hasGltfScene()) {
        const auto& gltfScene = scene.getGltfScene();

        // Initialize node transforms
        m_state.nodeTransforms.resize(gltfScene.nodes.size());
        for (size_t i = 0; i < gltfScene.nodes.size(); i++) {
            const auto& node = gltfScene.nodes[i];
            const glm::mat4& m = node.worldTransform;
            m_state.nodeTransforms[i].position[0] = m[3][0];
            m_state.nodeTransforms[i].position[1] = m[3][1];
            m_state.nodeTransforms[i].position[2] = m[3][2];

            float sx = glm::length(glm::vec3(m[0]));
            float sy = glm::length(glm::vec3(m[1]));
            float sz = glm::length(glm::vec3(m[2]));
            m_state.nodeTransforms[i].scale[0] = sx;
            m_state.nodeTransforms[i].scale[1] = sy;
            m_state.nodeTransforms[i].scale[2] = sz;

            if (sx > 0.0001f && sy > 0.0001f && sz > 0.0001f) {
                glm::mat3 rot(
                    glm::vec3(m[0]) / sx,
                    glm::vec3(m[1]) / sy,
                    glm::vec3(m[2]) / sz
                );
                float pitch = asinf(glm::clamp(-rot[2][0], -1.0f, 1.0f));
                float yaw, roll;
                if (fabsf(rot[2][0]) < 0.999f) {
                    yaw = atan2f(rot[2][1], rot[2][2]);
                    roll = atan2f(rot[1][0], rot[0][0]);
                } else {
                    yaw = atan2f(-rot[0][1], rot[1][1]);
                    roll = 0.0f;
                }
                m_state.nodeTransforms[i].rotation[0] = glm::degrees(pitch);
                m_state.nodeTransforms[i].rotation[1] = glm::degrees(yaw);
                m_state.nodeTransforms[i].rotation[2] = glm::degrees(roll);
            }
        }

        // Initialize material overrides
        m_state.materialOverrides.resize(gltfScene.materials.size());
        for (size_t i = 0; i < gltfScene.materials.size(); i++) {
            const auto& mat = gltfScene.materials[i];
            auto& ov = m_state.materialOverrides[i];
            ov.baseColor[0] = mat.baseColor.r;
            ov.baseColor[1] = mat.baseColor.g;
            ov.baseColor[2] = mat.baseColor.b;
            ov.baseColor[3] = mat.baseColor.a;
            ov.metallic = mat.metallic;
            ov.roughness = mat.roughness;
            ov.emissive[0] = mat.emissive.r;
            ov.emissive[1] = mat.emissive.g;
            ov.emissive[2] = mat.emissive.b;
        }
    }

    m_sceneInitialized = true;
}

template<typename SceneType>
void EditorPanels::drawSceneTreePanel(SceneType& scene) {
    ImGuiIO& io = ImGui::GetIO();
    float menuBarHeight = ImGui::GetFrameHeight();
    float toolbarHeight = 36.0f;
    float panelWidth = 280.0f;
    float panelTop = menuBarHeight + toolbarHeight;
    float panelHeight = io.DisplaySize.y - panelTop;

    ImGui::SetNextWindowPos(ImVec2(0, panelTop));
    ImGui::SetNextWindowSize(ImVec2(panelWidth, panelHeight));
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse;

    if (ImGui::Begin("Scene Tree", nullptr, flags)) {
        // glTF load section
        ImGui::Text("Load Scene:");
        ImGui::PushItemWidth(-60);
        ImGui::InputText("##GltfPath", m_state.gltfPathInput, sizeof(m_state.gltfPathInput));
        ImGui::PopItemWidth();
        ImGui::SameLine();
        if (ImGui::Button("Load")) {
            std::string path(m_state.gltfPathInput);
            if (!path.empty()) {
                m_state.pendingGltfPath = path;
                m_state.gltfLoadRequested = true;
            }
        }

        ImGui::Separator();

        ImGui::Text("Source: %s", scene.getSceneSource().c_str());

        if (scene.hasGltfScene()) {
            const auto& gltfScene = scene.getGltfScene();
            ImGui::Text("Nodes: %zu  Meshes: %zu  Materials: %zu",
                        gltfScene.nodes.size(), gltfScene.meshes.size(), gltfScene.materials.size());

            ImGui::Separator();

            if (ImGui::CollapsingHeader("Nodes", ImGuiTreeNodeFlags_DefaultOpen)) {
                for (int rootIdx : gltfScene.rootNodes) {
                    drawNodeTree(gltfScene, rootIdx);
                }
            }

            if (ImGui::CollapsingHeader("Materials")) {
                for (size_t i = 0; i < gltfScene.materials.size(); i++) {
                    const auto& mat = gltfScene.materials[i];
                    std::string label = mat.name.empty()
                        ? "Material " + std::to_string(i)
                        : mat.name;

                    bool isSelected = (m_state.selectedMaterialIndex == static_cast<int>(i));
                    if (ImGui::Selectable(label.c_str(), isSelected)) {
                        m_state.selectedMaterialIndex = static_cast<int>(i);
                        m_state.selectedNodeIndex = -1;
                    }
                }
            }

            if (ImGui::CollapsingHeader("Lights")) {
                // Note: lights panel - Metal scenes don't expose getLights() directly,
                // so we just show material/mesh counts for non-Vulkan backends.
                ImGui::Text("(Light editing available in Vulkan backend)");
            }
        } else {
            ImGui::Text("(Procedural scene - no node hierarchy)");
            ImGui::Text("Vertices: %zu", scene.getVertices().size());
        }
    }
    ImGui::End();
}

template<typename SceneType>
void EditorPanels::drawPropertiesPanel(SceneType& scene) {
    ImGuiIO& io = ImGui::GetIO();
    float menuBarHeight = ImGui::GetFrameHeight();
    float toolbarHeight = 36.0f;
    float panelWidth = 300.0f;
    float panelTop = menuBarHeight + toolbarHeight;
    float panelHeight = io.DisplaySize.y - panelTop;

    ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x - panelWidth, panelTop));
    ImGui::SetNextWindowSize(ImVec2(panelWidth, panelHeight));
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse;

    if (ImGui::Begin("Properties", nullptr, flags)) {
        if (!scene.hasGltfScene()) {
            ImGui::Text("No glTF scene loaded.");
            ImGui::End();
            return;
        }

        const auto& gltfScene = scene.getGltfScene();

        // Node properties
        if (m_state.selectedNodeIndex >= 0 &&
            m_state.selectedNodeIndex < static_cast<int>(gltfScene.nodes.size())) {

            const auto& node = gltfScene.nodes[m_state.selectedNodeIndex];

            ImGui::Text("Node: %s", node.name.empty()
                ? ("Node " + std::to_string(m_state.selectedNodeIndex)).c_str()
                : node.name.c_str());
            ImGui::Separator();

            if (m_state.selectedNodeIndex < static_cast<int>(m_state.nodeTransforms.size())) {
                auto& xform = m_state.nodeTransforms[m_state.selectedNodeIndex];

                if (ImGui::CollapsingHeader("Transform", ImGuiTreeNodeFlags_DefaultOpen)) {
                    ImGui::DragFloat3("Position", xform.position, 0.1f);
                    ImGui::DragFloat3("Rotation", xform.rotation, 1.0f, -180.0f, 180.0f);
                    ImGui::DragFloat3("Scale", xform.scale, 0.01f, 0.001f, 100.0f);
                }
            }

            if (node.meshIndex >= 0 && node.meshIndex < static_cast<int>(gltfScene.meshes.size())) {
                const auto& mesh = gltfScene.meshes[node.meshIndex];
                if (ImGui::CollapsingHeader("Mesh Info", ImGuiTreeNodeFlags_DefaultOpen)) {
                    ImGui::Text("Name: %s", mesh.name.empty() ? "(unnamed)" : mesh.name.c_str());
                    ImGui::Text("Vertices: %zu", mesh.vertices.size());
                    ImGui::Text("Indices: %zu", mesh.indices.size());
                    ImGui::Text("Material: %d", mesh.materialIndex);
                    ImGui::Text("Has tangents: %s", mesh.hasTangents ? "yes" : "no");
                }
            }

            ImGui::Separator();
        }

        // Material properties
        if (m_state.selectedMaterialIndex >= 0 &&
            m_state.selectedMaterialIndex < static_cast<int>(gltfScene.materials.size())) {

            const auto& mat = gltfScene.materials[m_state.selectedMaterialIndex];
            int matIdx = m_state.selectedMaterialIndex;

            std::string matName = mat.name.empty()
                ? "Material " + std::to_string(matIdx)
                : mat.name;
            ImGui::Text("Material: %s", matName.c_str());
            ImGui::Separator();

            if (matIdx < static_cast<int>(m_state.materialOverrides.size())) {
                auto& ov = m_state.materialOverrides[matIdx];

                if (ImGui::CollapsingHeader("PBR Properties", ImGuiTreeNodeFlags_DefaultOpen)) {
                    ov.modified |= ImGui::ColorEdit4("Base Color", ov.baseColor);
                    ov.modified |= ImGui::SliderFloat("Metallic", &ov.metallic, 0.0f, 1.0f);
                    ov.modified |= ImGui::SliderFloat("Roughness", &ov.roughness, 0.0f, 1.0f);
                    ov.modified |= ImGui::ColorEdit3("Emissive", ov.emissive);
                }

                if (ImGui::CollapsingHeader("Material Info")) {
                    ImGui::Text("Alpha mode: %s", mat.alphaMode.c_str());
                    ImGui::Text("Double sided: %s", mat.doubleSided ? "yes" : "no");
                    ImGui::Text("Has normal map: %s", mat.normal_tex.valid() ? "yes" : "no");
                    ImGui::Text("Has occlusion: %s", mat.occlusion_tex.valid() ? "yes" : "no");
                    ImGui::Text("Has emissive tex: %s", mat.emissive_tex.valid() ? "yes" : "no");
                    ImGui::Text("Has metallic/roughness tex: %s", mat.metallic_roughness_tex.valid() ? "yes" : "no");

                    if (mat.hasClearcoat) {
                        ImGui::Text("Clearcoat: %.2f (roughness: %.2f)",
                                    mat.clearcoatFactor, mat.clearcoatRoughnessFactor);
                    }
                    if (mat.hasSheen) {
                        ImGui::Text("Sheen color: (%.2f, %.2f, %.2f)",
                                    mat.sheenColorFactor.r, mat.sheenColorFactor.g, mat.sheenColorFactor.b);
                    }
                    if (mat.hasTransmission) {
                        ImGui::Text("Transmission: %.2f", mat.transmissionFactor);
                    }
                    ImGui::Text("IOR: %.3f", mat.ior);
                    if (mat.isUnlit) {
                        ImGui::Text("Unlit: yes");
                    }
                }
            }
        } else if (m_state.selectedNodeIndex < 0) {
            ImGui::Text("Select a node or material to view properties.");
        }

        ImGui::Separator();
        if (ImGui::CollapsingHeader("Active Pipeline")) {
            ImGui::Text("Base: %s", m_state.currentPipelineBase.c_str());
        }
    }
    ImGui::End();
}
