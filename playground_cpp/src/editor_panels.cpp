#include "editor_panels.h"

#include <imgui.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <iostream>
#include <filesystem>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <set>

namespace fs = std::filesystem;

// --------------------------------------------------------------------------
// FPS
// --------------------------------------------------------------------------

void EditorPanels::updateFPS() {
    m_state.frameCount++;
    auto now = std::chrono::high_resolution_clock::now();
    float elapsed = std::chrono::duration<float>(now - m_state.lastFpsUpdate).count();
    if (elapsed >= 0.5f) {
        m_state.fps = static_cast<float>(m_state.frameCount) / elapsed;
        m_state.frameTimeMs = elapsed / static_cast<float>(m_state.frameCount) * 1000.0f;
        m_state.frameCount = 0;
        m_state.lastFpsUpdate = now;
    }
}

bool EditorPanels::wantCaptureMouse() const {
    return ImGui::GetIO().WantCaptureMouse;
}

bool EditorPanels::wantCaptureKeyboard() const {
    return ImGui::GetIO().WantCaptureKeyboard;
}

// --------------------------------------------------------------------------
// Pipeline scanning
// --------------------------------------------------------------------------

void EditorPanels::scanPipelines(const std::string& directory) {
    try {
        if (!fs::exists(directory)) {
            std::cout << "[editor] Pipeline directory not found: " << directory << std::endl;
            return;
        }

        std::set<std::string> existingBases(m_state.availablePipelines.begin(),
                                             m_state.availablePipelines.end());

        std::set<std::string> newBases;
        for (const auto& entry : fs::recursive_directory_iterator(directory)) {
            if (!entry.is_regular_file()) continue;
            std::string path = entry.path().string();
            std::replace(path.begin(), path.end(), '\\', '/');

            std::string suffix = ".vert.spv";
            if (path.size() > suffix.size() && path.substr(path.size() - suffix.size()) == suffix) {
                std::string base = path.substr(0, path.size() - suffix.size());
                newBases.insert(base);
            }
            suffix = ".frag.spv";
            if (path.size() > suffix.size() && path.substr(path.size() - suffix.size()) == suffix) {
                std::string base = path.substr(0, path.size() - suffix.size());
                newBases.insert(base);
            }
        }

        int added = 0;
        for (const auto& base : newBases) {
            if (existingBases.find(base) == existingBases.end()) {
                m_state.availablePipelines.push_back(base);
                added++;
            }
        }

        std::sort(m_state.availablePipelines.begin(), m_state.availablePipelines.end());

        std::cout << "[editor] Found " << added
                  << " pipeline(s) in " << directory
                  << " (total: " << m_state.availablePipelines.size() << ")" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[editor] Error scanning pipelines: " << e.what() << std::endl;
    }
}

// --------------------------------------------------------------------------
// Menu bar
// --------------------------------------------------------------------------

void EditorPanels::drawMenuBar() {
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Load glTF...")) {
                // Will be handled by the path input in scene tree panel
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Quit", "ESC")) {
                // Signal quit - handled by main loop via GLFW
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("View")) {
            ImGui::MenuItem("Wireframe", "W", &m_state.wireframe);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }
}

// --------------------------------------------------------------------------
// Viewport controls (top toolbar)
// --------------------------------------------------------------------------

void EditorPanels::drawViewportControls() {
    ImGuiIO& io = ImGui::GetIO();
    float menuBarHeight = ImGui::GetFrameHeight();

    ImGui::SetNextWindowPos(ImVec2(0, menuBarHeight));
    ImGui::SetNextWindowSize(ImVec2(io.DisplaySize.x, 36));
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                             ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar |
                             ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8, 4));
    if (ImGui::Begin("##Toolbar", nullptr, flags)) {
        // Camera mode
        ImGui::Text("Camera:");
        ImGui::SameLine();
        int camMode = static_cast<int>(m_state.cameraMode);
        ImGui::PushItemWidth(80);
        if (ImGui::Combo("##CamMode", &camMode, "Orbit\0Fly\0")) {
            m_state.cameraMode = static_cast<CameraMode>(camMode);
        }
        ImGui::PopItemWidth();

        ImGui::SameLine();
        ImGui::Separator();
        ImGui::SameLine();

        // Wireframe toggle
        bool prevWireframe = m_state.wireframe;
        ImGui::Checkbox("Wireframe", &m_state.wireframe);
        if (m_state.wireframe != prevWireframe) {
            m_state.wireframeChanged = true;
        }

        ImGui::SameLine();
        ImGui::Separator();
        ImGui::SameLine();

        // Background color
        ImGui::Text("BG:");
        ImGui::SameLine();
        ImGui::PushItemWidth(120);
        if (ImGui::ColorEdit3("##BG", m_state.backgroundColor,
                              ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel)) {
            m_state.backgroundChanged = true;
        }
        ImGui::PopItemWidth();

        ImGui::SameLine();
        ImGui::Separator();
        ImGui::SameLine();

        // FPS counter
        ImGui::Text("%.1f FPS (%.2f ms)", m_state.fps, m_state.frameTimeMs);
    }
    ImGui::End();
    ImGui::PopStyleVar();
}

// --------------------------------------------------------------------------
// Node tree helper
// --------------------------------------------------------------------------

void EditorPanels::drawNodeTree(const GltfScene& gltfScene, int nodeIndex) {
    if (nodeIndex < 0 || nodeIndex >= static_cast<int>(gltfScene.nodes.size())) return;

    const auto& node = gltfScene.nodes[nodeIndex];
    std::string label = node.name.empty()
        ? "Node " + std::to_string(nodeIndex)
        : node.name;

    if (node.meshIndex >= 0) label += " [Mesh]";
    if (node.cameraIndex >= 0) label += " [Camera]";
    if (node.lightIndex >= 0) label += " [Light]";

    ImGuiTreeNodeFlags nodeFlags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth;
    if (node.children.empty()) nodeFlags |= ImGuiTreeNodeFlags_Leaf;
    if (m_state.selectedNodeIndex == nodeIndex) nodeFlags |= ImGuiTreeNodeFlags_Selected;

    bool opened = ImGui::TreeNodeEx(
        (label + "##" + std::to_string(nodeIndex)).c_str(), nodeFlags);

    if (ImGui::IsItemClicked()) {
        m_state.selectedNodeIndex = nodeIndex;
        m_state.selectedMaterialIndex = -1;
        if (node.meshIndex >= 0 && node.meshIndex < static_cast<int>(gltfScene.meshes.size())) {
            m_state.selectedMaterialIndex = gltfScene.meshes[node.meshIndex].materialIndex;
        }
    }

    if (opened) {
        for (int childIdx : node.children) {
            drawNodeTree(gltfScene, childIdx);
        }
        ImGui::TreePop();
    }
}

// --------------------------------------------------------------------------
// Pipeline selector panel
// --------------------------------------------------------------------------

void EditorPanels::drawPipelineSelector() {
    ImGuiIO& io = ImGui::GetIO();
    float leftPanelWidth = 280.0f;
    float rightPanelWidth = 300.0f;
    float panelWidth = io.DisplaySize.x - leftPanelWidth - rightPanelWidth;
    float panelHeight = 180.0f;
    float panelY = io.DisplaySize.y - panelHeight;

    ImGui::SetNextWindowPos(ImVec2(leftPanelWidth, panelY));
    ImGui::SetNextWindowSize(ImVec2(panelWidth, panelHeight));
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse;

    if (ImGui::Begin("Pipeline", nullptr, flags)) {
        ImGui::Text("Scan directory:");
        ImGui::SameLine();
        ImGui::PushItemWidth(200);
        char scanDir[256];
        strncpy(scanDir, m_state.pipelineScanDir.c_str(), sizeof(scanDir) - 1);
        scanDir[sizeof(scanDir) - 1] = '\0';
        if (ImGui::InputText("##ScanDir", scanDir, sizeof(scanDir),
                             ImGuiInputTextFlags_EnterReturnsTrue)) {
            m_state.pipelineScanDir = scanDir;
            scanPipelines(m_state.pipelineScanDir);
        }
        ImGui::PopItemWidth();
        ImGui::SameLine();
        if (ImGui::Button("Scan")) {
            m_state.pipelineScanDir = scanDir;
            scanPipelines(m_state.pipelineScanDir);
        }

        if (!m_state.availablePipelines.empty()) {
            ImGui::Text("Available pipelines:");

            std::string currentLabel = m_state.selectedPipelineIndex >= 0 &&
                m_state.selectedPipelineIndex < static_cast<int>(m_state.availablePipelines.size())
                    ? m_state.availablePipelines[m_state.selectedPipelineIndex]
                    : "(none)";

            ImGui::PushItemWidth(-80);
            if (ImGui::BeginCombo("##Pipelines", currentLabel.c_str())) {
                for (int i = 0; i < static_cast<int>(m_state.availablePipelines.size()); i++) {
                    bool isSelected = (m_state.selectedPipelineIndex == i);
                    if (ImGui::Selectable(m_state.availablePipelines[i].c_str(), isSelected)) {
                        m_state.selectedPipelineIndex = i;
                    }
                    if (isSelected) ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
            ImGui::PopItemWidth();
            ImGui::SameLine();

            if (ImGui::Button("Apply")) {
                if (m_state.selectedPipelineIndex >= 0 &&
                    m_state.selectedPipelineIndex < static_cast<int>(m_state.availablePipelines.size())) {
                    m_state.pendingPipelinePath = m_state.availablePipelines[m_state.selectedPipelineIndex];
                    m_state.pipelineReloadRequested = true;
                }
            }
        } else {
            ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.3f, 1.0f),
                               "No pipelines found. Click 'Scan' to search.");
        }

        ImGui::Separator();
        ImGui::Text("Current: %s", m_state.currentPipelineBase.c_str());
    }
    ImGui::End();
}
