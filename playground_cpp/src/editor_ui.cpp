#include "editor_ui.h"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

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
// Lifecycle
// --------------------------------------------------------------------------

EditorUI::~EditorUI() {
    // cleanup() should be called explicitly before destruction
}

void EditorUI::init(VulkanContext& ctx, GLFWwindow* window, uint32_t imageCount) {
    if (m_initialized) return;

    // Create a descriptor pool for ImGui
    VkDescriptorPoolSize poolSizes[] = {
        { VK_DESCRIPTOR_TYPE_SAMPLER, 100 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 100 },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 100 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 100 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 100 },
        { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 100 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 100 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 100 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 100 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 100 },
        { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 100 }
    };

    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.maxSets = 100;
    poolInfo.poolSizeCount = static_cast<uint32_t>(std::size(poolSizes));
    poolInfo.pPoolSizes = poolSizes;

    if (vkCreateDescriptorPool(ctx.device, &poolInfo, nullptr, &m_imguiDescriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create ImGui descriptor pool");
    }

    // Create a render pass for ImGui (renders on top of existing content)
    VkAttachmentDescription colorAttachment = {};
    colorAttachment.format = ctx.swapchainFormat;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; // preserve existing content
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorRef = {};
    colorRef.attachment = 0;
    colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;

    VkSubpassDependency dependency = {};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;

    VkRenderPassCreateInfo rpInfo = {};
    rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpInfo.attachmentCount = 1;
    rpInfo.pAttachments = &colorAttachment;
    rpInfo.subpassCount = 1;
    rpInfo.pSubpasses = &subpass;
    rpInfo.dependencyCount = 1;
    rpInfo.pDependencies = &dependency;

    if (vkCreateRenderPass(ctx.device, &rpInfo, nullptr, &m_imguiRenderPass) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create ImGui render pass");
    }

    // Create framebuffers for each swapchain image
    m_imguiFramebuffers.resize(ctx.swapchainImageViews.size());
    for (size_t i = 0; i < ctx.swapchainImageViews.size(); i++) {
        VkFramebufferCreateInfo fbInfo = {};
        fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fbInfo.renderPass = m_imguiRenderPass;
        fbInfo.attachmentCount = 1;
        fbInfo.pAttachments = &ctx.swapchainImageViews[i];
        fbInfo.width = ctx.swapchainExtent.width;
        fbInfo.height = ctx.swapchainExtent.height;
        fbInfo.layers = 1;

        if (vkCreateFramebuffer(ctx.device, &fbInfo, nullptr, &m_imguiFramebuffers[i]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create ImGui framebuffer");
        }
    }

    // Initialize ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    // Set up ImGui style
    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 4.0f;
    style.FrameRounding = 2.0f;
    style.GrabRounding = 2.0f;
    style.WindowBorderSize = 1.0f;
    style.FrameBorderSize = 0.0f;
    style.Alpha = 0.95f;

    // Lux-themed dark color scheme
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

    // Initialize ImGui GLFW backend
    ImGui_ImplGlfw_InitForVulkan(window, true);

    // Initialize ImGui Vulkan backend
    ImGui_ImplVulkan_InitInfo initInfo = {};
    initInfo.Instance = ctx.instance;
    initInfo.PhysicalDevice = ctx.physicalDevice;
    initInfo.Device = ctx.device;
    initInfo.QueueFamily = ctx.graphicsQueueFamily;
    initInfo.Queue = ctx.graphicsQueue;
    initInfo.DescriptorPool = m_imguiDescriptorPool;
    initInfo.MinImageCount = imageCount;
    initInfo.ImageCount = imageCount;
    initInfo.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    initInfo.RenderPass = m_imguiRenderPass;
    initInfo.Subpass = 0;

    ImGui_ImplVulkan_Init(&initInfo);

    // Upload fonts
    ImGui_ImplVulkan_CreateFontsTexture();

    m_state.lastFpsUpdate = std::chrono::high_resolution_clock::now();
    m_initialized = true;

    std::cout << "[editor] ImGui initialized successfully" << std::endl;
}

void EditorUI::cleanup(VulkanContext& ctx) {
    if (!m_initialized) return;

    vkDeviceWaitIdle(ctx.device);

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    for (auto fb : m_imguiFramebuffers) {
        if (fb != VK_NULL_HANDLE) {
            vkDestroyFramebuffer(ctx.device, fb, nullptr);
        }
    }
    m_imguiFramebuffers.clear();

    if (m_imguiRenderPass != VK_NULL_HANDLE) {
        vkDestroyRenderPass(ctx.device, m_imguiRenderPass, nullptr);
        m_imguiRenderPass = VK_NULL_HANDLE;
    }

    if (m_imguiDescriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(ctx.device, m_imguiDescriptorPool, nullptr);
        m_imguiDescriptorPool = VK_NULL_HANDLE;
    }

    m_initialized = false;
}

// --------------------------------------------------------------------------
// Frame lifecycle
// --------------------------------------------------------------------------

void EditorUI::beginFrame() {
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void EditorUI::endFrame(VkCommandBuffer cmd) {
    ImGui::Render();
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
}

void EditorUI::renderToSwapchain(VulkanContext& ctx, VkCommandBuffer cmd,
                                  VkImage /*swapImage*/, VkImageView /*swapView*/,
                                  VkFormat /*swapFormat*/, VkExtent2D extent) {
    // We rely on the framebuffers created during init, indexed by the current image
    // The caller should find the right index. For simplicity, we do a linear search.
    // Actually, the caller should pass the imageIndex directly. We'll use a simpler approach:
    // just record the render pass with the framebuffer matching the swapchain image view.

    // This method is called with a command buffer already in recording state.
    // We start the ImGui render pass.

    // Find the framebuffer for this swapchain image (caller should track imageIndex)
    // For now, assume this is handled externally.
    (void)extent; // unused in this path - we use the full framebuffer extent
}

void EditorUI::updateFPS() {
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

bool EditorUI::wantCaptureMouse() const {
    return ImGui::GetIO().WantCaptureMouse;
}

bool EditorUI::wantCaptureKeyboard() const {
    return ImGui::GetIO().WantCaptureKeyboard;
}

// --------------------------------------------------------------------------
// Pipeline scanning
// --------------------------------------------------------------------------

void EditorUI::scanPipelines(const std::string& directory) {
    // Accumulate into existing list (caller can clear first if needed)
    try {
        if (!fs::exists(directory)) {
            std::cout << "[editor] Pipeline directory not found: " << directory << std::endl;
            return;
        }

        // Collect existing bases to avoid duplicates
        std::set<std::string> existingBases(m_state.availablePipelines.begin(),
                                             m_state.availablePipelines.end());

        // Scan for .vert.spv or .frag.spv files to identify pipelines
        std::set<std::string> newBases;
        for (const auto& entry : fs::recursive_directory_iterator(directory)) {
            if (!entry.is_regular_file()) continue;
            std::string path = entry.path().string();
            // Normalize path separators
            std::replace(path.begin(), path.end(), '\\', '/');

            // Look for .vert.spv files as pipeline indicators
            std::string suffix = ".vert.spv";
            if (path.size() > suffix.size() && path.substr(path.size() - suffix.size()) == suffix) {
                std::string base = path.substr(0, path.size() - suffix.size());
                newBases.insert(base);
            }
            // Also check for .frag.spv (fullscreen shaders)
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
// Initialize editable state from scene data
// --------------------------------------------------------------------------

void EditorUI::initFromScene(SceneManager& scene) {
    if (m_sceneInitialized) return;

    if (scene.hasGltfScene()) {
        const auto& gltfScene = scene.getGltfScene();

        // Initialize node transforms
        m_state.nodeTransforms.resize(gltfScene.nodes.size());
        for (size_t i = 0; i < gltfScene.nodes.size(); i++) {
            const auto& node = gltfScene.nodes[i];
            // Decompose world transform to get position (translation column)
            const glm::mat4& m = node.worldTransform;
            m_state.nodeTransforms[i].position[0] = m[3][0];
            m_state.nodeTransforms[i].position[1] = m[3][1];
            m_state.nodeTransforms[i].position[2] = m[3][2];

            // Extract scale from column lengths
            float sx = glm::length(glm::vec3(m[0]));
            float sy = glm::length(glm::vec3(m[1]));
            float sz = glm::length(glm::vec3(m[2]));
            m_state.nodeTransforms[i].scale[0] = sx;
            m_state.nodeTransforms[i].scale[1] = sy;
            m_state.nodeTransforms[i].scale[2] = sz;

            // Rotation extraction is approximate (euler from rotation matrix)
            if (sx > 0.0001f && sy > 0.0001f && sz > 0.0001f) {
                glm::mat3 rot(
                    glm::vec3(m[0]) / sx,
                    glm::vec3(m[1]) / sy,
                    glm::vec3(m[2]) / sz
                );
                // Euler angles (pitch, yaw, roll) from rotation matrix
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

// --------------------------------------------------------------------------
// Panel drawing: main entry
// --------------------------------------------------------------------------

void EditorUI::drawPanels(SceneManager& scene) {
    initFromScene(scene);

    drawMenuBar();
    drawViewportControls();
    drawSceneTreePanel(scene);
    drawPropertiesPanel(scene);
    drawPipelineSelector();
}

// --------------------------------------------------------------------------
// Menu bar
// --------------------------------------------------------------------------

void EditorUI::drawMenuBar() {
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

void EditorUI::drawViewportControls() {
    ImGuiIO& io = ImGui::GetIO();
    float menuBarHeight = ImGui::GetFrameHeight(); // main menu bar height

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
// Scene tree panel (left side)
// --------------------------------------------------------------------------

void EditorUI::drawSceneTreePanel(SceneManager& scene) {
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

        // Scene info
        ImGui::Text("Source: %s", scene.getSceneSource().c_str());

        if (scene.hasGltfScene()) {
            const auto& gltfScene = scene.getGltfScene();
            ImGui::Text("Nodes: %zu  Meshes: %zu  Materials: %zu",
                        gltfScene.nodes.size(), gltfScene.meshes.size(), gltfScene.materials.size());

            ImGui::Separator();

            // Node tree
            if (ImGui::CollapsingHeader("Nodes", ImGuiTreeNodeFlags_DefaultOpen)) {
                for (int rootIdx : gltfScene.rootNodes) {
                    drawNodeTree(gltfScene, rootIdx);
                }
            }

            // Materials list
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

            // Lights list
            if (ImGui::CollapsingHeader("Lights")) {
                const auto& lights = scene.getLights();
                for (size_t i = 0; i < lights.size(); i++) {
                    const auto& light = lights[i];
                    const char* typeStr = "Unknown";
                    switch (light.type) {
                        case SceneLight::Directional: typeStr = "Directional"; break;
                        case SceneLight::Point: typeStr = "Point"; break;
                        case SceneLight::Spot: typeStr = "Spot"; break;
                    }
                    ImGui::BulletText("Light %zu: %s (%.1f)", i, typeStr, light.intensity);
                }
            }
        } else {
            ImGui::Text("(Procedural scene - no node hierarchy)");
            ImGui::Text("Vertices: %zu", scene.getVertices().size());
        }
    }
    ImGui::End();
}

void EditorUI::drawNodeTree(const GltfScene& gltfScene, int nodeIndex) {
    if (nodeIndex < 0 || nodeIndex >= static_cast<int>(gltfScene.nodes.size())) return;

    const auto& node = gltfScene.nodes[nodeIndex];
    std::string label = node.name.empty()
        ? "Node " + std::to_string(nodeIndex)
        : node.name;

    // Add mesh/camera/light indicators
    if (node.meshIndex >= 0) {
        label += " [Mesh]";
    }
    if (node.cameraIndex >= 0) {
        label += " [Camera]";
    }
    if (node.lightIndex >= 0) {
        label += " [Light]";
    }

    ImGuiTreeNodeFlags nodeFlags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth;
    if (node.children.empty()) {
        nodeFlags |= ImGuiTreeNodeFlags_Leaf;
    }
    if (m_state.selectedNodeIndex == nodeIndex) {
        nodeFlags |= ImGuiTreeNodeFlags_Selected;
    }

    bool opened = ImGui::TreeNodeEx(
        (label + "##" + std::to_string(nodeIndex)).c_str(), nodeFlags);

    if (ImGui::IsItemClicked()) {
        m_state.selectedNodeIndex = nodeIndex;
        m_state.selectedMaterialIndex = -1;
        // If the node has a mesh, also select its material
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
// Properties panel (right side)
// --------------------------------------------------------------------------

void EditorUI::drawPropertiesPanel(SceneManager& scene) {
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

            // Mesh info
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

        // Pipeline info
        ImGui::Separator();
        if (ImGui::CollapsingHeader("Active Pipeline")) {
            ImGui::Text("Base: %s", m_state.currentPipelineBase.c_str());
        }
    }
    ImGui::End();
}

// --------------------------------------------------------------------------
// Pipeline selector panel
// --------------------------------------------------------------------------

void EditorUI::drawPipelineSelector() {
    ImGuiIO& io = ImGui::GetIO();
    float leftPanelWidth = 280.0f;
    float rightPanelWidth = 300.0f;
    float panelWidth = io.DisplaySize.x - leftPanelWidth - rightPanelWidth;
    float panelHeight = 180.0f;
    float panelY = io.DisplaySize.y - panelHeight;

    // Bottom panel between left and right panels
    ImGui::SetNextWindowPos(ImVec2(leftPanelWidth, panelY));
    ImGui::SetNextWindowSize(ImVec2(panelWidth, panelHeight));
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse;

    if (ImGui::Begin("Pipeline", nullptr, flags)) {
        // Scan directory
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

        // Pipeline dropdown
        if (!m_state.availablePipelines.empty()) {
            ImGui::Text("Available pipelines:");

            // Build combo items
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
                    if (isSelected) {
                        ImGui::SetItemDefaultFocus();
                    }
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

        // Current pipeline info
        ImGui::Separator();
        ImGui::Text("Current: %s", m_state.currentPipelineBase.c_str());
    }
    ImGui::End();
}
