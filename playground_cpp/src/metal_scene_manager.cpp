#include "metal_scene_manager.h"
#include "metal_scene.h"
#include "metal_context.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <cstring>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <limits>
#include <algorithm>
#include <unordered_set>

namespace fs = std::filesystem;

// --------------------------------------------------------------------------
// Static helper
// --------------------------------------------------------------------------

bool MetalSceneManager::isGltfFile(const std::string& source) {
    if (source.size() > 4 && source.substr(source.size()-4) == ".glb") return true;
    if (source.size() > 5 && source.substr(source.size()-5) == ".gltf") return true;
    return false;
}

// --------------------------------------------------------------------------
// Load scene geometry (CPU-side)
// --------------------------------------------------------------------------

void MetalSceneManager::loadScene(const std::string& sceneSource) {
    m_sceneSource = sceneSource;

    if (isGltfFile(sceneSource)) {
        std::cout << "[metal] Loading glTF scene: " << sceneSource << std::endl;
        m_gltfScene = loadGltf(sceneSource);
        m_hasGltfScene = true;

        if (!m_gltfScene.meshes.empty()) {
            auto drawItems = flattenScene(m_gltfScene);

            uint32_t vertexOffset = 0;
            for (auto& item : drawItems) {
                auto& gmesh = m_gltfScene.meshes[item.meshIndex];
                glm::mat4 world = item.worldTransform;
                glm::mat3 normalMat = glm::transpose(glm::inverse(glm::mat3(world)));

                for (size_t i = 0; i < gmesh.vertices.size(); i++) {
                    Vertex v;
                    glm::vec4 pos4 = world * glm::vec4(gmesh.vertices[i].position, 1.0f);
                    v.position = glm::vec3(pos4);
                    v.normal = glm::normalize(normalMat * gmesh.vertices[i].normal);
                    v.uv = gmesh.vertices[i].uv;
                    m_vertices.push_back(v);
                }
                for (auto idx : gmesh.indices) {
                    m_indices.push_back(idx + vertexOffset);
                }
                vertexOffset += static_cast<uint32_t>(gmesh.vertices.size());
            }
            std::cout << "[metal] Loaded glTF: " << m_vertices.size()
                      << " vertices, " << m_indices.size() << " indices" << std::endl;
        } else {
            std::cout << "[metal] glTF has no meshes, falling back to sphere" << std::endl;
            Scene::generateSphere(32, 32, m_vertices, m_indices);
        }
    } else if (sceneSource == "sphere" || sceneSource == "triangle" ||
               sceneSource == "fullscreen") {
        if (sceneSource == "sphere") {
            Scene::generateSphere(32, 32, m_vertices, m_indices);
        }
    } else {
        Scene::generateSphere(32, 32, m_vertices, m_indices);
    }

    if (!m_vertices.empty()) {
        computeAutoCamera(m_vertices);
    }
}

// --------------------------------------------------------------------------
// Upload mesh to GPU
// --------------------------------------------------------------------------

void MetalSceneManager::uploadToGPU(MetalContext& ctx, int vertexStride) {
    if (m_vertices.empty() && m_indices.empty()) return;

    if (vertexStride == 48 && m_hasGltfScene) {
        auto drawItems = flattenScene(m_gltfScene);

        std::vector<float> vdata;
        std::vector<uint32_t> allIndices;
        uint32_t vertexOffset = 0;
        uint32_t indexOffset = 0;
        m_drawRanges.clear();

        for (auto& item : drawItems) {
            auto& gmesh = m_gltfScene.meshes[item.meshIndex];
            glm::mat4 world = item.worldTransform;
            glm::mat3 upper3x3 = glm::mat3(world);
            glm::mat3 normalMat = glm::transpose(glm::inverse(upper3x3));

            for (size_t i = 0; i < gmesh.vertices.size(); i++) {
                auto& gv = gmesh.vertices[i];
                glm::vec4 pos4 = world * glm::vec4(gv.position, 1.0f);
                glm::vec3 pos(pos4);
                glm::vec3 norm = glm::normalize(normalMat * gv.normal);

                vdata.push_back(pos.x); vdata.push_back(pos.y); vdata.push_back(pos.z);
                vdata.push_back(norm.x); vdata.push_back(norm.y); vdata.push_back(norm.z);
                vdata.push_back(gv.uv.x); vdata.push_back(gv.uv.y);

                if (gmesh.hasTangents) {
                    glm::vec3 tang = glm::normalize(upper3x3 * glm::vec3(gv.tangent));
                    vdata.push_back(tang.x); vdata.push_back(tang.y); vdata.push_back(tang.z);
                    vdata.push_back(gv.tangent.w);
                } else {
                    vdata.push_back(1.0f); vdata.push_back(0.0f);
                    vdata.push_back(0.0f); vdata.push_back(1.0f);
                }
            }

            for (auto idx : gmesh.indices) {
                allIndices.push_back(idx + vertexOffset);
            }
            m_drawRanges.push_back({indexOffset, static_cast<uint32_t>(gmesh.indices.size()), item.materialIndex});
            indexOffset += static_cast<uint32_t>(gmesh.indices.size());
            vertexOffset += static_cast<uint32_t>(gmesh.vertices.size());
        }

        m_mesh = MetalScene::uploadMesh(ctx, vdata.data(), vdata.size() * sizeof(float),
                                         allIndices.data(), allIndices.size(), 48);
        m_mesh.vertexCount = vertexOffset;

        std::cout << "[metal] Uploaded glTF mesh 48-byte stride ("
                  << m_mesh.vertexCount << " verts, " << m_mesh.indexCount << " indices)" << std::endl;
    } else if (vertexStride == 48 && !m_hasGltfScene) {
        std::vector<float> vdata;
        vdata.reserve(m_vertices.size() * 12);
        for (auto& v : m_vertices) {
            vdata.push_back(v.position.x); vdata.push_back(v.position.y); vdata.push_back(v.position.z);
            vdata.push_back(v.normal.x); vdata.push_back(v.normal.y); vdata.push_back(v.normal.z);
            vdata.push_back(v.uv.x); vdata.push_back(v.uv.y);
            vdata.push_back(1.0f); vdata.push_back(0.0f); vdata.push_back(0.0f); vdata.push_back(1.0f);
        }

        m_mesh = MetalScene::uploadMesh(ctx, vdata.data(), vdata.size() * sizeof(float),
                                         m_indices.data(), m_indices.size(), 48);
    } else {
        // Standard 32-byte Vertex upload
        if (!m_vertices.empty() && !m_indices.empty()) {
            m_mesh = MetalScene::uploadMesh(ctx, m_vertices.data(),
                                             m_vertices.size() * sizeof(Vertex),
                                             m_indices.data(), m_indices.size(), 32);
        }
    }
}

// --------------------------------------------------------------------------
// Upload textures
// --------------------------------------------------------------------------

void MetalSceneManager::uploadTextures(MetalContext& ctx) {
    createDefaultTextures(ctx);
    uploadGltfTextures(ctx);

    if (m_namedTextures.empty()) {
        auto texPixels = Scene::generateProceduralTexture(512);
        auto gpuTex = MetalScene::uploadTexture(ctx, texPixels.data(), 512, 512, 4, true);
        m_namedTextures["albedo_tex"] = gpuTex;
        m_namedTextures["base_color_tex"] = gpuTex;
    }
}

void MetalSceneManager::createDefaultTextures(MetalContext& ctx) {
    m_defaultWhiteTexture = MetalScene::createDefaultTexture(ctx, 255, 255, 255, 255);
    m_defaultBlackTexture = MetalScene::createDefaultTexture(ctx, 0, 0, 0, 255);
    m_defaultNormalTexture = MetalScene::createDefaultTexture(ctx, 128, 128, 255, 255);
    std::cout << "[metal] Created default textures (white, black, flat-normal)" << std::endl;
}

void MetalSceneManager::uploadGltfTextures(MetalContext& ctx) {
    if (!m_hasGltfScene || m_gltfScene.materials.empty()) return;

    m_perMaterialTextures.resize(m_gltfScene.materials.size());

    for (size_t i = 0; i < m_gltfScene.materials.size(); i++) {
        auto& mat = m_gltfScene.materials[i];
        auto& texMap = m_perMaterialTextures[i];

        auto tryUpload = [&](const GltfTextureData& texData, const std::string& name) {
            if (texData.valid()) {
                texMap[name] = MetalScene::uploadTexture(ctx, texData.pixels.data(),
                    static_cast<uint32_t>(texData.width),
                    static_cast<uint32_t>(texData.height), 4, true);
                std::cout << "[metal] Uploaded texture: " << name << "[" << i
                          << "] (" << texData.width << "x" << texData.height << ")" << std::endl;
            }
        };

        tryUpload(mat.base_color_tex, "base_color_tex");
        tryUpload(mat.normal_tex, "normal_tex");
        tryUpload(mat.metallic_roughness_tex, "metallic_roughness_tex");
        tryUpload(mat.occlusion_tex, "occlusion_tex");
        tryUpload(mat.emissive_tex, "emissive_tex");
        tryUpload(mat.clearcoat_tex, "clearcoat_tex");
        tryUpload(mat.clearcoat_roughness_tex, "clearcoat_roughness_tex");
        tryUpload(mat.sheen_color_tex, "sheen_color_tex");
        tryUpload(mat.transmission_tex, "transmission_tex");
    }

    // Backward compat: populate named textures from material 0
    if (!m_perMaterialTextures.empty()) {
        for (auto& [name, tex] : m_perMaterialTextures[0]) {
            m_namedTextures[name] = tex;
        }
    }
}

// --------------------------------------------------------------------------
// IBL assets
// --------------------------------------------------------------------------

void MetalSceneManager::loadIBLAssets(MetalContext& ctx, const std::string& requestedName) {
    // Look for IBL assets directory with manifest.json
    std::string iblDir;
    if (!requestedName.empty()) {
        std::string testPath = "assets/ibl/" + requestedName + "/manifest.json";
        if (fs::exists(testPath)) iblDir = "assets/ibl/" + requestedName;
    }
    if (iblDir.empty()) {
        std::string names[] = {"pisa", "neutral"};
        for (auto& name : names) {
            if (fs::exists("assets/ibl/" + name + "/manifest.json")) {
                iblDir = "assets/ibl/" + name;
                break;
            }
        }
    }
    if (iblDir.empty()) {
        std::cout << "[metal] No IBL assets found, skipping" << std::endl;
        return;
    }

    // Parse manifest.json for dimensions
    std::string manifestPath = iblDir + "/manifest.json";
    std::ifstream manifestFile(manifestPath);
    if (!manifestFile.is_open()) return;
    std::stringstream manifestSS;
    manifestSS << manifestFile.rdbuf();
    std::string manifestContent = manifestSS.str();

    // Simple JSON int extractor (same as Vulkan code)
    auto extractNestedInt = [&](const std::string& json, const std::string& obj, const std::string& key) -> int {
        size_t opos = json.find("\"" + obj + "\"");
        if (opos == std::string::npos) return -1;
        size_t kpos = json.find("\"" + key + "\"", opos);
        if (kpos == std::string::npos) return -1;
        kpos = json.find(":", kpos);
        if (kpos == std::string::npos) return -1;
        std::string sub = json.substr(kpos + 1, 20);
        // Skip whitespace
        size_t start = sub.find_first_of("0123456789");
        if (start == std::string::npos) return -1;
        return std::stoi(sub.substr(start));
    };

    int specSize = extractNestedInt(manifestContent, "specular", "face_size");
    int specMips = extractNestedInt(manifestContent, "specular", "mip_count");
    int irrSize = extractNestedInt(manifestContent, "irradiance", "face_size");
    int brdfSize = extractNestedInt(manifestContent, "brdf_lut", "size");

    if (specSize <= 0 || specMips <= 0 || irrSize <= 0 || brdfSize <= 0) {
        std::cout << "[metal] Invalid IBL manifest values, skipping" << std::endl;
        return;
    }

    std::cout << "[metal] Loading IBL assets (spec=" << specSize << "x" << specSize
              << " mips=" << specMips << ", irr=" << irrSize << "x" << irrSize
              << ", brdf=" << brdfSize << "x" << brdfSize << ")" << std::endl;

    // Load specular cubemap (raw F16 RGBA data, no header)
    {
        std::string specPath = iblDir + "/specular.bin";
        std::ifstream f(specPath, std::ios::binary | std::ios::ate);
        if (f.is_open()) {
            size_t fileSize = static_cast<size_t>(f.tellg());
            f.seekg(0);
            std::vector<uint16_t> data(fileSize / sizeof(uint16_t));
            f.read(reinterpret_cast<char*>(data.data()), fileSize);

            m_iblTextures["env_specular"] = MetalScene::uploadCubemapF16(
                ctx, static_cast<uint32_t>(specSize), static_cast<uint32_t>(specMips),
                data.data(), data.size());
            std::cout << "[metal] Loaded IBL specular cubemap" << std::endl;
        }
    }

    // Load irradiance cubemap (raw F16 RGBA data, no header, 1 mip)
    {
        std::string irrPath = iblDir + "/irradiance.bin";
        std::ifstream f(irrPath, std::ios::binary | std::ios::ate);
        if (f.is_open()) {
            size_t fileSize = static_cast<size_t>(f.tellg());
            f.seekg(0);
            std::vector<uint16_t> data(fileSize / sizeof(uint16_t));
            f.read(reinterpret_cast<char*>(data.data()), fileSize);

            m_iblTextures["env_irradiance"] = MetalScene::uploadCubemapF16(
                ctx, static_cast<uint32_t>(irrSize), 1, data.data(), data.size());
            std::cout << "[metal] Loaded IBL irradiance cubemap" << std::endl;
        }
    }

    // Load BRDF LUT (2D texture, RG16F raw data — pad to RGBA16F for Metal)
    {
        std::string brdfPath = iblDir + "/brdf_lut.bin";
        std::ifstream f(brdfPath, std::ios::binary | std::ios::ate);
        if (f.is_open()) {
            size_t fileSize = static_cast<size_t>(f.tellg());
            f.seekg(0);
            std::vector<uint16_t> rawData(fileSize / sizeof(uint16_t));
            f.read(reinterpret_cast<char*>(rawData.data()), fileSize);

            size_t totalPixels = static_cast<size_t>(brdfSize) * brdfSize;
            std::vector<uint16_t> rgbaData;

            // Pad RG16F to RGBA16F
            if (rawData.size() == totalPixels * 2) {
                rgbaData.resize(totalPixels * 4);
                for (size_t p = 0; p < totalPixels; p++) {
                    rgbaData[p * 4 + 0] = rawData[p * 2 + 0];
                    rgbaData[p * 4 + 1] = rawData[p * 2 + 1];
                    rgbaData[p * 4 + 2] = 0;
                    rgbaData[p * 4 + 3] = 0x3C00; // 1.0 in half-float
                }
            } else {
                rgbaData = rawData;
            }

            auto* desc = MTL::TextureDescriptor::texture2DDescriptor(
                MTL::PixelFormatRGBA16Float, static_cast<uint32_t>(brdfSize),
                static_cast<uint32_t>(brdfSize), false);
            desc->setUsage(MTL::TextureUsageShaderRead);
            desc->setStorageMode(MTL::StorageModePrivate);

            auto* tex = ctx.newTexture(desc);

            size_t bytesPerRow = brdfSize * 4 * sizeof(uint16_t); // RGBA16F = 8 bytes/pixel
            auto* staging = ctx.newBuffer(rgbaData.data(), rgbaData.size() * sizeof(uint16_t),
                                           MTL::ResourceStorageModeShared);

            auto* cmdBuf = ctx.beginCommandBuffer();
            auto* blit = cmdBuf->blitCommandEncoder();
            blit->copyFromBuffer(staging, 0, bytesPerRow, 0,
                                 MTL::Size(static_cast<uint32_t>(brdfSize),
                                           static_cast<uint32_t>(brdfSize), 1),
                                 tex, 0, 0, MTL::Origin(0, 0, 0));
            blit->endEncoding();
            ctx.submitAndWait(cmdBuf);
            staging->release();

            MetalGPUTexture brdfTex;
            brdfTex.texture = tex;
            brdfTex.sampler = MetalScene::createSampler(ctx, MTL::SamplerMinMagFilterLinear,
                                                         MTL::SamplerAddressModeClampToEdge, false);
            brdfTex.width = static_cast<uint32_t>(brdfSize);
            brdfTex.height = static_cast<uint32_t>(brdfSize);
            m_iblTextures["brdf_lut"] = brdfTex;

            std::cout << "[metal] Loaded IBL BRDF LUT" << std::endl;
        }
    }

    std::cout << "[metal] IBL loading complete: " << m_iblTextures.size() << " texture(s)" << std::endl;
}

// --------------------------------------------------------------------------
// Texture lookup
// --------------------------------------------------------------------------

MetalGPUTexture& MetalSceneManager::getTextureForBinding(const std::string& name, int materialIndex) {
    // Check per-material textures first
    if (materialIndex >= 0 && materialIndex < static_cast<int>(m_perMaterialTextures.size())) {
        auto& texMap = m_perMaterialTextures[materialIndex];
        auto it = texMap.find(name);
        if (it != texMap.end()) return it->second;
    }

    // Check named textures
    auto it = m_namedTextures.find(name);
    if (it != m_namedTextures.end()) return it->second;

    // Check IBL textures
    it = m_iblTextures.find(name);
    if (it != m_iblTextures.end()) return it->second;

    // Return default textures based on name
    if (name.find("normal") != std::string::npos) return m_defaultNormalTexture;
    if (name.find("occlusion") != std::string::npos) return m_defaultWhiteTexture;
    if (name.find("emissive") != std::string::npos) return m_defaultBlackTexture;
    return m_defaultWhiteTexture;
}

// --------------------------------------------------------------------------
// Scene feature detection
// --------------------------------------------------------------------------

std::set<std::string> MetalSceneManager::detectSceneFeatures() const {
    std::set<std::string> features;
    if (!m_hasGltfScene) return features;

    for (auto& mat : m_gltfScene.materials) {
        if (mat.normal_tex.valid()) features.insert("has_normal_map");
        if (mat.emissive_tex.valid() || glm::length(mat.emissive) > 0.0f) features.insert("has_emission");
        if (mat.hasClearcoat) features.insert("has_clearcoat");
        if (mat.hasSheen) features.insert("has_sheen");
        if (mat.hasTransmission) features.insert("has_transmission");
    }
    return features;
}

std::set<std::string> MetalSceneManager::detectMaterialFeatures(int materialIndex) const {
    std::set<std::string> features;
    if (!m_hasGltfScene || materialIndex < 0 ||
        materialIndex >= static_cast<int>(m_gltfScene.materials.size())) return features;

    auto& mat = m_gltfScene.materials[materialIndex];
    if (mat.normal_tex.valid()) features.insert("has_normal_map");
    if (mat.emissive_tex.valid() || glm::length(mat.emissive) > 0.0f) features.insert("has_emission");
    if (mat.hasClearcoat) features.insert("has_clearcoat");
    if (mat.hasSheen) features.insert("has_sheen");
    if (mat.hasTransmission) features.insert("has_transmission");
    return features;
}

std::map<std::string, std::vector<int>> MetalSceneManager::groupMaterialsByFeatures() const {
    std::map<std::string, std::vector<int>> groups;
    if (!m_hasGltfScene) return groups;

    for (size_t i = 0; i < m_gltfScene.materials.size(); i++) {
        auto features = detectMaterialFeatures(static_cast<int>(i));
        std::string suffix;
        for (auto& f : features) suffix += "+" + f;
        groups[suffix].push_back(static_cast<int>(i));
    }
    return groups;
}

// --------------------------------------------------------------------------
// Auto-camera
// --------------------------------------------------------------------------

void MetalSceneManager::computeAutoCamera(const std::vector<Vertex>& vertices) {
    glm::vec3 bmin(std::numeric_limits<float>::max());
    glm::vec3 bmax(std::numeric_limits<float>::lowest());

    for (auto& v : vertices) {
        bmin = glm::min(bmin, v.position);
        bmax = glm::max(bmax, v.position);
    }

    glm::vec3 center = (bmin + bmax) * 0.5f;
    float radius = glm::length(bmax - bmin) * 0.5f;
    if (radius < 0.001f) radius = 1.0f;

    m_autoTarget = center;
    m_autoEye = center + glm::vec3(0.0f, radius * 0.3f, radius * 2.0f);
    m_autoUp = glm::vec3(0.0f, 1.0f, 0.0f);
    m_autoFar = radius * 10.0f;
    m_hasSceneBounds = true;

    std::cout << "[metal] Auto-camera: center=(" << center.x << "," << center.y << "," << center.z
              << "), radius=" << radius << std::endl;
}

// --------------------------------------------------------------------------
// Cleanup
// --------------------------------------------------------------------------

void MetalSceneManager::cleanup() {
    MetalScene::destroyMesh(m_mesh);

    // Track released texture pointers to avoid double-free.
    // m_namedTextures may alias m_perMaterialTextures[0] entries,
    // and procedural textures may be stored under multiple keys.
    std::unordered_set<MTL::Texture*> released;

    auto safeDestroy = [&](MetalGPUTexture& tex) {
        if (tex.texture && released.find(tex.texture) == released.end()) {
            released.insert(tex.texture);
            MetalScene::destroyTexture(tex);
        } else {
            // Already released — just null out pointers
            tex.texture = nullptr;
            tex.sampler = nullptr;
        }
    };

    // Release per-material textures first (they are the canonical owners)
    for (auto& texMap : m_perMaterialTextures) {
        for (auto& [_, tex] : texMap) safeDestroy(tex);
    }
    m_perMaterialTextures.clear();

    // Release named textures (may alias per-material textures)
    for (auto& [_, tex] : m_namedTextures) safeDestroy(tex);
    m_namedTextures.clear();

    // Release IBL textures
    for (auto& [_, tex] : m_iblTextures) safeDestroy(tex);
    m_iblTextures.clear();

    // Release default textures
    safeDestroy(m_defaultWhiteTexture);
    safeDestroy(m_defaultBlackTexture);
    safeDestroy(m_defaultNormalTexture);
}
