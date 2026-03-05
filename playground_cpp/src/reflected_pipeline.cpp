#include "reflected_pipeline.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <map>
#include <filesystem>
#include <set>

// ===========================================================================
// Minimal JSON parser (no external dependency required)
// Parses the subset of JSON that .lux.json files produce.
// ===========================================================================

namespace {

struct JsonValue;
using JsonObject = std::unordered_map<std::string, JsonValue>;
using JsonArray = std::vector<JsonValue>;

struct JsonValue {
    enum Type { Null, Number, String, Bool, Object, Array } type = Null;
    double number = 0;
    std::string str;
    bool boolean = false;
    JsonObject object;
    JsonArray array;

    int asInt() const { return static_cast<int>(number); }
    const std::string& asString() const { return str; }
};

class JsonParser {
public:
    explicit JsonParser(const std::string& input) : src(input), pos(0) {}

    JsonValue parse() {
        skipWhitespace();
        return parseValue();
    }

private:
    const std::string& src;
    size_t pos;

    void skipWhitespace() {
        while (pos < src.size() && (src[pos] == ' ' || src[pos] == '\n' || src[pos] == '\r' || src[pos] == '\t'))
            pos++;
    }

    char peek() { return pos < src.size() ? src[pos] : '\0'; }
    char advance() { return src[pos++]; }

    JsonValue parseValue() {
        skipWhitespace();
        char c = peek();
        if (c == '"') return parseString();
        if (c == '{') return parseObject();
        if (c == '[') return parseArray();
        if (c == 't' || c == 'f') return parseBool();
        if (c == 'n') return parseNull();
        if (c == '-' || (c >= '0' && c <= '9')) return parseNumber();
        throw std::runtime_error(std::string("Unexpected char: ") + c);
    }

    JsonValue parseString() {
        advance(); // skip opening "
        std::string result;
        while (peek() != '"') {
            if (peek() == '\\') {
                advance();
                char esc = advance();
                switch (esc) {
                    case '"': result += '"'; break;
                    case '\\': result += '\\'; break;
                    case '/': result += '/'; break;
                    case 'n': result += '\n'; break;
                    case 't': result += '\t'; break;
                    case 'r': result += '\r'; break;
                    default: result += esc; break;
                }
            } else {
                result += advance();
            }
        }
        advance(); // skip closing "
        JsonValue v;
        v.type = JsonValue::String;
        v.str = result;
        return v;
    }

    JsonValue parseNumber() {
        size_t start = pos;
        if (peek() == '-') advance();
        while (pos < src.size() && ((src[pos] >= '0' && src[pos] <= '9') || src[pos] == '.' || src[pos] == 'e' || src[pos] == 'E' || src[pos] == '+' || src[pos] == '-'))
            advance();
        JsonValue v;
        v.type = JsonValue::Number;
        v.number = std::stod(src.substr(start, pos - start));
        return v;
    }

    JsonValue parseBool() {
        JsonValue v;
        v.type = JsonValue::Bool;
        if (src.substr(pos, 4) == "true") { pos += 4; v.boolean = true; }
        else { pos += 5; v.boolean = false; }
        return v;
    }

    JsonValue parseNull() {
        pos += 4;
        return JsonValue{};
    }

    JsonValue parseObject() {
        advance(); // skip {
        JsonValue v;
        v.type = JsonValue::Object;
        skipWhitespace();
        if (peek() == '}') { advance(); return v; }
        while (true) {
            skipWhitespace();
            auto key = parseString();
            skipWhitespace();
            advance(); // skip :
            skipWhitespace();
            v.object[key.str] = parseValue();
            skipWhitespace();
            if (peek() == ',') { advance(); continue; }
            if (peek() == '}') { advance(); break; }
        }
        return v;
    }

    JsonValue parseArray() {
        advance(); // skip [
        JsonValue v;
        v.type = JsonValue::Array;
        skipWhitespace();
        if (peek() == ']') { advance(); return v; }
        while (true) {
            skipWhitespace();
            v.array.push_back(parseValue());
            skipWhitespace();
            if (peek() == ',') { advance(); continue; }
            if (peek() == ']') { advance(); break; }
        }
        return v;
    }
};

ReflectionData jsonToReflection(const JsonValue& root) {
    ReflectionData data;
    if (root.type != JsonValue::Object) return data;

    auto& obj = root.object;
    auto it = obj.find("version");
    if (it != obj.end()) data.version = it->second.asInt();

    it = obj.find("source");
    if (it != obj.end()) data.source = it->second.asString();

    it = obj.find("stage");
    if (it != obj.end()) data.stage = it->second.asString();

    it = obj.find("execution_model");
    if (it != obj.end()) data.execution_model = it->second.asString();

    // Inputs
    it = obj.find("inputs");
    if (it != obj.end() && it->second.type == JsonValue::Array) {
        for (auto& inp : it->second.array) {
            ReflectionData::VarInfo vi;
            auto ni = inp.object.find("name");
            if (ni != inp.object.end()) vi.name = ni->second.asString();
            auto ti = inp.object.find("type");
            if (ti != inp.object.end()) vi.type = ti->second.asString();
            auto li = inp.object.find("location");
            if (li != inp.object.end()) vi.location = li->second.asInt();
            data.inputs.push_back(vi);
        }
    }

    // Outputs
    it = obj.find("outputs");
    if (it != obj.end() && it->second.type == JsonValue::Array) {
        for (auto& out : it->second.array) {
            ReflectionData::VarInfo vi;
            auto ni = out.object.find("name");
            if (ni != out.object.end()) vi.name = ni->second.asString();
            auto ti = out.object.find("type");
            if (ti != out.object.end()) vi.type = ti->second.asString();
            auto li = out.object.find("location");
            if (li != out.object.end()) vi.location = li->second.asInt();
            data.outputs.push_back(vi);
        }
    }

    // Descriptor sets
    it = obj.find("descriptor_sets");
    if (it != obj.end() && it->second.type == JsonValue::Object) {
        for (auto& [setKey, bindings] : it->second.object) {
            int setIdx = std::stoi(setKey);
            if (bindings.type != JsonValue::Array) continue;
            for (auto& b : bindings.array) {
                ReflectionData::BindingInfo bi;
                auto bi_it = b.object.find("binding");
                if (bi_it != b.object.end()) bi.binding = bi_it->second.asInt();
                bi_it = b.object.find("type");
                if (bi_it != b.object.end()) bi.type = bi_it->second.asString();
                bi_it = b.object.find("name");
                if (bi_it != b.object.end()) bi.name = bi_it->second.asString();
                bi_it = b.object.find("size");
                if (bi_it != b.object.end()) bi.size = bi_it->second.asInt();
                bi_it = b.object.find("max_count");
                if (bi_it != b.object.end()) bi.max_count = bi_it->second.asInt();

                // Fields
                bi_it = b.object.find("fields");
                if (bi_it != b.object.end() && bi_it->second.type == JsonValue::Array) {
                    for (auto& f : bi_it->second.array) {
                        ReflectionData::FieldInfo fi;
                        auto fi_it = f.object.find("name");
                        if (fi_it != f.object.end()) fi.name = fi_it->second.asString();
                        fi_it = f.object.find("type");
                        if (fi_it != f.object.end()) fi.type = fi_it->second.asString();
                        fi_it = f.object.find("offset");
                        if (fi_it != f.object.end()) fi.offset = fi_it->second.asInt();
                        fi_it = f.object.find("size");
                        if (fi_it != f.object.end()) fi.size = fi_it->second.asInt();
                        bi.fields.push_back(fi);
                    }
                }

                // Stage flags
                bi_it = b.object.find("stage_flags");
                if (bi_it != b.object.end() && bi_it->second.type == JsonValue::Array) {
                    for (auto& sf : bi_it->second.array) {
                        bi.stage_flags.push_back(sf.asString());
                    }
                }

                data.descriptor_sets[setIdx].push_back(bi);
            }
        }
    }

    // Push constants
    it = obj.find("push_constants");
    if (it != obj.end() && it->second.type == JsonValue::Array) {
        for (auto& pc : it->second.array) {
            ReflectionData::PushConstantInfo pci;
            auto pi = pc.object.find("name");
            if (pi != pc.object.end()) pci.name = pi->second.asString();
            pi = pc.object.find("size");
            if (pi != pc.object.end()) pci.size = pi->second.asInt();

            pi = pc.object.find("fields");
            if (pi != pc.object.end() && pi->second.type == JsonValue::Array) {
                for (auto& f : pi->second.array) {
                    ReflectionData::FieldInfo fi;
                    auto fi_it = f.object.find("name");
                    if (fi_it != f.object.end()) fi.name = fi_it->second.asString();
                    fi_it = f.object.find("type");
                    if (fi_it != f.object.end()) fi.type = fi_it->second.asString();
                    fi_it = f.object.find("offset");
                    if (fi_it != f.object.end()) fi.offset = fi_it->second.asInt();
                    fi_it = f.object.find("size");
                    if (fi_it != f.object.end()) fi.size = fi_it->second.asInt();
                    pci.fields.push_back(fi);
                }
            }

            pi = pc.object.find("stage_flags");
            if (pi != pc.object.end() && pi->second.type == JsonValue::Array) {
                for (auto& sf : pi->second.array) {
                    pci.stage_flags.push_back(sf.asString());
                }
            }

            data.push_constants.push_back(pci);
        }
    }

    // Vertex attributes
    it = obj.find("vertex_attributes");
    if (it != obj.end() && it->second.type == JsonValue::Array) {
        for (auto& attr : it->second.array) {
            ReflectionData::VertexAttribute va;
            auto ai = attr.object.find("location");
            if (ai != attr.object.end()) va.location = ai->second.asInt();
            ai = attr.object.find("type");
            if (ai != attr.object.end()) va.type = ai->second.asString();
            ai = attr.object.find("name");
            if (ai != attr.object.end()) va.name = ai->second.asString();
            ai = attr.object.find("format");
            if (ai != attr.object.end()) va.format = ai->second.asString();
            ai = attr.object.find("offset");
            if (ai != attr.object.end()) va.offset = ai->second.asInt();
            data.vertex_attributes.push_back(va);
        }
    }

    it = obj.find("vertex_stride");
    if (it != obj.end()) data.vertex_stride = it->second.asInt();

    // Parse bindless section if present
    it = obj.find("bindless");
    if (it != obj.end() && it->second.type == JsonValue::Object) {
        auto& bless = it->second.object;
        auto enabledIt = bless.find("enabled");
        if (enabledIt != bless.end() && enabledIt->second.type == JsonValue::Bool) {
            data.bindlessEnabled = enabledIt->second.boolean;
        }

        // Parse dynamic struct layout for extended bindless structs
        auto sizeIt = bless.find("struct_size");
        if (sizeIt != bless.end()) {
            data.bindlessStructSize = static_cast<uint32_t>(sizeIt->second.asInt());
        }

        auto fieldsIt = bless.find("struct_fields");
        if (fieldsIt != bless.end() && fieldsIt->second.type == JsonValue::Array) {
            data.bindlessHasCustomProperties = true;
            for (auto& fieldVal : fieldsIt->second.array) {
                if (fieldVal.type != JsonValue::Object) continue;
                ReflectionData::BindlessFieldMeta meta;
                auto fi = fieldVal.object.find("name");
                if (fi != fieldVal.object.end()) meta.name = fi->second.asString();
                fi = fieldVal.object.find("type");
                if (fi != fieldVal.object.end()) meta.type = fi->second.asString();
                fi = fieldVal.object.find("offset");
                if (fi != fieldVal.object.end()) meta.offset = static_cast<uint32_t>(fi->second.asInt());
                fi = fieldVal.object.find("size");
                if (fi != fieldVal.object.end()) meta.size = static_cast<uint32_t>(fi->second.asInt());
                data.bindlessStructFields.push_back(meta);
            }
        }
    }

    // Parse mesh_output metadata if present
    it = obj.find("mesh_output");
    if (it != obj.end() && it->second.type == JsonValue::Object) {
        ReflectionData::MeshOutputInfo info;
        auto& mo = it->second.object;
        auto moIt = mo.find("max_vertices");
        if (moIt != mo.end()) info.maxVertices = static_cast<uint32_t>(moIt->second.asInt());
        moIt = mo.find("max_primitives");
        if (moIt != mo.end()) info.maxPrimitives = static_cast<uint32_t>(moIt->second.asInt());
        moIt = mo.find("workgroup_size");
        if (moIt != mo.end() && moIt->second.type == JsonValue::Array && !moIt->second.array.empty()) {
            info.workgroupSize = static_cast<uint32_t>(moIt->second.array[0].asInt());
        }
        moIt = mo.find("output_topology");
        if (moIt != mo.end()) info.outputTopology = moIt->second.asString();
        data.meshOutput = info;
    }

    return data;
}

} // anonymous namespace

// ===========================================================================
// Public API
// ===========================================================================

ReflectionData parseReflectionJson(const std::string& json_path) {
    std::ifstream file(json_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open reflection JSON: " + json_path);
    }
    std::stringstream ss;
    ss << file.rdbuf();
    std::string content = ss.str();

    JsonParser parser(content);
    JsonValue root = parser.parse();
    return jsonToReflection(root);
}

#ifndef LUX_METAL_BACKEND

VkFormat reflectionFormatToVkFormat(const std::string& fmt) {
    if (fmt == "R32_SFLOAT") return VK_FORMAT_R32_SFLOAT;
    if (fmt == "R32G32_SFLOAT") return VK_FORMAT_R32G32_SFLOAT;
    if (fmt == "R32G32B32_SFLOAT") return VK_FORMAT_R32G32B32_SFLOAT;
    if (fmt == "R32G32B32A32_SFLOAT") return VK_FORMAT_R32G32B32A32_SFLOAT;
    if (fmt == "R32_SINT") return VK_FORMAT_R32_SINT;
    if (fmt == "R32G32_SINT") return VK_FORMAT_R32G32_SINT;
    if (fmt == "R32G32B32_SINT") return VK_FORMAT_R32G32B32_SINT;
    if (fmt == "R32G32B32A32_SINT") return VK_FORMAT_R32G32B32A32_SINT;
    if (fmt == "R32_UINT") return VK_FORMAT_R32_UINT;
    if (fmt == "R32G32_UINT") return VK_FORMAT_R32G32_UINT;
    if (fmt == "R32G32B32_UINT") return VK_FORMAT_R32G32B32_UINT;
    if (fmt == "R32G32B32A32_UINT") return VK_FORMAT_R32G32B32A32_UINT;
    return VK_FORMAT_R32G32B32A32_SFLOAT; // fallback
}

VkDescriptorType bindingTypeToVkDescriptorType(const std::string& type) {
    if (type == "uniform_buffer") return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    if (type == "sampler") return VK_DESCRIPTOR_TYPE_SAMPLER;
    if (type == "sampled_image") return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    if (type == "sampled_cube_image") return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    if (type == "storage_image") return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    if (type == "storage_buffer") return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    if (type == "acceleration_structure") return VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    if (type == "bindless_combined_image_sampler_array") return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
}

static VkShaderStageFlags stageFlagsFromStrings(const std::vector<std::string>& flags) {
    VkShaderStageFlags result = 0;
    for (const auto& f : flags) {
        if (f == "vertex") result |= VK_SHADER_STAGE_VERTEX_BIT;
        else if (f == "fragment") result |= VK_SHADER_STAGE_FRAGMENT_BIT;
        else if (f == "raygen") result |= VK_SHADER_STAGE_RAYGEN_BIT_KHR;
        else if (f == "closest_hit") result |= VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
        else if (f == "miss") result |= VK_SHADER_STAGE_MISS_BIT_KHR;
        else if (f == "any_hit") result |= VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
        else if (f == "intersection") result |= VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
        else if (f == "callable") result |= VK_SHADER_STAGE_CALLABLE_BIT_KHR;
        else if (f == "mesh") result |= VK_SHADER_STAGE_MESH_BIT_EXT;
        else if (f == "task") result |= VK_SHADER_STAGE_TASK_BIT_EXT;
    }
    if (result == 0) {
        // Safe fallback: make binding visible to all shader stages
        result = VK_SHADER_STAGE_ALL;
    }
    return result;
}

// Internal helper struct to track bindless info per (set, binding) during merging
struct BindingMergeInfo {
    VkDescriptorSetLayoutBinding vkBinding = {};
    bool isBindless = false;
    int maxCount = 0;
};

std::unordered_map<int, VkDescriptorSetLayout> createDescriptorSetLayouts(
    VkDevice device,
    const ReflectionData& vertReflection,
    const ReflectionData& fragReflection
) {
    // Merge descriptor sets from both stages
    std::unordered_map<int, std::unordered_map<int, BindingMergeInfo>> merged;

    auto mergeStage = [&](const ReflectionData& refl) {
        for (auto& [setIdx, bindings] : refl.descriptor_sets) {
            for (auto& b : bindings) {
                auto& existing = merged[setIdx][b.binding];
                if (existing.vkBinding.descriptorCount == 0) {
                    existing.vkBinding.binding = static_cast<uint32_t>(b.binding);
                    existing.vkBinding.descriptorType = bindingTypeToVkDescriptorType(b.type);
                    existing.vkBinding.stageFlags = stageFlagsFromStrings(b.stage_flags);
                    existing.vkBinding.pImmutableSamplers = nullptr;
                    if (b.type == "bindless_combined_image_sampler_array") {
                        existing.isBindless = true;
                        existing.maxCount = b.max_count > 0 ? b.max_count : 1024;
                        existing.vkBinding.descriptorCount = static_cast<uint32_t>(existing.maxCount);
                    } else {
                        existing.vkBinding.descriptorCount = 1;
                    }
                } else {
                    existing.vkBinding.stageFlags |= stageFlagsFromStrings(b.stage_flags);
                }
            }
        }
    };

    mergeStage(vertReflection);
    mergeStage(fragReflection);

    std::unordered_map<int, VkDescriptorSetLayout> layouts;
    for (auto& [setIdx, bindingMap] : merged) {
        std::vector<VkDescriptorSetLayoutBinding> bindings;
        std::vector<int> bindingOrder;  // track original binding indices for flag arrays
        bool hasBindless = false;

        for (auto& [bIdx, info] : bindingMap) {
            bindings.push_back(info.vkBinding);
            bindingOrder.push_back(bIdx);
            if (info.isBindless) hasBindless = true;
        }
        // Sort by binding number
        std::vector<size_t> sortIndices(bindings.size());
        for (size_t i = 0; i < sortIndices.size(); i++) sortIndices[i] = i;
        std::sort(sortIndices.begin(), sortIndices.end(),
                  [&](size_t a, size_t b) { return bindings[a].binding < bindings[b].binding; });
        std::vector<VkDescriptorSetLayoutBinding> sortedBindings(bindings.size());
        std::vector<int> sortedOrder(bindings.size());
        for (size_t i = 0; i < sortIndices.size(); i++) {
            sortedBindings[i] = bindings[sortIndices[i]];
            sortedOrder[i] = bindingOrder[sortIndices[i]];
        }

        VkDescriptorSetLayoutCreateInfo layoutInfo = {};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(sortedBindings.size());
        layoutInfo.pBindings = sortedBindings.data();

        // Set up bindless flags if needed
        std::vector<VkDescriptorBindingFlags> bindingFlags;
        VkDescriptorSetLayoutBindingFlagsCreateInfo flagsInfo = {};

        if (hasBindless) {
            layoutInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;

            bindingFlags.resize(sortedBindings.size(), 0);
            for (size_t i = 0; i < sortedBindings.size(); i++) {
                int bIdx = sortedOrder[i];
                auto& info = bindingMap[bIdx];
                if (info.isBindless) {
                    bindingFlags[i] = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
                                      VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
                                      VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT;
                }
            }

            flagsInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
            flagsInfo.bindingCount = static_cast<uint32_t>(bindingFlags.size());
            flagsInfo.pBindingFlags = bindingFlags.data();
            layoutInfo.pNext = &flagsInfo;
        }

        VkDescriptorSetLayout layout;
        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &layout) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create descriptor set layout from reflection");
        }
        layouts[setIdx] = layout;
    }

    return layouts;
}

std::unordered_map<int, VkDescriptorSetLayout> createDescriptorSetLayoutsMultiStage(
    VkDevice device,
    const std::vector<ReflectionData>& stages
) {
    // Merge descriptor sets from all stages
    std::unordered_map<int, std::unordered_map<int, BindingMergeInfo>> merged;

    for (auto& refl : stages) {
        for (auto& [setIdx, bindings] : refl.descriptor_sets) {
            for (auto& b : bindings) {
                auto& existing = merged[setIdx][b.binding];
                if (existing.vkBinding.descriptorCount == 0) {
                    existing.vkBinding.binding = static_cast<uint32_t>(b.binding);
                    existing.vkBinding.descriptorType = bindingTypeToVkDescriptorType(b.type);
                    existing.vkBinding.stageFlags = stageFlagsFromStrings(b.stage_flags);
                    existing.vkBinding.pImmutableSamplers = nullptr;
                    if (b.type == "bindless_combined_image_sampler_array") {
                        existing.isBindless = true;
                        existing.maxCount = b.max_count > 0 ? b.max_count : 1024;
                        existing.vkBinding.descriptorCount = static_cast<uint32_t>(existing.maxCount);
                    } else {
                        existing.vkBinding.descriptorCount = 1;
                    }
                } else {
                    existing.vkBinding.stageFlags |= stageFlagsFromStrings(b.stage_flags);
                }
            }
        }
    }

    std::unordered_map<int, VkDescriptorSetLayout> layouts;
    for (auto& [setIdx, bindingMap] : merged) {
        std::vector<VkDescriptorSetLayoutBinding> bindings;
        std::vector<int> bindingOrder;
        bool hasBindless = false;

        for (auto& [bIdx, info] : bindingMap) {
            bindings.push_back(info.vkBinding);
            bindingOrder.push_back(bIdx);
            if (info.isBindless) hasBindless = true;
        }
        // Sort by binding number
        std::vector<size_t> sortIndices(bindings.size());
        for (size_t i = 0; i < sortIndices.size(); i++) sortIndices[i] = i;
        std::sort(sortIndices.begin(), sortIndices.end(),
                  [&](size_t a, size_t b) { return bindings[a].binding < bindings[b].binding; });
        std::vector<VkDescriptorSetLayoutBinding> sortedBindings(bindings.size());
        std::vector<int> sortedOrder(bindings.size());
        for (size_t i = 0; i < sortIndices.size(); i++) {
            sortedBindings[i] = bindings[sortIndices[i]];
            sortedOrder[i] = bindingOrder[sortIndices[i]];
        }

        VkDescriptorSetLayoutCreateInfo layoutInfo = {};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(sortedBindings.size());
        layoutInfo.pBindings = sortedBindings.data();

        // Set up bindless flags if needed
        std::vector<VkDescriptorBindingFlags> bindingFlags;
        VkDescriptorSetLayoutBindingFlagsCreateInfo flagsInfo = {};

        if (hasBindless) {
            layoutInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;

            bindingFlags.resize(sortedBindings.size(), 0);
            for (size_t i = 0; i < sortedBindings.size(); i++) {
                int bIdx = sortedOrder[i];
                auto& info = bindingMap[bIdx];
                if (info.isBindless) {
                    bindingFlags[i] = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
                                      VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
                                      VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT;
                }
            }

            flagsInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
            flagsInfo.bindingCount = static_cast<uint32_t>(bindingFlags.size());
            flagsInfo.pBindingFlags = bindingFlags.data();
            layoutInfo.pNext = &flagsInfo;
        }

        VkDescriptorSetLayout layout;
        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &layout) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create descriptor set layout from reflection (multi-stage)");
        }
        layouts[setIdx] = layout;
    }

    return layouts;
}

VkPipelineLayout createReflectedPipelineLayoutMultiStage(
    VkDevice device,
    const std::unordered_map<int, VkDescriptorSetLayout>& setLayouts,
    const std::vector<ReflectionData>& stages
) {
    // Collect layouts in order
    int maxSet = -1;
    for (auto& [idx, _] : setLayouts) {
        maxSet = std::max(maxSet, idx);
    }

    std::vector<VkDescriptorSetLayout> orderedLayouts;
    for (int i = 0; i <= maxSet; i++) {
        auto it = setLayouts.find(i);
        if (it != setLayouts.end()) {
            orderedLayouts.push_back(it->second);
        }
    }

    // Collect push constant ranges from all stages
    std::vector<VkPushConstantRange> pushRanges;
    for (auto& refl : stages) {
        for (auto& pc : refl.push_constants) {
            VkPushConstantRange range = {};
            range.stageFlags = stageFlagsFromStrings(pc.stage_flags);
            range.offset = 0;
            range.size = static_cast<uint32_t>(pc.size);
            pushRanges.push_back(range);
        }
    }

    VkPipelineLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = static_cast<uint32_t>(orderedLayouts.size());
    layoutInfo.pSetLayouts = orderedLayouts.data();
    layoutInfo.pushConstantRangeCount = static_cast<uint32_t>(pushRanges.size());
    layoutInfo.pPushConstantRanges = pushRanges.data();

    VkPipelineLayout pipelineLayout;
    if (vkCreatePipelineLayout(device, &layoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create pipeline layout from reflection (multi-stage)");
    }

    return pipelineLayout;
}

std::vector<MergedBindingInfo> getMergedBindings(
    const std::vector<ReflectionData>& stages
) {
    // Use a map keyed by (set, binding) to merge
    std::map<std::pair<int,int>, MergedBindingInfo> merged;

    for (auto& refl : stages) {
        for (auto& [setIdx, bindings] : refl.descriptor_sets) {
            for (auto& b : bindings) {
                auto key = std::make_pair(setIdx, b.binding);
                auto it = merged.find(key);
                if (it == merged.end()) {
                    MergedBindingInfo info;
                    info.set = setIdx;
                    info.binding = b.binding;
                    info.type = b.type;
                    info.name = b.name;
                    info.size = b.size;
                    info.max_count = b.max_count;
                    info.stageFlags = stageFlagsFromStrings(b.stage_flags);
                    merged[key] = info;
                } else {
                    it->second.stageFlags |= stageFlagsFromStrings(b.stage_flags);
                    if (b.max_count > 0) it->second.max_count = b.max_count;
                }
            }
        }
    }

    std::vector<MergedBindingInfo> result;
    for (auto& [_, info] : merged) {
        result.push_back(info);
    }
    return result;
}

VkPipelineLayout createReflectedPipelineLayout(
    VkDevice device,
    const std::unordered_map<int, VkDescriptorSetLayout>& setLayouts,
    const ReflectionData& vertReflection,
    const ReflectionData& fragReflection
) {
    // Collect layouts in order
    int maxSet = -1;
    for (auto& [idx, _] : setLayouts) {
        maxSet = std::max(maxSet, idx);
    }

    std::vector<VkDescriptorSetLayout> orderedLayouts;
    for (int i = 0; i <= maxSet; i++) {
        auto it = setLayouts.find(i);
        if (it != setLayouts.end()) {
            orderedLayouts.push_back(it->second);
        }
        // Note: gaps would need empty layouts in production
    }

    // Collect push constant ranges
    std::vector<VkPushConstantRange> pushRanges;
    auto addPushRanges = [&](const ReflectionData& refl) {
        for (auto& pc : refl.push_constants) {
            VkPushConstantRange range = {};
            range.stageFlags = stageFlagsFromStrings(pc.stage_flags);
            range.offset = 0;
            range.size = static_cast<uint32_t>(pc.size);
            pushRanges.push_back(range);
        }
    };
    addPushRanges(vertReflection);
    addPushRanges(fragReflection);

    VkPipelineLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = static_cast<uint32_t>(orderedLayouts.size());
    layoutInfo.pSetLayouts = orderedLayouts.data();
    layoutInfo.pushConstantRangeCount = static_cast<uint32_t>(pushRanges.size());
    layoutInfo.pPushConstantRanges = pushRanges.data();

    VkPipelineLayout pipelineLayout;
    if (vkCreatePipelineLayout(device, &layoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create pipeline layout from reflection");
    }

    return pipelineLayout;
}

#endif // !LUX_METAL_BACKEND

// ===========================================================================
// Manifest parsing (shared by both Vulkan and Metal)
// ===========================================================================

ShaderManifest parseManifestJson(const std::string& manifest_path) {
    ShaderManifest manifest;
    std::ifstream file(manifest_path);
    if (!file.is_open()) return manifest;

    std::stringstream ss;
    ss << file.rdbuf();
    std::string content = ss.str();

    JsonParser parser(content);
    JsonValue root = parser.parse();
    if (root.type != JsonValue::Object) return manifest;

    auto it = root.object.find("pipeline");
    if (it != root.object.end()) manifest.pipeline = it->second.asString();

    it = root.object.find("features");
    if (it != root.object.end() && it->second.type == JsonValue::Array) {
        for (auto& f : it->second.array) {
            manifest.featureNames.push_back(f.asString());
        }
    }

    it = root.object.find("permutations");
    if (it != root.object.end() && it->second.type == JsonValue::Array) {
        for (auto& p : it->second.array) {
            ManifestPermutation perm;
            auto si = p.object.find("suffix");
            if (si != p.object.end()) perm.suffix = si->second.asString();

            auto fi = p.object.find("features");
            if (fi != p.object.end() && fi->second.type == JsonValue::Object) {
                for (auto& [key, val] : fi->second.object) {
                    perm.features[key] = val.boolean;
                }
            }
            manifest.permutations.push_back(perm);
        }
    }

    return manifest;
}

ShaderManifest tryLoadManifest(const std::string& pipelineBase) {
    namespace fs = std::filesystem;

    // Try direct: pipelineBase + ".manifest.json"
    std::string path1 = pipelineBase + ".manifest.json";
    if (fs::exists(path1)) {
        std::cout << "[info] Loading shader manifest: " << path1 << std::endl;
        return parseManifestJson(path1);
    }

    // Try legacy subdirectory format: shadercache/gltf_pbr/gltf_pbr_layered.manifest.json
    // Extract filename from pipelineBase
    std::string filename = fs::path(pipelineBase).filename().string();
    std::string dir = fs::path(pipelineBase).parent_path().string();
    if (!dir.empty()) {
        std::string path2 = dir + "/gltf_pbr/" + filename + ".manifest.json";
        if (fs::exists(path2)) {
            std::cout << "[info] Loading shader manifest: " << path2 << std::endl;
            return parseManifestJson(path2);
        }
    }

    return ShaderManifest{}; // empty = no manifest
}

std::string findPermutationSuffix(const ShaderManifest& manifest,
                                   const std::set<std::string>& materialFeatures) {
    // Build feature map from the material's features
    std::unordered_map<std::string, bool> wanted;
    for (auto& fname : manifest.featureNames) {
        wanted[fname] = (materialFeatures.count(fname) > 0);
    }

    // Find exact match in manifest permutations
    for (auto& perm : manifest.permutations) {
        bool match = true;
        for (auto& fname : manifest.featureNames) {
            auto pi = perm.features.find(fname);
            bool permHas = (pi != perm.features.end() && pi->second);
            if (permHas != wanted[fname]) { match = false; break; }
        }
        if (match) return perm.suffix;
    }

    // Fallback: return base (empty suffix)
    return "";
}

#ifndef LUX_METAL_BACKEND

ReflectedVertexInput createReflectedVertexInput(const ReflectionData& vertReflection, int overrideStride) {
    ReflectedVertexInput result;

    result.binding = {};
    result.binding.binding = 0;
    // Use override stride when the actual vertex buffer has a larger stride than the
    // shader requires (e.g. buffer always has 48-byte stride with tangent, but shader
    // only reads position+normal+uv = 32 bytes). Vulkan allows stride > attribute sum.
    result.binding.stride = (overrideStride > 0)
        ? static_cast<uint32_t>(overrideStride)
        : static_cast<uint32_t>(vertReflection.vertex_stride);
    result.binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    for (auto& attr : vertReflection.vertex_attributes) {
        VkVertexInputAttributeDescription desc = {};
        desc.location = static_cast<uint32_t>(attr.location);
        desc.binding = 0;
        desc.format = reflectionFormatToVkFormat(attr.format);
        desc.offset = static_cast<uint32_t>(attr.offset);
        result.attributes.push_back(desc);
    }

    return result;
}

#endif // !LUX_METAL_BACKEND

// ===========================================================================
// Metal format mappings (Apple platforms only)
// ===========================================================================

#if defined(__APPLE__) && defined(LUX_METAL_BACKEND)

MTL::VertexFormat reflectionFormatToMTLVertexFormat(const std::string& fmt) {
    if (fmt == "R32_SFLOAT") return MTL::VertexFormatFloat;
    if (fmt == "R32G32_SFLOAT") return MTL::VertexFormatFloat2;
    if (fmt == "R32G32B32_SFLOAT") return MTL::VertexFormatFloat3;
    if (fmt == "R32G32B32A32_SFLOAT") return MTL::VertexFormatFloat4;
    if (fmt == "R32_SINT") return MTL::VertexFormatInt;
    if (fmt == "R32G32_SINT") return MTL::VertexFormatInt2;
    if (fmt == "R32G32B32_SINT") return MTL::VertexFormatInt3;
    if (fmt == "R32G32B32A32_SINT") return MTL::VertexFormatInt4;
    if (fmt == "R32_UINT") return MTL::VertexFormatUInt;
    if (fmt == "R32G32_UINT") return MTL::VertexFormatUInt2;
    if (fmt == "R32G32B32_UINT") return MTL::VertexFormatUInt3;
    if (fmt == "R32G32B32A32_UINT") return MTL::VertexFormatUInt4;
    if (fmt == "R16G16_SFLOAT") return MTL::VertexFormatHalf2;
    if (fmt == "R16G16B16A16_SFLOAT") return MTL::VertexFormatHalf4;
    return MTL::VertexFormatFloat4; // fallback
}

MTL::PixelFormat reflectionFormatToMTLPixelFormat(const std::string& fmt) {
    if (fmt == "R8G8B8A8_UNORM") return MTL::PixelFormatRGBA8Unorm;
    if (fmt == "B8G8R8A8_UNORM") return MTL::PixelFormatBGRA8Unorm;
    if (fmt == "R16G16B16A16_SFLOAT") return MTL::PixelFormatRGBA16Float;
    if (fmt == "R32_SFLOAT") return MTL::PixelFormatR32Float;
    if (fmt == "D32_SFLOAT") return MTL::PixelFormatDepth32Float;
    return MTL::PixelFormatRGBA8Unorm; // fallback
}

#endif
