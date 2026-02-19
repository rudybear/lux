# P8: Native Playgrounds — C++ (Vulkan) and Rust (ash)

## Goal

Create native GPU playgrounds in C++ and Rust that support ALL Lux shader features:
- Rasterization (vertex+fragment, fullscreen quad, PBR sphere)
- Ray tracing (raygen + closest_hit + miss via VK_KHR_ray_tracing_pipeline)
- Autodiff shaders (rendered as standard fragment shaders)
- SPIR-V loading from pre-compiled .spv files
- Screenshot capture + pixel readback for validation
- Interactive preview window

## Architecture Overview

Both implementations share the same architecture:

```
┌──────────────────────────────────────────────┐
│                 LuxPlayground                │
├──────────────┬───────────────┬───────────────┤
│  Raster Mode │ Fullscreen    │   RT Mode     │
│  (triangle/  │ (frag-only,   │  (raygen +    │
│   PBR sphere)│  UV coords)   │   rchit +     │
│              │               │   rmiss)      │
├──────────────┴───────────────┴───────────────┤
│              Vulkan Backend                  │
│  Instance → PhysicalDevice → Device → Queue  │
│  Swapchain / Offscreen render target         │
│  VMA/gpu-allocator for memory               │
│  BLAS/TLAS for RT mode                       │
│  SBT for RT dispatch                         │
├──────────────────────────────────────────────┤
│         Window (GLFW / winit)                │
└──────────────────────────────────────────────┘
```

---

## C++ Implementation

### Technology Stack

| Component | Library | Version |
|-----------|---------|---------|
| Vulkan bindings | Vulkan SDK (vulkan.h / vulkan.hpp) | 1.3+ |
| Windowing | GLFW 3.x | 3.4+ |
| Memory allocator | VMA (vk_mem_alloc.h) | 3.x |
| Device bootstrap | vk-bootstrap | latest |
| Math | GLM | 1.0+ |
| Image output | stb_image_write.h | latest |
| Build system | CMake | 3.20+ |

### Directory Structure

```
playground_cpp/
  CMakeLists.txt
  src/
    main.cpp                  -- CLI entry point + interactive preview
    vulkan_context.h/cpp      -- Instance, device, queues, swapchain (via vk-bootstrap)
    memory.h/cpp              -- VMA wrapper
    spv_loader.h/cpp          -- SPIR-V file loading + shader module creation
    raster_pipeline.h/cpp     -- Raster pipelines (triangle, fullscreen, PBR)
    rt_pipeline.h/cpp         -- RT pipeline (raygen/rchit/rmiss + SBT + AS)
    scene.h/cpp               -- Sphere mesh, vertex layouts, procedural texture
    camera.h/cpp              -- Perspective + look-at matrices (Vulkan conventions)
    screenshot.h/cpp          -- Readback to CPU + PNG save
    test_runner.h/cpp         -- Automated test: compile → render → validate
  deps/
    vk_mem_alloc.h            -- Single-header VMA
    stb_image_write.h         -- Single-header PNG writer
    vk_bootstrap.h/cpp        -- vk-bootstrap
  shaders/
    fullscreen.vert.glsl      -- Built-in fullscreen triangle vertex shader
```

### Key Classes

```cpp
// vulkan_context.h
class VulkanContext {
    VkInstance instance;
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkQueue graphicsQueue, presentQueue;
    VkSwapchainKHR swapchain;          // for interactive mode
    VkCommandPool commandPool;
    VmaAllocator allocator;
    // RT extension function pointers
    PFN_vkCreateAccelerationStructureKHR vkCreateAccelerationStructureKHR;
    PFN_vkCmdBuildAccelerationStructuresKHR vkCmdBuildAccelerationStructuresKHR;
    PFN_vkCmdTraceRaysKHR vkCmdTraceRaysKHR;
    PFN_vkCreateRayTracingPipelinesKHR vkCreateRayTracingPipelinesKHR;
    PFN_vkGetRayTracingShaderGroupHandlesKHR vkGetRayTracingShaderGroupHandlesKHR;
    PFN_vkGetAccelerationStructureDeviceAddressKHR vkGetAccelerationStructureDeviceAddressKHR;
    PFN_vkGetAccelerationStructureBuildSizesKHR vkGetAccelerationStructureBuildSizesKHR;
    PFN_vkDestroyAccelerationStructureKHR vkDestroyAccelerationStructureKHR;
    // Methods
    void init(bool enableRT, bool headless);
    void cleanup();
    bool supportsRT() const;
};

// raster_pipeline.h
class RasterPipeline {
    // Modes: TRIANGLE, FULLSCREEN, PBR
    enum class Mode { Triangle, Fullscreen, PBR };
    VkPipeline pipeline;
    VkPipelineLayout layout;
    VkDescriptorSetLayout descriptorSetLayout;
    void create(VulkanContext& ctx, Mode mode,
                const std::string& vertSpv, const std::string& fragSpv);
    void render(VkCommandBuffer cmd, uint32_t width, uint32_t height);
};

// rt_pipeline.h
class RTPipeline {
    VkPipeline pipeline;
    VkPipelineLayout layout;
    VkDescriptorSetLayout descriptorSetLayout;
    // Acceleration structures
    VkAccelerationStructureKHR blas, tlas;
    VkBuffer blasBuffer, tlasBuffer, scratchBuffer, instanceBuffer;
    // SBT
    VkBuffer sbtBuffer;
    VkStridedDeviceAddressRegionKHR raygenSBT, missSBT, hitSBT, callableSBT;
    // Storage image (RT output)
    VkImage storageImage;
    VkImageView storageImageView;

    void createAccelerationStructures(VulkanContext& ctx, const Mesh& mesh);
    void createPipeline(VulkanContext& ctx,
                        const std::string& rgenSpv,
                        const std::string& rchitSpv,
                        const std::string& rmissSpv);
    void createSBT(VulkanContext& ctx);
    void render(VkCommandBuffer cmd, uint32_t width, uint32_t height);
};
```

### Rendering Modes (matching Python playground)

#### Mode 1: Triangle
- Hardcoded 3-vertex RGB triangle
- Vertex layout: position(vec3) + color(vec3) = 24 bytes/vertex
- Loads user's vertex.spv + fragment.spv
- No uniforms

#### Mode 2: Fullscreen Quad
- Built-in fullscreen triangle vertex shader (compiled from GLSL or hardcoded SPIR-V)
- Passes UV coordinates at location 0
- Loads user's fragment.spv only
- No uniforms

#### Mode 3: PBR Sphere
- UV sphere mesh (32×32, position+normal+UV = 32 bytes/vertex)
- Descriptor set 0: MVP uniform buffer (3×mat4 = 192 bytes)
- Descriptor set 1: Light UBO (32 bytes) + sampler + texture
- Depth buffer (D32_SFLOAT)
- Backface culling, CCW front face
- Loads user's vertex.spv + fragment.spv

#### Mode 4: Ray Tracing
- Loads user's raygen.spv, closest_hit.spv, miss.spv
- Builds BLAS from sphere mesh triangles
- Builds TLAS with single instance (identity transform)
- Creates RT pipeline with 3 shader groups
- Creates SBT with proper alignment
- Descriptor set: TLAS (binding 0) + storage image (binding 1) + camera UBO (binding 2)
- Camera UBO: inverse view + inverse projection matrices
- Dispatches vkCmdTraceRaysKHR(width, height, 1)
- Copies storage image to readback buffer

### CLI Interface

```
lux-playground [OPTIONS] <shader>

Arguments:
  <shader>           Path to .lux file or base name for .spv files

Options:
  --mode <MODE>      Rendering mode: triangle|fullscreen|pbr|rt [default: auto-detect]
  --width <W>        Output width [default: 512]
  --height <H>       Output height [default: 512]
  --output <PATH>    Output PNG path [default: output.png]
  --interactive      Open GLFW preview window
  --validate         Run pixel validation after render
  --compile          Compile .lux file first (requires luxc on PATH)
  --headless         No window, offscreen render only
```

### Required Vulkan Extensions

```
VK_KHR_swapchain                    (interactive mode)
VK_KHR_acceleration_structure       (RT mode)
VK_KHR_ray_tracing_pipeline         (RT mode)
VK_KHR_deferred_host_operations     (RT mode)
VK_KHR_buffer_device_address        (RT mode, Vulkan 1.2 core)
VK_EXT_descriptor_indexing          (RT mode, Vulkan 1.2 core)
VK_KHR_pipeline_library             (RT mode)
```

---

## Rust Implementation

### Technology Stack

| Component | Crate | Version |
|-----------|-------|---------|
| Vulkan bindings | ash | 0.38+ |
| Windowing | winit | 0.30+ |
| Surface creation | ash-window | 0.13+ |
| Memory allocator | gpu-allocator | 0.28+ |
| Math | glam | 0.29+ |
| Image output | image (PNG) | 0.25+ |
| SPIR-V reflection | spirv-reflect | 0.2+ |
| Byte casting | bytemuck | 1.x |

### Directory Structure

```
playground_rust/
  Cargo.toml
  src/
    main.rs                   -- CLI entry + event loop
    vulkan_context.rs         -- Instance, device, queues, swapchain
    memory.rs                 -- gpu-allocator wrapper
    spv_loader.rs             -- SPIR-V file loading + shader module creation
    raster_pipeline.rs        -- Raster pipelines (triangle, fullscreen, PBR)
    rt_pipeline.rs            -- RT pipeline (raygen/rchit/rmiss + SBT + AS)
    scene.rs                  -- Sphere mesh, vertex layouts, procedural texture
    camera.rs                 -- Perspective + look-at (Vulkan conventions)
    screenshot.rs             -- Readback to CPU + PNG save
    test_runner.rs            -- Automated validation
  shaders/
    fullscreen.vert.spv       -- Pre-compiled fullscreen triangle vertex shader
```

### Key Structures

```rust
// vulkan_context.rs
pub struct VulkanContext {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub graphics_queue: vk::Queue,
    pub command_pool: vk::CommandPool,
    pub allocator: gpu_allocator::vulkan::Allocator,
    // RT extension loaders (Option — None if RT not available)
    pub rt_pipeline_loader: Option<ash::khr::ray_tracing_pipeline::Device>,
    pub accel_struct_loader: Option<ash::khr::acceleration_structure::Device>,
    pub rt_properties: Option<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR<'static>>,
}

// rt_pipeline.rs
pub struct RTPipeline {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_set: vk::DescriptorSet,
    pub blas: vk::AccelerationStructureKHR,
    pub tlas: vk::AccelerationStructureKHR,
    pub sbt_buffer: vk::Buffer,
    pub raygen_region: vk::StridedDeviceAddressRegionKHR,
    pub miss_region: vk::StridedDeviceAddressRegionKHR,
    pub hit_region: vk::StridedDeviceAddressRegionKHR,
    pub storage_image: vk::Image,
    pub storage_image_view: vk::ImageView,
}
```

---

## Shared Features (Both Implementations)

### SPIR-V Loading
1. Read binary file, verify magic number 0x07230203
2. Create VkShaderModule from raw bytes
3. Auto-detect shader stage from file extension (.vert/.frag/.rgen/.rchit/.rmiss)
4. Auto-detect rendering mode from which .spv files exist

### Screenshot Pipeline
1. Render to offscreen image (RGBA8_UNORM, 512×512)
2. Copy image → staging buffer (host-visible)
3. Map staging buffer, read pixels
4. Save as PNG
5. Optional: run validation checks (coverage, brightness, color channels)

### Camera Setup (matching Python playground)
- Eye: (0, 0, 3), Target: (0, 0, 0), Up: (0, 1, 0)
- FOV: 45°, Near: 0.1, Far: 100.0
- Vulkan Y-flip in projection matrix
- RT mode: pass inverse view + inverse projection

### Sphere Mesh (matching Python playground)
- 32 stacks × 32 slices UV sphere
- Vertex format: position(vec3) + normal(vec3) + uv(vec2) = 32 bytes
- uint32 index buffer
- Used for both PBR rasterization and RT BLAS

### Procedural Texture (matching Python playground)
- 512×512 RGBA8 checker pattern
- 8×8 grid, warm/cool alternating colors
- Sinusoidal color variation

---

## Validation Tests

Each playground should be able to run all existing Lux screenshot tests:

| Test | Mode | Validation |
|------|------|------------|
| hello_triangle | triangle | Color interpolation, vertex positions |
| pbr_surface | pbr | Sphere coverage, specular, shading |
| scheduled_pbr | pbr | Copper color, tonemap |
| brdf_gallery | fullscreen | 4 BRDF bands |
| colorspace_demo | fullscreen | HSV sweep, hue variation |
| texture_demo | fullscreen | Spatial patterns |
| autodiff_demo | fullscreen | Top/bottom halves differ |
| advanced_materials_demo | fullscreen | 4 material quadrants |
| sdf_shapes | fullscreen | SDF boundaries |
| procedural_noise | fullscreen | Noise patterns |
| rt_pathtracer | rt | RT pipeline rendering (NEW — actual pixels!) |
| rt_manual | rt | RT pipeline rendering (NEW — actual pixels!) |

---

## Implementation Order

### Phase 1: C++ Core (P8a)
1. CMake project setup with dependencies
2. VulkanContext (vk-bootstrap + VMA)
3. SPIR-V loader
4. Fullscreen quad rendering
5. Triangle rendering
6. PBR sphere rendering
7. Screenshot capture + PNG output
8. Interactive GLFW preview window

### Phase 2: C++ Ray Tracing (P8b)
1. RT extension loading + feature detection
2. BLAS creation from sphere mesh
3. TLAS creation with single instance
4. RT pipeline creation (load rgen/rchit/rmiss .spv)
5. SBT creation with proper alignment
6. Storage image + descriptor set
7. Ray dispatch + copy to readback
8. RT screenshot test

### Phase 3: Rust Core (P8c)
1. Cargo project setup with dependencies
2. VulkanContext (ash + gpu-allocator + ash-window)
3. SPIR-V loader
4. Fullscreen quad rendering
5. Triangle rendering
6. PBR sphere rendering
7. Screenshot capture + PNG output
8. Interactive winit preview window

### Phase 4: Rust Ray Tracing (P8d)
1. RT extension loading + feature detection
2. BLAS creation from sphere mesh
3. TLAS creation with single instance
4. RT pipeline creation
5. SBT creation
6. Storage image + descriptor set
7. Ray dispatch + copy to readback
8. RT screenshot test

---

## Build Instructions

### C++
```bash
cd playground_cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/lux-playground --mode fullscreen ../playground/brdf_gallery.frag.spv --output test.png
```

### Rust
```bash
cd playground_rust
cargo build --release
./target/release/lux-playground --mode fullscreen ../playground/brdf_gallery.frag.spv --output test.png
```

---

## Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| C++ Vulkan approach | vk-bootstrap + VMA | Minimizes boilerplate, well-documented |
| Rust Vulkan approach | ash + gpu-allocator | Full API control, most RT examples use ash |
| RT geometry | Triangle BLAS (not procedural AABB) | Matches existing sphere mesh, simpler |
| SBT layout | Separate regions, single buffer | Follows Khronos best practices |
| Output format | RGBA8_UNORM storage image | Avoids BGRA swizzle, matches Python |
| Fullscreen vertex shader | Pre-compiled SPIR-V | Avoids runtime GLSL compilation |
| Window library | GLFW (C++) / winit (Rust) | De facto standards |
| RT fallback | Graceful degradation | RT features optional; raster always works |
