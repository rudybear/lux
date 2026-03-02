# CPU-Side Rendering Guidance

## Buffer Layout Rules

### std140 (Uniform Buffers)
- Scalars: 4-byte aligned
- vec2: 8-byte aligned
- vec3/vec4: 16-byte aligned
- Arrays: each element rounded up to 16-byte stride
- Structs: rounded up to multiple of 16 bytes
- **Pitfall**: vec3 wastes 4 bytes per element — prefer vec4 or pack manually

### std430 (Storage Buffers)
- Same as std140 but arrays and structs use tighter packing
- vec3 arrays use 12-byte stride instead of 16
- Preferred for SSBOs — use wherever possible to reduce padding
- Not available for uniform buffers in core Vulkan

### Practical Packing Tips
- Place vec4 and mat4 fields first, then vec2, then scalars
- Group scalars into vec4 (e.g., `roughness`, `metallic`, `ao`, `pad` -> one vec4)
- Use `float[4]` instead of `vec3 + float` to avoid hidden padding
- Align uniform buffers to `minUniformBufferOffsetAlignment` (typically 256 bytes)

## Descriptor Set Strategy

### Recommended Set Layout
- **Set 0 — Per-frame**: Camera matrices, time, global lighting. Bind once per frame.
- **Set 1 — Per-material**: Textures, material parameters. Bind once per material group.
- **Set 2 — Per-object**: Model matrix, object ID, skinning data. Bind per draw call.
- **Set 3 — Per-pass** (optional): Shadow map, G-buffer attachments. Bind per render pass.

### Key Principles
- Higher-numbered sets change more frequently
- Changing set N invalidates sets N+1, N+2, ... (Vulkan spec)
- Minimize set 0 and set 1 rebinds — they cascade
- Use dynamic uniform buffers for per-object data to avoid per-draw descriptor updates
- Pre-allocate descriptor pools sized for worst-case frame

## Draw Call Batching and Sorting

### Sort Order (front-to-back priority)
1. Render pass / framebuffer
2. Pipeline (shader + fixed-function state)
3. Material / descriptor set 1
4. Front-to-back depth (opaque) or back-to-front (transparent)

### Batching Techniques
- **Instancing**: Use `vkCmdDrawIndexedIndirect` for objects sharing the same mesh + material
- **Multi-draw indirect**: Batch multiple meshes into a single indirect draw buffer
- **Merge by atlas**: Atlas small textures to share a single descriptor set
- **Vertex pulling**: Store vertices in SSBO, use gl_VertexIndex to pull — allows merging dissimilar meshes

### Targets
- Desktop: aim for <1000 draw calls per frame
- Mobile: aim for <200 draw calls per frame
- Each pipeline bind costs ~5-10 us on mobile, ~1-2 us on desktop

## Push Constants vs UBO

### Push Constants
- Up to 128 bytes guaranteed (256 on most desktop GPUs)
- Zero-latency: inlined into the command buffer
- Ideal for: model matrix, material index, per-draw parameters
- No descriptor set needed — just `vkCmdPushConstants`

### When to Use UBO Instead
- Data > 128 bytes (push constant limit)
- Data shared across many draws without change (per-frame camera)
- Arrays of structures (light lists, bone matrices)

### Rule of Thumb
- Per-draw data that fits in 128 bytes: push constants
- Per-frame or per-material data: UBO in set 0 or set 1
- Large variable-size arrays: SSBO with std430

## Vulkan-Specific Guidance

### Pipeline Management
- Create pipelines at load time, not during rendering
- Use `VkPipelineCache` to speed up pipeline creation across runs
- Use pipeline derivatives (`basePipelineHandle`) for variants that differ only in blend state or depth test
- Consider `VK_EXT_graphics_pipeline_library` to compile stages independently

### Command Buffer Recording
- Use secondary command buffers for parallel recording of render pass contents
- Reset command pools per frame rather than individual command buffers
- One command pool per thread — never share across threads

### Synchronization
- Prefer pipeline barriers over events for simple producer-consumer patterns
- Batch barriers: combine multiple image layout transitions into a single `vkCmdPipelineBarrier`
- Use `VK_DEPENDENCY_BY_REGION_BIT` for tile-based GPUs (mobile)
- Avoid `vkDeviceWaitIdle` — use fences per frame in flight

### Memory
- Use `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT` for GPU-only resources
- Use staging buffers + transfer queue for uploading large textures
- Sub-allocate from large `VkDeviceMemory` blocks (or use VMA)
- Align buffer offsets to `minStorageBufferOffsetAlignment` and `minUniformBufferOffsetAlignment`

### Render Passes
- Merge compatible passes to enable subpass dependencies (especially on tile-based GPUs)
- Use `LOAD_OP_CLEAR` or `LOAD_OP_DONT_CARE` instead of `LOAD_OP_LOAD` when possible
- Use `STORE_OP_DONT_CARE` for depth/stencil when not read later
- Transient attachments (`LAZILY_ALLOCATED`) save bandwidth on mobile
