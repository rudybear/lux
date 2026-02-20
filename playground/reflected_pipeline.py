"""Reflection-driven pipeline setup for wgpu.

Reads .lux.json sidecar files and creates wgpu bind group layouts,
pipeline layouts, and vertex buffer layouts automatically — no hardcoded
descriptor sets or vertex formats needed.

Usage:
    import json
    with open("shader.frag.json") as f:
        reflection = json.load(f)
    pipeline = ReflectedPipeline(device, vert_reflection, frag_reflection,
                                 vert_module, frag_module, render_format)
    bind_groups = pipeline.create_bind_groups(device, {
        "MVP": mvp_buffer,
        "Light": light_buffer,
        "albedo_tex": (sampler, texture_view),
    })
"""

from __future__ import annotations

import json
from pathlib import Path

import wgpu


# Map from reflection format strings to wgpu VertexFormat
_FORMAT_MAP = {
    "R32_SFLOAT": wgpu.VertexFormat.float32,
    "R32G32_SFLOAT": wgpu.VertexFormat.float32x2,
    "R32G32B32_SFLOAT": wgpu.VertexFormat.float32x3,
    "R32G32B32A32_SFLOAT": wgpu.VertexFormat.float32x4,
    "R32_SINT": wgpu.VertexFormat.sint32,
    "R32G32_SINT": wgpu.VertexFormat.sint32x2,
    "R32G32B32_SINT": wgpu.VertexFormat.sint32x3,
    "R32G32B32A32_SINT": wgpu.VertexFormat.sint32x4,
    "R32_UINT": wgpu.VertexFormat.uint32,
    "R32G32_UINT": wgpu.VertexFormat.uint32x2,
    "R32G32B32_UINT": wgpu.VertexFormat.uint32x3,
    "R32G32B32A32_UINT": wgpu.VertexFormat.uint32x4,
}


def load_reflection(json_path: Path) -> dict:
    """Load a .lux.json reflection file."""
    return json.loads(json_path.read_text(encoding="utf-8"))


def create_default_texture(device: wgpu.GPUDevice):
    """Create a 1x1 white RGBA texture for missing sampler bindings."""
    import numpy as np
    data = np.array([255, 255, 255, 255], dtype=np.uint8)
    texture = device.create_texture(
        size=(1, 1, 1),
        format=wgpu.TextureFormat.rgba8unorm,
        usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
    )
    device.queue.write_texture(
        wgpu.TexelCopyTextureInfo(texture=texture),
        data.tobytes(),
        wgpu.TexelCopyBufferLayout(offset=0, bytes_per_row=4, rows_per_image=1),
        (1, 1, 1),
    )
    view = texture.create_view()
    sampler = device.create_sampler(
        mag_filter=wgpu.FilterMode.linear,
        min_filter=wgpu.FilterMode.linear,
    )
    return sampler, view


class ReflectedPipeline:
    """Data-driven pipeline created from reflection metadata.

    Merges vertex and fragment reflection data to create a complete
    pipeline with proper bind group layouts and vertex buffer layout.
    """

    def __init__(
        self,
        device: wgpu.GPUDevice,
        vert_reflection: dict,
        frag_reflection: dict,
        vert_module: wgpu.GPUShaderModule,
        frag_module: wgpu.GPUShaderModule,
        color_format: wgpu.TextureFormat = wgpu.TextureFormat.rgba8unorm,
        depth_format: wgpu.TextureFormat = wgpu.TextureFormat.depth24plus,
        cull_mode: wgpu.CullMode = wgpu.CullMode.back,
    ):
        self.vert_reflection = vert_reflection
        self.frag_reflection = frag_reflection

        # Merge descriptor sets from both stages
        merged_sets = self._merge_descriptor_sets()

        # Create bind group layouts
        self.bind_group_layouts: dict[int, wgpu.GPUBindGroupLayout] = {}
        self._binding_info: dict[int, list[dict]] = {}  # set -> list of binding info

        for set_idx in sorted(merged_sets.keys()):
            entries = []
            binding_info = []
            for binding_data in sorted(merged_sets[set_idx], key=lambda b: b["binding"]):
                entry = self._create_bind_group_layout_entry(binding_data)
                entries.append(entry)
                binding_info.append(binding_data)

            self.bind_group_layouts[set_idx] = device.create_bind_group_layout(
                entries=entries
            )
            self._binding_info[set_idx] = binding_info

        # Create pipeline layout (sorted by set index)
        max_set = max(self.bind_group_layouts.keys()) if self.bind_group_layouts else -1
        layout_list = []
        for i in range(max_set + 1):
            if i in self.bind_group_layouts:
                layout_list.append(self.bind_group_layouts[i])
            else:
                # Create empty layout for gaps
                layout_list.append(device.create_bind_group_layout(entries=[]))

        self.pipeline_layout = device.create_pipeline_layout(
            bind_group_layouts=layout_list
        )

        # Create vertex buffer layout from vertex reflection
        vertex_buffers = self._create_vertex_buffers()

        # Create render pipeline
        self.pipeline = device.create_render_pipeline(
            layout=self.pipeline_layout,
            vertex=wgpu.VertexState(
                module=vert_module,
                entry_point="main",
                buffers=vertex_buffers,
            ),
            primitive=wgpu.PrimitiveState(
                topology=wgpu.PrimitiveTopology.triangle_list,
                cull_mode=cull_mode,
                front_face=wgpu.FrontFace.cw,
            ),
            depth_stencil=wgpu.DepthStencilState(
                format=depth_format,
                depth_write_enabled=True,
                depth_compare=wgpu.CompareFunction.less,
            ),
            multisample=wgpu.MultisampleState(count=1, mask=0xFFFFFFFF),
            fragment=wgpu.FragmentState(
                module=frag_module,
                entry_point="main",
                targets=[
                    wgpu.ColorTargetState(format=color_format),
                ],
            ),
        )

    def create_bind_groups(
        self,
        device: wgpu.GPUDevice,
        resources: dict,
    ) -> dict[int, wgpu.GPUBindGroup]:
        """Create bind groups from a name->resource mapping.

        Resources can be:
        - wgpu.GPUBuffer: for uniform_buffer bindings (auto-wraps in BufferBinding)
        - (sampler, texture_view) tuple: for sampler+sampled_image pairs
        - wgpu.GPUSampler: for standalone sampler bindings
        - wgpu.GPUTextureView: for standalone texture bindings

        The mapping is by name as it appears in the reflection JSON.
        """
        bind_groups = {}

        for set_idx, layout in self.bind_group_layouts.items():
            entries = []
            for binding_data in self._binding_info[set_idx]:
                name = binding_data["name"]
                binding_num = binding_data["binding"]
                btype = binding_data["type"]

                if name not in resources:
                    continue

                resource = resources[name]

                if btype == "uniform_buffer":
                    buf = resource
                    size = binding_data.get("size", 0)
                    entries.append(wgpu.BindGroupEntry(
                        binding=binding_num,
                        resource=wgpu.BufferBinding(buffer=buf, size=size),
                    ))

                elif btype == "sampler":
                    # Expect (sampler, texture_view) tuple
                    if isinstance(resource, tuple):
                        sampler, _ = resource
                    else:
                        sampler = resource
                    entries.append(wgpu.BindGroupEntry(
                        binding=binding_num,
                        resource=sampler,
                    ))

                elif btype in ("sampled_image", "sampled_cube_image"):
                    # Expect (sampler, texture_view) tuple
                    if isinstance(resource, tuple):
                        _, tex_view = resource
                    else:
                        tex_view = resource
                    entries.append(wgpu.BindGroupEntry(
                        binding=binding_num,
                        resource=tex_view,
                    ))

                elif btype == "storage_image":
                    entries.append(wgpu.BindGroupEntry(
                        binding=binding_num,
                        resource=resource,
                    ))

            if entries:
                bind_groups[set_idx] = device.create_bind_group(
                    layout=layout,
                    entries=entries,
                )

        return bind_groups

    def _merge_descriptor_sets(self) -> dict[int, list[dict]]:
        """Merge descriptor sets from vertex and fragment reflections."""
        merged: dict[int, dict[int, dict]] = {}  # set -> {binding -> data}

        for reflection in [self.vert_reflection, self.frag_reflection]:
            for set_str, bindings in reflection.get("descriptor_sets", {}).items():
                set_idx = int(set_str)
                if set_idx not in merged:
                    merged[set_idx] = {}
                for b in bindings:
                    binding_num = b["binding"]
                    if binding_num in merged[set_idx]:
                        # Merge stage flags
                        existing = merged[set_idx][binding_num]
                        existing_flags = set(existing.get("stage_flags", []))
                        new_flags = set(b.get("stage_flags", []))
                        existing["stage_flags"] = list(existing_flags | new_flags)
                    else:
                        merged[set_idx][binding_num] = dict(b)

        return {k: list(v.values()) for k, v in merged.items()}

    def _create_bind_group_layout_entry(self, binding_data: dict) -> wgpu.BindGroupLayoutEntry:
        """Create a single bind group layout entry from reflection data."""
        binding_num = binding_data["binding"]
        btype = binding_data["type"]

        # Determine visibility
        stage_flags = binding_data.get("stage_flags", [])
        if not stage_flags:
            # Default: both stages
            visibility = wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT
        else:
            visibility = 0
            if "vertex" in stage_flags:
                visibility |= wgpu.ShaderStage.VERTEX
            if "fragment" in stage_flags:
                visibility |= wgpu.ShaderStage.FRAGMENT
            if visibility == 0:
                visibility = wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT

        if btype == "uniform_buffer":
            return wgpu.BindGroupLayoutEntry(
                binding=binding_num,
                visibility=visibility,
                buffer=wgpu.BufferBindingLayout(type=wgpu.BufferBindingType.uniform),
            )
        elif btype == "sampler":
            return wgpu.BindGroupLayoutEntry(
                binding=binding_num,
                visibility=visibility,
                sampler=wgpu.SamplerBindingLayout(type=wgpu.SamplerBindingType.filtering),
            )
        elif btype == "sampled_image":
            return wgpu.BindGroupLayoutEntry(
                binding=binding_num,
                visibility=visibility,
                texture=wgpu.TextureBindingLayout(
                    sample_type=wgpu.TextureSampleType.float,
                    view_dimension=wgpu.TextureViewDimension.d2,
                ),
            )
        elif btype == "sampled_cube_image":
            return wgpu.BindGroupLayoutEntry(
                binding=binding_num,
                visibility=visibility,
                texture=wgpu.TextureBindingLayout(
                    sample_type=wgpu.TextureSampleType.float,
                    view_dimension=wgpu.TextureViewDimension.cube,
                ),
            )
        elif btype == "storage_image":
            return wgpu.BindGroupLayoutEntry(
                binding=binding_num,
                visibility=visibility,
                storage_texture=wgpu.StorageTextureBindingLayout(
                    access=wgpu.StorageTextureAccess.write_only,
                    format=wgpu.TextureFormat.rgba8unorm,
                    view_dimension=wgpu.TextureViewDimension.d2,
                ),
            )
        elif btype == "acceleration_structure":
            # wgpu doesn't directly support RT — placeholder
            return wgpu.BindGroupLayoutEntry(
                binding=binding_num,
                visibility=visibility,
                buffer=wgpu.BufferBindingLayout(type=wgpu.BufferBindingType.read_only_storage),
            )
        else:
            raise ValueError(f"Unknown binding type in reflection: {btype}")

    def _create_vertex_buffers(self) -> list[wgpu.VertexBufferLayout]:
        """Create vertex buffer layout from vertex reflection."""
        attrs = self.vert_reflection.get("vertex_attributes", [])
        stride = self.vert_reflection.get("vertex_stride", 0)

        if not attrs:
            return []

        wgpu_attrs = []
        for attr in attrs:
            fmt_str = attr.get("format", "R32G32B32A32_SFLOAT")
            fmt = _FORMAT_MAP.get(fmt_str, wgpu.VertexFormat.float32x4)
            wgpu_attrs.append(wgpu.VertexAttribute(
                format=fmt,
                offset=attr["offset"],
                shader_location=attr["location"],
            ))

        return [wgpu.VertexBufferLayout(
            array_stride=stride,
            step_mode=wgpu.VertexStepMode.vertex,
            attributes=wgpu_attrs,
        )]

    @classmethod
    def create_fullscreen(
        cls,
        device: wgpu.GPUDevice,
        frag_reflection: dict,
        frag_module: wgpu.GPUShaderModule,
        vert_module: wgpu.GPUShaderModule,
        color_format: wgpu.TextureFormat = wgpu.TextureFormat.rgba8unorm,
    ) -> "ReflectedPipeline":
        """Create a fullscreen pipeline (no vertex buffers, no depth test)."""
        instance = cls.__new__(cls)
        instance.vert_reflection = {}
        instance.frag_reflection = frag_reflection

        # Only fragment descriptor sets
        merged_sets = {}
        for set_str, bindings in frag_reflection.get("descriptor_sets", {}).items():
            set_idx = int(set_str)
            merged_sets[set_idx] = {b["binding"]: dict(b) for b in bindings}

        instance.bind_group_layouts = {}
        instance._binding_info = {}
        for set_idx in sorted(merged_sets.keys()):
            entries = []
            binding_info = []
            for binding_data in sorted(merged_sets[set_idx].values(), key=lambda b: b["binding"]):
                entry = instance._create_bind_group_layout_entry(binding_data)
                entries.append(entry)
                binding_info.append(binding_data)
            instance.bind_group_layouts[set_idx] = device.create_bind_group_layout(entries=entries)
            instance._binding_info[set_idx] = binding_info

        max_set = max(instance.bind_group_layouts.keys()) if instance.bind_group_layouts else -1
        layout_list = []
        for i in range(max_set + 1):
            if i in instance.bind_group_layouts:
                layout_list.append(instance.bind_group_layouts[i])
            else:
                layout_list.append(device.create_bind_group_layout(entries=[]))

        instance.pipeline_layout = device.create_pipeline_layout(bind_group_layouts=layout_list)

        instance.pipeline = device.create_render_pipeline(
            layout=instance.pipeline_layout,
            vertex=wgpu.VertexState(module=vert_module, entry_point="main", buffers=[]),
            primitive=wgpu.PrimitiveState(topology=wgpu.PrimitiveTopology.triangle_list),
            multisample=wgpu.MultisampleState(count=1, mask=0xFFFFFFFF),
            fragment=wgpu.FragmentState(
                module=frag_module, entry_point="main",
                targets=[wgpu.ColorTargetState(format=color_format)],
            ),
        )
        return instance

    def create_bind_groups_with_defaults(
        self,
        device: wgpu.GPUDevice,
        resources: dict,
    ) -> dict[int, wgpu.GPUBindGroup]:
        """Create bind groups, filling missing resources with defaults."""
        filled = dict(resources)
        default_tex = None

        for set_idx, bindings in self._binding_info.items():
            for b in bindings:
                name = b["name"]
                if name in filled:
                    continue
                if b["type"] in ("sampler", "sampled_image", "sampled_cube_image"):
                    if default_tex is None:
                        default_tex = create_default_texture(device)
                    filled[name] = default_tex
                elif b["type"] == "uniform_buffer":
                    size = b.get("size", 64)
                    filled[name] = device.create_buffer_with_data(
                        data=bytes(size),
                        usage=wgpu.BufferUsage.UNIFORM,
                    )

        return self.create_bind_groups(device, filled)
