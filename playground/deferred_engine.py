"""Deferred rendering engine for the Lux shader compiler playground.

Two-pass deferred renderer using wgpu-py:
  1. G-buffer pass: Renders scene geometry to 3 color attachments (rgba16float)
     plus depth (depth32float). Outputs albedo+metallic, normals+roughness,
     emission+occlusion.
  2. Lighting pass: Fullscreen triangle reads G-buffer textures and performs
     PBR lighting with IBL, outputting final color to an rgba8unorm target.

Usage:
    python playground/deferred_engine.py examples/deferred_basic \\
        --scene assets/DamagedHelmet.glb [--interactive] [--width 1024] [--height 768]

Dependencies:
    pip install wgpu numpy Pillow pygltflib
    For interactive mode: pip install glfw rendercanvas
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import wgpu

# Ensure playground/ is importable regardless of invocation method.
_PLAYGROUND_DIR = str(Path(__file__).resolve().parent)
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PLAYGROUND_DIR not in sys.path:
    sys.path.insert(0, _PLAYGROUND_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GBUF_COLOR_FORMAT = wgpu.TextureFormat.rgba16float
_GBUF_DEPTH_FORMAT = wgpu.TextureFormat.depth32float
_OUTPUT_FORMAT = wgpu.TextureFormat.rgba8unorm
_CLEAR_COLOR = (0.0, 0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# G-buffer resource management
# ---------------------------------------------------------------------------

@dataclass
class GBufferResources:
    """GPU textures, views, and sampler for the G-buffer."""
    rt_textures: list = field(default_factory=list)   # 3 color textures
    rt_views: list = field(default_factory=list)       # 3 color views
    depth_texture: object = None
    depth_view: object = None                          # for render attachment
    depth_sample_view: object = None                   # for sampling (depth-only aspect)
    sampler: object = None                             # shared sampler for lighting pass


def create_gbuffer(device: wgpu.GPUDevice, width: int, height: int) -> GBufferResources:
    """Create G-buffer textures and views."""
    gb = GBufferResources()

    # 3 color render targets: rgba16float
    for _i in range(3):
        tex = device.create_texture(
            size=(width, height, 1),
            format=_GBUF_COLOR_FORMAT,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING,
        )
        gb.rt_textures.append(tex)
        gb.rt_views.append(tex.create_view())

    # Depth: depth32float (readable in lighting pass)
    gb.depth_texture = device.create_texture(
        size=(width, height, 1),
        format=_GBUF_DEPTH_FORMAT,
        usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING,
    )
    gb.depth_view = gb.depth_texture.create_view()
    gb.depth_sample_view = gb.depth_texture.create_view(
        aspect=wgpu.TextureAspect.depth_only,
    )

    # Shared sampler for reading G-buffer in lighting pass
    gb.sampler = device.create_sampler(
        mag_filter=wgpu.FilterMode.nearest,
        min_filter=wgpu.FilterMode.nearest,
        address_mode_u=wgpu.AddressMode.clamp_to_edge,
        address_mode_v=wgpu.AddressMode.clamp_to_edge,
        address_mode_w=wgpu.AddressMode.clamp_to_edge,
    )

    return gb


# ---------------------------------------------------------------------------
# Material UBO packing (matching shader std140 layout, 144 bytes)
# ---------------------------------------------------------------------------

def _pack_material_ubo(mat) -> bytes:
    """Pack material properties into std140 UBO (144 bytes).

    Layout:
        offset   0: base_color_factor  vec4
        offset  16: emissive_factor    vec3
        offset  28: metallic_factor    float
        offset  32: roughness_factor   float
        offset  36: emissive_strength  float
        offset  40: ior                float
        offset  44: clearcoat_factor   float
        offset  48: clearcoat_roughness float
        offset  52: sheen_roughness    float
        offset  56: transmission       float
        offset  60: _pad               float
        offset  64: sheen_color_factor vec3
        offset  76: _pad               float
        offset  80: baseColorUvSt      vec4
        offset  96: normalUvSt         vec4
        offset 112: mrUvSt             vec4
        offset 128: baseColorUvRot     float
        offset 132: normalUvRot        float
        offset 136: mrUvRot            float
        offset 140: _pad               float
    """
    buf = bytearray(144)
    bc = mat.base_color
    struct.pack_into("4f", buf, 0, bc[0], bc[1], bc[2], bc[3] if len(bc) > 3 else 1.0)
    em = mat.emissive
    struct.pack_into("3f", buf, 16, em[0], em[1], em[2] if len(em) > 2 else 0.0)
    struct.pack_into("f", buf, 28, mat.metallic)
    struct.pack_into("f", buf, 32, mat.roughness)
    struct.pack_into("f", buf, 36, getattr(mat, 'emissive_strength', 1.0))
    struct.pack_into("f", buf, 40, getattr(mat, 'ior', 1.5))
    struct.pack_into("f", buf, 44, getattr(mat, 'clearcoat_factor', 0.0))
    struct.pack_into("f", buf, 48, getattr(mat, 'clearcoat_roughness_factor', 0.0))
    struct.pack_into("f", buf, 52, getattr(mat, 'sheen_roughness_factor', 0.0))
    struct.pack_into("f", buf, 56, getattr(mat, 'transmission_factor', 0.0))
    # pad at 60
    sc = getattr(mat, 'sheen_color_factor', (0.0, 0.0, 0.0))
    struct.pack_into("3f", buf, 64, sc[0], sc[1], sc[2] if len(sc) > 2 else 0.0)
    # UV transforms: identity
    struct.pack_into("4f", buf, 80, 0.0, 0.0, 1.0, 1.0)
    struct.pack_into("4f", buf, 96, 0.0, 0.0, 1.0, 1.0)
    struct.pack_into("4f", buf, 112, 0.0, 0.0, 1.0, 1.0)
    # rotations default to 0 (already zeroed)
    return bytes(buf)


# ---------------------------------------------------------------------------
# DeferredCamera UBO (80 bytes)
# ---------------------------------------------------------------------------

def _pack_deferred_camera_ubo(inv_view_proj: np.ndarray, view_pos: np.ndarray) -> bytes:
    """Pack DeferredCamera UBO: mat4 inv_view_proj (64) + vec3 view_pos + pad (16)."""
    buf = bytearray(80)
    # inv_view_proj as 16 floats (column-major)
    flat = inv_view_proj.flatten().tolist()
    struct.pack_into("16f", buf, 0, *flat[:16])
    struct.pack_into("3f", buf, 64, float(view_pos[0]), float(view_pos[1]), float(view_pos[2]))
    # pad at 76 (already 0)
    return bytes(buf)


# ---------------------------------------------------------------------------
# Lights buffer packing (std430: 64 bytes per light)
# ---------------------------------------------------------------------------

def _pack_lights_buffer(lights: list) -> bytes:
    """Pack scene lights into GPU buffer (64 bytes per light).

    Layout per light:
        vec4: (light_type, intensity, range, inner_cone)
        vec4: (position.xyz, outer_cone)
        vec4: (direction.xyz, shadow_index)
        vec4: (color.xyz, pad)
    """
    buf = bytearray()
    for light in lights:
        light_type = {"directional": 0.0, "point": 1.0, "spot": 2.0}.get(
            getattr(light, 'type', 'directional'), 0.0)
        intensity = getattr(light, 'intensity', 1.0)
        range_val = getattr(light, 'range', 0.0)
        inner_cone = getattr(light, 'inner_cone_angle', 0.0)
        pos = getattr(light, 'position', np.zeros(3))
        outer_cone = getattr(light, 'outer_cone_angle', 0.7854)
        direction = getattr(light, 'direction', np.array([0, -1, 0]))
        shadow_idx = getattr(light, 'shadow_index', -1.0)
        color = getattr(light, 'color', (1.0, 1.0, 1.0))

        buf += struct.pack("4f", light_type, intensity, range_val, inner_cone)
        buf += struct.pack("3f", float(pos[0]), float(pos[1]), float(pos[2]))
        buf += struct.pack("f", outer_cone)
        buf += struct.pack("3f", float(direction[0]), float(direction[1]), float(direction[2]))
        buf += struct.pack("f", float(shadow_idx))
        buf += struct.pack("3f", float(color[0]), float(color[1]), float(color[2]))
        buf += struct.pack("f", 0.0)

    if not buf:
        # Dummy directional light (white, pointing down)
        buf = bytearray(64)
        struct.pack_into("f", buf, 0, 0.0)       # type: directional
        struct.pack_into("f", buf, 4, 1.0)       # intensity
        struct.pack_into("f", buf, 28, 0.7854)   # outer_cone
        struct.pack_into("f", buf, 36, -1.0)     # dir.y
        struct.pack_into("f", buf, 48, 1.0)      # color.r
        struct.pack_into("f", buf, 52, 1.0)      # color.g
        struct.pack_into("f", buf, 56, 1.0)      # color.b

    return bytes(buf)


# ---------------------------------------------------------------------------
# Texture upload helpers
# ---------------------------------------------------------------------------

def _upload_texture(device: wgpu.GPUDevice, data: np.ndarray,
                    address_mode: str = "repeat"):
    """Upload an RGBA uint8 image to GPU. Returns (sampler, texture_view)."""
    h, w = data.shape[:2]
    texture = device.create_texture(
        size=(w, h, 1),
        format=wgpu.TextureFormat.rgba8unorm,
        usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
    )
    device.queue.write_texture(
        wgpu.TexelCopyTextureInfo(texture=texture),
        data.tobytes(),
        wgpu.TexelCopyBufferLayout(offset=0, bytes_per_row=w * 4, rows_per_image=h),
        (w, h, 1),
    )
    view = texture.create_view()
    addr = wgpu.AddressMode.repeat if address_mode == "repeat" else wgpu.AddressMode.clamp_to_edge
    sampler = device.create_sampler(
        mag_filter=wgpu.FilterMode.linear,
        min_filter=wgpu.FilterMode.linear,
        address_mode_u=addr,
        address_mode_v=addr,
        address_mode_w=addr,
    )
    return sampler, view


def _create_default_texture(device: wgpu.GPUDevice):
    """1x1 white RGBA texture."""
    data = np.array([[255, 255, 255, 255]], dtype=np.uint8).reshape(1, 1, 4)
    return _upload_texture(device, data)


def _create_default_normal_texture(device: wgpu.GPUDevice):
    """1x1 flat normal map (0.5, 0.5, 1.0) -- no perturbation."""
    data = np.array([[128, 128, 255, 255]], dtype=np.uint8).reshape(1, 1, 4)
    return _upload_texture(device, data)


def _create_black_texture(device: wgpu.GPUDevice):
    """1x1 black RGBA texture for emissive defaults."""
    data = np.array([[0, 0, 0, 255]], dtype=np.uint8).reshape(1, 1, 4)
    return _upload_texture(device, data)


def _create_default_brdf_lut(device: wgpu.GPUDevice):
    """1x1 BRDF LUT with sensible PBR defaults."""
    brdf_rgba = np.array([[0.5, 0.0, 0.0, 0.0]], dtype=np.float16).reshape(1, 1, 4)
    texture = device.create_texture(
        size=(1, 1, 1),
        format=wgpu.TextureFormat.rgba16float,
        usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
    )
    device.queue.write_texture(
        wgpu.TexelCopyTextureInfo(texture=texture),
        brdf_rgba.tobytes(),
        wgpu.TexelCopyBufferLayout(offset=0, bytes_per_row=8, rows_per_image=1),
        (1, 1, 1),
    )
    view = texture.create_view()
    sampler = device.create_sampler(
        mag_filter=wgpu.FilterMode.linear,
        min_filter=wgpu.FilterMode.linear,
    )
    return sampler, view


def _create_default_cubemap(device: wgpu.GPUDevice, face_size: int = 1):
    """1x1 dim grey cubemap (0.2, 0.2, 0.2) for missing IBL."""
    data = np.full((6, face_size, face_size, 4), 0.2, dtype=np.float16)
    data[:, :, :, 3] = 1.0
    return _upload_cubemap_f16(device, face_size, data)


def _upload_cubemap_f16(device: wgpu.GPUDevice, face_size: int, data: np.ndarray):
    """Upload float16 RGBA cubemap. data: (6, h, w, 4) float16."""
    texture = device.create_texture(
        size=(face_size, face_size, 6),
        dimension="2d",
        format=wgpu.TextureFormat.rgba16float,
        usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
    )
    bytes_per_row = face_size * 4 * 2
    for face_idx in range(6):
        face_data = np.ascontiguousarray(data[face_idx])
        device.queue.write_texture(
            wgpu.TexelCopyTextureInfo(texture=texture, origin=(0, 0, face_idx)),
            face_data.tobytes(),
            wgpu.TexelCopyBufferLayout(offset=0, bytes_per_row=bytes_per_row,
                                        rows_per_image=face_size),
            (face_size, face_size, 1),
        )
    view = texture.create_view(dimension="cube")
    sampler = device.create_sampler(
        mag_filter=wgpu.FilterMode.linear,
        min_filter=wgpu.FilterMode.linear,
        mipmap_filter=wgpu.MipmapFilterMode.linear,
    )
    return sampler, view


# ---------------------------------------------------------------------------
# IBL loading (reused from engine.py patterns)
# ---------------------------------------------------------------------------

def _load_ibl_assets(device: wgpu.GPUDevice, ibl_name: str) -> dict:
    """Load preprocessed IBL assets. Returns name -> (sampler, view) dict."""
    import json as _json
    assets_dir = Path(_PROJECT_ROOT) / "assets" / "ibl" / ibl_name
    manifest_path = assets_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"[warn] IBL assets not found at {assets_dir}, using defaults")
        return {}

    with open(manifest_path) as f:
        manifest = _json.load(f)

    result = {}

    spec_info = manifest.get("specular", {})
    irr_info = manifest.get("irradiance", {})
    brdf_info = manifest.get("brdf_lut", {})

    # Specular cubemap (with mips)
    spec_path = assets_dir / "specular.bin"
    if spec_path.exists():
        spec_data = np.frombuffer(spec_path.read_bytes(), dtype=np.float16)
        face_size = spec_info.get("face_size", manifest.get("specular_face_size", 256))
        mip_count = spec_info.get("mip_count", manifest.get("specular_mip_count", 5))
        texture = device.create_texture(
            size=(face_size, face_size, 6),
            dimension="2d",
            format=wgpu.TextureFormat.rgba16float,
            mip_level_count=mip_count,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        )
        offset = 0
        for mip in range(mip_count):
            mip_size = max(face_size >> mip, 1)
            face_texels = mip_size * mip_size * 4
            for face_idx in range(6):
                face_data = spec_data[offset:offset + face_texels].copy()
                offset += face_texels
                bytes_per_row = mip_size * 4 * 2
                device.queue.write_texture(
                    wgpu.TexelCopyTextureInfo(texture=texture, mip_level=mip,
                                              origin=(0, 0, face_idx)),
                    face_data.tobytes(),
                    wgpu.TexelCopyBufferLayout(offset=0, bytes_per_row=bytes_per_row,
                                                rows_per_image=mip_size),
                    (mip_size, mip_size, 1),
                )
        view = texture.create_view(dimension="cube")
        sampler = device.create_sampler(
            mag_filter=wgpu.FilterMode.linear,
            min_filter=wgpu.FilterMode.linear,
            mipmap_filter=wgpu.MipmapFilterMode.linear,
        )
        result["env_specular"] = (sampler, view)

    # Irradiance cubemap
    irr_path = assets_dir / "irradiance.bin"
    if irr_path.exists():
        irr_size = irr_info.get("face_size", manifest.get("irradiance_face_size", 32))
        irr_data = np.frombuffer(irr_path.read_bytes(), dtype=np.float16)
        irr_data = irr_data.reshape(6, irr_size, irr_size, 4)
        result["env_irradiance"] = _upload_cubemap_f16(device, irr_size, irr_data)

    # BRDF LUT
    brdf_path = assets_dir / "brdf_lut.bin"
    if brdf_path.exists():
        lut_size = brdf_info.get("size", manifest.get("brdf_lut_size", 512))
        brdf_raw = np.frombuffer(brdf_path.read_bytes(), dtype=np.float16)
        brdf_rg = brdf_raw.reshape(lut_size, lut_size, 2)
        brdf_rgba = np.zeros((lut_size, lut_size, 4), dtype=np.float16)
        brdf_rgba[:, :, 0] = brdf_rg[:, :, 0]
        brdf_rgba[:, :, 1] = brdf_rg[:, :, 1]
        texture = device.create_texture(
            size=(lut_size, lut_size, 1),
            format=wgpu.TextureFormat.rgba16float,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        )
        device.queue.write_texture(
            wgpu.TexelCopyTextureInfo(texture=texture),
            brdf_rgba.tobytes(),
            wgpu.TexelCopyBufferLayout(offset=0, bytes_per_row=lut_size * 8,
                                        rows_per_image=lut_size),
            (lut_size, lut_size, 1),
        )
        view = texture.create_view()
        sampler = device.create_sampler(
            mag_filter=wgpu.FilterMode.linear,
            min_filter=wgpu.FilterMode.linear,
        )
        result["brdf_lut"] = (sampler, view)

    print(f"[info] Loaded IBL assets: {list(result.keys())} from {assets_dir}")
    return result


def _auto_load_ibl(device: wgpu.GPUDevice) -> dict:
    """Auto-detect and load IBL assets (prefer pisa, then neutral)."""
    ibl_dir = Path(_PROJECT_ROOT) / "assets" / "ibl"
    if not ibl_dir.exists():
        return {}
    preferred = ["pisa", "neutral"]
    candidates = [d.name for d in ibl_dir.iterdir()
                  if d.is_dir() and (d / "manifest.json").exists()]
    ordered = [n for n in preferred if n in candidates]
    ordered += [n for n in sorted(candidates) if n not in preferred]
    if ordered:
        return _load_ibl_assets(device, ordered[0])
    return {}


# ---------------------------------------------------------------------------
# DeferredRenderer
# ---------------------------------------------------------------------------

class DeferredRenderer:
    """Two-pass deferred renderer for Lux compiled shaders.

    Pass 1 (G-buffer): renders scene meshes to 3 rgba16float MRT + depth32float.
    Pass 2 (Lighting): fullscreen triangle reads G-buffer, applies PBR lighting.
    """

    def __init__(self, pipeline_base: str, width: int = 1024, height: int = 768):
        self.pipeline_base = pipeline_base
        self.width = width
        self.height = height

        # GPU setup
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        if adapter is None:
            raise RuntimeError("No suitable GPU adapter found")
        self.device = adapter.request_device_sync()

        # Load reflection JSON for all 4 stages
        base = Path(pipeline_base)
        self.gbuf_vert_refl = self._load_json(base.parent / (base.name + ".gbuf.vert.json"))
        self.gbuf_frag_refl = self._load_json(base.parent / (base.name + ".gbuf.frag.json"))
        self.light_vert_refl = self._load_json(base.parent / (base.name + ".light.vert.json"))
        self.light_frag_refl = self._load_json(base.parent / (base.name + ".light.frag.json"))

        # Load SPIR-V shader modules
        from render_harness import load_shader_module
        self.gbuf_vert_mod = load_shader_module(
            self.device, base.parent / (base.name + ".gbuf.vert.spv"))
        self.gbuf_frag_mod = load_shader_module(
            self.device, base.parent / (base.name + ".gbuf.frag.spv"))
        self.light_vert_mod = load_shader_module(
            self.device, base.parent / (base.name + ".light.vert.spv"))
        self.light_frag_mod = load_shader_module(
            self.device, base.parent / (base.name + ".light.frag.spv"))

        # Create G-buffer textures
        self.gbuffer = create_gbuffer(self.device, width, height)

        # Create pipelines
        self._create_gbuffer_pipeline()
        self._create_lighting_pipeline()

        # IBL textures (loaded later when scene is known)
        self.ibl_textures: dict = {}

        print(f"[info] DeferredRenderer initialized: {width}x{height}")

    @staticmethod
    def _load_json(path: Path) -> dict:
        """Load a reflection JSON file."""
        if not path.exists():
            raise FileNotFoundError(f"Reflection JSON not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    # -------------------------------------------------------------------
    # G-buffer pipeline creation
    # -------------------------------------------------------------------

    def _create_gbuffer_pipeline(self):
        """Create the G-buffer render pipeline (MRT + depth)."""
        from reflected_pipeline import ReflectedPipeline

        # Merge descriptor sets from vert and frag reflections
        merged_sets = self._merge_descriptor_sets(
            self.gbuf_vert_refl, self.gbuf_frag_refl)

        # Build bind group layouts
        self.gbuf_bind_group_layouts: dict[int, wgpu.GPUBindGroupLayout] = {}
        self.gbuf_binding_info: dict[int, list[dict]] = {}

        for set_idx in sorted(merged_sets.keys()):
            entries = []
            binding_info = []
            for binding_data in sorted(merged_sets[set_idx], key=lambda b: b["binding"]):
                entry = self._create_bgl_entry(binding_data)
                entries.append(entry)
                binding_info.append(binding_data)
            self.gbuf_bind_group_layouts[set_idx] = self.device.create_bind_group_layout(
                entries=entries)
            self.gbuf_binding_info[set_idx] = binding_info

        # Pipeline layout
        max_set = max(self.gbuf_bind_group_layouts.keys()) if self.gbuf_bind_group_layouts else -1
        layout_list = []
        for i in range(max_set + 1):
            if i in self.gbuf_bind_group_layouts:
                layout_list.append(self.gbuf_bind_group_layouts[i])
            else:
                layout_list.append(self.device.create_bind_group_layout(entries=[]))
        self.gbuf_pipeline_layout = self.device.create_pipeline_layout(
            bind_group_layouts=layout_list)

        # Vertex buffer layout from reflection
        vertex_buffers = self._create_vertex_buffers(self.gbuf_vert_refl)

        # Create render pipeline with 3 color targets + depth
        self.gbuf_pipeline = self.device.create_render_pipeline(
            layout=self.gbuf_pipeline_layout,
            vertex=wgpu.VertexState(
                module=self.gbuf_vert_mod,
                entry_point="main",
                buffers=vertex_buffers,
            ),
            primitive=wgpu.PrimitiveState(
                topology=wgpu.PrimitiveTopology.triangle_list,
                cull_mode=wgpu.CullMode.back,
                front_face=wgpu.FrontFace.cw,
            ),
            depth_stencil=wgpu.DepthStencilState(
                format=_GBUF_DEPTH_FORMAT,
                depth_write_enabled=True,
                depth_compare=wgpu.CompareFunction.less,
            ),
            multisample=wgpu.MultisampleState(count=1, mask=0xFFFFFFFF),
            fragment=wgpu.FragmentState(
                module=self.gbuf_frag_mod,
                entry_point="main",
                targets=[
                    wgpu.ColorTargetState(format=_GBUF_COLOR_FORMAT),
                    wgpu.ColorTargetState(format=_GBUF_COLOR_FORMAT),
                    wgpu.ColorTargetState(format=_GBUF_COLOR_FORMAT),
                ],
            ),
        )

        # Store vertex stride for padding
        self.gbuf_vertex_stride = self.gbuf_vert_refl.get("vertex_stride", 48)

    # -------------------------------------------------------------------
    # Lighting pipeline creation (fullscreen triangle)
    # -------------------------------------------------------------------

    def _create_lighting_pipeline(self):
        """Create the fullscreen lighting pipeline."""
        # Only fragment descriptor sets matter for the lighting pass
        frag_sets = {}
        for set_str, bindings in self.light_frag_refl.get("descriptor_sets", {}).items():
            set_idx = int(set_str)
            frag_sets[set_idx] = {b["binding"]: dict(b) for b in bindings}

        self.light_bind_group_layouts: dict[int, wgpu.GPUBindGroupLayout] = {}
        self.light_binding_info: dict[int, list[dict]] = {}

        for set_idx in sorted(frag_sets.keys()):
            entries = []
            binding_info = []
            for binding_data in sorted(frag_sets[set_idx].values(), key=lambda b: b["binding"]):
                entry = self._create_bgl_entry(binding_data)
                entries.append(entry)
                binding_info.append(binding_data)
            self.light_bind_group_layouts[set_idx] = self.device.create_bind_group_layout(
                entries=entries)
            self.light_binding_info[set_idx] = binding_info

        max_set = max(self.light_bind_group_layouts.keys()) if self.light_bind_group_layouts else -1
        layout_list = []
        for i in range(max_set + 1):
            if i in self.light_bind_group_layouts:
                layout_list.append(self.light_bind_group_layouts[i])
            else:
                layout_list.append(self.device.create_bind_group_layout(entries=[]))
        self.light_pipeline_layout = self.device.create_pipeline_layout(
            bind_group_layouts=layout_list)

        # Create render pipeline: no vertex buffers, no depth test
        self.light_pipeline = self.device.create_render_pipeline(
            layout=self.light_pipeline_layout,
            vertex=wgpu.VertexState(
                module=self.light_vert_mod,
                entry_point="main",
                buffers=[],
            ),
            primitive=wgpu.PrimitiveState(
                topology=wgpu.PrimitiveTopology.triangle_list,
            ),
            multisample=wgpu.MultisampleState(count=1, mask=0xFFFFFFFF),
            fragment=wgpu.FragmentState(
                module=self.light_frag_mod,
                entry_point="main",
                targets=[
                    wgpu.ColorTargetState(format=_OUTPUT_FORMAT),
                ],
            ),
        )

    # -------------------------------------------------------------------
    # Bind group helpers
    # -------------------------------------------------------------------

    def _merge_descriptor_sets(self, vert_refl: dict, frag_refl: dict) -> dict[int, list[dict]]:
        """Merge descriptor sets from vertex and fragment reflection."""
        merged: dict[int, dict[int, dict]] = {}
        for reflection in [vert_refl, frag_refl]:
            for set_str, bindings in reflection.get("descriptor_sets", {}).items():
                set_idx = int(set_str)
                if set_idx not in merged:
                    merged[set_idx] = {}
                for b in bindings:
                    binding_num = b["binding"]
                    if binding_num in merged[set_idx]:
                        existing = merged[set_idx][binding_num]
                        existing_flags = set(existing.get("stage_flags", []))
                        new_flags = set(b.get("stage_flags", []))
                        existing["stage_flags"] = list(existing_flags | new_flags)
                    else:
                        merged[set_idx][binding_num] = dict(b)
        return {k: list(v.values()) for k, v in merged.items()}

    def _create_bgl_entry(self, binding_data: dict) -> wgpu.BindGroupLayoutEntry:
        """Create a single bind group layout entry from reflection data."""
        binding_num = binding_data["binding"]
        btype = binding_data["type"]
        name = binding_data.get("name", "")

        # Determine visibility
        stage_flags = binding_data.get("stage_flags", [])
        if not stage_flags:
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
            # Depth texture sampler must be non-filtering (pairs with unfilterable_float)
            if name == "gbuf_depth":
                return wgpu.BindGroupLayoutEntry(
                    binding=binding_num,
                    visibility=visibility,
                    sampler=wgpu.SamplerBindingLayout(
                        type=wgpu.SamplerBindingType.non_filtering),
                )
            return wgpu.BindGroupLayoutEntry(
                binding=binding_num,
                visibility=visibility,
                sampler=wgpu.SamplerBindingLayout(type=wgpu.SamplerBindingType.filtering),
            )
        elif btype == "sampled_image":
            # Depth texture binding needs special sample type
            if name == "gbuf_depth":
                return wgpu.BindGroupLayoutEntry(
                    binding=binding_num,
                    visibility=visibility,
                    texture=wgpu.TextureBindingLayout(
                        sample_type=wgpu.TextureSampleType.unfilterable_float,
                        view_dimension=wgpu.TextureViewDimension.d2,
                    ),
                )
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
        elif btype == "storage_buffer":
            return wgpu.BindGroupLayoutEntry(
                binding=binding_num,
                visibility=visibility,
                buffer=wgpu.BufferBindingLayout(type=wgpu.BufferBindingType.read_only_storage),
            )
        else:
            raise ValueError(f"Unknown binding type: {btype}")

    _FORMAT_MAP = {
        "R32_SFLOAT": wgpu.VertexFormat.float32,
        "R32G32_SFLOAT": wgpu.VertexFormat.float32x2,
        "R32G32B32_SFLOAT": wgpu.VertexFormat.float32x3,
        "R32G32B32A32_SFLOAT": wgpu.VertexFormat.float32x4,
    }

    def _create_vertex_buffers(self, vert_refl: dict) -> list[wgpu.VertexBufferLayout]:
        """Create vertex buffer layout from vertex reflection."""
        attrs = vert_refl.get("vertex_attributes", [])
        stride = vert_refl.get("vertex_stride", 0)
        if not attrs:
            return []
        wgpu_attrs = []
        for attr in attrs:
            fmt_str = attr.get("format", "R32G32B32A32_SFLOAT")
            fmt = self._FORMAT_MAP.get(fmt_str, wgpu.VertexFormat.float32x4)
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

    def _create_bind_groups(self, layout_map: dict, binding_info: dict,
                            resources: dict) -> dict[int, wgpu.GPUBindGroup]:
        """Create bind groups from a resource map, matching the binding_info metadata."""
        bind_groups = {}
        for set_idx, layout in layout_map.items():
            entries = []
            for bdata in binding_info[set_idx]:
                name = bdata["name"]
                binding_num = bdata["binding"]
                btype = bdata["type"]

                if name not in resources:
                    continue

                resource = resources[name]

                if btype == "uniform_buffer":
                    size = bdata.get("size", 0)
                    entries.append(wgpu.BindGroupEntry(
                        binding=binding_num,
                        resource=wgpu.BufferBinding(buffer=resource, size=size),
                    ))
                elif btype == "storage_buffer":
                    entries.append(wgpu.BindGroupEntry(
                        binding=binding_num,
                        resource=wgpu.BufferBinding(buffer=resource, size=resource.size),
                    ))
                elif btype == "sampler":
                    if isinstance(resource, tuple):
                        sampler, _ = resource
                    else:
                        sampler = resource
                    entries.append(wgpu.BindGroupEntry(
                        binding=binding_num,
                        resource=sampler,
                    ))
                elif btype in ("sampled_image", "sampled_cube_image"):
                    if isinstance(resource, tuple):
                        _, tex_view = resource
                    else:
                        tex_view = resource
                    entries.append(wgpu.BindGroupEntry(
                        binding=binding_num,
                        resource=tex_view,
                    ))

            if entries:
                bind_groups[set_idx] = self.device.create_bind_group(
                    layout=layout,
                    entries=entries,
                )
        return bind_groups

    def _fill_missing_resources(self, binding_info: dict, resources: dict):
        """Fill missing texture/buffer resources with defaults."""
        cube_names = set()
        for _set_idx, bindings in binding_info.items():
            for b in bindings:
                if b["type"] == "sampled_cube_image":
                    cube_names.add(b["name"])

        default_tex = None
        default_cube = None
        default_brdf = None
        default_normal = None
        default_black = None
        _BLACK_TEX_NAMES = {"emissive_tex", "sheen_color_tex", "transmission_tex"}

        for _set_idx, bindings in binding_info.items():
            for b in bindings:
                name = b["name"]
                btype = b["type"]
                if btype in ("sampler", "sampled_image", "sampled_cube_image") and name not in resources:
                    if name in cube_names:
                        if default_cube is None:
                            default_cube = _create_default_cubemap(self.device)
                        resources[name] = default_cube
                    elif name == "normal_tex":
                        if default_normal is None:
                            default_normal = _create_default_normal_texture(self.device)
                        resources[name] = default_normal
                    elif name in _BLACK_TEX_NAMES:
                        if default_black is None:
                            default_black = _create_black_texture(self.device)
                        resources[name] = default_black
                    elif name == "brdf_lut":
                        if default_brdf is None:
                            default_brdf = _create_default_brdf_lut(self.device)
                        resources[name] = default_brdf
                    else:
                        if default_tex is None:
                            default_tex = _create_default_texture(self.device)
                        resources[name] = default_tex

                elif btype == "uniform_buffer" and name not in resources:
                    size = b.get("size", 64)
                    resources[name] = self.device.create_buffer_with_data(
                        data=bytes(size),
                        usage=wgpu.BufferUsage.UNIFORM,
                    )
                elif btype == "storage_buffer" and name not in resources:
                    size = b.get("size", 64)
                    resources[name] = self.device.create_buffer_with_data(
                        data=bytes(max(size, 64)),
                        usage=wgpu.BufferUsage.STORAGE,
                    )

    # -------------------------------------------------------------------
    # Scene loading and GPU upload
    # -------------------------------------------------------------------

    def _load_scene(self, scene_path: str):
        """Load a glTF/glb scene file using the standard engine loader."""
        from engine import load_scene, SceneData
        scene = load_scene(scene_path)
        return scene

    def _upload_scene(self, scene):
        """Upload scene geometry, materials, and textures to GPU.

        Returns dicts of GPU resources keyed by resource name.
        """
        from scene_utils import perspective, look_at

        device = self.device

        # Upload mesh vertex/index buffers
        self.vertex_buffers = []  # (vbo, stride, num_verts)
        self.index_buffers = []   # (ibo, num_indices)
        self.draw_materials = []  # material_index per mesh

        for mesh in scene.meshes:
            vdata = mesh.vertex_data
            vstride = mesh.vertex_stride

            # Pad vertex data to match pipeline stride if needed
            if vstride < self.gbuf_vertex_stride and vdata:
                vdata = self._pad_vertices(vdata, vstride, self.gbuf_vertex_stride,
                                           mesh.num_vertices)
                vstride = self.gbuf_vertex_stride

            vbo = device.create_buffer_with_data(
                data=vdata,
                usage=wgpu.BufferUsage.VERTEX,
            ) if vdata else None
            ibo = device.create_buffer_with_data(
                data=mesh.index_data,
                usage=wgpu.BufferUsage.INDEX,
            ) if mesh.index_data else None

            self.vertex_buffers.append((vbo, vstride, mesh.num_vertices))
            self.index_buffers.append((ibo, mesh.num_indices))
            self.draw_materials.append(mesh.material_index)

        # MVP UBO (192 bytes: model + view + projection)
        cam = scene.camera
        model = np.eye(4, dtype=np.float32)
        view = look_at(cam.eye, cam.target, cam.up)
        proj = perspective(cam.fov_y, self.width / self.height, cam.near, cam.far)
        mvp_data = model.tobytes() + view.tobytes() + proj.tobytes()
        self.mvp_buffer = device.create_buffer_with_data(
            data=mvp_data,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        # DeferredCamera UBO (80 bytes: inv_view_proj + view_pos)
        view_proj = proj @ view  # column-major
        inv_view_proj = np.linalg.inv(view_proj).astype(np.float32)
        deferred_cam_data = _pack_deferred_camera_ubo(inv_view_proj, cam.eye)
        self.deferred_camera_buffer = device.create_buffer_with_data(
            data=deferred_cam_data,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        # SceneLight UBO (16 bytes: view_pos vec3 + light_count int)
        light_count = len(scene.lights) if scene.lights else 1
        scene_light_data = struct.pack("3fi", float(cam.eye[0]), float(cam.eye[1]),
                                       float(cam.eye[2]), light_count)
        self.scene_light_buffer = device.create_buffer_with_data(
            data=scene_light_data,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        # Lights SSBO
        lights_data = _pack_lights_buffer(scene.lights if scene.lights else [])
        self.lights_ssbo = device.create_buffer_with_data(
            data=lights_data,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )

        # Per-material UBOs and textures
        self.per_material_ubos = []
        self.per_material_textures = []  # list of {name: (sampler, view)}
        for mat in scene.materials:
            ubo_data = _pack_material_ubo(mat)
            ubo = device.create_buffer_with_data(
                data=ubo_data,
                usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            )
            self.per_material_ubos.append(ubo)

            mat_textures = {}
            for tex_name, tex_data in mat.textures.items():
                sampler, view = _upload_texture(device, tex_data)
                mat_textures[tex_name] = (sampler, view)
            self.per_material_textures.append(mat_textures)

        # Load IBL
        self.ibl_textures = _auto_load_ibl(device)

        return scene

    @staticmethod
    def _pad_vertices(vertex_data: bytes, src_stride: int, dst_stride: int,
                      num_vertices: int) -> bytes:
        """Pad vertex data from src_stride to dst_stride per vertex."""
        if src_stride >= dst_stride:
            return vertex_data
        pad_size = dst_stride - src_stride
        if pad_size == 16:
            pad = struct.pack("4f", 1.0, 0.0, 0.0, 1.0)  # default tangent
        else:
            pad = b'\x00' * pad_size
        result = bytearray()
        for i in range(num_vertices):
            start = i * src_stride
            result.extend(vertex_data[start:start + src_stride])
            result.extend(pad)
        return bytes(result)

    # -------------------------------------------------------------------
    # G-buffer bind group creation (per-material)
    # -------------------------------------------------------------------

    def _create_gbuf_bind_groups(self, material_index: int) -> dict[int, wgpu.GPUBindGroup]:
        """Create G-buffer pass bind groups for a given material."""
        resources = {}

        # Set 0: MVP UBO
        resources["MVP"] = self.mvp_buffer

        # Set 1: Material UBO + textures
        if material_index < len(self.per_material_ubos):
            resources["Material"] = self.per_material_ubos[material_index]
        if material_index < len(self.per_material_textures):
            for name, tex_pair in self.per_material_textures[material_index].items():
                resources[name] = tex_pair

        # Fill missing textures and buffers with defaults
        self._fill_missing_resources(self.gbuf_binding_info, resources)

        return self._create_bind_groups(
            self.gbuf_bind_group_layouts, self.gbuf_binding_info, resources)

    # -------------------------------------------------------------------
    # Lighting bind group creation
    # -------------------------------------------------------------------

    def _create_lighting_bind_groups(self) -> dict[int, wgpu.GPUBindGroup]:
        """Create lighting pass bind groups (G-buffer textures + UBOs + IBL)."""
        resources = {}

        # UBOs
        resources["DeferredCamera"] = self.deferred_camera_buffer
        resources["SceneLight"] = self.scene_light_buffer

        # G-buffer textures: sampler + texture view pairs
        # The lighting shader uses split sampler/texture bindings.
        # We provide the G-buffer sampler for all G-buffer texture names.
        gb = self.gbuffer
        resources["gbuf_tex0"] = (gb.sampler, gb.rt_views[0])
        resources["gbuf_tex1"] = (gb.sampler, gb.rt_views[1])
        resources["gbuf_tex2"] = (gb.sampler, gb.rt_views[2])

        # Depth texture uses the depth-only sample view
        # We create a separate non-filtering sampler for depth since the layout
        # may require non-filtering for unfilterable-float textures.
        depth_sampler = self.device.create_sampler(
            mag_filter=wgpu.FilterMode.nearest,
            min_filter=wgpu.FilterMode.nearest,
            address_mode_u=wgpu.AddressMode.clamp_to_edge,
            address_mode_v=wgpu.AddressMode.clamp_to_edge,
        )
        resources["gbuf_depth"] = (depth_sampler, gb.depth_sample_view)

        # IBL textures
        for name, tex_pair in self.ibl_textures.items():
            resources[name] = tex_pair

        # Lights SSBO
        resources["lights"] = self.lights_ssbo

        # Fill missing resources (cubemaps, brdf_lut, etc.)
        self._fill_missing_resources(self.light_binding_info, resources)

        return self._create_bind_groups(
            self.light_bind_group_layouts, self.light_binding_info, resources)

    # -------------------------------------------------------------------
    # Rendering
    # -------------------------------------------------------------------

    def render(self, scene_source: str) -> np.ndarray:
        """Render a scene to an offscreen texture and return pixels.

        Args:
            scene_source: Path to a .glb/.gltf file, or a built-in scene name.

        Returns:
            np.ndarray of shape (height, width, 4) with uint8 RGBA pixels.
        """
        scene = self._load_scene(scene_source)
        self._upload_scene(scene)

        # Pre-create per-material G-buffer bind groups
        material_bind_groups = {}
        for i in range(len(scene.materials)):
            material_bind_groups[i] = self._create_gbuf_bind_groups(i)

        # Create lighting bind groups
        lighting_bind_groups = self._create_lighting_bind_groups()

        # Render target for final output
        output_texture = self.device.create_texture(
            size=(self.width, self.height, 1),
            format=_OUTPUT_FORMAT,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
        )
        output_view = output_texture.create_view()

        # Command encoder
        encoder = self.device.create_command_encoder()

        # --- Pass 1: G-buffer ---
        self._encode_gbuffer_pass(encoder, scene, material_bind_groups)

        # --- Pass 2: Lighting ---
        self._encode_lighting_pass(encoder, output_view, lighting_bind_groups)

        # --- Readback ---
        pixels = self._readback(encoder, output_texture)

        print(f"[info] Deferred render complete: {self.width}x{self.height}")
        return pixels

    def render_interactive(self, scene_source: str):
        """Open an interactive window for deferred rendering.

        Args:
            scene_source: Path to a .glb/.gltf file, or a built-in scene name.
        """
        try:
            from rendercanvas.glfw import RenderCanvas, loop
        except ImportError:
            print("Interactive mode requires: pip install glfw rendercanvas")
            print("Falling back to offscreen render + save.")
            pixels = self.render(scene_source)
            from render_harness import save_png
            save_png(pixels, Path("deferred_output.png"))
            return

        scene = self._load_scene(scene_source)
        self._upload_scene(scene)

        # Pre-create per-material G-buffer bind groups
        material_bind_groups = {}
        for i in range(len(scene.materials)):
            material_bind_groups[i] = self._create_gbuf_bind_groups(i)

        # Create lighting bind groups
        lighting_bind_groups = self._create_lighting_bind_groups()

        # Canvas setup
        canvas = RenderCanvas(
            title=f"Lux Deferred Renderer - {Path(scene_source).name}",
            size=(self.width, self.height),
        )
        context = canvas.get_context("wgpu")
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        present_format = context.get_preferred_format(adapter)
        context.configure(device=self.device, format=present_format)

        # If present format differs from our output format, we need a blit.
        # For simplicity, render the lighting pass directly to the swapchain.
        # Recreate lighting pipeline with swapchain format if needed.
        if present_format != _OUTPUT_FORMAT:
            self._recreate_lighting_pipeline_for_format(present_format)
            lighting_bind_groups = self._create_lighting_bind_groups()

        device = self.device

        def draw_frame():
            current_texture_view = context.get_current_texture().create_view()
            encoder = device.create_command_encoder()

            # Pass 1: G-buffer
            self._encode_gbuffer_pass(encoder, scene, material_bind_groups)

            # Pass 2: Lighting (render directly to swapchain)
            self._encode_lighting_pass(encoder, current_texture_view, lighting_bind_groups)

            device.queue.submit([encoder.finish()])

        canvas.request_draw(draw_frame)
        print(f"[info] Interactive deferred renderer started. Close window to exit.")
        loop.run()

    def _recreate_lighting_pipeline_for_format(self, color_format: wgpu.TextureFormat):
        """Recreate the lighting pipeline for a different output format (e.g. swapchain)."""
        self.light_pipeline = self.device.create_render_pipeline(
            layout=self.light_pipeline_layout,
            vertex=wgpu.VertexState(
                module=self.light_vert_mod,
                entry_point="main",
                buffers=[],
            ),
            primitive=wgpu.PrimitiveState(
                topology=wgpu.PrimitiveTopology.triangle_list,
            ),
            multisample=wgpu.MultisampleState(count=1, mask=0xFFFFFFFF),
            fragment=wgpu.FragmentState(
                module=self.light_frag_mod,
                entry_point="main",
                targets=[
                    wgpu.ColorTargetState(format=color_format),
                ],
            ),
        )

    def _encode_gbuffer_pass(self, encoder, scene, material_bind_groups: dict):
        """Encode the G-buffer render pass: draws scene meshes to MRT."""
        gb = self.gbuffer

        render_pass = encoder.begin_render_pass(
            color_attachments=[
                wgpu.RenderPassColorAttachment(
                    view=gb.rt_views[0],
                    load_op=wgpu.LoadOp.clear,
                    store_op=wgpu.StoreOp.store,
                    clear_value=(0.0, 0.0, 0.0, 0.0),
                ),
                wgpu.RenderPassColorAttachment(
                    view=gb.rt_views[1],
                    load_op=wgpu.LoadOp.clear,
                    store_op=wgpu.StoreOp.store,
                    clear_value=(0.0, 0.0, 0.0, 0.0),
                ),
                wgpu.RenderPassColorAttachment(
                    view=gb.rt_views[2],
                    load_op=wgpu.LoadOp.clear,
                    store_op=wgpu.StoreOp.store,
                    clear_value=(0.0, 0.0, 0.0, 0.0),
                ),
            ],
            depth_stencil_attachment=wgpu.RenderPassDepthStencilAttachment(
                view=gb.depth_view,
                depth_load_op=wgpu.LoadOp.clear,
                depth_store_op=wgpu.StoreOp.store,
                depth_clear_value=1.0,
            ),
        )

        render_pass.set_pipeline(self.gbuf_pipeline)

        current_material = -1

        for mesh_idx in range(len(self.vertex_buffers)):
            mat_idx = self.draw_materials[mesh_idx] if mesh_idx < len(self.draw_materials) else 0

            # Switch material bind groups when material changes
            if mat_idx != current_material:
                current_material = mat_idx
                bind_groups = material_bind_groups.get(mat_idx, material_bind_groups.get(0, {}))
                for set_idx in sorted(bind_groups.keys()):
                    render_pass.set_bind_group(set_idx, bind_groups[set_idx])

            vbo, stride, num_verts = self.vertex_buffers[mesh_idx]
            ibo, num_indices = self.index_buffers[mesh_idx]

            if vbo:
                render_pass.set_vertex_buffer(0, vbo)
            if ibo and num_indices > 0:
                render_pass.set_index_buffer(ibo, wgpu.IndexFormat.uint32)
                render_pass.draw_indexed(num_indices)
            elif vbo:
                render_pass.draw(num_verts)

        render_pass.end()

    def _encode_lighting_pass(self, encoder, output_view, lighting_bind_groups: dict):
        """Encode the lighting pass: fullscreen triangle reads G-buffer."""
        render_pass = encoder.begin_render_pass(
            color_attachments=[
                wgpu.RenderPassColorAttachment(
                    view=output_view,
                    load_op=wgpu.LoadOp.clear,
                    store_op=wgpu.StoreOp.store,
                    clear_value=(0.05, 0.05, 0.08, 1.0),
                ),
            ],
        )

        render_pass.set_pipeline(self.light_pipeline)
        for set_idx in sorted(lighting_bind_groups.keys()):
            render_pass.set_bind_group(set_idx, lighting_bind_groups[set_idx])

        # Fullscreen triangle: 3 vertices, no vertex buffer
        render_pass.draw(3)
        render_pass.end()

    def _readback(self, encoder, texture) -> np.ndarray:
        """Read pixels from a texture back to CPU."""
        bytes_per_row = self.width * 4
        bytes_per_row_aligned = (bytes_per_row + 255) & ~255
        readback_buffer = self.device.create_buffer(
            size=bytes_per_row_aligned * self.height,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )
        encoder.copy_texture_to_buffer(
            wgpu.TexelCopyTextureInfo(texture=texture),
            wgpu.TexelCopyBufferInfo(
                buffer=readback_buffer, offset=0,
                bytes_per_row=bytes_per_row_aligned, rows_per_image=self.height,
            ),
            (self.width, self.height, 1),
        )
        self.device.queue.submit([encoder.finish()])

        readback_buffer.map_sync(wgpu.MapMode.READ)
        raw = readback_buffer.read_mapped()
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(self.height, bytes_per_row_aligned)
        arr = arr[:, :self.width * 4].reshape(self.height, self.width, 4).copy()
        readback_buffer.unmap()
        return arr

    def cleanup(self):
        """Release GPU resources."""
        # wgpu-py handles resource cleanup via Python GC, but we can clear references.
        self.gbuffer = None
        self.vertex_buffers = []
        self.index_buffers = []
        self.per_material_ubos = []
        self.per_material_textures = []
        self.ibl_textures = {}
        print("[info] DeferredRenderer resources released.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Lux deferred rendering engine (Python/wgpu)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python playground/deferred_engine.py examples/deferred_basic --scene assets/DamagedHelmet.glb
    python playground/deferred_engine.py examples/deferred_basic --scene assets/DamagedHelmet.glb --interactive
    python playground/deferred_engine.py examples/deferred_basic --scene sphere -o output.png
""",
    )
    parser.add_argument("pipeline_base", type=str,
                        help="Shader pipeline base path (e.g. examples/deferred_basic)")
    parser.add_argument("--scene", type=str, default="sphere",
                        help="Scene to render: .glb/.gltf path or 'sphere' (default: sphere)")
    parser.add_argument("-o", "--output", type=str, default="deferred_output.png",
                        help="Output PNG path (default: deferred_output.png)")
    parser.add_argument("--interactive", action="store_true",
                        help="Open interactive window instead of saving PNG")
    parser.add_argument("--width", type=int, default=1024, help="Render width (default: 1024)")
    parser.add_argument("--height", type=int, default=768, help="Render height (default: 768)")

    args = parser.parse_args()

    renderer = DeferredRenderer(args.pipeline_base, args.width, args.height)

    if args.interactive:
        renderer.render_interactive(args.scene)
    else:
        pixels = renderer.render(args.scene)
        from PIL import Image
        img = Image.fromarray(pixels, "RGBA")
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(output_path))
        print(f"[info] Saved deferred render to {output_path}")

    renderer.cleanup()


if __name__ == "__main__":
    main()
