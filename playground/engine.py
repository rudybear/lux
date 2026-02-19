"""Lux rendering engine: scene/pipeline separation with reflection-driven rendering.

Usage:
    python -m playground.engine --scene <source> --pipeline <shader_base> [options]

Scenes:
    sphere      — procedural UV sphere + orbit camera + directional light
    fullscreen  — fullscreen quad (screen-space effects)
    triangle    — colored triangle
    *.glb/*.gltf — glTF scene

Pipelines:
    Path to compiled shader base (finds .vert.spv, .frag.spv, .lux.json)
    Auto-detects render path from available shader stages.
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import wgpu
from PIL import Image

# Ensure playground/ is on sys.path so lazy imports of sibling modules work
# regardless of whether we're run as `python -m playground.engine` or directly.
_PLAYGROUND_DIR = str(Path(__file__).resolve().parent)
if _PLAYGROUND_DIR not in sys.path:
    sys.path.insert(0, _PLAYGROUND_DIR)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MeshData:
    """CPU-side mesh data ready for GPU upload."""
    vertex_data: bytes
    index_data: bytes
    num_vertices: int
    num_indices: int
    vertex_stride: int = 32  # default: pos(3)+normal(3)+uv(2) = 32 bytes
    material_index: int = 0


@dataclass
class MaterialData:
    """Material properties and texture images."""
    base_color: tuple = (1.0, 1.0, 1.0, 1.0)
    metallic: float = 0.0
    roughness: float = 1.0
    emissive: tuple = (0.0, 0.0, 0.0)
    textures: dict = field(default_factory=dict)  # name -> np.ndarray RGBA uint8


@dataclass
class CameraData:
    """Camera parameters."""
    eye: np.ndarray = field(default_factory=lambda: np.array([0, 0, 3], dtype=np.float32))
    target: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    up: np.ndarray = field(default_factory=lambda: np.array([0, 1, 0], dtype=np.float32))
    fov_y: float = math.radians(45.0)
    near: float = 0.1
    far: float = 100.0


@dataclass
class LightData:
    """Light source parameters."""
    direction: np.ndarray = field(default_factory=lambda: np.array([1, 0.8, 0.6], dtype=np.float32))
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    color: tuple = (1.0, 1.0, 1.0)
    intensity: float = 1.0


@dataclass
class SceneData:
    """Complete scene description, independent of rendering pipeline."""
    meshes: list[MeshData] = field(default_factory=list)
    materials: list[MaterialData] = field(default_factory=list)
    camera: CameraData = field(default_factory=CameraData)
    lights: list[LightData] = field(default_factory=list)
    render_path: str = "raster"  # hint: "raster", "fullscreen"


@dataclass
class GPUScene:
    """Scene resources uploaded to GPU, independent of any pipeline."""
    vertex_buffers: list = field(default_factory=list)  # list of (GPUBuffer, stride, num_verts)
    index_buffers: list = field(default_factory=list)    # list of (GPUBuffer, num_indices)
    textures: dict = field(default_factory=dict)         # name -> (sampler, texture_view)
    uniform_buffers: dict = field(default_factory=dict)  # name -> GPUBuffer


# ---------------------------------------------------------------------------
# Scene loading
# ---------------------------------------------------------------------------

def load_scene(source: str) -> SceneData:
    """Load a scene from a source specifier."""
    if source == "sphere":
        return _builtin_sphere()
    elif source == "fullscreen":
        return _builtin_fullscreen()
    elif source == "triangle":
        return _builtin_triangle()
    elif source.endswith((".glb", ".gltf")):
        return _load_gltf_scene(source)
    else:
        raise ValueError(f"Unknown scene source: {source}")


def _builtin_sphere() -> SceneData:
    from scene_utils import generate_sphere, generate_procedural_texture
    verts, indices = generate_sphere(stacks=32, slices=32)
    vertex_data = struct.pack(f"{len(verts)}f", *verts)
    index_data = struct.pack(f"{len(indices)}I", *indices)
    mesh = MeshData(
        vertex_data=vertex_data,
        index_data=index_data,
        num_vertices=len(verts) // 8,
        num_indices=len(indices),
        vertex_stride=32,
    )
    mat = MaterialData()
    mat.textures["albedo_tex"] = generate_procedural_texture(512)
    return SceneData(
        meshes=[mesh],
        materials=[mat],
        camera=CameraData(),
        lights=[LightData()],
        render_path="raster",
    )


def _builtin_fullscreen() -> SceneData:
    return SceneData(render_path="fullscreen")


def _builtin_triangle() -> SceneData:
    from scene_utils import generate_triangle
    vertex_data, num_verts, stride = generate_triangle()
    mesh = MeshData(
        vertex_data=vertex_data,
        index_data=b"",
        num_vertices=num_verts,
        num_indices=0,
        vertex_stride=stride,
    )
    return SceneData(meshes=[mesh], render_path="raster")


def _load_gltf_scene(path: str) -> SceneData:
    from gltf_loader import load_gltf, flatten_scene
    gltf_scene = load_gltf(Path(path))
    draw_items = flatten_scene(gltf_scene)

    scene = SceneData(render_path="raster")

    for mesh in gltf_scene.meshes:
        scene.meshes.append(MeshData(
            vertex_data=mesh.vertex_data,
            index_data=mesh.index_data,
            num_vertices=mesh.num_vertices,
            num_indices=mesh.num_indices,
            vertex_stride=getattr(mesh, 'vertex_stride', 32),
            material_index=mesh.material_index,
        ))

    for mat in gltf_scene.materials:
        mdata = MaterialData(
            base_color=mat.base_color,
            metallic=mat.metallic,
            roughness=mat.roughness,
            emissive=mat.emissive,
        )
        if mat.base_color_texture is not None:
            mdata.textures["base_color_tex"] = mat.base_color_texture
        if mat.normal_texture is not None:
            mdata.textures["normal_tex"] = mat.normal_texture
        if mat.metallic_roughness_texture is not None:
            mdata.textures["metallic_roughness_tex"] = mat.metallic_roughness_texture
        if mat.occlusion_texture is not None:
            mdata.textures["occlusion_tex"] = mat.occlusion_texture
        if mat.emissive_texture is not None:
            mdata.textures["emissive_tex"] = mat.emissive_texture
        scene.materials.append(mdata)

    # Extract camera from glTF if available
    if gltf_scene.cameras:
        cam = gltf_scene.cameras[0]
        scene.camera = CameraData(fov_y=cam.fov_y, near=cam.near, far=cam.far)

    # Extract lights
    for light in gltf_scene.lights:
        scene.lights.append(LightData(
            direction=light.direction.copy(),
            position=light.position.copy(),
            color=light.color,
            intensity=light.intensity,
        ))

    if not scene.lights:
        scene.lights.append(LightData())

    return scene


# ---------------------------------------------------------------------------
# Pipeline resolution
# ---------------------------------------------------------------------------

def resolve_pipeline(pipeline_arg: Optional[str], scene_source: str) -> str:
    """If no --pipeline given, pick a sensible default for the scene type."""
    if pipeline_arg:
        return pipeline_arg
    if scene_source.endswith((".glb", ".gltf")):
        return "examples/gltf_pbr"
    elif scene_source == "fullscreen":
        raise ValueError("--pipeline is required for fullscreen scenes")
    elif scene_source == "triangle":
        return "examples/hello_triangle"
    else:
        return "examples/pbr_basic"


def detect_render_path(pipeline_base: str) -> str:
    """Auto-detect rendering path from available shader files."""
    base = Path(pipeline_base)
    if base.with_suffix(".rgen.spv").exists():
        return "rt"
    elif base.with_suffix(".vert.spv").exists() and base.with_suffix(".frag.spv").exists():
        return "raster"
    elif base.with_suffix(".frag.spv").exists():
        return "fullscreen"
    else:
        raise FileNotFoundError(
            f"No shader files found for pipeline base: {pipeline_base}\n"
            f"Expected: {base}.vert.spv + {base}.frag.spv, or {base}.frag.spv, or {base}.rgen.spv"
        )


# ---------------------------------------------------------------------------
# GPU upload (Phase 1: scene → GPU, independent of pipeline)
# ---------------------------------------------------------------------------

def _pad_vertices(vertex_data: bytes, src_stride: int, dst_stride: int,
                   num_vertices: int) -> bytes:
    """Pad vertex data from src_stride to dst_stride per vertex.
    Extra bytes are filled with default tangent (1,0,0,1) if padding 32->48."""
    if src_stride >= dst_stride:
        return vertex_data
    pad_size = dst_stride - src_stride
    # Default tangent vec4(1,0,0,1) for 32->48 byte padding
    if pad_size == 16:
        pad = struct.pack("4f", 1.0, 0.0, 0.0, 1.0)
    else:
        pad = b'\x00' * pad_size
    result = bytearray()
    for i in range(num_vertices):
        start = i * src_stride
        result.extend(vertex_data[start:start + src_stride])
        result.extend(pad)
    return bytes(result)


def upload_scene(device: wgpu.GPUDevice, scene: SceneData, width: int, height: int,
                 pipeline_stride: int = 0) -> GPUScene:
    """Upload scene resources to GPU. Independent of any pipeline."""
    gpu = GPUScene()

    # Upload meshes (with optional vertex padding)
    for mesh in scene.meshes:
        vdata = mesh.vertex_data
        vstride = mesh.vertex_stride
        if pipeline_stride > 0 and vstride < pipeline_stride and vdata:
            vdata = _pad_vertices(vdata, vstride, pipeline_stride, mesh.num_vertices)
            vstride = pipeline_stride
        vbo = device.create_buffer_with_data(
            data=vdata,
            usage=wgpu.BufferUsage.VERTEX,
        ) if vdata else None
        ibo = device.create_buffer_with_data(
            data=mesh.index_data,
            usage=wgpu.BufferUsage.INDEX,
        ) if mesh.index_data else None
        gpu.vertex_buffers.append((vbo, vstride, mesh.num_vertices))
        gpu.index_buffers.append((ibo, mesh.num_indices))

    # Upload textures from materials
    for mat in scene.materials:
        for tex_name, tex_data in mat.textures.items():
            if tex_name in gpu.textures:
                continue
            sampler, view = _upload_texture(device, tex_data)
            gpu.textures[tex_name] = (sampler, view)

    # Create MVP uniform buffer
    from scene_utils import perspective, look_at
    cam = scene.camera
    model = np.eye(4, dtype=np.float32)
    view = look_at(cam.eye, cam.target, cam.up)
    proj = perspective(cam.fov_y, width / height, cam.near, cam.far)
    mvp_data = model.tobytes() + view.tobytes() + proj.tobytes()
    gpu.uniform_buffers["MVP"] = device.create_buffer_with_data(
        data=mvp_data,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )

    # Create Light uniform buffer (std140: vec3 padded to 16 bytes each)
    if scene.lights:
        light = scene.lights[0]
        light_dir = light.direction / max(np.linalg.norm(light.direction), 1e-6)
        light_data = struct.pack("3f", *light_dir) + struct.pack("f", 0.0)
        light_data += struct.pack("3f", *cam.eye) + struct.pack("f", 0.0)
        gpu.uniform_buffers["Light"] = device.create_buffer_with_data(
            data=light_data,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        gpu.uniform_buffers["Lighting"] = gpu.uniform_buffers["Light"]

    return gpu


def _upload_texture(device: wgpu.GPUDevice, data: np.ndarray):
    """Upload a texture image to GPU. Returns (sampler, texture_view)."""
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
    sampler = device.create_sampler(
        mag_filter=wgpu.FilterMode.linear,
        min_filter=wgpu.FilterMode.linear,
    )
    return sampler, view


def _create_default_texture(device: wgpu.GPUDevice):
    """Create a 1x1 white RGBA texture for missing sampler bindings."""
    data = np.array([[255, 255, 255, 255]], dtype=np.uint8).reshape(1, 1, 4)
    return _upload_texture(device, data)


def _create_default_cubemap(device: wgpu.GPUDevice, face_size: int = 1):
    """Create a 1x1 black cubemap for missing cubemap bindings."""
    return _upload_cubemap_f16(device, face_size, num_faces=6,
                               data=np.zeros((6, face_size, face_size, 4), dtype=np.float16))


def _upload_cubemap_f16(device: wgpu.GPUDevice, face_size: int, num_faces: int,
                        data: np.ndarray):
    """Upload float16 RGBA cubemap faces. data shape: (6, h, w, 4) float16.
    Returns (sampler, cube_texture_view)."""
    texture = device.create_texture(
        size=(face_size, face_size, 6),
        dimension="2d",
        format=wgpu.TextureFormat.rgba16float,
        usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
    )
    bytes_per_row = face_size * 4 * 2  # 4 components × 2 bytes per float16
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


def _load_ibl_assets(device: wgpu.GPUDevice, ibl_name: str) -> dict:
    """Load preprocessed IBL assets and upload cubemaps. Returns name -> (sampler, view) dict."""
    import json
    assets_dir = Path(__file__).parent / "assets" / "ibl" / ibl_name
    manifest_path = assets_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"[warn] IBL assets not found at {assets_dir}, using defaults")
        return {}

    with open(manifest_path) as f:
        manifest = json.load(f)

    result = {}

    # Helper: read nested manifest format from preprocess_ibl.py
    spec_info = manifest.get("specular", {})
    irr_info = manifest.get("irradiance", {})
    brdf_info = manifest.get("brdf_lut", {})

    # Load specular cubemap (with mips)
    spec_path = assets_dir / "specular.bin"
    if spec_path.exists():
        spec_data = np.frombuffer(spec_path.read_bytes(), dtype=np.float16)
        face_size = spec_info.get("face_size", manifest.get("specular_face_size", 256))
        mip_count = spec_info.get("mip_count", manifest.get("specular_mip_count", 5))
        # Create texture with mips
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

    # Load irradiance cubemap
    irr_path = assets_dir / "irradiance.bin"
    if irr_path.exists():
        irr_size = irr_info.get("face_size", manifest.get("irradiance_face_size", 32))
        irr_data = np.frombuffer(irr_path.read_bytes(), dtype=np.float16)
        irr_data = irr_data.reshape(6, irr_size, irr_size, 4)
        result["env_irradiance"] = _upload_cubemap_f16(device, irr_size, 6, irr_data)

    # Load BRDF LUT (2D texture, float16 RG → pad to RGBA)
    brdf_path = assets_dir / "brdf_lut.bin"
    if brdf_path.exists():
        lut_size = brdf_info.get("size", manifest.get("brdf_lut_size", 512))
        brdf_raw = np.frombuffer(brdf_path.read_bytes(), dtype=np.float16)
        brdf_rg = brdf_raw.reshape(lut_size, lut_size, 2)
        # Pad RG to RGBA
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


# ---------------------------------------------------------------------------
# Pipeline creation (Phase 2: from reflection, independent of scene)
# ---------------------------------------------------------------------------

def create_pipeline(
    device: wgpu.GPUDevice,
    pipeline_base: str,
    render_path: str,
    color_format=wgpu.TextureFormat.rgba8unorm,
):
    """Create a GPU pipeline from reflection metadata."""
    from reflected_pipeline import ReflectedPipeline, load_reflection
    from render_harness import load_shader_module

    base = Path(pipeline_base)

    if render_path == "fullscreen":
        frag_spv = base.with_suffix(".frag.spv")
        frag_json = base.with_suffix(".frag.json")

        frag_module = load_shader_module(device, frag_spv)
        frag_reflection = load_reflection(frag_json) if frag_json.exists() else {}

        # Use built-in fullscreen vertex shader (WGSL)
        vert_wgsl = """
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};
@vertex
fn main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    let p = positions[idx];
    var out: VertexOutput;
    out.position = vec4<f32>(p, 0.0, 1.0);
    out.uv = vec2<f32>(p.x * 0.5 + 0.5, 1.0 - (p.y * 0.5 + 0.5));
    return out;
}
"""
        vert_module = device.create_shader_module(code=vert_wgsl)

        # Create pipeline without depth, no vertex buffers
        pipeline = device.create_render_pipeline(
            layout="auto",
            vertex=wgpu.VertexState(module=vert_module, entry_point="main", buffers=[]),
            primitive=wgpu.PrimitiveState(topology=wgpu.PrimitiveTopology.triangle_list),
            multisample=wgpu.MultisampleState(count=1, mask=0xFFFFFFFF),
            fragment=wgpu.FragmentState(
                module=frag_module, entry_point="main",
                targets=[wgpu.ColorTargetState(format=color_format)],
            ),
        )
        return pipeline, render_path, frag_reflection, {}

    elif render_path == "raster":
        vert_spv = base.with_suffix(".vert.spv")
        frag_spv = base.with_suffix(".frag.spv")
        vert_json = base.with_suffix(".vert.json")
        frag_json = base.with_suffix(".frag.json")

        vert_module = load_shader_module(device, vert_spv)
        frag_module = load_shader_module(device, frag_spv)
        vert_reflection = load_reflection(vert_json) if vert_json.exists() else {}
        frag_reflection = load_reflection(frag_json) if frag_json.exists() else {}

        reflected = ReflectedPipeline(
            device, vert_reflection, frag_reflection,
            vert_module, frag_module, color_format,
        )
        return reflected.pipeline, render_path, frag_reflection, {
            "reflected": reflected,
            "vert_reflection": vert_reflection,
            "frag_reflection": frag_reflection,
        }

    else:
        raise ValueError(f"Unsupported render path: {render_path}")


# ---------------------------------------------------------------------------
# Bind scene to pipeline (Phase 3)
# ---------------------------------------------------------------------------

def bind_scene_to_pipeline(
    device: wgpu.GPUDevice,
    gpu_scene: GPUScene,
    pipeline_info: dict,
) -> dict:
    """Map scene resources to pipeline descriptors by name.
    Returns dict of set_idx -> GPUBindGroup.
    """
    reflected = pipeline_info.get("reflected")
    if reflected is None:
        return {}

    # Build resource map: name -> resource
    resources = {}
    for name, buf in gpu_scene.uniform_buffers.items():
        resources[name] = buf
    for name, (sampler, view) in gpu_scene.textures.items():
        resources[name] = (sampler, view)

    # Identify which names need cube textures
    cube_names = set()
    for set_idx, bindings in reflected._binding_info.items():
        for b in bindings:
            if b["type"] == "sampled_cube_image":
                cube_names.add(b["name"])

    # Fill missing textures with defaults (cube or 2D as appropriate)
    default_tex = None
    default_cube = None
    for set_idx, bindings in reflected._binding_info.items():
        for b in bindings:
            if b["type"] in ("sampler", "sampled_image", "sampled_cube_image") and b["name"] not in resources:
                if b["name"] in cube_names:
                    if default_cube is None:
                        default_cube = _create_default_cubemap(device)
                    resources[b["name"]] = default_cube
                else:
                    if default_tex is None:
                        default_tex = _create_default_texture(device)
                    resources[b["name"]] = default_tex

    # Fill missing uniform buffers with zero-filled buffers
    for set_idx, bindings in reflected._binding_info.items():
        for b in bindings:
            if b["type"] == "uniform_buffer" and b["name"] not in resources:
                size = b.get("size", 64)
                resources[b["name"]] = device.create_buffer_with_data(
                    data=bytes(size),
                    usage=wgpu.BufferUsage.UNIFORM,
                )

    return reflected.create_bind_groups(device, resources)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render(
    scene_source: str,
    pipeline_base: str,
    output: str = "output.png",
    width: int = 512,
    height: int = 512,
    ibl_name: str = "",
) -> np.ndarray:
    """Main rendering entry point: load scene, create pipeline, bind, render."""

    # Load scene
    scene = load_scene(scene_source)

    # Detect render path
    render_path = detect_render_path(pipeline_base)

    # Override render path based on scene hint
    if scene.render_path == "fullscreen" and render_path == "fullscreen":
        pass  # consistent
    elif scene.render_path == "fullscreen":
        render_path = "fullscreen"

    # GPU setup
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    if adapter is None:
        raise RuntimeError("No suitable GPU adapter found")
    device = adapter.request_device_sync()

    # Peek at pipeline vertex stride from reflection for vertex padding
    import json
    vert_json = Path(pipeline_base).with_suffix(".vert.json")
    pipeline_stride = 0
    if vert_json.exists():
        with open(vert_json) as f:
            vert_refl = json.load(f)
        pipeline_stride = vert_refl.get("vertex_stride", 0)

    # Phase 1: Upload scene to GPU
    print(f"Uploading scene '{scene_source}' to GPU...")
    gpu_scene = upload_scene(device, scene, width, height, pipeline_stride=pipeline_stride)

    # Load IBL assets if available
    if ibl_name:
        ibl_textures = _load_ibl_assets(device, ibl_name)
        for name, tex in ibl_textures.items():
            gpu_scene.textures[name] = tex
    else:
        # Auto-detect: use first available IBL directory
        ibl_dir = Path(__file__).parent / "assets" / "ibl"
        if ibl_dir.exists():
            for d in sorted(ibl_dir.iterdir()):
                if d.is_dir() and (d / "manifest.json").exists():
                    ibl_textures = _load_ibl_assets(device, d.name)
                    for name, tex in ibl_textures.items():
                        gpu_scene.textures[name] = tex
                    break

    # Phase 2: Create pipeline from reflection
    print(f"Creating pipeline from '{pipeline_base}'...")
    gpu_pipeline, path, frag_refl, pipeline_info = create_pipeline(
        device, pipeline_base, render_path,
    )

    # Phase 3: Bind scene to pipeline
    bind_groups = bind_scene_to_pipeline(device, gpu_scene, pipeline_info)

    # Execute render pass
    print(f"Rendering {width}x{height}...")
    pixels = _execute_render(
        device, gpu_pipeline, render_path, bind_groups,
        gpu_scene, width, height,
    )

    # Save output
    from render_harness import save_png
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_png(pixels, output_path)
    print(f"Saved {width}x{height} render to {output_path}")

    return pixels


def _execute_render(
    device, pipeline, render_path, bind_groups, gpu_scene, width, height,
) -> np.ndarray:
    """Execute the actual render pass and readback pixels."""

    color_format = wgpu.TextureFormat.rgba8unorm

    # Render target
    texture = device.create_texture(
        size=(width, height, 1),
        format=color_format,
        usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
    )
    texture_view = texture.create_view()

    encoder = device.create_command_encoder()

    if render_path == "fullscreen":
        render_pass = encoder.begin_render_pass(
            color_attachments=[wgpu.RenderPassColorAttachment(
                view=texture_view,
                load_op=wgpu.LoadOp.clear,
                store_op=wgpu.StoreOp.store,
                clear_value=(0.0, 0.0, 0.0, 1.0),
            )],
        )
        render_pass.set_pipeline(pipeline)
        render_pass.draw(3)
        render_pass.end()

    elif render_path == "raster":
        # Depth buffer
        depth_texture = device.create_texture(
            size=(width, height, 1),
            format=wgpu.TextureFormat.depth24plus,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
        )
        depth_view = depth_texture.create_view()

        render_pass = encoder.begin_render_pass(
            color_attachments=[wgpu.RenderPassColorAttachment(
                view=texture_view,
                load_op=wgpu.LoadOp.clear,
                store_op=wgpu.StoreOp.store,
                clear_value=(0.05, 0.05, 0.08, 1.0),
            )],
            depth_stencil_attachment=wgpu.RenderPassDepthStencilAttachment(
                view=depth_view,
                depth_load_op=wgpu.LoadOp.clear,
                depth_store_op=wgpu.StoreOp.store,
                depth_clear_value=1.0,
            ),
        )
        render_pass.set_pipeline(pipeline)

        # Set bind groups
        for set_idx in sorted(bind_groups.keys()):
            render_pass.set_bind_group(set_idx, bind_groups[set_idx])

        # Draw each mesh
        for i, (vbo, stride, num_verts) in enumerate(gpu_scene.vertex_buffers):
            ibo, num_indices = gpu_scene.index_buffers[i]
            if vbo:
                render_pass.set_vertex_buffer(0, vbo)
            if ibo and num_indices > 0:
                render_pass.set_index_buffer(ibo, wgpu.IndexFormat.uint32)
                render_pass.draw_indexed(num_indices)
            elif vbo:
                render_pass.draw(num_verts)

        render_pass.end()

    # Readback
    bytes_per_row = width * 4
    bytes_per_row_aligned = (bytes_per_row + 255) & ~255
    readback_buffer = device.create_buffer(
        size=bytes_per_row_aligned * height,
        usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
    )
    encoder.copy_texture_to_buffer(
        wgpu.TexelCopyTextureInfo(texture=texture),
        wgpu.TexelCopyBufferInfo(
            buffer=readback_buffer, offset=0,
            bytes_per_row=bytes_per_row_aligned, rows_per_image=height,
        ),
        (width, height, 1),
    )
    device.queue.submit([encoder.finish()])

    readback_buffer.map_sync(wgpu.MapMode.READ)
    raw = readback_buffer.read_mapped()
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(height, bytes_per_row_aligned)
    arr = arr[:, :width * 4].reshape(height, width, 4).copy()
    readback_buffer.unmap()
    return arr


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Lux rendering engine")
    parser.add_argument("--scene", required=True, help="Scene source: sphere, fullscreen, triangle, or path to .glb/.gltf")
    parser.add_argument("--pipeline", default=None, help="Path to compiled shader base (finds .spv + .json)")
    parser.add_argument("--output", "-o", default="output.png", help="Output PNG path")
    parser.add_argument("--width", type=int, default=512, help="Render width")
    parser.add_argument("--height", type=int, default=512, help="Render height")
    parser.add_argument("--ibl", default="", help="IBL environment name (from playground/assets/ibl/)")
    args = parser.parse_args()

    pipeline_base = resolve_pipeline(args.pipeline, args.scene)
    pixels = render(args.scene, pipeline_base, args.output, args.width, args.height,
                    ibl_name=args.ibl)

    # Stats
    non_black = (pixels[:, :, :3].sum(axis=2) > 10).sum()
    total = pixels.shape[0] * pixels.shape[1]
    print(f"Coverage: {non_black / total * 100:.1f}%")


if __name__ == "__main__":
    main()
