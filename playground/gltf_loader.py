"""glTF 2.0 scene loader for the Lux playground (Python/wgpu).

Loads .glb/.gltf files and produces GPU-ready mesh data, material tables,
node hierarchies, cameras, and lights (via KHR_lights_punctual).

Dependencies:
    pip install pygltflib numpy Pillow
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# ===========================================================================
# Data structures
# ===========================================================================

@dataclass
class GltfMesh:
    """GPU-ready mesh data."""
    name: str
    vertex_data: bytes       # interleaved vertex data
    index_data: bytes        # uint32 indices
    num_vertices: int
    num_indices: int
    material_index: int = 0
    has_tangents: bool = False
    vertex_stride: int = 32  # 32 = pos+normal+uv, 48 = pos+normal+uv+tangent


@dataclass
class GltfMaterial:
    """PBR material properties."""
    name: str
    base_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    metallic: float = 0.0
    roughness: float = 1.0
    emissive: tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_color_texture: Optional[np.ndarray] = None  # RGBA uint8 image
    normal_texture: Optional[np.ndarray] = None
    metallic_roughness_texture: Optional[np.ndarray] = None
    occlusion_texture: Optional[np.ndarray] = None
    emissive_texture: Optional[np.ndarray] = None
    alpha_mode: str = "OPAQUE"
    alpha_cutoff: float = 0.5
    double_sided: bool = False


@dataclass
class GltfNode:
    """Scene graph node."""
    name: str
    transform: np.ndarray          # 4x4 local transform
    world_transform: np.ndarray    # 4x4 world transform (computed during traversal)
    mesh_index: int = -1
    material_index: int = -1
    camera_index: int = -1
    light_index: int = -1
    children: list[int] = field(default_factory=list)
    parent: int = -1


@dataclass
class GltfCamera:
    """Camera parameters."""
    name: str
    type: str = "perspective"  # "perspective" or "orthographic"
    fov_y: float = math.radians(60.0)
    aspect: float = 1.0
    near: float = 0.01
    far: float = 1000.0
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    direction: np.ndarray = field(default_factory=lambda: np.array([0, 0, -1], dtype=np.float32))


@dataclass
class GltfLight:
    """Light source (KHR_lights_punctual compatible)."""
    name: str
    type: str = "directional"  # "directional", "point", "spot"
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    intensity: float = 1.0
    range: float = 0.0  # 0 = infinite
    inner_cone_angle: float = 0.0  # spot only
    outer_cone_angle: float = math.pi / 4  # spot only
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    direction: np.ndarray = field(default_factory=lambda: np.array([0, 0, -1], dtype=np.float32))


@dataclass
class GltfScene:
    """Complete loaded scene."""
    meshes: list[GltfMesh] = field(default_factory=list)
    materials: list[GltfMaterial] = field(default_factory=list)
    nodes: list[GltfNode] = field(default_factory=list)
    cameras: list[GltfCamera] = field(default_factory=list)
    lights: list[GltfLight] = field(default_factory=list)
    root_nodes: list[int] = field(default_factory=list)


@dataclass
class DrawItem:
    """Flat draw list item produced by scene traversal."""
    world_transform: np.ndarray
    mesh_index: int
    material_index: int


# ===========================================================================
# Scene traversal
# ===========================================================================

def flatten_scene(scene: GltfScene) -> list[DrawItem]:
    """Traverse the scene graph and produce a flat draw list sorted by material."""
    items = []

    def _traverse(node_idx: int, parent_world: np.ndarray):
        node = scene.nodes[node_idx]
        # Row-vector convention: child_world = child_local @ parent_world
        node.world_transform = node.transform @ parent_world

        if node.mesh_index >= 0:
            mesh = scene.meshes[node.mesh_index]
            items.append(DrawItem(
                world_transform=node.world_transform.copy(),
                mesh_index=node.mesh_index,
                material_index=mesh.material_index,
            ))

        for child_idx in node.children:
            _traverse(child_idx, node.world_transform)

    identity = np.eye(4, dtype=np.float32)
    for root_idx in scene.root_nodes:
        _traverse(root_idx, identity)

    # Sort by material to minimize pipeline switches
    items.sort(key=lambda d: d.material_index)
    return items


# ===========================================================================
# glTF loading
# ===========================================================================

def load_gltf(path: Path) -> GltfScene:
    """Load a .glb or .gltf file into a GltfScene.

    Uses pygltflib for parsing. Falls back to a minimal loader if unavailable.
    """
    try:
        from pygltflib import GLTF2
    except ImportError:
        raise ImportError("pygltflib is required for glTF loading: pip install pygltflib")

    gltf = GLTF2().load(str(path))
    scene = GltfScene()

    # --- Load materials ---
    if gltf.materials:
        for mat in gltf.materials:
            gmat = GltfMaterial(name=mat.name or "unnamed")
            pbr = mat.pbrMetallicRoughness
            if pbr:
                bc = pbr.baseColorFactor or [1, 1, 1, 1]
                gmat.base_color = tuple(bc)
                gmat.metallic = pbr.metallicFactor if pbr.metallicFactor is not None else 0.0
                gmat.roughness = pbr.roughnessFactor if pbr.roughnessFactor is not None else 1.0
            if mat.emissiveFactor:
                gmat.emissive = tuple(mat.emissiveFactor)
            gmat.alpha_mode = mat.alphaMode or "OPAQUE"
            gmat.alpha_cutoff = mat.alphaCutoff if mat.alphaCutoff is not None else 0.5
            gmat.double_sided = mat.doubleSided or False
            # Load texture images
            if pbr and pbr.baseColorTexture:
                gmat.base_color_texture = _load_texture_image(gltf, pbr.baseColorTexture.index)
            if pbr and pbr.metallicRoughnessTexture:
                gmat.metallic_roughness_texture = _load_texture_image(gltf, pbr.metallicRoughnessTexture.index)
            if mat.normalTexture:
                gmat.normal_texture = _load_texture_image(gltf, mat.normalTexture.index)
            if mat.occlusionTexture:
                gmat.occlusion_texture = _load_texture_image(gltf, mat.occlusionTexture.index)
            if mat.emissiveTexture:
                gmat.emissive_texture = _load_texture_image(gltf, mat.emissiveTexture.index)
            scene.materials.append(gmat)
    else:
        scene.materials.append(GltfMaterial(name="default"))

    # --- Load meshes ---
    if gltf.meshes:
        for mesh in gltf.meshes:
            for prim in mesh.primitives:
                gmesh = _load_primitive(gltf, prim, mesh.name or "unnamed")
                scene.meshes.append(gmesh)

    # --- Load nodes ---
    if gltf.nodes:
        for i, node in enumerate(gltf.nodes):
            transform = _node_transform(node)
            gnode = GltfNode(
                name=node.name or f"node_{i}",
                transform=transform,
                world_transform=np.eye(4, dtype=np.float32),
                mesh_index=node.mesh if node.mesh is not None else -1,
                camera_index=node.camera if node.camera is not None else -1,
                children=list(node.children) if node.children else [],
            )
            scene.nodes.append(gnode)

        # Set parent indices
        for i, node in enumerate(scene.nodes):
            for child_idx in node.children:
                if 0 <= child_idx < len(scene.nodes):
                    scene.nodes[child_idx].parent = i

    # --- Root nodes ---
    if gltf.scenes and gltf.scene is not None:
        gltf_scene = gltf.scenes[gltf.scene]
        scene.root_nodes = list(gltf_scene.nodes) if gltf_scene.nodes else []
    elif scene.nodes:
        # Find orphan nodes (no parent)
        scene.root_nodes = [
            i for i, n in enumerate(scene.nodes) if n.parent == -1
        ]

    # --- Load cameras ---
    if gltf.cameras:
        for cam in gltf.cameras:
            gcam = GltfCamera(name=cam.name or "camera")
            if cam.type == "perspective" and cam.perspective:
                gcam.fov_y = cam.perspective.yfov or math.radians(60)
                gcam.aspect = cam.perspective.aspectRatio or 1.0
                gcam.near = cam.perspective.znear or 0.01
                gcam.far = cam.perspective.zfar or 1000.0
            scene.cameras.append(gcam)

    # --- Load lights (KHR_lights_punctual) ---
    if hasattr(gltf, 'extensions') and gltf.extensions:
        lights_ext = gltf.extensions.get("KHR_lights_punctual", {}) if isinstance(gltf.extensions, dict) else {}
        lights_list = lights_ext.get("lights", [])
        for light_data in lights_list:
            glight = GltfLight(
                name=light_data.get("name", "light"),
                type=light_data.get("type", "directional"),
                color=tuple(light_data.get("color", [1, 1, 1])),
                intensity=light_data.get("intensity", 1.0),
                range=light_data.get("range", 0.0),
            )
            if glight.type == "spot":
                spot = light_data.get("spot", {})
                glight.inner_cone_angle = spot.get("innerConeAngle", 0.0)
                glight.outer_cone_angle = spot.get("outerConeAngle", math.pi / 4)
            scene.lights.append(glight)

    return scene


# ===========================================================================
# Helpers
# ===========================================================================

def _node_transform(node) -> np.ndarray:
    """Compute 4x4 transform matrix from a glTF node's TRS or matrix."""
    if node.matrix:
        return np.array(node.matrix, dtype=np.float32).reshape(4, 4)

    t = np.eye(4, dtype=np.float32)
    if node.translation:
        t[3, 0] = node.translation[0]
        t[3, 1] = node.translation[1]
        t[3, 2] = node.translation[2]

    r = np.eye(4, dtype=np.float32)
    if node.rotation:
        r = _quat_to_mat4(node.rotation)

    s = np.eye(4, dtype=np.float32)
    if node.scale:
        s[0, 0] = node.scale[0]
        s[1, 1] = node.scale[1]
        s[2, 2] = node.scale[2]

    # Row-vector convention: v' = v @ S @ R @ T (scale, then rotate, then translate)
    return s @ r @ t


def _quat_to_mat4(q) -> np.ndarray:
    """Convert quaternion [x, y, z, w] to 4x4 rotation matrix."""
    x, y, z, w = q
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = 1 - 2 * (y*y + z*z)
    m[0, 1] = 2 * (x*y + z*w)
    m[0, 2] = 2 * (x*z - y*w)
    m[1, 0] = 2 * (x*y - z*w)
    m[1, 1] = 1 - 2 * (x*x + z*z)
    m[1, 2] = 2 * (y*z + x*w)
    m[2, 0] = 2 * (x*z + y*w)
    m[2, 1] = 2 * (y*z - x*w)
    m[2, 2] = 1 - 2 * (x*x + y*y)
    return m


def _load_texture_image(gltf, texture_index: int) -> Optional[np.ndarray]:
    """Load a texture image from glTF as RGBA uint8 numpy array."""
    try:
        from PIL import Image as PILImage
        import io
    except ImportError:
        return None

    if texture_index is None or texture_index < 0:
        return None
    if texture_index >= len(gltf.textures):
        return None

    tex = gltf.textures[texture_index]
    if tex.source is None or tex.source < 0:
        return None
    if tex.source >= len(gltf.images):
        return None

    img_info = gltf.images[tex.source]
    data = gltf.binary_blob()
    if data is None:
        return None

    # Image from bufferView (embedded in GLB)
    if img_info.bufferView is not None:
        bv = gltf.bufferViews[img_info.bufferView]
        offset = bv.byteOffset or 0
        length = bv.byteLength
        img_bytes = data[offset:offset + length]
        try:
            img = PILImage.open(io.BytesIO(img_bytes))
            img = img.convert("RGBA")
            return np.array(img, dtype=np.uint8)
        except Exception:
            return None

    return None


def _load_primitive(gltf, prim, mesh_name: str) -> GltfMesh:
    """Load a single mesh primitive, producing interleaved vertex data."""
    from pygltflib import GLTF2
    import pygltflib

    positions = _read_accessor(gltf, prim.attributes.POSITION) if prim.attributes.POSITION is not None else None
    normals = _read_accessor(gltf, prim.attributes.NORMAL) if hasattr(prim.attributes, 'NORMAL') and prim.attributes.NORMAL is not None else None
    texcoords = _read_accessor(gltf, prim.attributes.TEXCOORD_0) if hasattr(prim.attributes, 'TEXCOORD_0') and prim.attributes.TEXCOORD_0 is not None else None
    tangents = _read_accessor(gltf, prim.attributes.TANGENT) if hasattr(prim.attributes, 'TANGENT') and prim.attributes.TANGENT is not None else None

    if positions is None:
        return GltfMesh(mesh_name, b"", b"", 0, 0)

    num_vertices = len(positions) // 3

    has_tangents = tangents is not None
    floats_per_vertex = 12 if has_tangents else 8  # pos(3)+normal(3)+uv(2)+tangent(4) or pos(3)+normal(3)+uv(2)
    vertex_stride = 48 if has_tangents else 32

    vertices = []
    for i in range(num_vertices):
        px, py, pz = positions[i*3], positions[i*3+1], positions[i*3+2]
        nx, ny, nz = (normals[i*3], normals[i*3+1], normals[i*3+2]) if normals else (0, 1, 0)
        u, v = (texcoords[i*2], texcoords[i*2+1]) if texcoords else (0, 0)
        vertices.extend([px, py, pz, nx, ny, nz, u, v])
        if has_tangents:
            tx, ty, tz, tw = tangents[i*4], tangents[i*4+1], tangents[i*4+2], tangents[i*4+3]
            vertices.extend([tx, ty, tz, tw])

    vertex_data = struct.pack(f"{len(vertices)}f", *vertices)

    # Load indices
    if prim.indices is not None:
        indices = _read_accessor_int(gltf, prim.indices)
        index_data = struct.pack(f"{len(indices)}I", *indices)
        num_indices = len(indices)
    else:
        # No indices — generate sequential
        indices = list(range(num_vertices))
        index_data = struct.pack(f"{num_vertices}I", *indices)
        num_indices = num_vertices

    material_index = prim.material if prim.material is not None else 0

    return GltfMesh(
        name=mesh_name,
        vertex_data=vertex_data,
        index_data=index_data,
        num_vertices=num_vertices,
        num_indices=num_indices,
        material_index=material_index,
        has_tangents=has_tangents,
        vertex_stride=vertex_stride,
    )


def _read_accessor(gltf, accessor_index: int) -> list[float]:
    """Read a float accessor from the glTF binary data."""
    accessor = gltf.accessors[accessor_index]
    buffer_view = gltf.bufferViews[accessor.bufferView]
    buffer = gltf.buffers[buffer_view.buffer]

    # Get binary data
    data = gltf.binary_blob()
    if data is None:
        return []

    offset = (buffer_view.byteOffset or 0) + (accessor.byteOffset or 0)
    component_count = _component_count(accessor.type)
    element_size = component_count * 4  # 4 bytes per float32
    byte_stride = getattr(buffer_view, 'byteStride', None) or 0

    if byte_stride == 0 or byte_stride == element_size:
        # Contiguous: fast path
        total_components = accessor.count * component_count
        values = struct.unpack_from(f"<{total_components}f", data, offset)
        return list(values)
    else:
        # Interleaved: read element-by-element with stride stepping
        values = []
        for i in range(accessor.count):
            elem_offset = offset + i * byte_stride
            elem = struct.unpack_from(f"<{component_count}f", data, elem_offset)
            values.extend(elem)
        return values


def _read_accessor_int(gltf, accessor_index: int) -> list[int]:
    """Read an integer accessor (indices) from the glTF binary data."""
    accessor = gltf.accessors[accessor_index]
    buffer_view = gltf.bufferViews[accessor.bufferView]

    data = gltf.binary_blob()
    if data is None:
        return []

    offset = (buffer_view.byteOffset or 0) + (accessor.byteOffset or 0)

    # Component type: 5121=UNSIGNED_BYTE, 5123=UNSIGNED_SHORT, 5125=UNSIGNED_INT
    if accessor.componentType == 5121:
        elem_fmt, elem_size = "B", 1
    elif accessor.componentType == 5123:
        elem_fmt, elem_size = "H", 2
    else:
        elem_fmt, elem_size = "I", 4

    byte_stride = getattr(buffer_view, 'byteStride', None) or 0

    if byte_stride == 0 or byte_stride == elem_size:
        # Contiguous: fast path
        fmt = f"<{accessor.count}{elem_fmt}"
        values = struct.unpack_from(fmt, data, offset)
        return list(values)
    else:
        # Strided: read element-by-element
        values = []
        for i in range(accessor.count):
            elem_offset = offset + i * byte_stride
            val = struct.unpack_from(f"<{elem_fmt}", data, elem_offset)
            values.extend(val)
        return values


def _component_count(accessor_type: str) -> int:
    """Number of components for a glTF accessor type."""
    return {
        "SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4,
        "MAT2": 4, "MAT3": 9, "MAT4": 16,
    }.get(accessor_type, 1)
