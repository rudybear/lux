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
import json as _json
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
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
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
    emissive_strength: float = 1.0
    ior: float = 1.5
    clearcoat_factor: float = 0.0
    clearcoat_roughness_factor: float = 0.0
    sheen_color_factor: tuple = (0.0, 0.0, 0.0)
    sheen_roughness_factor: float = 0.0
    transmission_factor: float = 0.0
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
    type: str = "directional"       # "directional", "point", "spot"
    range: float = 0.0              # 0 = infinite
    inner_cone_angle: float = 0.0   # spot only
    outer_cone_angle: float = 0.7854  # spot only (~pi/4)
    shadow_index: float = -1.0


@dataclass
class ShadowEntry:
    """Shadow map data for a single shadow-casting light."""
    view_projection: list = field(default_factory=lambda: [
        1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])  # 4x4 column-major flat
    bias: float = 0.005
    normal_bias: float = 0.02
    resolution: float = 1024.0
    light_size: float = 0.02


@dataclass
class DrawRange:
    """Index range for a single draw call within a merged vertex/index buffer."""
    index_offset: int
    index_count: int
    material_index: int


@dataclass
class SceneData:
    """Complete scene description, independent of rendering pipeline."""
    meshes: list[MeshData] = field(default_factory=list)
    materials: list[MaterialData] = field(default_factory=list)
    camera: CameraData = field(default_factory=CameraData)
    lights: list[LightData] = field(default_factory=list)
    render_path: str = "raster"  # hint: "raster", "fullscreen"
    camera_elevation: float = 5.0  # degrees; 0 = straight-on
    scene_features: set = field(default_factory=set)  # detected material features


@dataclass
class GPUScene:
    """Scene resources uploaded to GPU, independent of any pipeline."""
    vertex_buffers: list = field(default_factory=list)  # list of (GPUBuffer, stride, num_verts)
    index_buffers: list = field(default_factory=list)    # list of (GPUBuffer, num_indices)
    textures: dict = field(default_factory=dict)         # name -> (sampler, texture_view)
    uniform_buffers: dict = field(default_factory=dict)  # name -> GPUBuffer
    storage_buffers: dict = field(default_factory=dict)  # name -> GPUBuffer (SSBOs)
    per_material_textures: list = field(default_factory=list)  # list of dict: name -> (sampler, view)
    per_material_ubos: list = field(default_factory=list)      # list of GPUBuffer
    draw_ranges: list = field(default_factory=list)            # list of DrawRange
    reflection_data: dict = field(default_factory=dict)        # stage -> reflection dict
    # Shadow map resources
    shadow_texture: object = None
    shadow_sampler: object = None
    shadow_array_view: object = None
    shadow_layer_views: list = field(default_factory=list)
    shadow_matrices_buffer: object = None
    shadow_depth_pipeline: object = None
    shadow_depth_bind_layout: object = None
    shadow_depth_pipeline_layout: object = None
    shadow_mvp_buffers: list = field(default_factory=list)
    has_shadows: bool = False
    num_shadow_maps: int = 0


# ---------------------------------------------------------------------------
# Scene loading
# ---------------------------------------------------------------------------

def load_scene(source: str, camera_elevation: float = 5.0) -> SceneData:
    """Load a scene from a source specifier."""
    if source == "sphere":
        return _builtin_sphere()
    elif source == "fullscreen":
        return _builtin_fullscreen()
    elif source == "triangle":
        return _builtin_triangle()
    elif source.endswith((".glb", ".gltf")):
        return _load_gltf_scene(source, camera_elevation=camera_elevation)
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


def _transform_vertices(vertex_data: bytes, vertex_stride: int, num_vertices: int,
                        transform: np.ndarray) -> bytes:
    """Apply a 4x4 world transform to vertex positions and normals.

    Uses row-vector convention: v_transformed = v @ M (translation in last row).
    """
    if np.allclose(transform, np.eye(4)):
        return vertex_data
    stride_floats = vertex_stride // 4
    vdata = np.frombuffer(vertex_data, dtype=np.float32).copy().reshape(-1, stride_floats)
    # Transform positions (first 3 floats): row-vector * matrix
    pos = vdata[:, :3]
    ones = np.ones((num_vertices, 1), dtype=np.float32)
    pos_h = np.hstack([pos, ones])  # (N, 4)
    pos_transformed = (pos_h @ transform)[:, :3]
    vdata[:, :3] = pos_transformed
    # Transform normals (floats 3-5) using upper 3x3 (row-vector)
    upper3x3 = transform[:3, :3]
    normal_mat = np.linalg.inv(upper3x3).T  # inverse-transpose for normals
    normals = vdata[:, 3:6]
    normals_transformed = normals @ normal_mat
    norms = np.linalg.norm(normals_transformed, axis=1, keepdims=True)
    normals_transformed /= np.maximum(norms, 1e-8)
    vdata[:, 3:6] = normals_transformed
    # Transform tangent direction (floats 8-10 if stride >= 48)
    if stride_floats >= 12:
        tangent_dir = vdata[:, 8:11]
        tangent_transformed = tangent_dir @ upper3x3
        tnorms = np.linalg.norm(tangent_transformed, axis=1, keepdims=True)
        tangent_transformed /= np.maximum(tnorms, 1e-8)
        vdata[:, 8:11] = tangent_transformed
    return vdata.tobytes()


def _load_gltf_scene(path: str, camera_elevation: float = 5.0) -> SceneData:
    from gltf_loader import load_gltf, flatten_scene
    gltf_scene = load_gltf(Path(path))
    draw_items = flatten_scene(gltf_scene)

    scene = SceneData(render_path="raster", camera_elevation=camera_elevation)

    # Use draw_items to apply per-node world transforms to vertices
    for item in draw_items:
        mesh = gltf_scene.meshes[item.mesh_index]
        vdata = _transform_vertices(
            mesh.vertex_data, getattr(mesh, 'vertex_stride', 32),
            mesh.num_vertices, item.world_transform,
        )
        # Detect negative-determinant transforms (reflections/axis swaps)
        # which flip triangle winding order.  Reverse winding to compensate.
        idata = mesh.index_data
        upper3x3 = item.world_transform[:3, :3]
        if np.linalg.det(upper3x3) < 0 and mesh.num_indices > 0:
            indices = np.frombuffer(idata, dtype=np.uint32).copy()
            # Swap 2nd and 3rd vertex of every triangle: (v0,v1,v2) -> (v0,v2,v1)
            for tri in range(0, len(indices) - 2, 3):
                indices[tri + 1], indices[tri + 2] = indices[tri + 2], indices[tri + 1]
            idata = indices.tobytes()
        scene.meshes.append(MeshData(
            vertex_data=vdata,
            index_data=idata,
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
        # Wire extension properties from GltfMaterial
        ext = getattr(mat, 'extensions', {})
        if 'emissive_strength' in ext:
            mdata.emissive_strength = ext['emissive_strength'].get('emissiveStrength', 1.0)
        if 'ior' in ext:
            mdata.ior = ext['ior'].get('ior', 1.5)
        if 'clearcoat' in ext:
            mdata.clearcoat_factor = ext['clearcoat'].get('factor', 0.0)
            mdata.clearcoat_roughness_factor = ext['clearcoat'].get('roughnessFactor', 0.0)
        if 'sheen' in ext:
            mdata.sheen_color_factor = tuple(ext['sheen'].get('colorFactor', [0, 0, 0]))
            mdata.sheen_roughness_factor = ext['sheen'].get('roughnessFactor', 0.0)
        if 'transmission' in ext:
            mdata.transmission_factor = ext['transmission'].get('factor', 0.0)
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
        # Extension textures
        if hasattr(mat, 'extensions'):
            if 'clearcoat' in mat.extensions:
                cc = mat.extensions['clearcoat']
                if 'texture' in cc and cc['texture'] is not None:
                    mdata.textures["clearcoat_tex"] = cc['texture']
                if 'roughness_texture' in cc and cc['roughness_texture'] is not None:
                    mdata.textures["clearcoat_roughness_tex"] = cc['roughness_texture']
            if 'sheen' in mat.extensions:
                sh = mat.extensions['sheen']
                if 'color_texture' in sh and sh['color_texture'] is not None:
                    mdata.textures["sheen_color_tex"] = sh['color_texture']
            if 'transmission' in mat.extensions:
                tr = mat.extensions['transmission']
                if 'texture' in tr and tr['texture'] is not None:
                    mdata.textures["transmission_tex"] = tr['texture']
        scene.materials.append(mdata)

    # Extract camera from glTF if available, or compute from scene bounds
    if gltf_scene.cameras:
        cam = gltf_scene.cameras[0]
        scene.camera = CameraData(fov_y=cam.fov_y, near=cam.near, far=cam.far)
    elif scene.meshes:
        # Auto-compute camera using node transform to find model's front.
        # Blender exports apply a rotation (±90° X) to convert Z-up to Y-up;
        # we use that rotation to determine where the model's "front" and "up" are.
        all_positions = []
        for mesh in scene.meshes:
            stride_floats = mesh.vertex_stride // 4
            vdata = np.frombuffer(mesh.vertex_data, dtype=np.float32).reshape(-1, stride_floats)
            all_positions.append(vdata[:, :3])
        all_positions = np.concatenate(all_positions, axis=0)
        bbox_min = all_positions.min(axis=0)
        bbox_max = all_positions.max(axis=0)
        center = (bbox_min + bbox_max) / 2.0
        extent = bbox_max - bbox_min

        # Determine camera direction from node transform
        cam_dir = np.array([0, 0, 1], dtype=np.float32)   # default: +Z
        cam_up = np.array([0, 1, 0], dtype=np.float32)    # default: +Y
        if draw_items:
            upper3x3 = draw_items[0].world_transform[:3, :3]
            row_norms = np.linalg.norm(upper3x3, axis=1, keepdims=True)
            R_approx = upper3x3 / np.maximum(row_norms, 1e-8)
            # Blender front (-Y local) and up (+Z local) in world space
            # R_approx is R_col^T (row-vector convention); use R_approx @ v
            # to compute R_col^T * v, matching the C++/Rust engines.
            front = R_approx @ np.array([0, -1, 0], dtype=np.float32)
            up_w = R_approx @ np.array([0, 0, 1], dtype=np.float32)
            fl = float(np.linalg.norm(front))
            ul = float(np.linalg.norm(up_w))
            if fl > 0.5 and ul > 0.5:
                candidate_dir = front / fl
                candidate_up = up_w / ul
                # Skip if view direction is nearly vertical
                if abs(float(candidate_dir[1])) < 0.9:
                    cam_dir = candidate_dir
                    cam_up = candidate_up
                    if abs(float(np.dot(cam_dir, cam_up))) > 0.9:
                        cam_up = np.array([0, 1, 0], dtype=np.float32)

        # Compute perpendicular extent for distance calculation
        view_right = np.cross(-cam_dir, cam_up)
        vr_len = float(np.linalg.norm(view_right))
        if vr_len < 1e-6:
            cam_up = np.array([0, 0, 1], dtype=np.float32)
            view_right = np.cross(-cam_dir, cam_up)
            vr_len = float(np.linalg.norm(view_right))
        view_right /= vr_len
        view_up = np.cross(view_right, -cam_dir)
        view_up /= max(float(np.linalg.norm(view_up)), 1e-8)

        proj_right = sum(abs(float(view_right[i])) * float(extent[i]) for i in range(3))
        proj_up = sum(abs(float(view_up[i])) * float(extent[i]) for i in range(3))
        max_perp_extent = max(proj_right, proj_up)

        distance = (max_perp_extent / 2.0) / math.tan(scene.camera.fov_y / 2.0)
        distance *= 1.1  # slight margin

        # Position camera along front direction with elevation
        eye = center + distance * cam_dir
        elev_rad = math.radians(scene.camera_elevation)
        eye = eye + distance * math.sin(elev_rad) * view_up

        scene.camera.eye = eye.astype(np.float32)
        scene.camera.target = center.astype(np.float32)
        scene.camera.up = cam_up.astype(np.float32)
        scene.camera.far = max(distance * 3.0, 100.0)

    # Extract lights
    for light in gltf_scene.lights:
        scene.lights.append(LightData(
            direction=light.direction.copy(),
            position=light.position.copy(),
            color=light.color,
            intensity=light.intensity,
            type=getattr(light, 'type', 'directional'),
            range=getattr(light, 'range', 0.0),
            inner_cone_angle=getattr(light, 'inner_cone_angle', 0.0),
            outer_cone_angle=getattr(light, 'outer_cone_angle', 0.7854),
        ))

    # Extract light positions/directions from node world transforms
    for node in gltf_scene.nodes:
        if node.light_index >= 0 and node.light_index < len(scene.lights):
            light = scene.lights[node.light_index]
            # Position from row 3 of world transform (row-vector convention: translation in last row)
            light.position = node.world_transform[3, :3].copy()
            # Direction: transform local -Z through rotation
            rot = node.world_transform[:3, :3]
            light.direction = (rot @ np.array([0, 0, -1], dtype=np.float32))
            norm = np.linalg.norm(light.direction)
            if norm > 1e-6:
                light.direction /= norm

    if not scene.lights:
        scene.lights.append(LightData())

    # Detect scene material features
    for mat in gltf_scene.materials:
        if mat.normal_texture is not None:
            scene.scene_features.add("has_normal_map")
        if mat.emissive_texture is not None or (mat.emissive and any(e > 0 for e in mat.emissive)):
            scene.scene_features.add("has_emission")
        if hasattr(mat, 'extensions'):
            if 'clearcoat' in mat.extensions:
                scene.scene_features.add("has_clearcoat")
            if 'sheen' in mat.extensions:
                scene.scene_features.add("has_sheen")
            if 'transmission' in mat.extensions:
                scene.scene_features.add("has_transmission")
    # Detect shadow support: if any light has shadow capability, enable has_shadows
    if scene.lights:
        scene.scene_features.add("has_shadows")
    if scene.scene_features:
        print(f"[info] Detected material features: {scene.scene_features}")

    return scene


def detect_scene_features(scene_source: str, scene: SceneData, gltf_scene) -> set:
    """Detect which material features are used across the scene."""
    features = set()
    if not hasattr(gltf_scene, 'materials'):
        return features
    for mat in gltf_scene.materials:
        if mat.normal_texture is not None:
            features.add("has_normal_map")
        if mat.emissive_texture is not None or (mat.emissive and any(e > 0 for e in mat.emissive)):
            features.add("has_emission")
        if hasattr(mat, 'extensions'):
            if 'clearcoat' in mat.extensions:
                features.add("has_clearcoat")
            if 'sheen' in mat.extensions:
                features.add("has_sheen")
            if 'transmission' in mat.extensions:
                features.add("has_transmission")
    # Detect shadow support
    if scene.lights:
        features.add("has_shadows")
    return features


# ---------------------------------------------------------------------------
# Per-material feature detection
# ---------------------------------------------------------------------------

def detect_material_features(material: MaterialData, material_textures: dict) -> set:
    """Detect features for a single material.

    Examines both the material's scalar properties and its texture map to
    determine which shader features (normal mapping, emission, clearcoat,
    sheen, transmission) are active for this particular material.
    """
    features = set()
    if "normal_tex" in material_textures and material_textures["normal_tex"] is not None:
        features.add("has_normal_map")
    if material.emissive != (0, 0, 0) or "emissive_tex" in material_textures:
        features.add("has_emission")
    if material.clearcoat_factor > 0:
        features.add("has_clearcoat")
    if any(c > 0 for c in material.sheen_color_factor) or material.sheen_roughness_factor > 0:
        features.add("has_sheen")
    if material.transmission_factor > 0:
        features.add("has_transmission")
    return features


def features_to_suffix(features: set) -> str:
    """Convert feature set to permutation suffix.

    Example: {has_normal_map, has_sheen} -> '+normal_map+sheen'
    Features are sorted alphabetically for deterministic ordering.
    """
    if not features:
        return ""
    sorted_names = sorted(f.replace("has_", "") for f in features)
    return "+" + "+".join(sorted_names)


def group_materials_by_features(materials: list, per_material_textures: list) -> dict:
    """Group materials by their feature suffix.

    Returns a dict mapping suffix string to a list of material indices that
    share those features, e.g. {'': [0], '+normal_map+sheen': [1, 3]}.
    """
    groups = {}
    for i, mat in enumerate(materials):
        tex = per_material_textures[i] if i < len(per_material_textures) else {}
        suffix = features_to_suffix(detect_material_features(mat, tex))
        groups.setdefault(suffix, []).append(i)

    if len(groups) > 1:
        print("[info] Material permutation groups:")
        for suffix, indices in groups.items():
            label = suffix if suffix else "(base)"
            print(f"  \"{label}\": materials {indices}")

    return groups


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------

def parse_manifest_json(manifest_path: Path) -> dict:
    """Parse a .manifest.json file.

    Returns a dict with keys:
        'pipeline': str - pipeline name
        'features': list[str] - ordered feature names
        'permutations': list[dict] - each with 'suffix' and 'features' dict
    """
    with open(manifest_path, encoding="utf-8") as f:
        data = _json.load(f)

    result = {
        "pipeline": data.get("pipeline", ""),
        "features": data.get("features", []),
        "permutations": [],
    }
    for p in data.get("permutations", []):
        perm = {
            "suffix": p.get("suffix", ""),
            "features": p.get("features", {}),
        }
        result["permutations"].append(perm)
    return result


def try_load_manifest(pipeline_base: str) -> Optional[dict]:
    """Try to load a manifest from pipeline_base + '.manifest.json'.

    Also checks a legacy subdirectory format used by older shader caches.
    Returns the parsed manifest dict, or None if no manifest exists.
    """
    # Try direct: pipelineBase + ".manifest.json"
    path1 = Path(pipeline_base + ".manifest.json")
    if path1.exists():
        print(f"[info] Loading shader manifest: {path1}")
        return parse_manifest_json(path1)

    # Try legacy subdirectory format
    base_path = Path(pipeline_base)
    filename = base_path.name
    parent = base_path.parent
    if parent != base_path:
        path2 = parent / "gltf_pbr" / (filename + ".manifest.json")
        if path2.exists():
            print(f"[info] Loading shader manifest: {path2}")
            return parse_manifest_json(path2)

    return None


def find_permutation_suffix(manifest: dict, material_features: set) -> str:
    """Find the matching permutation suffix for a given set of material features.

    Checks the manifest's permutation list for an exact match of the feature
    flags. Falls back to the empty suffix (base permutation) if no match.
    """
    feature_names = manifest.get("features", [])

    # Build wanted feature map
    wanted = {}
    for fname in feature_names:
        wanted[fname] = (fname in material_features)

    # Find exact match
    for perm in manifest.get("permutations", []):
        match = True
        for fname in feature_names:
            perm_has = perm["features"].get(fname, False)
            if perm_has != wanted[fname]:
                match = False
                break
        if match:
            return perm["suffix"]

    # Fallback: base permutation
    return ""


# ---------------------------------------------------------------------------
# Pipeline resolution
# ---------------------------------------------------------------------------

def resolve_pipeline(pipeline_arg: Optional[str], scene_source: str,
                     scene_features: set = None) -> str:
    """If no --pipeline given, pick a sensible default for the scene type."""
    if pipeline_arg:
        return pipeline_arg
    if scene_source.endswith((".glb", ".gltf")):
        if scene_features:
            suffix = "+".join(sorted(f.replace("has_", "") for f in scene_features))
            candidate = f"shadercache/gltf_pbr_layered+{suffix}"
            if Path(candidate + ".frag.spv").exists():
                return candidate
        return "shadercache/gltf_pbr"
    elif scene_source == "fullscreen":
        raise ValueError("--pipeline is required for fullscreen scenes")
    elif scene_source == "triangle":
        return "shadercache/hello_triangle"
    else:
        return "shadercache/pbr_basic"


def detect_render_path(pipeline_base: str) -> str:
    """Auto-detect rendering path from available shader files.

    Prefers raster over RT since the Python/wgpu engine only supports raster.
    """
    base = Path(pipeline_base)
    if base.with_suffix(".vert.spv").exists() and base.with_suffix(".frag.spv").exists():
        return "raster"
    elif base.with_suffix(".frag.spv").exists():
        return "fullscreen"
    elif base.with_suffix(".mesh.spv").exists():
        raise RuntimeError(
            "Mesh shaders are not supported by the Python/wgpu engine. "
            "Use the C++ or Rust engine instead."
        )
    elif base.with_suffix(".rgen.spv").exists():
        return "rt"
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


# ---------------------------------------------------------------------------
# Material UBO packing
# ---------------------------------------------------------------------------

# glTF-specific mapping: reflection field name -> MaterialData attribute
_GLTF_FIELD_MAP = {
    "base_color_factor": lambda m: m.base_color,        # vec4
    "metallic_factor":   lambda m: m.metallic,           # scalar
    "roughness_factor":  lambda m: m.roughness,          # scalar
    "emissive_factor":   lambda m: m.emissive,           # vec3
    "emissive_strength": lambda m: m.emissive_strength,  # scalar
    "ior":               lambda m: m.ior,                # scalar
    "clearcoat_factor":  lambda m: m.clearcoat_factor,   # scalar
    "clearcoat_roughness_factor": lambda m: m.clearcoat_roughness_factor,
    "sheen_color_factor":    lambda m: m.sheen_color_factor,     # vec3
    "sheen_roughness_factor": lambda m: m.sheen_roughness_factor,
    "transmission_factor":    lambda m: m.transmission_factor,
}


def _pack_field(buf: bytearray, offset: int, type_name: str, value):
    """Pack a single field value into a buffer at the given offset."""
    if type_name == "scalar":
        struct.pack_into("f", buf, offset, float(value))
    elif type_name == "vec2":
        v = value if hasattr(value, '__len__') else (float(value), 0.0)
        struct.pack_into("2f", buf, offset, *v[:2])
    elif type_name == "vec3":
        v = value if hasattr(value, '__len__') else (float(value), 0.0, 0.0)
        struct.pack_into("3f", buf, offset, *v[:3])
    elif type_name == "vec4":
        v = value if hasattr(value, '__len__') else (float(value), 0.0, 0.0, 0.0)
        struct.pack_into("4f", buf, offset, *v[:4])


def _pack_properties_ubo(mat: MaterialData, ubo_reflection: dict) -> bytes:
    """Pack material data into buffer matching reflection-declared field offsets."""
    buf = bytearray(ubo_reflection["size"])
    for field in ubo_reflection["fields"]:
        getter = _GLTF_FIELD_MAP.get(field["name"])
        if getter:
            value = getter(mat)
        elif "default" in field:
            value = field["default"]
        else:
            continue
        _pack_field(buf, field["offset"], field["type"], value)
    return bytes(buf)


def _pack_material_ubo(mat: MaterialData) -> bytes:
    """Pack material properties into std140 UBO matching properties Material block.

    Fixed layout (144 bytes total):
        offset  0: base_color_factor  vec4  (16 bytes)
        offset 16: emissive_factor    vec3  (12 bytes)
        offset 28: metallic_factor    scalar (4 bytes)
        offset 32: roughness_factor   scalar (4 bytes)
        offset 36: emissive_strength  scalar (4 bytes)
        offset 40: ior                scalar (4 bytes)
        offset 44: clearcoat_factor   scalar (4 bytes)
        offset 48: clearcoat_roughness_factor scalar (4 bytes)
        offset 52: sheen_roughness_factor     scalar (4 bytes)
        offset 56: transmission_factor        scalar (4 bytes)
        offset 60: _pad                       (4 bytes, vec3 alignment)
        offset 64: sheen_color_factor  vec3  (12 bytes)
        offset 76: _pad                       (4 bytes)
        offset 80: baseColorUvSt      vec4  (offset.xy, scale.xy)
        offset 96: normalUvSt         vec4  (offset.xy, scale.xy)
        offset112: mrUvSt             vec4  (offset.xy, scale.xy)
        offset128: baseColorUvRot     scalar (4 bytes)
        offset132: normalUvRot        scalar (4 bytes)
        offset136: mrUvRot            scalar (4 bytes)
        offset140: _pad               (4 bytes)
    """
    buf = bytearray(144)
    bc = mat.base_color
    struct.pack_into("4f", buf, 0, bc[0], bc[1], bc[2], bc[3] if len(bc) > 3 else 1.0)
    em = mat.emissive
    struct.pack_into("3f", buf, 16, em[0], em[1], em[2] if len(em) > 2 else 0.0)
    struct.pack_into("f", buf, 28, mat.metallic)
    struct.pack_into("f", buf, 32, mat.roughness)
    struct.pack_into("f", buf, 36, mat.emissive_strength)
    struct.pack_into("f", buf, 40, mat.ior)
    struct.pack_into("f", buf, 44, mat.clearcoat_factor)
    struct.pack_into("f", buf, 48, mat.clearcoat_roughness_factor)
    struct.pack_into("f", buf, 52, mat.sheen_roughness_factor)
    struct.pack_into("f", buf, 56, mat.transmission_factor)
    # pad at offset 60
    sc = mat.sheen_color_factor
    struct.pack_into("3f", buf, 64, sc[0], sc[1], sc[2] if len(sc) > 2 else 0.0)
    # KHR_texture_transform: identity UV transforms (offset 0, scale 1, rotation 0)
    struct.pack_into("4f", buf, 80, 0.0, 0.0, 1.0, 1.0)   # baseColorUvSt
    struct.pack_into("4f", buf, 96, 0.0, 0.0, 1.0, 1.0)   # normalUvSt
    struct.pack_into("4f", buf, 112, 0.0, 0.0, 1.0, 1.0)  # mrUvSt
    # rotations default to 0.0 (already zero-initialized)
    return bytes(buf)


def _pack_lights_buffer(scene: SceneData) -> bytes:
    """Pack scene lights into GPU buffer (std430 layout: 64 bytes per light)."""
    lights = scene.lights if scene.lights else []

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

        # vec4: (light_type, intensity, range, inner_cone)
        buf += struct.pack("4f", light_type, intensity, range_val, inner_cone)
        # vec4: (position.xyz, outer_cone)
        buf += struct.pack("3f", float(pos[0]), float(pos[1]), float(pos[2]))
        buf += struct.pack("f", outer_cone)
        # vec4: (direction.xyz, shadow_index)
        buf += struct.pack("3f", float(direction[0]), float(direction[1]), float(direction[2]))
        buf += struct.pack("f", float(shadow_idx))
        # vec4: (color.xyz, pad)
        buf += struct.pack("3f", float(color[0]), float(color[1]), float(color[2]))
        buf += struct.pack("f", 0.0)

    if not buf:
        # Dummy light (directional, white, pointing down)
        # Layout: vec4(type,intensity,range,inner_cone), vec4(pos.xyz,outer_cone),
        #         vec4(dir.xyz,shadow_idx), vec4(color.xyz,pad)
        buf = bytearray(64)
        struct.pack_into("f", buf, 0, 0.0)       # type: directional
        struct.pack_into("f", buf, 4, 1.0)       # intensity
        struct.pack_into("f", buf, 28, 0.7854)   # outer_cone
        struct.pack_into("f", buf, 32, 0.0)      # dir.x
        struct.pack_into("f", buf, 36, -1.0)     # dir.y
        struct.pack_into("f", buf, 40, 0.0)      # dir.z
        struct.pack_into("f", buf, 48, 1.0)      # color.r
        struct.pack_into("f", buf, 52, 1.0)      # color.g
        struct.pack_into("f", buf, 56, 1.0)      # color.b

    return bytes(buf)


# ---------------------------------------------------------------------------
# Shadow map helpers (Phase F)
# ---------------------------------------------------------------------------

_SHADOW_MAP_RESOLUTION = 1024
_MAX_SHADOW_MAPS = 8


def _look_at_shadow(eye, target, up) -> list:
    """Compute a column-major 4x4 look-at view matrix (16 floats).

    Follows the same conventions as scene_utils.look_at but returns a flat
    list instead of a numpy array so it can be used without numpy dependency
    in pure-Python shadow math.
    """
    import math as _m
    # forward
    fx = target[0] - eye[0]
    fy = target[1] - eye[1]
    fz = target[2] - eye[2]
    fl = _m.sqrt(fx * fx + fy * fy + fz * fz)
    if fl < 1e-12:
        fl = 1e-12
    fx /= fl; fy /= fl; fz /= fl
    # side = cross(f, up)
    sx = fy * up[2] - fz * up[1]
    sy = fz * up[0] - fx * up[2]
    sz = fx * up[1] - fy * up[0]
    sl = _m.sqrt(sx * sx + sy * sy + sz * sz)
    if sl < 1e-12:
        # Degenerate: pick a different up vector
        alt_up = (1.0, 0.0, 0.0) if abs(fx) < 0.9 else (0.0, 1.0, 0.0)
        sx = fy * alt_up[2] - fz * alt_up[1]
        sy = fz * alt_up[0] - fx * alt_up[2]
        sz = fx * alt_up[1] - fy * alt_up[0]
        sl = _m.sqrt(sx * sx + sy * sy + sz * sz)
    sx /= sl; sy /= sl; sz /= sl
    # up = cross(s, f)
    ux = sy * fz - sz * fy
    uy = sz * fx - sx * fz
    uz = sx * fy - sy * fx
    # dot products
    ds = -(sx * eye[0] + sy * eye[1] + sz * eye[2])
    du = -(ux * eye[0] + uy * eye[1] + uz * eye[2])
    df = fx * eye[0] + fy * eye[1] + fz * eye[2]
    # Column-major 4x4: m[col][row]
    # col 0: (s.x, u.x, -f.x, 0)
    # col 1: (s.y, u.y, -f.y, 0)
    # col 2: (s.z, u.z, -f.z, 0)
    # col 3: (ds,  du,   df,  1)
    return [
        sx,  ux,  -fx, 0.0,
        sy,  uy,  -fy, 0.0,
        sz,  uz,  -fz, 0.0,
        ds,  du,   df, 1.0,
    ]


def _ortho_shadow(left, right, bottom, top, near, far) -> list:
    """Compute a column-major 4x4 orthographic projection (16 floats).

    Maps to Vulkan/wgpu clip space: y-down not flipped (caller handles),
    z in [0, 1].
    """
    rl = right - left
    tb = top - bottom
    fn = far - near
    if abs(rl) < 1e-12:
        rl = 1e-12
    if abs(tb) < 1e-12:
        tb = 1e-12
    if abs(fn) < 1e-12:
        fn = 1e-12
    # Column-major
    return [
        2.0 / rl,               0.0,                    0.0,            0.0,
        0.0,                    2.0 / tb,                0.0,            0.0,
        0.0,                    0.0,                   -1.0 / fn,        0.0,
       -(right + left) / rl,  -(top + bottom) / tb,    -near / fn,      1.0,
    ]


def _perspective_shadow(fovy, aspect, near, far) -> list:
    """Compute a column-major 4x4 perspective projection (16 floats).

    Maps to Vulkan/wgpu clip space: y-down, z in [0, 1].
    """
    import math as _m
    f = 1.0 / _m.tan(fovy / 2.0)
    fn = far - near
    if abs(fn) < 1e-12:
        fn = 1e-12
    return [
        f / aspect, 0.0,  0.0,                     0.0,
        0.0,       -f,    0.0,                     0.0,
        0.0,        0.0,  far / (near - far),      -1.0,
        0.0,        0.0,  (near * far) / (near - far), 0.0,
    ]


def _mat4_multiply(a, b) -> list:
    """Multiply two column-major 4x4 matrices (each 16 floats). Returns 16 floats."""
    result = [0.0] * 16
    for col in range(4):
        for row in range(4):
            s = 0.0
            for k in range(4):
                s += a[k * 4 + row] * b[col * 4 + k]
            result[col * 4 + row] = s
    return result


def _compute_shadow_data(lights: list, camera_view=None,
                          camera_proj=None, near=0.1, far=100.0) -> list:
    """Compute shadow matrices for shadow-casting lights.

    Returns a list of ShadowEntry (one per shadow-casting light, up to
    _MAX_SHADOW_MAPS).  Also updates each light's shadow_index to reference
    the correct shadow entry.
    """
    entries = []
    shadow_idx = 0

    for light in lights:
        if shadow_idx >= _MAX_SHADOW_MAPS:
            break

        light_type = getattr(light, 'type', 'directional')

        if light_type == "directional":
            d = light.direction
            d_len = math.sqrt(d[0]**2 + d[1]**2 + d[2]**2)
            if d_len < 1e-8:
                d = np.array([0.0, -1.0, 0.0])
                d_len = 1.0
            dn = [d[0] / d_len, d[1] / d_len, d[2] / d_len]

            # Place light "eye" far along the incoming direction
            dist = far * 0.5
            eye = [-dn[0] * dist, -dn[1] * dist, -dn[2] * dist]
            target = [0.0, 0.0, 0.0]
            up = [0.0, 1.0, 0.0]
            if abs(dn[1]) > 0.95:
                up = [0.0, 0.0, 1.0]

            view_mat = _look_at_shadow(eye, target, up)
            # Ortho projection enclosing the scene
            ortho_extent = far * 0.5
            proj_mat = _ortho_shadow(
                -ortho_extent, ortho_extent,
                -ortho_extent, ortho_extent,
                near, far,
            )
            vp = _mat4_multiply(proj_mat, view_mat)

            entry = ShadowEntry(view_projection=vp, bias=0.005, normal_bias=0.02)
            entries.append(entry)
            light.shadow_index = float(shadow_idx)
            shadow_idx += 1

        elif light_type == "spot":
            pos = light.position
            d = light.direction
            d_len = math.sqrt(d[0]**2 + d[1]**2 + d[2]**2)
            if d_len < 1e-8:
                d = np.array([0.0, -1.0, 0.0])
                d_len = 1.0
            dn = [d[0] / d_len, d[1] / d_len, d[2] / d_len]

            eye = [float(pos[0]), float(pos[1]), float(pos[2])]
            target = [eye[0] + dn[0], eye[1] + dn[1], eye[2] + dn[2]]
            up = [0.0, 1.0, 0.0]
            if abs(dn[1]) > 0.95:
                up = [0.0, 0.0, 1.0]

            view_mat = _look_at_shadow(eye, target, up)
            outer_angle = getattr(light, 'outer_cone_angle', 0.7854)
            fovy = outer_angle * 2.0
            if fovy < 0.01:
                fovy = 0.01
            spot_far = getattr(light, 'range', 0.0)
            if spot_far <= 0.0:
                spot_far = far
            proj_mat = _perspective_shadow(fovy, 1.0, near, spot_far)
            vp = _mat4_multiply(proj_mat, view_mat)

            entry = ShadowEntry(view_projection=vp, bias=0.005, normal_bias=0.02)
            entries.append(entry)
            light.shadow_index = float(shadow_idx)
            shadow_idx += 1

        elif light_type == "point":
            # Point lights could use cubemap shadows; for now skip
            light.shadow_index = -1.0

        else:
            light.shadow_index = -1.0

    return entries


def _pack_shadow_buffer(entries: list) -> bytes:
    """Pack shadow entries into GPU buffer (std430: 80 bytes per entry).

    Layout per entry:
        16 floats — mat4 view_projection (64 bytes)
        4 floats  — (bias, normalBias, resolution, light_size) (16 bytes)
    Total: 80 bytes per entry.
    """
    buf = bytearray()
    for entry in entries:
        vp = entry.view_projection
        for i in range(16):
            buf += struct.pack("f", float(vp[i]))
        buf += struct.pack("4f", entry.bias, entry.normal_bias,
                           entry.resolution, entry.light_size)
    # Ensure at least one entry (dummy) if empty
    if not buf:
        buf = bytearray(80)
    return bytes(buf)


# Shadow depth shader (WGSL) — position-only vertex transform
_SHADOW_DEPTH_WGSL = """
struct ShadowUniforms {
    mvp: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> shadow_uniforms: ShadowUniforms;

@vertex
fn vs_main(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
    return shadow_uniforms.mvp * vec4<f32>(position, 1.0);
}
"""


def _setup_shadow_maps(device, scene, gpu_scene, reflection_data):
    """Create shadow map textures, sampler, and depth pipeline if shadows are needed.

    Detects shadow support from reflection data by looking for "shadow_maps"
    (sampler binding) and/or "shadow_matrices" (storage_buffer binding).

    Updates gpu_scene in place with shadow resources.
    """
    # Check reflection for shadow bindings
    has_shadow_maps = False
    has_shadow_matrices = False
    for _stage_key, stage_ref in reflection_data.items():
        for _set_key, bindings in stage_ref.get("descriptor_sets", {}).items():
            for binding in bindings:
                name = binding.get("name", "")
                btype = binding.get("type", "")
                if name == "shadow_maps" and btype in ("sampler", "sampled_image"):
                    has_shadow_maps = True
                elif name == "shadow_matrices" and btype == "storage_buffer":
                    has_shadow_matrices = True

    if not has_shadow_maps and not has_shadow_matrices:
        return  # shader doesn't use shadows

    # Compute shadow data for each shadow-casting light
    cam = scene.camera
    shadow_entries = _compute_shadow_data(
        scene.lights, near=cam.near, far=cam.far,
    )

    num_shadows = len(shadow_entries)
    if num_shadows == 0:
        # No shadow-casting lights; create a dummy so bindings are valid
        shadow_entries = [ShadowEntry()]
        num_shadows = 1

    gpu_scene.num_shadow_maps = num_shadows
    gpu_scene.has_shadows = True

    # 1. Create depth texture array
    gpu_scene.shadow_texture = device.create_texture(
        size=(_SHADOW_MAP_RESOLUTION, _SHADOW_MAP_RESOLUTION, num_shadows),
        format=wgpu.TextureFormat.depth32float,
        usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING,
    )

    # 2. Create comparison sampler
    gpu_scene.shadow_sampler = device.create_sampler(
        compare=wgpu.CompareFunction.less_equal,
        mag_filter=wgpu.FilterMode.linear,
        min_filter=wgpu.FilterMode.linear,
    )

    # 3. Create per-layer views for rendering into individual layers
    gpu_scene.shadow_layer_views = [
        gpu_scene.shadow_texture.create_view(
            dimension=wgpu.TextureViewDimension.d2,
            base_array_layer=i,
            array_layer_count=1,
        )
        for i in range(num_shadows)
    ]

    # 4. Create array view for shader binding
    gpu_scene.shadow_array_view = gpu_scene.shadow_texture.create_view(
        dimension=wgpu.TextureViewDimension.d2_array,
    )

    # 5. Pack and upload shadow matrices SSBO
    shadow_data = _pack_shadow_buffer(shadow_entries)
    gpu_scene.shadow_matrices_buffer = device.create_buffer_with_data(
        data=shadow_data,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )
    gpu_scene.storage_buffers["shadow_matrices"] = gpu_scene.shadow_matrices_buffer

    # 6. Create shadow depth pipeline (WGSL, position-only)
    shadow_module = device.create_shader_module(code=_SHADOW_DEPTH_WGSL)

    # Bind group layout: one uniform buffer for the shadow MVP
    shadow_bind_layout = device.create_bind_group_layout(entries=[
        wgpu.BindGroupLayoutEntry(
            binding=0,
            visibility=wgpu.ShaderStage.VERTEX,
            buffer=wgpu.BufferBindingLayout(type=wgpu.BufferBindingType.uniform),
        ),
    ])
    gpu_scene.shadow_depth_bind_layout = shadow_bind_layout

    shadow_pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[shadow_bind_layout],
    )
    gpu_scene.shadow_depth_pipeline_layout = shadow_pipeline_layout

    # Determine vertex stride from the scene's first mesh (position at offset 0)
    vertex_stride = 32  # default: pos(3) + normal(3) + uv(2) = 32
    if scene.meshes:
        vertex_stride = scene.meshes[0].vertex_stride
    # Also check GPU-side vertex buffers (may have been padded)
    if gpu_scene.vertex_buffers:
        _, uploaded_stride, _ = gpu_scene.vertex_buffers[0]
        vertex_stride = max(vertex_stride, uploaded_stride)

    gpu_scene.shadow_depth_pipeline = device.create_render_pipeline(
        layout=shadow_pipeline_layout,
        vertex=wgpu.VertexState(
            module=shadow_module,
            entry_point="vs_main",
            buffers=[wgpu.VertexBufferLayout(
                array_stride=vertex_stride,
                step_mode=wgpu.VertexStepMode.vertex,
                attributes=[wgpu.VertexAttribute(
                    format=wgpu.VertexFormat.float32x3,
                    offset=0,
                    shader_location=0,
                )],
            )],
        ),
        primitive=wgpu.PrimitiveState(
            topology=wgpu.PrimitiveTopology.triangle_list,
            cull_mode=wgpu.CullMode.back,
            front_face=wgpu.FrontFace.cw,
        ),
        depth_stencil=wgpu.DepthStencilState(
            format=wgpu.TextureFormat.depth32float,
            depth_write_enabled=True,
            depth_compare=wgpu.CompareFunction.less,
            depth_bias=2,
            depth_bias_slope_scale=1.5,
        ),
        multisample=wgpu.MultisampleState(count=1, mask=0xFFFFFFFF),
    )

    # 7. Create per-shadow-map MVP uniform buffers and bind groups
    gpu_scene.shadow_mvp_buffers = []
    for entry in shadow_entries:
        # MVP for shadow = light VP * model (model is identity for now)
        mvp_data = struct.pack(f"{16}f", *entry.view_projection)
        mvp_buf = device.create_buffer_with_data(
            data=mvp_data,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        gpu_scene.shadow_mvp_buffers.append(mvp_buf)

    print(f"[info] Shadow maps: {num_shadows} shadow caster(s) at "
          f"{_SHADOW_MAP_RESOLUTION}x{_SHADOW_MAP_RESOLUTION}")


def _render_shadow_pass(device, encoder, gpu_scene):
    """Execute all shadow depth render passes."""
    if not gpu_scene.has_shadows or gpu_scene.shadow_depth_pipeline is None:
        return

    for layer_idx in range(gpu_scene.num_shadow_maps):
        if layer_idx >= len(gpu_scene.shadow_layer_views):
            break
        if layer_idx >= len(gpu_scene.shadow_mvp_buffers):
            break

        layer_view = gpu_scene.shadow_layer_views[layer_idx]
        mvp_buf = gpu_scene.shadow_mvp_buffers[layer_idx]

        # Create bind group for this shadow map's MVP
        bind_group = device.create_bind_group(
            layout=gpu_scene.shadow_depth_bind_layout,
            entries=[wgpu.BindGroupEntry(
                binding=0,
                resource=wgpu.BufferBinding(buffer=mvp_buf, size=64),
            )],
        )

        # Begin depth-only render pass for this layer
        shadow_pass = encoder.begin_render_pass(
            color_attachments=[],
            depth_stencil_attachment=wgpu.RenderPassDepthStencilAttachment(
                view=layer_view,
                depth_load_op=wgpu.LoadOp.clear,
                depth_store_op=wgpu.StoreOp.store,
                depth_clear_value=1.0,
            ),
        )
        shadow_pass.set_pipeline(gpu_scene.shadow_depth_pipeline)
        shadow_pass.set_bind_group(0, bind_group)

        # Draw all scene meshes
        for i, (vbo, stride, num_verts) in enumerate(gpu_scene.vertex_buffers):
            ibo, num_indices = gpu_scene.index_buffers[i]
            if vbo:
                shadow_pass.set_vertex_buffer(0, vbo)
            if ibo and num_indices > 0:
                shadow_pass.set_index_buffer(ibo, wgpu.IndexFormat.uint32)
                shadow_pass.draw_indexed(num_indices)
            elif vbo:
                shadow_pass.draw(num_verts)

        shadow_pass.end()


def upload_scene(device: wgpu.GPUDevice, scene: SceneData, width: int, height: int,
                 pipeline_stride: int = 0, reflection_data: dict = None) -> GPUScene:
    """Upload scene resources to GPU. Independent of any pipeline."""
    gpu = GPUScene()
    if reflection_data:
        gpu.reflection_data = reflection_data

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

    # Upload textures from materials (global set, backward-compatible)
    for mat in scene.materials:
        for tex_name, tex_data in mat.textures.items():
            if tex_name in gpu.textures:
                continue
            sampler, view = _upload_texture(device, tex_data)
            gpu.textures[tex_name] = (sampler, view)

    # Upload per-material textures (each material gets its own texture map)
    for mat in scene.materials:
        mat_textures = {}
        for tex_name, tex_data in mat.textures.items():
            sampler, view = _upload_texture(device, tex_data)
            mat_textures[tex_name] = (sampler, view)
        gpu.per_material_textures.append(mat_textures)

    # Create per-material UBO buffers
    for mat in scene.materials:
        ubo_buf = device.create_buffer_with_data(
            data=_pack_material_ubo(mat),
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        gpu.per_material_ubos.append(ubo_buf)

    # Build draw ranges: track index offset as meshes are appended
    index_offset = 0
    for mesh in scene.meshes:
        if mesh.num_indices > 0:
            gpu.draw_ranges.append(DrawRange(
                index_offset=index_offset,
                index_count=mesh.num_indices,
                material_index=mesh.material_index,
            ))
            index_offset += mesh.num_indices

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

    # Setup shadow maps BEFORE packing lights so shadow_index is populated
    if reflection_data:
        _setup_shadow_maps(device, scene, gpu, reflection_data)

    # Multi-light SSBO creation (when shader uses "lights" SSBO)
    # Detect from reflection data if "lights" storage buffer is expected
    has_multi_light = False
    for stage_ref in gpu.reflection_data.values():
        for _set_key, bindings in stage_ref.get("descriptor_sets", {}).items():
            for binding in bindings:
                if binding.get("name") == "lights" and binding.get("type") == "storage_buffer":
                    has_multi_light = True
                    break

    if has_multi_light:
        light_data = _pack_lights_buffer(scene)
        gpu.storage_buffers["lights"] = device.create_buffer_with_data(
            data=light_data,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )

        # SceneLight UBO: vec3 view_pos at offset 0, int light_count at offset 12 (16 bytes total)
        light_count = len(scene.lights) if scene.lights else 1
        scene_light_data = struct.pack("3fi", *cam.eye, light_count)
        gpu.uniform_buffers["SceneLight"] = device.create_buffer_with_data(
            data=scene_light_data,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

    # Create Material properties UBO (fixed std140 layout) — for single-pipeline compat
    if scene.materials:
        gpu.uniform_buffers["Material"] = device.create_buffer_with_data(
            data=_pack_material_ubo(scene.materials[0]),
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

    return gpu


def _upload_texture(device: wgpu.GPUDevice, data: np.ndarray,
                    address_mode: str = "repeat"):
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
    """Create a 1x1 white RGBA texture for missing sampler bindings."""
    data = np.array([[255, 255, 255, 255]], dtype=np.uint8).reshape(1, 1, 4)
    return _upload_texture(device, data)


def _create_default_normal_texture(device: wgpu.GPUDevice):
    """Create a 1x1 flat normal map texture (0.5, 0.5, 1.0, 1.0).

    In tangent space, this decodes to (0, 0, 1) — no perturbation.
    A white default (1,1,1) would decode to a 45-degree tilted normal.
    """
    data = np.array([[128, 128, 255, 255]], dtype=np.uint8).reshape(1, 1, 4)
    return _upload_texture(device, data)


def _create_black_texture(device: wgpu.GPUDevice):
    """Create a 1x1 black RGBA texture for emissive/additive defaults.

    Emissive textures default to black (no emission). Using white would add
    (1,1,1) to every pixel's HDR output, completely washing out materials.
    """
    data = np.array([[0, 0, 0, 255]], dtype=np.uint8).reshape(1, 1, 4)
    return _upload_texture(device, data)


def _create_default_brdf_lut(device: wgpu.GPUDevice):
    """Create a 1x1 BRDF LUT with sensible PBR defaults.

    Values (0.5, 0.0) approximate the BRDF integral at mid-range NdotV
    and roughness, avoiding the over-bright result of (1.0, 1.0).
    """
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
    """Create a 1x1 dim grey cubemap for missing cubemap bindings.

    Uses (0.2, 0.2, 0.2) instead of black so metallic surfaces have
    minimal ambient light to reflect even without IBL assets.
    """
    data = np.full((6, face_size, face_size, 4), 0.2, dtype=np.float16)
    data[:, :, :, 3] = 1.0  # alpha = 1
    return _upload_cubemap_f16(device, face_size, num_faces=6, data=data)


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
    assets_dir = Path(_PROJECT_ROOT) / "assets" / "ibl" / ibl_name
    manifest_path = assets_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"[warn] IBL assets not found at {assets_dir}, using defaults")
        return {}

    with open(manifest_path) as f:
        manifest = _json.load(f)

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

def _bind_shadow_resources(gpu_scene, resources, reflected):
    """Add shadow map resources to the resource map if the shader expects them.

    Checks reflected binding info for "shadow_maps" (sampler + texture) and
    "shadow_matrices" (storage buffer) names and maps the gpu_scene's shadow
    resources accordingly.
    """
    if not gpu_scene.has_shadows:
        return

    for _set_idx, bindings in reflected._binding_info.items():
        for b in bindings:
            name = b["name"]
            btype = b["type"]

            if name == "shadow_maps":
                if btype == "sampler" and gpu_scene.shadow_sampler is not None:
                    resources["shadow_maps"] = (gpu_scene.shadow_sampler,
                                                 gpu_scene.shadow_array_view)
                elif btype == "sampled_image" and gpu_scene.shadow_array_view is not None:
                    resources["shadow_maps"] = (gpu_scene.shadow_sampler,
                                                 gpu_scene.shadow_array_view)

            elif name == "shadow_matrices" and btype == "storage_buffer":
                if gpu_scene.shadow_matrices_buffer is not None:
                    resources["shadow_matrices"] = gpu_scene.shadow_matrices_buffer


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
    for name, buf in gpu_scene.storage_buffers.items():
        resources[name] = buf
    for name, (sampler, view) in gpu_scene.textures.items():
        resources[name] = (sampler, view)

    # Bind shadow resources if available
    _bind_shadow_resources(gpu_scene, resources, reflected)

    # Identify which names need cube textures
    cube_names = set()
    for set_idx, bindings in reflected._binding_info.items():
        for b in bindings:
            if b["type"] == "sampled_cube_image":
                cube_names.add(b["name"])

    # Fill missing textures with defaults (cube or 2D as appropriate)
    # Different textures need different defaults:
    #   white (1,1,1): base_color_tex, occlusion_tex (AO=1 means no occlusion)
    #   flat normal (0.5, 0.5, 1.0): normal_tex
    #   black (0,0,0): emissive_tex (no emission by default)
    #   BRDF LUT (0.5, 0.0): brdf_lut
    #   grey (0.2): cubemaps
    default_tex = None
    default_cube = None
    default_brdf = None
    default_normal = None
    default_black = None
    _BLACK_TEX_NAMES = {"emissive_tex", "sheen_color_tex", "transmission_tex"}
    for set_idx, bindings in reflected._binding_info.items():
        for b in bindings:
            if b["type"] in ("sampler", "sampled_image", "sampled_cube_image") and b["name"] not in resources:
                if b["name"] in cube_names:
                    if default_cube is None:
                        default_cube = _create_default_cubemap(device)
                    resources[b["name"]] = default_cube
                elif b["name"] == "normal_tex":
                    if default_normal is None:
                        default_normal = _create_default_normal_texture(device)
                    resources[b["name"]] = default_normal
                elif b["name"] in _BLACK_TEX_NAMES:
                    if default_black is None:
                        default_black = _create_black_texture(device)
                    resources[b["name"]] = default_black
                elif b["name"] == "brdf_lut":
                    if default_brdf is None:
                        default_brdf = _create_default_brdf_lut(device)
                    resources[b["name"]] = default_brdf
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
            elif b["type"] == "storage_buffer" and b["name"] not in resources:
                size = b.get("size", 64)
                resources[b["name"]] = device.create_buffer_with_data(
                    data=bytes(max(size, 64)),
                    usage=wgpu.BufferUsage.STORAGE,
                )

    return reflected.create_bind_groups(device, resources)


# ---------------------------------------------------------------------------
# Multi-pipeline setup
# ---------------------------------------------------------------------------

@dataclass
class PermutationPipeline:
    """A single permutation's pipeline and associated resources."""
    suffix: str = ""
    base_path: str = ""
    material_indices: list = field(default_factory=list)
    pipeline: object = None       # wgpu.GPURenderPipeline
    reflected: object = None      # ReflectedPipeline
    per_material_bind_groups: list = field(default_factory=list)  # list of bind group dicts


def _build_per_material_resources(
    device: wgpu.GPUDevice,
    reflected,
    gpu_scene: GPUScene,
    material_index: int,
    scene: SceneData,
) -> dict:
    """Build a resource map for a single material's bind group (fragment set).

    Combines shared scene resources (IBL, global UBOs) with per-material
    textures and UBO for the given material index.
    """
    resources = {}

    # Shared uniform buffers (MVP, Light/Lighting)
    for name, buf in gpu_scene.uniform_buffers.items():
        if name != "Material":  # Material is per-material, not shared
            resources[name] = buf

    # Shared storage buffers (lights SSBO, etc.)
    for name, buf in gpu_scene.storage_buffers.items():
        resources[name] = buf

    # Per-material UBO
    if material_index < len(gpu_scene.per_material_ubos):
        resources["Material"] = gpu_scene.per_material_ubos[material_index]

    # IBL and other global textures (cubemaps, brdf_lut)
    for name, tex_pair in gpu_scene.textures.items():
        resources[name] = tex_pair

    # Override with per-material textures
    if material_index < len(gpu_scene.per_material_textures):
        for name, tex_pair in gpu_scene.per_material_textures[material_index].items():
            resources[name] = tex_pair

    # Bind shadow resources if available
    _bind_shadow_resources(gpu_scene, resources, reflected)

    # Identify which names need cube textures
    cube_names = set()
    for set_idx, bindings in reflected._binding_info.items():
        for b in bindings:
            if b["type"] == "sampled_cube_image":
                cube_names.add(b["name"])

    # Fill missing textures with semantically correct defaults
    default_tex = None
    default_cube = None
    default_brdf = None
    default_normal = None
    default_black = None
    _BLACK_TEX_NAMES = {"emissive_tex", "sheen_color_tex", "transmission_tex"}
    for set_idx, bindings in reflected._binding_info.items():
        for b in bindings:
            if b["type"] in ("sampler", "sampled_image", "sampled_cube_image") and b["name"] not in resources:
                if b["name"] in cube_names:
                    if default_cube is None:
                        default_cube = _create_default_cubemap(device)
                    resources[b["name"]] = default_cube
                elif b["name"] == "normal_tex":
                    if default_normal is None:
                        default_normal = _create_default_normal_texture(device)
                    resources[b["name"]] = default_normal
                elif b["name"] in _BLACK_TEX_NAMES:
                    if default_black is None:
                        default_black = _create_black_texture(device)
                    resources[b["name"]] = default_black
                elif b["name"] == "brdf_lut":
                    if default_brdf is None:
                        default_brdf = _create_default_brdf_lut(device)
                    resources[b["name"]] = default_brdf
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
            elif b["type"] == "storage_buffer" and b["name"] not in resources:
                size = b.get("size", 64)
                resources[b["name"]] = device.create_buffer_with_data(
                    data=bytes(max(size, 64)),
                    usage=wgpu.BufferUsage.STORAGE,
                )

    return resources


def setup_multi_pipeline(
    device: wgpu.GPUDevice,
    pipeline_base: str,
    manifest: dict,
    scene: SceneData,
    gpu_scene: GPUScene,
    color_format=wgpu.TextureFormat.rgba8unorm,
) -> list:
    """Set up multi-pipeline rendering: one pipeline per permutation group.

    Groups materials by their feature sets, creates a ReflectedPipeline for
    each permutation, and builds per-material bind groups.

    Returns a list of PermutationPipeline objects.
    """
    from reflected_pipeline import ReflectedPipeline, load_reflection
    from render_harness import load_shader_module

    # Build per-material texture name maps (just names, for feature detection)
    per_mat_tex_names = []
    for mat in scene.materials:
        per_mat_tex_names.append(mat.textures)

    # Group materials by features
    groups = group_materials_by_features(scene.materials, per_mat_tex_names)

    # Compute max vertex stride across all permutations (for shared vertex buffer)
    max_stride = 0
    for suffix in groups.keys():
        perm_base = pipeline_base + suffix if suffix else pipeline_base
        perm_vert_json = Path(perm_base + ".vert.json")
        if perm_vert_json.exists():
            with open(perm_vert_json) as f:
                perm_vr = _json.load(f)
            max_stride = max(max_stride, perm_vr.get("vertex_stride", 0))

    # Build material -> permutation index mapping
    total_materials = len(scene.materials)
    material_to_perm = [0] * total_materials

    permutations = []
    perm_idx = 0

    for suffix, material_indices in groups.items():
        # Build full path for this permutation
        perm_base = pipeline_base + suffix
        vert_path = Path(perm_base + ".vert.spv")
        frag_path = Path(perm_base + ".frag.spv")

        if not vert_path.exists() or not frag_path.exists():
            print(f"[warn] Missing shader for permutation '{suffix}', falling back to base")
            perm_base = pipeline_base
            vert_path = Path(pipeline_base + ".vert.spv")
            frag_path = Path(pipeline_base + ".frag.spv")
            suffix = ""

        vert_json = Path(perm_base + ".vert.json")
        frag_json = Path(perm_base + ".frag.json")

        vert_module = load_shader_module(device, vert_path)
        frag_module = load_shader_module(device, frag_path)
        vert_reflection = load_reflection(vert_json) if vert_json.exists() else {}
        frag_reflection = load_reflection(frag_json) if frag_json.exists() else {}

        reflected = ReflectedPipeline(
            device, vert_reflection, frag_reflection,
            vert_module, frag_module, color_format,
            stride_override=max_stride,
        )

        perm = PermutationPipeline(
            suffix=suffix,
            base_path=perm_base,
            material_indices=material_indices,
            pipeline=reflected.pipeline,
            reflected=reflected,
        )

        # Create per-material bind groups for each material in this permutation
        for mi in material_indices:
            resources = _build_per_material_resources(
                device, reflected, gpu_scene, mi, scene,
            )
            bind_groups = reflected.create_bind_groups(device, resources)
            perm.per_material_bind_groups.append(bind_groups)

        # Map materials to this permutation
        for mi in material_indices:
            material_to_perm[mi] = perm_idx

        permutations.append(perm)
        perm_idx += 1

        print(f"[info] Loaded permutation '{suffix or '(base)'}' from {perm_base}"
              f" ({len(material_indices)} material(s))")

    print(f"[info] Multi-pipeline: {len(permutations)} permutation(s)"
          f" for {total_materials} material(s)")

    return permutations


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
    camera_elevation: float = 5.0,
) -> np.ndarray:
    """Main rendering entry point: load scene, create pipeline, bind, render."""

    # Load scene
    scene = load_scene(scene_source, camera_elevation=camera_elevation)

    # Detect render path (use base pipeline to check for shader files)
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

    # Check for multi-pipeline mode: manifest exists AND multiple materials
    manifest = try_load_manifest(pipeline_base) if render_path == "raster" else None
    has_multiple_materials = len(scene.materials) > 1
    use_multi_pipeline = (manifest is not None
                          and len(manifest.get("permutations", [])) > 0
                          and has_multiple_materials)

    # Resolve effective pipeline base and vertex stride BEFORE uploading.
    # For multi-pipeline: compute max stride across all needed permutations.
    # For single-pipeline: resolve the feature suffix first, then read stride.
    effective_base = pipeline_base
    if use_multi_pipeline:
        # Determine max vertex stride across all permutations that will be used
        per_mat_tex = [mat.textures for mat in scene.materials]
        mat_groups = group_materials_by_features(scene.materials, per_mat_tex)
        pipeline_stride = 0
        for suffix, _ in mat_groups.items():
            perm_base = pipeline_base + suffix if suffix else pipeline_base
            perm_vert_json = Path(perm_base + ".vert.json")
            if perm_vert_json.exists():
                with open(perm_vert_json) as f:
                    perm_vr = _json.load(f)
                pipeline_stride = max(pipeline_stride, perm_vr.get("vertex_stride", 0))
    else:
        # Single-pipeline: resolve feature suffix first
        if scene.scene_features and "gltf_pbr_layered" in pipeline_base:
            suffix = "+".join(sorted(f.replace("has_", "") for f in scene.scene_features))
            candidate = f"shadercache/gltf_pbr_layered+{suffix}"
            if Path(candidate + ".frag.spv").exists():
                print(f"[info] Auto-selected pipeline variant: {candidate}")
                effective_base = candidate

        # Read vertex stride from the resolved pipeline
        vert_json = Path(effective_base).with_suffix(".vert.json")
        pipeline_stride = 0
        if vert_json.exists():
            with open(vert_json) as f:
                vert_refl = _json.load(f)
            pipeline_stride = vert_refl.get("vertex_stride", 0)

    # Pre-load reflection data for SSBO detection (multi-light, materials, etc.)
    reflection_data = {}
    resolve_base = effective_base if not use_multi_pipeline else pipeline_base
    for stage_suffix in [".vert.json", ".frag.json"]:
        rpath = Path(resolve_base + stage_suffix)
        if rpath.exists():
            with open(rpath, encoding="utf-8") as f:
                reflection_data[stage_suffix] = _json.load(f)

    # Phase 1: Upload scene to GPU (with correct stride for padding)
    print(f"Uploading scene '{scene_source}' to GPU...")
    gpu_scene = upload_scene(device, scene, width, height, pipeline_stride=pipeline_stride,
                             reflection_data=reflection_data)

    # Load IBL assets if available
    if ibl_name:
        ibl_textures = _load_ibl_assets(device, ibl_name)
        for name, tex in ibl_textures.items():
            gpu_scene.textures[name] = tex
    else:
        # Auto-detect: prefer "pisa" then "neutral", matching C++/Rust engines
        ibl_dir = Path(_PROJECT_ROOT) / "assets" / "ibl"
        if ibl_dir.exists():
            preferred = ["pisa", "neutral"]
            candidates = [d.name for d in ibl_dir.iterdir()
                          if d.is_dir() and (d / "manifest.json").exists()]
            ordered = [n for n in preferred if n in candidates]
            ordered += [n for n in sorted(candidates) if n not in preferred]
            if ordered:
                ibl_textures = _load_ibl_assets(device, ordered[0])
                for name, tex in ibl_textures.items():
                    gpu_scene.textures[name] = tex

    if use_multi_pipeline:
        # Multi-pipeline mode: one pipeline per material permutation group
        print(f"Creating multi-pipeline from '{pipeline_base}'...")
        permutations = setup_multi_pipeline(
            device, pipeline_base, manifest, scene, gpu_scene,
        )

        # Execute render pass with multi-pipeline
        print(f"Rendering {width}x{height} (multi-pipeline)...")
        pixels = _execute_render_multi_pipeline(
            device, permutations, gpu_scene, scene, width, height,
        )
    else:
        # Phase 2: Create pipeline from reflection
        print(f"Creating pipeline from '{effective_base}'...")
        gpu_pipeline, path, frag_refl, pipeline_info = create_pipeline(
            device, effective_base, render_path,
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

    # Render shadow maps before the main color pass
    _render_shadow_pass(device, encoder, gpu_scene)

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


def _execute_render_multi_pipeline(
    device, permutations, gpu_scene, scene, width, height,
) -> np.ndarray:
    """Execute a multi-pipeline render pass.

    Each draw range is rendered with the pipeline matching its material's
    permutation group. Per-material bind groups provide the correct Material
    UBO and textures for each draw call.
    """
    color_format = wgpu.TextureFormat.rgba8unorm

    # Render target
    texture = device.create_texture(
        size=(width, height, 1),
        format=color_format,
        usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
    )
    texture_view = texture.create_view()

    # Depth buffer
    depth_texture = device.create_texture(
        size=(width, height, 1),
        format=wgpu.TextureFormat.depth24plus,
        usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
    )
    depth_view = depth_texture.create_view()

    encoder = device.create_command_encoder()

    # Render shadow maps before the main color pass
    _render_shadow_pass(device, encoder, gpu_scene)

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

    # Build material -> permutation index mapping
    total_materials = len(scene.materials)
    material_to_perm = [0] * total_materials
    for perm_idx, perm in enumerate(permutations):
        for mi in perm.material_indices:
            if mi < total_materials:
                material_to_perm[mi] = perm_idx

    # If we have draw ranges, use them for proper multi-material rendering
    if gpu_scene.draw_ranges:
        # Each draw range maps 1:1 to a mesh (built in mesh order in upload_scene).
        # Each mesh has its own VBO/IBO in the Python engine.
        current_perm_idx = -1
        current_mat_idx = -1

        for mesh_idx, draw_range in enumerate(gpu_scene.draw_ranges):
            mat_idx = draw_range.material_index
            perm_idx = material_to_perm[mat_idx] if mat_idx < total_materials else 0
            perm = permutations[perm_idx]

            # Switch pipeline if permutation changed
            if perm_idx != current_perm_idx:
                current_perm_idx = perm_idx
                render_pass.set_pipeline(perm.pipeline)

            # Rebind material when material or permutation changes
            if mat_idx != current_mat_idx:
                current_mat_idx = mat_idx

                # Find which local index within this permutation's material list
                local_mat_idx = -1
                for j, mi in enumerate(perm.material_indices):
                    if mi == mat_idx:
                        local_mat_idx = j
                        break

                if local_mat_idx >= 0 and local_mat_idx < len(perm.per_material_bind_groups):
                    bind_groups = perm.per_material_bind_groups[local_mat_idx]
                    for set_idx in sorted(bind_groups.keys()):
                        render_pass.set_bind_group(set_idx, bind_groups[set_idx])

            # Bind vertex/index buffers for this mesh and draw
            if mesh_idx < len(gpu_scene.vertex_buffers):
                vbo, stride, num_verts = gpu_scene.vertex_buffers[mesh_idx]
                ibo, num_indices = gpu_scene.index_buffers[mesh_idx]
                if vbo:
                    render_pass.set_vertex_buffer(0, vbo)
                if ibo and num_indices > 0:
                    render_pass.set_index_buffer(ibo, wgpu.IndexFormat.uint32)
                    render_pass.draw_indexed(num_indices)
    else:
        # Fallback: draw meshes individually (no draw ranges)
        for i, (vbo, stride, num_verts) in enumerate(gpu_scene.vertex_buffers):
            ibo, num_indices = gpu_scene.index_buffers[i]
            mat_idx = scene.meshes[i].material_index if i < len(scene.meshes) else 0
            perm_idx = material_to_perm[mat_idx] if mat_idx < total_materials else 0
            perm = permutations[perm_idx]

            render_pass.set_pipeline(perm.pipeline)

            # Find local material index
            local_mat_idx = -1
            for j, mi in enumerate(perm.material_indices):
                if mi == mat_idx:
                    local_mat_idx = j
                    break

            if local_mat_idx >= 0 and local_mat_idx < len(perm.per_material_bind_groups):
                bind_groups = perm.per_material_bind_groups[local_mat_idx]
                for set_idx in sorted(bind_groups.keys()):
                    render_pass.set_bind_group(set_idx, bind_groups[set_idx])

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
    parser.add_argument("--ibl", default="", help="IBL environment name (from assets/ibl/)")
    parser.add_argument("--camera-elevation", type=float, default=5.0,
                        help="Camera elevation angle in degrees (0 = straight-on)")
    args = parser.parse_args()

    pipeline_base = resolve_pipeline(args.pipeline, args.scene)
    pixels = render(args.scene, pipeline_base, args.output, args.width, args.height,
                    ibl_name=args.ibl, camera_elevation=args.camera_elevation)

    # Stats
    non_black = (pixels[:, :, :3].sum(axis=2) > 10).sum()
    total = pixels.shape[0] * pixels.shape[1]
    print(f"Coverage: {non_black / total * 100:.1f}%")


if __name__ == "__main__":
    main()
