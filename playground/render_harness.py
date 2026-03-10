"""Offscreen render harness for Lux shader playground.

Loads compiled SPIR-V vertex and fragment shaders, renders a colored
triangle to a 512x512 offscreen texture, and saves the result as PNG.

Usage:
    python render_harness.py <name>.vert.spv <name>.frag.spv -o output.png

Dependencies:
    pip install wgpu Pillow numpy
"""

import argparse
import json
import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import wgpu
from PIL import Image


# ---------------------------------------------------------------------------
# Shader loading helpers
# ---------------------------------------------------------------------------

def _try_spirv_to_wgsl(spv_path: Path) -> str | None:
    """Attempt SPIR-V -> WGSL conversion using naga-cli (if available).

    spirv-cross does NOT support WGSL output, so we look for naga-cli
    which ships with some wgpu/Rust toolchains.  Returns None on failure.
    """
    for tool in ("naga", "naga-cli"):
        try:
            with tempfile.NamedTemporaryFile(suffix=".wgsl", delete=False) as tmp:
                tmp_path = tmp.name
            result = subprocess.run(
                [tool, str(spv_path), tmp_path],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                wgsl = Path(tmp_path).read_text(encoding="utf-8")
                Path(tmp_path).unlink(missing_ok=True)
                return wgsl
            Path(tmp_path).unlink(missing_ok=True)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


def load_shader_module(device: wgpu.GPUDevice, spv_path: Path) -> wgpu.GPUShaderModule:
    """Load a SPIR-V binary as a wgpu shader module.

    Primary path: pass raw SPIR-V bytes to wgpu (supported by wgpu-native
    0.29.x via the spirv feature).  Fallback: convert to WGSL via naga-cli.
    """
    spv_bytes = spv_path.read_bytes()

    # Verify SPIR-V magic number
    if len(spv_bytes) < 4:
        raise ValueError(f"{spv_path}: file too small to be valid SPIR-V")
    magic = struct.unpack("<I", spv_bytes[:4])[0]
    if magic != 0x07230203:
        raise ValueError(
            f"{spv_path}: bad SPIR-V magic 0x{magic:08X} "
            f"(expected 0x07230203)"
        )

    # Try loading SPIR-V directly
    try:
        module = device.create_shader_module(code=spv_bytes)
        return module
    except Exception as exc:
        msg = str(exc).encode("ascii", errors="replace").decode("ascii")
        print(f"[warn] direct SPIR-V load failed ({msg}); trying WGSL fallback...")

    # Fallback: SPIR-V -> WGSL via naga-cli
    wgsl = _try_spirv_to_wgsl(spv_path)
    if wgsl is not None:
        print(f"[info] converted {spv_path.name} to WGSL via naga")
        return device.create_shader_module(code=wgsl)

    raise RuntimeError(
        f"Cannot load {spv_path}: wgpu rejected the SPIR-V and no WGSL "
        f"converter is available.  Install naga-cli or use a wgpu version "
        f"with SPIR-V support."
    )


# ---------------------------------------------------------------------------
# GLB / Gaussian splat loading
# ---------------------------------------------------------------------------

def load_splat_glb(path: Path) -> dict:
    """Parse a GLB file with KHR_gaussian_splatting extension.

    Returns a dict with numpy arrays: positions (N,3), rotations (N,4),
    scales (N,3), opacities (N,), sh_coeffs list of arrays, sh_degree int,
    and num_splats int.
    """
    data = path.read_bytes()
    magic, version, length = struct.unpack_from('<III', data, 0)
    if magic != 0x46546C67:
        raise ValueError(f"Not a GLB file: bad magic 0x{magic:08X}")

    # JSON chunk
    json_len, json_type = struct.unpack_from('<II', data, 12)
    if json_type != 0x4E4F534A:
        raise ValueError("GLB: first chunk is not JSON")
    gltf = json.loads(data[20:20 + json_len])

    # BIN chunk
    bin_off = 20 + json_len
    bin_len, bin_type = struct.unpack_from('<II', data, bin_off)
    if bin_type != 0x004E4942:
        raise ValueError("GLB: second chunk is not BIN")
    bin_data = data[bin_off + 8:bin_off + 8 + bin_len]

    accessors = gltf['accessors']
    buffer_views = gltf['bufferViews']

    def read_accessor(idx):
        acc = accessors[idx]
        bv = buffer_views[acc['bufferView']]
        offset = bv.get('byteOffset', 0)
        count = acc['count']
        components = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}[acc['type']]
        arr = np.frombuffer(bin_data, dtype=np.float32,
                            count=count * components, offset=offset)
        return arr.reshape(count, components) if components > 1 else arr.copy()

    prim = gltf['meshes'][0]['primitives'][0]
    gs = prim['extensions']['KHR_gaussian_splatting']

    result = {
        'positions': read_accessor(prim['attributes']['POSITION']),
        'rotations': read_accessor(gs['attributes']['ROTATION']),
        'scales': read_accessor(gs['attributes']['SCALE']),
        'opacities': read_accessor(gs['attributes']['OPACITY']),
        'sh_coeffs': [read_accessor(e['coefficients']) for e in gs.get('sh', [])],
        'sh_degree': max((e['degree'] for e in gs.get('sh', [])), default=0),
        'num_splats': accessors[prim['attributes']['POSITION']]['count'],
    }
    return result


# ---------------------------------------------------------------------------
# Camera math helpers (numpy)
# ---------------------------------------------------------------------------

def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Compute a 4x4 view matrix (column-major, OpenGL convention)."""
    f = target - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[3, :3] = 0.0
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


def _perspective(fov_y: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Compute a 4x4 perspective matrix (Vulkan Y-flip applied)."""
    t = math.tan(fov_y / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = 1.0 / (aspect * t)
    m[1, 1] = -1.0 / t  # Vulkan Y-flip
    m[2, 2] = far / (near - far)
    m[2, 3] = (near * far) / (near - far)
    m[3, 2] = -1.0
    return m


# ---------------------------------------------------------------------------
# Vertex data
# ---------------------------------------------------------------------------

# Hardcoded colored triangle in clip space (no MVP transform needed).
# Each vertex: position (vec3) + color (vec3) = 6 x float32
TRIANGLE_VERTICES = [
    # position (x, y, z),     color (r, g, b)
    ( 0.0,  0.5, 0.0,        1.0, 0.0, 0.0),   # top,          red
    (-0.5, -0.5, 0.0,        0.0, 1.0, 0.0),   # bottom-left,  green
    ( 0.5, -0.5, 0.0,        0.0, 0.0, 1.0),   # bottom-right, blue
]

VERTEX_FORMAT = f"{len(TRIANGLE_VERTICES) * 6}f"
VERTEX_DATA = struct.pack(
    VERTEX_FORMAT,
    *[component for vertex in TRIANGLE_VERTICES for component in vertex],
)
FLOATS_PER_VERTEX = 6
VERTEX_STRIDE = FLOATS_PER_VERTEX * 4  # 24 bytes


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_triangle(
    vert_spv_path: Path,
    frag_spv_path: Path,
    width: int = 512,
    height: int = 512,
) -> np.ndarray:
    """Render one frame and return an (H, W, 4) uint8 RGBA numpy array."""

    # --- GPU setup ---
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    if adapter is None:
        raise RuntimeError("No suitable GPU adapter found")
    device = adapter.request_device_sync()

    # --- Shaders ---
    vert_module = load_shader_module(device, vert_spv_path)
    frag_module = load_shader_module(device, frag_spv_path)

    # --- Vertex buffer ---
    vbo = device.create_buffer_with_data(
        data=VERTEX_DATA, usage=wgpu.BufferUsage.VERTEX
    )

    # --- Render target texture ---
    texture = device.create_texture(
        size=(width, height, 1),
        format=wgpu.TextureFormat.rgba8unorm,
        usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
    )
    texture_view = texture.create_view()

    # --- Render pipeline ---
    pipeline = device.create_render_pipeline(
        layout="auto",
        vertex=wgpu.VertexState(
            module=vert_module,
            entry_point="main",
            buffers=[
                wgpu.VertexBufferLayout(
                    array_stride=VERTEX_STRIDE,
                    step_mode=wgpu.VertexStepMode.vertex,
                    attributes=[
                        wgpu.VertexAttribute(
                            format=wgpu.VertexFormat.float32x3,
                            offset=0,
                            shader_location=0,  # position
                        ),
                        wgpu.VertexAttribute(
                            format=wgpu.VertexFormat.float32x3,
                            offset=12,
                            shader_location=1,  # color
                        ),
                    ],
                )
            ],
        ),
        primitive=wgpu.PrimitiveState(
            topology=wgpu.PrimitiveTopology.triangle_list,
        ),
        multisample=wgpu.MultisampleState(count=1, mask=0xFFFFFFFF),
        fragment=wgpu.FragmentState(
            module=frag_module,
            entry_point="main",
            targets=[
                wgpu.ColorTargetState(format=wgpu.TextureFormat.rgba8unorm),
            ],
        ),
    )

    # --- Encode render pass ---
    encoder = device.create_command_encoder()
    render_pass = encoder.begin_render_pass(
        color_attachments=[
            wgpu.RenderPassColorAttachment(
                view=texture_view,
                load_op=wgpu.LoadOp.clear,
                store_op=wgpu.StoreOp.store,
                clear_value=(0.0, 0.0, 0.0, 1.0),
            )
        ],
    )
    render_pass.set_pipeline(pipeline)
    render_pass.set_vertex_buffer(0, vbo)
    render_pass.draw(3)
    render_pass.end()

    # --- Readback: copy texture -> buffer ---
    bytes_per_row = width * 4
    # wgpu requires bytes_per_row aligned to 256
    bytes_per_row_aligned = (bytes_per_row + 255) & ~255

    readback_buffer = device.create_buffer(
        size=bytes_per_row_aligned * height,
        usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
    )
    encoder.copy_texture_to_buffer(
        wgpu.TexelCopyTextureInfo(texture=texture),
        wgpu.TexelCopyBufferInfo(
            buffer=readback_buffer,
            offset=0,
            bytes_per_row=bytes_per_row_aligned,
            rows_per_image=height,
        ),
        (width, height, 1),
    )
    device.queue.submit([encoder.finish()])

    # --- Map buffer and extract pixels ---
    readback_buffer.map_sync(wgpu.MapMode.READ)
    raw = readback_buffer.read_mapped()

    # Handle potential row padding
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(height, bytes_per_row_aligned)
    arr = arr[:, : width * 4].reshape(height, width, 4).copy()

    readback_buffer.unmap()
    return arr


# ---------------------------------------------------------------------------
# Fullscreen quad rendering (fragment-only shaders)
# ---------------------------------------------------------------------------

# Built-in WGSL vertex shader for fullscreen coverage.
# Draws a single triangle that covers the entire screen and passes
# UV coordinates (0,0)-(1,1) to the fragment shader at location 0.
_FULLSCREEN_VERT_WGSL = """
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    // Fullscreen triangle: 3 vertices that cover [-1,1] clip space
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    let p = positions[idx];
    var out: VertexOutput;
    out.position = vec4<f32>(p, 0.0, 1.0);
    // Map clip coords to UV: [-1,1] -> [0,1], with Y flipped so
    // UV (0,0) = top-left to match Vulkan/wgpu convention
    out.uv = vec2<f32>(p.x * 0.5 + 0.5, 1.0 - (p.y * 0.5 + 0.5));
    return out;
}
"""


def render_fullscreen(
    frag_spv_path: Path,
    width: int = 512,
    height: int = 512,
) -> np.ndarray:
    """Render a fullscreen quad with the given fragment shader.

    The fragment shader must accept `in uv: vec2` at location 0.
    Returns an (H, W, 4) uint8 RGBA numpy array.
    """

    # --- GPU setup ---
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    if adapter is None:
        raise RuntimeError("No suitable GPU adapter found")
    device = adapter.request_device_sync()

    # --- Shaders ---
    vert_module = device.create_shader_module(code=_FULLSCREEN_VERT_WGSL)
    frag_module = load_shader_module(device, frag_spv_path)

    # --- Render target texture ---
    texture = device.create_texture(
        size=(width, height, 1),
        format=wgpu.TextureFormat.rgba8unorm,
        usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
    )
    texture_view = texture.create_view()

    # --- Render pipeline (no vertex buffers needed) ---
    pipeline = device.create_render_pipeline(
        layout="auto",
        vertex=wgpu.VertexState(
            module=vert_module,
            entry_point="main",
            buffers=[],
        ),
        primitive=wgpu.PrimitiveState(
            topology=wgpu.PrimitiveTopology.triangle_list,
        ),
        multisample=wgpu.MultisampleState(count=1, mask=0xFFFFFFFF),
        fragment=wgpu.FragmentState(
            module=frag_module,
            entry_point="main",
            targets=[
                wgpu.ColorTargetState(format=wgpu.TextureFormat.rgba8unorm),
            ],
        ),
    )

    # --- Encode render pass ---
    encoder = device.create_command_encoder()
    render_pass = encoder.begin_render_pass(
        color_attachments=[
            wgpu.RenderPassColorAttachment(
                view=texture_view,
                load_op=wgpu.LoadOp.clear,
                store_op=wgpu.StoreOp.store,
                clear_value=(0.0, 0.0, 0.0, 1.0),
            )
        ],
    )
    render_pass.set_pipeline(pipeline)
    render_pass.draw(3)  # 3 vertices for fullscreen triangle
    render_pass.end()

    # --- Readback: copy texture -> buffer ---
    bytes_per_row = width * 4
    bytes_per_row_aligned = (bytes_per_row + 255) & ~255

    readback_buffer = device.create_buffer(
        size=bytes_per_row_aligned * height,
        usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
    )
    encoder.copy_texture_to_buffer(
        wgpu.TexelCopyTextureInfo(texture=texture),
        wgpu.TexelCopyBufferInfo(
            buffer=readback_buffer,
            offset=0,
            bytes_per_row=bytes_per_row_aligned,
            rows_per_image=height,
        ),
        (width, height, 1),
    )
    device.queue.submit([encoder.finish()])

    # --- Map buffer and extract pixels ---
    readback_buffer.map_sync(wgpu.MapMode.READ)
    raw = readback_buffer.read_mapped()

    arr = np.frombuffer(raw, dtype=np.uint8).reshape(height, bytes_per_row_aligned)
    arr = arr[:, : width * 4].reshape(height, width, 4).copy()

    readback_buffer.unmap()
    return arr


# ---------------------------------------------------------------------------
# Gaussian splat rendering
# ---------------------------------------------------------------------------

def render_splats(
    comp_spv_path: Path,
    vert_spv_path: Path,
    frag_spv_path: Path,
    splat_glb_path: Path,
    width: int = 512,
    height: int = 512,
    eye: np.ndarray = None,
    target: np.ndarray = None,
) -> np.ndarray:
    """Render gaussian splats and return an (H, W, 4) uint8 RGBA array.

    Pure CPU/numpy implementation matching the GPU pipeline exactly:
    compute projection → CPU depth sort → rasterize 2D Gaussians.
    """
    splat = load_splat_glb(splat_glb_path)
    return render_splats_from_data(splat, width, height, eye=eye, target=target)


def render_splats_from_data(
    splat: dict,
    width: int = 512,
    height: int = 512,
    eye: np.ndarray = None,
    target: np.ndarray = None,
) -> np.ndarray:
    """Render preloaded splat data and return an (H, W, 4) uint8 RGBA array."""
    n = splat['num_splats']

    positions = splat['positions']      # (N, 3)
    rotations = splat['rotations']      # (N, 4) XYZW
    scales_log = splat['scales']        # (N, 3) log-space
    opacities_logit = splat['opacities']  # (N,) logit-space
    sh_coeffs = splat['sh_coeffs']      # list of (N, 3) arrays

    # --- Camera setup (matches C++/Rust auto-camera) ---
    bb_min, bb_max = positions.min(axis=0), positions.max(axis=0)
    center = (bb_min + bb_max) / 2.0
    radius = float(np.linalg.norm(bb_max - bb_min)) * 0.5
    if radius < 0.001:
        radius = 1.0

    if target is None:
        target = center
    if eye is None:
        eye = center + np.array([0.0, radius * 0.5, radius * 2.5], dtype=np.float32)

    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    fov_y = math.radians(45.0)
    aspect = width / height
    near_plane = 0.01
    far_plane = radius * 10.0

    view = _look_at(eye, target, up)
    proj = _perspective(fov_y, aspect, near_plane, far_plane)

    focal_y = 0.5 * height / math.tan(fov_y / 2.0)
    focal_x = focal_y  # square pixels

    # --- Preprocess each splat (CPU equivalent of compute shader) ---
    # Arrays for projected splat data
    proj_centers = np.zeros((n, 4), dtype=np.float32)  # ndc_x, ndc_y, ndc_z, radius
    proj_conics = np.zeros((n, 4), dtype=np.float32)   # conic_x, conic_y, conic_z, opacity
    proj_colors = np.zeros((n, 4), dtype=np.float32)   # r, g, b, opacity
    view_depths = np.zeros(n, dtype=np.float32)

    SH_C0 = 0.28209479177387814

    for i in range(n):
        pos = positions[i]
        # Transform to view space
        vp = view @ np.append(pos, 1.0)
        vx, vy, vz = vp[:3]

        # Near-plane cull
        if vz > -0.1:
            view_depths[i] = 1e6
            continue

        # Positive depth
        t = -vz
        t2 = t * t

        # Quaternion to rotation matrix (XYZW = scalar-last)
        qi, qj, qk, qr = rotations[i]
        qi2, qj2, qk2 = qi*qi, qj*qj, qk*qk
        qri, qrj, qrk = qr*qi, qr*qj, qr*qk
        qij, qik, qjk = qi*qj, qi*qk, qj*qk

        R = np.array([
            [1 - 2*(qj2+qk2), 2*(qij-qrk),   2*(qik+qrj)],
            [2*(qij+qrk),     1 - 2*(qi2+qk2), 2*(qjk-qri)],
            [2*(qik-qrj),     2*(qjk+qri),     1 - 2*(qi2+qj2)],
        ], dtype=np.float64)

        # Scale (exp of log-scale)
        s = np.exp(scales_log[i]).astype(np.float64)

        # 3D covariance: Sigma = R @ diag(s^2) @ R^T
        S2 = s * s
        cov3d = np.zeros((3, 3), dtype=np.float64)
        for a in range(3):
            for b in range(a, 3):
                val = sum(R[a, k] * R[b, k] * S2[k] for k in range(3))
                cov3d[a, b] = val
                cov3d[b, a] = val

        # Jacobian of projection
        j00 = focal_x / t
        j02 = -focal_x * vx / t2
        j11 = focal_y / t
        j12 = -focal_y * vy / t2

        # J = [[j00, 0, j02], [0, j11, j12]]
        # 2D covariance = J @ cov3d @ J^T
        c00 = j00*j00*cov3d[0,0] + 2*j00*j02*cov3d[0,2] + j02*j02*cov3d[2,2]
        c01 = j00*j11*cov3d[0,1] + j00*j12*cov3d[0,2] + j02*j11*cov3d[1,2] + j02*j12*cov3d[2,2]
        c11 = j11*j11*cov3d[1,1] + 2*j11*j12*cov3d[1,2] + j12*j12*cov3d[2,2]

        # Low-pass filter
        c00 += 0.3
        c11 += 0.3

        det = c00 * c11 - c01 * c01
        if det <= 0:
            view_depths[i] = 1e6
            continue

        inv_det = 1.0 / det
        conic_x = float(c11 * inv_det)
        conic_y = float(-c01 * inv_det)
        conic_z = float(c00 * inv_det)

        # Radius from eigenvalues
        mid = 0.5 * (c00 + c11)
        half_diff = math.sqrt(max(mid*mid - det, 0.0))
        lambda_max = mid + half_diff
        splat_radius = math.ceil(3.0 * math.sqrt(lambda_max))

        # Project to NDC
        clip = proj @ np.array([vx, vy, vz, 1.0], dtype=np.float64)
        ndc_x = float(clip[0] / clip[3])
        ndc_y = float(clip[1] / clip[3])
        ndc_z = float(clip[2] / clip[3])

        # Opacity (sigmoid)
        opacity = 1.0 / (1.0 + math.exp(-float(opacities_logit[i])))

        # SH color (degree 0)
        if sh_coeffs:
            sh0 = sh_coeffs[0][i]
            color = np.clip(SH_C0 * sh0 + 0.5, 0, 1)
        else:
            color = np.array([1.0, 1.0, 1.0])

        proj_centers[i] = [ndc_x, ndc_y, ndc_z, splat_radius]
        proj_conics[i] = [conic_x, conic_y, conic_z, opacity]
        proj_colors[i] = [color[0], color[1], color[2], opacity]
        view_depths[i] = vz  # negative = in front

    # --- Sort back-to-front ---
    sorted_indices = np.argsort(view_depths)  # most negative first = farthest

    # --- Rasterize (premultiplied alpha, back-to-front, matches GPU exactly) ---
    # GPU clears to opaque black (0,0,0,1)
    fb = np.zeros((height, width, 3), dtype=np.float64)
    alpha_cutoff = 0.004
    srgb = True  # match the default gaussian_splat.lux color_space: srgb

    for idx in sorted_indices:
        ndc_x, ndc_y, ndc_z, splat_radius = proj_centers[idx]
        conic_x, conic_y, conic_z, opacity = proj_conics[idx]
        cr, cg, cb, _ = proj_colors[idx]

        if splat_radius <= 0 or opacity <= 0:
            continue

        # NDC to pixel center
        px = (ndc_x * 0.5 + 0.5) * width
        py = (ndc_y * 0.5 + 0.5) * height

        r = int(splat_radius)
        x0 = max(0, int(px - r))
        x1 = min(width, int(px + r) + 1)
        y0 = max(0, int(py - r))
        y1 = min(height, int(py + r) + 1)

        if x0 >= x1 or y0 >= y1:
            continue

        # Vectorized Gaussian evaluation over the bounding box
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
        dx = xx - px
        dy = yy - py

        power = -0.5 * (conic_x * dx*dx + 2*conic_y * dx*dy + conic_z * dy*dy)

        # Discard fragments outside Gaussian tail
        gauss = np.where(power > -4.0, np.exp(power), 0.0)
        alpha = gauss * opacity
        alpha = np.where(alpha >= alpha_cutoff, alpha, 0.0)

        if not np.any(alpha > 0):
            continue

        # sRGB conversion (linear → sRGB)
        if srgb:
            cr = cr ** 0.45454545 if cr > 0 else 0.0
            cg = cg ** 0.45454545 if cg > 0 else 0.0
            cb = cb ** 0.45454545 if cb > 0 else 0.0

        # GPU premultiplied alpha blending: dst = src_premul + (1 - src_a) * dst
        # src_premul.rgb = color.rgb * alpha
        one_minus_a = 1.0 - alpha
        region = fb[y0:y1, x0:x1]
        region[:, :, 0] = cr * alpha + one_minus_a * region[:, :, 0]
        region[:, :, 1] = cg * alpha + one_minus_a * region[:, :, 1]
        region[:, :, 2] = cb * alpha + one_minus_a * region[:, :, 2]

    # Convert to uint8 RGBA (opaque output, matches GPU clear alpha=1)
    fb = np.clip(fb, 0, 1)
    result = np.zeros((height, width, 4), dtype=np.uint8)
    result[:, :, :3] = (fb * 255).astype(np.uint8)
    result[:, :, 3] = 255
    return result


def interactive_splat_viewer(
    splat_glb_path: Path,
    width: int = 512,
    height: int = 512,
) -> None:
    """Open an interactive matplotlib window for orbiting around a splat scene.

    Controls: drag to orbit, scroll to zoom, ESC/Q to exit.
    """
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import MouseButton

    splat = load_splat_glb(splat_glb_path)
    n = splat['num_splats']
    print(f"Loaded {n} splats (SH degree {splat['sh_degree']})")

    positions = splat['positions']
    bb_min, bb_max = positions.min(axis=0), positions.max(axis=0)
    center = (bb_min + bb_max) / 2.0
    radius = float(np.linalg.norm(bb_max - bb_min)) * 0.5
    if radius < 0.001:
        radius = 1.0

    # Orbit camera state: azimuth, elevation, distance
    state = {
        'azimuth': 0.0,       # radians
        'elevation': 0.2,     # radians
        'distance': radius * 3.0,
        'dragging': False,
        'last_x': 0,
        'last_y': 0,
    }

    def _eye_from_orbit():
        az, el, dist = state['azimuth'], state['elevation'], state['distance']
        x = dist * math.cos(el) * math.sin(az)
        y = dist * math.sin(el)
        z = dist * math.cos(el) * math.cos(az)
        return center + np.array([x, y, z], dtype=np.float32)

    def _render():
        eye = _eye_from_orbit()
        return render_splats_from_data(splat, width, height, eye=eye, target=center)

    # Initial render
    print("Rendering initial view...")
    pixels = _render()

    fig, ax = plt.subplots(1, 1, figsize=(width / 100, height / 100), dpi=100)
    fig.canvas.manager.set_window_title('Lux Gaussian Splat Viewer')
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    img_plot = ax.imshow(pixels)

    def _refresh():
        print("Re-rendering...", end=' ', flush=True)
        pixels = _render()
        img_plot.set_data(pixels)
        fig.canvas.draw_idle()
        print("done.")

    def on_press(event):
        if event.button == MouseButton.LEFT:
            state['dragging'] = True
            state['last_x'] = event.x
            state['last_y'] = event.y

    def on_release(event):
        if event.button == MouseButton.LEFT and state['dragging']:
            state['dragging'] = False
            _refresh()

    def on_motion(event):
        if not state['dragging'] or event.x is None or event.y is None:
            return
        dx = event.x - state['last_x']
        dy = event.y - state['last_y']
        state['last_x'] = event.x
        state['last_y'] = event.y
        state['azimuth'] -= dx * 0.01
        state['elevation'] = max(-math.pi / 2 + 0.01,
                                  min(math.pi / 2 - 0.01,
                                      state['elevation'] + dy * 0.01))

    def on_scroll(event):
        factor = 0.9 if event.step > 0 else 1.1
        state['distance'] = max(radius * 0.5, state['distance'] * factor)
        _refresh()

    def on_key(event):
        if event.key in ('escape', 'q'):
            plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('key_press_event', on_key)

    print("Interactive viewer: drag to orbit, scroll to zoom, ESC/Q to exit")
    plt.show()


def save_png(pixels: np.ndarray, output_path: Path) -> None:
    """Save an (H, W, 4) uint8 RGBA array as a PNG file."""
    img = Image.fromarray(pixels, "RGBA")
    img.save(str(output_path))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render compiled Lux SPIR-V shaders offscreen",
    )
    parser.add_argument("vert_spv", nargs='?', type=Path,
                        help="Vertex shader .vert.spv (triangle mode)")
    parser.add_argument("frag_spv", nargs='?', type=Path,
                        help="Fragment shader .frag.spv (triangle mode)")
    parser.add_argument("-o", "--output", type=Path, default=Path("output.png"),
                        help="Output PNG path (default: output.png)")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    # Splat mode
    parser.add_argument("--splat-comp", type=Path, help="Compute shader .comp.spv")
    parser.add_argument("--splat-vert", type=Path, help="Vertex shader .vert.spv")
    parser.add_argument("--splat-frag", type=Path, help="Fragment shader .frag.spv")
    parser.add_argument("--splat-scene", type=Path, help="Splat scene .glb file")
    parser.add_argument("--interactive", action="store_true",
                        help="Open interactive orbit viewer (splat mode only)")
    # Fullscreen mode
    parser.add_argument("--fullscreen", type=Path,
                        help="Fragment shader for fullscreen quad mode")
    args = parser.parse_args()

    is_splat = args.splat_scene is not None

    if is_splat:
        if not args.splat_scene.exists():
            print(f"Error: --splat-scene not found: {args.splat_scene}", file=sys.stderr)
            sys.exit(1)
        if args.interactive:
            interactive_splat_viewer(
                args.splat_scene, width=args.width, height=args.height)
            sys.exit(0)
        print(f"Splat mode (CPU): scene={args.splat_scene.name}")
        pixels = render_splats(
            args.splat_comp, args.splat_vert, args.splat_frag, args.splat_scene,
            width=args.width, height=args.height,
        )
    elif args.fullscreen:
        if not args.fullscreen.exists():
            print(f"Error: shader not found: {args.fullscreen}", file=sys.stderr)
            sys.exit(1)
        print(f"Fullscreen mode: {args.fullscreen}")
        pixels = render_fullscreen(
            args.fullscreen, width=args.width, height=args.height)
    else:
        if args.vert_spv is None or args.frag_spv is None:
            parser.error("Provide vert_spv and frag_spv, or use --splat-* / --fullscreen")
        if not args.vert_spv.exists():
            print(f"Error: vertex shader not found: {args.vert_spv}", file=sys.stderr)
            sys.exit(1)
        if not args.frag_spv.exists():
            print(f"Error: fragment shader not found: {args.frag_spv}", file=sys.stderr)
            sys.exit(1)
        print(f"Triangle mode: {args.vert_spv}, {args.frag_spv}")
        pixels = render_triangle(
            args.vert_spv, args.frag_spv,
            width=args.width, height=args.height,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_png(pixels, args.output)
    print(f"Saved {args.width}x{args.height} render to {args.output}")

    non_black = (pixels[:, :, :3].sum(axis=2) > 0).sum()
    total = pixels.shape[0] * pixels.shape[1]
    coverage = non_black / total * 100
    print(f"Coverage: {coverage:.1f}% ({non_black} non-black pixels)")


if __name__ == "__main__":
    main()
