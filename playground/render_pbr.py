"""PBR render harness for Lux shader playground.

Renders a 3D sphere with the PBR surface shader, using proper MVP
matrices, lighting via uniforms, normal data, and diffuse texture.

Usage:
    python render_pbr.py <name>.vert.spv <name>.frag.spv -o output.png

Dependencies:
    pip install wgpu Pillow numpy
"""

import argparse
import math
import struct
import sys
from pathlib import Path

import numpy as np
import wgpu
from PIL import Image

from render_harness import load_shader_module, save_png


# ---------------------------------------------------------------------------
# Matrix math (pure numpy, no external deps)
# ---------------------------------------------------------------------------

def perspective(fov_y: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Column-major perspective projection matrix (Vulkan clip space: y-down, z [0,1])."""
    f = 1.0 / math.tan(fov_y / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = -f  # Vulkan y-flip
    m[2, 2] = far / (near - far)
    m[2, 3] = -1.0
    m[3, 2] = (near * far) / (near - far)
    return m


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Column-major look-at view matrix."""
    f = target - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)

    m = np.eye(4, dtype=np.float32)
    m[0, 0] = s[0];  m[1, 0] = s[1];  m[2, 0] = s[2]
    m[0, 1] = u[0];  m[1, 1] = u[1];  m[2, 1] = u[2]
    m[0, 2] = -f[0]; m[1, 2] = -f[1]; m[2, 2] = -f[2]
    m[3, 0] = -np.dot(s, eye)
    m[3, 1] = -np.dot(u, eye)
    m[3, 2] = np.dot(f, eye)
    return m


# ---------------------------------------------------------------------------
# Sphere mesh generation
# ---------------------------------------------------------------------------

def generate_sphere(stacks: int = 32, slices: int = 32, radius: float = 1.0):
    """Generate a UV sphere. Returns (vertices, indices).

    Vertices: list of (px, py, pz, nx, ny, nz, u, v) — position + normal + UV.
    Indices: list of uint32 triangle indices.
    """
    vertices = []
    indices = []

    for i in range(stacks + 1):
        phi = math.pi * i / stacks
        v_coord = i / stacks
        for j in range(slices + 1):
            theta = 2.0 * math.pi * j / slices
            u_coord = j / slices
            x = math.sin(phi) * math.cos(theta)
            y = math.cos(phi)
            z = math.sin(phi) * math.sin(theta)
            vertices.extend([
                x * radius, y * radius, z * radius,  # position
                x, y, z,                               # normal
                u_coord, v_coord,                      # UV
            ])

    for i in range(stacks):
        for j in range(slices):
            a = i * (slices + 1) + j
            b = a + slices + 1
            indices.extend([a, b, a + 1])
            indices.extend([b, b + 1, a + 1])

    return vertices, indices


# ---------------------------------------------------------------------------
# Procedural texture generation
# ---------------------------------------------------------------------------

def generate_procedural_texture(size: int = 512) -> np.ndarray:
    """Generate a colorful procedural texture. Returns (size, size, 4) uint8 RGBA."""
    img = np.zeros((size, size, 4), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            u = x / size
            v = y / size
            # Checker pattern with two color palettes
            cx = int(u * 8) % 2
            cy = int(v * 8) % 2
            checker = cx ^ cy
            if checker:
                # Warm: terracotta / orange tones
                r = int(200 + 40 * math.sin(u * 12.0))
                g = int(120 + 30 * math.sin(v * 10.0))
                b = int(80 + 20 * math.cos(u * 8.0 + v * 6.0))
            else:
                # Cool: teal / blue-green tones
                r = int(60 + 30 * math.sin(u * 10.0 + v * 4.0))
                g = int(150 + 40 * math.cos(v * 8.0))
                b = int(170 + 50 * math.sin(u * 6.0))
            img[y, x] = [min(255, max(0, r)), min(255, max(0, g)), min(255, max(0, b)), 255]
    return img


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_pbr(
    vert_spv_path: Path,
    frag_spv_path: Path,
    width: int = 512,
    height: int = 512,
) -> np.ndarray:
    """Render a PBR-lit sphere and return an (H, W, 4) uint8 RGBA array."""

    # --- GPU setup ---
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    if adapter is None:
        raise RuntimeError("No suitable GPU adapter found")
    device = adapter.request_device_sync()

    # --- Shaders ---
    vert_module = load_shader_module(device, vert_spv_path)
    frag_module = load_shader_module(device, frag_spv_path)

    # --- Mesh data (position + normal + UV = 8 floats per vertex) ---
    sphere_verts, sphere_indices = generate_sphere(stacks=32, slices=32)
    vertex_data = struct.pack(f"{len(sphere_verts)}f", *sphere_verts)
    index_data = struct.pack(f"{len(sphere_indices)}I", *sphere_indices)
    num_indices = len(sphere_indices)

    vbo = device.create_buffer_with_data(data=vertex_data, usage=wgpu.BufferUsage.VERTEX)
    ibo = device.create_buffer_with_data(data=index_data, usage=wgpu.BufferUsage.INDEX)

    # --- MVP uniform buffer ---
    eye = np.array([0.0, 0.0, 3.0], dtype=np.float32)
    target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    model = np.eye(4, dtype=np.float32)
    view = look_at(eye, target, up)
    proj = perspective(math.radians(45.0), width / height, 0.1, 100.0)

    # Pack as column-major (numpy default) — 3 x mat4 = 192 bytes
    mvp_data = model.tobytes() + view.tobytes() + proj.tobytes()
    mvp_buffer = device.create_buffer_with_data(
        data=mvp_data,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )

    # --- Light uniform buffer (std140: vec3 padded to 16 bytes each) ---
    light_dir = np.array([0.5, 0.8, 1.0], dtype=np.float32)
    light_dir = light_dir / np.linalg.norm(light_dir)
    light_data = struct.pack("3f", *light_dir) + struct.pack("f", 0.0)  # pad
    light_data += struct.pack("3f", *eye) + struct.pack("f", 0.0)       # pad
    light_buffer = device.create_buffer_with_data(
        data=light_data,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )

    # --- Procedural albedo texture ---
    print("Generating procedural texture...")
    tex_data = generate_procedural_texture(512)
    tex_size = tex_data.shape[1]  # square

    albedo_texture = device.create_texture(
        size=(tex_size, tex_size, 1),
        format=wgpu.TextureFormat.rgba8unorm,
        usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
    )
    device.queue.write_texture(
        wgpu.TexelCopyTextureInfo(texture=albedo_texture),
        tex_data.tobytes(),
        wgpu.TexelCopyBufferLayout(
            offset=0,
            bytes_per_row=tex_size * 4,
            rows_per_image=tex_size,
        ),
        (tex_size, tex_size, 1),
    )
    albedo_view = albedo_texture.create_view()
    albedo_sampler = device.create_sampler(
        mag_filter=wgpu.FilterMode.linear,
        min_filter=wgpu.FilterMode.linear,
    )

    # --- Bind group layouts ---
    # Set 0: vertex MVP uniform
    mvp_bind_group_layout = device.create_bind_group_layout(
        entries=[
            wgpu.BindGroupLayoutEntry(
                binding=0,
                visibility=wgpu.ShaderStage.VERTEX,
                buffer=wgpu.BufferBindingLayout(type=wgpu.BufferBindingType.uniform),
            ),
        ]
    )
    # Set 1: fragment Light uniform (binding=0) + albedo texture (binding=1)
    frag_bind_group_layout = device.create_bind_group_layout(
        entries=[
            wgpu.BindGroupLayoutEntry(
                binding=0,
                visibility=wgpu.ShaderStage.FRAGMENT,
                buffer=wgpu.BufferBindingLayout(type=wgpu.BufferBindingType.uniform),
            ),
            wgpu.BindGroupLayoutEntry(
                binding=1,
                visibility=wgpu.ShaderStage.FRAGMENT,
                sampler=wgpu.SamplerBindingLayout(type=wgpu.SamplerBindingType.filtering),
            ),
            wgpu.BindGroupLayoutEntry(
                binding=2,
                visibility=wgpu.ShaderStage.FRAGMENT,
                texture=wgpu.TextureBindingLayout(
                    sample_type=wgpu.TextureSampleType.float,
                    view_dimension=wgpu.TextureViewDimension.d2,
                ),
            ),
        ]
    )

    mvp_bind_group = device.create_bind_group(
        layout=mvp_bind_group_layout,
        entries=[
            wgpu.BindGroupEntry(binding=0, resource=wgpu.BufferBinding(buffer=mvp_buffer, size=192)),
        ],
    )
    frag_bind_group = device.create_bind_group(
        layout=frag_bind_group_layout,
        entries=[
            wgpu.BindGroupEntry(binding=0, resource=wgpu.BufferBinding(buffer=light_buffer, size=32)),
            wgpu.BindGroupEntry(binding=1, resource=albedo_sampler),
            wgpu.BindGroupEntry(binding=2, resource=albedo_view),
        ],
    )

    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[mvp_bind_group_layout, frag_bind_group_layout],
    )

    # --- Render target ---
    texture = device.create_texture(
        size=(width, height, 1),
        format=wgpu.TextureFormat.rgba8unorm,
        usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
    )
    texture_view = texture.create_view()

    # --- Depth buffer ---
    depth_texture = device.create_texture(
        size=(width, height, 1),
        format=wgpu.TextureFormat.depth24plus,
        usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
    )
    depth_view = depth_texture.create_view()

    # --- Render pipeline ---
    pipeline = device.create_render_pipeline(
        layout=pipeline_layout,
        vertex=wgpu.VertexState(
            module=vert_module,
            entry_point="main",
            buffers=[
                wgpu.VertexBufferLayout(
                    array_stride=32,  # 8 floats = 32 bytes
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
                            shader_location=1,  # normal
                        ),
                        wgpu.VertexAttribute(
                            format=wgpu.VertexFormat.float32x2,
                            offset=24,
                            shader_location=2,  # uv
                        ),
                    ],
                ),
            ],
        ),
        primitive=wgpu.PrimitiveState(
            topology=wgpu.PrimitiveTopology.triangle_list,
            cull_mode=wgpu.CullMode.back,
            front_face=wgpu.FrontFace.ccw,
        ),
        depth_stencil=wgpu.DepthStencilState(
            format=wgpu.TextureFormat.depth24plus,
            depth_write_enabled=True,
            depth_compare=wgpu.CompareFunction.less,
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
                clear_value=(0.05, 0.05, 0.08, 1.0),  # dark background
            ),
        ],
        depth_stencil_attachment=wgpu.RenderPassDepthStencilAttachment(
            view=depth_view,
            depth_load_op=wgpu.LoadOp.clear,
            depth_store_op=wgpu.StoreOp.store,
            depth_clear_value=1.0,
        ),
    )

    render_pass.set_pipeline(pipeline)
    render_pass.set_bind_group(0, mvp_bind_group)
    render_pass.set_bind_group(1, frag_bind_group)
    render_pass.set_vertex_buffer(0, vbo)
    render_pass.set_index_buffer(ibo, wgpu.IndexFormat.uint32)
    render_pass.draw_indexed(num_indices)
    render_pass.end()

    # --- Readback ---
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

    readback_buffer.map_sync(wgpu.MapMode.READ)
    raw = readback_buffer.read_mapped()
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(height, bytes_per_row_aligned)
    arr = arr[:, : width * 4].reshape(height, width, 4).copy()
    readback_buffer.unmap()
    return arr


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a PBR-lit sphere using compiled Lux SPIR-V shaders",
    )
    parser.add_argument("vert_spv", type=Path, help="Vertex shader .vert.spv file")
    parser.add_argument("frag_spv", type=Path, help="Fragment shader .frag.spv file")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("pbr_output.png"),
        help="Output PNG path (default: pbr_output.png)",
    )
    parser.add_argument("--width", type=int, default=512, help="Render width")
    parser.add_argument("--height", type=int, default=512, help="Render height")
    args = parser.parse_args()

    if not args.vert_spv.exists():
        print(f"Error: vertex shader not found: {args.vert_spv}", file=sys.stderr)
        sys.exit(1)
    if not args.frag_spv.exists():
        print(f"Error: fragment shader not found: {args.frag_spv}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading shaders: {args.vert_spv}, {args.frag_spv}")
    pixels = render_pbr(
        args.vert_spv, args.frag_spv,
        width=args.width, height=args.height,
    )

    save_png(pixels, args.output)
    print(f"Saved {args.width}x{args.height} PBR render to {args.output}")

    # Stats
    non_black = (pixels[:, :, :3].sum(axis=2) > 10).sum()
    total = pixels.shape[0] * pixels.shape[1]
    print(f"Sphere coverage: {non_black / total * 100:.1f}%")


if __name__ == "__main__":
    main()
