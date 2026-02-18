"""Offscreen render harness for Lux shader playground.

Loads compiled SPIR-V vertex and fragment shaders, renders a colored
triangle to a 512x512 offscreen texture, and saves the result as PNG.

Usage:
    python render_harness.py <name>.vert.spv <name>.frag.spv -o output.png

Dependencies:
    pip install wgpu Pillow numpy
"""

import argparse
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


def save_png(pixels: np.ndarray, output_path: Path) -> None:
    """Save an (H, W, 4) uint8 RGBA array as a PNG file."""
    img = Image.fromarray(pixels, "RGBA")
    img.save(str(output_path))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a triangle using compiled Lux SPIR-V shaders",
    )
    parser.add_argument("vert_spv", type=Path, help="Vertex shader .vert.spv file")
    parser.add_argument("frag_spv", type=Path, help="Fragment shader .frag.spv file")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("output.png"),
        help="Output PNG path (default: output.png)",
    )
    parser.add_argument(
        "--width", type=int, default=512, help="Render width (default: 512)",
    )
    parser.add_argument(
        "--height", type=int, default=512, help="Render height (default: 512)",
    )
    args = parser.parse_args()

    if not args.vert_spv.exists():
        print(f"Error: vertex shader not found: {args.vert_spv}", file=sys.stderr)
        sys.exit(1)
    if not args.frag_spv.exists():
        print(f"Error: fragment shader not found: {args.frag_spv}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading shaders: {args.vert_spv}, {args.frag_spv}")
    pixels = render_triangle(
        args.vert_spv, args.frag_spv,
        width=args.width, height=args.height,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_png(pixels, args.output)
    print(f"Saved {args.width}x{args.height} render to {args.output}")

    # Quick sanity check
    non_black = (pixels[:, :, :3].sum(axis=2) > 0).sum()
    total = pixels.shape[0] * pixels.shape[1]
    coverage = non_black / total * 100
    print(f"Triangle coverage: {coverage:.1f}% ({non_black} non-black pixels)")


if __name__ == "__main__":
    main()
