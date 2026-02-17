"""Interactive GLFW window preview for Lux shader playground.

Opens a resizable window that continuously renders a colored triangle
using the provided SPIR-V vertex and fragment shaders.

Usage:
    python preview.py <name>.vert.spv <name>.frag.spv

Dependencies:
    pip install wgpu glfw rendercanvas
"""

import argparse
import struct
import sys
from pathlib import Path

import wgpu
from rendercanvas.glfw import RenderCanvas, loop

# Re-use the shader loader from the render harness
from render_harness import load_shader_module


# ---------------------------------------------------------------------------
# Vertex data (same as render_harness)
# ---------------------------------------------------------------------------

TRIANGLE_VERTICES = [
    ( 0.0,  0.5, 0.0,   1.0, 0.0, 0.0),  # top,          red
    (-0.5, -0.5, 0.0,   0.0, 1.0, 0.0),  # bottom-left,  green
    ( 0.5, -0.5, 0.0,   0.0, 0.0, 1.0),  # bottom-right, blue
]

VERTEX_FORMAT = f"{len(TRIANGLE_VERTICES) * 6}f"
VERTEX_DATA = struct.pack(
    VERTEX_FORMAT,
    *[component for vertex in TRIANGLE_VERTICES for component in vertex],
)
VERTEX_STRIDE = 6 * 4  # 24 bytes


# ---------------------------------------------------------------------------
# Interactive preview
# ---------------------------------------------------------------------------

def run_preview(vert_spv_path: Path, frag_spv_path: Path) -> None:
    """Open a GLFW window and render the triangle each frame."""

    # --- Canvas & GPU setup ---
    canvas = RenderCanvas(title="Lux Shader Preview", size=(640, 480))
    context = canvas.get_context("wgpu")

    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    if adapter is None:
        raise RuntimeError("No suitable GPU adapter found")
    device = adapter.request_device_sync()

    present_format = context.get_preferred_format(adapter)
    context.configure(device=device, format=present_format)

    # --- Shaders ---
    vert_module = load_shader_module(device, vert_spv_path)
    frag_module = load_shader_module(device, frag_spv_path)

    # --- Vertex buffer ---
    vbo = device.create_buffer_with_data(
        data=VERTEX_DATA, usage=wgpu.BufferUsage.VERTEX
    )

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
                            shader_location=0,
                        ),
                        wgpu.VertexAttribute(
                            format=wgpu.VertexFormat.float32x3,
                            offset=12,
                            shader_location=1,
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
                wgpu.ColorTargetState(format=present_format),
            ],
        ),
    )

    # --- Draw callback ---
    def draw_frame():
        current_texture_view = context.get_current_texture().create_view()

        encoder = device.create_command_encoder()
        render_pass = encoder.begin_render_pass(
            color_attachments=[
                wgpu.RenderPassColorAttachment(
                    view=current_texture_view,
                    load_op=wgpu.LoadOp.clear,
                    store_op=wgpu.StoreOp.store,
                    clear_value=(0.05, 0.05, 0.05, 1.0),
                )
            ],
        )
        render_pass.set_pipeline(pipeline)
        render_pass.set_vertex_buffer(0, vbo)
        render_pass.draw(3)
        render_pass.end()

        device.queue.submit([encoder.finish()])

    canvas.request_draw(draw_frame)
    loop.run()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive GLFW preview of Lux SPIR-V shaders",
    )
    parser.add_argument("vert_spv", type=Path, help="Vertex shader .vert.spv file")
    parser.add_argument("frag_spv", type=Path, help="Fragment shader .frag.spv file")
    args = parser.parse_args()

    if not args.vert_spv.exists():
        print(f"Error: vertex shader not found: {args.vert_spv}", file=sys.stderr)
        sys.exit(1)
    if not args.frag_spv.exists():
        print(f"Error: fragment shader not found: {args.frag_spv}", file=sys.stderr)
        sys.exit(1)

    print(f"Opening preview with: {args.vert_spv}, {args.frag_spv}")
    print("Close the window or press Ctrl+C to exit.")
    run_preview(args.vert_spv, args.frag_spv)


if __name__ == "__main__":
    main()
