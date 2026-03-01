"""Compute shader runner — dispatches a Lux compute shader and saves output as PNG.

Demonstrates compute → image pipeline: the compute shader writes to a storage
image, which is then read back to CPU and saved as a PNG file.

Usage:
    python run_compute.py <shader>.comp.spv -o output.png [--width 512] [--height 512]

    # Full pipeline from .lux source:
    luxc examples/compute_mandelbrot.lux --define workgroup_size_x=16 --define workgroup_size_y=16 -o build/compute
    python playground/run_compute.py build/compute/compute_mandelbrot.comp.spv -o mandelbrot.png

Dependencies:
    pip install wgpu Pillow numpy
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import wgpu
from PIL import Image

from render_harness import load_shader_module, save_png


def run_compute(
    comp_spv_path: Path,
    width: int = 512,
    height: int = 512,
    push_constants: dict[str, float] | None = None,
) -> np.ndarray:
    """Dispatch a compute shader that writes to a storage image.

    Returns an (H, W, 4) uint8 RGBA numpy array.
    """
    push_constants = push_constants or {}

    # --- GPU setup ---
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    if adapter is None:
        raise RuntimeError("No suitable GPU adapter found")
    device = adapter.request_device_sync()

    # --- Load compute shader ---
    comp_module = load_shader_module(device, comp_spv_path)

    # --- Storage image (output texture) ---
    storage_texture = device.create_texture(
        size=(width, height, 1),
        format=wgpu.TextureFormat.rgba8unorm,
        usage=(
            wgpu.TextureUsage.STORAGE_BINDING
            | wgpu.TextureUsage.COPY_SRC
        ),
    )
    storage_view = storage_texture.create_view()

    # --- Bind group layout + bind group ---
    bind_group_layout = device.create_bind_group_layout(
        entries=[
            wgpu.BindGroupLayoutEntry(
                binding=0,
                visibility=wgpu.ShaderStage.COMPUTE,
                storage_texture=wgpu.StorageTextureBindingLayout(
                    access=wgpu.StorageTextureAccess.write_only,
                    format=wgpu.TextureFormat.rgba8unorm,
                    view_dimension=wgpu.TextureViewDimension.d2,
                ),
            ),
        ],
    )

    bind_group = device.create_bind_group(
        layout=bind_group_layout,
        entries=[
            wgpu.BindGroupEntry(binding=0, resource=storage_view),
        ],
    )

    # --- Pipeline layout with push constants ---
    # Load reflection JSON to determine push constant size
    json_path = comp_spv_path.with_suffix("").with_suffix(".comp.json")
    push_size = 0
    push_fields = []
    if json_path.exists():
        reflection = json.loads(json_path.read_text())
        if reflection.get("push_constants"):
            pc = reflection["push_constants"][0]
            push_size = pc["size"]
            push_fields = pc["fields"]

    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout],
    )

    # --- Compute pipeline ---
    pipeline = device.create_compute_pipeline(
        layout=pipeline_layout,
        compute=wgpu.ProgrammableStage(
            module=comp_module,
            entry_point="main",
        ),
    )

    # --- Encode dispatch ---
    encoder = device.create_command_encoder()
    compute_pass = encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups(
        (width + 15) // 16,   # ceil(width / workgroup_size_x)
        (height + 15) // 16,  # ceil(height / workgroup_size_y)
        1,
    )
    compute_pass.end()

    # --- Readback: copy texture -> buffer ---
    bytes_per_row = width * 4
    bytes_per_row_aligned = (bytes_per_row + 255) & ~255

    readback_buffer = device.create_buffer(
        size=bytes_per_row_aligned * height,
        usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
    )
    encoder.copy_texture_to_buffer(
        wgpu.TexelCopyTextureInfo(texture=storage_texture),
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
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a Lux compute shader and save the storage image as PNG",
    )
    parser.add_argument("comp_spv", type=Path, help="Compute shader .comp.spv file")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("compute_output.png"),
        help="Output PNG path (default: compute_output.png)",
    )
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    args = parser.parse_args()

    if not args.comp_spv.exists():
        print(f"Error: compute shader not found: {args.comp_spv}", file=sys.stderr)
        sys.exit(1)

    print(f"Dispatching compute shader: {args.comp_spv}")
    print(f"Output size: {args.width}x{args.height}")

    pixels = run_compute(args.comp_spv, args.width, args.height)
    save_png(pixels, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
