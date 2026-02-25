#!/bin/bash
# Headless mesh shader rendering — Metal engine (macOS)
# Renders and saves PNG to screenshots/ directory

METAL_BIN=playground_cpp/build-metal/lux-playground-metal

if [ ! -f "$METAL_BIN" ]; then
    echo "Metal engine not built. Build with:"
    echo "  cd playground_cpp && cmake -B build-metal && cmake --build build-metal --target lux-playground-metal"
    exit 1
fi

if [ ! -f "shadercache/gltf_pbr_layered+emission.mesh.spv" ]; then
    echo "No compiled mesh shaders found. Compile first with:"
    echo "  python -m luxc examples/gltf_pbr_layered.lux -o shadercache/ --pipeline MeshPBR"
    exit 1
fi

mkdir -p screenshots

echo "=== Rendering with Metal Mesh Shader Engine ==="
$METAL_BIN --scene assets/DamagedHelmet.glb --pipeline "shadercache/gltf_pbr_layered+emission" --ibl pisa --output screenshots/mesh_helmet_metal.png --width 512 --height 512
echo
echo "Done. Output saved to screenshots/mesh_helmet_metal.png"
