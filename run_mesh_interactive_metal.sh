#!/bin/bash
# Interactive mesh shader viewer — Metal engine (macOS, requires Metal 3 / Apple7+ GPU)
# Controls: mouse drag to orbit, scroll to zoom, ESC to exit

METAL_BIN=playground_cpp/build-metal/lux-playground-metal

if [ ! -f "$METAL_BIN" ]; then
    echo "Metal engine not built. Build with:"
    echo "  cd playground_cpp && cmake -B build-metal && cmake --build build-metal --target lux-playground-metal"
    exit 1
fi

SCENE=assets/DamagedHelmet.glb
IBL=pisa

# Check for compiled mesh shader pipeline
if [ ! -f "shadercache/gltf_pbr_layered+emission.mesh.spv" ]; then
    echo "No compiled mesh shaders found. Compile first with:"
    echo "  python -m luxc examples/gltf_pbr_layered.lux -o shadercache/ --pipeline MeshPBR"
    exit 1
fi

PIPELINE="shadercache/gltf_pbr_layered+emission"

echo "=== Interactive Metal Mesh Shader Viewer ==="
echo "  Scene:    $SCENE"
echo "  Pipeline: $PIPELINE"
echo "  IBL:      $IBL"
echo "  Controls: mouse drag = orbit, scroll = zoom, ESC = exit"
echo

$METAL_BIN --scene $SCENE --pipeline $PIPELINE --ibl $IBL --mode mesh --interactive
