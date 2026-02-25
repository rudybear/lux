#!/bin/bash
# Interactive SheenChair viewer — Metal engine (macOS, Raster)
# Controls: mouse drag to orbit, scroll to zoom, ESC to exit

METAL_BIN=playground_cpp/build-metal/lux-playground-metal

if [ ! -f "$METAL_BIN" ]; then
    echo "Metal engine not built. Build with:"
    echo "  cd playground_cpp && cmake -B build-metal && cmake --build build-metal --target lux-playground-metal"
    exit 1
fi

SCENE=assets/SheenChair.glb
IBL=pisa

# Prefer manifest-based auto-selection (per-material permutations),
# fall back to explicit sheen permutation
if [ -f shadercache/gltf_pbr_layered.manifest.json ]; then
    PIPELINE=shadercache/gltf_pbr_layered
    echo "Using layered PBR raster pipeline (manifest auto-selection)"
elif [ -f "shadercache/gltf_pbr_layered+normal_map+sheen.frag.spv" ]; then
    PIPELINE="shadercache/gltf_pbr_layered+normal_map+sheen"
    echo "Using explicit sheen permutation"
else
    echo "No compiled shaders found. Compile first with:"
    echo "  python -m luxc examples/gltf_pbr_layered.lux -o shadercache/"
    exit 1
fi

echo "=== Interactive Metal Sheen Viewer ==="
echo "  Scene:    $SCENE"
echo "  Pipeline: $PIPELINE"
echo "  IBL:      $IBL"
echo "  Controls: mouse drag = orbit, scroll = zoom, ESC = exit"
echo

$METAL_BIN --scene $SCENE --pipeline $PIPELINE --ibl $IBL --interactive
