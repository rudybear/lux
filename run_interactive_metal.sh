#!/bin/bash
# Interactive glTF PBR viewer — Metal engine (macOS)
# Controls: mouse drag to orbit, scroll to zoom, ESC to exit

METAL_BIN=playground_cpp/build-metal/lux-playground-metal

if [ ! -f "$METAL_BIN" ]; then
    echo "Metal engine not built. Build with:"
    echo "  cd playground_cpp && cmake -B build-metal && cmake --build build-metal --target lux-playground-metal"
    exit 1
fi

SCENE=assets/DamagedHelmet.glb
IBL=pisa

# Prefer manifest-based auto-selection, fall back to explicit permutation, then hand-written
if [ -f shadercache/gltf_pbr_layered.manifest.json ]; then
    PIPELINE=shadercache/gltf_pbr_layered
    echo "Using layered PBR raster pipeline (manifest auto-selection)"
elif [ -f shadercache/gltf_pbr_layered+emission+normal_map.frag.spv ]; then
    PIPELINE=shadercache/gltf_pbr_layered+emission+normal_map
    echo "Using layered PBR raster pipeline (explicit permutation)"
elif [ -f shadercache/gltf_pbr.frag.spv ]; then
    PIPELINE=shadercache/gltf_pbr
    echo "Using hand-written PBR raster pipeline"
else
    echo "No compiled shaders found. Compile first with:"
    echo "  python -m luxc examples/gltf_pbr_layered.lux -o shadercache/"
    exit 1
fi

echo "=== Interactive Metal Viewer ==="
echo "  Scene:    $SCENE"
echo "  Pipeline: $PIPELINE"
echo "  IBL:      $IBL"
echo "  Controls: mouse drag = orbit, scroll = zoom, ESC = exit"
echo

$METAL_BIN --scene $SCENE --pipeline $PIPELINE --ibl $IBL --interactive
