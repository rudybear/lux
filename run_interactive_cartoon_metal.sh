#!/bin/bash
# Interactive Cartoon Toon raster viewer — Metal engine (macOS)
# Controls: mouse drag to orbit, scroll to zoom, ESC to exit

METAL_BIN=playground_cpp/build-metal/lux-playground-metal

if [ ! -f "$METAL_BIN" ]; then
    echo "Metal engine not built. Build with:"
    echo "  cd playground_cpp && cmake -B build-metal && cmake --build build-metal --target lux-playground-metal"
    exit 1
fi

# Compile cartoon_toon forward if needed
if [ ! -f shadercache/cartoon_toon.frag.spv ]; then
    echo "Compiling cartoon_toon forward shaders..."
    python -m luxc examples/cartoon_toon.lux -o shadercache/ --pipeline ToonForward
fi
if [ ! -f shadercache/cartoon_toon.frag.spv ]; then
    echo "Failed to compile shaders."
    exit 1
fi

PIPELINE=shadercache/cartoon_toon
SCENE=assets/DamagedHelmet.glb
IBL=pisa

echo "=== Interactive Metal Cartoon Raster Viewer ==="
echo "  Scene:    $SCENE"
echo "  Pipeline: $PIPELINE (raster)"
echo "  IBL:      $IBL"
echo "  Controls: mouse drag = orbit, scroll = zoom, ESC = exit"
echo

$METAL_BIN --scene $SCENE --pipeline $PIPELINE --ibl $IBL --interactive
