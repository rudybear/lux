@echo off
REM Interactive SheenChair viewer — Rust Vulkan engine (Raster)
REM Controls: mouse drag to orbit, scroll to zoom, ESC to exit

if not exist playground_rust\target\release\lux-playground.exe (
    echo Rust engine not built. Build with:
    echo   cd playground_rust ^&^& cargo build --release
    exit /b 1
)

set SCENE=assets/SheenChair.glb
set PIPELINE=shadercache/gltf_pbr_layered+normal_map+sheen
set IBL=pisa

if not exist shadercache\gltf_pbr_layered+normal_map+sheen.frag.spv (
    echo No compiled sheen shaders found. Compile first with:
    echo   python -m luxc examples/gltf_pbr_layered.lux --pipeline GltfForward --features has_normal_map,has_sheen -o shadercache/
    exit /b 1
)

echo === Interactive Rust Sheen Viewer ===
echo   Scene:    %SCENE%
echo   Pipeline: %PIPELINE%
echo   IBL:      %IBL%
echo   Controls: mouse drag = orbit, scroll = zoom, ESC = exit
echo.

playground_rust\target\release\lux-playground.exe --scene %SCENE% --pipeline %PIPELINE% --ibl %IBL% --interactive
