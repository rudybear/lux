@echo off
REM Interactive glTF PBR viewer — Rust Vulkan engine (Ray Tracing)
REM Controls: mouse drag to orbit, scroll to zoom, ESC to exit

if not exist playground_rust\target\release\lux-playground.exe (
    echo Rust engine not built. Build with:
    echo   cd playground_rust ^&^& cargo build --release
    exit /b 1
)

set SCENE=assets/DamagedHelmet.glb
set IBL=pisa

REM Prefer layered RT pipeline, fall back to hand-written RT
if exist shadercache\gltf_pbr_layered+emission+normal_map.rgen.spv (
    set PIPELINE=shadercache/gltf_pbr_layered+emission+normal_map
    echo Using layered PBR RT pipeline
) else if exist shadercache\gltf_pbr_rt.rgen.spv (
    set PIPELINE=shadercache/gltf_pbr_rt
    echo Using hand-written PBR RT pipeline
) else (
    echo No compiled RT shaders found. Compile first with:
    echo   python -m luxc examples/gltf_pbr_layered.lux -o shadercache/ --pipeline GltfRT --features has_normal_map,has_emission
    exit /b 1
)

echo === Interactive Rust RT Viewer ===
echo   Scene:    %SCENE%
echo   Pipeline: %PIPELINE%
echo   IBL:      %IBL%
echo   Controls: mouse drag = orbit, scroll = zoom, ESC = exit
echo.

playground_rust\target\release\lux-playground.exe --scene %SCENE% --pipeline %PIPELINE% --ibl %IBL% --interactive
