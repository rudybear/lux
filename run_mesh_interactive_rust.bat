@echo off
REM Interactive mesh shader viewer — Rust Vulkan engine
REM Controls: mouse drag to orbit, scroll to zoom, ESC to exit

if not exist playground_rust\target\release\lux-playground.exe (
    echo Rust engine not built. Build with:
    echo   cd playground_rust ^&^& cargo build --release
    exit /b 1
)

set SCENE=assets/DamagedHelmet.glb
set IBL=pisa

REM Prefer manifest-based auto-selection, fall back to explicit permutation
if exist shadercache\gltf_pbr_layered.manifest.json (
    set PIPELINE=shadercache/gltf_pbr_layered
    echo Using layered PBR mesh pipeline (manifest auto-selection^)
) else if exist shadercache\gltf_pbr_layered+emission.mesh.spv (
    set PIPELINE=shadercache/gltf_pbr_layered+emission
    echo Using layered PBR mesh pipeline (emission permutation^)
) else (
    echo No compiled mesh shaders found. Compile first with:
    echo   compile_all.bat
    exit /b 1
)

echo === Interactive Rust Mesh Shader Viewer ===
echo   Scene:    %SCENE%
echo   Pipeline: %PIPELINE%
echo   IBL:      %IBL%
echo   Controls: mouse drag = orbit, scroll = zoom, ESC = exit
echo.

playground_rust\target\release\lux-playground.exe --scene %SCENE% --pipeline %PIPELINE% --ibl %IBL% --mode mesh --interactive
