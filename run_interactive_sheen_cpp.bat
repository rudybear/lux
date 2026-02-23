@echo off
REM Interactive SheenChair viewer — C++ Vulkan engine (Raster)
REM Controls: mouse drag to orbit, scroll to zoom, ESC to exit

if not exist playground_cpp\build\Release\lux-playground.exe (
    echo C++ engine not built. Build with:
    echo   cd playground_cpp ^&^& cmake -B build ^&^& cmake --build build --config Release
    exit /b 1
)

set SCENE=assets/SheenChair.glb
set IBL=pisa

REM Prefer manifest-based auto-selection (per-material permutations),
REM fall back to explicit sheen permutation
if exist shadercache\gltf_pbr_layered.manifest.json (
    set PIPELINE=shadercache/gltf_pbr_layered
    echo Using layered PBR raster pipeline (manifest auto-selection^)
) else if exist shadercache\gltf_pbr_layered+normal_map+sheen.frag.spv (
    set PIPELINE=shadercache/gltf_pbr_layered+normal_map+sheen
    echo Using explicit sheen permutation
) else (
    echo No compiled shaders found. Compile first with:
    echo   compile_all.bat
    exit /b 1
)

echo === Interactive C++ Sheen Viewer ===
echo   Scene:    %SCENE%
echo   Pipeline: %PIPELINE%
echo   IBL:      %IBL%
echo   Controls: mouse drag = orbit, scroll = zoom, ESC = exit
echo.

playground_cpp\build\Release\lux-playground.exe --scene %SCENE% --pipeline %PIPELINE% --ibl %IBL% --interactive
