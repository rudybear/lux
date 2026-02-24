@echo off
REM Interactive SheenChair viewer — C++ Vulkan engine (Ray Tracing)
REM Controls: mouse drag to orbit, scroll to zoom, ESC to exit

if not exist playground_cpp\build\Release\lux-playground.exe (
    echo C++ engine not built. Build with:
    echo   cd playground_cpp ^&^& cmake -B build ^&^& cmake --build build --config Release
    exit /b 1
)

set SCENE=assets/SheenChair.glb
set IBL=pisa

REM Prefer bindless uber-shader (multi-material via gl_GeometryIndexEXT),
REM fall back to manifest-based permutations, then explicit sheen RT permutation
if exist shadercache\bindless\gltf_pbr_layered.rchit.spv (
    set PIPELINE=shadercache/bindless/gltf_pbr_layered
    echo Using bindless PBR RT pipeline (multi-material uber-shader^)
) else if exist shadercache\gltf_pbr_layered.manifest.json (
    set PIPELINE=shadercache/gltf_pbr_layered
    echo Using layered PBR RT pipeline (manifest auto-selection^)
) else if exist shadercache\gltf_pbr_layered+normal_map+sheen.rgen.spv (
    set PIPELINE=shadercache/gltf_pbr_layered+normal_map+sheen
    echo Using explicit sheen RT permutation
) else (
    echo No compiled RT shaders found. Compile first with:
    echo   compile_all.bat
    exit /b 1
)

echo === Interactive C++ Sheen RT Viewer ===
echo   Scene:    %SCENE%
echo   Pipeline: %PIPELINE%
echo   IBL:      %IBL%
echo   Controls: mouse drag = orbit, scroll = zoom, ESC = exit
echo.

playground_cpp\build\Release\lux-playground.exe --scene %SCENE% --pipeline %PIPELINE% --ibl %IBL% --mode rt --interactive
