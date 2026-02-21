@echo off
REM Interactive glTF PBR viewer — C++ Vulkan engine
REM Controls: mouse drag to orbit, scroll to zoom, ESC to exit

if not exist playground_cpp\build\Release\lux-playground.exe (
    echo C++ engine not built. Build with:
    echo   cd playground_cpp ^&^& cmake -B build ^&^& cmake --build build --config Release
    exit /b 1
)

REM Default: RT mode with IBL
set PIPELINE=shadercache/gltf_pbr_rt
set SCENE=assets/DamagedHelmet.glb
set IBL=pisa

REM Fall back to raster if RT shaders not compiled
if not exist shadercache\gltf_pbr_rt.rgen.spv (
    echo RT shaders not found, falling back to raster mode
    set PIPELINE=shadercache/gltf_pbr
    if not exist shadercache\gltf_pbr.vert.spv (
        echo No compiled shaders found. Compile first with:
        echo   python -m luxc examples/gltf_pbr_rt.lux -o shadercache/
        exit /b 1
    )
)

echo === Interactive C++ Viewer ===
echo   Scene:    %SCENE%
echo   Pipeline: %PIPELINE%
echo   IBL:      %IBL%
echo   Controls: mouse drag = orbit, scroll = zoom, ESC = exit
echo.

playground_cpp\build\Release\lux-playground.exe --scene %SCENE% --pipeline %PIPELINE% --ibl %IBL% --interactive
