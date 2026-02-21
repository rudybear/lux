@echo off
REM Interactive mesh shader viewer — C++ Vulkan engine
REM Controls: mouse drag to orbit, scroll to zoom, ESC to exit

if not exist playground_cpp\build\Release\lux-playground.exe (
    echo C++ engine not built. Build with:
    echo   cd playground_cpp ^&^& cmake -B build ^&^& cmake --build build --config Release
    exit /b 1
)

set SCENE=assets/DamagedHelmet.glb
set IBL=pisa

REM Check for compiled mesh shader pipeline
if not exist shadercache\gltf_pbr_layered+emission.mesh.spv (
    echo No compiled mesh shaders found. Compile first with:
    echo   compile_mesh.bat
    exit /b 1
)

set PIPELINE=shadercache/gltf_pbr_layered+emission

echo === Interactive C++ Mesh Shader Viewer ===
echo   Scene:    %SCENE%
echo   Pipeline: %PIPELINE%
echo   IBL:      %IBL%
echo   Controls: mouse drag = orbit, scroll = zoom, ESC = exit
echo.

playground_cpp\build\Release\lux-playground.exe --scene %SCENE% --pipeline %PIPELINE% --ibl %IBL% --interactive
