@echo off
REM Headless mesh shader rendering with C++ Vulkan engine
REM Renders and saves PNG to screenshots/ directory

if not exist playground_cpp\build\Release\lux-playground.exe (
    echo C++ engine not built. Build with:
    echo   cd playground_cpp ^&^& cmake -B build ^&^& cmake --build build --config Release
    exit /b 1
)

if not exist shadercache\gltf_pbr_layered+emission.mesh.spv (
    echo No compiled mesh shaders found. Compile first with:
    echo   compile_mesh.bat
    exit /b 1
)

echo === Rendering with C++ Mesh Shader Engine ===
playground_cpp\build\Release\lux-playground.exe --scene assets/DamagedHelmet.glb --pipeline shadercache/gltf_pbr_layered+emission --ibl pisa --output screenshots/mesh_helmet_cpp.png --width 512 --height 512
echo.
echo Done. Output saved to screenshots/mesh_helmet_cpp.png
