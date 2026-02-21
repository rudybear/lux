@echo off
REM Headless mesh shader rendering with Rust Vulkan engine
REM Renders and saves PNG to screenshots/ directory

if not exist playground_rust\target\release\lux-playground.exe (
    echo Rust engine not built. Build with:
    echo   cd playground_rust ^&^& cargo build --release
    exit /b 1
)

if not exist shadercache\gltf_pbr_layered+emission.mesh.spv (
    echo No compiled mesh shaders found. Compile first with:
    echo   compile_mesh.bat
    exit /b 1
)

echo === Rendering with Rust Mesh Shader Engine ===
playground_rust\target\release\lux-playground.exe --scene assets/DamagedHelmet.glb --pipeline shadercache/gltf_pbr_layered+emission --ibl pisa --output screenshots/mesh_helmet_rust.png --width 512 --height 512
echo.
echo Done. Output saved to screenshots/mesh_helmet_rust.png
