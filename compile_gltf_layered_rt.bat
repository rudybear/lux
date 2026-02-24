@echo off
REM Compile glTF PBR Layered — Ray Tracing pipeline, all permutations
REM Uses the unified layered surface declaration with --all-permutations
echo === Compiling glTF PBR Layered - RT, all permutations ===
python -m luxc examples/gltf_pbr_layered.lux --pipeline GltfRT --all-permutations -o shadercache/
echo.

REM Render with C++ engine - RT requires Vulkan RT extensions
if exist playground_cpp\build\Release\lux-playground.exe (
    echo === Rendering with C++ engine - RT ===
    playground_cpp\build\Release\lux-playground.exe --scene assets/DamagedHelmet.glb --pipeline shadercache/gltf_pbr_layered --ibl pisa --mode rt --output screenshots/test_gltf_layered_rt_cpp.png --width 512 --height 512
    echo.
) else (
    echo C++ engine not built. Build with: cd playground_cpp ^&^& cmake -B build ^&^& cmake --build build --config Release
)

REM Render with Rust engine - RT requires Vulkan RT extensions
if exist playground_rust\target\release\lux-playground.exe (
    echo === Rendering with Rust engine - RT ===
    playground_rust\target\release\lux-playground.exe --scene assets/DamagedHelmet.glb --pipeline shadercache/gltf_pbr_layered --ibl pisa --mode rt --output screenshots/test_gltf_layered_rt_rust.png --width 512 --height 512
    echo.
) else (
    echo Rust engine not built. Build with: cd playground_rust ^&^& cargo build --release
)

echo Note: Ray tracing requires C++ or Rust engine - Python/wgpu does not support RT
echo Done.
