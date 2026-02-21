@echo off
REM Compile glTF PBR Ray Tracing pipeline
REM Uses the hand-written gltf_pbr_rt.lux shader
echo === Compiling glTF PBR RT (hand-written) ===
python -m luxc examples/gltf_pbr_rt.lux -o shadercache/
echo.

REM Render with C++ engine (RT requires Vulkan RT extensions)
if exist playground_cpp\build\Release\lux-playground.exe (
    echo === Rendering with C++ engine (RT) ===
    playground_cpp\build\Release\lux-playground.exe --scene assets/DamagedHelmet.glb --pipeline shadercache/gltf_pbr_rt --output screenshots/test_gltf_rt_cpp.png --width 512 --height 512
    echo.
) else (
    echo C++ engine not built. Build with: cd playground_cpp ^&^& cmake -B build ^&^& cmake --build build --config Release
)

REM Render with Rust engine (RT requires Vulkan RT extensions)
if exist playground_rust\target\release\lux-playground.exe (
    echo === Rendering with Rust engine (RT) ===
    playground_rust\target\release\lux-playground.exe --scene assets/DamagedHelmet.glb --pipeline shadercache/gltf_pbr_rt --output screenshots/test_gltf_rt_rust.png --width 512 --height 512
    echo.
) else (
    echo Rust engine not built. Build with: cd playground_rust ^&^& cargo build --release
)

echo Note: Ray tracing requires C++ or Rust engine (Python/wgpu does not support RT)
echo Done.
