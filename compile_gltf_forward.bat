@echo off
REM Compile glTF PBR Forward (raster) pipeline
REM Uses the hand-written gltf_pbr.lux shader
echo === Compiling glTF PBR Forward (hand-written) ===
python -m luxc examples/gltf_pbr.lux -o shadercache/
echo.

REM Render with Python engine
echo === Rendering with Python engine ===
python -m playground.engine --scene assets/DamagedHelmet.glb --pipeline shadercache/gltf_pbr --output screenshots/test_gltf_forward.png --width 512 --height 512
echo.

REM Render with C++ engine (if built)
if exist playground_cpp\build\Release\lux-playground.exe (
    echo === Rendering with C++ engine ===
    playground_cpp\build\Release\lux-playground.exe --scene assets/DamagedHelmet.glb --pipeline shadercache/gltf_pbr --output screenshots/test_gltf_forward_cpp.png --width 512 --height 512
    echo.
)

REM Render with Rust engine (if built)
if exist playground_rust\target\release\lux-playground.exe (
    echo === Rendering with Rust engine ===
    playground_rust\target\release\lux-playground.exe --scene assets/DamagedHelmet.glb --pipeline shadercache/gltf_pbr --output screenshots/test_gltf_forward_rust.png --width 512 --height 512
    echo.
)

echo Done.
