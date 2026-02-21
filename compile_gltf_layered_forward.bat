@echo off
REM Compile glTF PBR Layered — Forward (raster) pipeline only
REM Uses the new unified layered surface declaration with features
echo === Compiling glTF PBR Layered (Forward, normal_map+emission) ===
python -m luxc examples/gltf_pbr_layered.lux --pipeline GltfForward --features has_normal_map,has_emission -o shadercache/
echo.

REM Render with Python engine
echo === Rendering with Python engine ===
python -m playground.engine --scene assets/DamagedHelmet.glb --pipeline shadercache/gltf_pbr_layered+emission+normal_map --output screenshots/test_gltf_layered_forward_python.png --width 512 --height 512
echo.

REM Render with C++ engine (if built)
if exist playground_cpp\build\Release\lux-playground.exe (
    echo === Rendering with C++ engine ===
    playground_cpp\build\Release\lux-playground.exe --scene assets/DamagedHelmet.glb --pipeline shadercache/gltf_pbr_layered+emission+normal_map --ibl pisa --output screenshots/test_gltf_layered_forward_cpp.png --width 512 --height 512
    echo.
)

REM Render with Rust engine (if built)
if exist playground_rust\target\release\lux-playground.exe (
    echo === Rendering with Rust engine ===
    playground_rust\target\release\lux-playground.exe --scene assets/DamagedHelmet.glb --pipeline shadercache/gltf_pbr_layered+emission+normal_map --ibl pisa --output screenshots/test_gltf_layered_forward_rust.png --width 512 --height 512
    echo.
)

echo Done.
