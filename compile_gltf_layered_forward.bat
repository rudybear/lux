@echo off
REM Compile glTF PBR Layered — Forward (raster) pipeline, all permutations
REM Uses the unified layered surface declaration with --all-permutations
echo === Compiling glTF PBR Layered (Forward, all permutations) ===
python -m luxc examples/gltf_pbr_layered.lux --pipeline GltfForward --all-permutations -o shadercache/
echo.

REM Render with Python engine (auto-selects permutation from manifest)
echo === Rendering with Python engine ===
python -m playground.engine --scene assets/DamagedHelmet.glb --pipeline shadercache/gltf_pbr_layered --output screenshots/test_gltf_layered_forward_python.png --width 512 --height 512
echo.

REM Render with C++ engine (if built)
if exist playground_cpp\build\Release\lux-playground.exe (
    echo === Rendering with C++ engine ===
    playground_cpp\build\Release\lux-playground.exe --scene assets/DamagedHelmet.glb --pipeline shadercache/gltf_pbr_layered --ibl pisa --output screenshots/test_gltf_layered_forward_cpp.png --width 512 --height 512
    echo.
)

REM Render with Rust engine (if built)
if exist playground_rust\target\release\lux-playground.exe (
    echo === Rendering with Rust engine ===
    playground_rust\target\release\lux-playground.exe --scene assets/DamagedHelmet.glb --pipeline shadercache/gltf_pbr_layered --ibl pisa --output screenshots/test_gltf_layered_forward_rust.png --width 512 --height 512
    echo.
)

echo Done.
