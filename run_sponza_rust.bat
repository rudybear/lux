@echo off
REM Sponza Multi-Light Shadow Demo — Rust Vulkan engine
REM Controls: mouse drag to orbit, scroll = zoom, ESC to exit
REM
REM Features:
REM   - Sun directional light casting shadows through arches
REM   - Orbiting torch spot light with animated shadows
REM   - Blue accent point light

if not exist assets\Sponza.glb (
    echo Sponza model not found. Download with:
    echo   python download_sponza.py
    exit /b 1
)

if not exist playground_rust\target\release\lux-playground.exe (
    echo Rust engine not built. Build with:
    echo   cd playground_rust ^&^& cargo build --release
    exit /b 1
)

REM Compile multi-light shadow shaders if needed
if not exist shadercache\gltf_pbr_layered+normal_map+shadows.frag.spv (
    echo Compiling multi-light shadow shaders...
    python -m luxc examples/gltf_pbr_layered.lux --pipeline GltfForward --all-permutations -o shadercache/
)
if not exist shadercache\gltf_pbr_layered+normal_map+shadows.frag.spv (
    echo Failed to compile shaders.
    exit /b 1
)

set PIPELINE=shadercache/gltf_pbr_layered
set SCENE=assets/Sponza.glb

echo === Sponza Multi-Light Shadow Demo (Rust) ===
echo   Scene:    %SCENE%
echo   Pipeline: %PIPELINE% (raster, multi-light + shadows)
echo   Lights:   sun (shadow) + orbiting torch (shadow) + blue accent
echo   Controls: mouse drag = orbit, scroll = zoom, ESC = exit
echo.

playground_rust\target\release\lux-playground.exe --scene %SCENE% --pipeline %PIPELINE% --sponza-lights --interactive
