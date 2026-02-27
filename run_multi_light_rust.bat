@echo off
REM Interactive multi-light shadow demo — Rust Vulkan engine
REM 3 lights: warm directional (shadow), blue point, red spot (shadow)
REM Controls: mouse drag to orbit, scroll to zoom, ESC to exit

if not exist playground_rust\target\release\lux-playground.exe (
    echo Rust engine not built. Build with:
    echo   cd playground_rust ^&^& cargo build --release
    exit /b 1
)

set SCENE=assets/DamagedHelmet.glb
set IBL=pisa

REM Check for compiled shaders
if not exist shadercache\gltf_pbr_layered.manifest.json (
    echo No compiled shaders found. Compile first with:
    echo   compile_multi_light.bat
    exit /b 1
)

set PIPELINE=shadercache/gltf_pbr_layered

echo === Multi-Light Shadow Demo (Rust) ===
echo   Scene:    %SCENE%
echo   Pipeline: %PIPELINE% (auto-selects shadow permutation)
echo   IBL:      %IBL%
echo   Lights:   3 demo lights (directional+shadow, point, spot+shadow)
echo   Controls: mouse drag = orbit, scroll = zoom, ESC = exit
echo.

playground_rust\target\release\lux-playground.exe --scene %SCENE% --pipeline %PIPELINE% --ibl %IBL% --demo-lights --interactive
