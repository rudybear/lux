@echo off
REM Multi-light shadow demo — headless rendering for all engines
REM Compiles shaders, then renders screenshots with C++, Rust, and Python engines

echo ============================================================
echo  Lux: Multi-Light Shadow Demo — Headless Test
echo ============================================================
echo.

REM Step 1: Compile shaders
echo [Step 1] Compiling shaders...
python -m luxc examples/gltf_pbr_layered.lux --pipeline GltfForward --all-permutations -o shadercache/
echo.

set SCENE=assets/DamagedHelmet.glb
set PIPELINE=shadercache/gltf_pbr_layered
set IBL=pisa
set W=512
set H=512

REM Step 2: Python engine
echo [Step 2] Rendering with Python engine...
python -m playground.engine --scene %SCENE% --pipeline %PIPELINE% --output screenshots/test_multi_light_python.png --width %W% --height %H%
echo.

REM Step 3: C++ engine
if exist playground_cpp\build\Release\lux-playground.exe (
    echo [Step 3] Rendering with C++ engine (demo lights)...
    playground_cpp\build\Release\lux-playground.exe --scene %SCENE% --pipeline %PIPELINE% --ibl %IBL% --demo-lights --output screenshots/test_multi_light_cpp.png --width %W% --height %H%
    echo.
) else (
    echo [Step 3] Skipped — C++ engine not built
    echo.
)

REM Step 4: Rust engine
if exist playground_rust\target\release\lux-playground.exe (
    echo [Step 4] Rendering with Rust engine (demo lights)...
    playground_rust\target\release\lux-playground.exe --scene %SCENE% --pipeline %PIPELINE% --ibl %IBL% --demo-lights --output screenshots/test_multi_light_rust.png --width %W% --height %H%
    echo.
) else (
    echo [Step 4] Skipped — Rust engine not built
    echo.
)

echo ============================================================
echo  Done. Screenshots saved to screenshots/test_multi_light_*.png
echo ============================================================
