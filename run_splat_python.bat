@echo off
REM Gaussian splat headless render — Python CPU rasterizer
REM Outputs a PNG screenshot of the splat scene.

REM --- Compile splat shaders if needed ---
if not exist examples\gaussian_splat.comp.spv (
    echo Compiling gaussian_splat.lux...
    python -m luxc examples/gaussian_splat.lux
    if errorlevel 1 (
        echo Shader compilation failed.
        exit /b 1
    )
)

REM --- Scene: use provided .glb or default to test_splats ---
set SCENE=%1
if not "%SCENE%"=="" goto :have_scene

if not exist tests\assets\test_splats.glb (
    echo Generating test splats...
    if not exist tests\assets mkdir tests\assets
    python -m tools.generate_test_splats tests/assets/test_splats.glb
)
set SCENE=tests/assets/test_splats.glb

:have_scene
echo === Python Gaussian Splat Renderer ===
echo   Scene: %SCENE%
echo.

python -m playground.render_harness --splat-scene %SCENE% -o splat_render.png
