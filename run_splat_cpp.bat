@echo off
REM Interactive Gaussian splat viewer - C++ Vulkan engine
REM Controls: mouse drag to orbit, scroll to zoom, ESC to exit

if not exist playground_cpp\build\Release\lux-playground.exe (
    echo C++ engine not built. Build with:
    echo   cd playground_cpp ^&^& cmake -B build ^&^& cmake --build build --config Release
    exit /b 1
)

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
set PIPELINE=examples/gaussian_splat

echo === Interactive C++ Gaussian Splat Viewer ===
echo   Scene:    %SCENE%
echo   Pipeline: %PIPELINE%
echo   Controls: mouse drag = orbit, scroll = zoom, ESC = exit
echo.

playground_cpp\build\Release\lux-playground.exe --scene %SCENE% --pipeline %PIPELINE% --interactive
