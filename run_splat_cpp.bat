@echo off
REM Interactive Gaussian splat viewer — C++ Vulkan engine
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

REM --- Scene: use provided .glb or download luigi test asset ---
set SCENE=%1
if "%SCENE%"=="" (
    if not exist tests\assets\luigi.glb (
        echo Downloading luigi.ply from HuggingFace...
        if not exist tests\assets mkdir tests\assets
        curl -L -o tests\assets\luigi.ply "https://huggingface.co/datasets/dylanebert/3dgs/resolve/main/luigi/luigi.ply?download=true"
        echo Converting PLY to glTF...
        python -m tools.ply_to_gltf tests/assets/luigi.ply tests/assets/luigi.glb
    )
    set SCENE=tests/assets/luigi.glb
)

set PIPELINE=examples/gaussian_splat

echo === Interactive C++ Gaussian Splat Viewer ===
echo   Scene:    %SCENE%
echo   Pipeline: %PIPELINE%
echo   Controls: mouse drag = orbit, scroll = zoom, ESC = exit
echo.

playground_cpp\build\Release\lux-playground.exe --scene %SCENE% --pipeline %PIPELINE% --interactive
