@echo off
REM Interactive Gaussian splat viewer - Rust Vulkan engine
REM Controls: mouse drag to orbit, scroll to zoom, ESC to exit

REM --- Compile splat shaders if needed ---
if not exist examples\gaussian_splat.comp.spv (
    echo Compiling gaussian_splat.lux...
    python -m luxc examples\gaussian_splat.lux
    if errorlevel 1 (
        echo Shader compilation failed.
        exit /b 1
    )
)

REM --- Scene: use provided .glb or download luigi ---
set SCENE=%1
if not "%SCENE%"=="" goto :have_scene

if not exist tests\assets\luigi.glb (
    echo Downloading luigi.ply from HuggingFace...
    if not exist tests\assets mkdir tests\assets
    curl -L -o tests\assets\luigi.ply "https://huggingface.co/datasets/dylanebert/3dgs/resolve/main/luigi/luigi.ply?download=true"
    echo Converting PLY to glTF...
    python -m tools.ply_to_gltf tests\assets\luigi.ply tests\assets\luigi.glb
)
set SCENE=tests\assets\luigi.glb

:have_scene
set PIPELINE=examples\gaussian_splat

echo === Interactive Rust Gaussian Splat Viewer ===
echo   Scene:    %SCENE%
echo   Pipeline: %PIPELINE%
echo   Controls: mouse drag = orbit, scroll = zoom, ESC = exit
echo.

cd playground_rust && cargo run --release -- --scene "..\%SCENE%" --pipeline ..\%PIPELINE% --interactive
