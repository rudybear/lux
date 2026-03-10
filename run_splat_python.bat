@echo off
REM Interactive Gaussian splat viewer - Python CPU rasterizer
REM Controls: drag to orbit, scroll to zoom, ESC/Q to exit

echo === Python Gaussian Splat Viewer ===

REM --- Check python is available ---
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: python not found on PATH
    pause
    exit /b 1
)

REM --- Scene: use provided .glb or download luigi ---
set SCENE=%1
if not "%SCENE%"=="" goto :have_scene

if not exist tests\assets\luigi.glb (
    echo Downloading luigi.ply from HuggingFace...
    if not exist tests\assets mkdir tests\assets
    curl -L -o tests\assets\luigi.ply "https://huggingface.co/datasets/dylanebert/3dgs/resolve/main/luigi/luigi.ply?download=true"
    if errorlevel 1 (
        echo Download failed.
        pause
        exit /b 1
    )
    echo Converting PLY to glTF...
    python -m tools.ply_to_gltf tests\assets\luigi.ply tests\assets\luigi.glb
    if errorlevel 1 (
        echo Conversion failed.
        pause
        exit /b 1
    )
)
set SCENE=tests\assets\luigi.glb

:have_scene
echo Scene: %SCENE%
echo Controls: drag to orbit, scroll to zoom, ESC/Q to exit
echo.

python -m playground.render_harness --splat-scene "%SCENE%" --interactive
if errorlevel 1 (
    echo.
    echo Viewer exited with error.
    pause
    exit /b 1
)
