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

REM --- Scene: use provided .glb or default to test_splats ---
set SCENE=%1
if not "%SCENE%"=="" goto :have_scene

if not exist tests\assets\test_splats.glb (
    echo Generating test splats...
    if not exist tests\assets mkdir tests\assets
    python -m tools.generate_test_splats tests\assets\test_splats.glb
    if errorlevel 1 (
        echo Failed to generate test splats.
        pause
        exit /b 1
    )
)
set SCENE=tests\assets\test_splats.glb

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
