@echo off
REM Launch deferred renderer in interactive mode (Rust)
REM Usage: run_deferred_rust.bat [scene.glb]
REM Default scene: assets\DamagedHelmet.glb

setlocal

REM 1. Compile the deferred shader
echo Compiling deferred_basic.lux...
python -m luxc examples\deferred_basic.lux
if errorlevel 1 (
    echo ERROR: Shader compilation failed
    pause
    exit /b 1
)

REM 2. Build the Rust viewer (if needed)
if not exist playground_rust\target\release\lux-playground.exe (
    echo Building Rust viewer...
    cd playground_rust
    cargo build --release
    cd ..
)

REM 3. Launch interactive mode
set SCENE=%~1
if "%SCENE%"=="" set SCENE=assets\DamagedHelmet.glb

echo Running deferred renderer (scene=%SCENE%)...
playground_rust\target\release\lux-playground.exe --scene %SCENE% --pipeline examples/deferred_basic --ibl pisa --mode deferred --interactive --validation
if errorlevel 1 (
    echo.
    echo ERROR: Renderer exited with error code %ERRORLEVEL%
    pause
)
