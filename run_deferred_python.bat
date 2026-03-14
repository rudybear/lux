@echo off
REM Launch Python deferred renderer in interactive mode
REM Usage: run_deferred_python.bat [scene.glb]
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

REM 2. Launch interactive mode
set SCENE=%~1
if "%SCENE%"=="" set SCENE=assets\DamagedHelmet.glb

echo Running Python deferred renderer (scene=%SCENE%)...
python playground\deferred_engine.py examples\deferred_basic --scene "%SCENE%" --interactive
if errorlevel 1 (
    echo.
    echo ERROR: Renderer exited with error code %ERRORLEVEL%
    pause
)
