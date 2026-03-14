@echo off
REM Launch deferred renderer in interactive mode
REM Usage: run_deferred_cpp.bat [scene.glb]
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

REM 2. Build the C++ viewer (if needed)
if not exist playground_cpp\build\Release\lux-playground.exe (
    echo Building C++ viewer...
    cd playground_cpp\build
    cmake --build . --config Release
    cd ..\..
)

REM 3. Launch interactive mode
set SCENE=%~1
if "%SCENE%"=="" set SCENE=assets\DamagedHelmet.glb

echo Running deferred renderer (scene=%SCENE%)...
playground_cpp\build\Release\lux-playground.exe examples/deferred_basic --interactive --scene "%SCENE%" --validation
if errorlevel 1 (
    echo.
    echo ERROR: Renderer exited with error code %ERRORLEVEL%
    pause
)
