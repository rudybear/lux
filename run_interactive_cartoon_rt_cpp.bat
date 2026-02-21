@echo off
REM Interactive Cartoon Toon RT viewer — C++ Vulkan engine
REM Controls: mouse drag to orbit, scroll to zoom, ESC to exit

if not exist playground_cpp\build\Release\lux-playground.exe (
    echo C++ engine not built. Build with:
    echo   cd playground_cpp ^&^& cmake -B build ^&^& cmake --build build --config Release
    exit /b 1
)

REM Compile cartoon_toon RT if needed
if not exist shadercache\cartoon_toon.rgen.spv (
    echo Compiling cartoon_toon RT shaders...
    python -m luxc examples/cartoon_toon.lux -o shadercache/ --pipeline ToonRT
)
if not exist shadercache\cartoon_toon.rgen.spv (
    echo Failed to compile RT shaders.
    exit /b 1
)

set PIPELINE=shadercache/cartoon_toon
set SCENE=assets/DamagedHelmet.glb
set IBL=pisa

echo === Interactive C++ Cartoon RT Viewer ===
echo   Scene:    %SCENE%
echo   Pipeline: %PIPELINE% (ray tracing)
echo   IBL:      %IBL%
echo   Controls: mouse drag = orbit, scroll = zoom, ESC = exit
echo.

playground_cpp\build\Release\lux-playground.exe --scene %SCENE% --pipeline %PIPELINE% --ibl %IBL% --interactive
