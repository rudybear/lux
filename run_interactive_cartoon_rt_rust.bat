@echo off
REM Interactive Cartoon Toon RT viewer — Rust Vulkan engine
REM Controls: mouse drag to orbit, scroll to zoom, ESC to exit

if not exist playground_rust\target\release\lux-playground.exe (
    echo Rust engine not built. Build with:
    echo   cd playground_rust ^&^& cargo build --release
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

echo === Interactive Rust Cartoon RT Viewer ===
echo   Scene:    %SCENE%
echo   Pipeline: %PIPELINE% (ray tracing)
echo   IBL:      %IBL%
echo   Controls: mouse drag = orbit, scroll = zoom, ESC = exit
echo.

playground_rust\target\release\lux-playground.exe --scene %SCENE% --pipeline %PIPELINE% --ibl %IBL% --interactive
