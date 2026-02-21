@echo off
REM Interactive Cartoon Toon raster viewer — Rust Vulkan engine
REM Controls: mouse drag to orbit, scroll = zoom, ESC to exit

if not exist playground_rust\target\release\lux-playground.exe (
    echo Rust engine not built. Build with:
    echo   cd playground_rust ^&^& cargo build --release
    exit /b 1
)

REM Compile cartoon_toon forward if needed
if not exist shadercache\cartoon_toon.frag.spv (
    echo Compiling cartoon_toon forward shaders...
    python -m luxc examples/cartoon_toon.lux -o shadercache/ --pipeline ToonForward
)
if not exist shadercache\cartoon_toon.frag.spv (
    echo Failed to compile shaders.
    exit /b 1
)

set PIPELINE=shadercache/cartoon_toon
set SCENE=assets/DamagedHelmet.glb
set IBL=pisa

echo === Interactive Rust Cartoon Raster Viewer ===
echo   Scene:    %SCENE%
echo   Pipeline: %PIPELINE% (raster)
echo   IBL:      %IBL%
echo   Controls: mouse drag = orbit, scroll = zoom, ESC = exit
echo.

playground_rust\target\release\lux-playground.exe --scene %SCENE% --pipeline %PIPELINE% --ibl %IBL% --interactive
