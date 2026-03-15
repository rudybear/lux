@echo off
REM Render the truck PLY interactively with Gaussian splatting

set PLY=C:\Users\rudyb\Downloads\point_cloud.ply
set GLB=C:\Users\rudyb\Downloads\point_cloud_noconv.glb
set PIPELINE=build\gaussian_splat_sh3

REM Convert PLY if GLB doesn't exist
if not exist "%GLB%" (
    echo Converting PLY to GLB...
    python tools\ply_to_gltf.py "%PLY%" "%GLB%" --no-convert
)

REM Always recompile shaders to pick up latest fixes
echo Compiling splat shaders...
python -m luxc examples\gaussian_splat_sh3.lux -o build\

echo Launching interactive viewer...
playground_rust\target\release\lux-playground.exe --scene "%GLB%" --pipeline %PIPELINE% --interactive --width 1280 --height 960
