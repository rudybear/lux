@echo off
REM Gaussian splat headless render — Python wgpu harness
REM Outputs a PNG screenshot of the splat scene.

REM --- Compile splat shaders if needed ---
if not exist examples\gaussian_splat.comp.spv (
    echo Compiling gaussian_splat.lux...
    python -m luxc examples/gaussian_splat.lux
    if errorlevel 1 (
        echo Shader compilation failed.
        exit /b 1
    )
)

REM --- Scene: use provided .glb or download luigi test asset ---
set SCENE=%1
if "%SCENE%"=="" (
    if not exist tests\assets\luigi.glb (
        echo Downloading luigi.ply from HuggingFace...
        if not exist tests\assets mkdir tests\assets
        curl -L -o tests\assets\luigi.ply "https://huggingface.co/datasets/dylanebert/3dgs/resolve/main/luigi/luigi.ply?download=true"
        echo Converting PLY to glTF...
        python -m tools.ply_to_gltf tests/assets/luigi.ply tests/assets/luigi.glb
    )
    set SCENE=tests/assets/luigi.glb
)

echo === Python Gaussian Splat Renderer ===
echo   Scene: %SCENE%
echo.

python -m playground.render_harness --splat-comp examples/gaussian_splat.comp.spv --splat-vert examples/gaussian_splat.vert.spv --splat-frag examples/gaussian_splat.frag.spv --splat-scene %SCENE% -o splat_render.png
