@echo off
REM Compile all glTF PBR variants: hand-written, layered (all permutations), and bindless.
REM Uses --all-permutations to generate every feature combination automatically.
REM Uses --bindless to generate uber-shaders with bindless descriptor arrays.
REM Engines auto-select the correct permutation per material from the manifest.

echo ============================================================
echo  glTF PBR — All Variants
echo ============================================================
echo.

REM --- Hand-written Forward ---
echo [1/7] Compiling hand-written Forward (gltf_pbr.lux)
python -m luxc examples/gltf_pbr.lux -o shadercache/
echo.

REM --- Hand-written RT ---
echo [2/7] Compiling hand-written RT (gltf_pbr_rt.lux)
python -m luxc examples/gltf_pbr_rt.lux -o shadercache/
echo.

REM --- Layered Forward: all permutations ---
echo [3/7] Compiling layered Forward (all permutations)
python -m luxc examples/gltf_pbr_layered.lux --pipeline GltfForward --all-permutations -o shadercache/
echo.

REM --- Layered RT: all permutations ---
echo [4/7] Compiling layered RT (all permutations)
python -m luxc examples/gltf_pbr_layered.lux --pipeline GltfRT --all-permutations -o shadercache/ --no-validate
echo.

REM --- Bindless Forward: single uber-shader ---
echo [5/7] Compiling bindless Forward (uber-shader)
python -m luxc examples/gltf_pbr_layered.lux --pipeline GltfForward --bindless -o shadercache/bindless/
echo.

REM --- Bindless RT: single uber-shader ---
echo [6/7] Compiling bindless RT (uber-shader)
python -m luxc examples/gltf_pbr_layered.lux --pipeline GltfRT --bindless -o shadercache/bindless/
echo.

REM --- Bindless Mesh: single uber-shader ---
echo [7/7] Compiling bindless Mesh (uber-shader)
python -m luxc examples/gltf_pbr_layered.lux --pipeline GltfMesh --bindless -o shadercache/bindless/
echo.

echo ============================================================
echo  Rendering
echo ============================================================
echo.

REM Python renders (raster only)
REM With multi-pipeline support, engines auto-select permutations from manifest
echo --- Python engine: base models (raster) ---
python -m playground.engine --scene assets/DamagedHelmet.glb --pipeline shadercache/gltf_pbr --output screenshots/test_gltf_forward_python.png --width 512 --height 512
python -m playground.engine --scene assets/DamagedHelmet.glb --pipeline shadercache/gltf_pbr_layered --output screenshots/test_gltf_layered_forward_python.png --width 512 --height 512
echo.

REM Python renders: Khronos extension test models
if exist assets\ClearCoatTest.glb (
    echo --- Python engine: Khronos extension test models ---
    python -m playground.engine --scene assets/ClearCoatTest.glb --pipeline shadercache/gltf_pbr_layered --output screenshots/test_clearcoat_python.png --width 512 --height 512
    python -m playground.engine --scene assets/SheenChair.glb --pipeline shadercache/gltf_pbr_layered --output screenshots/test_sheen_python.png --width 512 --height 512
    python -m playground.engine --scene assets/TransmissionTest.glb --pipeline shadercache/gltf_pbr_layered --output screenshots/test_transmission_python.png --width 512 --height 512
    echo.
)

REM C++ renders (raster + RT)
if exist playground_cpp\build\Release\lux-playground.exe (
    echo --- C++ engine: base models, raster + RT ---
    playground_cpp\build\Release\lux-playground.exe --scene assets/DamagedHelmet.glb --pipeline shadercache/gltf_pbr --ibl pisa --output screenshots/test_gltf_forward_cpp.png --width 512 --height 512
    playground_cpp\build\Release\lux-playground.exe --scene assets/DamagedHelmet.glb --pipeline shadercache/gltf_pbr_rt --ibl pisa --output screenshots/test_gltf_rt_cpp.png --width 512 --height 512
    playground_cpp\build\Release\lux-playground.exe --scene assets/DamagedHelmet.glb --pipeline shadercache/gltf_pbr_layered --ibl pisa --output screenshots/test_gltf_layered_forward_cpp.png --width 512 --height 512
    if exist assets\ClearCoatTest.glb (
        echo --- C++ engine: Khronos extension test models ---
        playground_cpp\build\Release\lux-playground.exe --scene assets/SheenChair.glb --pipeline shadercache/gltf_pbr_layered --ibl pisa --output screenshots/test_sheen_cpp.png --width 512 --height 512
        playground_cpp\build\Release\lux-playground.exe --scene assets/SheenChair.glb --pipeline shadercache/gltf_pbr_layered --mode rt --ibl pisa --output screenshots/test_sheen_rt_cpp.png --width 512 --height 512
        playground_cpp\build\Release\lux-playground.exe --scene assets/SheenChair.glb --pipeline shadercache/gltf_pbr_layered --mode mesh --ibl pisa --output screenshots/test_sheen_mesh_cpp.png --width 512 --height 512
        playground_cpp\build\Release\lux-playground.exe --scene assets/TransmissionTest.glb --pipeline shadercache/gltf_pbr_layered --ibl pisa --output screenshots/test_transmission_cpp.png --width 512 --height 512
        playground_cpp\build\Release\lux-playground.exe --scene assets/ClearCoatTest.glb --pipeline shadercache/gltf_pbr_layered --ibl pisa --output screenshots/test_clearcoat_cpp.png --width 512 --height 512
    )
    echo.
)

REM Rust renders (raster + RT)
if exist playground_rust\target\release\lux-playground.exe (
    echo --- Rust engine: base models, raster + RT ---
    playground_rust\target\release\lux-playground.exe --scene assets/DamagedHelmet.glb --pipeline shadercache/gltf_pbr --ibl pisa --output screenshots/test_gltf_forward_rust.png --width 512 --height 512
    playground_rust\target\release\lux-playground.exe --scene assets/DamagedHelmet.glb --pipeline shadercache/gltf_pbr_rt --ibl pisa --output screenshots/test_gltf_rt_rust.png --width 512 --height 512
    playground_rust\target\release\lux-playground.exe --scene assets/DamagedHelmet.glb --pipeline shadercache/gltf_pbr_layered --ibl pisa --output screenshots/test_gltf_layered_forward_rust.png --width 512 --height 512
    if exist assets\ClearCoatTest.glb (
        echo --- Rust engine: Khronos extension test models ---
        playground_rust\target\release\lux-playground.exe --scene assets/SheenChair.glb --pipeline shadercache/gltf_pbr_layered --ibl pisa --output screenshots/test_sheen_rust.png --width 512 --height 512
        playground_rust\target\release\lux-playground.exe --scene assets/SheenChair.glb --pipeline shadercache/gltf_pbr_layered --mode rt --ibl pisa --output screenshots/test_sheen_rt_rust.png --width 512 --height 512
        playground_rust\target\release\lux-playground.exe --scene assets/SheenChair.glb --pipeline shadercache/gltf_pbr_layered --mode mesh --ibl pisa --output screenshots/test_sheen_mesh_rust.png --width 512 --height 512
        playground_rust\target\release\lux-playground.exe --scene assets/ClearCoatTest.glb --pipeline shadercache/gltf_pbr_layered --ibl pisa --output screenshots/test_clearcoat_rust.png --width 512 --height 512
        playground_rust\target\release\lux-playground.exe --scene assets/TransmissionTest.glb --pipeline shadercache/gltf_pbr_layered --ibl pisa --output screenshots/test_transmission_rust.png --width 512 --height 512
    )
    echo.
)

REM ============================================================
REM  Bindless rendering (C++ and Rust only — Python/wgpu lacks descriptor indexing)
REM ============================================================
echo.
echo ============================================================
echo  Bindless Rendering
echo ============================================================
echo.

REM C++ bindless renders
if exist playground_cpp\build\Release\lux-playground.exe (
    echo --- C++ engine: bindless raster ---
    playground_cpp\build\Release\lux-playground.exe --scene assets/DamagedHelmet.glb --pipeline shadercache/bindless/gltf_pbr_layered --ibl pisa --output screenshots/test_bindless_raster_cpp.png --width 512 --height 512
    if exist assets\SheenChair.glb (
        playground_cpp\build\Release\lux-playground.exe --scene assets/SheenChair.glb --pipeline shadercache/bindless/gltf_pbr_layered --ibl pisa --output screenshots/test_bindless_sheen_raster_cpp.png --width 512 --height 512
        echo --- C++ engine: bindless RT ---
        playground_cpp\build\Release\lux-playground.exe --scene assets/SheenChair.glb --pipeline shadercache/bindless/gltf_pbr_layered --mode rt --ibl pisa --output screenshots/test_bindless_sheen_rt_cpp.png --width 512 --height 512
        echo --- C++ engine: bindless mesh ---
        playground_cpp\build\Release\lux-playground.exe --scene assets/SheenChair.glb --pipeline shadercache/bindless/gltf_pbr_layered --mode mesh --ibl pisa --output screenshots/test_bindless_sheen_mesh_cpp.png --width 512 --height 512
    )
    echo.
)

REM Rust bindless renders
if exist playground_rust\target\release\lux-playground.exe (
    echo --- Rust engine: bindless raster ---
    playground_rust\target\release\lux-playground.exe --scene assets/DamagedHelmet.glb --pipeline shadercache/bindless/gltf_pbr_layered --ibl pisa --output screenshots/test_bindless_raster_rust.png --width 512 --height 512
    if exist assets\SheenChair.glb (
        playground_rust\target\release\lux-playground.exe --scene assets/SheenChair.glb --pipeline shadercache/bindless/gltf_pbr_layered --ibl pisa --output screenshots/test_bindless_sheen_raster_rust.png --width 512 --height 512
        echo --- Rust engine: bindless RT ---
        playground_rust\target\release\lux-playground.exe --scene assets/SheenChair.glb --pipeline shadercache/bindless/gltf_pbr_layered --mode rt --ibl pisa --output screenshots/test_bindless_sheen_rt_rust.png --width 512 --height 512
        echo --- Rust engine: bindless mesh ---
        playground_rust\target\release\lux-playground.exe --scene assets/SheenChair.glb --pipeline shadercache/bindless/gltf_pbr_layered --mode mesh --ibl pisa --output screenshots/test_bindless_sheen_mesh_rust.png --width 512 --height 512
    )
    echo.
)

echo ============================================================
echo  All done. Check screenshots/ for rendered PNGs.
echo ============================================================
