@echo off
REM Compile all glTF PBR variants: hand-written and layered, forward and RT
REM Includes coat/sheen/transmission permutations for KHR_materials_* extensions
REM Outputs: shadercache/*.spv files + rendered PNGs

echo ============================================================
echo  glTF PBR — All Variants
echo ============================================================
echo.

REM --- Hand-written Forward ---
echo [1/10] Compiling hand-written Forward (gltf_pbr.lux)
python -m luxc examples/gltf_pbr.lux -o shadercache/
echo.

REM --- Hand-written RT ---
echo [2/10] Compiling hand-written RT (gltf_pbr_rt.lux)
python -m luxc examples/gltf_pbr_rt.lux -o shadercache/
echo.

REM --- Layered Forward: base (normal_map + emission) ---
echo [3/10] Compiling layered Forward (normal_map + emission)
python -m luxc examples/gltf_pbr_layered.lux --pipeline GltfForward --features has_normal_map,has_emission -o shadercache/
echo.

REM --- Layered RT: base (normal_map + emission) ---
echo [4/10] Compiling layered RT (normal_map + emission)
python -m luxc examples/gltf_pbr_layered.lux --pipeline GltfRT --features has_normal_map,has_emission -o shadercache/ --no-validate
echo.

REM --- Layered Forward: clearcoat ---
echo [5/10] Compiling layered Forward (normal_map + emission + clearcoat)
python -m luxc examples/gltf_pbr_layered.lux --pipeline GltfForward --features has_normal_map,has_emission,has_clearcoat -o shadercache/
echo.

REM --- Layered Forward: sheen ---
echo [6/10] Compiling layered Forward (normal_map + emission + sheen)
python -m luxc examples/gltf_pbr_layered.lux --pipeline GltfForward --features has_normal_map,has_emission,has_sheen -o shadercache/
echo.

REM --- Layered Forward: transmission ---
echo [7/10] Compiling layered Forward (normal_map + emission + transmission)
python -m luxc examples/gltf_pbr_layered.lux --pipeline GltfForward --features has_normal_map,has_emission,has_transmission -o shadercache/
echo.

REM --- Layered Forward: all extensions ---
echo [8/10] Compiling layered Forward (all extensions)
python -m luxc examples/gltf_pbr_layered.lux --pipeline GltfForward --features has_normal_map,has_emission,has_clearcoat,has_sheen,has_transmission -o shadercache/
echo.

REM --- Layered RT: clearcoat ---
echo [9/10] Compiling layered RT (normal_map + emission + clearcoat)
python -m luxc examples/gltf_pbr_layered.lux --pipeline GltfRT --features has_normal_map,has_emission,has_clearcoat -o shadercache/ --no-validate
echo.

REM --- Layered RT: all extensions ---
echo [10/14] Compiling layered RT (all extensions)
python -m luxc examples/gltf_pbr_layered.lux --pipeline GltfRT --features has_normal_map,has_emission,has_clearcoat,has_sheen,has_transmission -o shadercache/ --no-validate
echo.

REM --- Model-specific variants (exact feature sets matching Khronos test assets) ---
echo [11/14] Compiling layered Forward for SheenChair (normal_map + sheen)
python -m luxc examples/gltf_pbr_layered.lux --pipeline GltfForward --features has_normal_map,has_sheen -o shadercache/
echo.

echo [12/14] Compiling layered Forward for TransmissionTest (emission + transmission)
python -m luxc examples/gltf_pbr_layered.lux --pipeline GltfForward --features has_emission,has_transmission -o shadercache/
echo.

REM --- Layered Forward: clearcoat + normal_map (no emission) ---
echo [13/14] Compiling layered Forward (clearcoat + normal_map)
python -m luxc examples/gltf_pbr_layered.lux --pipeline GltfForward --features has_normal_map,has_clearcoat -o shadercache/
echo.

REM --- Layered Forward: clearcoat + sheen + normal_map (no emission) ---
echo [14/14] Compiling layered Forward (clearcoat + normal_map + sheen)
python -m luxc examples/gltf_pbr_layered.lux --pipeline GltfForward --features has_normal_map,has_clearcoat,has_sheen -o shadercache/
echo.

echo ============================================================
echo  Rendering
echo ============================================================
echo.

REM Python renders (raster only)
echo --- Python engine: base models (raster) ---
python -m playground.engine --scene assets/DamagedHelmet.glb --pipeline shadercache/gltf_pbr --output screenshots/test_gltf_forward_python.png --width 512 --height 512
python -m playground.engine --scene assets/DamagedHelmet.glb --pipeline shadercache/gltf_pbr_layered+emission+normal_map --output screenshots/test_gltf_layered_forward_python.png --width 512 --height 512
echo.

REM Python renders: Khronos extension test models
if exist assets\ClearCoatTest.glb (
    echo --- Python engine: Khronos extension test models ---
    python -m playground.engine --scene assets/ClearCoatTest.glb --pipeline shadercache/gltf_pbr_layered+clearcoat+emission+normal_map --output screenshots/test_clearcoat_python.png --width 512 --height 512
    python -m playground.engine --scene assets/SheenChair.glb --pipeline shadercache/gltf_pbr_layered+normal_map+sheen --output screenshots/test_sheen_python.png --width 512 --height 512
    python -m playground.engine --scene assets/TransmissionTest.glb --pipeline shadercache/gltf_pbr_layered+emission+transmission --output screenshots/test_transmission_python.png --width 512 --height 512
    echo.
)

REM C++ renders (raster + RT)
if exist playground_cpp\build\Release\lux-playground.exe (
    echo --- C++ engine: base models (raster + RT) ---
    playground_cpp\build\Release\lux-playground.exe --scene assets/DamagedHelmet.glb --pipeline shadercache/gltf_pbr --ibl pisa --output screenshots/test_gltf_forward_cpp.png --width 512 --height 512
    playground_cpp\build\Release\lux-playground.exe --scene assets/DamagedHelmet.glb --pipeline shadercache/gltf_pbr_rt --ibl pisa --output screenshots/test_gltf_rt_cpp.png --width 512 --height 512
    playground_cpp\build\Release\lux-playground.exe --scene assets/DamagedHelmet.glb --pipeline shadercache/gltf_pbr_layered+emission+normal_map --ibl pisa --output screenshots/test_gltf_layered_forward_cpp.png --width 512 --height 512
    if exist assets\ClearCoatTest.glb (
        echo --- C++ engine: Khronos extension test models (raster) ---
        playground_cpp\build\Release\lux-playground.exe --scene assets/SheenChair.glb --pipeline shadercache/gltf_pbr_layered+normal_map+sheen --ibl pisa --output screenshots/test_sheen_cpp.png --width 512 --height 512
        playground_cpp\build\Release\lux-playground.exe --scene assets/TransmissionTest.glb --pipeline shadercache/gltf_pbr_layered+emission+transmission --ibl pisa --output screenshots/test_transmission_cpp.png --width 512 --height 512
        playground_cpp\build\Release\lux-playground.exe --scene assets/ClearCoatTest.glb --pipeline shadercache/gltf_pbr_layered+clearcoat+emission+normal_map --ibl pisa --output screenshots/test_clearcoat_cpp.png --width 512 --height 512
    )
    echo.
)

REM Rust renders (raster + RT)
if exist playground_rust\target\release\lux-playground.exe (
    echo --- Rust engine: base models (raster + RT) ---
    playground_rust\target\release\lux-playground.exe --scene assets/DamagedHelmet.glb --pipeline shadercache/gltf_pbr --ibl pisa --output screenshots/test_gltf_forward_rust.png --width 512 --height 512
    playground_rust\target\release\lux-playground.exe --scene assets/DamagedHelmet.glb --pipeline shadercache/gltf_pbr_rt --ibl pisa --output screenshots/test_gltf_rt_rust.png --width 512 --height 512
    playground_rust\target\release\lux-playground.exe --scene assets/DamagedHelmet.glb --pipeline shadercache/gltf_pbr_layered+emission+normal_map --ibl pisa --output screenshots/test_gltf_layered_forward_rust.png --width 512 --height 512
    if exist assets\ClearCoatTest.glb (
        echo --- Rust engine: Khronos extension test models (raster) ---
        playground_rust\target\release\lux-playground.exe --scene assets/ClearCoatTest.glb --pipeline shadercache/gltf_pbr --ibl pisa --output screenshots/test_clearcoat_rust.png --width 512 --height 512
        playground_rust\target\release\lux-playground.exe --scene assets/TransmissionTest.glb --pipeline shadercache/gltf_pbr_layered+emission+transmission --ibl pisa --output screenshots/test_transmission_rust.png --width 512 --height 512
    )
    echo.
)

echo ============================================================
echo  All done. Check screenshots/ for rendered PNGs.
echo ============================================================
