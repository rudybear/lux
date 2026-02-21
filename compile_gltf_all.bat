@echo off
REM Compile all glTF PBR variants: hand-written and layered, forward and RT
REM Outputs: shadercache/*.spv files + rendered PNGs

echo ============================================================
echo  glTF PBR — All Variants
echo ============================================================
echo.

REM --- Hand-written Forward ---
echo [1/4] Compiling hand-written Forward (gltf_pbr.lux)
python -m luxc examples/gltf_pbr.lux -o shadercache/
echo.

REM --- Hand-written RT ---
echo [2/4] Compiling hand-written RT (gltf_pbr_rt.lux)
python -m luxc examples/gltf_pbr_rt.lux -o shadercache/
echo.

REM --- Layered Forward (with features) ---
echo [3/4] Compiling layered Forward (gltf_pbr_layered.lux --pipeline GltfForward --features)
python -m luxc examples/gltf_pbr_layered.lux --pipeline GltfForward --features has_normal_map,has_emission -o shadercache/
echo.

REM --- Layered RT (with features) ---
echo [4/4] Compiling layered RT (gltf_pbr_layered.lux --pipeline GltfRT --features)
python -m luxc examples/gltf_pbr_layered.lux --pipeline GltfRT --features has_normal_map,has_emission -o shadercache/ --no-validate
echo.

echo ============================================================
echo  Rendering
echo ============================================================
echo.

REM Python renders (raster only)
echo --- Python engine (raster) ---
python -m playground.engine --scene assets/DamagedHelmet.glb --pipeline shadercache/gltf_pbr --output screenshots/test_gltf_forward_python.png --width 512 --height 512
python -m playground.engine --scene assets/DamagedHelmet.glb --pipeline shadercache/gltf_pbr_layered+emission+normal_map --output screenshots/test_gltf_layered_forward_python.png --width 512 --height 512
echo.

REM C++ renders (raster + RT)
if exist playground_cpp\build\Release\lux-playground.exe (
    echo --- C++ engine (raster + RT) ---
    playground_cpp\build\Release\lux-playground.exe --scene assets/DamagedHelmet.glb --pipeline shadercache/gltf_pbr --output screenshots/test_gltf_forward_cpp.png --width 512 --height 512
    playground_cpp\build\Release\lux-playground.exe --scene assets/DamagedHelmet.glb --pipeline shadercache/gltf_pbr_rt --output screenshots/test_gltf_rt_cpp.png --width 512 --height 512 --mode rt
    playground_cpp\build\Release\lux-playground.exe --scene assets/DamagedHelmet.glb --pipeline shadercache/gltf_pbr_layered+emission+normal_map --output screenshots/test_gltf_layered_forward_cpp.png --width 512 --height 512
    playground_cpp\build\Release\lux-playground.exe --scene assets/DamagedHelmet.glb --pipeline shadercache/gltf_pbr_layered+emission+normal_map --output screenshots/test_gltf_layered_rt_cpp.png --width 512 --height 512 --mode rt
    echo.
)

REM Rust renders (raster + RT)
if exist playground_rust\target\release\lux-playground.exe (
    echo --- Rust engine (raster + RT) ---
    playground_rust\target\release\lux-playground.exe --scene assets/DamagedHelmet.glb --pipeline shadercache/gltf_pbr --output screenshots/test_gltf_forward_rust.png --width 512 --height 512
    playground_rust\target\release\lux-playground.exe --scene assets/DamagedHelmet.glb --pipeline shadercache/gltf_pbr_rt --output screenshots/test_gltf_rt_rust.png --width 512 --height 512 --mode rt
    playground_rust\target\release\lux-playground.exe --scene assets/DamagedHelmet.glb --pipeline shadercache/gltf_pbr_layered+emission+normal_map --output screenshots/test_gltf_layered_forward_rust.png --width 512 --height 512
    playground_rust\target\release\lux-playground.exe --scene assets/DamagedHelmet.glb --pipeline shadercache/gltf_pbr_layered+emission+normal_map --output screenshots/test_gltf_layered_rt_rust.png --width 512 --height 512 --mode rt
    echo.
)

echo ============================================================
echo  All done. Check screenshots/ for rendered PNGs.
echo ============================================================
