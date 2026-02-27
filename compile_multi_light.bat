@echo off
REM Compile glTF PBR Layered shaders with shadow support for multi-light demo
REM Generates all permutations including has_shadows variants

echo ============================================================
echo  Lux: Multi-Light Shadow Demo — Shader Compilation
echo ============================================================
echo.

REM --- Layered Forward: all permutations (includes has_shadows) ---
echo [1/2] Compiling layered Forward (all permutations, including shadows)
python -m luxc examples/gltf_pbr_layered.lux --pipeline GltfForward --all-permutations -o shadercache/
echo.

REM --- Also compile a specific shadow+emission+normal_map variant for direct use ---
echo [2/2] Compiling specific shadow variant (emission+normal_map+shadows)
python -m luxc examples/gltf_pbr_layered.lux --pipeline GltfForward --features has_normal_map,has_emission,has_shadows -o shadercache/
echo.

echo ============================================================
echo  Compilation complete. Shadow-enabled shaders in shadercache/
echo ============================================================
