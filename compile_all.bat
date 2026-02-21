@echo off
REM Compile all shader pipelines: glTF PBR (forward, RT, layered) and mesh shaders
REM Outputs: shadercache/*.spv files

echo ============================================================
echo  Lux: All Shader Pipelines
echo ============================================================
echo.

REM --- Hand-written Forward ---
echo [1/6] Compiling hand-written Forward (gltf_pbr.lux)
python -m luxc examples/gltf_pbr.lux -o shadercache/
echo.

REM --- Hand-written RT ---
echo [2/6] Compiling hand-written RT (gltf_pbr_rt.lux)
python -m luxc examples/gltf_pbr_rt.lux -o shadercache/
echo.

REM --- Layered Forward: base (normal_map + emission) ---
echo [3/6] Compiling layered Forward (normal_map + emission)
python -m luxc examples/gltf_pbr_layered.lux --pipeline GltfForward --features has_normal_map,has_emission -o shadercache/
echo.

REM --- Layered RT: base (normal_map + emission) ---
echo [4/6] Compiling layered RT (normal_map + emission)
python -m luxc examples/gltf_pbr_layered.lux --pipeline GltfRT --features has_normal_map,has_emission -o shadercache/ --no-validate
echo.

REM --- Mesh Shader Pipeline (from layered) ---
echo [5/6] Compiling Mesh Shader (GltfMesh from gltf_pbr_layered.lux)
python -m luxc examples/gltf_pbr_layered.lux --pipeline GltfMesh --features has_emission -o shadercache/ --define max_vertices=64 --define max_primitives=124 --define workgroup_size=32
echo.

REM --- Cartoon Shader (Vertex) ---
echo [6/6] Compiling Cartoon shader
python -m luxc examples/cartoon_shader.lux -o shadercache/
echo.

echo ============================================================
echo  All shader pipelines compiled. Check shadercache/ for .spv files.
echo ============================================================
