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

REM --- Layered Forward: all permutations ---
echo [3/6] Compiling layered Forward (all permutations)
python -m luxc examples/gltf_pbr_layered.lux --pipeline GltfForward --all-permutations -o shadercache/
echo.

REM --- Layered RT: all permutations ---
echo [4/6] Compiling layered RT (all permutations)
python -m luxc examples/gltf_pbr_layered.lux --pipeline GltfRT --all-permutations -o shadercache/
echo.

REM --- Mesh Shader Pipeline (from layered): all permutations ---
echo [5/6] Compiling Mesh Shader (GltfMesh, all permutations)
python -m luxc examples/gltf_pbr_layered.lux --pipeline GltfMesh --all-permutations -o shadercache/ --define max_vertices=64 --define max_primitives=124 --define workgroup_size=32
echo.

REM --- Cartoon Shader (Vertex) ---
echo [6/6] Compiling Cartoon shader
python -m luxc examples/cartoon_shader.lux -o shadercache/
echo.

echo ============================================================
echo  All shader pipelines compiled. Check shadercache/ for .spv files.
echo ============================================================
