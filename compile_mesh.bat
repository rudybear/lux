@echo off
REM Compile mesh shader pipeline from glTF PBR Layered
REM Uses mesh shaders instead of traditional vertex shaders
REM Requires GPU with mesh shader support (NVIDIA, AMD recent gen)

echo === Compiling glTF PBR Mesh Shader Pipeline (GltfMesh) ===
python -m luxc examples/gltf_pbr_layered.lux --pipeline GltfMesh --features has_emission -o shadercache/ --define max_vertices=64 --define max_primitives=124 --define workgroup_size=32
echo.
echo Done.
