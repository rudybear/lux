@echo off
cd /d "%~dp0"
playground_cpp\build\Release\lux-playground.exe --scene lighttest --pipeline shadercache/gltf_pbr_layered --interactive
