@echo off
REM Download KHR_gaussian_splatting conformance test assets.
REM Assets are extracted to tests\assets\khr_splat_conformance\
REM
REM If the automatic download fails, download manually from:
REM   https://github.com/user-attachments/files/25351745/gltf-splat-examples-2026-02-17.zip
REM and place the zip in tests\assets\khr_splat_conformance\

python -m tools.download_khr_splat_tests
