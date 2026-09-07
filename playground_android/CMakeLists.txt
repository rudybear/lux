cmake_minimum_required(VERSION 3.20)
project(lux_playground_android LANGUAGES CXX C)

if(NOT ANDROID)
    message(FATAL_ERROR "Configure with -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-29")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Reused-by-reference lux sources (playground_cpp + its FetchContent deps,
# built once already by the desktop CMake build in
# playground_cpp/build/_deps). We do NOT modify any of these files -- only
# vulkan_context.{h,cpp} is excluded (it #includes GLFW, unavailable on
# Android); android_vulkan_context.{h,cpp} in this directory replaces it.
# ---------------------------------------------------------------------------
get_filename_component(LUX_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(PGCPP "${LUX_ROOT}/playground_cpp")
set(DEPS_DIR "${PGCPP}/build/_deps")

if(NOT EXISTS "${DEPS_DIR}/glm-src" OR NOT EXISTS "${DEPS_DIR}/vk-bootstrap-src")
    message(FATAL_ERROR "Expected ${PGCPP}/build/_deps/{glm-src,vk-bootstrap-src} "
                         "to exist (run the desktop playground_cpp CMake configure once "
                         "to fetch them, or point DEPS_DIR at a manual checkout).")
endif()

find_package(Vulkan REQUIRED)

# The NDK r29 sysroot bundles Vulkan headers at version 1.3.275, but the
# vendored vk-bootstrap-src (fetched by playground_cpp's desktop CMake,
# reused here by reference) generates its dispatch table against a newer
# Vulkan-Headers release (references VK_EXT_depth_clamp_control types added
# after 1.3.275) and fails to compile against the older NDK headers. Rather
# than patch vk-bootstrap-src (a third-party FetchContent checkout, not one
# of "our" files either), fetch current Vulkan-Headers and put its include
# dir ahead of the NDK sysroot's on the search path -- header-only, so this
# only affects what vulkan_core.h symbols are visible at compile time, not
# what's actually linked (still the NDK's libvulkan.so at runtime).
include(FetchContent)
FetchContent_Declare(
    vulkan_headers
    GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
    GIT_TAG        v1.3.296
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(vulkan_headers)
set(FRESH_VULKAN_INCLUDE "${vulkan_headers_SOURCE_DIR}/include")

add_library(vk-bootstrap STATIC
    "${DEPS_DIR}/vk-bootstrap-src/src/VkBootstrap.cpp"
)
target_include_directories(vk-bootstrap PUBLIC "${FRESH_VULKAN_INCLUDE}" "${DEPS_DIR}/vk-bootstrap-src/src")
target_link_libraries(vk-bootstrap PUBLIC Vulkan::Vulkan)

# ---------------------------------------------------------------------------
# Stage 4: TensorFlow Lite C API + GPU delegate (mobile-DLSS Android live
# demo). No Maven/Gradle available in this NDK-only CMake build, so the
# arm64-v8a .so's are vendored directly under third_party/tflite/ (extracted
# from the official org.tensorflow:tensorflow-lite / tensorflow-lite-gpu
# 2.16.1 AARs -- 2.17.0's "tensorflow-lite" Maven artifact is a bodyless POM
# now, superseded upstream by the LiteRT AARs, but 2.16.1 still ships a real
# one). Headers are vendored from the matching tensorflow v2.16.1 tag (the
# AAR ships no headers, only .so's + Java).
#
# The AAR's jni/arm64-v8a/libtensorflowlite_jni.so is Java/JNI-oriented by
# name, but `nm -D` on it confirms it exports the full versioned
# (VERS_1.0) plain-C ABI (TfLiteInterpreterCreate, TfLiteModelCreate, ...) --
# i.e. it doubles as the "libtensorflowlite_c.so" the task brief names, so
# no separate C-only artifact is needed. Same story for
# libtensorflowlite_gpu_jni.so and TfLiteGpuDelegateV2Create/Delete. Both
# ship their real SONAME baked in as their JNI filename (confirmed via
# `readelf -d`), so they're deliberately kept under their AAR-native
# filenames here and in the APK (see build.sh) rather than renamed --
# renaming the file without also using `patchelf`/relinking to fix the
# SONAME would leave the dynamic linker looking for a DT_NEEDED name that
# doesn't match the on-disk file inside the APK's lib/arm64-v8a/.
set(TFLITE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/tflite")
add_library(tflite_c SHARED IMPORTED)
set_target_properties(tflite_c PROPERTIES
    IMPORTED_LOCATION "${TFLITE_DIR}/lib/arm64-v8a/libtensorflowlite_jni.so")
add_library(tflite_gpu_delegate SHARED IMPORTED)
set_target_properties(tflite_gpu_delegate PROPERTIES
    IMPORTED_LOCATION "${TFLITE_DIR}/lib/arm64-v8a/libtensorflowlite_gpu_jni.so")

add_library(lux_android SHARED
    # --- Android glue / this directory ---
    "${ANDROID_NDK}/sources/android/native_app_glue/android_native_app_glue.c"
    src/android_main.cpp
    src/android_vulkan_context.cpp
    src/input_assembly.cpp
    src/net_runner.cpp
    src/reconstruct_pass.cpp

    # --- Reused-by-reference lux Vulkan splat pipeline (playground_cpp/src) ---
    "${PGCPP}/src/splat_renderer.cpp"
    "${PGCPP}/src/gltf_loader.cpp"
    "${PGCPP}/src/scene_manager.cpp"
    "${PGCPP}/src/scene.cpp"
    "${PGCPP}/src/camera.cpp"
    "${PGCPP}/src/dlss_io.cpp"
    "${PGCPP}/src/spv_loader.cpp"
    "${PGCPP}/src/reflected_pipeline.cpp"
)

target_include_directories(lux_android PRIVATE
    "${FRESH_VULKAN_INCLUDE}"
    "${ANDROID_NDK}/sources/android/native_app_glue"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${PGCPP}/src"
    "${PGCPP}/deps"
    "${DEPS_DIR}/glm-src"
    "${DEPS_DIR}/vulkanmemoryallocator-src/include"
    "${TFLITE_DIR}/include"
)

target_compile_definitions(lux_android PRIVATE VK_USE_PLATFORM_ANDROID_KHR)
target_compile_options(lux_android PRIVATE -Wno-nullability-completeness)
target_compile_options(vk-bootstrap PRIVATE -Wno-nullability-completeness)

target_link_libraries(lux_android PRIVATE
    vk-bootstrap
    Vulkan::Vulkan
    android
    log
    tflite_c
    tflite_gpu_delegate
)

set_target_properties(lux_android PROPERTIES
    LIBRARY_OUTPUT_NAME "lux_android"
)
