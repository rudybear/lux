cmake_minimum_required(VERSION 3.20)
if(APPLE)
    project(lux-playground LANGUAGES CXX OBJCXX)
else()
    project(lux-playground LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# On Apple, Vulkan is optional (Metal backend doesn't need it)
if(APPLE)
    find_package(Vulkan QUIET)
else()
    find_package(Vulkan REQUIRED)
endif()

# Use FetchContent for all dependencies
include(FetchContent)

# GLFW
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.4
    GIT_SHALLOW    TRUE
)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

# GLM
FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG        1.0.1
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(glm)

# vk-bootstrap
FetchContent_Declare(
    vk-bootstrap
    GIT_REPOSITORY https://github.com/charles-lunarg/vk-bootstrap.git
    GIT_TAG        v1.3.296
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(vk-bootstrap)

# VulkanMemoryAllocator
FetchContent_Declare(
    VulkanMemoryAllocator
    GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
    GIT_TAG        v3.1.0
    GIT_SHALLOW    TRUE
)
set(VMA_BUILD_DOCUMENTATION OFF CACHE BOOL "" FORCE)
set(VMA_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(VulkanMemoryAllocator)

# Dear ImGui (downloaded via FetchContent)
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        v1.91.8
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(imgui)

# ===========================================================================
# Vulkan backend (all platforms with Vulkan SDK)
# ===========================================================================
if(Vulkan_FOUND)
    # ImGui source files (core + Vulkan/GLFW backends)
    set(IMGUI_SOURCES
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_demo.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
    )

    # Source files
    set(SOURCES
        src/main.cpp
        src/vulkan_context.cpp
        src/spv_loader.cpp
        src/raster_renderer.cpp
        src/deferred_renderer.cpp
        src/rt_renderer.cpp
        src/scene_manager.cpp
        src/scene.cpp
        src/camera.cpp
        src/screenshot.cpp
        src/reflected_pipeline.cpp
        src/gltf_loader.cpp
        src/mesh_renderer.cpp
        src/meshlet.cpp
        src/splat_renderer.cpp
        src/rt_splat_renderer.cpp
        src/editor_ui.cpp
        ${IMGUI_SOURCES}
    )

    add_executable(${PROJECT_NAME} ${SOURCES})

    target_include_directories(${PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/deps
        ${imgui_SOURCE_DIR}
        ${imgui_SOURCE_DIR}/backends
    )

    target_link_libraries(${PROJECT_NAME} PRIVATE
        Vulkan::Vulkan
        glfw
        glm::glm
        vk-bootstrap::vk-bootstrap
        GPUOpen::VulkanMemoryAllocator
    )

    # Platform-specific definitions
    if(WIN32)
        target_compile_definitions(${PROJECT_NAME} PRIVATE
            VK_USE_PLATFORM_WIN32_KHR
            NOMINMAX
            WIN32_LEAN_AND_MEAN
        )
    endif()

    target_compile_definitions(${PROJECT_NAME} PRIVATE
        GLFW_INCLUDE_VULKAN
        GLM_FORCE_DEPTH_ZERO_TO_ONE
    )

    # Suppress specific MSVC warnings for third-party code
    if(MSVC)
        target_compile_options(${PROJECT_NAME} PRIVATE /W3 /wd4996 /wd4244 /wd4267)
    else()
        target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wno-unused-parameter)
    endif()
endif()

# ===========================================================================
# Metal backend (macOS only)
# ===========================================================================
if(APPLE)
    # SPIRV-Cross for SPIR-V → MSL transpilation
    FetchContent_Declare(
        spirv-cross
        GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Cross.git
        GIT_TAG        vulkan-sdk-1.4.313.0
        GIT_SHALLOW    TRUE
    )
    set(SPIRV_CROSS_CLI OFF CACHE BOOL "" FORCE)
    set(SPIRV_CROSS_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
    set(SPIRV_CROSS_ENABLE_HLSL OFF CACHE BOOL "" FORCE)
    set(SPIRV_CROSS_ENABLE_GLSL ON CACHE BOOL "" FORCE)  # Required by MSL backend
    set(SPIRV_CROSS_ENABLE_MSL ON CACHE BOOL "" FORCE)
    set(SPIRV_CROSS_ENABLE_CPP OFF CACHE BOOL "" FORCE)
    set(SPIRV_CROSS_ENABLE_REFLECT OFF CACHE BOOL "" FORCE)
    set(SPIRV_CROSS_ENABLE_C_API OFF CACHE BOOL "" FORCE)
    set(SPIRV_CROSS_ENABLE_UTIL OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(spirv-cross)

    # Metal-cpp headers (vendored, header-only)
    set(METAL_CPP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/deps/metal-cpp)

    # ImGui source files for Metal backend (core + GLFW + Metal backends)
    set(IMGUI_METAL_SOURCES
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_demo.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_metal.mm
    )

    # Metal source files
    set(METAL_SOURCES
        src/metal_main.cpp
        src/metal_context.cpp
        src/metal_bridge.mm
        src/metal_shader_transpiler.cpp
        src/metal_raster_renderer.cpp
        src/metal_mesh_renderer.cpp
        src/metal_splat_renderer.cpp
        src/metal_scene.cpp
        src/metal_scene_manager.cpp
        src/metal_screenshot.cpp
        # Shared files (no Vulkan deps)
        src/gltf_loader.cpp
        src/meshlet.cpp
        src/camera.cpp
        src/reflected_pipeline.cpp
        src/scene_geometry.cpp
        # Editor (backend-agnostic panel drawing)
        src/editor_panels.cpp
        # ImGui
        ${IMGUI_METAL_SOURCES}
    )

    # One translation unit must define the Metal-cpp implementations
    set_source_files_properties(src/metal_context.cpp PROPERTIES
        COMPILE_DEFINITIONS "NS_PRIVATE_IMPLEMENTATION;CA_PRIVATE_IMPLEMENTATION;MTL_PRIVATE_IMPLEMENTATION"
    )

    # Mark .mm files as ObjC++
    set_source_files_properties(src/metal_bridge.mm PROPERTIES
        LANGUAGE OBJCXX
        COMPILE_FLAGS "-fobjc-arc"
    )

    # ImGui Metal backend is ObjC++ (NO ARC - imgui uses manual retain/release with IMGUI_IMPL_METAL_CPP)
    set_source_files_properties(${imgui_SOURCE_DIR}/backends/imgui_impl_metal.mm PROPERTIES
        LANGUAGE OBJCXX
        COMPILE_FLAGS "-fno-objc-arc -DIMGUI_IMPL_METAL_CPP"
    )

    add_executable(lux-playground-metal ${METAL_SOURCES})

    target_include_directories(lux-playground-metal PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/deps
        ${METAL_CPP_DIR}
        ${imgui_SOURCE_DIR}
        ${imgui_SOURCE_DIR}/backends
        ${spirv-cross_SOURCE_DIR}/..
        ${spirv-cross_SOURCE_DIR}
    )

    # SPIRV-Cross exports its source root as a public include dir, but the code
    # includes <spirv_cross/spirv_msl.hpp>. Create a compatibility symlink so
    # the include resolves from the parent directory.
    if(NOT EXISTS "${spirv-cross_SOURCE_DIR}/../spirv_cross")
        file(CREATE_LINK "${spirv-cross_SOURCE_DIR}" "${spirv-cross_SOURCE_DIR}/../spirv_cross" SYMBOLIC)
    endif()
    target_include_directories(lux-playground-metal PRIVATE
        "${spirv-cross_SOURCE_DIR}/.."
    )

    target_compile_definitions(lux-playground-metal PRIVATE
        LUX_METAL_BACKEND
        GLM_FORCE_DEPTH_ZERO_TO_ONE
        IMGUI_IMPL_METAL_CPP
    )

    # Link frameworks and libraries
    target_link_libraries(lux-playground-metal PRIVATE
        glfw
        glm::glm
        spirv-cross-msl
        spirv-cross-core
        "-framework Metal"
        "-framework QuartzCore"
        "-framework Foundation"
        "-framework AppKit"
        "-framework IOKit"
        "-framework CoreVideo"
    )

    target_compile_options(lux-playground-metal PRIVATE
        -Wall -Wextra -Wno-unused-parameter -Wno-nullability-completeness
    )

    # Enable ObjC++ for the target
    set_target_properties(lux-playground-metal PROPERTIES
        ENABLE_EXPORTS TRUE
    )
endif()
