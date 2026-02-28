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

# ===========================================================================
# Vulkan backend (all platforms with Vulkan SDK)
# ===========================================================================
if(Vulkan_FOUND)
    # Source files
    set(SOURCES
        src/main.cpp
        src/vulkan_context.cpp
        src/spv_loader.cpp
        src/raster_renderer.cpp
        src/rt_renderer.cpp
        src/scene_manager.cpp
        src/scene.cpp
        src/camera.cpp
        src/screenshot.cpp
        src/reflected_pipeline.cpp
        src/gltf_loader.cpp
        src/mesh_renderer.cpp
        src/meshlet.cpp
    )

    add_executable(${PROJECT_NAME} ${SOURCES})

    target_include_directories(${PROJECT_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/deps
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

    # Metal source files
    set(METAL_SOURCES
        src/metal_main.cpp
        src/metal_context.cpp
        src/metal_bridge.mm
        src/metal_shader_transpiler.cpp
        src/metal_raster_renderer.cpp
        src/metal_mesh_renderer.cpp
        src/metal_scene.cpp
        src/metal_scene_manager.cpp
        src/metal_screenshot.cpp
        # Shared files (no Vulkan deps)
        src/gltf_loader.cpp
        src/meshlet.cpp
        src/camera.cpp
        src/reflected_pipeline.cpp
        src/scene_geometry.cpp
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

    add_executable(lux-playground-metal ${METAL_SOURCES})

    target_include_directories(lux-playground-metal PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/deps
        ${METAL_CPP_DIR}
    )

    target_compile_definitions(lux-playground-metal PRIVATE
        LUX_METAL_BACKEND
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
