cmake_minimum_required(VERSION 3.14)
project(PBR_Lux)

set(CMAKE_CXX_STANDARD 17)

# Paths relative to this CMakeLists.txt
set(UPSTREAM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../upstream)
set(LIB_DIR ${UPSTREAM_DIR}/lib)
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(COMMON_DIR ${UPSTREAM_DIR}/src/common)
set(DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/data)

# Find Vulkan SDK
find_package(Vulkan REQUIRED)

# C++ sources
set(CXX_SOURCES
    ${COMMON_DIR}/application.cpp
    ${COMMON_DIR}/image.cpp
    ${COMMON_DIR}/main.cpp
    ${COMMON_DIR}/mesh.cpp
    ${COMMON_DIR}/optimus.cpp
    ${COMMON_DIR}/utils.cpp
    ${SRC_DIR}/vulkan.cpp
)

# C sources (separate to avoid C++ force-includes)
set(C_SOURCES
    ${LIB_DIR}/stb/src/libstb.c
    ${LIB_DIR}/volk/src/volk.c
)

add_executable(PBR_Lux ${CXX_SOURCES} ${C_SOURCES})

target_compile_definitions(PBR_Lux PRIVATE
    NOMINMAX
    GLFW_INCLUDE_NONE
    GLM_ENABLE_EXPERIMENTAL
    ENABLE_VULKAN
    ENABLE_BENCHMARK
)

# Newer MSVC requires explicit includes that older versions included transitively.
# Only apply to C++ sources via set_source_files_properties.
if(MSVC)
    set_source_files_properties(${CXX_SOURCES} PROPERTIES
        COMPILE_FLAGS "/FI\"stdexcept\" /FI\"cstring\""
    )
endif()

target_include_directories(PBR_Lux PRIVATE
    ${UPSTREAM_DIR}/src
    ${LIB_DIR}/glm/include
    ${LIB_DIR}/stb/include
    ${LIB_DIR}/glfw/include
    ${LIB_DIR}/assimp/include
    ${LIB_DIR}/volk/include
    ${Vulkan_INCLUDE_DIRS}
)

# Link libraries (Windows pre-built binaries from upstream)
target_link_directories(PBR_Lux PRIVATE
    ${LIB_DIR}/glfw/win64
    ${LIB_DIR}/assimp/win64
)

target_link_libraries(PBR_Lux PRIVATE
    glfw3dll
    assimp
)

# Copy DLLs and data to output directory
add_custom_command(TARGET PBR_Lux POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${LIB_DIR}/glfw/win64/glfw3.dll
        $<TARGET_FILE_DIR:PBR_Lux>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${LIB_DIR}/assimp/win64/assimp.dll
        $<TARGET_FILE_DIR:PBR_Lux>
)

# Install to data directory for easy running
install(TARGETS PBR_Lux DESTINATION ${DATA_DIR})
