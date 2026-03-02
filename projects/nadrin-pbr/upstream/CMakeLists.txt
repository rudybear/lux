cmake_minimum_required(VERSION 3.14)
project(PBR_Original)

set(CMAKE_CXX_STANDARD 17)

set(LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(COMMON_DIR ${SRC_DIR}/common)

find_package(Vulkan REQUIRED)

set(CXX_SOURCES
    ${COMMON_DIR}/application.cpp
    ${COMMON_DIR}/image.cpp
    ${COMMON_DIR}/main.cpp
    ${COMMON_DIR}/mesh.cpp
    ${COMMON_DIR}/optimus.cpp
    ${COMMON_DIR}/utils.cpp
    ${SRC_DIR}/vulkan.cpp
)

set(C_SOURCES
    ${LIB_DIR}/stb/src/libstb.c
    ${LIB_DIR}/volk/src/volk.c
)

add_executable(PBR_Original ${CXX_SOURCES} ${C_SOURCES})

target_compile_definitions(PBR_Original PRIVATE
    NOMINMAX
    GLFW_INCLUDE_NONE
    GLM_ENABLE_EXPERIMENTAL
    ENABLE_VULKAN
)

if(MSVC)
    set_source_files_properties(${CXX_SOURCES} PROPERTIES
        COMPILE_FLAGS "/FI\"stdexcept\" /FI\"cstring\""
    )
endif()

target_include_directories(PBR_Original PRIVATE
    ${SRC_DIR}
    ${LIB_DIR}/glm/include
    ${LIB_DIR}/stb/include
    ${LIB_DIR}/glfw/include
    ${LIB_DIR}/assimp/include
    ${LIB_DIR}/volk/include
    ${Vulkan_INCLUDE_DIRS}
)

target_link_directories(PBR_Original PRIVATE
    ${LIB_DIR}/glfw/win64
    ${LIB_DIR}/assimp/win64
)

target_link_libraries(PBR_Original PRIVATE
    glfw3dll
    assimp
)

add_custom_command(TARGET PBR_Original POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${LIB_DIR}/glfw/win64/glfw3.dll
        $<TARGET_FILE_DIR:PBR_Original>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${LIB_DIR}/assimp/win64/assimp.dll
        $<TARGET_FILE_DIR:PBR_Original>
)
