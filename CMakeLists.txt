cmake_minimum_required(VERSION 3.14)
project(lightAI VERSION 1.0.0 LANGUAGES CXX)

option(BUILD_SHARED_LIBS "Build shared library" OFF)
option(BUILD_STATIC_LIBS "Build static library" OFF)

include(FetchContent)

FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
    GIT_SHALLOW    TRUE
)

set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(eigen)

add_library(lightAI_core_objects OBJECT
    core/layer.cpp
    core/lightAI.cpp
)

set_target_properties(lightAI_core_objects PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(lightAI_core_objects
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/core
        ${CMAKE_CURRENT_SOURCE_DIR}/utils
)

target_link_libraries(lightAI_core_objects PUBLIC Eigen3::Eigen)
target_compile_features(lightAI_core_objects PUBLIC cxx_std_17)

if(BUILD_SHARED_LIBS)
    add_library(lightAI_core_shared SHARED $<TARGET_OBJECTS:lightAI_core_objects>)
    target_link_libraries(lightAI_core_shared PUBLIC lightAI_core_objects)
    # Опционально: задать версию и SOVERSION
    set_target_properties(lightAI_core_shared PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
    )
endif()

if(BUILD_STATIC_LIBS)
    add_library(lightAI_core_static STATIC $<TARGET_OBJECTS:lightAI_core_objects>)
    target_link_libraries(lightAI_core_static PUBLIC lightAI_core_objects)
endif()

if(BUILD_SHARED_LIBS)
    add_library(lightAI::core ALIAS lightAI_core_shared)
elseif(BUILD_STATIC_LIBS)
    add_library(lightAI::core ALIAS lightAI_core_static)
endif()
