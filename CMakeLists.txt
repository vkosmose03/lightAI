cmake_minimum_required(VERSION 3.14)
project(lightAI VERSION 1.0.0 LANGUAGES CXX)

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

add_library(lightAI_core
    core/layer.cpp
    core/lightAI.cpp
)

set_target_properties(lightAI_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(lightAI_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/core
        ${CMAKE_CURRENT_SOURCE_DIR}/utils
)

target_link_libraries(lightAI_core PUBLIC Eigen3::Eigen)

target_compile_features(lightAI_core PUBLIC cxx_std_17)

add_library(lightAI::core ALIAS lightAI_core)
