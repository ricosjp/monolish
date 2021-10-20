# Copyright 2020 RICOS Co. Ltd.
#
# This file is a part of ricosjp/monolish, distributed under Apache-2.0 License
# https://github.com/ricosjp/monolish
#

#
# Root CMakeLists.txt of monolish project
#
# Naming convention
# ------------------
# - Every variable should have prefix `monolish_` or `MONOLISH_`
#   since cmake does not have namespace.
#
# - Options to be exposed to users should be SCREAMING_SNAKE_CASE
#   - e.g. `MONOLISH_USE_NVIDIA_GPU`
#
# - Internal variables should be snake_case
#   - e.g. `monolish_gpu_target`
#

# CUDA native support and FindCUDAToolkit requires 3.17
# https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html
cmake_minimum_required(VERSION 3.17)

# The target_sources() command converts relative paths to absolute.
# https://cmake.org/cmake/help/latest/policy/CMP0076.html
cmake_policy(SET CMP0076 NEW)

# Generic, but not included modules in cmake itself
# should be placed at cmake/
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# clangd setting
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# FIXME: Add option for building static library
set(BUILD_SHARED_LIBS ON)

project(monolish
  VERSION 0.15.0
  LANGUAGES CXX Fortran # Fortran is needed when using find_package(BLAS)
  HOMEPAGE_URL "https://github.com/ricosjp/monolish"
  DESCRIPTION "MONOlithic LInear equation Solvers for Highly-parallel architecture."
)

#
# Build options
#
function(option_string name default help)
  set(${name} ${default} CACHE STRING ${help})
  # environment variable takes priority
  if(DEFINED ENV{${name}})
    message(STATUS "Overwrite by env variable: ${name}=$ENV{${name}}")
    set(${name} "$ENV{${name}}" CACHE STRING ${help} FORCE)
  endif()
endfunction()

option(MONOLISH_USE_AVX "Require AVX" ON)
option(MONOLISH_USE_MPI "Build with MPI" OFF)
option(MONOLISH_USE_NVIDIA_GPU "Build with NVIDIA GPU" OFF)
option(MONOLISH_NVIDIA_GPU_ARCH_ALL "Build library for all known CUDA architectures" OFF)
option(MONOLISH_CCACHE_BUILD "Use ccache" ON)
option_string(
  MONOLISH_NVIDIA_GPU_ARCH
  ""
  "NVIDIA GPU architecture used while OpenMP Offloading. e.g. sm_52"
)
option_string(
  MONOLISH_PACKAGE_VERSION
  ""
  "Package version for deb and containers."
)

# For container
option_string(
  MONOLISH_CONTAINER_REGISTRY
  "ghcr.io/ricosjp/monolish"
  "Container registry where monolish container will be uploaded"
)
option_string(
  MONOLISH_CONTAINER_BASE_IMAGE
  "ghcr.io/ricosjp/allgebra/cuda11_4/mkl"
  "Base image upon which monolish container will be created"
)
option_string(
  MONOLISH_CONTAINER_BASE_TAG
  "21.09.0"
  "Tag of MONOLISH_CONTAINER_BASE_IMAGE"
)

#
# Validate options
#
# FIXME: OpenMP Offloading in GCC is broken.
#        See https://gitlab.ritc.jp/ricos/omp-offload-bug for detail
if(MONOLISH_USE_NVIDIA_GPU AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  message(WARNING "OpenMP Offloading in monolish is only supported with clang")
endif()

if(NOT MONOLISH_PACKAGE_VERSION)
  # Get last release tag by `git describe`
  execute_process(
    COMMAND git describe --tags --abbrev=0
    OUTPUT_VARIABLE git_last_tag
    RESULT_VARIABLE exit_code
    OUTPUT_STRIP_TRAILING_WHITESPACE
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
  if(NOT exit_code EQUAL 0)
    message(SEND_ERROR "`git describe` command failed. Cannot generate package version.")
  endif()

  # count the number of commit from there by `git rev-list`
  execute_process(
    COMMAND git rev-list ${git_last_tag}.. --count
    OUTPUT_VARIABLE git_dev_commits
    RESULT_VARIABLE exit_code
    OUTPUT_STRIP_TRAILING_WHITESPACE
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
  if(NOT exit_code EQUAL 0)
    message(SEND_ERROR "`git rev-list` command failed. Cannot generate package version.")
  endif()

  # If the current commit is equal to last tag,
  # it must be a release commit.
  if(git_dev_commits EQUAL 0)
    set(monolish_package_version "${PROJECT_VERSION}")
  else()
    set(monolish_package_version "${PROJECT_VERSION}-dev.${git_dev_commits}")
  endif()
else()
  set(monolish_package_version ${MONOLISH_PACKAGE_VERSION})
endif()
message(STATUS "monolish package version = ${monolish_package_version}")

# Build with ccache
if(MONOLISH_CCACHE_BUILD)
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${CCACHE_PROGRAM})
  else()
    message(WARNING "Unable to find the program ccache.")
  endif()
endif()

# Detect host NVIDIA GPU architecture if not specified
# since clang requires CUDA architecture flag e.g. `-march=sm_80`
if(MONOLISH_USE_NVIDIA_GPU AND NOT MONOLISH_NVIDIA_GPU_ARCH_ALL)
  if(NOT MONOLISH_NVIDIA_GPU_ARCH)
    execute_process(
      COMMAND nvcc
        -o get_device_cc
        --run
        --run-args 0
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/get_device_cc.cu
      RESULT_VARIABLE exit_code
      OUTPUT_VARIABLE gpu_cc
      OUTPUT_STRIP_TRAILING_WHITESPACE
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    if(NOT exit_code EQUAL 0)
      message(SEND_ERROR "NVIDIA GPU architecture is not specified, and allgebra_get_device_cc failed to detect host GPU architecture.")
    endif()
    # clang does not support sm_86 yet
    if("${gpu_cc}" EQUAL "86")
      message(WARNING "Clang does not support sm_86 yet. Default to sm_80.")
      set(gpu_cc "80")
    endif()
    set(nvidia_gpu_arch "sm_${gpu_cc}")
  else()
    set(nvidia_gpu_arch "${MONOLISH_NVIDIA_GPU_ARCH}")
  endif()
  message(STATUS "NVIDIA GPU Architecture for OpenMP Offloading = ${nvidia_gpu_arch}")
endif()

#
# Find external packages
#
find_package(OpenMP REQUIRED)
# See cmake/FindMKL.cmake
find_package(MKL)
# Do not use FindBLAS.cmake and FindLAPACK.cmake if MKL is already found
if(NOT MKL_FOUND)
  find_package(BLAS REQUIRED)
  find_package(LAPACK)
  # BLAS_INCLUDE_DIRS is missing in FindBLAS.cmake.
  # see https://gitlab.kitware.com/cmake/cmake/-/issues/20268
  find_path(BLAS_INCLUDE_DIRS
    NAMES cblas.h
    HINTS
      /usr/include
      /usr/local/include
      /usr/include/openblas
  )
  set(monolish_backend "oss")
else()
  set(monolish_backend "mkl")
endif()
if(MONOLISH_USE_NVIDIA_GPU)
  find_package(CUDAToolkit REQUIRED)
  set(monolish_backend "${monolish_backend}-nvidia")
  if(NOT MONOLISH_NVIDIA_GPU_ARCH_ALL)
    # Remove `_` since it should be a special charactor for deb filename
    string(REPLACE "_" "" gpu_arch ${nvidia_gpu_arch})
    set(monolish_backend "${monolish_backend}-${gpu_arch}")
  endif()
endif()
message(STATUS "Backend = ${monolish_backend}")

#
# Glob source files
#
file(GLOB_RECURSE monolish_sources
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  src/*.cpp
)
install(
  DIRECTORY include/
  DESTINATION include
  FILES_MATCHING PATTERN "*.hpp"
)
install(
  DIRECTORY examples/
  DESTINATION share/monolish/examples
)
install(
  DIRECTORY benchmark/
  DESTINATION share/monolish/benchmark
)

#
# Target properties
#
function(set_common_properties target)
  set_property(TARGET ${target} PROPERTY CXX_STANDARD 17)
  target_compile_options(${target} PRIVATE -O3 -Wall)
  # AVX
  if(MONOLISH_USE_AVX)
    target_compile_options(${target} PRIVATE "-mavx")
    target_compile_definitions(${target} PRIVATE MONOLISH_USE_AVX)
    set(MKL_VML_AVX TRUE)
  endif()
  # MPI
  if(MONOLISH_USE_MPI)
    target_compile_definitions(${target} PRIVATE MONOLISH_USE_MPI)
  endif()
  # BLAS/LAPACK
  if(MKL_FOUND)
    target_include_directories(${target} PRIVATE ${MKL_INCLUDE_PATH})
    target_link_libraries(${target} PRIVATE ${MKL_LIBRARIES})
    target_compile_definitions(${target} PRIVATE MONOLISH_USE_MKL)
  else()
    target_include_directories(${target} PRIVATE ${BLAS_INCLUDE_DIRS})
    target_link_libraries(${target} PRIVATE ${BLAS_LIBRARIES})
    if(LAPACK_FOUND)
      target_include_directories(${target} PRIVATE ${LAPACK_INCLUDE_DIRS})
      target_link_libraries(${target} PRIVATE ${LAPACK_LIBRARIES})
      target_compile_definitions(${target} PRIVATE MONOLISH_USE_LAPACK)
    endif()
  endif()
endfunction()

function(set_gpu_properties target gpu_arch)
  target_compile_definitions(${target}
  PRIVATE
    MONOLISH_USE_NVIDIA_GPU
  )
  target_link_libraries(${target}
  PRIVATE
    CUDA::cublas
    CUDA::cusolver
    CUDA::cusparse
    CUDA::cudart
  )
  # OpenMP Offloading setting
  target_compile_options(${target} PRIVATE
    -fopenmp
    -fopenmp-targets=nvptx64
    -Xopenmp-target -march=${gpu_arch}
    -Wno-unknown-cuda-version
  )
  target_link_options(${target} PRIVATE
    -fopenmp
    -fopenmp-targets=nvptx64
    -Xopenmp-target -march=${gpu_arch}
    -Wno-unknown-cuda-version
  )
endfunction()

#
# Define targets
#

# monolish on CPU
if(NOT MONOLISH_USE_NVIDIA_GPU)
  set(monolish_cpu_target "monolish_cpu")
  if(MONOLISH_USE_MPI)
    set(monolish_cpu_target ${monolish_cpu_target}_mpi)
  endif()
  add_library(${monolish_cpu_target} ${monolish_sources})
  set_common_properties(${monolish_cpu_target})
  # XXX: Should we use the imported target `OpenMP::OpenMP_CXX`?
  target_compile_options(${monolish_cpu_target} PRIVATE ${OpenMP_CXX_FLAGS})
  target_link_options(${monolish_cpu_target} PRIVATE ${OpenMP_CXX_FLAGS})
  target_link_libraries(${monolish_cpu_target} PRIVATE ${OpenMP_CXX_LIBRARIES})
  install(
    TARGETS ${monolish_cpu_target}
    LIBRARY
    DESTINATION lib
  )
endif()

# monolish with NVIDIA GPU
if(MONOLISH_USE_NVIDIA_GPU)
  set(monolish_gpu_target "monolish_gpu")
  if(MONOLISH_USE_MPI)
    set(monolish_gpu_target ${monolish_gpu_target}_mpi)
  endif()

  if(NOT MONOLISH_NVIDIA_GPU_ARCH_ALL)
    # Build for specified or detected architecture
    add_library(${monolish_gpu_target} ${monolish_sources})
    set_common_properties(${monolish_gpu_target})
    set_gpu_properties(${monolish_gpu_target} ${nvidia_gpu_arch})
    install(
      TARGETS ${monolish_gpu_target}
      LIBRARY
      DESTINATION lib
    )
  else()
    # Build for every CUDA architectures after Maxwell
    # `sm_53` and `sm_62` are ommited since there is no GPU in
    # https://developer.nvidia.com/cuda-gpus
    foreach(gpu_cc IN ITEMS 52 60 61 70 75 80)
      set(target "${monolish_gpu_target}_${gpu_cc}")
      message(STATUS "Add ${target} target")
      add_library(${target} ${monolish_sources})
      set_common_properties(${target})
      set_gpu_properties(${target} "sm_${gpu_cc}")
      install(
        TARGETS ${target}
        LIBRARY
        DESTINATION lib
      )
    endforeach()
  endif()
endif()

#
# Packaging
#
# Sell also the "CPack DEB Generator" page
# https://cmake.org/cmake/help/latest/cpack_gen/deb.html
#
set(CPACK_PACKAGE_VENDOR "RICOS Co. Ltd.")
set(CPACK_PACKAGE_CONTACT "Toshiki Teramura <toshiki.teramura@gmail.com>")
set(CPACK_PACKAGE_VERSION "${monolish_package_version}+${monolish_backend}")
set(CPACK_SYSTEM_NAME "amd64")
set(CPACK_GENERATOR "DEB")

# Set dependencies for .deb
#
# The package names are packages in nvidia/cuda image based on Ubuntu 20.04
# distributed on DocekrHub. They may different for other `*.deb`-based Linux distributions.
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libc6, libgcc-s1, libgomp1, libstdc++6")
if(MKL_FOUND)
  set(CPACK_DEBIAN_PACKAGE_DEPENDS "${CPACK_DEBIAN_PACKAGE_DEPENDS}, intel-mkl")
else()
  set(CPACK_DEBIAN_PACKAGE_DEPENDS "${CPACK_DEBIAN_PACKAGE_DEPENDS}, libopenblas0-openmp")
endif()
if(MONOLISH_USE_NVIDIA_GPU)
  set(postfix "${CUDAToolkit_VERSION_MAJOR}-${CUDAToolkit_VERSION_MINOR}")
  set(CPACK_DEBIAN_PACKAGE_DEPENDS "${CPACK_DEBIAN_PACKAGE_DEPENDS}, cuda-cudart-${postfix}, libcusolver-${postfix}, libcusparse-${postfix}, libcublas-${postfix}, libelf1")
  unset(postfix)
endif()

# Although `DEB-DEFAULT` should be preferred, this value will be used in following process.
# Here we define it manually.
#
# https://cmake.org/cmake/help/latest/cpack_gen/deb.html#variable:CPACK_DEBIAN_FILE_NAME
set(CPACK_DEBIAN_FILE_NAME "monolish_${CPACK_PACKAGE_VERSION}_${CPACK_SYSTEM_NAME}.deb")

# FIXME: Add RPM setting

include(CPack)

#
# Build container
#
set(docker_tag ${MONOLISH_CONTAINER_REGISTRY}/${monolish_backend}:${monolish_package_version})
configure_file(docker/Dockerfile.in Dockerfile)
configure_file(docker/compose.yml.in compose.yml)
add_custom_target(docker
  COMMAND docker-compose build
  COMMENT "Build container ${docker_tag}"
)
