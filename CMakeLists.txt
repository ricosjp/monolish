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

include(cmake/option.cmake)
include(cmake/version.cmake)
include(cmake/detect_gpu.cmake)

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

include(cmake/document.cmake)
include(cmake/package.cmake)
