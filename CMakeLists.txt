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
)

option(MONOLISH_USE_NVIDIA_GPU "build NVIDIA GPU variant" OFF)
option(MONOLISH_USE_MPI "build MPI variant" OFF)
option(MONOLISH_USE_AVX "Require AVX" ON)
option(MONOLISH_FOR_PACKAGING "build for packaging" OFF)

# FIXME: OpenMP Offloading in GCC is broken.
#        See https://gitlab.ritc.jp/ricos/omp-offload-bug for detail
if(MONOLISH_USE_NVIDIA_GPU AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  message(SEND_ERROR "OpenMP Offloading in monolish is only supported with clang")
endif()

file(GLOB_RECURSE MONOLISH_SOURCE_FILES
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  src/*.cpp
)
install(
  DIRECTORY include/
  DESTINATION include
  FILES_MATCHING PATTERN "*.hpp"
)

function(set_common_properties target)
  set_property(TARGET ${target} PROPERTY CXX_STANDARD 17)
  target_compile_options(${target} PRIVATE -g -O3 -Wall)
  # AVX
  if(MONOLISH_USE_AVX)
    target_compile_options(${target} PRIVATE "-mavx")
    target_compile_definitions(${target} PRIVATE MONOLISH_USE_AVX)
  endif()
  # MPI
  if(MONOLISH_USE_MPI)
    target_compile_definitions(${target} PRIVATE MONOLISH_USE_MPI)
  endif()
  install(
    TARGETS ${target}
    LIBRARY
    DESTINATION lib
  )
endfunction(set_common_properties)

if(MONOLISH_USE_MPI)
  if(MONOLISH_USE_NVIDIA_GPU)
    if (MONOLISH_FOR_PACKAGING)
      set(MONOLISH_TARGET monolish_gpu_mpi_${GPU_CC})
    else()
      set(MONOLISH_TARGET monolish_gpu_mpi)
    endif()
  else()
    set(MONOLISH_TARGET monolish_cpu_mpi)
  endif()
else()
  if(MONOLISH_USE_NVIDIA_GPU)
    if (MONOLISH_FOR_PACKAGING)
      set(MONOLISH_TARGET monolish_gpu_${GPU_CC})
    else()
      set(MONOLISH_TARGET monolish_gpu)
    endif()
  else()
    set(MONOLISH_TARGET monolish_cpu)
  endif()
endif()

add_library(${MONOLISH_TARGET} ${MONOLISH_SOURCE_FILES})
set_common_properties(${MONOLISH_TARGET})

# OpenMP
find_package(OpenMP REQUIRED)
if(MONOLISH_USE_NVIDIA_GPU)
  # GPU offloading

  if (NOT MONOLISH_FOR_PACKAGING)
    # get CC
    execute_process(
      COMMAND allgebra_get_device_cc
      OUTPUT_VARIABLE GPU_CC
      ERROR_QUIET
    )
    string(REGEX REPLACE "\n$" "" GPU_CC "${GPU_CC}")

    # get compile_option (for cc86, see #82, #83)
    execute_process(
      COMMAND bash -c "if [ `allgebra_get_device_cc` -eq 86 ]; then printf 80; else allgebra_get_device_cc; fi"
      OUTPUT_VARIABLE GPU_CC_AMENDED
      ERROR_QUIET
    )
    string(REGEX REPLACE "\n$" "" GPU_CC_AMENDED "${GPU_CC_AMENDED}")
  endif()
  message(STATUS "GPU Compute Capability is ${GPU_CC}")

  # for cc86
  if(NOT("${GPU_CC}" EQUAL "${GPU_CC_AMENDED}"))
      message(STATUS "Clang does not support Compute Capability ${GPU_CC}. Instead, compile for Compute Capability ${GPU_CC_AMENDED}.")
  endif()

  target_compile_options(${MONOLISH_TARGET} PRIVATE
    -fopenmp
    -fopenmp-targets=nvptx64
    -Xopenmp-target -march=sm_${GPU_CC_AMENDED}
    -Wno-unknown-cuda-version
  )
  target_link_options(${MONOLISH_TARGET} PRIVATE
    -fopenmp
    -fopenmp-targets=nvptx64
    -Xopenmp-target -march=sm_${GPU_CC_AMENDED}
    -Wno-unknown-cuda-version
  )
else()
  target_compile_options(${MONOLISH_TARGET} PRIVATE ${OpenMP_CXX_FLAGS})
  target_link_options(${MONOLISH_TARGET} PRIVATE ${OpenMP_CXX_FLAGS})
  target_link_libraries(${MONOLISH_TARGET} PRIVATE ${OpenMP_CXX_LIBRARIES})
endif()

# MKL
if(MONOLISH_USE_AVX)
  set(MKL_VML_AVX TRUE)
endif()
find_package(MKL)
if(MKL_FOUND)
  # Not use FindBLAS and FindLAPACK if MKL is already found
  target_include_directories(${MONOLISH_TARGET} PRIVATE ${MKL_INCLUDE_PATH})
  target_link_libraries(${MONOLISH_TARGET} PRIVATE ${MKL_LIBRARIES})
  target_compile_definitions(${MONOLISH_TARGET} PRIVATE MONOLISH_USE_MKL)
else()
  find_package(BLAS REQUIRED)
  # BLAS_INCLUDE_DIRS is missing in FindBLAS.cmake.
  # see https://gitlab.kitware.com/cmake/cmake/-/issues/20268
  find_path(BLAS_INCLUDE_DIRS
    NAMES cblas.h
    HINTS
      /usr/include
      /usr/local/include
      /usr/include/openblas
  )
  target_include_directories(${MONOLISH_TARGET} PRIVATE ${BLAS_INCLUDE_DIRS})
  target_link_libraries(${MONOLISH_TARGET} PRIVATE ${BLAS_LIBRARIES})

  find_package(LAPACK)
  if(LAPACK_FOUND)
    target_include_directories(${MONOLISH_TARGET} PRIVATE ${LAPACK_INCLUDE_DIRS})
    target_link_libraries(${MONOLISH_TARGET} PRIVATE ${LAPACK_LIBRARIES})
    target_compile_definitions(${MONOLISH_TARGET} PRIVATE MONOLISH_USE_LAPACK)
  endif()
endif()

if(MONOLISH_USE_NVIDIA_GPU)
  target_compile_definitions(${MONOLISH_TARGET} PRIVATE MONOLISH_USE_NVIDIA_GPU)
  # Link CUDA libraries
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(${MONOLISH_TARGET}
    PRIVATE
    CUDA::cublas
    CUDA::cusolver
    CUDA::cusparse
    CUDA::cudart
  )
endif()
