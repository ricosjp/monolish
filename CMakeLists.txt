cmake_minimum_required(VERSION 3.17)
cmake_policy(SET CMP0076 NEW)

# Add cmake/ as module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# clangd setting
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Fortran is needed when using find_package(BLAS)
project(monolish CXX Fortran)

option(MONOLISH_USE_NVIDIA_GPU "build NVIDIA GPU variant" OFF)
option(MONOLISH_USE_MPI "build MPI variant" OFF)
option(MONOLISH_USE_AVX "Require AVX" ON)
option(MONOLISH_FOR_PACKAGING "build for packaging" OFF)

if(MONOLISH_USE_MPI)
  if(MONOLISH_USE_NVIDIA_GPU)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      if (MONOLISH_FOR_PACKAGING)
        set(MONOLISH_TARGET monolish_gpu_mpi_${GPU_CC})
      else()
        set(MONOLISH_TARGET monolish_gpu_mpi)
      endif()
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      message(FATAL_ERROR "GPU mode cannot be enabled with gcc")
    endif()
  else()
    set(MONOLISH_TARGET monolish_cpu_mpi)
  endif()
else()
  if(MONOLISH_USE_NVIDIA_GPU)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      if (MONOLISH_FOR_PACKAGING)
        set(MONOLISH_TARGET monolish_gpu_${GPU_CC})
      else()
        set(MONOLISH_TARGET monolish_gpu)
      endif()
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      message(FATAL_ERROR "GPU mode cannot be enabled with gcc")
    endif()
  else()
    set(MONOLISH_TARGET monolish_cpu)
  endif()
endif()

file(GLOB_RECURSE MONOLISH_SOURCE_FILES
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  src/*.cpp
)

add_library(${MONOLISH_TARGET} SHARED "")

target_sources(${MONOLISH_TARGET}
  PRIVATE
  ${MONOLISH_SOURCE_FILES}
)

set_property(TARGET ${MONOLISH_TARGET} PROPERTY CXX_STANDARD 17)
target_compile_options(${MONOLISH_TARGET} PRIVATE -g -O3 -Wall)

install(
  TARGETS ${MONOLISH_TARGET}
  LIBRARY
  DESTINATION lib
)

# AVX
if(MONOLISH_USE_AVX)
  target_compile_options(${MONOLISH_TARGET} PRIVATE "-mavx")
  target_compile_definitions(${MONOLISH_TARGET} PRIVATE MONOLISH_USE_AVX)
endif()

# MPI
if(MONOLISH_USE_MPI)
  target_compile_definitions(${MONOLISH_TARGET} PRIVATE MONOLISH_USE_MPI)
endif()

install(
  TARGETS ${MONOLISH_TARGET}
  LIBRARY
  DESTINATION lib
)

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
  endif()
  message(STATUS "GPU Compute Capability is ${GPU_CC}")

  target_compile_options(${MONOLISH_TARGET} PRIVATE
    -fopenmp
    -fopenmp-targets=nvptx64
    -Xopenmp-target -march=sm_${GPU_CC}
  )
  target_link_options(${MONOLISH_TARGET} PRIVATE
    -fopenmp
    -fopenmp-targets=nvptx64
    -Xopenmp-target -march=sm_${GPU_CC}
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

install(
  DIRECTORY include/
  DESTINATION include
  FILES_MATCHING PATTERN "*.hpp"
)
