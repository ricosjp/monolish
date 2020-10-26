cmake_minimum_required(VERSION 3.17)
cmake_policy(SET CMP0076 NEW)

# Add cmake/ as module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# clangd setting
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Fortran is needed when using find_package(BLAS)
project(monolish CXX Fortran)

option(MONOLISH_USE_GPU "build GPU variant" OFF)
option(MONOLISH_USE_AVX "Require AVX" ON)

if(MONOLISH_USE_GPU)
  set(MONOLISTH_TARGET monolish_gpu)
else()
  set(MONOLISTH_TARGET monolish_cpu)
endif()

file(GLOB_RECURSE MONOLISH_SOURCE_FILES
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  src/*.cpp
)

add_library(${MONOLISTH_TARGET} SHARED "")

target_sources(${MONOLISTH_TARGET}
  PRIVATE
  ${MONOLISH_SOURCE_FILES}
)

set_property(TARGET ${MONOLISTH_TARGET} PROPERTY CXX_STANDARD 14)
target_compile_options(${MONOLISTH_TARGET} PRIVATE -g -O3 -Wall -Wextra)

# AVX
if(MONOLISH_USE_AVX)
  target_compile_options(${MONOLISTH_TARGET} PRIVATE "-mavx")
  target_compile_definitions(${MONOLISTH_TARGET} PRIVATE MONOLISH_USE_AVX)
endif()

install(
  TARGETS ${MONOLISTH_TARGET}
  LIBRARY
  DESTINATION lib
)

# OpenMP
find_package(OpenMP REQUIRED)
if(MONOLISH_USE_GPU)
  # GPU offloading
  target_compile_options(${MONOLISTH_TARGET} PRIVATE
    -fopenmp
    -fno-stack-protector
    -fcf-protection=none
    -foffload=nvptx-none="-O3"
  )
  target_link_options(${MONOLISTH_TARGET} PRIVATE
    -fopenmp
    -fno-stack-protector
    -fcf-protection=none
    -foffload=nvptx-none="-O3"
  )
else()
  target_compile_options(${MONOLISTH_TARGET} PRIVATE ${OpenMP_CXX_FLAGS})
  target_link_options(${MONOLISTH_TARGET} PRIVATE ${OpenMP_CXX_FLAGS})
  target_link_libraries(${MONOLISTH_TARGET} PRIVATE ${OpenMP_CXX_LIBRARIES})
endif()

# MKL
if(MONOLISH_USE_AVX)
  set(MKL_VML_AVX TRUE)
endif()
find_package(MKL)
if(MKL_FOUND)
  # Not use FindBLAS if MKL is already found
  target_include_directories(${MONOLISTH_TARGET} PRIVATE ${MKL_INCLUDE_PATH})
  target_link_libraries(${MONOLISTH_TARGET} PRIVATE ${MKL_LIBRARIES})
  target_compile_definitions(${MONOLISTH_TARGET} PRIVATE MONOLISH_USE_MKL)
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
  target_include_directories(${MONOLISTH_TARGET} PRIVATE ${BLAS_INCLUDE_DIRS})
  target_link_libraries(${MONOLISTH_TARGET} PRIVATE ${BLAS_LIBRARIES})
endif()

if(MONOLISH_USE_GPU)
  target_compile_definitions(${MONOLISTH_TARGET} PRIVATE MONOLISH_USE_GPU)
  # Link CUDA libraries
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(${MONOLISTH_TARGET}
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
