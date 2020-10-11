cmake_minimum_required(VERSION 3.17)
cmake_policy(SET CMP0076 NEW)

# Add cmake/ as module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# clangd setting
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Fortran is needed when using find_package(BLAS)
project(monolish CXX Fortran)

option(MONOLISH_USE_GPU "build GPU variant" OFF)
option(MONOLISH_USE_AVX   "Require AVX" ON)

file(GLOB_RECURSE MONOLISH_SOURCE_FILES
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  src/*.cpp
)

function(common_setup target_name)
  add_library(${target_name} SHARED "")

  target_sources(${target_name}
    PRIVATE
    ${MONOLISH_SOURCE_FILES}
    )

  set_property(TARGET ${target_name} PROPERTY CXX_STANDARD 14)
  target_compile_options(${target_name} PRIVATE -g -O3 -Wall -Wextra)

  # AVX
  if(MONOLISH_USE_AVX)
    target_compile_options(${target_name} PRIVATE "-mavx")
    target_compile_definitions(${target_name} PRIVATE MONOLISH_USE_AVX)
  endif()

  install(
    TARGETS ${target_name}
    LIBRARY
      DESTINATION lib
  )
endfunction()

common_setup(monolish_cpu)

# OpenMP
find_package(OpenMP REQUIRED)
target_compile_options(monolish_cpu PRIVATE ${OpenMP_CXX_FLAGS})
target_link_options(monolish_cpu PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(monolish_cpu PRIVATE ${OpenMP_CXX_LIBRARIES})

# MKL
if(MONOLISH_USE_AVX)
  set(MKL_VML_AVX TRUE)
endif()
find_package(MKL)
if(MKL_FOUND)
  # Not use FindBLAS if MKL is already found
  target_include_directories(monolish_cpu PRIVATE ${MKL_INCLUDE_PATH})
  target_link_libraries(monolish_cpu PRIVATE ${MKL_LIBRARIES})
  target_compile_definitions(monolish_cpu PRIVATE MONOLISH_USE_MKL)
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
  target_include_directories(monolish_cpu PRIVATE ${BLAS_INCLUDE_DIRS})
  target_link_libraries(monolish_cpu PRIVATE ${BLAS_LIBRARIES})
endif()

if(MONOLISH_USE_GPU)
  common_setup(monolish_gpu)
  target_compile_definitions(monolish_gpu PRIVATE USE_GPU)

  # OpenACC
  target_compile_options(monolish_gpu PRIVATE -fopenmp -fno-stack-protector -foffload=nvptx-none="-O3")
#   set(OpenACC_ACCEL_TARGET "nvptx-none")
#   find_package(OpenACC REQUIRED)
#   target_compile_options(monolish_gpu PRIVATE -foffload=nvptx-none="-misa=sm_35")

  # CUDA
  find_package(CUDAToolkit REQUIRED)

  target_link_libraries(monolish_gpu
    PRIVATE
    #OpenACC::OpenACC_CXX
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
