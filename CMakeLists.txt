cmake_minimum_required(VERSION 3.17)
cmake_policy(SET CMP0076 NEW)

# Add cmake/ as module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# clangd setting
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Fortran is needed when using find_package(BLAS)
project(monolish CXX Fortran)

file(GLOB_RECURSE MONOLISH_SOURCE_FILES
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  src/*.cpp
)

###############################################################################
# monolish_gpu
###############################################################################

option(BUILD_GPU "build GPU variant" OFF)

if (BUILD_GPU)
  add_library(monolish_gpu SHARED "")

  # TODO: source files should be collected in subdirectory CMakeLists.txt
  #add_subdirectory(src)
  target_sources(monolish_gpu
    PRIVATE
    ${MONOLISH_SOURCE_FILES}
    )
  set_property(TARGET monolish_gpu PROPERTY CXX_STANDARD 14)
  target_compile_options(monolish_gpu PRIVATE -O3)
  target_compile_definitions(monolish_gpu PRIVATE USE_GPU)
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(monolish_gpu PRIVATE "-g3" "-fvar-tracking-assignments")
  endif()

  # OpenACC
  set(OpenACC_ACCEL_TARGET "nvptx-none")
  find_package(OpenACC REQUIRED)
  target_compile_options(monolish_gpu PRIVATE -foffload=nvptx-none="-misa=sm_35")

  # CUDA
  find_package(CUDAToolkit REQUIRED)

  target_link_libraries(monolish_gpu
    PRIVATE
    OpenACC::OpenACC_CXX
    CUDA::cublas
    CUDA::cusolver
    CUDA::cusparse
    CUDA::cudart
  )

  install(
    TARGETS monolish_gpu
    LIBRARY
      DESTINATION lib
  )
endif(BUILD_GPU)

###############################################################################
# monolish_cpu
###############################################################################

add_library(monolish_cpu SHARED "")

# TODO: source files should be collected in subdirectory CMakeLists.txt
#add_subdirectory(src)
target_sources(monolish_cpu
  PRIVATE
  ${MONOLISH_SOURCE_FILES}
  )

set_property(TARGET monolish_cpu PROPERTY CXX_STANDARD 14)
target_compile_options(monolish_cpu PRIVATE -O3)

# AVX
target_compile_options(monolish_cpu PRIVATE "-mavx")
target_compile_definitions(monolish_cpu PRIVATE USE_AVX)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(monolish_cpu PRIVATE "-g3" "-fvar-tracking-assignments")
endif()

# OpenMP
find_package(OpenMP)
if (OpenMP_FOUND)
  target_compile_options(monolish_cpu PRIVATE ${OpenMP_CXX_FLAGS})
  target_link_options(monolish_cpu PRIVATE ${OpenMP_CXX_FLAGS})
  target_link_libraries(monolish_cpu PRIVATE ${OpenMP_CXX_LIBRARIES})
endif (OpenMP_FOUND)

# Intel MKL
find_package(MKL)

# BLAS
if(MKL_FOUND)
  # Not use FindBLAS if MKL is already found
  target_include_directories(monolish_cpu PRIVATE ${MKL_INCLUDE_PATH})
  target_link_libraries(monolish_cpu PRIVATE ${MKL_LIBRARIES})
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

install(
  DIRECTORY include/
  DESTINATION include
  FILES_MATCHING PATTERN "*.hpp")

install(
  TARGETS monolish_cpu
  LIBRARY
    DESTINATION lib
  )


