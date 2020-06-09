# In order to use CMP0076, CMake 3.13 is required, but CentOS8 only provides 3.11.4
#cmake_minimum_required(VERSION 3.13)
#cmake_policy(SET CMP0076 NEW)

project(monolish CXX)

add_library(monolish_gpu SHARED "")
#add_subdirectory(src)

# TODO: GLOB is deprecated but used for now
file(GLOB_RECURSE MONOLISH_SOURCE_FILES
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  src/*.cpp
)

target_sources(monolish_gpu
  PRIVATE
  ${MONOLISH_SOURCE_FILES}
)

target_compile_features(monolish_gpu PUBLIC cxx_std_17)

target_compile_options(monolish_gpu PRIVATE -O3)
target_compile_options(monolish_gpu PRIVATE -fopenacc)
target_compile_definitions(monolish_gpu PRIVATE USE_GPU)

# CUDA
foreach(CUDALIB IN ITEMS cublas cusolver cusparse cudart)
  add_library(${CUDALIB} SHARED IMPORTED)
  set_target_properties(${CUDALIB} PROPERTIES
    IMPORTED_LOCATION /usr/local/cuda-10.0/targets/x86_64-linux/lib/lib${CUDALIB}.so
    INTERFACE_INCLUDE_DIRECTORIES /usr/local/cuda-10.0/targets/x86_64-linux/include
  )
endforeach()
target_link_libraries(monolish_gpu
  cublas
  cusolver
  cusparse
  cudart
)
