# In order to use CMP0076, CMake 3.13 is required, but CentOS8 only provides 3.11.4
#cmake_minimum_required(VERSION 3.13)
#cmake_policy(SET CMP0076 NEW)
cmake_minimum_required(VERSION 3.10)

# Fortran is needed when using find_package(BLAS)
project(monolish CXX Fortran)

# TODO: GLOB is deprecated but used for now
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

  target_compile_features(monolish_gpu PUBLIC cxx_std_17)

  target_compile_options(monolish_gpu PRIVATE -O3)

  # OpenACC
  set(OpenACC_CXX_FLAGS "-fopenacc")
  target_compile_options(monolish_gpu PRIVATE ${OpenACC_CXX_FLAGS})
  # This should be target_link_options(), but only available after CMake 3.13
  set_target_properties(monolish_gpu PROPERTIES LINK_FLAGS ${OpenACC_CXX_FLAGS})

  target_compile_definitions(monolish_gpu PRIVATE USE_GPU)

  # CUDA
  foreach(CUDALIB IN ITEMS cublas cusolver cusparse cudart)
    add_library(${CUDALIB} SHARED IMPORTED)
    set_target_properties(${CUDALIB} PROPERTIES
      IMPORTED_LOCATION /usr/local/cuda-10.0/targets/x86_64-linux/lib/lib${CUDALIB}.so
      INTERFACE_INCLUDE_DIRECTORIES /usr/local/cuda-10.0/targets/x86_64-linux/include
      )
  endforeach(CUDALIB)
  target_link_libraries(monolish_gpu
    PRIVATE
    cublas
    cusolver
    cusparse
    cudart
    )

  install(
    TARGETS monolish_gpu
    LIBRARY
      DESTINATION lib/monolish
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

target_compile_features(monolish_cpu PUBLIC cxx_std_17)
target_compile_options(monolish_cpu PRIVATE -O3)

find_package(OpenMP)
if (OpenMP_FOUND)
  target_compile_options(monolish_cpu PRIVATE ${OpenMP_CXX_FLAGS})
  # This should be target_link_options(), but only available after CMake 3.13
  set_target_properties(monolish_cpu PROPERTIES LINK_FLAGS ${OpenMP_CXX_FLAGS})
  target_link_libraries(monolish_cpu PRIVATE ${OpenMP_CXX_LIBRARIES})
endif (OpenMP_FOUND)

# TODO: set BLA_VENDOR in modern way
#set(BLA_VENDOR OpenBLAS)
find_package(BLAS)
if (BLAS_FOUND)
  # BLAS_INCLUDE_DIRS is missing in FindBLAS.cmake.
  # see https://gitlab.kitware.com/cmake/cmake/-/issues/20268
  find_path(BLAS_INCLUDE_DIRS cblas.h
    /usr/include
    /usr/local/include
    /usr/include/openblas
    )
  target_include_directories(monolish_cpu PRIVATE ${BLAS_INCLUDE_DIRS})
  target_link_libraries(monolish_cpu PRIVATE ${BLAS_LIBRARIES})
endif (BLAS_FOUND)

install(
  TARGETS monolish_cpu
  LIBRARY
    DESTINATION lib/monolish
  )
