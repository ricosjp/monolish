#
# Glob source files
#
file(GLOB_RECURSE monolish_sources
  RELATIVE ${PROJECT_SOURCE_DIR}
  src/*.cpp
)

#
# Common properties
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

if(NOT MONOLISH_USE_NVIDIA_GPU)
  set(monolish_cpu_target "monolish_cpu")
  if(MONOLISH_USE_MPI)
    set(monolish_cpu_target ${monolish_cpu_target}_mpi)
  endif()
  add_library(${monolish_cpu_target} ${monolish_sources})
  set_common_properties(${monolish_cpu_target})
  target_link_libraries(${monolish_cpu_target} PRIVATE OpenMP::OpenMP_CXX)

  foreach(name IN LISTS OpenMP_CXX_LIB_NAMES)
    if(name STREQUAL "omp")
      install(PROGRAMS ${OpenMP_${name}_LIBRARY} TYPE LIB)
    endif()
  endforeach(name)
  install(
    TARGETS ${monolish_cpu_target}
    LIBRARY
    DESTINATION lib
  )
endif()
