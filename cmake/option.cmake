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
option(MONOLISH_PACKAGE_DEV "Create development package including C++ header and examples" OFF)
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
  "ghcr.io/ricosjp/allgebra/cuda11_4/clang13/mkl"
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

# Build with ccache
if(MONOLISH_CCACHE_BUILD)
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${CCACHE_PROGRAM})
  else()
    message(WARNING "Unable to find the program ccache.")
  endif()
endif()
