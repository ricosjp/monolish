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
