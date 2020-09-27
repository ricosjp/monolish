# This file has been originated from FrontISTR project.
#
# Copyright (c) 2019 FrontISTR Commons
# Released under the MIT License
#
# Original LICENSE file is following:
# - https://gitlab.com/FrontISTR-Commons/FrontISTR/-/blob/master/License.txt
#

#
# Output Variables
# -----------------
#
# - MKL_FOUND         TRUE if FindMetis found Intel (R) Math Kernel Library (MKL)
# - MKL_INCLUDE_PATH  Include path of Intel (R) Math Kernel Library (MKL)
# - MKL_LIBRARIES     Intel (R) Math Kernel Library (MKL) libraries
#
# Input Variables
# ----------------
#
# - env MKL_ROOT      Set MKL_ROOT environment variable,
#
# - MKL_VML_AVX       VML, VSL and DF are provided for each SIMD ISA extension
# - MKL_VML_AVX2      These flags specify which library is used.
# - MKL_VML_AVX512    Their priority are AVX512 > AVX2 > AVX,
#                     and default kernel is used if all of them are false.
#

if(MKL_INCLUDE_PATH)
  # Already found
  set(MKL_FOUND TRUE)
  return()
endif()

set(_MKL_INCLUDE_HINTS
  $ENV{MKL_ROOT}/include
  $ENV{HOME}/local/include
  $ENV{HOME}/.local/include
  /opt/intel/mkl/include
  /usr/local/include/mkl
  /usr/include/mkl
  /usr/local/include
  /usr/include
)
find_path(MKL_INCLUDE_PATH NAMES mkl.h HINTS ${_MKL_INCLUDE_HINTS})
unset(_MKL_INCLUDE_HINTS)

set(_MKL_LIBRARY_HINTS
  $ENV{MKL_ROOT}/lib/intel64
  $ENV{HOME}/local/lib
  $ENV{HOME}/.local/lib
  /opt/intel/mkl/lib/intel64
  /usr/lib/x86_64-linux-gnu
  /usr/local/lib
  /usr/lib
)

# Optimized VML/VSL/DF for AVX/AVX2/AVX512
foreach(simd AVX AVX2 AVX512)
  string(TOLOWER "mkl_vml_${simd}" lib_name)
  string(TOUPPER "MKL_VML_${simd}" input_flag)
  if(${input_flag})
    find_library(MKL_VML_LIBRARY NAMES ${lib_name} HINTS ${_MKL_LIBRARY_HINTS})
    if(MKL_VML_LIBRARY)
      message(STATUS "MKL VML/VSL/DF use ${simd} kernel: ${MKL_VML_LIBRARY}")
    endif()
  endif()
endforeach()
# Fallback to default kernel
if(NOT MKL_VML_LIBRARY)
  find_library(MKL_VML_LIBRARY NAMES mkl_vml_def HINTS ${_MKL_LIBRARY_HINTS})
  if(MKL_VML_LIBRARY)
    message(STATUS "MKL VML/VSL/DF use default kernel: ${MKL_VML_LIBRARY}")
  endif()
endif()

# Minimal components
foreach(name intel_lp64 gnu_thread core)
  string(TOUPPER "MKL_${name}_LIBRARY" target)
  string(TOLOWER "mkl_${name}" lib_name)
  find_library(${target} NAMES ${lib_name} HINTS ${_MKL_LIBRARY_HINTS})
  if(${target})
    list(APPEND MKL_LIBRARIES ${${target}})
  else()
    message(WARNING "${lib_name} not found")
  endif()
endforeach()

if(MKL_INCLUDE_PATH AND MKL_LIBRARIES)
  set(MKL_FOUND TRUE)
endif()

list(APPEND MKL_LIBRARIES gomp pthread m dl)

mark_as_advanced(
  MKL_INCLUDE_PATH
  MKL_LIBRARIES
  MKL_VML_LIBRARY
  MKL_INTEL_LP64_LIBRARY
  MKL_GNU_THREAD_LIBRARY
  MKL_CORE_LIBRARY
)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MKL DEFAULT_MSG MKL_LIBRARIES MKL_INCLUDE_PATH)

