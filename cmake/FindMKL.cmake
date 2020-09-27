###############################################################################
# Copyright (c) 2020 FrontISTR Commons
# This software is released under the MIT License, see License.txt
###############################################################################

# Variables:
#
# MKL_FOUND         TRUE if FindMetis found metis
# MKL_INCLUDE_PATH  Inclue path of metis
# MKL_LIBRARIES     metis libraries
#
# env MKL_ROOT      Set MKL_ROOT environment variable,
#
if(MKL_LIBRARIES)
  set(MKL_FOUND TRUE)
  RETURN()
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
find_library(MKL_INTEL_LP64   NAMES mkl_intel_lp64   HINTS ${_MKL_LIBRARY_HINTS})
find_library(MKL_INTEL_THREAD NAMES mkl_intel_thread HINTS ${_MKL_LIBRARY_HINTS})
find_library(MKL_GNU_THREAD   NAMES mkl_gnu_thread   HINTS ${_MKL_LIBRARY_HINTS})
find_library(MKL_CORE         NAMES mkl_core         HINTS ${_MKL_LIBRARY_HINTS})
unset(_MKL_LIBRARY_HINTS)

set(MKL_LIBRARIES
  ${MKL_INTEL_LP64}
  ${MKL_GNU_THREAD}
  ${MKL_CORE}
  gomp
  pthread
  m
  dl
)

if(MKL_INCLUDE_PATH AND MKL_LIBRARIES)
  set(MKL_FOUND TRUE)
endif()

mark_as_advanced(MKL_INCLUDE_PATH MKL_LIBRARIES)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MKL DEFAULT_MSG MKL_LIBRARIES MKL_INCLUDE_PATH)

