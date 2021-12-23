# Copyright 2020 RICOS Co. Ltd.
#
# This file is a part of ricosjp/monolish, distributed under Apache-2.0 License
# https://github.com/ricosjp/monolish
#

#
# Root CMakeLists.txt of monolish project
#
# Naming convention
# ------------------
# - Every variable should have prefix `monolish_` or `MONOLISH_`
#   since cmake does not have namespace.
#
# - Options to be exposed to users should be SCREAMING_SNAKE_CASE
#   - e.g. `MONOLISH_USE_NVIDIA_GPU`
#
# - Internal variables should be snake_case
#   - e.g. `monolish_gpu_target`
#

# CUDA native support and FindCUDAToolkit requires 3.17
# https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html
cmake_minimum_required(VERSION 3.17)

# The target_sources() command converts relative paths to absolute.
# https://cmake.org/cmake/help/latest/policy/CMP0076.html
cmake_policy(SET CMP0076 NEW)

# Generic, but not included modules in cmake itself
# should be placed at cmake/
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# clangd setting
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# FIXME: Add option for building static library
set(BUILD_SHARED_LIBS ON)

project(monolish
  VERSION 0.15.1
  LANGUAGES CXX Fortran # Fortran is needed when using find_package(BLAS)
  HOMEPAGE_URL "https://github.com/ricosjp/monolish"
  DESCRIPTION "MONOlithic LInear equation Solvers for Highly-parallel architecture."
)

# Be sure that include order is not commutable
include(cmake/option.cmake)
include(cmake/version.cmake)

# Do not build anything if `MONOLISH_PACKAGE_DEV` is ON
if(NOT MONOLISH_PACKAGE_DEV)
  include(cmake/detect_gpu.cmake)
  include(cmake/backend.cmake)
  include(cmake/cpu_target.cmake)
  include(cmake/nvidia_gpu_target.cmake)
  include(cmake/package.cmake)
  include(cmake/container.cmake)
else()
  include(cmake/dev.cmake)
endif()

include(cmake/document.cmake)
