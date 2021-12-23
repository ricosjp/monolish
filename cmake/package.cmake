#
# Install OpenMP runtime library (libomp and libomptarget)
#
# FIXME: This should use libomp distributed by ubuntu
foreach(name IN LISTS OpenMP_CXX_LIB_NAMES)
  if(name STREQUAL "omp")
    install(PROGRAMS ${OpenMP_${name}_LIBRARY} TYPE LIB)
  endif()
endforeach()
if(MONOLISH_USE_NVIDIA_GPU)
  if(NOT DEFINED ENV{ALLGEBRA_LLVM_INSTALL_DIR})
    message(SEND_ERROR "Packaging of GPU variant must run in allgebra container")
  endif()
  install(PROGRAMS
    $ENV{ALLGEBRA_LLVM_INSTALL_DIR}/lib/libomptarget.so
    $ENV{ALLGEBRA_LLVM_INSTALL_DIR}/lib/libomptarget.rtl.cuda.so
    TYPE LIB
  )
endif()

# See also the "CPack DEB Generator" page
# https://cmake.org/cmake/help/latest/cpack_gen/deb.html
set(CPACK_PACKAGE_VENDOR "RICOS Co. Ltd.")
set(CPACK_PACKAGE_CONTACT "Toshiki Teramura <toshiki.teramura@gmail.com>")
set(CPACK_PACKAGE_VERSION "${monolish_package_version}+${monolish_backend}")
set(CPACK_SYSTEM_NAME "amd64")
set(CPACK_GENERATOR "DEB")

# Set dependencies for .deb
#
# The package names are packages in nvidia/cuda image based on Ubuntu 20.04
# distributed on DocekrHub. They may different for other `*.deb`-based Linux distributions.
set(monolish_deb_dependencies
  libc6
  libgcc-s1
  libgomp1
  libstdc++6
  monolish-dev
  )
if(MKL_FOUND)
  list(APPEND monolish_deb_dependencies intel-mkl)
else()
  list(APPEND monolish_deb_dependencies libopenblas0-openmp)
endif()
if(MONOLISH_USE_NVIDIA_GPU)
  set(postfix "${CUDAToolkit_VERSION_MAJOR}-${CUDAToolkit_VERSION_MINOR}")
  list(APPEND
    monolish_deb_dependencies
    cuda-cudart-${postfix}
    libcusolver-${postfix}
    libcusparse-${postfix}
    libcublas-${postfix}
    libelf1
    )
  unset(postfix)
endif()

# Although `DEB-DEFAULT` should be preferred, this value will be used in following process.
# Here we define it manually.
#
# https://cmake.org/cmake/help/latest/cpack_gen/deb.html#variable:CPACK_DEBIAN_FILE_NAME
set(CPACK_DEBIAN_FILE_NAME "monolish_${CPACK_PACKAGE_VERSION}_${CPACK_SYSTEM_NAME}.deb")

if(MONOLISH_NVIDIA_GPU_ARCH_ALL)
  list(JOIN monolish_nvidia_gpu_arch_supported " " monolish_nvidia_gpu_arch_supported_str)
  configure_file(
    ${PROJECT_SOURCE_DIR}/package/postinst.in
    ${PROJECT_SOURCE_DIR}/package/postinst
    @ONLY
  )
  install(FILES
    ${PROJECT_SOURCE_DIR}/package/get_device_cc.cu
    DESTINATION share/monolish/
  )
  install(PROGRAMS
    ${PROJECT_SOURCE_DIR}/package/link_monolish_gpu.sh
    DESTINATION share/monolish/
  )
  # Add post-install script for update-alternative
  set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA
    ${PROJECT_SOURCE_DIR}/package/postinst
  )
endif()

string(JOIN ", " CPACK_DEBIAN_PACKAGE_DEPENDS ${monolish_deb_dependencies})

# FIXME: Add RPM setting

include(CPack)
