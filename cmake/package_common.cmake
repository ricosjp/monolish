#
# Packaging monolish-dev.deb
#
install(
  DIRECTORY include/
  DESTINATION include
  FILES_MATCHING PATTERN "*.hpp"
)
install(
  DIRECTORY examples/
  DESTINATION share/monolish/examples
)
install(
  DIRECTORY benchmark/
  DESTINATION share/monolish/benchmark
)

#
# Install OpenMP runtime library (libomp, libomptarget)
#
# FIXME: libomp* should be an independent deb package.
#
if(NOT DEFINED ENV{ALLGEBRA_LLVM_INSTALL_DIR})
  message(SEND_ERROR "Packaging must run in allgebra container")
endif()
install(PROGRAMS
  $ENV{ALLGEBRA_LLVM_INSTALL_DIR}/lib/libomp.so
  $ENV{ALLGEBRA_LLVM_INSTALL_DIR}/lib/libomptarget.so
  $ENV{ALLGEBRA_LLVM_INSTALL_DIR}/lib/libomptarget.rtl.cuda.so
  TYPE LIB
)

set(CPACK_PACKAGE_NAME "libmonolish-common")
set(CPACK_PACKAGE_VENDOR "RICOS Co. Ltd.")
set(CPACK_PACKAGE_CONTACT "Toshiki Teramura <toshiki.teramura@gmail.com>")
set(CPACK_PACKAGE_VERSION "${monolish_package_version}")

set(CPACK_GENERATOR "DEB")
set(CPACK_DEBIAN_PACKAGE_DEPENDS "")

set(CPACK_DEBIAN_FILE_NAME "monolish_common_${CPACK_PACKAGE_VERSION}.deb")

include(CPack)
