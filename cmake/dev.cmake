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

set(CPACK_PACKAGE_NAME "libmonolish-dev")
set(CPACK_PACKAGE_VENDOR "RICOS Co. Ltd.")
set(CPACK_PACKAGE_CONTACT "Toshiki Teramura <toshiki.teramura@gmail.com>")
set(CPACK_PACKAGE_VERSION "${monolish_package_version}")

set(CPACK_GENERATOR "DEB")
set(CPACK_DEBIAN_PACKAGE_DEPENDS "")

set(CPACK_DEBIAN_FILE_NAME "monolish_dev_${CPACK_PACKAGE_VERSION}.deb")

include(CPack)
