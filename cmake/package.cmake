#
# Packaging
#
# Sell also the "CPack DEB Generator" page
# https://cmake.org/cmake/help/latest/cpack_gen/deb.html
#
set(CPACK_PACKAGE_VENDOR "RICOS Co. Ltd.")
set(CPACK_PACKAGE_CONTACT "Toshiki Teramura <toshiki.teramura@gmail.com>")
set(CPACK_PACKAGE_VERSION "${monolish_package_version}+${monolish_backend}")
set(CPACK_SYSTEM_NAME "amd64")
set(CPACK_GENERATOR "DEB")

# Set dependencies for .deb
#
# The package names are packages in nvidia/cuda image based on Ubuntu 20.04
# distributed on DocekrHub. They may different for other `*.deb`-based Linux distributions.
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libc6, libgcc-s1, libgomp1, libstdc++6")
if(MKL_FOUND)
  set(CPACK_DEBIAN_PACKAGE_DEPENDS "${CPACK_DEBIAN_PACKAGE_DEPENDS}, intel-mkl")
else()
  set(CPACK_DEBIAN_PACKAGE_DEPENDS "${CPACK_DEBIAN_PACKAGE_DEPENDS}, libopenblas0-openmp")
endif()
if(MONOLISH_USE_NVIDIA_GPU)
  set(postfix "${CUDAToolkit_VERSION_MAJOR}-${CUDAToolkit_VERSION_MINOR}")
  set(CPACK_DEBIAN_PACKAGE_DEPENDS "${CPACK_DEBIAN_PACKAGE_DEPENDS}, cuda-cudart-${postfix}, libcusolver-${postfix}, libcusparse-${postfix}, libcublas-${postfix}, libelf1")
  unset(postfix)
endif()

# Although `DEB-DEFAULT` should be preferred, this value will be used in following process.
# Here we define it manually.
#
# https://cmake.org/cmake/help/latest/cpack_gen/deb.html#variable:CPACK_DEBIAN_FILE_NAME
set(CPACK_DEBIAN_FILE_NAME "monolish_${CPACK_PACKAGE_VERSION}_${CPACK_SYSTEM_NAME}.deb")

# FIXME: Add RPM setting

include(CPack)

#
# Build container
#
set(monolish_docker_image ${MONOLISH_CONTAINER_REGISTRY}/${monolish_backend}:${monolish_package_version})
configure_file(docker/Dockerfile.in Dockerfile)
configure_file(docker/compose.yml.in compose.yml)
add_custom_target(docker
  COMMAND docker-compose build
  COMMENT "Build container ${monolish_docker_image}"
)
