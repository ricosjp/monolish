#
# document target
#
set(monolish_release_url           "https://github.com/ricosjp/monolish/releases/tag/${PROJECT_VERSION}")
set(monolish_release_download_base "https://github.com/ricosjp/monolish/releases/download/${PROJECT_VERSION}")
set(monolish_deb_oss        "monolish_${PROJECT_VERSION}+oss_amd64.deb")
set(monolish_deb_mkl        "monolish_${PROJECT_VERSION}+mkl_amd64.deb")
set(monolish_deb_oss_nvidia "monolish_${PROJECT_VERSION}+oss-nvidia_amd64.deb")
set(monolish_deb_mkl_nvidia "monolish_${PROJECT_VERSION}+mkl-nvidia_amd64.deb")
configure_file(
  ${PROJECT_SOURCE_DIR}/doc/installation/installation.md.in
  ${PROJECT_SOURCE_DIR}/doc/installation/installation.md
)
configure_file(Doxyfile.in Doxyfile)
add_custom_target(document
  COMMAND doxygen
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Generate document by doxygen"
)
