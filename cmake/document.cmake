#
# document target
#
set(monolish_release_url           "https://github.com/ricosjp/monolish/releases/tag/${PROJECT_VERSION}")
set(monolish_release_download_base "https://github.com/ricosjp/monolish/releases/download/${PROJECT_VERSION}")
set(monolish_deb_common        "monolish_common_${PROJECT_VERSION}.deb")
set(monolish_deb_oss        "monolish_${PROJECT_VERSION}+oss_amd64.deb")
set(monolish_deb_mkl        "monolish_${PROJECT_VERSION}+mkl_amd64.deb")
set(monolish_deb_oss_nvidia "monolish_${PROJECT_VERSION}+oss-nvidia_amd64.deb")
set(monolish_deb_mkl_nvidia "monolish_${PROJECT_VERSION}+mkl-nvidia_amd64.deb")

#
# Eval all markdowns in doc/
#
file(GLOB_RECURSE
  monolish_markdown_documents
  RELATIVE ${PROJECT_SOURCE_DIR}
  doc/*.md
)
foreach(md IN LISTS monolish_markdown_documents)
  configure_file(
    ${PROJECT_SOURCE_DIR}/${md}
    ${CMAKE_CURRENT_BINARY_DIR}/${md}
  )
endforeach()

configure_file(Doxyfile.in Doxyfile)

add_custom_target(document
  COMMAND doxygen
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Generate document by doxygen"
)
