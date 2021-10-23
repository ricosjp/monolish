function(check_exec)
  cmake_parse_arguments(CHECK_EXEC "" "OUTPUT_VARIABLE;ERROR_MSG" "COMMAND" ${ARGN})
  execute_process(
    COMMAND ${CHECK_EXEC_COMMAND}
    OUTPUT_VARIABLE ${CHECK_EXEC_OUTPUT_VARIABLE}
    RESULT_VARIABLE exit_code
    OUTPUT_STRIP_TRAILING_WHITESPACE
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
  if(NOT exit_code EQUAL 0)
    message(SEND_ERROR ${CHECK_EXEC_ERROR_MSG})
  endif()
  set(${CHECK_EXEC_OUTPUT_VARIABLE} "${${CHECK_EXEC_OUTPUT_VARIABLE}}" PARENT_SCOPE)
endfunction(check_exec)

if(NOT MONOLISH_PACKAGE_VERSION)
  # Get the hash of the last release
  check_exec(
    COMMAND git rev-list --tags --max-count=1
    OUTPUT_VARIABLE git_last_tag_hash
    ERROR_MSG "Cannot get hash of last release"
  )
  # count the number of commit from the last release
  check_exec(
    COMMAND git rev-list ${git_last_tag_hash}.. --count
    OUTPUT_VARIABLE git_dev_commits
    ERROR_MSG "Failed to count commit numbers from last release"
  )

  # If the current commit is equal to last tag,
  # it must be a release commit.
  if(git_dev_commits EQUAL 0)
    set(monolish_package_version "${PROJECT_VERSION}")
  else()
    set(monolish_package_version "${PROJECT_VERSION}-dev.${git_dev_commits}")
  endif()
else()
  set(monolish_package_version ${MONOLISH_PACKAGE_VERSION})
endif()
message(STATUS "monolish package version = ${monolish_package_version}")


