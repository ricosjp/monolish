#
# Build container
#
set(monolish_docker_image registry.ritc.jp/ricos/monolish/${monolish_backend}:${monolish_package_version})
set(monolish_docker_release_image ghcr.io/ricosjp/monolish/${monolish_backend}:${monolish_package_version})
check_exec(
  COMMAND git rev-parse --short HEAD
  OUTPUT_VARIABLE git_hash
  ERROR_MSG "Failed to get git hash"
)
check_exec(
  COMMAND date --rfc-3339=second
  OUTPUT_VARIABLE build_date
  ERROR_MSG "Failed to current date"
)
configure_file(package/Dockerfile.in Dockerfile)
configure_file(package/compose.yml.in compose.yml)
add_custom_target(docker
  COMMAND docker-compose build
  COMMENT "Build container ${monolish_docker_image}"
)
