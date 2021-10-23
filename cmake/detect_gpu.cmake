# Detect host NVIDIA GPU architecture if not specified
# since clang requires CUDA architecture flag e.g. `-march=sm_80`
if(MONOLISH_USE_NVIDIA_GPU AND NOT MONOLISH_NVIDIA_GPU_ARCH_ALL)
  if(NOT MONOLISH_NVIDIA_GPU_ARCH)
    execute_process(
      COMMAND nvcc
        -o get_device_cc
        --run
        --run-args 0
        ${PROJECT_SOURCE_DIR}/cmake/get_device_cc.cu
      RESULT_VARIABLE exit_code
      OUTPUT_VARIABLE gpu_cc
      OUTPUT_STRIP_TRAILING_WHITESPACE
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    if(NOT exit_code EQUAL 0)
      message(SEND_ERROR "NVIDIA GPU architecture is not specified, and failed to detect host GPU architecture.")
    endif()
    # clang does not support sm_86 yet
    if("${gpu_cc}" EQUAL "86")
      message(WARNING "Clang does not support sm_86 yet. Default to sm_80.")
      set(gpu_cc "80")
    endif()
    set(nvidia_gpu_arch "sm_${gpu_cc}")
  else()
    set(nvidia_gpu_arch "${MONOLISH_NVIDIA_GPU_ARCH}")
  endif()
  message(STATUS "NVIDIA GPU Architecture for OpenMP Offloading = ${nvidia_gpu_arch}")
endif()
