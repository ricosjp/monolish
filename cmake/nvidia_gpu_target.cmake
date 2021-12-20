#
# GPU-specific target properties
#
function(set_gpu_properties target gpu_arch)
  target_compile_definitions(${target}
  PRIVATE
    MONOLISH_USE_NVIDIA_GPU
  )
  target_link_libraries(${target}
  PRIVATE
    CUDA::cublas
    CUDA::cusolver
    CUDA::cusparse
    CUDA::cudart
  )
  # FIXME: clang will support sm_86 on LLVM 13.0.1
  if(gpu_arch STREQUAL "sm_86")
    message(WARNING
      "Clang does not support sm_86 yet. Default to sm_80."
    )
    set(gpu_arch "sm_80")
  endif()
  # OpenMP Offloading setting
  target_compile_options(${target} PRIVATE
    -fopenmp
    -fopenmp-targets=nvptx64
    -Xopenmp-target -march=${gpu_arch}
    -Wno-unknown-cuda-version
  )
  target_link_options(${target} PRIVATE
    -fopenmp
    -fopenmp-targets=nvptx64
    -Xopenmp-target -march=${gpu_arch}
    -Wno-unknown-cuda-version
  )
endfunction()

if(MONOLISH_USE_NVIDIA_GPU)
  set(monolish_gpu_target "monolish_gpu")
  if(MONOLISH_USE_MPI)
    set(monolish_gpu_target ${monolish_gpu_target}_mpi)
  endif()

  if(NOT MONOLISH_NVIDIA_GPU_ARCH_ALL)
    # Build for specified or detected architecture
    add_library(${monolish_gpu_target} ${monolish_sources})
    set_common_properties(${monolish_gpu_target})
    set_gpu_properties(${monolish_gpu_target} ${nvidia_gpu_arch})
    install(
      TARGETS ${monolish_gpu_target}
      LIBRARY
      DESTINATION lib
    )
  else()
    # Build for every CUDA architectures after Maxwell
    # `sm_53` and `sm_62` are ommited since there is no GPU in
    # https://developer.nvidia.com/cuda-gpus
    set(monolish_nvidia_gpu_arch_supported 52 60 61 70 75 80 86)
    foreach(gpu_cc IN LISTS monolish_nvidia_gpu_arch_supported)
      set(target "${monolish_gpu_target}_${gpu_cc}")
      message(STATUS "Add ${target} target")
      add_library(${target} ${monolish_sources})
      set_common_properties(${target})
      set_gpu_properties(${target} "sm_${gpu_cc}")
      install(
        TARGETS ${target}
        LIBRARY
        DESTINATION lib
      )
    endforeach()
  endif()
endif()
