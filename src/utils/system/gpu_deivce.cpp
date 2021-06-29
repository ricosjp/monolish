#include "../../internal/monolish_internal.hpp"

namespace monolish {
namespace util {

int get_num_devices() {
#if MONOLISH_USE_NVIDIA_GPU
  int devices = 0;
  cudaGetDeviceCount(&devices);
  return devices;
#else
  return -10;
#endif
}

bool set_default_device(size_t device_num) {
#if MONOLISH_USE_NVIDIA_GPU
  cudaSetDevice((int)device_num);
  omp_set_default_device(device_num);
  return true;
#else
  return false;
#endif
}

int get_default_device() {
#if MONOLISH_USE_NVIDIA_GPU
  int device = 0;
  cudaGetDevice(&device);
  return device;
#else
  return -10;
#endif
}

} // namespace util
} // namespace monolish
