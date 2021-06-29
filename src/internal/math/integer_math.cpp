#include "../../../include/monolish_blas.hpp"
#include "../monolish_internal.hpp"

namespace monolish {
namespace internal {

///////////////////////
//   vector utils    //
///////////////////////

// y[i] = a[i]
size_t vhash(const size_t N, const int *y, const int seed_value,
             bool gpu_status) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  size_t seed = seed_value;

  if (gpu_status == true) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target teams distribute parallel for reduction(^ : seed)
    for (size_t i = 0; i < N; i++) {
      seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
#pragma omp parallel for reduction(^ : seed)
    for (size_t i = 0; i < N; i++) {
      seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
  }

  logger.func_out();

  return seed;
}

size_t vhash(const size_t N, const size_t *y, const size_t seed_value,
             bool gpu_status) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  size_t seed = seed_value;

  if (gpu_status == true) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target teams distribute parallel for reduction(^ : seed)
    for (size_t i = 0; i < N; i++) {
      seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
#pragma omp parallel for reduction(^ : seed)
    for (size_t i = 0; i < N; i++) {
      seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
  }

  logger.func_out();

  return seed;
}

} // namespace internal
} // namespace monolish
