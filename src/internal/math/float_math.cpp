#include "../../../include/monolish_blas.hpp"
#include "../monolish_internal.hpp"

namespace monolish {
namespace internal {

void vtanh(const size_t N, const float *a, float *y, bool gpu_status) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  if (gpu_status == true) {
#if MONOLISH_USE_GPU
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < N; i++) {
      y[i] = std::tanh(a[i]);
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
#if MONOLISH_USE_MKL
    vsTanh(N, a, y);
#else
#pragma omp parallel for
    for (size_t i = 0; i < N; i++) {
      y[i] = std::tanh(a[i]);
    }
#endif
  }
  logger.func_out();
}

} // namespace internal
} // namespace monolish
