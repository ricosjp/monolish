#include "../../../include/monolish_blas.hpp"
#include "../monolish_internal.hpp"

namespace monolish::internal {

///////////////////////
//   vector utils    //
///////////////////////

// y[i] = a[i]
void vcopy(const size_t N, const int *a, int *y, bool gpu_status) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  if (gpu_status == true) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < N; i++) {
      y[i] = a[i];
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
#pragma omp parallel for
    for (size_t i = 0; i < N; i++) {
      y[i] = a[i];
    }
  }

  logger.func_out();
}

// y[i] = a[i]
void vcopy(const size_t N, const size_t *a, size_t *y, bool gpu_status) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  if (gpu_status == true) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < N; i++) {
      y[i] = a[i];
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
#pragma omp parallel for
    for (size_t i = 0; i < N; i++) {
      y[i] = a[i];
    }
  }

  logger.func_out();
}

// y[i] == a[i]
bool vequal(const size_t N, const int *a, const int *y, bool gpu_status) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  bool ans = true;

  if (gpu_status == true) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < N; i++) {
      if (y[i] != a[i]) {
        ans = false;
      }
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
#pragma omp parallel for
    for (size_t i = 0; i < N; i++) {
      if (y[i] != a[i]) {
        ans = false;
      }
    }
  }
  logger.func_out();
  return ans;
}

// y[i] == a[i]
bool vequal(const size_t N, const size_t *a, const size_t *y, bool gpu_status) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  bool ans = true;

  if (gpu_status == true) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < N; i++) {
      if (y[i] != a[i]) {
        ans = false;
      }
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
#pragma omp parallel for
    for (size_t i = 0; i < N; i++) {
      if (y[i] != a[i]) {
        ans = false;
      }
    }
  }
  logger.func_out();
  return ans;
}

} // namespace monolish::internal
