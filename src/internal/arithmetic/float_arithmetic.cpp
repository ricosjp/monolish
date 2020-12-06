#include "../../include/monolish_blas.hpp"
#include "../monolish_internal.hpp"

#ifdef MONOLISH_MONOLISH_USE_GPU
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#endif

#ifdef MONOLISH_USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#endif

namespace monolish {
  namespace internal {

    // y = a + b
    void vadd(const size_t N, const float* a, const float* b, float* y, bool gpu_status) {
      Logger &logger = Logger::get_instance();
      logger.func_in(monolish_func);

      if (gpu_status == true) {
#if MONOLISH_USE_GPU
#pragma omp target teams distribute parallel for
        for (size_t i = 0; i < N; i++) {
          y[i] = a[i] + b[i];
        }
#else
        throw std::runtime_error(
            "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
      } else {
#if MONOLISH_USE_MKL
        vsAdd(N, a, b, y);
#else
#pragma omp parallel for
        for (size_t i = 0; i < N; i++) {
          y[i] = a[i] + b[i];
        }
#endif
      }
      logger.func_out();
    }

    // y = a - b
    void vsub(const size_t N, const float* a, const float* b, float* y, bool gpu_status) {
      Logger &logger = Logger::get_instance();
      logger.func_in(monolish_func);

      if (gpu_status == true) {
#if MONOLISH_USE_GPU
#pragma omp target teams distribute parallel for
        for (size_t i = 0; i < N; i++) {
          y[i] = a[i] - b[i];
        }
#else
        throw std::runtime_error(
            "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
      } else {
#if MONOLISH_USE_MKL
        vsSub(N, a, b, y);
#else
#pragma omp parallel for
        for (size_t i = 0; i < N; i++) {
          y[i] = a[i] - b[i];
        }
#endif
      }
      logger.func_out();
    }

    // y = a * b
    void vmul(const size_t N, const float* a, const float* b, float* y, bool gpu_status) {
      Logger &logger = Logger::get_instance();
      logger.func_in(monolish_func);

      if (gpu_status == true) {
#if MONOLISH_USE_GPU
#pragma omp target teams distribute parallel for
        for (size_t i = 0; i < N; i++) {
          y[i] = a[i] * b[i];
        }
#else
        throw std::runtime_error(
            "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
      } else {
#if MONOLISH_USE_MKL
        vsMul(N, a, b, y);
#else
#pragma omp parallel for
        for (size_t i = 0; i < N; i++) {
          y[i] = a[i] * b[i];
        }
#endif
      }
      logger.func_out();
    }

    // y = a / b
    void vsiv(const size_t N, const float* a, const float* b, float* y, bool gpu_status) {
      Logger &logger = Logger::get_instance();
      logger.func_in(monolish_func);

      if (gpu_status == true) {
#if MONOLISH_USE_GPU
#pragma omp target teams distribute parallel for
        for (size_t i = 0; i < N; i++) {
          y[i] = a[i] / b[i];
        }
#else
        throw std::runtime_error(
            "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
      } else {
#pragma omp parallel for
        for (size_t i = 0; i < N; i++) {
          y[i] = a[i] / b[i];
        }
      }
      logger.func_out();
    }

  }
} // namespace monolish
