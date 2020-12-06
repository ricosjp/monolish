#include "../monolish_internal.hpp"
#include "../../../include/monolish_blas.hpp"

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
///////////////////////
//   scalar-vector   //
///////////////////////
    // y[i] = a[i] + alpha
    void vadd(const size_t N, const double* a, const double alpha, double* y, bool gpu_status) {
      Logger &logger = Logger::get_instance();
      logger.func_in(monolish_func);

      if (gpu_status == true) {
#if MONOLISH_USE_GPU
#pragma omp target teams distribute parallel for
        for (size_t i = 0; i < N; i++) {
          y[i] = a[i] + alpha;
        }
#else
        throw std::runtime_error(
            "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
      } else {
#pragma omp parallel for
        for (size_t i = 0; i < N; i++) {
          y[i] = a[i] + alpha;
        }
      }
      logger.func_out();
    }

    // y[i] = a[i] - alpha
    void vsub(const size_t N, const double* a, const double alpha, double* y, bool gpu_status) {
      Logger &logger = Logger::get_instance();
      logger.func_in(monolish_func);

      if (gpu_status == true) {
#if MONOLISH_USE_GPU
#pragma omp target teams distribute parallel for
        for (size_t i = 0; i < N; i++) {
          y[i] = a[i] - alpha;
        }
#else
        throw std::runtime_error(
            "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
      } else {
#pragma omp parallel for
        for (size_t i = 0; i < N; i++) {
          y[i] = a[i] - alpha;
        }
      }
      logger.func_out();
    }

    // y[i] = a[i] * alpha
    void vmul(const size_t N, const double* a, const double alpha, double* y, bool gpu_status) {
      Logger &logger = Logger::get_instance();
      logger.func_in(monolish_func);

      if (gpu_status == true) {
#if MONOLISH_USE_GPU
#pragma omp target teams distribute parallel for
        for (size_t i = 0; i < N; i++) {
          y[i] = a[i] * alpha;
        }
#else
        throw std::runtime_error(
            "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
      } else {
#pragma omp parallel for
        for (size_t i = 0; i < N; i++) {
          y[i] = a[i] * alpha;
        }
      }
      logger.func_out();
    }

    // y[i] = a[i] / alpha
    void vdiv(const size_t N, const double* a, const double alpha, double* y, bool gpu_status) {
      Logger &logger = Logger::get_instance();
      logger.func_in(monolish_func);

      if (gpu_status == true) {
#if MONOLISH_USE_GPU
#pragma omp target teams distribute parallel for
        for (size_t i = 0; i < N; i++) {
          y[i] = a[i] / alpha;
        }
#else
        throw std::runtime_error(
            "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
      } else {
#pragma omp parallel for
        for (size_t i = 0; i < N; i++) {
          y[i] = a[i] / alpha;
        }
      }
      logger.func_out();
    }

///////////////////////
//   vector-vector   //
///////////////////////
    // y[i] = a[i] + b[i]
    void vadd(const size_t N, const double* a, const double* b, double* y, bool gpu_status) {
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
        vdAdd(N, a, b, y);
#else
#pragma omp parallel for
        for (size_t i = 0; i < N; i++) {
          y[i] = a[i] + b[i];
        }
#endif
      }
      logger.func_out();
    }

    // y[i] = a[i] - b[i]
    void vsub(const size_t N, const double* a, const double* b, double* y, bool gpu_status) {
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
        vdSub(N, a, b, y);
#else
#pragma omp parallel for
        for (size_t i = 0; i < N; i++) {
          y[i] = a[i] - b[i];
        }
#endif
      }
      logger.func_out();
    }

    // y[i] = a[i] * b[i]
    void vmul(const size_t N, const double* a, const double* b, double* y, bool gpu_status) {
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
        vdMul(N, a, b, y);
#else
#pragma omp parallel for
        for (size_t i = 0; i < N; i++) {
          y[i] = a[i] * b[i];
        }
#endif
      }
      logger.func_out();
    }

    // y[i] = a[i] / b[i]
    void vdiv(const size_t N, const double* a, const double* b, double* y, bool gpu_status) {
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
