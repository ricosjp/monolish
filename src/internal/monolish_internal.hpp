#include <iostream>
#include <omp.h>

#ifdef MONOLISH_USE_GPU
#include <cuda_runtime.h>
#endif
// internal math
namespace monolish {
  namespace internal {

#ifdef MONOLISH_USE_GPU
    auto checkError = [](auto result, auto func, auto file, auto line) {
      if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result),
            cudaGetErrorName((cudaError_t)result), func);
        // cudaDeviceReset();
        exit(EXIT_FAILURE);
      }
    };
#define check(val) checkError((val), #val, __FILE__, __LINE__)
#endif

    void vadd(const size_t N, const double* a, const double* b, double* y, bool gpu_status);
    void vsub(const size_t N, const double* a, const double* b, double* y, bool gpu_status);
    void vmul(const size_t N, const double* a, const double* b, double* y, bool gpu_status);
    void vdiv(const size_t N, const double* a, const double* b, double* y, bool gpu_status);

    void vadd(const size_t N, const float* a, const float* b, float* y, bool gpu_status);
    void vsub(const size_t N, const float* a, const float* b, float* y, bool gpu_status);
    void vmul(const size_t N, const float* a, const float* b, float* y, bool gpu_status);
    void vdiv(const size_t N, const float* a, const float* b, float* y, bool gpu_status);
  }
} // namespace monolish
