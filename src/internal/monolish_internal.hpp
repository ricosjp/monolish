#include <iostream>
#include <omp.h>
#include <typeinfo>

#ifdef MONOLISH_USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#endif

#ifdef MONOLISH_USE_GPU
#include "cusparse.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif

#ifdef MONOLISH_USE_AVX
#include <immintrin.h>
#define SIMD_FUNC(NAME) _mm256_##NAME
using Sreg = __m256;
using Dreg = __m256d;
#endif

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
#define check_CUDA(val) checkError((val), #val, __FILE__, __LINE__)
#endif

// scalar-vector
void vadd(const size_t N, const double *a, const double alpha, double *y,
          bool gpu_status);
void vsub(const size_t N, const double *a, const double alpha, double *y,
          bool gpu_status);
void vmul(const size_t N, const double *a, const double alpha, double *y,
          bool gpu_status);
void vdiv(const size_t N, const double *a, const double alpha, double *y,
          bool gpu_status);

void vadd(const size_t N, const float *a, const float alpha, float *y,
          bool gpu_status);
void vsub(const size_t N, const float *a, const float alpha, float *y,
          bool gpu_status);
void vmul(const size_t N, const float *a, const float alpha, float *y,
          bool gpu_status);
void vdiv(const size_t N, const float *a, const float alpha, float *y,
          bool gpu_status);

// vector-vector
void vadd(const size_t N, const double *a, const double *b, double *y,
          bool gpu_status);
void vsub(const size_t N, const double *a, const double *b, double *y,
          bool gpu_status);
void vmul(const size_t N, const double *a, const double *b, double *y,
          bool gpu_status);
void vdiv(const size_t N, const double *a, const double *b, double *y,
          bool gpu_status);

void vadd(const size_t N, const float *a, const float *b, float *y,
          bool gpu_status);
void vsub(const size_t N, const float *a, const float *b, float *y,
          bool gpu_status);
void vmul(const size_t N, const float *a, const float *b, float *y,
          bool gpu_status);
void vdiv(const size_t N, const float *a, const float *b, float *y,
          bool gpu_status);

// utils
void vcopy(const size_t N, const double *a, double *y, bool gpu_status);
bool vequal(const size_t N, const double *a, const double *y, bool gpu_status);

void vcopy(const size_t N, const float *a, float *y, bool gpu_status);
bool vequal(const size_t N, const float *a, const float *y, bool gpu_status);

// utils integer
void vcopy(const size_t N, const int *a, int *y, bool gpu_status);
void vcopy(const size_t N, const size_t *a, size_t *y, bool gpu_status);
bool vequal(const size_t N, const int *a, const int *y, bool gpu_status);
bool vequal(const size_t N, const size_t *a, const size_t *y, bool gpu_status);
} // namespace internal
} // namespace monolish
