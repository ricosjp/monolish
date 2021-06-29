#pragma once

#include "../../include/monolish_mpi.hpp"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <omp.h>
#include <stdexcept>
#include <typeinfo>

#ifdef MONOLISH_USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#endif

#ifdef MONOLISH_USE_NVIDIA_GPU
#include "cusparse.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#endif

#ifdef MONOLISH_USE_AVX
#include <immintrin.h>
#define SIMD_FUNC(NAME) _mm256_##NAME
using Sreg = __m256;
using Dreg = __m256d;
#endif

namespace monolish::internal {

#ifdef MONOLISH_USE_NVIDIA_GPU
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
void vreciprocal(const size_t N, const float *a, float *y, bool gpu_status);
void vreciprocal(const size_t N, const double *a, double *y, bool gpu_status);

void vcopy(const size_t N, const float *a, float *y, bool gpu_status);
void vcopy(const size_t N, const double *a, double *y, bool gpu_status);
void vcopy(const size_t N, const int *a, int *y, bool gpu_status);
void vcopy(const size_t N, const size_t *a, size_t *y, bool gpu_status);

bool vequal(const size_t N, const float *a, const float *y, bool gpu_status);
bool vequal(const size_t N, const double *a, const double *y, bool gpu_status);
bool vequal(const size_t N, const int *a, const int *y, bool gpu_status);
bool vequal(const size_t N, const size_t *a, const size_t *y, bool gpu_status);

// math
void vsqrt(const size_t N, const double *a, double *y, bool gpu_status);
void vpow(const size_t N, const double *a, const double *b, double *y,
          bool gpu_status);
void vpow(const size_t N, const double *a, const double alpha, double *y,
          bool gpu_status);
void vsin(const size_t N, const double *a, double *y, bool gpu_status);
void vsinh(const size_t N, const double *a, double *y, bool gpu_status);
void vasin(const size_t N, const double *a, double *y, bool gpu_status);
void vasinh(const size_t N, const double *a, double *y, bool gpu_status);
void vtan(const size_t N, const double *a, double *y, bool gpu_status);
void vtanh(const size_t N, const double *a, double *y, bool gpu_status);
void vatan(const size_t N, const double *a, double *y, bool gpu_status);
void vatanh(const size_t N, const double *a, double *y, bool gpu_status);
void vceil(const size_t N, const double *a, double *y, bool gpu_status);
void vfloor(const size_t N, const double *a, double *y, bool gpu_status);
void vsign(const size_t N, const double *a, double *y, bool gpu_status);
double vmax(const size_t N, const double *y, bool gpu_status);
double vmin(const size_t N, const double *y, bool gpu_status);
void vmax(const size_t N, const double *a, const double *b, double *y,
          bool gpu_status);
void vmin(const size_t N, const double *a, const double *b, double *y,
          bool gpu_status);

void vsqrt(const size_t N, const float *a, float *y, bool gpu_status);
void vpow(const size_t N, const float *a, const float *b, float *y,
          bool gpu_status);
void vpow(const size_t N, const float *a, const float alpha, float *y,
          bool gpu_status);
void vsin(const size_t N, const float *a, float *y, bool gpu_status);
void vsinh(const size_t N, const float *a, float *y, bool gpu_status);
void vasin(const size_t N, const float *a, float *y, bool gpu_status);
void vasinh(const size_t N, const float *a, float *y, bool gpu_status);
void vtan(const size_t N, const float *a, float *y, bool gpu_status);
void vtanh(const size_t N, const float *a, float *y, bool gpu_status);
void vatan(const size_t N, const float *a, float *y, bool gpu_status);
void vatanh(const size_t N, const float *a, float *y, bool gpu_status);
void vceil(const size_t N, const float *a, float *y, bool gpu_status);
void vfloor(const size_t N, const float *a, float *y, bool gpu_status);
void vsign(const size_t N, const float *a, float *y, bool gpu_status);
float vmax(const size_t N, const float *y, bool gpu_status);
float vmin(const size_t N, const float *y, bool gpu_status);
void vmax(const size_t N, const float *a, const float *b, float *y,
          bool gpu_status);
void vmin(const size_t N, const float *a, const float *b, float *y,
          bool gpu_status);

size_t vhash(const size_t N, const int *y, const int seed_value,
             bool gpu_status);
size_t vhash(const size_t N, const size_t *y, const size_t seed_value,
             bool gpu_status);

} // namespace monolish::internal
