#include "../../../../include/monolish_blas.hpp"
#include "../../../monolish_internal.hpp"

#ifdef USE_GPU
#include <cublas_v2.h>
#else
#include <cblas.h>
#endif

//#ifdef USE_AVX
#include <immintrin.h>
#define SIMD_FUNC(NAME) _mm256_##NAME
using Sreg = __m256;
using Dreg = __m256d;
//#endif

namespace monolish {

// double ///////////////////
void blas::matmul(const matrix::CRS<double> &A, const matrix::Dense<double> &B,
                  matrix::Dense<double> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_col() != B.get_row()) {
    std::cout << "A.col: " << A.get_col() << std::flush;
    std::cout << ", " << std::flush;
    std::cout << "B.row: " << B.get_row() << std::endl;
    throw std::runtime_error("error A.col != B.row");
  }

  if (A.get_row() != C.get_row()) {
    std::cout << "A.row: " << A.get_row() << std::flush;
    std::cout << ", " << std::flush;
    std::cout << "C.row: " << C.get_row() << std::endl;
    throw std::runtime_error("error A.row != B.row");
  }

  if (B.get_col() != C.get_col()) {
    std::cout << "B.col: " << B.get_col() << std::flush;
    std::cout << ", " << std::flush;
    std::cout << "C.col: " << C.get_col() << std::endl;
    throw std::runtime_error("error B.col != C.col");
  }

  const double *vald = A.val.data();
  const int *rowd = A.row_ptr.data();
  const int *cold = A.col_ind.data();

  const double *Bd = B.val.data();
  double *Cd = C.val.data();

  // MN = MK * KN
  const size_t M = A.get_row();
  const size_t N = B.get_col();
  const size_t K = A.get_col();
  const size_t nnz = A.get_nnz();

#if USE_GPU
#pragma acc data present(vald [0:nnz], rowd [0:M + 1], cold [0:nnz],           \
                         Bd [0:K * N], Cd [0:M * N])
#pragma acc parallel wait
  {
#pragma acc loop independent
    for (size_t i = 0; i < M * N; i++) {
      Cd[i] = 0.0;
    }
  }

#pragma acc data present(vald [0:nnz], rowd [0:M + 1], cold [0:nnz],           \
                         Bd [0:K * N], Cd [0:M * N])
#pragma acc parallel wait
  {
#pragma acc loop independent
    for (size_t j = 0; j < N; j++) {
      for (size_t i = 0; i < M; i++) {
        for (size_t k = (size_t)rowd[i]; k < (size_t)rowd[i + 1]; k++) {
          Cd[i * N + j] += vald[k] * Bd[N * cold[k] + j];
        }
      }
    }
  }
#else
#if USE_AVX //avx_cpu
  const int vecL = 4;

#pragma omp parallel for
  for (int i = 0; i < (int)(M * N); i++) {
    Cd[i] = 0.0;
  }

#pragma omp parallel for
  for (int i = 0; i < (int)M; i++) {
      int start = (int)rowd[i];
      int end = (int)rowd[i + 1];
      const int Cr = i * N;
      for (int k = start; k < end; k++) {
          const int Br = N * cold[k];
          const Dreg Av = SIMD_FUNC(broadcast_sd)(&vald[k]);
          Dreg tv, Bv, Cv;
          int j;
          for (j = 0; j < (int)N - (vecL-1); j += vecL) {
              const int BB = Br + j;
              const int CC = Cr + j;

              Bv = SIMD_FUNC(loadu_pd)((double*)&Bd[BB]);
              Cv = SIMD_FUNC(loadu_pd)((double*)&Cd[CC]);
              tv = SIMD_FUNC(mul_pd)(Av, Bv);
              Cv = SIMD_FUNC(add_pd)(Cv, tv);
              SIMD_FUNC(storeu_pd)((double*)&Cd[CC], Cv);
          }

          for ( ; j < (int)N; j++) {
              Cd[Cr + j] += vald[k] * Bd[Br + j];
          }
      }
  }
#else //Scalar_cpu
#pragma omp parallel for
  for (int j = 0; j < (int)N; j++) {
      for (int i = 0; i < (int)M; i++) {
          double tmp = 0;
          int start = (int)rowd[i];
          int end = (int)rowd[i+1];
          for (int k = start; k < end; k++) {
              tmp += vald[k] * Bd[N * cold[k] + j];
          }
          Cd[i*N+j] = tmp;
      }
  }
#endif
#endif
  logger.func_out();
}

// float ///////////////////
void blas::matmul(const matrix::CRS<float> &A, const matrix::Dense<float> &B,
                  matrix::Dense<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_col() != B.get_row()) {
    std::cout << "A.col: " << A.get_col() << std::flush;
    std::cout << ", " << std::flush;
    std::cout << "B.row: " << B.get_row() << std::endl;
    throw std::runtime_error("error A.col != B.row");
  }

  if (A.get_row() != C.get_row()) {
    std::cout << "A.row: " << A.get_row() << std::flush;
    std::cout << ", " << std::flush;
    std::cout << "C.row: " << C.get_row() << std::endl;
    throw std::runtime_error("error A.row != B.row");
  }

  if (B.get_col() != C.get_col()) {
    std::cout << "B.col: " << B.get_col() << std::flush;
    std::cout << ", " << std::flush;
    std::cout << "C.col: " << C.get_col() << std::endl;
    throw std::runtime_error("error B.col != C.col");
  }

  const float *vald = A.val.data();
  const int *rowd = A.row_ptr.data();
  const int *cold = A.col_ind.data();

  const float *Bd = B.val.data();
  float *Cd = C.val.data();

  // MN = MK * KN
  const size_t M = A.get_row();
  const size_t N = B.get_col();
  const size_t K = A.get_col();
  const size_t nnz = A.get_nnz();

#if USE_GPU
#pragma acc data present(vald [0:nnz], rowd [0:M + 1], cold [0:nnz],           \
                         Bd [0:K * N], Cd [0:M * N])
#pragma acc parallel wait
  {
#pragma acc loop independent
    for (size_t i = 0; i < M * N; i++) {
      Cd[i] = 0.0;
    }
  }

#pragma acc data present(vald [0:nnz], rowd [0:M + 1], cold [0:nnz],           \
                         Bd [0:K * N], Cd [0:M * N])
#pragma acc parallel wait
  {
#pragma acc loop independent
    for (size_t j = 0; j < N; j++) {
      for (size_t i = 0; i < M; i++) {
        for (size_t k = (size_t)rowd[i]; k < (size_t)rowd[i + 1]; k++) {
          Cd[i * N + j] += vald[k] * Bd[N * cold[k] + j];
        }
      }
    }
  }
#else
#if USE_AVX //avx_cpu
  const int vecL = 8;

#pragma omp parallel for
  for (int i = 0; i < (int)(M * N); i++) {
    Cd[i] = 0.0;
  }

#pragma omp parallel for
  for (int i = 0; i < (int)M; i++) {
    int start = (int)rowd[i];
    int end = (int)rowd[i + 1];
    const int Cr = i * N;
    for (int k = start; k < end; k++) {
      const int Br = N * cold[k];
      const Sreg Av = SIMD_FUNC(broadcast_ss)(&vald[k]);
      Sreg tv, Bv, Cv;
      int j;
      for (j = 0; j < (int)N - 31; j += 32) {
        const int BB = Br + j;
        const int CC = Cr + j;

        Bv = SIMD_FUNC(loadu_ps)((float*)&Bd[BB]);
        Cv = SIMD_FUNC(loadu_ps)((float*)&Cd[CC]);
        tv = SIMD_FUNC(mul_ps)(Av, Bv);
        Cv = SIMD_FUNC(add_ps)(Cv, tv);
        SIMD_FUNC(storeu_ps)((float*)&Cd[CC], Cv);

        Bv = SIMD_FUNC(loadu_ps)((float*)&Bd[BB + 8]);
        Cv = SIMD_FUNC(loadu_ps)((float*)&Cd[CC + 8]);
        tv = SIMD_FUNC(mul_ps)(Av, Bv);
        Cv = SIMD_FUNC(add_ps)(Cv, tv);
        SIMD_FUNC(storeu_ps)((float*)&Cd[CC + 8], Cv);

        Bv = SIMD_FUNC(loadu_ps)((float*)&Bd[BB + 16]);
        Cv = SIMD_FUNC(loadu_ps)((float*)&Cd[CC + 16]);
        tv = SIMD_FUNC(mul_ps)(Av, Bv);
        Cv = SIMD_FUNC(add_ps)(Cv, tv);
        SIMD_FUNC(storeu_ps)((float*)&Cd[Cr + j + 16], Cv);

        Bv = SIMD_FUNC(loadu_ps)((float*)&Bd[BB + 24]);
        Cv = SIMD_FUNC(loadu_ps)((float*)&Cd[CC + 24]);
        tv = SIMD_FUNC(mul_ps)(Av, Bv);
        Cv = SIMD_FUNC(add_ps)(Cv, tv);
        SIMD_FUNC(storeu_ps)((float*)&Cd[CC + 24], Cv);
      }
      for ( ; j < (int)N - 7; j += 8) {
        const int BB = Br + j;
        const int CC = Cr + j;

        Bv = SIMD_FUNC(loadu_ps)((float*)&Bd[BB]);
        Cv = SIMD_FUNC(loadu_ps)((float*)&Cd[CC]);
        tv = SIMD_FUNC(mul_ps)(Av, Bv);
        Cv = SIMD_FUNC(add_ps)(Cv, tv);
        SIMD_FUNC(storeu_ps)((float*)&Cd[CC], Cv);
      }
      for ( ; j < (int)N; j++) {
        Cd[Cr + j] += vald[k] * Bd[Br + j];
      }
    }
  }
#else //Scalar_cpu
#pragma omp parallel for
  for (int j = 0; j < (int)N; j++) {
      for (int i = 0; i < (int)M; i++) {
          double tmp = 0;
          int start = (int)rowd[i];
          int end = (int)rowd[i+1];
          for (int k = start; k < end; k++) {
              tmp += vald[k] * Bd[N * cold[k] + j];
          }
          Cd[i*N+j] = tmp;
      }
  }
#endif
#endif
  logger.func_out();
}

template <typename T>
matrix::Dense<T> matrix::CRS<T>::operator*(const matrix::Dense<T> &B) {
  matrix::Dense<T> C(get_row(), B.get_col());
  C.send();

  blas::matmul(*this, B, C);

  return C;
}
template matrix::Dense<double>
matrix::CRS<double>::operator*(const matrix::Dense<double> &B);
template matrix::Dense<float>
matrix::CRS<float>::operator*(const matrix::Dense<float> &B);
} // namespace monolish
