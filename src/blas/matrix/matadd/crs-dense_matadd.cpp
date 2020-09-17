#include "../../../../include/monolish_blas.hpp"
#include "../../../monolish_internal.hpp"

#ifdef USE_GPU
#include <cublas_v2.h>
#else
#include <cblas.h>
#endif

namespace monolish {

// double ///////////////////
void blas::matadd(const matrix::CRS<double> &A, const matrix::CRS<double> &B,
                  matrix::CRS<double> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != B.get_row() && A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != B.row != C.row");
  }
  if (A.get_col() != B.get_col() && A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.col != B.col != C.col");
  }

  const double *Ad = A.val.data();
  const double *Bd = B.val.data();
  double *Cd = C.val.data();

  // MN = MK * KN
  const size_t nnz = A.get_nnz();

#if USE_GPU
  cublasHandle_t h;
  check(cublasCreate(&h));
#pragma acc data present(Ad [0:nnz], Bd [0:nnz], Cd [0:nnz])
#pragma acc parallel
  {
#pragma acc loop independent
    for (size_t i = 0; i < nnz; i++) {
      Cd[i] = Ad[i] + Bd[i];
    }
  }
#else
#pragma omp parallel for
  for (size_t i = 0; i < nnz; i++) {
    Cd[i] = Ad[i] + Bd[i];
  }
#endif
  logger.func_out();
}

// float ///////////////////
void blas::matadd(const matrix::CRS<float> &A, const matrix::CRS<float> &B,
                  matrix::CRS<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != B.get_row() && A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != B.row != C.row");
  }
  if (A.get_col() != B.get_col() && A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.col != B.col != C.col");
  }

  const float *Ad = A.val.data();
  const float *Bd = B.val.data();
  float *Cd = C.val.data();

  // MN = MK * KN
  const size_t nnz = A.get_nnz();

#if USE_GPU
  cublasHandle_t h;
  check(cublasCreate(&h));
#pragma acc data present(Ad [0:nnz], Bd [0:nnz], Cd [0:nnz])
#pragma acc parallel
  {
#pragma acc loop independent
    for (size_t i = 0; i < nnz; i++) {
      Cd[i] = Ad[i] + Bd[i];
    }
  }
#else
#pragma omp parallel for
  for (size_t i = 0; i < nnz; i++) {
    Cd[i] = Ad[i] + Bd[i];
  }
#endif

  logger.func_out();
}

template <typename T>
matrix::CRS<T> matrix::CRS<T>::operator+(const matrix::CRS<T> &B) {
  matrix::CRS<T> C(*this);
  C.send();

  blas::matadd(*this, B, C);

  return C;
}
template matrix::CRS<double>
matrix::CRS<double>::operator+(const matrix::CRS<double> &B);
template matrix::CRS<float>
matrix::CRS<float>::operator+(const matrix::CRS<float> &B);
} // namespace monolish
