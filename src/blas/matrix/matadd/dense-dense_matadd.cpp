#include "../../../../include/monolish_blas.hpp"
#include "../../../monolish_internal.hpp"

#ifdef MONOLISH_USE_GPU
#include <cublas_v2.h>
#else
#include <cblas.h>
#endif

namespace monolish {

// double ///////////////////
void blas::matadd(const matrix::Dense<double> &A,
                  const matrix::Dense<double> &B, matrix::Dense<double> &C) {
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

#if MONOLISH_USE_GPU
#pragma omp target teams distribute parallel for
  for (size_t i = 0; i < nnz; i++) {
    Cd[i] = Ad[i] + Bd[i];
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
void blas::matadd(const matrix::Dense<float> &A, const matrix::Dense<float> &B,
                  matrix::Dense<float> &C) {
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

#if MONOLISH_USE_GPU
#pragma omp target teams distribute parallel for
  for (size_t i = 0; i < nnz; i++) {
    Cd[i] = Ad[i] + Bd[i];
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
matrix::Dense<T> matrix::Dense<T>::operator+(const matrix::Dense<T> &B) {
  matrix::Dense<T> C(get_row(), get_col());
  C.send();

  blas::matadd(*this, B, C);

  return C;
}
template matrix::Dense<double>
matrix::Dense<double>::operator+(const matrix::Dense<double> &B);
template matrix::Dense<float>
matrix::Dense<float>::operator+(const matrix::Dense<float> &B);
} // namespace monolish
