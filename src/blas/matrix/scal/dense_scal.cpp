#include "../../../../include/monolish_blas.hpp"
#include "../../../monolish_internal.hpp"

#ifdef USE_GPU
#include <cublas_v2.h>
#else
#include <cblas.h>
#endif

namespace monolish {

// double ///////////////////
void blas::mscal(const double alpha, matrix::Dense<double> &A) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  size_t nnz = A.get_nnz();
  double *vald = A.val.data();

#if USE_GPU // gpu
#pragma omp target teams distribute parallel for
  for (size_t i = 0; i < nnz; i++) {
    vald[i] = alpha * vald[i];
  }

#else // cpu
#pragma omp parallel for
  for (size_t i = 0; i < nnz; i++)
    vald[i] = alpha * vald[i];

#endif
  logger.func_out();
}

// float ///////////////////
void blas::mscal(const float alpha, matrix::Dense<float> &A) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  size_t nnz = A.get_nnz();
  float *vald = A.val.data();

#if USE_GPU // gpu
#pragma omp target teams distribute parallel for
  for (size_t i = 0; i < nnz; i++) {
    vald[i] = alpha * vald[i];
  }

#else // cpu
#pragma omp parallel for
  for (size_t i = 0; i < nnz; i++)
    vald[i] = alpha * vald[i];

#endif
  logger.func_out();
}
template <typename T>
matrix::Dense<T> matrix::Dense<T>::operator*(const T value) {
  matrix::Dense<T> A = copy();

  blas::mscal(value, A);

  return A;
}
template matrix::Dense<double>
matrix::Dense<double>::operator*(const double value);
template matrix::Dense<float>
matrix::Dense<float>::operator*(const float value);
} // namespace monolish
