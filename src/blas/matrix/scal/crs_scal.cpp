#include <iostream>
#include <typeinfo>

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "../../../../include/monolish_blas.hpp"

namespace monolish {

void blas::mscal(const double alpha, matrix::CRS<double> &A) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  size_t nnz = A.get_nnz();

  double *vald = A.val.data();

#if USE_GPU // gpu
#pragma acc data present(vald [0:nnz])
#pragma acc parallel
#pragma acc loop independent
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

void blas::mscal(const float alpha, matrix::CRS<float> &A) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  size_t nnz = A.get_nnz();

  float *vald = A.val.data();

#if USE_GPU // gpu
#pragma acc parallel
#pragma acc loop independent
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

template <typename T> matrix::CRS<T> matrix::CRS<T>::operator*(const T value) {
  matrix::CRS<T> A = copy();

  blas::mscal(value, A);

  return A;
}
template matrix::CRS<double> matrix::CRS<double>::operator*(const double value);
template matrix::CRS<float> matrix::CRS<float>::operator*(const float value);
} // namespace monolish
