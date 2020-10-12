#include <iostream>
#include <typeinfo>

#include "../../../include/monolish_blas.hpp"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef MONOLISH_USE_GPU
#include <cublas.h>
#else
#include <cblas.h>
#endif

namespace monolish {

// double ///////////////////
void blas::xpay(const double alpha, const vector<double> &x,
                vector<double> &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (x.size() != y.size()) {
    throw std::runtime_error("error vector size is not same");
  }

  const double *xd = x.data();
  double *yd = y.data();
  size_t size = x.size();

#if MONOLISH_USE_GPU
#pragma omp target teams distribute parallel for
  for (size_t i = 0; i < size; i++) {
    yd[i] = xd[i] + alpha * yd[i];
  }
#else
#pragma omp parallel for
  for (size_t i = 0; i < size; i++) {
    yd[i] = xd[i] + alpha * yd[i];
  }
#endif
  logger.func_out();
}

// float ///////////////////
void blas::xpay(const float alpha, const vector<float> &x, vector<float> &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (x.size() != y.size()) {
    throw std::runtime_error("error vector size is not same");
  }

  const float *xd = x.data();
  float *yd = y.data();
  size_t size = x.size();

#if MONOLISH_USE_GPU
#pragma omp target teams distribute parallel for
  for (size_t i = 0; i < size; i++) {
    yd[i] = xd[i] + alpha * yd[i];
  }
#else
#pragma omp parallel for
  for (size_t i = 0; i < size; i++) {
    yd[i] = xd[i] + alpha * yd[i];
  }
#endif
  logger.func_out();
}
} // namespace monolish
