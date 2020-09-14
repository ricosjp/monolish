#include "../../../include/monolish_blas.hpp"
#include "../../monolish_internal.hpp"

#ifdef USE_GPU
#include <cublas_v2.h>
#else
#include <cblas.h>
#endif

namespace monolish {

// double ///////////////////
void blas::scal(const double alpha, vector<double> &x) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  double *xd = x.data();
  size_t size = x.size();

#if USE_GPU
  cublasHandle_t h;
  check(cublasCreate(&h));

#pragma acc host_data use_device(xd)
  { check(cublasDscal(h, size, &alpha, xd, 1)); }
#else
  cblas_dscal(size, alpha, xd, 1);
#endif
  logger.func_out();
}

// float ///////////////////
void blas::scal(const float alpha, vector<float> &x) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  float *xd = x.data();
  size_t size = x.size();

#if USE_GPU
  cublasHandle_t h;
  check(cublasCreate(&h));
#pragma acc host_data use_device(xd)
  { check(cublasSscal(h, size, &alpha, xd, 1)); }
#else
  cblas_sscal(size, alpha, xd, 1);
#endif
  logger.func_out();
}
} // namespace monolish
