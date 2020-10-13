#include "../../../include/monolish_blas.hpp"
#include "../../monolish_internal.hpp"

#ifdef MONOLISH_USE_GPU
#include <cublas_v2.h>
#else
#include <cblas.h>
#endif

namespace monolish {
// asum ///////////////////
float blas::asum(const vector<float> &x) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  float ans = 0;
  const float *xd = x.data();
  size_t size = x.size();

#if MONOLISH_USE_GPU
  cublasHandle_t h;
  check(cublasCreate(&h));
#pragma omp target data use_device_ptr(xd)
  { check(cublasSasum(h, size, xd, 1, &ans)); }
  cublasDestroy(h);
#else
  ans = cblas_sasum(size, xd, 1);
#endif
  logger.func_out();
  return ans;
}
void blas::asum(const vector<float> &x, float &ans) { ans = asum(x); }

// axpy ///////////////////
void blas::axpy(const float alpha, const vector<float> &x, vector<float> &y) {
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
  cublasHandle_t h;
  check(cublasCreate(&h));
#pragma omp target data use_device_ptr(xd, yd)
  { check(cublasSaxpy(h, size, &alpha, xd, 1, yd, 1)); }
  cublasDestroy(h);
#else
  cblas_saxpy(size, alpha, xd, 1, yd, 1);
#endif
  logger.func_out();
}

// dot ///////////////////
float blas::dot(const vector<float> &x, const vector<float> &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (x.size() != y.size()) {
    throw std::runtime_error("error vector size is not same");
  }

  float ans = 0;
  const float *xd = x.data();
  const float *yd = y.data();
  const size_t size = x.size();

#if MONOLISH_USE_GPU
  cublasHandle_t h;
  check(cublasCreate(&h));
#pragma omp target data use_device_ptr(xd, yd)
  { check(cublasSdot(h, size, xd, 1, yd, 1, &ans)); }
  cublasDestroy(h);
#else
  ans = cblas_sdot(size, xd, 1, yd, 1);
#endif
  logger.func_out();
  return ans;
}
void blas::dot(const vector<float> &x, const vector<float> &y, float &ans) {
  ans = dot(x, y);
}

// nrm2 ///////////////////
float blas::nrm2(const vector<float> &x) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  float ans = 0;
  const float *xd = x.data();
  size_t size = x.size();

#if MONOLISH_USE_GPU
  cublasHandle_t h;
  check(cublasCreate(&h));
#pragma omp target data use_device_ptr(xd)
  { check(cublasSnrm2(h, size, xd, 1, &ans)); }
  cublasDestroy(h);
#else
  ans = cblas_snrm2(size, xd, 1);
#endif
  logger.func_out();
  return ans;
}
void blas::nrm2(const vector<float> &x, float &ans) { ans = nrm2(x); }

// scal ///////////////////
void blas::scal(const float alpha, vector<float> &x) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  float *xd = x.data();
  size_t size = x.size();

#if MONOLISH_USE_GPU
  cublasHandle_t h;
  check(cublasCreate(&h));

#pragma omp target data use_device_ptr(xd)
  { check(cublasSscal(h, size, &alpha, xd, 1)); }
#else
  cblas_sscal(size, alpha, xd, 1);
#endif
  logger.func_out();
}

// axpyz ///////////////////
void blas::axpyz(const float alpha, const vector<float> &x,
                 const vector<float> &y, vector<float> &z) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (x.size() != y.size() || x.size() != z.size()) {
    throw std::runtime_error("error vector size is not same");
  }

  const float *xd = x.data();
  const float *yd = y.data();
  float *zd = z.data();
  size_t size = x.size();

#if MONOLISH_USE_GPU
#pragma omp target teams distribute parallel for
  for (size_t i = 0; i < size; i++) {
    zd[i] = alpha * xd[i] + yd[i];
  }
#else
#pragma omp parallel for
  for (size_t i = 0; i < size; i++) {
    zd[i] = alpha * xd[i] + yd[i];
  }
#endif
  logger.func_out();
}

// xpay ///////////////////
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

// sum ///////////////////
float blas::sum(const vector<float> &x) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  float ans = 0;
  const float *xd = x.data();
  size_t size = x.size();

#if MONOLISH_USE_GPU
#pragma omp target teams distribute parallel for reduction(+ : ans) map (tofrom: ans)
  for (size_t i = 0; i < size; i++) {
    ans += xd[i];
  }
#else
#pragma omp parallel for reduction(+ : ans)
  for (size_t i = 0; i < size; i++) {
    ans += xd[i];
  }
#endif

  logger.func_out();
  return ans;
}

} // namespace monolish
