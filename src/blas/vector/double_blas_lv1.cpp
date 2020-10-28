#include "../../../include/monolish_blas.hpp"
#include "../../monolish_internal.hpp"
#include <cblas.h>

#ifdef MONOLISH_USE_GPU
#include <cublas_v2.h>
#endif

namespace monolish {
// asum ///////////////////
double blas::asum(const vector<double> &x) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  double ans = 0;
  const double *xd = x.data();
  size_t size = x.size();

  if (x.get_device_mem_stat() == true) {
#if MONOLISH_USE_GPU
    cublasHandle_t h;
    check(cublasCreate(&h));
#pragma omp target data use_device_ptr(xd)
    { check(cublasDasum(h, size, xd, 1, &ans)); }
    cublasDestroy(h);
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
    ans = cblas_dasum(size, xd, 1);
  }
  logger.func_out();
  return ans;
}
void blas::asum(const vector<double> &x, double &ans) { ans = asum(x); }

// axpy ///////////////////
void blas::axpy(const double alpha, const vector<double> &x,
                vector<double> &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (x.size() != y.size()) {
    throw std::runtime_error("error vector size is not same");
  }
  if (x.get_device_mem_stat() != y.get_device_mem_stat()) {
    throw std::runtime_error("error vector get_device_mem_stat() is not same");
  }

  const double *xd = x.data();
  double *yd = y.data();
  size_t size = x.size();

  if (x.get_device_mem_stat() == true) {
#if MONOLISH_USE_GPU
    cublasHandle_t h;
    check(cublasCreate(&h));
#pragma omp target data use_device_ptr(xd, yd)
    { check(cublasDaxpy(h, size, &alpha, xd, 1, yd, 1)); }
    cublasDestroy(h);
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
    cblas_daxpy(size, alpha, xd, 1, yd, 1);
  }
  logger.func_out();
}

// dot ///////////////////
double blas::dot(const vector<double> &x, const vector<double> &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (x.size() != y.size()) {
    throw std::runtime_error("error vector size is not same");
  }
  if (x.get_device_mem_stat() != y.get_device_mem_stat()) {
    throw std::runtime_error("error vector get_device_mem_stat() is not same");
  }

  double ans = 0;
  const double *xd = x.data();
  const double *yd = y.data();
  const size_t size = x.size();

  if (x.get_device_mem_stat() == true) {
#if MONOLISH_USE_GPU
    cublasHandle_t h;
    check(cublasCreate(&h));
#pragma omp target data use_device_ptr(xd, yd)
    { check(cublasDdot(h, size, xd, 1, yd, 1, &ans)); }
    cublasDestroy(h);
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
    ans = cblas_ddot(size, xd, 1, yd, 1);
  }
  logger.func_out();
  return ans;
}
void blas::dot(const vector<double> &x, const vector<double> &y, double &ans) {
  ans = dot(x, y);
}

// nrm2 ///////////////////
double blas::nrm2(const vector<double> &x) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  double ans = 0;
  const double *xd = x.data();
  size_t size = x.size();

  if (x.get_device_mem_stat() == true) {
#if MONOLISH_USE_GPU
    cublasHandle_t h;
    check(cublasCreate(&h));
#pragma omp target data use_device_ptr(xd)
    { check(cublasDnrm2(h, size, xd, 1, &ans)); }
    cublasDestroy(h);
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
    ans = cblas_dnrm2(size, xd, 1);
  }
  logger.func_out();
  return ans;
}
void blas::nrm2(const vector<double> &x, double &ans) { ans = nrm2(x); }

// scal ///////////////////
void blas::scal(const double alpha, vector<double> &x) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  double *xd = x.data();
  size_t size = x.size();

  if (x.get_device_mem_stat() == true) {
#if MONOLISH_USE_GPU
    cublasHandle_t h;
    check(cublasCreate(&h));
#pragma omp target data use_device_ptr(xd)
    { check(cublasDscal(h, size, &alpha, xd, 1)); }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
    cblas_dscal(size, alpha, xd, 1);
  }
  logger.func_out();
}

// axpyz ///////////////////
void blas::axpyz(const double alpha, const vector<double> &x,
                 const vector<double> &y, vector<double> &z) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (x.size() != y.size() || x.size() != z.size()) {
    throw std::runtime_error("error vector size is not same");
  }
  if (x.get_device_mem_stat() != y.get_device_mem_stat() ||
      x.get_device_mem_stat() != z.get_device_mem_stat()) {
    throw std::runtime_error("error vector get_device_mem_stat() is not same");
  }

  const double *xd = x.data();
  const double *yd = y.data();
  double *zd = z.data();
  size_t size = x.size();

  if (x.get_device_mem_stat() == true) {
#if MONOLISH_USE_GPU
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < size; i++) {
      zd[i] = alpha * xd[i] + yd[i];
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
#pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
      zd[i] = alpha * xd[i] + yd[i];
    }
  }
  logger.func_out();
}

// xpay ///////////////////
void blas::xpay(const double alpha, const vector<double> &x,
                vector<double> &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (x.size() != y.size()) {
    throw std::runtime_error("error vector size is not same");
  }
  if (x.get_device_mem_stat() != y.get_device_mem_stat()) {
    throw std::runtime_error("error vector get_device_mem_stat() is not same");
  }

  const double *xd = x.data();
  double *yd = y.data();
  size_t size = x.size();

  if (x.get_device_mem_stat() == true) {
#if MONOLISH_USE_GPU
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < size; i++) {
      yd[i] = xd[i] + alpha * yd[i];
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
#pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
      yd[i] = xd[i] + alpha * yd[i];
    }
  }
  logger.func_out();
}

// sum ///////////////////
double blas::sum(const vector<double> &x) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  double ans = 0;
  const double *xd = x.data();
  size_t size = x.size();

  if (x.get_device_mem_stat() == true) {
#if MONOLISH_USE_GPU
#pragma omp target teams distribute parallel for reduction(+ : ans) map (tofrom: ans)
    for (size_t i = 0; i < size; i++) {
      ans += xd[i];
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
#pragma omp parallel for reduction(+ : ans)
    for (size_t i = 0; i < size; i++) {
      ans += xd[i];
    }
  }

  logger.func_out();
  return ans;
}

} // namespace monolish
