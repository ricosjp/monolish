#include "../../../include/monolish_blas.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
// asum ///////////////////
float blas::asum(const vector<float> &x) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  float ans = 0;
  const float *xd = x.data();
  size_t size = x.size();

  if (x.get_device_mem_stat() == true) {
#if MONOLISH_USE_GPU
    cublasHandle_t h;
    internal::check_CUDA(cublasCreate(&h));
#pragma omp target data use_device_ptr(xd)
    { internal::check_CUDA(cublasSasum(h, size, xd, 1, &ans)); }
    cublasDestroy(h);
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
    ans = cblas_sasum(size, xd, 1);
  }
  logger.func_out();
  return ans;
}
void blas::asum(const vector<float> &x, float &ans) { ans = asum(x); }

// axpy ///////////////////
void blas::axpy(const float alpha, const vector<float> &x,
                vector<float> &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (x.size() != y.size()) {
    throw std::runtime_error("error vector size is not same");
  }
  if (x.get_device_mem_stat() != y.get_device_mem_stat()) {
    throw std::runtime_error("error vector get_device_mem_stat() is not same");
  }

  const float *xd = x.data();
  float *yd = y.data();
  size_t size = x.size();

  if (x.get_device_mem_stat() == true) {
#if MONOLISH_USE_GPU
    cublasHandle_t h;
    internal::check_CUDA(cublasCreate(&h));
#pragma omp target data use_device_ptr(xd, yd)
    { internal::check_CUDA(cublasSaxpy(h, size, &alpha, xd, 1, yd, 1)); }
    cublasDestroy(h);
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
    cblas_saxpy(size, alpha, xd, 1, yd, 1);
  }
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
  if (x.get_device_mem_stat() != y.get_device_mem_stat()) {
    throw std::runtime_error("error vector get_device_mem_stat() is not same");
  }

  float ans = 0;
  const float *xd = x.data();
  const float *yd = y.data();
  const size_t size = x.size();

  if (x.get_device_mem_stat() == true) {
#if MONOLISH_USE_GPU
    cublasHandle_t h;
    internal::check_CUDA(cublasCreate(&h));
#pragma omp target data use_device_ptr(xd, yd)
    { internal::check_CUDA(cublasSdot(h, size, xd, 1, yd, 1, &ans)); }
    cublasDestroy(h);
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
    ans = cblas_sdot(size, xd, 1, yd, 1);
  }
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

  if (x.get_device_mem_stat() == true) {
#if MONOLISH_USE_GPU
    cublasHandle_t h;
    internal::check_CUDA(cublasCreate(&h));
#pragma omp target data use_device_ptr(xd)
    { internal::check_CUDA(cublasSnrm2(h, size, xd, 1, &ans)); }
    cublasDestroy(h);
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
    ans = cblas_snrm2(size, xd, 1);
  }
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

  if (x.get_device_mem_stat() == true) {
#if MONOLISH_USE_GPU
    cublasHandle_t h;
    internal::check_CUDA(cublasCreate(&h));
#pragma omp target data use_device_ptr(xd)
    { internal::check_CUDA(cublasSscal(h, size, &alpha, xd, 1)); }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
    cblas_sscal(size, alpha, xd, 1);
  }
  logger.func_out();
}

// nrm1 ///////////////////
float blas::nrm1(const vector<float> &x) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  float ans = 0;
  const float *xd = x.data();
  size_t size = x.size();

  if (x.get_device_mem_stat() == true) {
#if MONOLISH_USE_GPU
#pragma omp target teams distribute parallel for reduction(+ : ans) map (tofrom: ans)
    for (size_t i = 0; i < size; i++) {
      ans += std::abs(xd[i]);
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
#pragma omp parallel for reduction(+ : ans)
    for (size_t i = 0; i < size; i++) {
      ans += std::abs(xd[i]);
    }
  }

  logger.func_out();
  return ans;
}
void blas::nrm1(const vector<float> &x, float &ans) { ans = nrm1(x); }

// axpyz ///////////////////
void blas::axpyz(const float alpha, const vector<float> &x,
                 const vector<float> &y, vector<float> &z) {
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

  const float *xd = x.data();
  const float *yd = y.data();
  float *zd = z.data();
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
void blas::xpay(const float alpha, const vector<float> &x,
                vector<float> &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (x.size() != y.size()) {
    throw std::runtime_error("error vector size is not same");
  }
  if (x.get_device_mem_stat() != y.get_device_mem_stat()) {
    throw std::runtime_error("error vector get_device_mem_stat() is not same");
  }

  const float *xd = x.data();
  float *yd = y.data();
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
float blas::sum(const vector<float> &x) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  float ans = 0;
  const float *xd = x.data();
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
