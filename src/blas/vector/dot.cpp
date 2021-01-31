#include "../../../include/monolish_blas.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {

namespace {
template <typename F1, typename F2> double Ddot_core(const F1 &x, const F2 &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(x, y));
  assert(util::is_same_device_mem_stat(x, y));

  double ans = 0;
  const double *xd = x.data();
  const double *yd = y.data();
  const size_t size = x.size();

  if (x.get_device_mem_stat() == true) {
#if MONOLISH_USE_GPU
    cublasHandle_t h;
    internal::check_CUDA(cublasCreate(&h));
#pragma omp target data use_device_ptr(xd, yd)
    { internal::check_CUDA(cublasDdot(h, size, xd, 1, yd, 1, &ans)); }
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

template <typename F1, typename F2> float Sdot_core(const F1 &x, const F2 &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(x, y));
  assert(util::is_same_device_mem_stat(x, y));

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

} // namespace

namespace blas {

double dot(const vector<double> &x, const vector<double> &y) {
  return Ddot_core(x, y);
}
void dot(const vector<double> &x, const vector<double> &y, double &ans) {
  ans = dot(x, y);
}

float dot(const vector<float> &x, const vector<float> &y) {
  return Sdot_core(x, y);
}
void dot(const vector<float> &x, const vector<float> &y, float &ans) {
  ans = dot(x, y);
}

} // namespace blas

} // namespace monolish
