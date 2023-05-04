#pragma once

namespace monolish {
namespace {

template <typename F1, typename F2, typename F3>
void Daxpy_core(const F1 alpha, const F2 &x, F3 &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(x, y));
  assert(util::is_same_device_mem_stat(x, y));

  const double *xd = x.begin();
  double *yd = y.begin();
  auto size = x.size();

  if (x.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU
    cublasHandle_t h;
    internal::check_CUDA(cublasCreate(&h));
#pragma omp target data use_device_ptr(xd, yd)
    { internal::check_CUDA(cublasDaxpy(h, size, &alpha, xd, 1, yd, 1)); }
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

template <typename F1, typename F2, typename F3>
void Saxpy_core(const F1 alpha, const F2 &x, F3 &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(x, y));
  assert(util::is_same_device_mem_stat(x, y));

  const float *xd = x.begin();
  float *yd = y.begin();
  auto size = x.size();

  if (x.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU
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

} // namespace

} // namespace monolish
