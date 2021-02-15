#pragma once

namespace monolish {

namespace {
template <typename F1> double Dnrm2_core(const F1 &x) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  double ans = 0;
  const double *xd = x.data();
  size_t size = x.size();
  const size_t xoffset = x.get_offset();

  if (x.get_device_mem_stat() == true) {
#if MONOLISH_USE_GPU
    cublasHandle_t h;
    internal::check_CUDA(cublasCreate(&h));
#pragma omp target data use_device_ptr(xd)
    { internal::check_CUDA(cublasDnrm2(h, size, xd + xoffset, 1, &ans)); }
    cublasDestroy(h);
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
    ans = cblas_dnrm2(size, xd + xoffset, 1);
  }
  logger.func_out();
  return ans;
}

template <typename F1> float Snrm2_core(const F1 &x) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  float ans = 0;
  const float *xd = x.data();
  size_t size = x.size();
  const size_t xoffset = x.get_offset();

  if (x.get_device_mem_stat() == true) {
#if MONOLISH_USE_GPU
    cublasHandle_t h;
    internal::check_CUDA(cublasCreate(&h));
#pragma omp target data use_device_ptr(xd)
    { internal::check_CUDA(cublasSnrm2(h, size, xd + xoffset, 1, &ans)); }
    cublasDestroy(h);
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
    ans = cblas_snrm2(size, xd + xoffset, 1);
  }
  logger.func_out();
  return ans;
}
}

} // namespace monolish
