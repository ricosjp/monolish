#include "../../../include/monolish_blas.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {

namespace {
template <typename F1, typename F2>
void Dscal_core(const F1 alpha, F2 &x) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  double *xd = x.data();
  size_t size = x.size();

  if (x.get_device_mem_stat() == true) {
#if MONOLISH_USE_GPU
    cublasHandle_t h;
    internal::check_CUDA(cublasCreate(&h));
#pragma omp target data use_device_ptr(xd)
    { internal::check_CUDA(cublasDscal(h, size, &alpha, xd, 1)); }
    cublasDestroy(h);
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
    cblas_dscal(size, alpha, xd, 1);
  }
  logger.func_out();
}

template <typename F1, typename F2>
void Sscal_core(const F1 alpha, F2 &x) {
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
    cublasDestroy(h);
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
    cblas_sscal(size, alpha, xd, 1);
  }
  logger.func_out();
}

} // namespace

namespace blas {
void scal(const double alpha, vector<double> &x){
  Dscal_core(alpha, x);
}

void scal(const float alpha, vector<float> &x){
  Sscal_core(alpha, x);
}

} // namespace blas
} // namespace monolish
