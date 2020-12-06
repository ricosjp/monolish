#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"

namespace monolish {

// double ///////////////////
void blas::matvec(const matrix::Dense<double> &A, const vector<double> &x,
                  vector<double> &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_col() != x.size()) {
    throw std::runtime_error("error A.col != x.size");
  }

  if (A.get_row() != y.size()) {
    throw std::runtime_error("error A.row != y.size");
  }

  if (A.get_device_mem_stat() != x.get_device_mem_stat() ||
      A.get_device_mem_stat() != y.get_device_mem_stat()) {
    throw std::runtime_error("error get_device_mem_stat() is not same");
  }

  const double *xd = x.data();
  double *yd = y.data();
  const double *vald = A.val.data();
  const size_t m = A.get_row();
  const size_t n = A.get_col();
  const double alpha = 1.0;
  const double beta = 0.0;

  if (A.get_device_mem_stat() == true) {
#if MONOLISH_USE_GPU
    cublasHandle_t h;
    internal::check_CUDA(cublasCreate(&h));
#pragma omp target data use_device_ptr(xd, yd, vald)
    {
      // cublas is col major
      internal::check_CUDA(cublasDgemv(h, CUBLAS_OP_T, n, m, &alpha, vald, n, xd, 1, &beta, yd,
                        1));
    }
    cublasDestroy(h);
#else
    throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
  } else {
    cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, alpha, vald, n, xd, 1, beta,
                yd, 1);
  }

  logger.func_out();
}

// float ///////////////////
void blas::matvec(const matrix::Dense<float> &A, const vector<float> &x,
                  vector<float> &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_col() != x.size()) {
    throw std::runtime_error("error A.col != x.size");
  }

  if (A.get_row() != y.size()) {
    throw std::runtime_error("error A.row != y.size");
  }

  if (A.get_device_mem_stat() != x.get_device_mem_stat() ||
      A.get_device_mem_stat() != y.get_device_mem_stat()) {
    throw std::runtime_error("error get_device_mem_stat() is not same");
  }

  const float *xd = x.data();
  float *yd = y.data();
  const float *vald = A.val.data();
  const size_t n = A.get_row();
  const size_t m = A.get_col();
  const float alpha = 1.0;
  const float beta = 0.0;

  if (A.get_device_mem_stat() == true) {
#if MONOLISH_USE_GPU
    cublasHandle_t h;
    internal::check_CUDA(cublasCreate(&h));
#pragma omp target data use_device_ptr(xd, yd, vald)
    {
      // cublas is col major
      internal::check_CUDA(cublasSgemv(h, CUBLAS_OP_T, m, n, &alpha, vald, m, xd, 1, &beta, yd,
                        1));
    }
    cublasDestroy(h);
#else
    throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
  } else {
    cblas_sgemv(CblasRowMajor, CblasNoTrans, n, m, alpha, vald, m, xd, 1, beta,
                yd, 1);
  }

  logger.func_out();
}

template <typename T> vector<T> matrix::Dense<T>::operator*(vector<T> &vec) {
  vector<T> y(get_row());
  if (gpu_status == true) {
    y.send();
  }

  blas::matvec(*this, vec, y);

  return y;
}
template vector<double> matrix::Dense<double>::operator*(vector<double> &vec);
template vector<float> matrix::Dense<float>::operator*(vector<float> &vec);
} // namespace monolish
