#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"

namespace monolish {

// double ///////////////////
void blas::matmul(const matrix::Dense<double> &A,
                  const matrix::Dense<double> &B, matrix::Dense<double> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_col() != B.get_row()) {
    std::cout << "A.col: " << A.get_col() << std::flush;
    std::cout << ", " << std::flush;
    std::cout << "B.row: " << B.get_row() << std::endl;
    throw std::runtime_error("error A.col != B.row");
  }

  if (A.get_row() != C.get_row()) {
    std::cout << "A.row: " << A.get_row() << std::flush;
    std::cout << ", " << std::flush;
    std::cout << "C.row: " << C.get_row() << std::endl;
    throw std::runtime_error("error A.row != B.row");
  }

  if (B.get_col() != C.get_col()) {
    std::cout << "B.col: " << B.get_col() << std::flush;
    std::cout << ", " << std::flush;
    std::cout << "C.col: " << C.get_col() << std::endl;
    throw std::runtime_error("error B.col != C.col");
  }

  if (A.get_device_mem_stat() != B.get_device_mem_stat() ||
      A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error get_device_mem_stat() is not same");
  }

  const double *Ad = A.val.data();
  const double *Bd = B.val.data();
  double *Cd = C.val.data();

  // MN = MK * KN
  const size_t m = A.get_row();
  const size_t n = B.get_col();
  const size_t k = A.get_col();
  const double alpha = 1.0;
  const double beta = 0.0;

  if (A.get_device_mem_stat() == true) {
#if MONOLISH_USE_GPU
    cublasHandle_t h;
    internal::check_CUDA(cublasCreate(&h));
#pragma omp target data use_device_ptr(Ad, Bd, Cd)
    {
      // cublas is col major
      internal::check_CUDA(cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, Bd, n, Ad,
                        k, &beta, Cd, n));
    }
    cublasDestroy(h);
#else
    throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
  } else {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, Ad,
                k, Bd, n, beta, Cd, n);
  }
  logger.func_out();
}

// float ///////////////////
void blas::matmul(const matrix::Dense<float> &A, const matrix::Dense<float> &B,
                  matrix::Dense<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_col() != B.get_row()) {
    std::cout << "A.col: " << A.get_col() << std::flush;
    std::cout << ", " << std::flush;
    std::cout << "B.row: " << B.get_row() << std::endl;
    throw std::runtime_error("error A.col != B.row");
  }

  if (A.get_row() != C.get_row()) {
    std::cout << "A.row: " << A.get_row() << std::flush;
    std::cout << ", " << std::flush;
    std::cout << "C.row: " << C.get_row() << std::endl;
    throw std::runtime_error("error A.row != B.row");
  }

  if (B.get_col() != C.get_col()) {
    std::cout << "B.col: " << B.get_col() << std::flush;
    std::cout << ", " << std::flush;
    std::cout << "C.col: " << C.get_col() << std::endl;
    throw std::runtime_error("error B.col != C.col");
  }

  if (A.get_device_mem_stat() != B.get_device_mem_stat() ||
      A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error get_device_mem_stat() is not same");
  }

  const float *Ad = A.val.data();
  const float *Bd = B.val.data();
  float *Cd = C.val.data();

  // MN = MK * KN
  const size_t m = A.get_row();
  const size_t n = B.get_col();
  const size_t k = A.get_col();
  const float alpha = 1.0;
  const float beta = 0.0;

  if (A.get_device_mem_stat() == true) {
#if MONOLISH_USE_GPU
    cublasHandle_t h;
    internal::check_CUDA(cublasCreate(&h));
#pragma omp target data use_device_ptr(Ad, Bd, Cd)
    {
      // cublas is col major
      internal::check_CUDA(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, Bd, n, Ad,
                        k, &beta, Cd, n));
    }
    cublasDestroy(h);
#else
    throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
  } else {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, Ad,
                k, Bd, n, beta, Cd, n);
  }

  logger.func_out();
}

template <typename T>
matrix::Dense<T> matrix::Dense<T>::operator*(const matrix::Dense<T> &B) {
  matrix::Dense<T> C(get_row(), B.get_col());
  if (gpu_status == true) {
    C.send();
  }

  blas::matmul(*this, B, C);

  return C;
}
template matrix::Dense<double>
matrix::Dense<double>::operator*(const matrix::Dense<double> &B);
template matrix::Dense<float>
matrix::Dense<float>::operator*(const matrix::Dense<float> &B);
} // namespace monolish
