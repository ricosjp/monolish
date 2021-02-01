#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"

namespace monolish {

namespace {
// double ///////////////////
template <typename VEC>
void Dmatvec_core(const matrix::Dense<double> &A, const VEC &x, VEC &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err, M = MN * N
  assert(A.get_row() == y.size());
  assert(A.get_col() == x.size());
  assert(util::is_same_device_mem_stat(A, x, y));

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
      internal::check_CUDA(cublasDgemv(h, CUBLAS_OP_T, n, m, &alpha, vald, n,
                                       xd, 1, &beta, yd, 1));
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
template <typename VEC>
void Smatvec_core(const matrix::Dense<float> &A, const VEC &x, VEC &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err, M = MN * N
  assert(A.get_row() == y.size());
  assert(A.get_col() == x.size());
  assert(util::is_same_device_mem_stat(A, x, y));

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
      internal::check_CUDA(cublasSgemv(h, CUBLAS_OP_T, m, n, &alpha, vald, m,
                                       xd, 1, &beta, yd, 1));
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
} // namespace

namespace blas {
void matvec(const matrix::Dense<double> &A, const vector<double> &x,
            vector<double> &y) {
  Dmatvec_core(A, x, y);
}

void matvec(const matrix::Dense<float> &A, const vector<float> &x,
            vector<float> &y) {
  Smatvec_core(A, x, y);
}
} // namespace blas
} // namespace monolish
