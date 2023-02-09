#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"

namespace monolish {
namespace {
std::string get_matmul_name(std::string func, bool transA, bool transB) {
  func += "_";
  if (transA == true) {
    func += "T";
  } else {
    func += "N";
  }
  if (transB == true) {
    func += "T";
  } else {
    func += "N";
  }
  return func;
}

// double ///////////////////
void Dense_Dense_Dmatmul_core(const double &a, const matrix::Dense<double> &A,
                              const matrix::Dense<double> &B, const double &b,
                              matrix::Dense<double> &C, bool transA,
                              bool transB) {
  Logger &logger = Logger::get_instance();
  logger.func_in(get_matmul_name(monolish_func, transA, transB));

  // err
  assert(A.get_col() == B.get_row());
  assert(A.get_row() == C.get_row());
  assert(B.get_col() == C.get_col());
  assert(util::is_same_device_mem_stat(A, B, C));

  const auto *Ad = A.vad;
  const auto *Bd = B.vad;
  auto *Cd = C.vad;

  // MN = MK * KN
  const auto m = A.get_row();
  const auto n = B.get_col();
  const auto k = A.get_col();
  const double alpha = a;
  const double beta = b;

  if (A.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU
    cublasHandle_t h;
    internal::check_CUDA(cublasCreate(&h));
#pragma omp target data use_device_ptr(Ad, Bd, Cd)
    {
      // cublas is col major
      internal::check_CUDA(cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                                       &alpha, Bd, n, Ad, k, &beta, Cd, n));
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
void Dense_Dense_Smatmul_core(const float &a, const matrix::Dense<float> &A,
                              const matrix::Dense<float> &B, const float &b,
                              matrix::Dense<float> &C, bool transA,
                              bool transB) {
  Logger &logger = Logger::get_instance();
  logger.func_in(get_matmul_name(monolish_func, transA, transB));

  // err
  assert(A.get_col() == B.get_row());
  assert(A.get_row() == C.get_row());
  assert(B.get_col() == C.get_col());
  assert(util::is_same_device_mem_stat(A, B, C));

  const auto *Ad = A.vad;
  const auto *Bd = B.vad;
  auto *Cd = C.vad;

  // MN = MK * KN
  const auto m = A.get_row();
  const auto n = B.get_col();
  const auto k = A.get_col();
  const float alpha = a;
  const float beta = b;

  if (A.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU
    cublasHandle_t h;
    internal::check_CUDA(cublasCreate(&h));
#pragma omp target data use_device_ptr(Ad, Bd, Cd)
    {
      // cublas is col major
      internal::check_CUDA(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                                       &alpha, Bd, n, Ad, k, &beta, Cd, n));
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
} // namespace
} // namespace monolish
