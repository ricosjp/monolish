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
template <typename MAT1, typename MAT2, typename MAT3>
void Dense_Dense_Dmatmul_core(const double &a, const MAT1 &A, const MAT2 &B,
                              const double &b, MAT3 &C, bool transA,
                              bool transB) {
  Logger &logger = Logger::get_instance();
  logger.func_in(get_matmul_name(monolish_func, transA, transB));

  // err
  assert(A.get_col() == B.get_row());
  assert(A.get_row() == C.get_row());
  assert(B.get_col() == C.get_col());
  assert(util::is_same_device_mem_stat(A, B, C));

  const auto *Ad = A.begin();
  const auto *Bd = B.begin();
  auto *Cd = C.begin();

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
template <typename MAT1, typename MAT2, typename MAT3>
void Dense_Dense_Smatmul_core(const float &a, const MAT1 &A, const MAT2 &B,
                              const float &b, MAT3 &C, bool transA,
                              bool transB) {
  Logger &logger = Logger::get_instance();
  logger.func_in(get_matmul_name(monolish_func, transA, transB));

  // err
  assert(A.get_col() == B.get_row());
  assert(A.get_row() == C.get_row());
  assert(B.get_col() == C.get_col());
  assert(util::is_same_device_mem_stat(A, B, C));

  const auto *Ad = A.begin();
  const auto *Bd = B.begin();
  auto *Cd = C.begin();

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
