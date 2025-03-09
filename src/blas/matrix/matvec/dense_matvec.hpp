#pragma once

namespace monolish {
namespace {
std::string get_matvec_name(std::string func, bool flag) {
  if (flag == true) {
    return func + "_T";
  } else {
    return func + "_N";
  }
}
// double ///////////////////
template <typename MAT1, typename VEC1, typename VEC2>
void Dmatvec_core(const double alpha, const MAT1 &A, const VEC1 &x,
                  const double beta, VEC2 &y, bool transA) {
  Logger &logger = Logger::get_instance();
  logger.func_in(get_matvec_name(monolish_func, transA));

  // err, M = MN * N
  if (transA) {
    assert(A.get_row() == x.size());
    assert(A.get_col() == y.size());
  } else {
    assert(A.get_row() == y.size());
    assert(A.get_col() == x.size());
  }
  assert(util::is_same_device_mem_stat(A, x, y));

  const auto *xd = x.begin();
  auto *yd = y.begin();
  const auto *vald = A.begin();
  const auto m = A.get_row();
  const auto n = A.get_col();

  if (A.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU
    cublasHandle_t h;
    internal::check_CUDA(cublasCreate(&h));
#pragma omp target data use_device_ptr(xd, yd, vald)
    {
      // cublas is col major
      internal::check_CUDA(cublasDgemv(h, internal::get_cublas_trans(!transA),
                                       n, m, &alpha, vald, n, xd, 1, &beta, yd,
                                       1));
    }
    cublasDestroy(h);
#else
    throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
  } else {
    cblas_dgemv(CblasRowMajor, internal::get_cblas_trans(transA), m, n, alpha,
                vald, n, xd, 1, beta, yd, 1);
  }

  logger.func_out();
}

// float ///////////////////
template <typename MAT1, typename VEC1, typename VEC2>
void Smatvec_core(const float alpha, const MAT1 &A, const VEC1 &x,
                  const float beta, VEC2 &y, bool transA) {
  Logger &logger = Logger::get_instance();
  logger.func_in(get_matvec_name(monolish_func, transA));

  // err, M = MN * N
  if (transA) {
    assert(A.get_row() == x.size());
    assert(A.get_col() == y.size());
  } else {
    assert(A.get_row() == y.size());
    assert(A.get_col() == x.size());
  }
  assert(util::is_same_device_mem_stat(A, x, y));

  const auto *xd = x.begin();
  auto *yd = y.begin();
  const auto *vald = A.begin();
  const auto n = A.get_row();
  const auto m = A.get_col();

  if (A.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU
    cublasHandle_t h;
    internal::check_CUDA(cublasCreate(&h));
#pragma omp target data use_device_ptr(xd, yd, vald)
    {
      // cublas is col major
      internal::check_CUDA(cublasSgemv(h, internal::get_cublas_trans(!transA),
                                       m, n, &alpha, vald, m, xd, 1, &beta, yd,
                                       1));
    }
    cublasDestroy(h);
#else
    throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
  } else {
    cblas_sgemv(CblasRowMajor, internal::get_cblas_trans(transA), n, m, alpha,
                vald, m, xd, 1, beta, yd, 1);
  }

  logger.func_out();
}
} // namespace

} // namespace monolish
