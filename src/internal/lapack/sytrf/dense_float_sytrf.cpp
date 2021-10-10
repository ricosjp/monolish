#include "../../monolish_internal.hpp"
#include "../monolish_lapack_float.hpp"
#ifndef MONOLISH_USE_MKL
#include "../lapack.h"
#endif

#include <vector>

namespace monolish {

// float
int internal::lapack::sytrf(matrix::Dense<float> &A, std::vector<int> &ipiv) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  int info = 0;
  const int M = (int)A.get_row(); // square(symmetic)
  float *Ad = A.val.data();
  int *ipivd = ipiv.data();
  const char U = 'U';

  if ((int)ipiv.size() != M) {
    logger.func_out();
    std::runtime_error("lapack::getrf, ipiv size error");
  }

  if (A.get_device_mem_stat()) {
#if MONOLISH_USE_NVIDIA_GPU
    cudaDeviceSynchronize();
    cusolverDnHandle_t h;
    internal::check_CUDA(cusolverDnCreate(&h));
    int lwork = -1;

#pragma omp target data use_device_ptr(Ad)
    { internal::check_CUDA(cusolverDnSsytrf_bufferSize(h, M, Ad, M, &lwork)); }

#pragma omp target enter data map(to : ipivd [0:M])
    monolish::vector<float> work(lwork);
    work.send();
    float *workd = work.data();

    std::vector<int> devinfo(1);
    int *devinfod = devinfo.data();

#pragma omp target enter data map(to : devinfod [0:1])
#pragma omp target data use_device_ptr(Ad, ipivd, workd, devinfod)
    {
      internal::check_CUDA(cusolverDnSsytrf(h, CUBLAS_FILL_MODE_UPPER, M, Ad, M,
                                            ipivd, workd, lwork, devinfod));
    }

#pragma omp target exit data map(from : devinfod [0:1])
#pragma omp target exit data map(from : ipivd [0:M])
    cudaDeviceSynchronize();
    info = devinfo[0];
    cusolverDnDestroy(h);

#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
    int lwork = M;
    std::vector<float> work(M);
    ssytrf_(&U, &M, Ad, &M, ipivd, work.data(), &lwork, &info);
  }

  logger.func_out();
  return info;
}

} // namespace monolish
