#include "../../monolish_internal.hpp"
#include "../monolish_lapack_float.hpp"
#ifndef MONOLISH_USE_MKL
#include "../lapack.h"
#endif

#include <vector>

namespace monolish {

// float
int internal::lapack::getrf(matrix::Dense<float> &A, std::vector<int> &ipiv) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  if (ipiv.size() != std::min(A.get_row(), A.get_col())) {
    logger.func_out();
    std::runtime_error("lapack::getrf, ipiv size error");
  }

  int info = 0;
  const int M = (int)A.get_row();
  const int N = (int)A.get_col();
  float *Ad = A.val.data();
  int *ipivd = ipiv.data();

  if (A.get_device_mem_stat()) {
#if MONOLISH_USE_NVIDIA_GPU
    int ipivl = ipiv.size();
    cudaDeviceSynchronize();
    cusolverDnHandle_t h;
    internal::check_CUDA(cusolverDnCreate(&h));
    int lwork = -1;

#pragma omp target data use_device_ptr(Ad)
    {
      internal::check_CUDA(cusolverDnSgetrf_bufferSize(h, M, N, Ad, M, &lwork));
    }

#pragma omp target enter data map(to : ipivd [0:ipivl])
    monolish::vector<float> work(lwork);
    work.send();
    float *workd = work.data();
    std::vector<int> devinfo(1);
    int *devinfod = devinfo.data();

#pragma omp target enter data map(to : devinfod [0:1])
#pragma omp target data use_device_ptr(Ad, ipivd, workd, devinfod)
    {
      internal::check_CUDA(
          cusolverDnSgetrf(h, M, N, Ad, M, workd, ipivd, devinfod));
    }
#pragma omp target exit data map(from : devinfod [0:1])
    cudaDeviceSynchronize();
    info = devinfo[0];
    cusolverDnDestroy(h);

#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
    sgetrf_(&M, &N, Ad, &M, ipivd, &info);
  }

  logger.func_out();
  return info;
}

} // namespace monolish
