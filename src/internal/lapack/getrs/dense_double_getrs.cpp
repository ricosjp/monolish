#include "../../monolish_internal.hpp"
#include "../monolish_lapack_double.hpp"
#ifndef MONOLISH_USE_MKL
#include "../lapack.h"
#endif

#include <vector>

namespace monolish {

// double
int internal::lapack::getrs(const matrix::Dense<double> &A, vector<double> &B, const std::vector<int>& ipiv){
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  
  int info = 0;
  const int M = (int)A.get_row();
  const int N = (int)A.get_col();
  const int K = 1;
  const double* Ad = A.val.data();
  double* Bd = B.data();
  const int* ipivd = ipiv.data();
  const char trans = 'N';

  if(A.get_device_mem_stat() == true && B.get_device_mem_stat() == true){
#if MONOLISH_USE_GPU
    cudaDeviceSynchronize();
    cusolverDnHandle_t h;
    internal::check_CUDA(cusolverDnCreate(&h));
#pragma omp target enter data map(to: ipivd[0:M])

#pragma omp target data use_device_ptr(Ad, ipivd, Bd) 
    {
      internal::check_CUDA(cusolverDnDgetrs(h, CUBLAS_OP_N, M, K, Ad, N, ipivd, Bd, M, &info));
    }

    //free
#pragma omp target exit data map(release : ipivd[0:M])
    cusolverDnDestroy(h);

#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  }
  else{
    dgetrs_(&trans, &M, &K, Ad, &N, ipivd, Bd, &M, &info);
  }

  logger.func_out();
  return info;
}

} // namespace monolish
