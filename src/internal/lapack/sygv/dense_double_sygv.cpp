#include "../../monolish_internal.hpp"
#include "../monolish_lapack_double.hpp"
#ifndef MONOLISH_USE_MKL
#include "../lapack.h"
#endif

#include <vector>

namespace monolish {

// double
int internal::lapack::sygvd(matrix::Dense<double> &A, matrix::Dense<double> &B,
                            vector<double> &W, const int itype,
                            const char *jobz, const char *uplo) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);
  int info = 0;
  int size = static_cast<int>(A.get_row());
  int lwork = -1;
#ifdef MONOLISH_USE_NVIDIA_GPU
  cudaDeviceSynchronize();
  cusolverDnHandle_t h;
  internal::check_CUDA(cusolverDnCreate(&h));
  cusolverEigType_t cu_itype;
  switch (itype) {
  case 1:
    cu_itype = CUSOLVER_EIG_TYPE_1;
    break;
  case 2:
    cu_itype = CUSOLVER_EIG_TYPE_2;
    break;
  case 3:
    cu_itype = CUSOLVER_EIG_TYPE_3;
    break;
  default:
    throw std::runtime_error("itype should be either 1, 2, 3");
  }
  cusolverEigMode_t cu_jobz;
  if (jobz[0] == 'N') {
    cu_jobz = CUSOLVER_EIG_MODE_NOVECTOR;
  } else if (jobz[0] == 'V') {
    cu_jobz = CUSOLVER_EIG_MODE_VECTOR;
  } else {
    throw std::runtime_error("jobz should be N or V");
  }
  cublasFillMode_t cu_uplo;
  if (uplo[0] == 'U') {
    cu_uplo = CUBLAS_FILL_MODE_UPPER;
  } else if (uplo[0] == 'L') {
    cu_uplo = CUBLAS_FILL_MODE_LOWER;
  } else {
    throw std::runtime_error("uplo should be U or L");
  }
  monolish::util::send(A, B, W);
  double *Avald = A.val.data();
  double *Bvald = B.val.data();
  double *Wd = W.data();
#pragma omp target data use_device_ptr(Avald, Bvald, Wd)
  {
    // workspace query
    internal::check_CUDA(cusolverDnDsygvd_bufferSize(h, cu_itype, cu_jobz,
                                                     cu_uplo, size, Avald, size,
                                                     Bvald, size, Wd, &lwork));
  }
  monolish::vector<double> work(lwork);
  work.send();
  double *workd = work.data();
  std::vector<int> devinfo(1);
  int *devinfod = devinfo.data();
#pragma omp target enter data map(to : devinfod [0:1])
#pragma omp target data use_device_ptr(Avald, Bvald, Wd, workd, devinfod)
  {
    internal::check_CUDA(cusolverDnDsygvd(h, cu_itype, cu_jobz, cu_uplo, size,
                                          Avald, size, Bvald, size, Wd, workd,
                                          lwork, devinfod));
  }
#pragma omp target exit data map(from : devinfod [0:1])
  cudaDeviceSynchronize();
  info = devinfo[0];
  monolish::util::recv(A, W);
  cusolverDnDestroy(h);
#else // MONOLISH_USE_NVIDIA_GPU
  std::vector<double> work(1);
  int liwork = -1;
  std::vector<int> iwork(1);
  // workspace query; no error returned
  dsygvd_(&itype, jobz, uplo, &size, A.val.data(), &size, B.val.data(), &size,
          W.data(), work.data(), &lwork, iwork.data(), &liwork, &info);

  // workspace preparation
  lwork = work[0];
  work.resize(lwork);
  liwork = iwork[0];
  iwork.resize(liwork);
  // actual calculation
  dsygvd_(&itype, jobz, uplo, &size, A.val.data(), &size, B.val.data(), &size,
          W.data(), work.data(), &lwork, iwork.data(), &liwork, &info);
#endif
  logger.func_out();
  return info;
}

} // namespace monolish
