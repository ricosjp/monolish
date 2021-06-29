#include "../../../include/monolish_blas.hpp"
#include "../../../include/monolish_equation.hpp"
#include "../../internal/monolish_internal.hpp"

#ifdef MONOLISH_USE_NVIDIA_GPU
#include "cusolverSp.h"
#endif

namespace monolish {

template <>
int equation::QR<matrix::CRS<double>, double>::cusolver_QR(
    matrix::CRS<double> &A, vector<double> &x, vector<double> &b) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

#ifdef MONOLISH_USE_NVIDIA_GPU
  cusolverSpHandle_t sp_handle;
  cusolverSpCreate(&sp_handle);

  cusparseMatDescr_t descrA;
  internal::check_CUDA(cusparseCreateMatDescr(&descrA));
  internal::check_CUDA(
      cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
  internal::check_CUDA(
      cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));
  internal::check_CUDA(
      cusparseSetMatDiagType(descrA, CUSPARSE_DIAG_TYPE_NON_UNIT));

  int n = A.get_row();
  int nnz = A.get_nnz();

  double *Dval = A.val.data();
  int *Dptr = A.row_ptr.data();
  int *Dind = A.col_ind.data();

  const double *Drhv = b.data();
  double *Dsol = x.data();

#pragma omp target data use_device_ptr(Dval, Dptr, Dind, Drhv, Dsol)
  {
    internal::check_CUDA(cusolverSpDcsrlsvqr(sp_handle, n, nnz, descrA, Dval,
                                             Dptr, Dind, Drhv, tol, reorder,
                                             Dsol, &singularity));
  }
#else
  (void)(&A);
  (void)(&x);
  (void)(&b);
  throw std::runtime_error("error sparse Cholesky is only GPU");
#endif
  logger.func_out();
  return 0;
}

template <>
int equation::QR<matrix::CRS<float>, float>::cusolver_QR(matrix::CRS<float> &A,
                                                         vector<float> &x,
                                                         vector<float> &b) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

#ifdef MONOLISH_USE_NVIDIA_GPU
  cusolverSpHandle_t sp_handle;
  cusolverSpCreate(&sp_handle);

  cusparseMatDescr_t descrA;
  internal::check_CUDA(cusparseCreateMatDescr(&descrA));
  internal::check_CUDA(
      cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
  internal::check_CUDA(
      cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));
  internal::check_CUDA(
      cusparseSetMatDiagType(descrA, CUSPARSE_DIAG_TYPE_NON_UNIT));

  int n = A.get_row();
  int nnz = A.get_nnz();

  float *Dval = A.val.data();
  int *Dptr = A.row_ptr.data();
  int *Dind = A.col_ind.data();

  const float *Drhv = b.data();
  float *Dsol = x.data();

#pragma omp target data use_device_ptr(Dval, Dptr, Dind, Drhv, Dsol)
  {
    internal::check_CUDA(cusolverSpScsrlsvqr(sp_handle, n, nnz, descrA, Dval,
                                             Dptr, Dind, Drhv, tol, reorder,
                                             Dsol, &singularity));
  }
#else
  (void)(&A);
  (void)(&x);
  (void)(&b);
  throw std::runtime_error("error sparse Cholesky is only GPU");
#endif
  logger.func_out();
  return 0;
}
} // namespace monolish
