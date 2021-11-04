#include "../../include/monolish_blas.hpp"
#include "../../include/monolish_equation.hpp"
#include "../../include/monolish_vml.hpp"
#include "../internal/monolish_internal.hpp"
#include "./kernel/ilu_kernel.hpp"

namespace monolish {

////////////////////////////
// precondition
////////////////////////////
template <typename MATRIX, typename T>
void equation::ILU<MATRIX, T>::create_precond(MATRIX &A) {
  Logger &logger = Logger::get_instance();
  logger.solver_in(monolish_func);

  if (A.get_row() != A.get_col()) {
    throw std::runtime_error("error A.row != A.col");
  }

#if MONOLISH_USE_NVIDIA_GPU
#else
    throw std::runtime_error("ILU on CPU does not impl.");
#endif











  this->precond.M.recv(); // sor does not work on gpu now

  this->precond.A = &A;

  logger.solver_out();
}
// template void equation::ILU<matrix::Dense<float>, float>::create_precond(
//     matrix::Dense<float> &A);
// template void equation::ILU<matrix::Dense<double>, double>::create_precond(
//     matrix::Dense<double> &A);

template void
equation::ILU<matrix::CRS<float>, float>::create_precond(matrix::CRS<float> &A);
template void equation::ILU<matrix::CRS<double>, double>::create_precond(
    matrix::CRS<double> &A);

// template void
// equation::ILU<matrix::LinearOperator<float>, float>::create_precond(
//     matrix::LinearOperator<float> &A);
// template void
// equation::ILU<matrix::LinearOperator<double>, double>::create_precond(
//     matrix::LinearOperator<double> &A);

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <typename MATRIX, typename T>
void equation::ILU<MATRIX, T>::apply_precond(const vector<T> &r, vector<T> &z) {
  Logger &logger = Logger::get_instance();
  logger.solver_in(monolish_func);

  //sor_kernel_precond(*this->precond.A, this->precond.M, z, r);

  logger.solver_out();
}
// template void equation::ILU<matrix::Dense<float>, float>::apply_precond(
//     const vector<float> &r, vector<float> &z);
// template void equation::ILU<matrix::Dense<double>, double>::apply_precond(
//     const vector<double> &r, vector<double> &z);

template void
equation::ILU<matrix::CRS<float>, float>::apply_precond(const vector<float> &r,
                                                        vector<float> &z);
template void equation::ILU<matrix::CRS<double>, double>::apply_precond(
    const vector<double> &r, vector<double> &z);

// template void
// equation::ILU<matrix::LinearOperator<float>, float>::apply_precond(
//     const vector<float> &r, vector<float> &z);
// template void
// equation::ILU<matrix::LinearOperator<double>, double>::apply_precond(
//     const vector<double> &r, vector<double> &z);

////////////////////////////
// solver
////////////////////////////

template <typename MATRIX, typename T>
int equation::ILU<MATRIX, T>::cusparse_ILU(MATRIX &A, vector<T> &x,
                                           vector<T> &b) {
  Logger &logger = Logger::get_instance();
  logger.solver_in(monolish_func);

  if (A.get_row() != A.get_col()) {
    throw std::runtime_error("error, A.row != A.col");
  }
  if (A.get_device_mem_stat() != x.get_device_mem_stat() &&
      A.get_device_mem_stat() != b.get_device_mem_stat()) {
    throw std::runtime_error("error, A.get_device_mem_stat != "
                             "x.get_device_mem_stat != b.get_device_mem_stat");
  }

#if MONOLISH_USE_NVIDIA_GPU
  auto M = A.get_row();
  auto nnz = A.get_nnz();
  T* d_x = x.data();
  T* d_b = b.data();

  monolish::vector<T> tmp(x.size(),0.0);
  tmp.send();
  T* d_tmp = tmp.data();

  int* d_csrRowPtr = A.row_ptr.data();
  int* d_csrColInd = A.col_ind.data();
  T* d_csrVal = A.val.data();

  cusparseHandle_t handle;
  cusparseCreate(&handle);
  cudaDeviceSynchronize();

  int pBufferSize_M;
  int pBufferSize_L;
  int pBufferSize_U;
  int pBufferSize;
  void *pBuffer = this->buf;

  int structural_zero;
  int numerical_zero;
  const T alpha = 1.0;

  // step 1: create a descriptor which contains
  const cusparseSolvePolicy_t policy_M = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
  cusparseMatDescr_t descr_M = 0;
  csrilu02Info_t info_M  = 0;

  const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
  const cusparseOperation_t trans_L  = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseMatDescr_t descr_L = 0;
  csrsv2Info_t  info_L  = 0;

  const cusparseSolvePolicy_t policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
  const cusparseOperation_t trans_U  = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseMatDescr_t descr_U = 0;
  csrsv2Info_t  info_U  = 0;

#pragma omp target data use_device_ptr(d_csrVal, d_csrRowPtr, d_csrColInd, d_x, d_b, d_tmp)
  {
      cusparseCreateMatDescr(&descr_M);
      cusparseSetMatIndexBase(descr_M, CUSPARSE_INDEX_BASE_ZERO);
      cusparseSetMatType(descr_M, CUSPARSE_MATRIX_TYPE_GENERAL);

      cusparseCreateMatDescr(&descr_L);
      cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ZERO);
      cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL);
      cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER);
      cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_UNIT);

      cusparseCreateMatDescr(&descr_U);
      cusparseSetMatIndexBase(descr_U, CUSPARSE_INDEX_BASE_ZERO);
      cusparseSetMatType(descr_U, CUSPARSE_MATRIX_TYPE_GENERAL);
      cusparseSetMatFillMode(descr_U, CUSPARSE_FILL_MODE_UPPER);
      cusparseSetMatDiagType(descr_U, CUSPARSE_DIAG_TYPE_NON_UNIT);

      // step 2: create a empty info structure
      // we need one info for csrilu02 and two info's for csrsv2
      cusparseCreateCsrilu02Info(&info_M);
      cusparseCreateCsrsv2Info(&info_L);
      cusparseCreateCsrsv2Info(&info_U);

      // step 3: query how much memory used in csrilu02 and csrsv2, and allocate the buffer
      cusparseDcsrilu02_bufferSize(handle, M, nnz,
              descr_M, d_csrVal, d_csrRowPtr, d_csrColInd, info_M, &pBufferSize_M);
      cusparseDcsrsv2_bufferSize(handle, trans_L, M, nnz,
              descr_L, d_csrVal, d_csrRowPtr, d_csrColInd, info_L, &pBufferSize_L);
      cusparseDcsrsv2_bufferSize(handle, trans_U, M, nnz,
              descr_U, d_csrVal, d_csrRowPtr, d_csrColInd, info_U, &pBufferSize_U);

      pBufferSize = std::max(pBufferSize_M, std::max(pBufferSize_L, pBufferSize_U));
  }

#pragma omp target data use_device_ptr(d_csrVal, d_csrRowPtr, d_csrColInd, d_x, d_b, d_tmp, pBuffer)
  {
      // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
      cudaMalloc((void**)&pBuffer, pBufferSize);

      // step 4: perform analysis of incomplete Cholesky on M
      //         perform analysis of triangular solve on L
      //         perform analysis of triangular solve on U
      // The lower(upper) triangular part of M has the same sparsity pattern as L(U),
      // we can do analysis of csrilu0 and csrsv2 simultaneously.

      cusparseDcsrilu02_analysis(handle, M, nnz, descr_M,
              d_csrVal, d_csrRowPtr, d_csrColInd, info_M,
              policy_M, pBuffer);
      auto status = cusparseXcsrilu02_zeroPivot(handle, info_M, &structural_zero);

      if (CUSPARSE_STATUS_ZERO_PIVOT == status){
          printf("A(%d,%d) is missing\n", structural_zero, structural_zero);
      }

      cusparseDcsrsv2_analysis(handle, trans_L, M, nnz, descr_L,
              d_csrVal, d_csrRowPtr, d_csrColInd,
              info_L, policy_L, pBuffer);

      cusparseDcsrsv2_analysis(handle, trans_U, M, nnz, descr_U,
              d_csrVal, d_csrRowPtr, d_csrColInd,
              info_U, policy_U, pBuffer);

      // step 5: M = L * U
      cusparseDcsrilu02(handle, M, nnz, descr_M,
              d_csrVal, d_csrRowPtr, d_csrColInd, info_M, policy_M, pBuffer);
      status = cusparseXcsrilu02_zeroPivot(handle, info_M, &numerical_zero);
      if (CUSPARSE_STATUS_ZERO_PIVOT == status){
          printf("U(%d,%d) is zero\n", numerical_zero, numerical_zero);
      }
  }

#pragma omp target data use_device_ptr(d_csrVal, d_csrRowPtr, d_csrColInd, d_x, d_b, d_tmp, pBuffer)
  {
      // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
      cudaMalloc((void**)&pBuffer, pBufferSize);
      // step 6: solve L*z = x
      cusparseDcsrsv2_solve(handle, trans_L, M, nnz, &alpha, descr_L,
              d_csrVal, d_csrRowPtr, d_csrColInd, info_L,
              d_b, d_tmp, policy_L, pBuffer);

      // step 7: solve U*y = z
      cusparseDcsrsv2_solve(handle, trans_U, M, nnz, &alpha, descr_U,
              d_csrVal, d_csrRowPtr, d_csrColInd, info_U,
              d_tmp, d_x, policy_U, pBuffer);

      // step 6: free resources
      cudaFree(pBuffer);
      cusparseDestroyMatDescr(descr_M);
      cusparseDestroyMatDescr(descr_L);
      cusparseDestroyMatDescr(descr_U);
      cusparseDestroyCsrilu02Info(info_M);
      cusparseDestroyCsrsv2Info(info_L);
      cusparseDestroyCsrsv2Info(info_U);
      cusparseDestroy(handle);
  }

#else
    throw std::runtime_error("ILU on CPU does not impl.");
#endif

  logger.solver_out();
  return MONOLISH_SOLVER_MAXITER;
}
// template int equation::ILU<matrix::Dense<double>, double>::monolish_ILU(
//     matrix::Dense<double> &A, vector<double> &x, vector<double> &b);
// template int equation::ILU<matrix::Dense<float>, float>::monolish_ILU(
//     matrix::Dense<float> &A, vector<float> &x, vector<float> &b);

template int equation::ILU<matrix::CRS<double>, double>::cusparse_ILU(
    matrix::CRS<double> &A, vector<double> &x, vector<double> &b);
// template int equation::ILU<matrix::CRS<float>, float>::cusparse_ILU(
//     matrix::CRS<float> &A, vector<float> &x, vector<float> &b);

// template int
// equation::ILU<matrix::LinearOperator<double>, double>::monolish_ILU(
//     matrix::LinearOperator<double> &A, vector<double> &x, vector<double> &b);
// template int
// equation::ILU<matrix::LinearOperator<float>, float>::monolish_ILU(
//     matrix::LinearOperator<float> &A, vector<float> &x, vector<float> &b);

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <typename MATRIX, typename T>
int equation::ILU<MATRIX, T>::solve(MATRIX &A, vector<T> &x, vector<T> &b) {
  Logger &logger = Logger::get_instance();
  logger.solver_in(monolish_func);

  int ret = 0;
  if (this->get_lib() == 0) {
    ret = cusparse_ILU(A, x, b);
  }

  logger.solver_out();
  return ret; // err code
}

// template int equation::ILU<matrix::Dense<float>, float>::solve(
//     matrix::Dense<float> &A, vector<float> &x, vector<float> &b);
// template int equation::ILU<matrix::Dense<double>, double>::solve(
//     matrix::Dense<double> &A, vector<double> &x, vector<double> &b);

// template int equation::ILU<matrix::CRS<float>, float>::solve(
//     matrix::CRS<float> &A, vector<float> &x, vector<float> &b);
template int equation::ILU<matrix::CRS<double>, double>::solve(
    matrix::CRS<double> &A, vector<double> &x, vector<double> &b);

// template int equation::ILU<matrix::LinearOperator<float>, float>::solve(
//     matrix::LinearOperator<float> &A, vector<float> &x, vector<float> &b);
// template int equation::ILU<matrix::LinearOperator<double>, double>::solve(
//     matrix::LinearOperator<double> &A, vector<double> &x, vector<double> &b);
} // namespace monolish
