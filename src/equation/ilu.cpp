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

  if (A.get_device_mem_stat() != true) {
    throw std::runtime_error("ILU on CPU does not impl.");
  }

#if MONOLISH_USE_NVIDIA_GPU
  cusparseHandle_t handle;
  cusparseCreate(&handle);
  cudaDeviceSynchronize();

  // step 1: create a descriptor which contains
  cusparseMatDescr_t descr_M = 0;
  csrilu02Info_t info_M = 0;
  cusparseMatDescr_t descr_L = 0;
  csrsv2Info_t info_L = 0;
  cusparseMatDescr_t descr_U = 0;
  csrsv2Info_t info_U = 0;

  const cusparseSolvePolicy_t policy_M = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
  const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
  const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE;
  const cusparseSolvePolicy_t policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
  const cusparseOperation_t trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;

  cusolver_ilu_create_descr(A, descr_M, info_M, descr_L, info_L, descr_U,
                            info_U, handle);

  bufsize =
      cusolver_ilu_get_buffersize(A, descr_M, info_M, descr_L, info_L, trans_L,
                                  descr_U, info_U, trans_U, handle);

  this->precond.M.resize(A.get_nnz());
#pragma omp parallel for
  for (size_t i = 0; i < A.get_nnz(); i++) {
    this->precond.M.data()[i] = A.val.data()[i];
  }
  this->precond.M.send();

  cusolver_ilu(A, this->precond.M.data(), descr_M, info_M, policy_M, descr_L,
               info_L, policy_L, trans_L, descr_U, info_U, policy_U, trans_U,
               bufsize, handle);

  matM = descr_M;
  infoM = info_M;

  matL = descr_L;
  infoL = info_L;

  matU = descr_U;
  infoU = info_U;

  this->precond.A = &A;
#else
  throw std::runtime_error("ILU on CPU does not impl.");
#endif

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

#if MONOLISH_USE_NVIDIA_GPU
  T *d_z = z.data();
  T *d_r = (T *)r.data();

  monolish::vector<T> tmp(z.size(), 0.0);
  tmp.send();
  T *d_tmp = tmp.data();

  cusparseHandle_t handle;
  cusparseCreate(&handle);
  cudaDeviceSynchronize();

  cusparseMatDescr_t descr_M = (cusparseMatDescr_t)matM;
  csrilu02Info_t info_M = (csrilu02Info_t)infoM;
  cusparseMatDescr_t descr_L = (cusparseMatDescr_t)matL;
  csrsv2Info_t info_L = (csrsv2Info_t)infoL;
  cusparseMatDescr_t descr_U = (cusparseMatDescr_t)matU;
  csrsv2Info_t info_U = (csrsv2Info_t)infoU;

  const cusparseSolvePolicy_t policy_M = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
  const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
  const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE;
  const cusparseSolvePolicy_t policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
  const cusparseOperation_t trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;

  cusolver_ilu_solve(*this->precond.A, this->precond.M.data(), descr_M, info_M,
                     policy_M, descr_L, info_L, policy_L, trans_L, descr_U,
                     info_U, policy_U, trans_U, d_z, d_r, d_tmp, bufsize,
                     handle);

#else
  throw std::runtime_error("ILU on CPU does not impl.");
#endif

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

  if (A.get_device_mem_stat() != true) {
    throw std::runtime_error("ILU on CPU does not impl.");
  }

#if MONOLISH_USE_NVIDIA_GPU
  T *d_x = x.data();
  T *d_b = b.data();

  monolish::vector<T> tmp(x.size(), 0.0);
  tmp.send();
  T *d_tmp = tmp.data();

  cusparseHandle_t handle;
  cusparseCreate(&handle);
  cudaDeviceSynchronize();

  // step 1: create a descriptor which contains
  cusparseMatDescr_t descr_M = (cusparseMatDescr_t)matM;
  csrilu02Info_t info_M = (csrilu02Info_t)infoM;
  cusparseMatDescr_t descr_L = (cusparseMatDescr_t)matL;
  csrsv2Info_t info_L = (csrsv2Info_t)infoL;
  cusparseMatDescr_t descr_U = (cusparseMatDescr_t)matU;
  csrsv2Info_t info_U = (csrsv2Info_t)infoU;

  const cusparseSolvePolicy_t policy_M = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
  const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
  const cusparseSolvePolicy_t policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
  const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE;
  const cusparseOperation_t trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;

  cusolver_ilu_create_descr(A, descr_M, info_M, descr_L, info_L, descr_U,
                            info_U, handle);
  bufsize =
      cusolver_ilu_get_buffersize(A, descr_M, info_M, descr_L, info_L, trans_L,
                                  descr_U, info_U, trans_U, handle);

  monolish::vector<T> tmpval(A.val);
  tmpval.send();

  cusolver_ilu(A, tmpval.data(), descr_M, info_M, policy_M, descr_L, info_L,
               policy_L, trans_L, descr_U, info_U, policy_U, trans_U, bufsize,
               handle);

  cusolver_ilu_solve(A, tmpval.data(), descr_M, info_M, policy_M, descr_L,
                     info_L, policy_L, trans_L, descr_U, info_U, policy_U,
                     trans_U, d_x, d_b, d_tmp, bufsize, handle);

#else
  throw std::runtime_error("ILU on CPU does not impl.");
#endif

  logger.solver_out();
  return MONOLISH_SOLVER_SUCCESS;
}
// template int equation::ILU<matrix::Dense<float>, float>::monolish_ILU(
//     matrix::Dense<float> &A, vector<float> &x, vector<float> &b);
// template int equation::ILU<matrix::Dense<double>, double>::monolish_ILU(
//     matrix::Dense<double> &A, vector<double> &x, vector<double> &b);

template int equation::ILU<matrix::CRS<float>, float>::cusparse_ILU(
    matrix::CRS<float> &A, vector<float> &x, vector<float> &b);
template int equation::ILU<matrix::CRS<double>, double>::cusparse_ILU(
    matrix::CRS<double> &A, vector<double> &x, vector<double> &b);

// template int
// equation::ILU<matrix::LinearOperator<float>, float>::monolish_ILU(
//     matrix::LinearOperator<float> &A, vector<float> &x, vector<float> &b);
// template int
// equation::ILU<matrix::LinearOperator<double>, double>::monolish_ILU(
//     matrix::LinearOperator<double> &A, vector<double> &x, vector<double> &b);

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

template int equation::ILU<matrix::CRS<float>, float>::solve(
    matrix::CRS<float> &A, vector<float> &x, vector<float> &b);
template int equation::ILU<matrix::CRS<double>, double>::solve(
    matrix::CRS<double> &A, vector<double> &x, vector<double> &b);

// template int equation::ILU<matrix::LinearOperator<float>, float>::solve(
//     matrix::LinearOperator<float> &A, vector<float> &x, vector<float> &b);
// template int equation::ILU<matrix::LinearOperator<double>, double>::solve(
//     matrix::LinearOperator<double> &A, vector<double> &x, vector<double> &b);

} // namespace monolish
