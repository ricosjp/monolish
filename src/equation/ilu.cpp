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
  T* d_x = x.data();
  T* d_b = b.data();

  monolish::vector<T> tmp(x.size(),0.0);
  tmp.send();
  T* d_tmp = tmp.data();

  cusparseHandle_t handle;
  cusparseCreate(&handle);
  cudaDeviceSynchronize();

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

  cusolver_ilu_create_descr(A, descr_M, info_M, descr_L, info_L, descr_U, info_U, handle);
  auto bufsize = cusolver_ilu_get_buffersize(A, descr_M, info_M, descr_L, info_L, trans_L, descr_U, info_U, trans_U, handle);

  cusolver_ilu(A, 
          descr_M, info_M, policy_M, 
          descr_L, info_L, policy_L, trans_L, 
          descr_U, info_U, policy_U, trans_U,
          bufsize, handle);

  cusolver_ilu_solve(A, 
          descr_M, info_M, policy_M, 
          descr_L, info_L, policy_L, trans_L, 
          descr_U, info_U, policy_U, trans_U,
          d_x, d_b, d_tmp, bufsize, handle);

#pragma omp target data use_device_ptr(descr_M, descr_L, descr_U, info_M, info_L, info_U, handle)
  {
      // step 6: free resources
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
