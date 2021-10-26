#include "../../include/monolish_blas.hpp"
#include "../../include/monolish_equation.hpp"
#include "../../include/monolish_vml.hpp"
#include "../internal/monolish_internal.hpp"
#include "./kernel/sor_kernel.hpp"

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

  this->precond.M.resize(A.get_row());

  // send M
  if (A.get_device_mem_stat() == true) {
    this->precond.M.send();
  }

  T w = this->get_omega();
  A.diag(this->precond.M);
  blas::scal(w, this->precond.M);
  vml::reciprocal(this->precond.M, this->precond.M);

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

  sor_kernel_precond(*this->precond.A, this->precond.M, z, r);

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


  logger.solver_out();
  return MONOLISH_SOLVER_MAXITER;
}
// template int equation::ILU<matrix::Dense<double>, double>::monolish_ILU(
//     matrix::Dense<double> &A, vector<double> &x, vector<double> &b);
// template int equation::ILU<matrix::Dense<float>, float>::monolish_ILU(
//     matrix::Dense<float> &A, vector<float> &x, vector<float> &b);

template int equation::ILU<matrix::CRS<double>, double>::cusparse_ILU(
    matrix::CRS<double> &A, vector<double> &x, vector<double> &b);
template int equation::ILU<matrix::CRS<float>, float>::cusparse_ILU(
    matrix::CRS<float> &A, vector<float> &x, vector<float> &b);

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

template int equation::ILU<matrix::CRS<float>, float>::solve(
    matrix::CRS<float> &A, vector<float> &x, vector<float> &b);
template int equation::ILU<matrix::CRS<double>, double>::solve(
    matrix::CRS<double> &A, vector<double> &x, vector<double> &b);

// template int equation::ILU<matrix::LinearOperator<float>, float>::solve(
//     matrix::LinearOperator<float> &A, vector<float> &x, vector<float> &b);
// template int equation::ILU<matrix::LinearOperator<double>, double>::solve(
//     matrix::LinearOperator<double> &A, vector<double> &x, vector<double> &b);
} // namespace monolish
