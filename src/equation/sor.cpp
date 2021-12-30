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
void equation::SOR<MATRIX, T>::create_precond(MATRIX &A) {
  Logger &logger = Logger::get_instance();
  logger.solver_in(monolish_func);

  if (A.get_row() != A.get_col()) {
    throw std::runtime_error("error A.row != A.col");
  }

  this->precond.M.device_free();
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
template void equation::SOR<matrix::Dense<float>, float>::create_precond(
    matrix::Dense<float> &A);
template void equation::SOR<matrix::Dense<double>, double>::create_precond(
    matrix::Dense<double> &A);

template void
equation::SOR<matrix::CRS<float>, float>::create_precond(matrix::CRS<float> &A);
template void equation::SOR<matrix::CRS<double>, double>::create_precond(
    matrix::CRS<double> &A);

// template void
// equation::SOR<matrix::LinearOperator<float>, float>::create_precond(
//     matrix::LinearOperator<float> &A);
// template void
// equation::SOR<matrix::LinearOperator<double>, double>::create_precond(
//     matrix::LinearOperator<double> &A);

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <typename MATRIX, typename T>
void equation::SOR<MATRIX, T>::apply_precond(const vector<T> &r, vector<T> &z) {
  Logger &logger = Logger::get_instance();
  logger.solver_in(monolish_func);

  sor_kernel_precond(*this->precond.A, this->precond.M, z, r);

  logger.solver_out();
}
template void equation::SOR<matrix::Dense<float>, float>::apply_precond(
    const vector<float> &r, vector<float> &z);
template void equation::SOR<matrix::Dense<double>, double>::apply_precond(
    const vector<double> &r, vector<double> &z);

template void
equation::SOR<matrix::CRS<float>, float>::apply_precond(const vector<float> &r,
                                                        vector<float> &z);
template void equation::SOR<matrix::CRS<double>, double>::apply_precond(
    const vector<double> &r, vector<double> &z);

// template void
// equation::SOR<matrix::LinearOperator<float>, float>::apply_precond(
//     const vector<float> &r, vector<float> &z);
// template void
// equation::SOR<matrix::LinearOperator<double>, double>::apply_precond(
//     const vector<double> &r, vector<double> &z);

////////////////////////////
// solver
////////////////////////////

template <typename MATRIX, typename T>
int equation::SOR<MATRIX, T>::monolish_SOR(MATRIX &A, vector<T> &x,
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

  vector<T> r(A.get_row(), 0.0);
  vector<T> t(A.get_row(), 0.0);
  vector<T> s(A.get_row(), 0.0);
  vector<T> d(A.get_row(), 0.0);
  util::send(r, t, s, d);

  T w = 1.0 / this->get_omega();
  A.diag(d);
  blas::scal(w, d);
  vml::reciprocal(d, d);
  d.nonfree_recv();

  auto bnrm2 = blas::nrm2(b);
  bnrm2 = 1.0 / bnrm2;
  T nrm2 = 0.0;

  this->precond.create_precond(A);

  for (size_t iter = 0; iter < this->get_maxiter(); iter++) {
    /* x += (D/w-L)^{-1}(b - Ax) */
    this->precond.apply_precond(x, s);
    blas::matvec(A, s, t);
    blas::axpyz(-1.0, t, b, r);
    nrm2 = blas::nrm2(r);

    r.nonfree_recv();

    sor_kernel_lower(A, d, t, r);

    t.send();

    blas::axpy(1.0, t, x);

    nrm2 = nrm2 * bnrm2;

    if (this->get_print_rhistory() == true) {
      *this->rhistory_stream << iter + 1 << "\t" << std::scientific << nrm2
                             << std::endl;
    }
    if (nrm2 < this->get_tol() && this->get_miniter() <= iter + 1) {
      logger.solver_out();
      return MONOLISH_SOLVER_SUCCESS;
    }
    if (std::isnan(nrm2)) {
      logger.solver_out();
      return MONOLISH_SOLVER_RESIDUAL_NAN;
    }
  }

  logger.solver_out();
  return MONOLISH_SOLVER_MAXITER;
}
template int equation::SOR<matrix::Dense<double>, double>::monolish_SOR(
    matrix::Dense<double> &A, vector<double> &x, vector<double> &b);
template int equation::SOR<matrix::Dense<float>, float>::monolish_SOR(
    matrix::Dense<float> &A, vector<float> &x, vector<float> &b);

template int equation::SOR<matrix::CRS<double>, double>::monolish_SOR(
    matrix::CRS<double> &A, vector<double> &x, vector<double> &b);
template int equation::SOR<matrix::CRS<float>, float>::monolish_SOR(
    matrix::CRS<float> &A, vector<float> &x, vector<float> &b);

// template int
// equation::SOR<matrix::LinearOperator<double>, double>::monolish_SOR(
//     matrix::LinearOperator<double> &A, vector<double> &x, vector<double> &b);
// template int
// equation::SOR<matrix::LinearOperator<float>, float>::monolish_SOR(
//     matrix::LinearOperator<float> &A, vector<float> &x, vector<float> &b);

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <typename MATRIX, typename T>
int equation::SOR<MATRIX, T>::solve(MATRIX &A, vector<T> &x, vector<T> &b) {
  Logger &logger = Logger::get_instance();
  logger.solver_in(monolish_func);

  int ret = 0;
  if (this->get_lib() == 0) {
    ret = monolish_SOR(A, x, b);
  }

  logger.solver_out();
  return ret; // err code
}

template int equation::SOR<matrix::Dense<float>, float>::solve(
    matrix::Dense<float> &A, vector<float> &x, vector<float> &b);
template int equation::SOR<matrix::Dense<double>, double>::solve(
    matrix::Dense<double> &A, vector<double> &x, vector<double> &b);

template int equation::SOR<matrix::CRS<float>, float>::solve(
    matrix::CRS<float> &A, vector<float> &x, vector<float> &b);
template int equation::SOR<matrix::CRS<double>, double>::solve(
    matrix::CRS<double> &A, vector<double> &x, vector<double> &b);

// template int equation::SOR<matrix::LinearOperator<float>, float>::solve(
//     matrix::LinearOperator<float> &A, vector<float> &x, vector<float> &b);
// template int equation::SOR<matrix::LinearOperator<double>, double>::solve(
//     matrix::LinearOperator<double> &A, vector<double> &x, vector<double> &b);
} // namespace monolish
