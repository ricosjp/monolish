#include "../../include/monolish_blas.hpp"
#include "../../include/monolish_equation.hpp"
#include "../../include/monolish_vml.hpp"
#include "../internal/monolish_internal.hpp"

#include <fstream>
#include <iomanip>
#include <string>

namespace monolish {

template <typename MATRIX, typename T>
int equation::CG<MATRIX, T>::monolish_CG(MATRIX &A, vector<T> &x,
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
  vector<T> p(A.get_row(), 0.0);
  vector<T> q(A.get_row(), 0.0);
  vector<T> z(A.get_row(), 0.0);

  if (A.get_device_mem_stat() == true) {
    monolish::util::send(r, p, q, z);
  }

  this->precond.create_precond(A);

  // r = b-Ax
  blas::matvec(A, x, q);
  vml::sub(b, q, r);

  // p0 = Mr0
  this->precond.apply_precond(r, z);
  blas::copy(z, p);

  for (size_t iter = 0; iter < this->get_maxiter(); iter++) {
    blas::matvec(A, p, q);

    auto tmp = blas::dot(z, r);
    auto alpha = tmp / blas::dot(p, q);

    blas::axpy(alpha, p, x);

    blas::axpy(-alpha, q, r);

    this->precond.apply_precond(r, z);
    auto beta = blas::dot(z, r) / tmp;

    blas::xpay(beta, z, p); // p = z + beta*p

    T resid = this->get_residual(r);
    if (this->get_print_rhistory() == true) {
      *this->rhistory_stream << iter + 1 << "\t" << std::scientific << resid
                             << std::endl;
    }

    if (resid < this->get_tol() && this->get_miniter() <= iter + 1) {
      logger.solver_out();
      return MONOLISH_SOLVER_SUCCESS;
    }

    if (std::isnan(resid)) {
      return MONOLISH_SOLVER_RESIDUAL_NAN;
    }
  }

  logger.solver_out();
  return MONOLISH_SOLVER_MAXITER;
}
template int equation::CG<matrix::CRS<double>, double>::monolish_CG(
    matrix::CRS<double> &A, vector<double> &x, vector<double> &b);
template int equation::CG<matrix::CRS<float>, float>::monolish_CG(
    matrix::CRS<float> &A, vector<float> &x, vector<float> &b);

///

template <typename MATRIX, typename T>
int equation::CG<MATRIX, T>::solve(MATRIX &A, vector<T> &x, vector<T> &b) {
  Logger &logger = Logger::get_instance();
  logger.solver_in(monolish_func);

  int ret = 0;
  if (this->get_lib() == 0) {
    ret = monolish_CG(A, x, b);
  }

  logger.solver_out();
  return ret; // err code
}
template int equation::CG<matrix::CRS<double>, double>::solve(
    matrix::CRS<double> &A, vector<double> &x, vector<double> &b);
template int equation::CG<matrix::CRS<float>, float>::solve(
    matrix::CRS<float> &A, vector<float> &x, vector<float> &b);
} // namespace monolish
