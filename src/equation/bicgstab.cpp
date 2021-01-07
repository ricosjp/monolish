#include "../../include/monolish_blas.hpp"
#include "../../include/monolish_equation.hpp"
#include "../../include/monolish_vml.hpp"
#include "../internal/monolish_internal.hpp"

#include <fstream>
#include <iomanip>
#include <string>

namespace monolish {

template <typename MATRIX, typename T>
int equation::BiCGSTAB<MATRIX, T>::monolish_BiCGSTAB(MATRIX &A, vector<T> &x,
                                             vector<T> &b) {
  Logger &logger = Logger::get_instance();
  logger.solver_in(monolish_func);

  if (A.get_row() != A.get_col()) {
    throw std::runtime_error("error A.row != A.col");
  }
  if (A.get_device_mem_stat() != x.get_device_mem_stat() &&
      A.get_device_mem_stat() != b.get_device_mem_stat()) {
    throw std::runtime_error("error, A.get_device_mem_stat != "
                             "x.get_device_mem_stat != b.get_device_mem_stat");
  }

  vector<T> r(A.get_row(), 0.0);
  vector<T> r0(A.get_row(), 0.0);

  vector<T> p(A.get_row(), 0.0);
  vector<T> phat(A.get_row(), 0.0);
  vector<T> s(A.get_row(), 0.0);
  vector<T> shat(A.get_row(), 0.0);

  vector<T> v(A.get_row(), 0.0);
  vector<T> t(A.get_row(), 0.0);

  if (A.get_device_mem_stat() == true) {
    monolish::util::send(r, r0, p, phat, s, shat, v, t);
  }

  T rho_old = 1, rho = 1, alpha = 1, beta, omega = 1;

  if (A.get_device_mem_stat() == false) {
    A.send();
  }
  if (x.get_device_mem_stat() == false) {
    x.send();
  }
  if (b.get_device_mem_stat() == false) {
    b.send();
  }

  this->precond.create_precond(A);

  // r = b-Ax
  blas::matvec(A, x, r);
  vml::sub(b, r, r);

  // r0 = r, (r*0, r0)!=0
  r0 = r;

  for (size_t iter = 0; iter < this->maxiter; iter++) {

    // alpha = (r(i-1), r0) / (AM^-1*p(i-1), r0)
    rho = blas::dot(r, r0);

    if (rho == 0.0) {
      printf("breakdown\n");
      return 0;
    }

    if (iter == 0) {
      p = r;
    } else {
      // beta = (rho / rho_old) * (alpha / omega)
      beta = (rho / rho_old) * (alpha / omega);

      // p = r + beta(p + omega * AM-1 p(i-1) )
      blas::axpy(-omega, v, p); // p = -omega*v + p
      blas::xpay(beta, r, p);   // p = r + beta*p
    }

    // phat = M^-1 p(i-1)
    this->precond.apply_precond(p, phat);
    // v = AM^-1p(i-1)
    blas::matvec(A, phat, v);
    alpha = rho / blas::dot(v, r0);

    // s(i) = r(i-1) - alpha v
    blas::axpyz(-alpha, v, r, s);

    // shat = M^-1 s(i)
    this->precond.apply_precond(s, shat);
    // t = A * shat
    blas::matvec(A, shat, t);

    // omega = (AM-1s, s) / (AM-1s, AM-1s)
    omega = blas::dot(t, s) / blas::dot(t, t);

    if (omega == 0.0) {
      return MONOLISH_SOLVER_BREAKDOWN;
    }

    // x(i) = x(i-1) + alpha * M^-1 p(i-1) + omega * M^-1 s(i)
    blas::axpy(alpha, phat, x);
    blas::axpy(omega, shat, x);

    // r(i) = s(i-1) - omega * AM^-1 s(i-1)
    blas::axpyz(-omega, t, s, r);

    // convergence check
    auto resid = this->get_residual(r);
    if (this->print_rhistory == true) {
      *this->rhistory_stream << iter + 1 << "\t" << std::scientific << resid
                             << std::endl;
    }

    if (resid < this->tol && this->miniter <= iter + 1) {
      logger.solver_out();
      return MONOLISH_SOLVER_SUCCESS;
    }

    if (std::isnan(resid)) {
      return MONOLISH_SOLVER_RESIDUAL_NAN;
    }

    rho_old = rho;
  }

  logger.solver_out();
  return MONOLISH_SOLVER_MAXITER;
}
template int equation::BiCGSTAB<matrix::CRS<double>, double>::monolish_BiCGSTAB(
    matrix::CRS<double> &A, vector<double> &x, vector<double> &b);
template int equation::BiCGSTAB<matrix::CRS<float>, float>::monolish_BiCGSTAB(matrix::CRS<float> &A,
                                                          vector<float> &x,
                                                          vector<float> &b);

template int equation::BiCGSTAB<matrix::LinearOperator<double>, double>::monolish_BiCGSTAB(
    matrix::LinearOperator<double> &A, vector<double> &x, vector<double> &b);
template int equation::BiCGSTAB<matrix::LinearOperator<float>, float>::monolish_BiCGSTAB(
    matrix::LinearOperator<float> &A, vector<float> &x, vector<float> &b);

template <typename MATRIX, typename T>
int equation::BiCGSTAB<MATRIX, T>::solve(MATRIX &A, vector<T> &x,
                                 vector<T> &b) {
  Logger &logger = Logger::get_instance();
  logger.solver_in(monolish_func);

  int ret = 0;
  if (this->lib == 0) {
    ret = monolish_BiCGSTAB(A, x, b);
  }

  logger.solver_out();
  return ret; // err code
}
template int equation::BiCGSTAB<matrix::CRS<double>, double>::solve(matrix::CRS<double> &A,
                                               vector<double> &x,
                                               vector<double> &b);
template int equation::BiCGSTAB<matrix::CRS<float>, float>::solve(matrix::CRS<float> &A,
                                              vector<float> &x,
                                              vector<float> &b);
template int equation::BiCGSTAB<matrix::LinearOperator<double>, double>::solve(matrix::LinearOperator<double> &A,
                                               vector<double> &x,
                                               vector<double> &b);
template int equation::BiCGSTAB<matrix::LinearOperator<float>, float>::solve(matrix::LinearOperator<float> &A,
                                               vector<float> &x,
                                               vector<float> &b);
} // namespace monolish
