#include "../../include/monolish_blas.hpp"
#include "../../include/monolish_equation.hpp"
#include "../../include/monolish_vml.hpp"
#include "../internal/monolish_internal.hpp"

#include <fstream>
#include <iomanip>
#include <string>

namespace monolish {

template <typename T>
int equation::BiCGSTAB<T>::monolish_BiCGSTAB(matrix::CRS<T> &A, vector<T> &x,
                                 vector<T> &b) {
  Logger &logger = Logger::get_instance();
  logger.solver_in(monolish_func);

  if (A.get_row() != A.get_col()) {
    throw std::runtime_error("error A.row != A.col");
  }

  vector<T> r(A.get_row(), 0.0);
  vector<T> rstar(A.get_row(), 0.0);

  vector<T> p(A.get_row(), 0.0);
  vector<T> phat(A.get_row(), 0.0);
  vector<T> s(A.get_row(), 0.0);
  vector<T> shat(A.get_row(), 0.0);

  vector<T> v(A.get_row(), 0.0);
  vector<T> t(A.get_row(), 0.0);

  monolish::util::send(r, p, phat, s, shat);

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

  // r*0 = r0, (r*0, r0)!=0
  rstar = r;

  // p0 = r0
  p = r;

  //p = Mp

  for (size_t iter = 0; iter < this->maxiter; iter++) {

    // alpha = (r(i-1), rstar) / (AM^-1*p(i-1), rstar)
    auto rho = blas::dot(r, rstar);

    if(rho == 0.0){
      // breakdown
    }
    // phat = M^-1 p(i-1)
    this->precond.apply_precond(p, phat);
    // v = AM^-1p(i-1)
    blas::matvec(A, phat, v);
    auto alpha = rho / blas::dot(v, rstar);

    // s(i) = r(i-1) - alpha v
    blas::axpyz(-alpha, v, r, s);

    // shat = M^-1 s(i)
    this->precond.apply_precond(s, shat);
    // t = A * shat
    blas::matvec(A, shat, t);

    // omega = (AM-1s, s) / (AM-1s, AM-1s)
    auto omega = blas::dot(t, s) / blas::dot(t, t);

    // x(i) = x(i-1) + alpha * M^-1 p(i-1) + omega * M^-1 s(i)
    blas::axpy(alpha,phat,x);
    blas::axpy(omega,shat,x);

    // r(i) = s(i-1) - omega * AM^-1 s(i-1)
    blas::axpyz(-omega, t, s, r);

    //convergence check
    auto resid = this->get_residual(r);
    if (this->print_rhistory == true) {
      *this->rhistory_stream << iter + 1 << "\t" << std::scientific << resid
                             << std::endl;
    }
    if (resid < this->tol && this->miniter <= iter + 1) {
      logger.solver_out();
      return MONOLISH_SOLVER_SUCCESS;
    }
    
    // beta = alpha/omega * (r(i),rstar) / (r(i-1), rstar)
    auto beta = alpha / omega * blas::dot(r, rstar) / rho;

    // p = r + beta(p + omega * AM-1 p(i-1) )
    blas::axpy(-omega, v, p); // p = -omega*v + p
    blas::xpay(beta, r, p); // p = r + beta*p
  }

  logger.solver_out();
  return MONOLISH_SOLVER_MAXITER;
}
template int equation::BiCGSTAB<double>::monolish_BiCGSTAB(matrix::CRS<double> &A,
                                               vector<double> &x,
                                               vector<double> &b);
template int equation::BiCGSTAB<float>::monolish_BiCGSTAB(matrix::CRS<float> &A,
                                              vector<float> &x,
                                              vector<float> &b);

///

template <typename T>
int equation::BiCGSTAB<T>::solve(matrix::CRS<T> &A, vector<T> &x, vector<T> &b) {
  Logger &logger = Logger::get_instance();
  logger.solver_in(monolish_func);

  int ret = 0;
  if (this->lib == 0) {
    ret = monolish_BiCGSTAB(A, x, b);
  }

  logger.solver_out();
  return ret; // err code
}
template int equation::BiCGSTAB<double>::solve(matrix::CRS<double> &A,
                                         vector<double> &x, vector<double> &b);
template int equation::BiCGSTAB<float>::solve(matrix::CRS<float> &A, vector<float> &x,
                                        vector<float> &b);
} // namespace monolish
