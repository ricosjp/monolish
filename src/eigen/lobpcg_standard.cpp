#include "../../include/monolish_blas.hpp"
#include "../../include/monolish_eigen.hpp"
#include "../internal/lapack/monolish_lapack.hpp"
#include "../internal/monolish_internal.hpp"

namespace monolish {

template <typename MATRIX, typename T>
int standard_eigen::LOBPCG<MATRIX, T>::monolish_LOBPCG(
    MATRIX &A, T &l, monolish::vector<T> &xinout) {
  T norm;
  Logger &logger = Logger::get_instance();
  logger.solver_in(monolish_func);

  this->precond.create_precond(A);
  // Algorithm following DOI:10.1007/978-3-319-69953-0_14
  xinout[0] = 1.0;
  xinout[1] = -1.0;
  blas::nrm2(xinout, norm);
  blas::scal(1.0 / norm, xinout);
  monolish::matrix::Dense<T> wxp(3, A.get_col());
  monolish::matrix::Dense<T> twxp(A.get_col(), 3);
  monolish::matrix::Dense<T> WXP(3, A.get_row());
  monolish::view1D<monolish::matrix::Dense<T>, T> w(wxp, 0, 1 * A.get_row());
  monolish::view1D<monolish::matrix::Dense<T>, T> x(wxp, 1 * A.get_row(),
                                             2 * A.get_row());
  monolish::view1D<monolish::matrix::Dense<T>, T> p(wxp, 2 * A.get_row(),
                                             3 * A.get_row());
  monolish::view1D<monolish::matrix::Dense<T>, T> W(WXP, 0, 1 * A.get_row());
  monolish::view1D<monolish::matrix::Dense<T>, T> X(WXP, 1 * A.get_row(),
                                             2 * A.get_row());
  monolish::view1D<monolish::matrix::Dense<T>, T> P(WXP, 2 * A.get_row(),
                                             3 * A.get_row());
  monolish::vector<T> vtmp1(A.get_row());
  monolish::vector<T> vtmp2(A.get_row());

  if (A.get_device_mem_stat() == true) {
    monolish::util::send(wxp, twxp, WXP, vtmp1, vtmp2, xinout);
  }

  blas::copy(xinout, x);
  // X = A x
  blas::matvec(A, x, X);
  // mu = (x, X)
  T mu;
  blas::dot(x, X, mu);
  // w = X - mu x
  blas::axpyz(-mu, x, X, w);
  blas::nrm2(w, norm);
  blas::scal(1.0 / norm, w);

  // B singular flag
  bool is_singular = false;
  // W = A w
  blas::matvec(A, w, W);
  matrix::Dense<T> Sam(3, 3);
  matrix::Dense<T> Sbm(3, 3);
  vector<T> lambda(Sam.get_col());
  if (A.get_device_mem_stat() == true) {
    monolish::util::send(Sam, Sbm, lambda);
  }

  for (std::size_t iter = 0; iter < this->get_maxiter(); iter++) {
    if (iter == 0 || is_singular) {
      // It is intended not to resize actual memory layout
      // and just use the beginning part of 
      // (i.e. not touching {Sam,Sbm,wxp,twxp,WXP}.{val,nnz})
      Sam.set_col(2);
      Sam.set_row(2);
      Sbm.set_col(2);
      Sbm.set_row(2);
      wxp.set_row(2);
      twxp.set_col(2);
      WXP.set_row(2);
    }
    if (A.get_device_mem_stat() == true) {
      wxp.nonfree_recv();
    }
    twxp.transpose(wxp);
    if (A.get_device_mem_stat() == true) {
      monolish::util::send(twxp, Sam, Sbm, lambda);
    }
    // Sa = { w, x, p }^T { W, X, P }
    //    = { w, x, p }^T A { w, x, p }
    blas::matmul(WXP, twxp, Sam);
    // Sb = { w, x, p }^T { w, x, p }
    blas::matmul(wxp, twxp, Sbm);

    if (A.get_device_mem_stat() == true) {
      Sam.nonfree_recv();
      Sbm.nonfree_recv();
    }
    // Generalized Eigendecomposition
    //   Sa v = lambda Sb v
    const char jobz = 'V';
    const char uplo = 'L';
    int info = internal::lapack::sygvd(Sam, Sbm, lambda, 1, &jobz, &uplo);
    // info==6 means order 3 of B is not positive definite, similar to step 0.
    // therefore we set is_singular flag to true and restart the iteration step.
    if (info == 6) {
      if (this->get_print_rhistory()) {
        *this->rhistory_stream << iter + 1 << "\t"
                               << "singular; restart the step" << std::endl;
      }
      is_singular = true;
      iter--;
      continue;
    } else if (info != 0) {
      throw std::runtime_error("internal LAPACK sygvd returned error");
    }
    if (A.get_device_mem_stat() == true) {
      monolish::util::recv(Sam, lambda);
    }
    std::size_t index = 0;
    l = lambda[index];

    // extract b which satisfies Aprime b = lambda_min b
    monolish::vector<T> b(Sam.get_col());
    Sam.row(index, b);

    if (iter == 0 || is_singular) {
      // x = b[0] w + b[1] x, normalize
      // p = b[0] w         , normalize
      blas::scal(b[0], w);
      blas::copy(w, p);
      blas::xpay(b[1], p, x);

      // X = b[0] W + b[1] X, normalize with x
      // P = b[0] W         , normalize with p
      blas::scal(b[0], W);
      blas::copy(W, P);
      blas::xpay(b[1], P, X);
    } else {
      // x = b[0] w + b[1] x + b[2] p, normalize
      // p = b[0] w          + b[2] p, normalize
      blas::scal(b[0], w);
      blas::xpay(b[2], w, p);
      blas::xpay(b[1], p, x);

      // X = b[0] W + b[1] X + b[2] P, normalize with x
      // P = b[0] W          + b[2] P, normalize with p
      blas::scal(b[0], W);
      blas::xpay(b[2], W, P);
      blas::xpay(b[1], P, X);
    }
    T normp;
    blas::nrm2(p, normp);
    blas::scal(1.0 / normp, p);
    blas::scal(1.0 / normp, P);
    T normx;
    blas::nrm2(x, normx);
    blas::scal(1.0 / normx, x);
    blas::scal(1.0 / normx, X);

    // w = X - lambda x
    blas::axpyz(-l, x, X, w);
    // apply preconditioner
    blas::copy(w, vtmp2);
    this->precond.apply_precond(vtmp2, vtmp1);
    blas::copy(vtmp1, w);

    // residual calculation
    T residual;
    blas::nrm2(w, residual);
    if (this->get_print_rhistory()) {
      *this->rhistory_stream << iter + 1 << "\t" << std::scientific << residual
                             << std::endl;
    }

    // early return when residual is small enough
    if (residual < this->get_tol() && this->get_miniter() < iter + 1) {
      blas::copy(x, xinout);
      logger.solver_out();
      return MONOLISH_SOLVER_SUCCESS;
    }

    if (std::isnan(residual)) {
      blas::copy(x, xinout);
      logger.solver_out();
      return MONOLISH_SOLVER_RESIDUAL_NAN;
    }

    // normalize w
    blas::scal(1.0 / residual, w);
    // W = A w
    blas::matvec(A, w, W);

    // reset is_singular flag
    if (iter == 0 || is_singular) {
      is_singular = false;
      Sam.set_row(3);
      Sam.set_col(3);
      Sbm.set_row(3);
      Sbm.set_col(3);
      wxp.set_row(3);
      twxp.set_col(3);
      WXP.set_row(3);
    }
  }
  blas::copy(x, xinout);
  logger.solver_out();
  return MONOLISH_SOLVER_MAXITER;
}

template int
standard_eigen::LOBPCG<matrix::CRS<double>, double>::monolish_LOBPCG(
    matrix::CRS<double> &A, double &l, vector<double> &x);
template int standard_eigen::LOBPCG<matrix::CRS<float>, float>::monolish_LOBPCG(
    matrix::CRS<float> &A, float &l, vector<float> &x);
// template int
// standard_eigen::LOBPCG<matrix::LinearOperator<double>,
// double>::monolish_LOBPCG(
//     matrix::LinearOperator<double> &A, double &l, vector<double> &x);
// template int
// standard_eigen::LOBPCG<matrix::LinearOperator<float>,
// float>::monolish_LOBPCG(
//     matrix::LinearOperator<float> &A, float &l, vector<float> &x);

template <typename MATRIX, typename T>
int standard_eigen::LOBPCG<MATRIX, T>::solve(MATRIX &A, T &l, vector<T> &x) {
  Logger &logger = Logger::get_instance();
  logger.solver_in(monolish_func);

  int ret = 0;
  if (this->get_lib() == 0) {
    ret = monolish_LOBPCG(A, l, x);
  }

  logger.solver_out();
  return ret; // err code
}

template int standard_eigen::LOBPCG<matrix::CRS<double>, double>::solve(
    matrix::CRS<double> &A, double &l, vector<double> &x);
template int standard_eigen::LOBPCG<matrix::CRS<float>, float>::solve(
    matrix::CRS<float> &A, float &l, vector<float> &x);
// template int
// standard_eigen::LOBPCG<matrix::LinearOperator<double>, double>::solve(
//     matrix::LinearOperator<double> &A, double &l, vector<double> &x);
// template int
// standard_eigen::LOBPCG<matrix::LinearOperator<float>, float>::solve(
//     matrix::LinearOperator<float> &A, float &l, vector<float> &x);

} // namespace monolish
