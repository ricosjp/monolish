#include "../../include/monolish_blas.hpp"
#include "../../include/monolish_vml.hpp"
#include "../../include/monolish_eigen.hpp"
#include "../internal/lapack/monolish_lapack.hpp"
#include "../internal/monolish_internal.hpp"

namespace monolish {

template <typename MATRIX, typename T>
int standard_eigen::LOBPCG<MATRIX, T>::monolish_LOBPCG(
    MATRIX &A, vector<T> &l, matrix::Dense<T> &xinout) {
  Logger &logger = Logger::get_instance();
  logger.solver_in(monolish_func);

  // size consistency check
  // the size of xinout is (m, A.get_col())
  if (A.get_col() != xinout.get_col()) {
    throw std::runtime_error("error, A.col != x.col");
  }
  if (l.size() != xinout.get_row()) {
    throw std::runtime_error("error, l.size != x.row");
  }

  this->precond.create_precond(A);

  // Algorithm following DOI:10.1007/978-3-319-69953-0_14
  // m is the number of eigenpairs to compute
  const std::size_t m = l.size();
  // n is the dimension of the original space
  const std::size_t n = A.get_col();

  // Internal memory
  // currently 6m+(6m+3m+2) vectors are used
  matrix::Dense<T> wxp(3 * m, n);
  matrix::Dense<T> WXP(3 * m, n);
  // TODO: wxp_p and WXP_p are not needed when m=1
  matrix::Dense<T> wxp_p(3 * m, n);
  matrix::Dense<T> WXP_p(3 * m, n);
  // TODO: twxp is not needed when transpose matmul is supported
  matrix::Dense<T> twxp(n, 3 * m);
  // TODO: vtmp1 and vtmp2 are not needed when in-place preconditioner is used
  //       and view1D is supported by preconditioners
  monolish::vector<T> vtmp1(A.get_row());
  monolish::vector<T> vtmp2(A.get_row());

  // view1Ds for calculation
  std::vector<view1D<matrix::Dense<T>, T>> w;
  std::vector<view1D<matrix::Dense<T>, T>> x;
  std::vector<view1D<matrix::Dense<T>, T>> p;
  std::vector<view1D<matrix::Dense<T>, T>> W;
  std::vector<view1D<matrix::Dense<T>, T>> X;
  std::vector<view1D<matrix::Dense<T>, T>> P;
  std::vector<view1D<matrix::Dense<T>, T>> wp;
  std::vector<view1D<matrix::Dense<T>, T>> xp;
  std::vector<view1D<matrix::Dense<T>, T>> pp;
  std::vector<view1D<matrix::Dense<T>, T>> Wp;
  std::vector<view1D<matrix::Dense<T>, T>> Xp;
  std::vector<view1D<matrix::Dense<T>, T>> Pp;
  for (std::size_t i = 0; i < m; ++i) {
    w.push_back(view1D<matrix::Dense<T>, T>(wxp, i * n, (i + 1) * n));
    x.push_back(view1D<matrix::Dense<T>, T>(wxp, (m + i) * n, (m + i + 1) * n));
    p.push_back(view1D<matrix::Dense<T>, T>(wxp, (2 * m + i) * n, (2 * m + i + 1) * n));
    W.push_back(view1D<matrix::Dense<T>, T>(WXP, i * n, (i + 1) * n));
    X.push_back(view1D<matrix::Dense<T>, T>(WXP, (m + i) * n, (m + i + 1) * n));
    P.push_back(view1D<matrix::Dense<T>, T>(WXP, (2 * m + i) * n, (2 * m + i + 1) * n));
    wp.push_back(view1D<matrix::Dense<T>, T>(wxp_p, i * n, (i + 1) * n));
    xp.push_back(view1D<matrix::Dense<T>, T>(wxp_p, (m + i) * n, (m + i + 1) * n));
    pp.push_back(view1D<matrix::Dense<T>, T>(wxp_p, (2 * m + i) * n, (2 * m + i + 1) * n));
    Wp.push_back(view1D<matrix::Dense<T>, T>(WXP_p, i * n, (i + 1) * n));
    Xp.push_back(view1D<matrix::Dense<T>, T>(WXP_p, (m + i) * n, (m + i + 1) * n));
    Pp.push_back(view1D<matrix::Dense<T>, T>(WXP_p, (2 * m + i) * n, (2 * m + i + 1) * n));
  }

  // Preparing initial input to be orthonormal to each other
  for (std::size_t i = 0; i < m; ++i) {
    T norm;
    x[i][i] = 1.0;
    x[i][i + 1] = -1.0;
    blas::nrm2(x[i], norm);
    blas::scal(1.0 / norm, x[i]);
  }

  if (A.get_device_mem_stat() == true) {
    monolish::util::send(wxp, twxp, WXP, vtmp1, vtmp2);
  }

  for (std::size_t i = 0; i < m; ++i) {
    // X = A x
    blas::matvec(A, x[i], X[i]);
    // mu = (x, X)
    T mu = blas::dot(x[i], X[i]);
    // w = X - mu x, normalize
    blas::axpyz(-mu, x[i], X[i], w[i]);
    T norm = blas::nrm2(w[i]);
    blas::scal(1.0 / norm, w[i]);
    // W = A w
    blas::matvec(A, w[i], W[i]);
  }

  // B singular flag
  bool is_singular = false;
  
  matrix::Dense<T> Sam(3 * m, 3 * m);
  matrix::Dense<T> Sbm(3 * m, 3 * m);
  vector<T> lambda(3 * m);

  for (std::size_t iter = 0; iter < this->get_maxiter(); iter++) {
    if (A.get_device_mem_stat() == true) {
      monolish::util::send(Sam, Sbm, lambda);
    }
    if (iter == 0 || is_singular) {
      // It is intended not to resize actual memory layout
      // and just use the beginning part of
      // (i.e. not touching {Sam,Sbm,wxp,twxp,WXP}.{val,nnz})
      Sam.set_col(2 * m);
      Sam.set_row(2 * m);
      Sbm.set_col(2 * m);
      Sbm.set_row(2 * m);
      wxp.set_row(2 * m);
      wxp_p.set_row(2 * m);
      twxp.set_col(2 * m);
      WXP.set_row(2 * m);
      WXP_p.set_row(2 * m);
    }
    if (A.get_device_mem_stat() == true) {
      wxp.nonfree_recv();
    }
    twxp.transpose(wxp);
    blas::copy(wxp, wxp_p);
    if (A.get_device_mem_stat() == true) {
      twxp.send();
    }
    // prepare previous step results
    wxp_p = wxp;
    WXP_p = WXP;
    // Sa = { w, x, p }^T { W, X, P }
    //    = { w, x, p }^T A { w, x, p }
    blas::matmul(WXP, twxp, Sam);
    // Sb = { w, x, p }^T { w, x, p }
    blas::matmul(wxp, twxp, Sbm);

    // workaround on GPU; synchronize Sam, Sbm
    if (A.get_device_mem_stat() == true) {
      Sam.nonfree_recv();
      Sbm.nonfree_recv();
    }
    // Generalized Eigendecomposition
    //   Sa v = lambda Sb v
    const char jobz = 'V';
    const char uplo = 'L';
    int info = internal::lapack::sygvd(Sam, Sbm, lambda, 1, &jobz, &uplo);
    // 5m < info <= 6m means order 3 of B is not positive definite, similar to step 0.
    // therefore we set is_singular flag to true and restart the iteration step.
    if (5 * m < info && info <= 6 * m) {
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

    vector<T> residual(m);
    for (std::size_t i = 0; i < m; ++i) {
      // copy current eigenvalue results
      l[i] = lambda[i];

      // extract eigenvector of Sa v = lambda Sb v
      vector<T> b(3 * m);
      Sam.row(i, b);
      
      if (iter == 0 || is_singular) {
        // b[2m+1]...b[3m-1] is not calculated so explicitly set to 0
        for (std::size_t j = 2 * m + 1; j < 3 * m; ++j) {
          b[j] = 0.0;
        }
      }

      for (std::size_t j = 0; j < m; ++j) {
        // x[i] = \Sum_j b[j] w[j] + b[m+j] x[j] + b[2m+j] p[j], normalize
        // p[i] = \Sum_j b[j] w[j]               + b[2m+j] p[j], normalize
        if (j == 0) {
          blas::scal(b[j], w[i]);
          blas::xpay(b[2 * m + j], w[i], p[i]);
          blas::xpay(b[m + j], p[i], x[i]);
        } else {
          blas::axpy(b[j], wp[j], w[i]); // w[i] can be used as temporary
          blas::axpy(b[2 * m + j], pp[j], p[i]);
          blas::axpy(b[j], wp[j], x[i]);
          blas::axpy(b[2 * m + j], pp[j], x[i]);
          blas::axpy(b[m + j], xp[j], x[i]);
        }

        // X[i] = \Sum_j b[j] W[j] + b[m+j] X[j] + b[2m+j] P[j], normalize with x
        // P[i] = \Sum_j b[j] W[j]               + b[2m+j] P[j], normalize with p
        if (j == 0) {
          blas::scal(b[j], W[i]);
          blas::xpay(b[2 * m + j], W[i], P[i]);
          blas::xpay(b[m + j], P[i], X[i]);
        } else {
          blas::axpy(b[j], Wp[j], W[i]); // w[i] can be used as temporary
          blas::axpy(b[2 * m + j], Pp[j], P[i]);
          blas::axpy(b[j], Wp[j], X[i]);
          blas::axpy(b[2 * m + j], Pp[j], X[i]);
          blas::axpy(b[m + j], Xp[j], X[i]);
        }
      }
    
      T normp;
      blas::nrm2(p[i], normp);
      blas::scal(1.0 / normp, p[i]);
      blas::scal(1.0 / normp, P[i]);
      T normx;
      blas::nrm2(x[i], normx);
      blas::scal(1.0 / normx, x[i]);
      blas::scal(1.0 / normx, X[i]);

      // w[i] = X[i] - lambda x[i]
      blas::axpyz(-l[i], x[i], X[i], w[i]);
      // apply preconditioner
      blas::copy(w[i], vtmp2);
      this->precond.apply_precond(vtmp2, vtmp1);
      blas::copy(vtmp1, w[i]);

      // residual calculation
      blas::nrm2(w[i], residual[i]);
      if (this->get_print_rhistory()) {
        *this->rhistory_stream << iter + 1 << "\t" << i << "\t" << std::scientific << residual[i]
                               << std::endl;
      }
      if (std::isnan(residual[i])) {
        for (std::size_t i = 0; i < m; ++i) {
          view1D<matrix::Dense<T>, T> xinout_i(xinout, i * n, (i + 1) * n);
          blas::copy(x[i], xinout_i);
        }
        logger.solver_out();
        return MONOLISH_SOLVER_RESIDUAL_NAN;
      }
      // normalize w
      blas::scal(1.0 / residual[i], w[i]);
      // W = A w
      blas::matvec(A, w[i], W[i]);
    }

    // early return when residual is small enough
    if (vml::max(residual) < this->get_tol() && this->get_miniter() < iter + 1) {
      for (std::size_t i = 0; i < m; ++i) {
        view1D<matrix::Dense<T>, T> xinout_i(xinout, i * n, (i + 1) * n);
        blas::copy(x[i], xinout_i);
      }
      logger.solver_out();
      return MONOLISH_SOLVER_SUCCESS;
    }

    // reset is_singular flag
    if (iter == 0 || is_singular) {
      is_singular = false;
      Sam.set_row(3 * m);
      Sam.set_col(3 * m);
      Sbm.set_row(3 * m);
      Sbm.set_col(3 * m);
      wxp.set_row(3 * m);
      wxp_p.set_row(3 * m);
      twxp.set_col(3 * m);
      WXP.set_row(3 * m);
      WXP_p.set_row(3 * m);
    }
  }
  for (std::size_t i = 0; i < m; ++i) {
    view1D<matrix::Dense<T>, T> xinout_i(xinout, i * n, (i + 1) * n);
    blas::copy(x[i], xinout_i);
  }
  logger.solver_out();
  return MONOLISH_SOLVER_MAXITER;
}

template int
standard_eigen::LOBPCG<matrix::CRS<double>, double>::monolish_LOBPCG(
    matrix::CRS<double> &A, vector<double> &l, matrix::Dense<double> &x);
template int standard_eigen::LOBPCG<matrix::CRS<float>, float>::monolish_LOBPCG(
    matrix::CRS<float> &A, vector<float> &l, matrix::Dense<float> &x);
// template int
// standard_eigen::LOBPCG<matrix::LinearOperator<double>,
// double>::monolish_LOBPCG(
//     matrix::LinearOperator<double> &A, double &l, vector<double> &x);
// template int
// standard_eigen::LOBPCG<matrix::LinearOperator<float>,
// float>::monolish_LOBPCG(
//     matrix::LinearOperator<float> &A, float &l, vector<float> &x);

template <typename MATRIX, typename T>
int standard_eigen::LOBPCG<MATRIX, T>::solve(MATRIX &A, vector<T> &l, matrix::Dense<T> &x) {
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
    matrix::CRS<double> &A, vector<double> &l, matrix::Dense<double> &x);
template int standard_eigen::LOBPCG<matrix::CRS<float>, float>::solve(
    matrix::CRS<float> &A, vector<float> &l, matrix::Dense<float> &x);
// template int
// standard_eigen::LOBPCG<matrix::LinearOperator<double>, double>::solve(
//     matrix::LinearOperator<double> &A, double &l, vector<double> &x);
// template int
// standard_eigen::LOBPCG<matrix::LinearOperator<float>, float>::solve(
//     matrix::LinearOperator<float> &A, float &l, vector<float> &x);

} // namespace monolish
