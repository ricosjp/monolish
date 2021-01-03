#include "../../../include/monolish_blas.hpp"
#include "../../../include/monolish_lapack.hpp"
#include "../../../include/monolish_eigenvalue.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {

template <typename T>
int
eigenvalue::monolish_LOBPCG(matrix::CRS<T> const &A,
                            T& l,
                            monolish::vector<T> &x) {
  int info = 0;
  T eps = 1e-6;
  T residual = 1.0;
  T norm;
  std::size_t iter = 0;
  std::size_t maxiter = 1000;

  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // Algorithm following DOI:10.1007/978-3-319-69953-0_14 
  x[0] = 1.0; // This is normalized
  monolish::vector<T> w(A.get_row());
  monolish::vector<T> p(A.get_row());
  monolish::vector<T> X(A.get_row());
  monolish::vector<T> W(A.get_row());
  monolish::vector<T> P(A.get_row());

  // X = A x
  blas::matvec(A, x, X);
  // mu = (x, X)
  T mu;
  blas::dot(x, X, mu);
  // w = X - mu x
  monolish::vector<T> vtmp1(A.get_row());
  vtmp1 = x;
  blas::scal(mu, vtmp1);
  blas::vecsub(X, vtmp1, w);
  blas::nrm2(w, norm);
  blas::scal(1.0 / norm, w);

  // 

  do {
    // W = A w
    blas::matvec(A, w, W);

    // Sa = { w, x, p }^T { W, X, P }
    //    = { w, x, p }^T A { w, x, p }
    std::vector<T> Sa(9);
    blas::dot(w, W, Sa[0]);
    blas::dot(w, X, Sa[1]);
    blas::dot(w, P, Sa[2]);
    blas::dot(x, W, Sa[3]);
    blas::dot(x, X, Sa[4]);
    blas::dot(x, P, Sa[5]);
    blas::dot(p, W, Sa[6]);
    blas::dot(p, X, Sa[7]);
    blas::dot(p, P, Sa[8]);
    if (iter == 0) { Sa[8] = 1.0; }
    matrix::Dense<T> Sam(3, 3, Sa);
    Sam.print_all();

    // Sb = { w, x, p }^T { w, x, p }
    std::vector<T> Sb(9);
    blas::dot(w, w, Sb[0]);
    blas::dot(w, x, Sb[1]);
    blas::dot(w, p, Sb[2]);
    blas::dot(x, w, Sb[3]);
    blas::dot(x, x, Sb[4]);
    blas::dot(x, p, Sb[5]);
    blas::dot(p, w, Sb[6]);
    blas::dot(p, x, Sb[7]);
    blas::dot(p, p, Sb[8]);
    if (iter == 0) { Sb[8] = 1.0; }
    matrix::Dense<T> Sbm(3, 3, Sb);

    // Generalized Eigendecomposition
    //   Sa v = lambda Sb v
    monolish::vector<T> lambda(3);
    const char jobz = 'V';
    const char uplo = 'U';
    bool bl = lapack::sygv(1, &jobz, &uplo, Sam, Sbm, lambda);
    if (!bl) { throw std::runtime_error("LAPACK sygv failed"); }
    std::size_t index = 0;
    l = lambda[index];
    // if (std::abs(l) < eps) { index = 1; l = lambda[index]; }
    std::cout << l << std::endl;

    // extract b which satisfies Aprime b = lambda_min b
    monolish::vector<T> b(Sam.get_col());
    Sam.col(index, b);

    // x = b[0] w + b[1] x + b[2] p, normalize
    // p = b[0] w          + b[2] p, normalize
    blas::scal(b[0], w);
    blas::scal(b[1], x);
    blas::scal(b[2], p);
    blas::vecadd(w, p, p);
    blas::vecadd(x, p, x);
    T normp;
    blas::nrm2(p, normp);
    blas::scal(1.0 / normp, p);
    T normx;
    blas::nrm2(x, normx);
    blas::scal(1.0 / normx, x);
    
    // X = b[0] W + b[1] X + b[2] P, normalize with x
    // P = b[0] W          + b[2] P, normalize with p
    blas::scal(b[0], W);
    blas::scal(b[1], X);
    blas::scal(b[2], P);
    blas::vecadd(W, P, P);
    blas::vecadd(X, P, X);
    blas::scal(1.0 / normp, P);
    blas::scal(1.0 / normx, X);

    // w = X - lambda x
    vtmp1 = x;
    blas::scal(l, vtmp1);
    blas::vecsub(X, vtmp1, w);

    // residual calculation and normalize
    blas::nrm2(w, residual);
    blas::scal(1.0 / residual, w);
    ++iter;
  } while (residual > eps && iter < maxiter);
  logger.func_out();
  return info;
}

template int eigenvalue::monolish_LOBPCG<double>(matrix::CRS<double> const &A,
                                                 double& l,
                                                 monolish::vector<double> &x);

} // namespace monolish
