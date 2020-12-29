#include "../../include/monolish_blas.hpp"
#include "../../include/monolish_lapack.hpp"
#include "../../include/monolish_equation.hpp"
#include "../monolish_internal.hpp"

namespace monolish {

template <typename T>
int
equation::eig<T>::monolish_LOBPCG(matrix::CRS<T> const &A,
                                  T w,
                                  vector<T> &w) {
  T eps = 1e-6;
  std::size_t maxiter = 10000;

  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  monolish::vector<T> x(A.get_row());
  x[0] = 1.0;
  monolish::vector<T> r(A.get_row());
  monolish::vector<T> p(A.get_row());

  // r = x - A x;
  monolish::vector<T> vtmp(A.get_row());
  blas::matvec(A, x, vtmp);
  r = x - vtmp;

  do {
    // V = { x, r, p }
    std::vector<monolish::vector<T>> V = { x, r, p };

    // Aprime = V^T A V
    //   Atmp^T = A V
    matrix::Dense<T> Atmp(3, A.get_row());
    for (std::size_t i = 0; i < V.size(); ++i) {
      monolish::vector<T> vtmp(A.get_row());
      blas::matvec(A, V[i], vtmp);
      Atmp.row_add(i, vtmp);
    }
    //   Aprime^T = Atmp V
    matrix::Dense<T> Aprime(3, 3);
    blas::matmul(Atmp, V, Aprime);
    Aprime.transpose;

    // Eigendecomposition of Aprime
    //   (Aprime overwritten)
    monolish::vector<T> lambda(3);
    lapack::syev('V', 'U', Aprime, lambda);

    // extract b which satisfies Aprime b = lambda_min b
    monolish::vector<T> b(Aprime.get_row());
    Aprime.col(0, b);

    // x = V b
    blas::matvec(V, b, x);
    // r = A x - lambda_min x
    blas::matvec(A, x, r);
    monolish::vector<T> xtmp(x);
    blas::scal(lambda[0], xtmp);
    blas::vecsub(r, xtmp, r);
    // p = { 0, rp, pp } b
    monolish::vector<T> zeros(Aprime.get_row());
    std::vector<monolish::vector<T>> Vtmp = { zeros, rp, pp };
    blas::matvec(Vtmp, b, p);

    // prepare next V = { x, r, p }
    blas::nrm2(V, residual);
    ++iter;
  } while (residual < eps || iter >= maxiter);
  logger.func_out();
}
