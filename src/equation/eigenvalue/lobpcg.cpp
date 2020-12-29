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
  monolish::vector<T> vtmp1(A.get_row());
  monolish::vector<T> vtmp2(A.get_row());
  blas::matvec(A, x, vtmp);
  r = x - vtmp;

  do {
    // V = { x, r, p }
    std::vector<monolish::vector<T>> V = { x, r, p };

    // Aprime = V^T A V
    //   Atmp^T = A V
    matrix::Dense<T> Atmp(3, A.get_row());
    for (std::size_t i = 0; i < V.size(); ++i) {
      blas::matvec(A, V[i], vtmp1);
      Atmp.row_add(i, vtmp1);
    }
    //   Aprime^T = Atmp V
    matrix::Dense<T> Aprime(3, 3);
    for (std::size_t i = 0; i < V.size(); ++i) {
      blas::matvec(Atmp, V[i]);
      Aprime.row_add(i, vtmp1);
    }
    Aprime.transpose;

    // Eigendecomposition of Aprime
    //   (Aprime overwritten)
    monolish::vector<T> lambda(3);
    lapack::syev('V', 'U', Aprime, lambda);

    // extract b which satisfies Aprime b = lambda_min b
    monolish::vector<T> b(Aprime.get_row());
    Aprime.col(0, b);

    // x = b[0] x + b[1] r + b[2] p
    blas::scal(b[0], x, x);
    blas::scal(b[1], r, vtmp1);
    blas::vecadd(x, vtmp1, x);
    blas::scal(b[2], p, vtmp1);
    blas::vecadd(x, vtmp1, x);
    
    // r = A x - lambda_min x
    vtmp2 = r;
    blas::matvec(A, x, r);
    blas::scal(lambda[0], vtmp1);
    blas::vecsub(r, vtmp1, r);

    // p = b[1] rp + b[2] pp
    blas::scal(b[1], vtmp2, vtmp2);
    blas::scal(b[2], p, p);
    blas::vecadd(vtmp2, p, p);

    // residual calculation
    blas::nrm2(r, residual);
    ++iter;
  } while (residual < eps || iter >= maxiter);
  logger.func_out();
}
