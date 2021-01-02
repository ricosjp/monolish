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
  std::size_t iter = 0;
  std::size_t maxiter = 10000;

  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  x[0] = 1.0;
  monolish::vector<T> r(A.get_row());
  monolish::vector<T> p(A.get_row());

  // r = x - A x;
  monolish::vector<T> vtmp1(A.get_row());
  monolish::vector<T> vtmp2(A.get_row());
  blas::matvec(A, x, vtmp1);
  blas::vecsub(x, vtmp1, r);

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
    monolish::vector<T> vtmp3(3);
    for (std::size_t i = 0; i < V.size(); ++i) {
      blas::matvec(Atmp, V[i], vtmp3);
      Aprime.row_add(i, vtmp3);
    }
    Aprime.transpose();

    // Eigendecomposition of Aprime
    //   (Aprime overwritten)
    monolish::vector<T> lambda(3);
    const char jobz = 'V';
    const char uplo = 'U';
    bool bl = lapack::syev(&jobz, &uplo, Aprime, lambda);
    if (!bl) { throw std::runtime_error("LAPACK syev failed"); }
    l = lambda[0];

    // extract b which satisfies Aprime b = lambda_min b
    monolish::vector<T> b(Aprime.get_row());
    Aprime.col(0, b);

    // x = b[0] x + b[1] r + b[2] p
    blas::scal(b[0], x);
    vtmp1 = r;
    blas::scal(b[1], vtmp1);
    blas::vecadd(x, vtmp1, x);
    vtmp1 = p;
    blas::scal(b[2], vtmp1);
    blas::vecadd(x, vtmp1, x);

    // r = A x - lambda_min x
    vtmp2 = r;
    blas::matvec(A, x, r);
    blas::scal(l, vtmp1);
    blas::vecsub(r, vtmp1, r);

    // p = b[1] rp + b[2] pp
    blas::scal(b[1], vtmp2);
    blas::scal(b[2], p);
    blas::vecadd(vtmp2, p, p);

    // residual calculation
    blas::nrm2(r, residual);
    ++iter;
  } while (residual > eps && iter < maxiter);
  logger.func_out();
  return info;
}

template int eigenvalue::monolish_LOBPCG<double>(matrix::CRS<double> const &A,
                                                 double& l,
                                                 monolish::vector<double> &x);

} // namespace monolish
