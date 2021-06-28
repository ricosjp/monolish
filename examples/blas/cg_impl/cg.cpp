#include "monolish_blas.hpp"
#include "monolish_equation.hpp"
#include "monolish_vml.hpp"
#include <iostream>

/*
This is a program of CG method implemented as an example of monolish::blas and
monolish::vml. The CG method is implemented in monolish::equation, so there is
not need for users to implement it.
*/

template <typename MATRIX, typename Float>
void my_cg(const MATRIX &A, monolish::vector<Float> &x,
           const monolish::vector<Float> &b) {
  monolish::Logger &logger = monolish::Logger::get_instance();
  logger.solver_in(monolish_func);

  Float tol = 1.0e-12;

  monolish::vector<Float> r(A.get_row(), 0.0);
  monolish::vector<Float> p(A.get_row(), 0.0);
  monolish::vector<Float> q(A.get_row(), 0.0);

  monolish::util::send(r, p, q); // sent r, p, q to GPU

  // r = b-Ax
  monolish::blas::matvec(A, x, q);
  monolish::vml::sub(b, q, r);

  // p0 = r0
  monolish::blas::copy(r, p);

  // CG loop, maxiter = A.get_row()
  for (size_t iter = 0; iter < A.get_row(); iter++) {
    monolish::blas::matvec(A, p, q); // q = Ap

    // alpha = (r, r) / (Ap,q)
    auto tmp = monolish::blas::dot(r, r);
    auto alpha = tmp / monolish::blas::dot(p, q);

    monolish::blas::axpy(alpha, p, x); // x = alpha*p + x

    monolish::blas::axpy(-alpha, q, r); // r = -alpha * q + r

    auto beta = monolish::blas::dot(r, r) / tmp; // beta = (r, r)

    monolish::blas::xpay(beta, r, p); // p = r + beta*p

    // check convergence
    auto resid = monolish::blas::nrm2(r);
    std::cout << iter + 1 << ": \t" << resid << std::endl;

    if (resid < tol) {
      logger.solver_out();
      return;
    }

    if (std::isnan(resid)) {
      return;
    }
  }

  logger.solver_out();
}

int main(int argc, char **argv) {

  // output log if you need
  // monolish::util::set_log_level(3);
  // monolish::util::set_log_filename("./monolish_test_log.txt");

  // create matrix from MatrixMarket format file
  // monolish::matrix::COO<double> COO("./sample.mtx");

  // or create tridiagonal toeplitz matrix. diagonal elements is 11,
  // non-diagonal elements are -1.0
  size_t DIM = 100;
  monolish::matrix::COO<double> COO =
      monolish::util::tridiagonal_toeplitz_matrix<double>(DIM, 11.0, -1.0);

  // check tridiagonal toeplitz matrix if you need
  // COO.print_all();

  // create CRS matrix from COO
  monolish::matrix::CRS<double> A(COO);

  std::cout << "===== Matrix informatrion =====" << std::endl;
  std::cout << "# of rows : " << A.get_row() << std::endl;
  std::cout << "# of cols : " << A.get_col() << std::endl;
  std::cout << "# of nnz  : " << A.get_nnz() << std::endl;
  std::cout << "===============================" << std::endl;

  // initial x is rand(0~1)
  monolish::vector<double> x(A.get_row(), 0.0, 1.0);

  // initial b is {1, 1, 1, ...,1}
  monolish::vector<double> b(A.get_row(), 1.0);

  // send A, x, b to GPU device
  monolish::util::send(A, x, b);

  // solve
  my_cg(A, x, b);

  // print vector x if you need.
  // x.print_all("solution.txt");

  return 0;
}
