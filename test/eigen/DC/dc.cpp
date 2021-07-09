#include "../../test_utils.hpp"
#include "../include/monolish_eigen.hpp"
#include "../include/monolish_equation.hpp"
#include <iostream>

template <typename T>
bool test_tridiagonal_toeplitz(const int check_ans, const T tol_ev,
                               const T tol_res) {
  int DIM = 100;
  monolish::matrix::COO<T> COO =
      monolish::util::tridiagonal_toeplitz_matrix<T>(DIM, 11.0, -1.0);
  monolish::matrix::Dense<T> A(COO);
  monolish::matrix::Dense<T> Aorig(COO);
  monolish::vector<T> lambda(DIM);

  monolish::eigen::DC<monolish::matrix::Dense<T>, T> solver;
  if (monolish::util::solver_check(solver.solve(A, lambda))) {
    return false;
  }

  if (check_ans == 1) {
    for (std::size_t i = 0; i < DIM; ++i) {
      std::cout << "Toeplitz: " << i << "th" << std::endl;
      // Check eiegnvalues based on analytic results
      T exact_result =
          monolish::util::tridiagonal_toeplitz_matrix_eigenvalue<T>(DIM, i,
                                                                    11.0, -1.0);
      std::string sval = "DC eigenvalue(Toeplitz)";
      if (ans_check<T>(sval, lambda[i], exact_result, tol_ev) == false) {
        return false;
      }
      // Check eigenvectors from |Ax - lambda x|
      monolish::vector<T> x(DIM);
      A.row(i, x);
      monolish::vector<T> tmp(DIM);
      monolish::vector<T> tmp2 = x;
      monolish::blas::matvec(Aorig, tmp2, tmp);
      monolish::blas::scal(lambda[i], x);
      std::string svec = "DC eigenvector(Toeplitz)";
      if (ans_check<T>(svec, x.data(), tmp.data(), x.size(), tol_res) ==
          false) {
        return false;
      }
    }
  }
  return true;
}

template <typename T>
bool test_laplacian_1d(const int check_ans, const T tol_ev, const T tol_res) {
  int DIM = 100;
  monolish::matrix::COO<T> COO = monolish::util::laplacian_matrix_1D<T>(DIM);
  monolish::matrix::Dense<T> A(COO);
  monolish::matrix::Dense<T> Aorig(COO);
  monolish::vector<T> lambda(DIM);

  monolish::eigen::DC<monolish::matrix::Dense<T>, T> solver;
  if (monolish::util::solver_check(solver.solve(A, lambda))) {
    return false;
  }

  if (check_ans == 1) {
    for (std::size_t i = 0; i < DIM; ++i) {
      std::cout << "Laplacian: " << i << "th" << std::endl;
      // Check eiegnvalues based on analytic results
      T exact_result =
          monolish::util::laplacian_matrix_1D_eigenvalue<T>(DIM, i);
      std::string sval = "DC eigenvalue(Laplacian)";
      if (ans_check<T>(sval, lambda[i], exact_result, tol_ev) == false) {
        return false;
      }
      // Check eigenvectors from |Ax - lambda x|
      monolish::vector<T> x(DIM);
      A.row(i, x);
      monolish::vector<T> tmp(DIM);
      monolish::vector<T> tmp2 = x;
      monolish::blas::matvec(Aorig, tmp2, tmp);
      monolish::blas::scal(lambda[i], x);
      std::string svec = "DC eigenvector(Laplacian)";
      if (ans_check<T>(svec, x.data(), tmp.data(), x.size(), tol_res) ==
          false) {
        return false;
      }
    }
  }
  return true;
}

template <typename T>
bool test_toeplitz_plus_hankel(const int check_ans, const T tol_ev,
                               const T tol_res) {
  int DIM = 100;
  monolish::matrix::COO<T> COO_A =
      monolish::util::toeplitz_plus_hankel_matrix<T>(DIM, 1.0, -1.0 / 3.0,
                                                     -1.0 / 6.0);
  monolish::matrix::COO<T> COO_B =
      monolish::util::toeplitz_plus_hankel_matrix<T>(DIM, 11.0 / 20.0,
                                                     13.0 / 60.0, 1.0 / 120.0);
  monolish::matrix::Dense<T> A(COO_A);
  monolish::matrix::Dense<T> Aorig(COO_A);
  monolish::matrix::Dense<T> B(COO_B);
  monolish::matrix::Dense<T> Borig(COO_B);
  monolish::vector<T> lambda(DIM);

  monolish::generalized_eigen::DC<monolish::matrix::Dense<T>, T> solver;
  if (monolish::util::solver_check(solver.solve(A, B, lambda, 1))) {
    return false;
  }

  if (check_ans == 1) {
    for (std::size_t i = 0; i < DIM; ++i) {
      std::cout << "Toeplitz + Hankel: " << i << "th" << std::endl;
      // Check eiegnvalues based on analytic results
      T exact_result =
          monolish::util::toeplitz_plus_hankel_matrix_eigenvalue<T>(
              DIM, i, 1.0, -1.0 / 3.0, -1.0 / 6.0, 11.0 / 20.0, 13.0 / 60.0,
              1.0 / 120.0);
      std::string sval = "DC eigenvalue(Toeplitz plus Hankel)";
      if (ans_check<T>(sval, lambda[i], exact_result, tol_ev) == false) {
        return false;
      }
      // Check eigenvectors from |Ax - lambda B x|
      monolish::vector<T> x(DIM);
      A.row(i, x);
      monolish::vector<T> tmp(DIM);
      monolish::vector<T> tmp2 = x;
      monolish::blas::matvec(Aorig, tmp2, tmp);
      monolish::blas::matvec(Borig, tmp2, x);
      monolish::blas::scal(lambda[i], x);
      std::string svec = "DC eigenvector(Toeplitz plus Hankel)";
      if (ans_check<T>(svec, x.data(), tmp.data(), x.size(), tol_res) ==
          false) {
        return false;
      }
    }
  }
  return true;
}

int main(int argc, char **argv) {

  if (argc != 3) {
    std::cout << "error $1:matrix filename, $2:error check (1/0)" << std::endl;
    return 1;
  }

  // monolish::util::set_log_level(3);
  // monolish::util::set_log_filename("./monolish_test_log.txt");

  char *file = argv[1];
  int check_ans = atoi(argv[2]);

  if (test_tridiagonal_toeplitz<double>(check_ans, 1.0e-8, 1.0e-4) == false) {
    return 1;
  }
  if (test_tridiagonal_toeplitz<float>(check_ans, 1.0e-4, 1.0e-2) == false) {
    return 1;
  }

  if (test_laplacian_1d<double>(check_ans, 1.0e-8, 2.0e-3) == false) {
    return 1;
  }
  if (test_laplacian_1d<float>(check_ans, 1.0e-3, 2.0e-1) == false) {
    return 1;
  }

  if (test_toeplitz_plus_hankel<double>(check_ans, 1.0e-8, 1.0e-4) == false) {
    return 1;
  }
  if (test_toeplitz_plus_hankel<float>(check_ans, 5.0e-3, 1.0e-1) == false) {
    return 1;
  }
  return 0;
}
