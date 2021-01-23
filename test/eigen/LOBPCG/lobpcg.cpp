#include "../../test_utils.hpp"
#include "../include/monolish_eigen.hpp"
#include "../include/monolish_equation.hpp"
#include <iostream>

template <typename T, typename PRECOND>
bool test_solve(monolish::matrix::COO<T> mat, const T exact_result,
                const int check_ans, const T tol_ev, const T tol_res) {
  monolish::matrix::CRS<T> A(mat);
  T lambda;
  monolish::vector<T> x(A.get_row());
  monolish::util::send(A);

  monolish::eigen::LOBPCG<monolish::matrix::CRS<T>, T> solver;

  solver.set_tol(tol_res);
  solver.set_lib(0);
  solver.set_miniter(0);
  solver.set_maxiter(1000);

  // precond setting
  PRECOND precond;
  solver.set_create_precond(precond);
  solver.set_apply_precond(precond);

  solver.set_print_rhistory(true);

  if (monolish::util::solver_check(solver.solve(A, lambda, x))) {
    return false;
  }

  if (check_ans == 1) {
    if (ans_check<T>("LOBPCG", lambda, exact_result, tol_ev) == false) {
      return false;
    }
  }
  return true;
}

template <typename T, typename PRECOND>
bool test_tridiagonal_toeplitz(const int check_ans, const T tol_ev,
                               const T tol_res) {
  int DIM = 100;
  monolish::matrix::COO<T> COO =
      monolish::util::tridiagonal_toeplitz_matrix<T>(DIM, 11.0, -1.0);
  // for (std::size_t i = 0; i < COO.get_row(); ++i) {
  //   for (std::size_t j = 0; j < COO.get_col(); ++j) {
  //     if (i == j) { COO.insert(i, j, 2.0); }
  //     if (i - j == 1 || j - i == 1) { COO.insert(i, j, -1.0); }
  //     if (i == 0 && j == DIM - 1) { COO.insert(i, j, -1.0); }
  //     if (j == 0 && i == DIM - 1) { COO.insert(i, j, -1.0); }
  //   }
  // }

  // Calculate exact result by solving full eigenproblem
  // monolish::matrix::Dense<T> AD(COO);
  // monolish::vector<T> ld(AD.get_row());
  // const char jobz = 'V';
  // const char uplo = 'U';
  // bool bl = monolish::lapack::syev(&jobz, &uplo, AD, ld);
  // if (!bl) {
  //   throw std::runtime_error("LAPACK syev failed");
  // }
  // T exact_result = ld[0];

  // Calculate exact eigenvalue from analytic solution
  T exact_result = monolish::util::tridiagonal_toeplitz_matrix_eigenvalue<T>(
      DIM, 0, 11.0, -1.0);

  return test_solve<T, PRECOND>(COO, exact_result, check_ans, tol_ev, tol_res);
}

template <typename T, typename PRECOND>
bool test_laplacian_1d(const int check_ans, const T tol_ev, const T tol_res) {
  int DIM = 10000;
  monolish::matrix::COO<T> COO = monolish::util::laplacian_matrix_1D<T>(DIM);
  T exact_result = monolish::util::laplacian_matrix_1D_eigenvalue<T>(DIM, 0);

  return test_solve<T, PRECOND>(COO, exact_result, check_ans, tol_ev, tol_res);
}

int main(int argc, char **argv) {

  if (argc != 3) {
    std::cout << "error $1:matrix filename, $2:error check (1/0)" << std::endl;
    return 1;
  }

  char *file = argv[1];
  int check_ans = atoi(argv[2]);

  // monolish::util::set_log_level(3);
  // monolish::util::set_log_filename("./monolish_test_log.txt");

  if (test_tridiagonal_toeplitz<
          double,
          monolish::equation::none<monolish::matrix::CRS<double>, double>>(
          check_ans, 1.0e-8, 1.0e-4) == false) {
    return 1;
  }
  if (test_tridiagonal_toeplitz<
          float, monolish::equation::none<monolish::matrix::CRS<float>, float>>(
          check_ans, 1.0e-2, 1.0e-2) == false) {
    return 1;
  }
  if (test_tridiagonal_toeplitz<
          double,
          monolish::equation::Jacobi<monolish::matrix::CRS<double>, double>>(
          check_ans, 1.0e-8, 1.0e-4) == false) {
    return 1;
  }
  if (test_tridiagonal_toeplitz<
          float,
          monolish::equation::Jacobi<monolish::matrix::CRS<float>, float>>(
          check_ans, 1.0e-2, 1.0e-2) == false) {
    return 1;
  }

  if (test_laplacian_1d<double, monolish::equation::none<
                                    monolish::matrix::CRS<double>, double>>(
          check_ans, 1.0e-3, 5.0e-3) == false) {
    return 1;
  }
  if (test_laplacian_1d<
          float, monolish::equation::none<monolish::matrix::CRS<float>, float>>(
          check_ans, 1.0e-2, 1.0e-2) == false) {
    return 1;
  }
  if (test_laplacian_1d<double, monolish::equation::Jacobi<
                                    monolish::matrix::CRS<double>, double>>(
          check_ans, 1.0e-3, 5.0e-3) == false) {
    return 1;
  }
  if (test_laplacian_1d<float, monolish::equation::Jacobi<
                                   monolish::matrix::CRS<float>, float>>(
          check_ans, 1.0e-2, 1.0e-2) == false) {
    return 1;
  }
  return 0;
}
