#include "../../test_utils.hpp"
#include "../include/monolish_eigen.hpp"
#include "../include/monolish_equation.hpp"
#include <iostream>

template <typename T, typename PRECOND>
bool test_solve(monolish::matrix::COO<T> mat, const T exact_result,
                const int check_ans, const T tol_ev, const T tol_res,
                const std::string s) {
  monolish::matrix::CRS<T> A(mat);
  T lambda;
  monolish::vector<T> x(A.get_row());
  monolish::util::send(A);

  monolish::eigen::LOBPCG<monolish::matrix::CRS<T>, T> solver;

  solver.set_tol(tol_res);
  solver.set_lib(0);
  solver.set_miniter(0);
  solver.set_maxiter(A.get_row());

  // precond setting
  PRECOND precond;
  solver.set_create_precond(precond);
  solver.set_apply_precond(precond);

  solver.set_print_rhistory(true);

  if (monolish::util::solver_check(solver.solve(A, lambda, x))) {
    return false;
  }

  if (check_ans == 1) {
    if (ans_check<T>("LOBPCG(" + s + ")", lambda, exact_result, tol_ev) ==
        false) {
      return false;
    }
  }
  return true;
}

template <typename T, typename PRECOND>
bool test_solve_GEVP(monolish::matrix::COO<T> matA, monolish::matrix::COO<T> matB, const T exact_result,
                     const int check_ans, const T tol_ev, const T tol_res,
                     const std::string s) {
  monolish::matrix::CRS<T> A(matA);
  monolish::matrix::CRS<T> B(matB);
  T lambda;
  monolish::vector<T> x(A.get_row());
  monolish::util::send(A, B);

  monolish::generalized_eigen::LOBPCG<monolish::matrix::CRS<T>, T> solver;

  solver.set_tol(tol_res);
  solver.set_lib(0);
  solver.set_miniter(0);
  solver.set_maxiter(A.get_row());

  // precond setting
  PRECOND precond;
  solver.set_create_precond(precond);
  solver.set_apply_precond(precond);

  solver.set_print_rhistory(true);

  if (monolish::util::solver_check(solver.solve(A, B, lambda, x, 1))) {
    return false;
  }

  if (check_ans == 1) {
    if (ans_check<T>("LOBPCG(" + s + ")", lambda, exact_result, tol_ev) ==
        false) {
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
      monolish::util::tridiagonal_toeplitz_matrix<T>(DIM, 10.0, -1.0);
  T exact_result = monolish::util::tridiagonal_toeplitz_matrix_eigenvalue<T>(
      DIM, 0, 10.0, -1.0);

  return test_solve<T, PRECOND>(COO, exact_result, check_ans, tol_ev,
                                tol_res * DIM, "Tridiagonal Toeplitz");
}

template <typename T, typename PRECOND>
bool test_frank(const int check_ans, const T tol_ev, const T tol_res) {
  int DIM = 100;
  monolish::matrix::COO<T> COO = monolish::util::frank_matrix<T>(DIM);
  T exact_result = monolish::util::frank_matrix_eigenvalue<T>(DIM, 0);

  return test_solve<T, PRECOND>(COO, exact_result, check_ans, tol_ev,
                                tol_res * DIM, "Frank");
}

template <typename T, typename PRECOND>
bool test_toeplitz_plus_hankel(const int check_ans, const T tol_ev,
                               const T tol_res) {
  int DIM = 100;
  monolish::matrix::COO<T> COO_A =
      monolish::util::toeplitz_plus_hankel_matrix<T>(DIM, 1.0, -1.0 / 3.0,
                                                     -1.0 / 6.0);
  monolish::matrix::COO<T> COO_B =
      monolish::util::toeplitz_plus_hankel_matrix<T>(DIM, 11.0 / 20.0,
                                                     13.0 / 60.0, 1.0 / 120.0);

  // Check eiegnvalues based on analytic results
  T exact_result =
      monolish::util::toeplitz_plus_hankel_matrix_eigenvalue<T>(
          DIM, 0, 1.0, -1.0 / 3.0, -1.0 / 6.0, 11.0 / 20.0, 13.0 / 60.0,
          1.0 / 120.0);

  return test_solve_GEVP<T, PRECOND>(COO_A, COO_B, exact_result, check_ans, tol_ev, tol_res * DIM, "Toeplitz + Hankel");
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
          check_ans, 3.0e-4, 1.0e-5) == false) {
    return 1;
  }
  if (test_tridiagonal_toeplitz<
          float, monolish::equation::none<monolish::matrix::CRS<float>, float>>(
          check_ans, 3.0e-4, 1.0e-5) == false) {
    return 1;
  }
  if (test_tridiagonal_toeplitz<
          double,
          monolish::equation::Jacobi<monolish::matrix::CRS<double>, double>>(
          check_ans, 3.0e-4, 1.0e-5) == false) {
    return 1;
  }
  if (test_tridiagonal_toeplitz<
          float,
          monolish::equation::Jacobi<monolish::matrix::CRS<float>, float>>(
          check_ans, 3.0e-4, 1.0e-5) == false) {
    return 1;
  }

  if (test_frank<double, monolish::equation::none<monolish::matrix::CRS<double>,
                                                  double>>(check_ans, 2.0e-1,
                                                           1.0e-5) == false) {
    return 1;
  }
  if (test_frank<float,
                 monolish::equation::none<monolish::matrix::CRS<float>, float>>(
          check_ans, 2.0e-1, 1.0e-5) == false) {
    return 1;
  }
  if (test_frank<double, monolish::equation::Jacobi<
                             monolish::matrix::CRS<double>, double>>(
          check_ans, 2.0e-1, 1.0e-5) == false) {
    return 1;
  }
  if (test_frank<float, monolish::equation::Jacobi<monolish::matrix::CRS<float>,
                                                   float>>(check_ans, 2.0e-1,
                                                           1.0e-5) == false) {
    return 1;
  }

  if (test_toeplitz_plus_hankel<double, monolish::equation::none<monolish::matrix::CRS<double>,
                                                  double>>(check_ans, 2.0e-1,
                                                           1.0e-5) == false) {
    return 1;
  }
  if (test_toeplitz_plus_hankel<float,
                 monolish::equation::none<monolish::matrix::CRS<float>, float>>(
          check_ans, 2.0e-1, 1.0e-5) == false) {
    return 1;
  }
  if (test_toeplitz_plus_hankel<double, monolish::equation::Jacobi<
                             monolish::matrix::CRS<double>, double>>(
          check_ans, 2.0e-1, 1.0e-5) == false) {
    return 1;
  }
  if (test_toeplitz_plus_hankel<float, monolish::equation::Jacobi<monolish::matrix::CRS<float>,
                                                   float>>(check_ans, 2.0e-1,
                                                           1.0e-5) == false) {
    return 1;
  }
return 0;
}
