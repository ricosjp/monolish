#include "../../test_utils.hpp"
#include "../include/monolish_eigen.hpp"
#include "../include/monolish_equation.hpp"
#include "../include/monolish_solver.hpp"
#include <chrono>
#include <iostream>

template <typename T, typename PRECOND>
bool benchmark_SEVP(const char *fileA, const int eignum, const T tol_res) {
  monolish::matrix::COO<T> COO(fileA);
  monolish::matrix::CRS<T> A(COO);
  A.send();

  monolish::vector<T> eigvals(eignum);
  monolish::matrix::Dense<T> eigvecs(eignum, A.get_row());
  for (size_t i = 0; i < eignum; ++i) {
    eigvecs.insert(i, (i * 7) % A.get_row(), 1.0);
  }

  monolish::standard_eigen::LOBPCG<monolish::matrix::CRS<T>, T> solver;

  solver.set_tol(tol_res);
  solver.set_lib(0);
  solver.set_miniter(0);
  solver.set_maxiter(A.get_row());
  solver.set_initvec_scheme(monolish::solver::initvec_scheme::USER);

  // precond setting
  PRECOND precond;
  solver.set_create_precond(precond);
  solver.set_apply_precond(precond);

  solver.set_print_rhistory(true);
  auto start = std::chrono::system_clock::now();
  if (monolish::util::solver_check(solver.solve(A, eigvals, eigvecs))) {
    return false;
  }
  auto end = std::chrono::system_clock::now();
  double sec = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                   .count() /
               1.0e+9;

  std::cerr << "time: " << sec << std::endl;
  return true;
}

template <typename T, typename PRECOND>
bool benchmark_GEVP(const char *fileA, const char *fileB, const int eignum,
                    const T tol_res) {
  monolish::matrix::COO<T> COOA(fileA);
  monolish::matrix::CRS<T> A(COOA);
  monolish::matrix::COO<T> COOB(fileB);
  monolish::matrix::CRS<T> B(COOB);
  A.send();
  B.send();

  monolish::vector<T> eigvals(eignum);
  monolish::matrix::Dense<T> eigvecs(eignum, A.get_row());
  for (size_t i = 0; i < eignum; ++i) {
    eigvecs.insert(i, (i * 7) % A.get_row(), 1.0);
  }

  monolish::generalized_eigen::LOBPCG<monolish::matrix::CRS<T>, T> solver;

  solver.set_tol(tol_res);
  solver.set_lib(0);
  solver.set_miniter(0);
  solver.set_maxiter(A.get_row());
  solver.set_initvec_scheme(monolish::solver::initvec_scheme::USER);

  // precond setting
  PRECOND precond;
  solver.set_create_precond(precond);
  solver.set_apply_precond(precond);

  solver.set_print_rhistory(true);
  auto start = std::chrono::system_clock::now();
  if (monolish::util::solver_check(solver.solve(A, B, eigvals, eigvecs, 1))) {
    return false;
  }
  auto end = std::chrono::system_clock::now();
  double sec = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                   .count() /
               1.0e+9;

  std::cerr << "time: " << sec << std::endl;
  return true;
}

template <typename MATRIX, typename T, typename PRECOND>
bool test_solve(monolish::matrix::COO<T> mat, monolish::vector<T> exact_result,
                const int check_ans, const T tol_ev, const T tol_res,
                const int maxiter, const std::string s) {
  MATRIX A(mat);
  monolish::vector<T> lambda(exact_result.size());
  monolish::matrix::Dense<T> x(exact_result.size(), A.get_row());
  for (size_t i = 0; i < exact_result.size(); ++i) {
    x.insert(i, (i * 7) % A.get_row(), 1.0);
  }
  monolish::util::send(A);

  monolish::eigen::LOBPCG<MATRIX, T> solver;

  solver.set_tol(tol_res);
  solver.set_lib(0);
  solver.set_miniter(0);
  solver.set_maxiter(maxiter);
  solver.set_initvec_scheme(monolish::solver::initvec_scheme::USER);

  // precond setting
  PRECOND precond;
  solver.set_create_precond(precond);
  solver.set_apply_precond(precond);

  solver.set_print_rhistory(true);

  if (monolish::util::solver_check(solver.solve(A, lambda, x))) {
    return false;
  }

  if (check_ans == 1) {
    for (int i = 0; i < lambda.size(); ++i) {
      if (ans_check<T>("LOBPCG(" + s + ")", lambda[i], exact_result[i],
                       tol_ev) == false) {
        return false;
      }
    }
  }
  return true;
}

template <typename MATRIX, typename T, typename PRECOND>
bool test_solve_LinearOperator(monolish::matrix::COO<T> mat,
                               monolish::vector<T> exact_result,
                               const int check_ans, const T tol_ev,
                               const T tol_res, const int maxiter,
                               const std::string s) {
  monolish::matrix::Dense<T> dense_mat(mat);
  monolish::util::send(dense_mat);
  MATRIX A(dense_mat);
  monolish::vector<T> lambda(exact_result.size());
  monolish::matrix::Dense<T> x(exact_result.size(), A.get_row());
  for (size_t i = 0; i < exact_result.size(); ++i) {
    x.insert(i, (i * 7) % A.get_row(), 1.0);
  }

  monolish::eigen::LOBPCG<monolish::matrix::LinearOperator<T>, T> solver;

  solver.set_tol(tol_res);
  solver.set_lib(0);
  solver.set_miniter(0);
  solver.set_maxiter(maxiter);
  solver.set_initvec_scheme(monolish::solver::initvec_scheme::USER);

  // precond setting
  PRECOND precond;
  solver.set_create_precond(precond);
  solver.set_apply_precond(precond);

  solver.set_print_rhistory(true);

  if (monolish::util::solver_check(solver.solve(A, lambda, x))) {
    return false;
  }

  if (check_ans == 1) {
    for (int i = 0; i < lambda.size(); ++i) {
      if (ans_check<T>("LOBPCG(" + s + ")", lambda[i], exact_result[i],
                       tol_ev) == false) {
        return false;
      }
    }
  }
  return true;
}

template <typename MATRIX, typename T, typename PRECOND>
bool test_solve_GEVP(monolish::matrix::COO<T> matA,
                     monolish::matrix::COO<T> matB,
                     monolish::vector<T> exact_result, const int check_ans,
                     const T tol_ev, const T tol_res, const std::string s) {
  MATRIX A(matA);
  MATRIX B(matB);
  monolish::vector<T> lambda(exact_result.size());
  monolish::matrix::Dense<T> x(exact_result.size(), A.get_row());
  for (size_t i = 0; i < exact_result.size(); ++i) {
    x.insert(i, (i * 7) % A.get_row(), 1.0);
  }
  monolish::util::send(A, B);

  monolish::generalized_eigen::LOBPCG<MATRIX, T> solver;

  solver.set_tol(tol_res);
  solver.set_lib(0);
  solver.set_miniter(0);
  solver.set_maxiter(A.get_row());
  solver.set_initvec_scheme(monolish::solver::initvec_scheme::USER);

  // precond setting
  PRECOND precond;
  solver.set_create_precond(precond);
  solver.set_apply_precond(precond);

  solver.set_print_rhistory(true);

  if (monolish::util::solver_check(solver.solve(A, B, lambda, x, 1))) {
    return false;
  }

  if (check_ans == 1) {
    for (int i = 0; i < lambda.size(); ++i) {
      if (ans_check<T>("LOBPCG(" + s + ")", lambda[i], exact_result[i],
                       tol_ev) == false) {
        return false;
      }
    }
  }
  return true;
}

template <typename MATRIX, typename T, typename PRECOND>
bool test_solve_GEVP_LinearOperator(monolish::matrix::COO<T> matA,
                                    monolish::matrix::COO<T> matB,
                                    monolish::vector<T> exact_result,
                                    const int check_ans, const T tol_ev,
                                    const T tol_res, const std::string s) {
  monolish::matrix::Dense<T> dense_matA(matA), dense_matB(matB);
  monolish::util::send(dense_matA, dense_matB);
  MATRIX A(dense_matA);
  MATRIX B(dense_matB);
  monolish::vector<T> lambda(exact_result.size());
  monolish::matrix::Dense<T> x(exact_result.size(), A.get_row());
  for (size_t i = 0; i < exact_result.size(); ++i) {
    x.insert(i, (i * 7) % A.get_row(), 1.0);
  }

  monolish::generalized_eigen::LOBPCG<MATRIX, T> solver;

  solver.set_tol(tol_res);
  solver.set_lib(0);
  solver.set_miniter(0);
  solver.set_maxiter(A.get_row());
  solver.set_initvec_scheme(monolish::solver::initvec_scheme::USER);

  // precond setting
  PRECOND precond;
  solver.set_create_precond(precond);
  solver.set_apply_precond(precond);

  solver.set_print_rhistory(true);

  if (monolish::util::solver_check(solver.solve(A, B, lambda, x, 1))) {
    return false;
  }

  if (check_ans == 1) {
    for (int i = 0; i < lambda.size(); ++i) {
      if (ans_check<T>("LOBPCG(" + s + ")", lambda[i], exact_result[i],
                       tol_ev) == false) {
        return false;
      }
    }
  }
  return true;
}

template <typename MATRIX, typename T, typename PRECOND>
bool test_tridiagonal_toeplitz(const int check_ans, const T tol_ev,
                               const T tol_res) {
  int DIM = 100;
  monolish::matrix::COO<T> COO =
      monolish::util::tridiagonal_toeplitz_matrix<T>(DIM, 10.0, -1.0);
  monolish::vector<T> exact_result(2);
  for (int i = 0; i < exact_result.size(); ++i) {
    exact_result[i] = monolish::util::tridiagonal_toeplitz_matrix_eigenvalue<T>(
        DIM, i, 10.0, -1.0);
  }

  return test_solve<MATRIX, T, PRECOND>(COO, exact_result, check_ans, tol_ev,
                                        tol_res * DIM, DIM,
                                        "Tridiagonal Toeplitz");
}

template <typename MATRIX, typename T, typename PRECOND>
bool test_tridiagonal_toeplitz_LinearOperator(const int check_ans,
                                              const T tol_ev, const T tol_res) {
  int DIM = 100;
  monolish::matrix::COO<T> COO =
      monolish::util::tridiagonal_toeplitz_matrix<T>(DIM, 10.0, -1.0);
  monolish::vector<T> exact_result(2);
  for (int i = 0; i < exact_result.size(); ++i) {
    exact_result[i] = monolish::util::tridiagonal_toeplitz_matrix_eigenvalue<T>(
        DIM, i, 10.0, -1.0);
  }

  return test_solve_LinearOperator<MATRIX, T, PRECOND>(
      COO, exact_result, check_ans, tol_ev, tol_res * DIM, DIM,
      "Tridiagonal Toeplitz");
}

template <typename MATRIX, typename T, typename PRECOND>
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
  monolish::vector<T> exact_result(2);
  for (int i = 0; i < exact_result.size(); ++i) {
    exact_result[i] = monolish::util::toeplitz_plus_hankel_matrix_eigenvalue<T>(
        DIM, i, 1.0, -1.0 / 3.0, -1.0 / 6.0, 11.0 / 20.0, 13.0 / 60.0,
        1.0 / 120.0);
  }

  return test_solve_GEVP<MATRIX, T, PRECOND>(COO_A, COO_B, exact_result,
                                             check_ans, tol_ev, tol_res * DIM,
                                             "Toeplitz + Hankel");
}

template <typename MATRIX, typename T, typename PRECOND>
bool test_toeplitz_plus_hankel_LinearOperator(const int check_ans,
                                              const T tol_ev, const T tol_res) {
  int DIM = 100;
  monolish::matrix::COO<T> COO_A =
      monolish::util::toeplitz_plus_hankel_matrix<T>(DIM, 1.0, -1.0 / 3.0,
                                                     -1.0 / 6.0);
  monolish::matrix::COO<T> COO_B =
      monolish::util::toeplitz_plus_hankel_matrix<T>(DIM, 11.0 / 20.0,
                                                     13.0 / 60.0, 1.0 / 120.0);

  // Check eiegnvalues based on analytic results
  monolish::vector<T> exact_result(2);
  for (int i = 0; i < exact_result.size(); ++i) {
    exact_result[i] = monolish::util::toeplitz_plus_hankel_matrix_eigenvalue<T>(
        DIM, i, 1.0, -1.0 / 3.0, -1.0 / 6.0, 11.0 / 20.0, 13.0 / 60.0,
        1.0 / 120.0);
  }

  return test_solve_GEVP_LinearOperator<MATRIX, T, PRECOND>(
      COO_A, COO_B, exact_result, check_ans, tol_ev, tol_res * DIM,
      "Toeplitz + Hankel");
}

int main(int argc, char **argv) {
  char *fileA;
  char *fileB;
  int check_ans;
  bool is_benchmark = false;
  switch (argc) {
  case 1:
    check_ans = 1;
    break;
  case 2:
    check_ans = atoi(argv[1]);
    break;
  case 3:
    check_ans = atoi(argv[2]);
    fileA = argv[1];
    is_benchmark = true;
    break;
  case 4:
    check_ans = atoi(argv[3]);
    fileA = argv[1];
    fileB = argv[2];
    is_benchmark = true;
    break;
  default:
    std::cout << "error $1:matrix A filename (optional), $2:matrix B filename "
                 "(optional), $3:error check (1/0) (optional)"
              << std::endl;
  }

  if (is_benchmark) {
    if (argc == 3) {
      if (!benchmark_SEVP<double, monolish::equation::none<
                                      monolish::matrix::CRS<double>, double>>(
              fileA, 10, 1.0e-4)) {
        return 1;
      }
      if (!benchmark_SEVP<float, monolish::equation::none<
                                     monolish::matrix::CRS<float>, float>>(
              fileA, 10, 1.0e-4)) {
        return 2;
      }
      if (!benchmark_SEVP<double, monolish::equation::Jacobi<
                                      monolish::matrix::CRS<double>, double>>(
              fileA, 10, 1.0e-4)) {
        return 3;
      }
      if (!benchmark_SEVP<float, monolish::equation::Jacobi<
                                     monolish::matrix::CRS<float>, float>>(
              fileA, 10, 1.0e-4)) {
        return 4;
      }
      return 0;
    } else if (argc == 4) {
      if (!benchmark_GEVP<double, monolish::equation::none<
                                      monolish::matrix::CRS<double>, double>>(
              fileA, fileB, 10, 1.0e-4)) {
        return 1;
      }
      if (!benchmark_GEVP<float, monolish::equation::none<
                                     monolish::matrix::CRS<float>, float>>(
              fileA, fileB, 10, 1.0e-4)) {
        return 2;
      }
      if (!benchmark_GEVP<double, monolish::equation::Jacobi<
                                      monolish::matrix::CRS<double>, double>>(
              fileA, fileB, 10, 1.0e-4)) {
        return 3;
      }
      if (!benchmark_GEVP<float, monolish::equation::Jacobi<
                                     monolish::matrix::CRS<float>, float>>(
              fileA, fileB, 10, 1.0e-4)) {
        return 4;
      }
      return 0;
    }
    return -1;
  }

  // monolish::util::set_log_level(3);
  // monolish::util::set_log_filename("./monolish_test_log.txt");

  std::cout << "CRS" << std::endl;

  if (test_tridiagonal_toeplitz<
          monolish::matrix::CRS<double>, double,
          monolish::equation::none<monolish::matrix::CRS<double>, double>>(
          check_ans, 1.0e-3, 1.0e-5) == false) {
    return 1;
  }
  if (test_tridiagonal_toeplitz<
          monolish::matrix::CRS<float>, float,
          monolish::equation::none<monolish::matrix::CRS<float>, float>>(
          check_ans, 1.0e-3, 1.0e-5) == false) {
    return 1;
  }
  if (test_tridiagonal_toeplitz<
          monolish::matrix::CRS<double>, double,
          monolish::equation::Jacobi<monolish::matrix::CRS<double>, double>>(
          check_ans, 1.0e-3, 1.0e-5) == false) {
    return 1;
  }
  if (test_tridiagonal_toeplitz<
          monolish::matrix::CRS<float>, float,
          monolish::equation::Jacobi<monolish::matrix::CRS<float>, float>>(
          check_ans, 1.0e-3, 1.0e-5) == false) {
    return 1;
  }

  if (test_toeplitz_plus_hankel<
          monolish::matrix::CRS<double>, double,
          monolish::equation::none<monolish::matrix::CRS<double>, double>>(
          check_ans, 5.0e-4, 1.0e-6) == false) {
    return 1;
  }
  if (test_toeplitz_plus_hankel<
          monolish::matrix::CRS<float>, float,
          monolish::equation::none<monolish::matrix::CRS<float>, float>>(
          check_ans, 5.0e-4, 1.0e-6) == false) {
    return 1;
  }
  if (test_toeplitz_plus_hankel<
          monolish::matrix::CRS<double>, double,
          monolish::equation::Jacobi<monolish::matrix::CRS<double>, double>>(
          check_ans, 5.0e-4, 1.0e-6) == false) {
    return 1;
  }
  if (test_toeplitz_plus_hankel<
          monolish::matrix::CRS<float>, float,
          monolish::equation::Jacobi<monolish::matrix::CRS<float>, float>>(
          check_ans, 5.0e-4, 1.0e-6) == false) {
    return 1;
  }

  std::cout << "Dense" << std::endl;

  if (test_tridiagonal_toeplitz<
          monolish::matrix::Dense<double>, double,
          monolish::equation::none<monolish::matrix::Dense<double>, double>>(
          check_ans, 1.0e-3, 1.0e-5) == false) {
    return 1;
  }
  if (test_tridiagonal_toeplitz<
          monolish::matrix::Dense<float>, float,
          monolish::equation::none<monolish::matrix::Dense<float>, float>>(
          check_ans, 1.0e-3, 1.0e-5) == false) {
    return 1;
  }
  if (test_tridiagonal_toeplitz<
          monolish::matrix::Dense<double>, double,
          monolish::equation::Jacobi<monolish::matrix::Dense<double>, double>>(
          check_ans, 1.0e-3, 1.0e-5) == false) {
    return 1;
  }
  if (test_tridiagonal_toeplitz<
          monolish::matrix::Dense<float>, float,
          monolish::equation::Jacobi<monolish::matrix::Dense<float>, float>>(
          check_ans, 1.0e-3, 1.0e-5) == false) {
    return 1;
  }

  if (test_toeplitz_plus_hankel<
          monolish::matrix::Dense<double>, double,
          monolish::equation::none<monolish::matrix::Dense<double>, double>>(
          check_ans, 5.0e-4, 1.0e-6) == false) {
    return 1;
  }
  if (test_toeplitz_plus_hankel<
          monolish::matrix::Dense<float>, float,
          monolish::equation::none<monolish::matrix::Dense<float>, float>>(
          check_ans, 5.0e-4, 1.0e-6) == false) {
    return 1;
  }
  if (test_toeplitz_plus_hankel<
          monolish::matrix::Dense<double>, double,
          monolish::equation::Jacobi<monolish::matrix::Dense<double>, double>>(
          check_ans, 5.0e-4, 1.0e-6) == false) {
    return 1;
  }
  if (test_toeplitz_plus_hankel<
          monolish::matrix::Dense<float>, float,
          monolish::equation::Jacobi<monolish::matrix::Dense<float>, float>>(
          check_ans, 5.0e-4, 1.0e-6) == false) {
    return 1;
  }

  std::cout << "LinearOperator" << std::endl;

  if (test_tridiagonal_toeplitz_LinearOperator<
          monolish::matrix::LinearOperator<double>, double,
          monolish::equation::none<monolish::matrix::LinearOperator<double>,
                                   double>>(check_ans, 1.0e-3, 1.0e-5) ==
      false) {
    return 1;
  }
  if (test_tridiagonal_toeplitz_LinearOperator<
          monolish::matrix::LinearOperator<float>, float,
          monolish::equation::none<monolish::matrix::LinearOperator<float>,
                                   float>>(check_ans, 1.0e-3, 1.0e-5) ==
      false) {
    return 1;
  }
  if (test_tridiagonal_toeplitz_LinearOperator<
          monolish::matrix::LinearOperator<double>, double,
          monolish::equation::Jacobi<monolish::matrix::LinearOperator<double>,
                                     double>>(check_ans, 1.0e-3, 1.0e-5) ==
      false) {
    return 1;
  }
  if (test_tridiagonal_toeplitz_LinearOperator<
          monolish::matrix::LinearOperator<float>, float,
          monolish::equation::Jacobi<monolish::matrix::LinearOperator<float>,
                                     float>>(check_ans, 1.0e-3, 1.0e-5) ==
      false) {
    return 1;
  }

  if (test_toeplitz_plus_hankel_LinearOperator<
          monolish::matrix::LinearOperator<double>, double,
          monolish::equation::none<monolish::matrix::LinearOperator<double>,
                                   double>>(check_ans, 5.0e-4, 1.0e-6) ==
      false) {
    return 1;
  }
  if (test_toeplitz_plus_hankel_LinearOperator<
          monolish::matrix::LinearOperator<float>, float,
          monolish::equation::none<monolish::matrix::LinearOperator<float>,
                                   float>>(check_ans, 5.0e-4, 1.0e-6) ==
      false) {
    return 1;
  }
  if (test_toeplitz_plus_hankel_LinearOperator<
          monolish::matrix::LinearOperator<double>, double,
          monolish::equation::Jacobi<monolish::matrix::LinearOperator<double>,
                                     double>>(check_ans, 5.0e-4, 1.0e-6) ==
      false) {
    return 1;
  }
  if (test_toeplitz_plus_hankel_LinearOperator<
          monolish::matrix::LinearOperator<float>, float,
          monolish::equation::Jacobi<monolish::matrix::LinearOperator<float>,
                                     float>>(check_ans, 5.0e-4, 1.0e-6) ==
      false) {
    return 1;
  }

  return 0;
}
