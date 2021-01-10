#include "../../test_utils.hpp"
#include "../include/monolish_eigen.hpp"
#include "../include/monolish_lapack.hpp"
#include <iostream>

template <typename T>
bool test(const char *file, const int check_ans, const T tol_ev, const T tol_res) {
  int DIM = 100;
  monolish::matrix::COO<T> COO = monolish::util::tridiagonal_toeplitz_matrix<T>(DIM, 11.0, -1.0);
  COO.sort(true);
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
  T exact_result = monolish::util::tridiagonal_toeplitz_matrix_eigenvalue<T>(DIM, 0, 11.0, -1.0);

  monolish::matrix::CRS<T> A(COO);

  T lambda;
  monolish::vector<T> x(A.get_row(), 0.0);

  monolish::eigen::LOBPCG<T> solver;

  solver.set_tol(tol_res);
  solver.set_lib(0);
  solver.set_miniter(0);
  solver.set_maxiter(1000);

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

int main(int argc, char **argv) {

  if (argc != 3) {
    std::cout << "error $1:matrix filename, $2:error check (1/0)" << std::endl;
    return 1;
  }

  char *file = argv[1];
  int check_ans = atoi(argv[2]);

  // monolish::util::set_log_level(3);
  // monolish::util::set_log_filename("./monolish_test_log.txt");

  if (test<double>(file, check_ans, 5.0e-3, 8.0e-2) == false) {
    return 1;
  }
  if (test<float>(file, check_ans, 1.0e-1, 1.0e-0) == false) {
    return 1;
  }

  return 0;
}
