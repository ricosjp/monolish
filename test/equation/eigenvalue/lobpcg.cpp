#include "../../test_utils.hpp"
#include "../include/monolish_eigenvalue.hpp"
#include "../include/monolish_lapack.hpp"
#include <iostream>

template <typename T>
bool test(const char *file, const int check_ans, const T tol) {
  const int DIM = 12;
  monolish::matrix::COO<T> COO(DIM, DIM);
  for (std::size_t i = 0; i < COO.get_row(); ++i) {
    for (std::size_t j = 0; j < COO.get_col(); ++j) {
      T val;
      if (i < j) {
        val = 1.0 + i;
      } else {
        val = 1.0 + j;
      }
      COO.insert(i, j, val);
    }
  }
  // Calculate exact result by solving full eigenproblem
  monolish::matrix::Dense<T> AD(COO);
  monolish::vector<T> ld(AD.get_row());
  const char jobz = 'V';
  const char uplo = 'U';
  bool bl = monolish::lapack::syev(&jobz, &uplo, AD, ld);
  if (!bl) { throw std::runtime_error("LAPACK syev failed"); }

  // Calculate exact eigenvalue from analytic solution
  T exact_result = 1.0 / (2.0 * (1.0 - std::cos(M_PI * (2 * (DIM-1) + 1) / (2 * DIM + 1))));

  monolish::matrix::CRS<T> A(COO);

  T lambda;
  monolish::vector<T> x(A.get_row());

  monolish::eigenvalue::monolish_LOBPCG(A, lambda, x);

  if (ans_check<T>("LOBPCG", lambda, exact_result, tol) == false) {
    return false;
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

  if (test<double>(file, check_ans, 1.0e-8) == false) {
    return 1;
  }
  // if (test<float>(file, check_ans, 1.0e-4) == false) {
  //   return 1;
  // }

  return 0;
}
