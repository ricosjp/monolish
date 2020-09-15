#include "../test_utils.hpp"
#include "monolish_blas.hpp"

#define FUNC "subvec_op"
#define DENSE_PERF 2 * M *N / time / 1.0e+9
#define CRS_PERF 2 * M *nnzrow / time / 1.0e+9

template <typename T> void get_ans(monolish::vector<T> &vec) {

  for (int i = 0; i < vec.size(); i++) {
    vec[i] = vec[i] + 3;
    vec[i] = vec[i] * 2;
    vec[i] = vec[i] - 1;
    vec[i] = vec[i] / 2;
  }
}

template <typename T> bool test(const size_t M, const size_t N, double tol) {
  monolish::matrix::Dense<T> A(M, N, 10.0); // M*N matrix

  //////////////////////////
  monolish::vector<T> vec(A.get_col(), 2.0);
  monolish::vector<T> ansvec(A.get_col(), 2.0);

  A.row(A.get_row() - 1, ansvec);
  get_ans(ansvec);

  A.row_add(A.get_row() - 1, 3.0);
  A.row_mul(A.get_row() - 1, 2.0);
  A.row_sub(A.get_row() - 1, 1.0);
  A.row_div(A.get_row() - 1, 2.0);
  A.row(A.get_row() - 1, vec);

  if (ans_check<T>(ansvec.data(), vec.data(), vec.size(), tol) == false) {
    std::cout << "=row=" << std::endl;
    std::cout << "==ans==" << std::endl;
    ansvec.print_all();
    std::cout << "==result==" << std::endl;
    vec.print_all();
    return false;
  };

  //////////////////////////

  monolish::vector<T> vec2(A.get_row(), 2.0);
  monolish::vector<T> ansvec2(A.get_row(), 2.0);

  A.col(A.get_col() - 1, ansvec2);
  get_ans(ansvec2);

  A.col_add(A.get_col() - 1, 3.0);
  A.col_mul(A.get_col() - 1, 2.0);
  A.col_sub(A.get_col() - 1, 1.0);
  A.col_div(A.get_col() - 1, 2.0);
  A.col(A.get_col() - 1, vec2);

  if (ans_check<T>(ansvec2.data(), vec2.data(), vec2.size(), tol) == false) {
    std::cout << "=col=" << std::endl;
    std::cout << "==ans==" << std::endl;
    ansvec2.print_all();
    std::cout << "==result==" << std::endl;
    vec2.print_all();

    return false;
  };

  //////////////////////////

  //     size_t n = std::min(A.get_row(), A.get_col());
  //     monolish::vector<T> vec3(n, 2.0);
  //     monolish::vector<T> ansvec3(n, 2.0);
  //
  //     A.diag(ansvec3);
  //     get_ans(ansvec3);
  //
  //     A.diag_add(3.0);
  //     A.diag_mul(2.0);
  //     A.diag_sub(1.0);
  //     A.diag_div(2.0);
  //
  //     A.diag(vec3);
  //
  //     if (ans_check<T>(ansvec3.data(), vec3.data(), vec.size(), tol) ==
  //     false) {
  //         std::cout << "=diag=" << std::endl;
  //         std::cout << "==ans==" << std::endl;
  //         ansvec3.print_all();
  //         std::cout << "==result==" << std::endl;
  //         vec3.print_all();
  //         return false;
  //     };
  std::cout << "Pass in " << get_type<T>() << " precision" << std::endl;

  return true;
}

int main(int argc, char **argv) {

  if (argc != 4) {
    std::cout << "error $1: precision (double or float) $2: row, $3: col"
              << std::endl;
    return 1;
  }

  const size_t M = atoi(argv[2]);
  const size_t N = atoi(argv[3]);

  // monolish::util::set_log_level(3);
  // monolish::util::set_log_filename("./monolish_test_log.txt");

  if (strcmp(argv[1], "double") == 0) {
    if (test<double>(M, N, 1.0e-8) == false) {
      return 1;
    }
  }

  if (strcmp(argv[1], "float") == 0) {
    if (test<float>(M, N, 1.0e-2) == false) {
      return 1;
    }
  }
  return 0;
}
