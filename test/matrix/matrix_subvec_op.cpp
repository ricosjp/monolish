#include "../test_utils.hpp"

template <typename T> void ans_subvec_op(monolish::vector<T> &vec) {

  for (int i = 0; i < vec.size(); i++) {
    vec[i] = vec[i] + 3;
    vec[i] = vec[i] * 2;
    vec[i] = vec[i] - 1;
    vec[i] = vec[i] / 2;
  }
}

template <typename T>
bool test_subvec_row(const size_t M, const size_t N, double tol) {
  monolish::matrix::Dense<T> A(M, N, 10.0); // M*N matrix

  monolish::vector<T> vec(A.get_col(), 2.0);
  monolish::vector<T> ansvec(A.get_col(), 2.0);

  A.row(A.get_row() - 1, ansvec);
  ans_subvec_op(ansvec);

  A.row_add(A.get_row() - 1, 3.0);
  A.row_mul(A.get_row() - 1, 2.0);
  A.row_sub(A.get_row() - 1, 1.0);
  A.row_div(A.get_row() - 1, 2.0);
  A.row(A.get_row() - 1, vec);

  return ans_check<T>(__func__, A.type(), ansvec.data(), vec.data(), vec.size(),
                      tol);
}

template <typename T>
bool test_subvec_col(const size_t M, const size_t N, double tol) {
  monolish::matrix::Dense<T> A(M, N, 10.0); // M*N matrix

  monolish::vector<T> vec(A.get_row(), 2.0);
  monolish::vector<T> ansvec(A.get_row(), 2.0);

  A.col(A.get_col() - 1, ansvec);
  ans_subvec_op(ansvec);

  A.col_add(A.get_col() - 1, 3.0);
  A.col_mul(A.get_col() - 1, 2.0);
  A.col_sub(A.get_col() - 1, 1.0);
  A.col_div(A.get_col() - 1, 2.0);
  A.col(A.get_col() - 1, vec);

  return ans_check<T>(__func__, A.type(), ansvec.data(), vec.data(), vec.size(),
                      tol);
}

template <typename T>
bool test_subvec_diag(const size_t M, const size_t N, double tol) {
  monolish::matrix::Dense<T> A(M, N, 10.0); // M*N matrix

  monolish::vector<T> vec(A.get_row(), 2.0);
  monolish::vector<T> ansvec(A.get_row(), 2.0);

  A.diag(ansvec);
  ans_subvec_op(ansvec);

  A.diag_add(3.0);
  A.diag_mul(2.0);
  A.diag_sub(1.0);
  A.diag_div(2.0);
  A.diag(vec);

  return ans_check<T>(__func__, A.type(), ansvec.data(), vec.data(), vec.size(),
                      tol);
}

int main(int argc, char **argv) {

  if (argc != 3) {
    std::cout << "$1: row, $2: col" << std::endl;
    return 1;
  }

  const size_t M = atoi(argv[1]);
  const size_t N = atoi(argv[2]);

  // monolish::util::set_log_level(3);
  // monolish::util::set_log_filename("./monolish_test_log.txt");

  // row //
  if (test_subvec_row<double>(M, N, 1.0e-8) == false) {
    return 1;
  }
  if (test_subvec_row<float>(M, N, 1.0e-8) == false) {
    return 1;
  }

  // col //
  if (test_subvec_col<double>(M, N, 1.0e-8) == false) {
    return 1;
  }
  if (test_subvec_col<float>(M, N, 1.0e-8) == false) {
    return 1;
  }

  // diag (only square now) //
  if (test_subvec_diag<double>(M, M, 1.0e-8) == false) {
    return 1;
  }
  if (test_subvec_diag<float>(M, M, 1.0e-8) == false) {
    return 1;
  }

  return 0;
}
