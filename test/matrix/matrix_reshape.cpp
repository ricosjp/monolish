#include "../test_utils.hpp"

template <typename MAT, typename T>
bool test_send_reshape(const size_t M, const size_t N, double tol) {

  size_t nnzrow = 27;
  if (nnzrow < N) {
    nnzrow = 27;
  } else {
    nnzrow = N - 1;
  }
  monolish::matrix::COO<T> seedA =
      monolish::util::random_structure_matrix<T>(M, N, nnzrow, 1.0);
  MAT A(seedA);

  monolish::util::send(A);
  A.reshape(N, M);
  if (A.get_row() != N || A.get_col() != M) {
    std::cout << "reshape error, reshapeA.row = " << A.get_row()
              << ", reshapeA.col = " << A.get_col() << std::endl;
    return false;
  }

  std::cout << __func__ << "(" << A.type() << ")"
            << " : pass" << std::endl;

  return true;
}

template <typename MAT, typename T>
bool test_reshape(const size_t M, const size_t N, double tol) {

  size_t nnzrow = 27;
  if (nnzrow < N) {
    nnzrow = 27;
  } else {
    nnzrow = N - 1;
  }
  monolish::matrix::COO<T> seedA =
      monolish::util::random_structure_matrix<T>(M, N, nnzrow, 1.0);
  MAT A(seedA);

  A.reshape(N, M);
  if (A.get_row() != N || A.get_col() != M) {
    std::cout << "reshape error, reshapeA.row = " << A.get_row()
              << ", reshapeA.col = " << A.get_col() << std::endl;
    return false;
  }

  std::cout << __func__ << "(" << A.type() << ")"
            << " : pass" << std::endl;

  return true;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cout << "$1: row, $2: col" << std::endl;
    return 1;
  }

  const size_t M = atoi(argv[1]);
  const size_t N = atoi(argv[2]);

  std::cout << "M=" << M << ", N=" << N << std::endl;

  if (test_send_reshape<monolish::matrix::Dense<double>, double>(
          M, N, 1.0e-6) == false) {
    return 1;
  }
  if (test_send_reshape<monolish::matrix::Dense<float>, float>(M, N, 1.0e-6) ==
      false) {
    return 1;
  }
  if (test_reshape<monolish::matrix::Dense<double>, double>(M, N, 1.0e-6) ==
      false) {
    return 1;
  }
  if (test_reshape<monolish::matrix::Dense<float>, float>(M, N, 1.0e-6) ==
      false) {
    return 1;
  }

  return 0;
}
