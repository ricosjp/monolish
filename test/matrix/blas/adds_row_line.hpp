#include "../../test_utils.hpp"

template <typename T>
void ans_adds_row_line(const monolish::matrix::Dense<T> &A, size_t num,
                       const monolish::vector<T> &mx,
                       monolish::matrix::Dense<T> &C) {
  if (A.get_col() != mx.size()) {
    std::runtime_error("A.row != y.size");
  }

  const T *x = mx.data();
  int M = A.get_row();
  int N = A.get_col();

  for (int j = 0; j < N; j++) {
    C.data()[num * N + j] = A.data()[num * N + j] + x[j];
  }
}

template <typename MAT, typename T>
bool test_send_adds_row_line(const size_t M, const size_t N, double tol) {
  size_t nnzrow = 27;
  if (nnzrow < N) {
    nnzrow = 27;
  } else {
    nnzrow = N - 1;
  }
  size_t line = M - 1;

  monolish::matrix::COO<T> seedA =
      monolish::util::random_structure_matrix<T>(M, N, nnzrow, 1.0);

  MAT A(seedA); // M*N matrix
  MAT C(seedA); // M*N matrix
  monolish::vector<T> x(A.get_col(), 0.0, 1.0, test_random_engine());

  monolish::matrix::Dense<T> AA(seedA);
  monolish::matrix::Dense<T> CC(seedA);
  ans_adds_row_line(AA, line, x, CC);
  monolish::matrix::COO<T> ansC(CC);

  monolish::util::send(A, x, C);
  monolish::blas::adds_row(A, line, x, C);
  C.recv();
  monolish::matrix::COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.data(), ansC.data(),
                      ansC.get_nnz(), tol);
}

template <typename MAT, typename T>
bool test_adds_row_line(const size_t M, const size_t N, double tol) {
  size_t nnzrow = 27;
  if (nnzrow < N) {
    nnzrow = 27;
  } else {
    nnzrow = N - 1;
  }
  size_t line = M - 1;

  monolish::matrix::COO<T> seedA =
      monolish::util::random_structure_matrix<T>(M, N, nnzrow, 1.0);

  MAT A(seedA); // M*N matrix
  MAT C(seedA); // M*N matrix
  monolish::vector<T> x(A.get_col(), 0.0, 1.0, test_random_engine());

  monolish::matrix::Dense<T> AA(seedA);
  monolish::matrix::Dense<T> CC(seedA);
  ans_adds_row_line(AA, line, x, CC);
  monolish::matrix::COO<T> ansC(CC);

  monolish::blas::adds_row(A, line, x, C);
  monolish::matrix::COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.data(), ansC.data(),
                      ansC.get_nnz(), tol);
}
