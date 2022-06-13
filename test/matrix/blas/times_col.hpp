#include "../../test_utils.hpp"

template <typename T>
void ans_times_col(const monolish::matrix::Dense<T> &A,
                   const monolish::vector<T> &mx,
                   monolish::matrix::Dense<T> &C) {
  if (A.get_row() != mx.size()) {
    std::runtime_error("A.row != y.size");
  }

  const T *x = mx.data();
  int M = A.get_row();
  int N = A.get_col();

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      C.val[i * N + j] = A.val[i * N + j] * x[i];
    }
  }
}

template <typename MAT, typename T>
bool test_send_times_col(const size_t M, const size_t N, double tol) {
  size_t nnzrow = 27;
  if (nnzrow < N) {
    nnzrow = 27;
  } else {
    nnzrow = N - 1;
  }

  monolish::matrix::COO<T> seedA =
      monolish::util::random_structure_matrix<T>(M, N, nnzrow, 1.0);

  MAT A(seedA); // M*N matrix
  MAT C(seedA); // M*N matrix
  monolish::vector<T> x(A.get_row(), 0.0, 1.0, test_random_engine());

  monolish::matrix::Dense<T> AA(seedA);
  monolish::matrix::Dense<T> CC(seedA);
  ans_times_col(AA, x, CC);
  monolish::matrix::COO<T> ansC(CC);

  monolish::util::send(A, x, C);
  monolish::blas::times_col(A, x, C);
  C.recv();
  monolish::matrix::COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.val.data(), ansC.val.data(),
                      ansC.get_nnz(), tol);
}

template <typename MAT, typename T>
bool test_times_col(const size_t M, const size_t N, double tol) {
  size_t nnzrow = 27;
  if (nnzrow < N) {
    nnzrow = 27;
  } else {
    nnzrow = N - 1;
  }

  monolish::matrix::COO<T> seedA =
      monolish::util::random_structure_matrix<T>(M, N, nnzrow, 1.0);

  MAT A(seedA); // M*N matrix
  MAT C(seedA); // M*N matrix
  monolish::vector<T> x(A.get_row(), 0.0, 1.0, test_random_engine());

  monolish::matrix::Dense<T> AA(seedA);
  monolish::matrix::Dense<T> CC(seedA);
  ans_times_col(AA, x, CC);
  monolish::matrix::COO<T> ansC(CC);

  monolish::blas::times_col(A, x, C);
  monolish::matrix::COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.val.data(), ansC.val.data(),
                      ansC.get_nnz(), tol);
}
