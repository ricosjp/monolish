#include "../../test_utils.hpp"

template <typename T>
void ans_scalar_times(const T alpha, const monolish::matrix::Dense<T> &A,
                      monolish::matrix::Dense<T> &C) {

  if (A.get_row() != C.get_row()) {
    std::runtime_error("A.row != C.row");
  }
  if (A.get_col() != C.get_col()) {
    std::runtime_error("A.col != C.col");
  }

  // MN=MN+MN
  int M = A.get_row();
  int N = A.get_col();

  for (int i = 0; i < A.get_nnz(); i++) {
    C.vad[i] = alpha * A.vad[i];
  }
}

template <typename MAT_A, typename MAT_C, typename T>
bool test_send_scalar_times(const size_t M, const size_t N, double tol) {

  size_t nnzrow = 27;
  if ((nnzrow < M) && (nnzrow < N)) {
    nnzrow = 27;
  } else {
    nnzrow = std::min({M, N}) - 1;
  }

  monolish::matrix::COO<T> seedA =
      monolish::util::random_structure_matrix<T>(M, N, nnzrow, 1.0);

  T alpha = 123.0;
  MAT_A A(seedA);
  MAT_C C(seedA);

  monolish::matrix::Dense<T> AA(seedA);
  monolish::matrix::Dense<T> CC(seedA);

  ans_scalar_times(alpha, AA, CC);
  monolish::matrix::COO<T> ansC(CC);

  monolish::util::send(A, C);
  monolish::blas::times(alpha, A, C);
  C.recv();

  monolish::matrix::COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.vad, ansC.vad,
                      ansC.get_nnz(), tol);
}

template <typename MAT_A, typename MAT_C, typename T>
bool test_scalar_times(const size_t M, const size_t N, double tol) {

  size_t nnzrow = 27;
  if ((nnzrow < M) && (nnzrow < N)) {
    nnzrow = 27;
  } else {
    nnzrow = std::min({M, N}) - 1;
  }

  monolish::matrix::COO<T> seedA =
      monolish::util::random_structure_matrix<T>(M, N, nnzrow, 1.0);

  T alpha = 123.0;
  MAT_A A(seedA);
  MAT_C C(seedA);

  monolish::matrix::Dense<T> AA(seedA);
  monolish::matrix::Dense<T> CC(seedA);

  ans_scalar_times(alpha, AA, CC);
  monolish::matrix::COO<T> ansC(CC);

  monolish::blas::times(alpha, A, C);

  monolish::matrix::COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.vad, ansC.vad,
                      ansC.get_nnz(), tol);
}
