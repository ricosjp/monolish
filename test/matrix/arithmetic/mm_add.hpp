#include "../../test_utils.hpp"

template <typename T>
void ans_mm_add(const monolish::matrix::Dense<T> &A,
                const monolish::matrix::Dense<T> &B,
                monolish::matrix::Dense<T> &C) {

  if (A.get_row() != B.get_row()) {
    std::runtime_error("A.row != B.row");
  }
  if (A.get_col() != B.get_col()) {
    std::runtime_error("A.col != B.col");
  }

  // MN=MN+MN
  int M = A.get_row();
  int N = A.get_col();

  for (int i = 0; i < A.get_nnz(); i++) {
    C.val[i] += A.val[i] * B.val[i];
  }
}

template <typename MAT_A, typename MAT_B, typename MAT_C, typename T>
bool test_send_mm_add(const size_t M, const size_t N, double tol) {

  size_t nnzrow = 27;
  if ((nnzrow < M) && (nnzrow < N)) {
    nnzrow = 27;
  } else {
    nnzrow = std::min({M, N}) - 1;
  }

  monolish::matrix::COO<T> seedA =
      monolish::util::random_structure_matrix<T>(M, N, nnzrow, 1.0);

  MAT_A A(seedA);
  MAT_B B(seedA);
  MAT_C C(seedA);

  monolish::matrix::Dense<T> AA(seedA);
  monolish::matrix::Dense<T> BB(seedA);
  monolish::matrix::Dense<T> CC(seedA);

  ans_mm_add(AA, BB, CC);
  monolish::matrix::COO<T> ansC(CC);

  monolish::util::send(A, B, C);
  monolish::blas::add(A,B,C);
  C.recv();

  monolish::matrix::COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.val.data(), ansC.val.data(),
                      ansC.get_nnz(), tol);
}

template <typename MAT_A, typename MAT_B, typename MAT_C, typename T>
bool test_mm_add(const size_t M, const size_t N, double tol) {

  size_t nnzrow = 27;
  if ((nnzrow < M) && (nnzrow < N)) {
    nnzrow = 27;
  } else {
    nnzrow = std::min({M, N}) - 1;
  }

  monolish::matrix::COO<T> seedA =
      monolish::util::random_structure_matrix<T>(M, N, nnzrow, 1.0);

  MAT_A A(seedA);
  MAT_B B(seedA);
  MAT_C C(seedA);

  monolish::matrix::Dense<T> AA(seedA);
  monolish::matrix::Dense<T> BB(seedA);
  monolish::matrix::Dense<T> CC(seedA);

  ans_mm_add(AA, BB, CC);
  monolish::matrix::COO<T> ansC(CC);

  monolish::blas::add(A,B,C);

  monolish::matrix::COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.val.data(), ansC.val.data(),
                      ansC.get_nnz(), tol);
}
