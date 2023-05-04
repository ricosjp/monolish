#include "../../test_utils.hpp"

template <typename T>
void ans_mm_mul(const monolish::matrix::Dense<T> &A,
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
    C.begin()[i] = A.begin()[i] * B.begin()[i];
  }
}

template <typename MAT_A, typename MAT_B, typename MAT_C, typename T>
bool test_send_mm_mul(const size_t M, const size_t N, double tol) {

  monolish::matrix::Dense<T> seed(M, N, 1.0, 2.0);
  monolish::matrix::COO<T> seedA(seed);

  MAT_A A(seedA);
  MAT_B B(seedA);
  MAT_C C(seedA);

  monolish::matrix::Dense<T> AA(seedA);
  monolish::matrix::Dense<T> BB(seedA);
  monolish::matrix::Dense<T> CC(seedA);

  ans_mm_mul(AA, BB, CC);
  monolish::matrix::COO<T> ansC(CC);

  monolish::util::send(A, B, C);
  monolish::vml::mul(A, B, C);
  C.recv();

  monolish::matrix::COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.begin(), ansC.begin(),
                      ansC.get_nnz(), tol);
}

template <typename MAT_A, typename MAT_B, typename MAT_C, typename T>
bool test_mm_mul(const size_t M, const size_t N, double tol) {

  monolish::matrix::Dense<T> seed(M, N, 1.0, 2.0);
  monolish::matrix::COO<T> seedA(seed);

  MAT_A A(seedA);
  MAT_B B(seedA);
  MAT_C C(seedA);

  monolish::matrix::Dense<T> AA(seedA);
  monolish::matrix::Dense<T> BB(seedA);
  monolish::matrix::Dense<T> CC(seedA);

  ans_mm_mul(AA, BB, CC);
  monolish::matrix::COO<T> ansC(CC);

  monolish::vml::mul(A, B, C);

  monolish::matrix::COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.begin(), ansC.begin(),
                      ansC.get_nnz(), tol);
}
