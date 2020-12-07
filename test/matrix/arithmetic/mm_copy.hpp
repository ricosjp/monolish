#include "../../test_utils.hpp"

template <typename T>
void ans_mm_copy(const monolish::matrix::Dense<T> &A,
                monolish::matrix::Dense<T> &C) {

  for (int i = 0; i < A.get_nnz(); i++) {
    C.val[i] = A.val[i];
  }
}

template <typename MAT_A, typename MAT_C, typename T>
bool test_send_mm_copy(const size_t M, const size_t N, double tol) {

  size_t nnzrow = 27;
  if ((nnzrow < M) && (nnzrow < N)) {
    nnzrow = 27;
  } else {
    nnzrow = std::min({M, N}) - 1;
  }

  monolish::matrix::COO<T> seedA =
      monolish::util::random_structure_matrix<T>(M, N, nnzrow, 1.0);

  MAT_A A(seedA);
  MAT_C C(seedA);

  monolish::matrix::Dense<T> AA(seedA);
  monolish::matrix::Dense<T> CC(seedA);

  ans_mm_copy(AA, CC);
  monolish::matrix::COO<T> ansC(CC);

  monolish::util::send(A, C);
  C = A;
  C.recv();

  monolish::matrix::COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.val.data(), ansC.val.data(),
                      ansC.get_nnz(), tol);
}

template <typename MAT_A, typename MAT_C, typename T>
bool test_mm_copy(const size_t M, const size_t N, double tol) {

  size_t nnzrow = 27;
  if ((nnzrow < M) && (nnzrow < N)) {
    nnzrow = 27;
  } else {
    nnzrow = std::min({M, N}) - 1;
  }

  monolish::matrix::COO<T> seedA =
      monolish::util::random_structure_matrix<T>(M, N, nnzrow, 1.0);

  MAT_A A(seedA);
  MAT_C C(seedA);

  monolish::matrix::Dense<T> AA(seedA);
  monolish::matrix::Dense<T> CC(seedA);

  ans_mm_copy(AA, CC);
  monolish::matrix::COO<T> ansC(CC);

  C = A;

  monolish::matrix::COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.val.data(), ansC.val.data(),
                      ansC.get_nnz(), tol);
}
