#include "../../test_utils.hpp"

template <typename T>
void ans_tt_copy(const monolish::tensor::tensor_Dense<T> &A,
                 monolish::tensor::tensor_Dense<T> &C) {

  for (int i = 0; i < A.get_nnz(); i++) {
    C.data()[i] = A.data()[i];
  }
}

template <typename MAT_A, typename MAT_C, typename T>
bool test_send_tt_copy(const size_t M, const size_t N, const size_t L, double tol) {

  size_t nnzrow = 27;
  if ((nnzrow < M) && (nnzrow < N) && (nnzrow < L)) {
    nnzrow = 27;
  } else {
    nnzrow = std::min({M, N, L}) - 1;
  }

  monolish::tensor::tensor_COO<T> seedA =
      monolish::util::random_structure_tensor<T>(M, N, L, nnzrow, 1.0);

  MAT_A A(seedA);
  MAT_C C(seedA);

  monolish::tensor::tensor_Dense<T> AA(seedA);
  monolish::tensor::tensor_Dense<T> CC(seedA);

  ans_tt_copy(AA, CC);
  monolish::tensor::tensor_COO<T> ansC(CC);

  monolish::util::send(A, C);
  monolish::blas::copy(C, A);
  C.recv();

  monolish::tensor::tensor_COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.data(), ansC.data(),
                      ansC.get_nnz(), tol);
}

template <typename MAT_A, typename MAT_C, typename T>
bool test_tt_copy(const size_t M, const size_t N, const size_t L, double tol) {

  size_t nnzrow = 27;
  if ((nnzrow < M) && (nnzrow < N) && (nnzrow < L)) {
    nnzrow = 27;
  } else {
    nnzrow = std::min({M, N, L}) - 1;
  }

  monolish::tensor::tensor_COO<T> seedA =
      monolish::util::random_structure_tensor<T>(M, N, L, nnzrow, 1.0);

  MAT_A A(seedA);
  MAT_C C(seedA);

  monolish::tensor::tensor_Dense<T> AA(seedA);
  monolish::tensor::tensor_Dense<T> CC(seedA);

  ans_tt_copy(AA, CC);
  monolish::tensor::tensor_COO<T> ansC(CC);

  monolish::blas::copy(C, A);

  monolish::tensor::tensor_COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.data(), ansC.data(),
                      ansC.get_nnz(), tol);
}
