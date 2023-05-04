#include "../../test_utils.hpp"

template <typename T>
void ans_st_pow(const monolish::tensor::tensor_Dense<T> &A, const T alpha,
                monolish::tensor::tensor_Dense<T> &C) {

  for (int i = 0; i < A.get_nnz(); i++) {
    C.begin()[i] = std::pow(A.begin()[i], alpha);
  }
}

template <typename MAT_A, typename MAT_C, typename T>
bool test_send_st_pow(const size_t M, const size_t N, const size_t L,
                      double tol) {
  size_t nnzrow = 27;
  if (nnzrow < L) {
    nnzrow = 27;
  } else {
    nnzrow = L - 1;
  }

  monolish::tensor::tensor_COO<T> seedA =
      monolish::util::random_structure_tensor<T>(M, N, L, nnzrow, 1.0);

  MAT_A A(seedA);
  T alpha = 123.0;
  MAT_C C(seedA);

  monolish::tensor::tensor_Dense<T> AA(seedA);
  monolish::tensor::tensor_Dense<T> CC(seedA);

  ans_st_pow(AA, alpha, CC);
  monolish::tensor::tensor_COO<T> ansC(CC);

  monolish::util::send(A, C);
  monolish::vml::pow(A, alpha, C);
  C.recv();

  monolish::tensor::tensor_COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.begin(), ansC.begin(),
                      ansC.get_nnz(), tol);
}

template <typename MAT_A, typename MAT_C, typename T>
bool test_st_pow(const size_t M, const size_t N, const size_t L, double tol) {
  size_t nnzrow = 27;
  if (nnzrow < L) {
    nnzrow = 27;
  } else {
    nnzrow = L - 1;
  }

  monolish::tensor::tensor_COO<T> seedA =
      monolish::util::random_structure_tensor<T>(M, N, L, nnzrow, 1.0);

  MAT_A A(seedA);
  T alpha = 123.0;
  MAT_C C(seedA);

  monolish::tensor::tensor_Dense<T> AA(seedA);
  monolish::tensor::tensor_Dense<T> CC(seedA);

  ans_st_pow(AA, alpha, CC);
  monolish::tensor::tensor_COO<T> ansC(CC);

  monolish::vml::pow(A, alpha, C);

  monolish::tensor::tensor_COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.begin(), ansC.begin(),
                      ansC.get_nnz(), tol);
}
