#include "../../test_utils.hpp"

template <typename T>
void ans_alo(monolish::tensor::tensor_Dense<T> &A, const T alpha, const T beta){

  for (int i = 0; i < A.get_nnz(); i++) {
    if (A.data()[i] > 0) {
      A.data()[i] = alpha * A.data()[i];
    } else {
      A.data()[i] = beta * A.data()[i];
    }
  }
}

template <typename MAT, typename T>
bool test_send_talo(const size_t M, const size_t N, const size_t L, double tol) {
  size_t nnzrow = 27;
  if (nnzrow < L) {
    nnzrow = 27;
  } else {
    nnzrow = L - 1;
  }

  monolish::tensor::tensor_COO<T> seedA =
      monolish::util::random_structure_tensor<T>(M, N, L, nnzrow, 1.0);

  MAT A(seedA);
  T alpha = 1.5;
  T beta = 0.5;

  monolish::tensor::tensor_Dense<T> AA(seedA);
  ans_alo(AA, alpha, beta);

  A.send();
  monolish::vml::alo(A, alpha, beta, A);
  A.recv();
  monolish::tensor::tensor_Dense<T> resultA(A);

  return ans_check<T>(__func__, A.type(), resultA.data(), A.data(),
                      A.get_nnz(), tol);
}

template <typename MAT, typename T>
bool test_talo(const size_t M, const size_t N, const size_t L, double tol) {
  size_t nnzrow = 27;
  if (nnzrow < L) {
    nnzrow = 27;
  } else {
    nnzrow = L - 1;
  }

  monolish::tensor::tensor_COO<T> seedA =
      monolish::util::random_structure_tensor<T>(M, N, L, nnzrow, 1.0);

  MAT A(seedA);
  T alpha = 1.5;
  T beta = 0.5;

  monolish::tensor::tensor_Dense<T> AA(seedA);
  ans_alo(AA, alpha, beta);

  monolish::vml::alo(A, alpha, beta, A);
  monolish::tensor::tensor_Dense<T> resultA(A);

  return ans_check<T>(__func__, A.type(), resultA.data(), AA.data(),
                      AA.get_nnz(), tol);
}
