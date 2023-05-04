#include "../../test_utils.hpp"
#include "monolish_blas.hpp"

template <typename T> void ans_floor(monolish::tensor::tensor_Dense<T> &A) {

  for (size_t i = 0; i < A.get_nnz(); i++) {
    A.begin()[i] = std::floor(A.begin()[i]);
  }
}

template <typename MAT, typename T>
bool test_send_tfloor(const size_t M, const size_t N, const size_t L,
                      double tol) {
  size_t nnzrow = 27;
  if (nnzrow < L) {
    nnzrow = 27;
  } else {
    nnzrow = L - 1;
  }

  monolish::tensor::tensor_COO<T> seedA =
      monolish::util::random_structure_tensor<T>(M, N, L, nnzrow, 1.0);

  MAT A(seedA); // M*N tensor

  monolish::tensor::tensor_Dense<T> AA(seedA);
  ans_floor(AA);

  A.send();
  monolish::vml::floor(A, A);
  A.recv();
  monolish::tensor::tensor_Dense<T> resultA(A);

  return ans_check<T>(__func__, A.type(), resultA.begin(), AA.begin(),
                      AA.get_nnz(), tol);
}

template <typename MAT, typename T>
bool test_tfloor(const size_t M, const size_t N, const size_t L, double tol) {
  size_t nnzrow = 27;
  if (nnzrow < L) {
    nnzrow = 27;
  } else {
    nnzrow = L - 1;
  }

  monolish::tensor::tensor_COO<T> seedA =
      monolish::util::random_structure_tensor<T>(M, N, L, nnzrow, 1.0);

  MAT A(seedA); // M*N tensor

  monolish::tensor::tensor_Dense<T> AA(seedA);
  ans_floor(AA);

  monolish::vml::floor(A, A);
  monolish::tensor::tensor_Dense<T> resultA(A);

  return ans_check<T>(__func__, A.type(), resultA.begin(), AA.begin(),
                      AA.get_nnz(), tol);
}
