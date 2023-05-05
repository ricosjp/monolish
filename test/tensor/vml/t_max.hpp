#include "../../test_utils.hpp"
#include "monolish_blas.hpp"

template <typename T> T ans_max(monolish::tensor::tensor_Dense<T> &A) {
  return *(std::max_element(A.begin(), A.begin() + A.get_nnz()));
}

template <typename MAT, typename T>
bool test_send_tmax(const size_t M, const size_t N, const size_t L,
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
  T ans = ans_max(AA);

  A.send();
  T result = monolish::vml::max(A);
  monolish::tensor::tensor_Dense<T> resultA(A);

  return ans_check<T>(__func__, result, ans, tol);
}

template <typename MAT, typename T>
bool test_tmax(const size_t M, const size_t N, const size_t L, double tol) {
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
  T ans = ans_max(AA);

  T result = monolish::vml::max(A);
  monolish::tensor::tensor_Dense<T> resultA(A);

  return ans_check<T>(__func__, result, ans, tol);
}
