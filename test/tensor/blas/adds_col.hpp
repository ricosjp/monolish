#include "../../test_utils.hpp"

template <typename T>
void ans_adds_col(const monolish::tensor::tensor_Dense<T> &A,
                  const monolish::vector<T> &mx,
                  monolish::tensor::tensor_Dense<T> &C) {
  if (A.get_shape()[0] != mx.size()) {
    std::runtime_error("A.row != y.size");
  }

  const T *x = mx.data();
  int M = A.get_shape()[0];
  int N = A.get_shape()[1] * A.get_shape()[2];

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      C.data()[i * N + j] = A.data()[i * N + j] + x[i];
    }
  }
}

template <typename MAT, typename T>
bool test_send_adds_col(const size_t M, const size_t N, const size_t L,
                        double tol) {
  size_t nnzrow = 27;
  if (nnzrow < L) {
    nnzrow = 27;
  } else {
    nnzrow = L - 1;
  }

  monolish::tensor::tensor_COO<T> seedA =
      monolish::util::random_structure_tensor<T>(M, N, L, nnzrow, 1.0);

  MAT A(seedA); // M*N*L tensor
  MAT C(seedA); // M*N*L tensor
  monolish::vector<T> x(M, 0.0, 1.0, test_random_engine());

  monolish::tensor::tensor_Dense<T> AA(seedA);
  monolish::tensor::tensor_Dense<T> CC(seedA);
  ans_adds_col(AA, x, CC);
  monolish::tensor::tensor_COO<T> ansC(CC);

  monolish::util::send(A, x, C);
  monolish::blas::adds_col(A, x, C);
  C.recv();
  monolish::tensor::tensor_COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.data(), ansC.data(),
                      ansC.get_nnz(), tol);
}

template <typename MAT, typename T>
bool test_adds_col(const size_t M, const size_t N, const size_t L, double tol) {
  size_t nnzrow = 27;
  if (nnzrow < L) {
    nnzrow = 27;
  } else {
    nnzrow = L - 1;
  }

  monolish::tensor::tensor_COO<T> seedA =
      monolish::util::random_structure_tensor<T>(M, N, L, nnzrow, 1.0);

  MAT A(seedA); // M*N*L tensor
  MAT C(seedA); // M*N*L tensor
  monolish::vector<T> x(M, 0.0, 1.0, test_random_engine());

  monolish::tensor::tensor_Dense<T> AA(seedA);
  monolish::tensor::tensor_Dense<T> CC(seedA);
  ans_adds_col(AA, x, CC);
  monolish::tensor::tensor_COO<T> ansC(CC);

  monolish::blas::adds_col(A, x, C);
  monolish::tensor::tensor_COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.data(), ansC.data(),
                      ansC.get_nnz(), tol);
}
