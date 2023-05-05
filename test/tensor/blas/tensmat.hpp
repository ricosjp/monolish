#include "../../test_utils.hpp"

template <typename T>
void ans_tensmat(monolish::tensor::tensor_Dense<T> &A,
                 monolish::matrix::Dense<T> &B,
                 monolish::tensor::tensor_Dense<T> &C) {

  if (A.get_shape()[2] != B.get_row()) {
    std::runtime_error("A.col != x.size");
  }
  if (A.get_shape()[0] != C.get_shape()[0] ||
      A.get_shape()[1] != C.get_shape()[1]) {
    std::runtime_error("A.row != C.row");
  }
  if (B.get_col() != C.get_shape()[2]) {
    std::runtime_error("B.col != C.col");
  }

  T *y = C.begin();
  int M = A.get_shape()[0] * A.get_shape()[1];
  int N = B.get_col();
  int K = A.get_shape()[2];

  for (int i = 0; i < C.get_nnz(); i++)
    y[i] = 0;

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < K; k++) {
        y[i * N + j] += A.begin()[i * K + k] * B.begin()[k * N + j];
      }
    }
  }
}

template <typename T>
void ans_tensmat(const T &a, monolish::tensor::tensor_Dense<T> &A,
                 monolish::matrix::Dense<T> &B, const T &b,
                 monolish::tensor::tensor_Dense<T> &C) {

  if (A.get_shape()[2] != B.get_row()) {
    std::runtime_error("A.col != x.size");
  }
  if (A.get_shape()[0] != C.get_shape()[0] ||
      A.get_shape()[1] != C.get_shape()[1]) {
    std::runtime_error("A.row != C.row");
  }
  if (B.get_col() != C.get_shape()[2]) {
    std::runtime_error("B.col != C.col");
  }

  T *y = C.begin();
  int M = A.get_shape()[0] * A.get_shape()[1];
  int N = B.get_col();
  int K = A.get_shape()[2];

  for (int i = 0; i < C.get_nnz(); i++)
    y[i] = b * y[i];

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < K; k++) {
        y[i * N + j] += a * A.begin()[i * K + k] * B.begin()[k * N + j];
      }
    }
  }
}

template <typename MAT_A, typename MAT_B, typename MAT_C, typename T>
bool test_send_tensmat(const size_t M, const size_t N, const size_t L,
                       const size_t K, double tol) {

  size_t nnzrow = 27;
  if ((nnzrow < M) && (nnzrow < N) && (nnzrow < L) && (nnzrow < K)) {
    nnzrow = 27;
  } else {
    nnzrow = std::min({M, N, L, K}) - 1;
  }

  monolish::tensor::tensor_COO<T> seedA =
      monolish::util::random_structure_tensor<T>(M, N, L, nnzrow, 1.0);
  monolish::matrix::COO<T> seedB =
      monolish::util::random_structure_matrix<T>(L, K, nnzrow, 1.0);
  monolish::tensor::tensor_COO<T> seedC =
      monolish::util::random_structure_tensor<T>(M, N, K, nnzrow, 1.0);

  MAT_A A(seedA); // M*N tensor
  MAT_B B(seedB);
  MAT_C C(seedC); // M*N tensor

  monolish::tensor::tensor_Dense<T> AA(seedA);
  monolish::matrix::Dense<T> BB(seedB);
  monolish::tensor::tensor_Dense<T> CC(seedC);

  ans_tensmat(AA, BB, CC);
  monolish::tensor::tensor_COO<T> ansC(CC);

  monolish::util::send(A, B, C);
  monolish::blas::tensmat(A, B, C);
  C.recv();

  monolish::tensor::tensor_COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.begin(), ansC.begin(),
                      ansC.get_nnz(), tol);
}

template <typename MAT_A, typename MAT_B, typename MAT_C, typename T>
bool test_send_tensmat(const size_t M, const size_t N, const size_t L,
                       const size_t K, const T a, const T b, double tol) {

  size_t nnzrow = 27;
  if ((nnzrow < M) && (nnzrow < N) && (nnzrow < L) && (nnzrow < K)) {
    nnzrow = 27;
  } else {
    nnzrow = std::min({M, N, L, K}) - 1;
  }

  monolish::tensor::tensor_COO<T> seedA =
      monolish::util::random_structure_tensor<T>(M, N, L, nnzrow, 1.0);
  monolish::matrix::COO<T> seedB =
      monolish::util::random_structure_matrix<T>(L, K, nnzrow, 1.0);
  monolish::tensor::tensor_COO<T> seedC =
      monolish::util::random_structure_tensor<T>(M, N, K, nnzrow, 1.0);

  MAT_A A(seedA); // M*N tensor
  MAT_B B(seedB);
  MAT_C C(seedC); // M*N tensor

  monolish::tensor::tensor_Dense<T> AA(seedA);
  monolish::matrix::Dense<T> BB(seedB);
  monolish::tensor::tensor_Dense<T> CC(seedC);

  ans_tensmat(a, AA, BB, b, CC);
  monolish::tensor::tensor_COO<T> ansC(CC);

  monolish::util::send(A, B, C);
  monolish::blas::tensmat(a, A, B, b, C);
  C.recv();

  monolish::tensor::tensor_COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.begin(), ansC.begin(),
                      ansC.get_nnz(), tol);
}

template <typename MAT_A, typename MAT_B, typename MAT_C, typename T>
bool test_tensmat(const size_t M, const size_t N, const size_t L,
                  const size_t K, double tol) {

  size_t nnzrow = 27;
  if ((nnzrow < M) && (nnzrow < N) && (nnzrow < L) && (nnzrow < K)) {
    nnzrow = 27;
  } else {
    nnzrow = std::min({M, N, L, K}) - 1;
  }

  monolish::tensor::tensor_COO<T> seedA =
      monolish::util::random_structure_tensor<T>(M, N, L, nnzrow, 1.0);
  monolish::matrix::COO<T> seedB =
      monolish::util::random_structure_matrix<T>(L, K, nnzrow, 1.0);
  monolish::tensor::tensor_COO<T> seedC =
      monolish::util::random_structure_tensor<T>(M, N, K, nnzrow, 1.0);

  MAT_A A(seedA); // M*N tensor
  MAT_B B(seedB);
  MAT_C C(seedC);

  monolish::tensor::tensor_Dense<T> AA(seedA);
  monolish::matrix::Dense<T> BB(seedB);
  monolish::tensor::tensor_Dense<T> CC(seedC);

  ans_tensmat(AA, BB, CC);
  monolish::tensor::tensor_COO<T> ansC(CC);

  monolish::blas::tensmat(A, B, C);

  monolish::tensor::tensor_COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.begin(), ansC.begin(),
                      ansC.get_nnz(), tol);
}

template <typename MAT_A, typename MAT_B, typename MAT_C, typename T>
bool test_tensmat(const size_t M, const size_t N, const size_t L,
                  const size_t K, const T a, const T b, double tol) {

  size_t nnzrow = 27;
  if ((nnzrow < M) && (nnzrow < N) && (nnzrow < L) && (nnzrow < K)) {
    nnzrow = 27;
  } else {
    nnzrow = std::min({M, N, L, K}) - 1;
  }

  monolish::tensor::tensor_COO<T> seedA =
      monolish::util::random_structure_tensor<T>(M, N, L, nnzrow, 1.0);
  monolish::matrix::COO<T> seedB =
      monolish::util::random_structure_matrix<T>(L, K, nnzrow, 1.0);
  monolish::tensor::tensor_COO<T> seedC =
      monolish::util::random_structure_tensor<T>(M, N, K, nnzrow, 1.0);

  MAT_A A(seedA); // M*N tensor
  MAT_B B(seedB);
  MAT_C C(seedC);

  monolish::tensor::tensor_Dense<T> AA(seedA);
  monolish::matrix::Dense<T> BB(seedB);
  monolish::tensor::tensor_Dense<T> CC(seedC);

  ans_tensmat(a, AA, BB, b, CC);
  monolish::tensor::tensor_COO<T> ansC(CC);

  monolish::blas::tensmat(a, A, B, b, C);

  monolish::tensor::tensor_COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.begin(), ansC.begin(),
                      ansC.get_nnz(), tol);
}
