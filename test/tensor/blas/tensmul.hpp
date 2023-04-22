#include "../../test_utils.hpp"

template <typename T>
void ans_tensmul(const monolish::tensor::tensor_Dense<T> &A,
                 const monolish::tensor::tensor_Dense<T> &B,
                 monolish::tensor::tensor_Dense<T> &C) {

  if (A.get_shape()[2] != B.get_shape()[0]) {
    std::cout << A.get_shape()[2] << B.get_shape()[0] << std::endl;
    std::runtime_error("test: A.shape[2] != B.shape[0]");
  }
  if (A.get_shape()[0] != C.get_shape()[0] ||
      A.get_shape()[1] != C.get_shape()[1]) {
    std::runtime_error("test: A.row != C.row");
  }
  if (C.get_shape()[2] != B.get_shape()[1] ||
      C.get_shape()[3] != B.get_shape()[2]) {
    std::runtime_error("test: C.col != B.col");
  }

  // MLJN=MLK*KJN
  int M = A.get_shape()[0];
  int L = A.get_shape()[1];
  int K = A.get_shape()[2];
  int J = B.get_shape()[1];
  int N = B.get_shape()[2];

  for (int i = 0; i < C.get_nnz(); i++) {
    C.data()[i] = 0;
  }

  for (int ml = 0; ml < M * L; ml++) {
    for (int jn = 0; jn < J * N; jn++) {
      for (int k = 0; k < K; k++) {
        C.data()[ml * J * N + jn] +=
            A.data()[ml * K + k] * B.data()[k * J * N + jn];
      }
    }
  }
}

template <typename T>
void ans_tensmul(const T &a, const monolish::tensor::tensor_Dense<T> &A,
                 const monolish::tensor::tensor_Dense<T> &B, const T &b,
                 monolish::tensor::tensor_Dense<T> &C) {

  if (A.get_shape()[2] != B.get_shape()[0]) {
    std::cout << A.get_shape()[2] << B.get_shape()[0] << std::endl;
    std::runtime_error("test: A.shape[2] != B.shape[0]");
  }
  if (A.get_shape()[0] != C.get_shape()[0] ||
      A.get_shape()[1] != C.get_shape()[1]) {
    std::runtime_error("test: A.row != C.row");
  }
  if (C.get_shape()[2] != B.get_shape()[1] ||
      C.get_shape()[3] != B.get_shape()[2]) {
    std::runtime_error("test: C.col != B.col");
  }

  // MLJN=MLK*KJN
  int M = A.get_shape()[0];
  int L = A.get_shape()[1];
  int K = A.get_shape()[2];
  int J = B.get_shape()[1];
  int N = B.get_shape()[2];

  for (int i = 0; i < C.get_nnz(); i++) {
    C.data()[i] = b * C.data()[i];
  }

  for (int ml = 0; ml < M * L; ml++) {
    for (int jn = 0; jn < J * N; jn++) {
      for (int k = 0; k < K; k++) {
        C.data()[ml * J * N + jn] +=
            a * A.data()[ml * K + k] * B.data()[k * J * N + jn];
      }
    }
  }
}

template <typename MAT_A, typename MAT_B, typename MAT_C, typename T>
bool test_send_tensmul(const size_t M, const size_t N, const size_t L,
                       const size_t J, const size_t K, double tol) {

  size_t nnzrow = 27;
  if ((nnzrow < M) && (nnzrow < N) && (nnzrow < K) && (nnzrow < J) &&
      (nnzrow < L)) {
    nnzrow = 27;
  } else {
    nnzrow = std::min({M, N, L, J, K}) - 1;
  }

  monolish::tensor::tensor_COO<T> seedA =
      monolish::util::random_structure_tensor<T>(M, L, K, nnzrow, 1.0);
  monolish::tensor::tensor_COO<T> seedB =
      monolish::util::random_structure_tensor<T>(K, J, N, nnzrow, 1.0);
  monolish::tensor::tensor_COO<T> seedC =
      monolish::util::random_structure_tensor<T>(M, L, J, N, nnzrow, 1.0);

  MAT_A A(seedA);
  MAT_B B(seedB);
  MAT_C C(seedC);

  monolish::tensor::tensor_Dense<T> AA(seedA);
  monolish::tensor::tensor_Dense<T> BB(seedB);
  monolish::tensor::tensor_Dense<T> CC(seedC);

  ans_tensmul(AA, BB, CC);
  monolish::tensor::tensor_COO<T> ansC(CC);

  monolish::util::send(A, B, C);
  monolish::blas::tensmul(A, B, C);
  C.recv();

  monolish::tensor::tensor_COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.data(), ansC.data(),
                      ansC.get_nnz(), tol);
}

template <typename MAT_A, typename MAT_B, typename MAT_C, typename T>
bool test_send_tensmul(const size_t M, const size_t N, const size_t L,
                       const size_t J, const size_t K, const T a, const T b,
                       double tol) {

  size_t nnzrow = 27;
  if ((nnzrow < M) && (nnzrow < N) && (nnzrow < K) && (nnzrow < J) &&
      (nnzrow < L)) {
    nnzrow = 27;
  } else {
    nnzrow = std::min({M, N, L, J, K}) - 1;
  }

  monolish::tensor::tensor_COO<T> seedA =
      monolish::util::random_structure_tensor<T>(M, L, K, nnzrow, 1.0);
  monolish::tensor::tensor_COO<T> seedB =
      monolish::util::random_structure_tensor<T>(K, J, N, nnzrow, 1.0);
  monolish::tensor::tensor_COO<T> seedC =
      monolish::util::random_structure_tensor<T>(M, L, J, N, nnzrow, 1.0);

  MAT_A A(seedA);
  MAT_B B(seedB);
  MAT_C C(seedC);

  monolish::tensor::tensor_Dense<T> AA(seedA);
  monolish::tensor::tensor_Dense<T> BB(seedB);
  monolish::tensor::tensor_Dense<T> CC(seedC);

  ans_tensmul(a, AA, BB, b, CC);
  monolish::tensor::tensor_COO<T> ansC(CC);

  monolish::util::send(A, B, C);
  monolish::blas::tensmul(a, A, B, b, C);
  C.recv();

  monolish::tensor::tensor_COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.data(), ansC.data(),
                      ansC.get_nnz(), tol);
}

template <typename MAT_A, typename MAT_B, typename MAT_C, typename T>
bool test_tensmul(const size_t M, const size_t N, const size_t L,
                  const size_t J, const size_t K, double tol) {

  size_t nnzrow = 27;
  if ((nnzrow < M) && (nnzrow < N) && (nnzrow < K) && (nnzrow < J) &&
      (nnzrow < L)) {
    nnzrow = 27;
  } else {
    nnzrow = std::min({M, N, L, J, K}) - 1;
  }

  monolish::tensor::tensor_COO<T> seedA =
      monolish::util::random_structure_tensor<T>(M, L, K, nnzrow, 1.0);
  monolish::tensor::tensor_COO<T> seedB =
      monolish::util::random_structure_tensor<T>(K, J, N, nnzrow, 1.0);
  monolish::tensor::tensor_COO<T> seedC =
      monolish::util::random_structure_tensor<T>(M, L, J, N, nnzrow, 1.0);

  MAT_A A(seedA);
  MAT_B B(seedB);
  MAT_C C(seedC);

  monolish::tensor::tensor_Dense<T> AA(seedA);
  monolish::tensor::tensor_Dense<T> BB(seedB);
  monolish::tensor::tensor_Dense<T> CC(seedC);

  ans_tensmul(AA, BB, CC);
  monolish::tensor::tensor_COO<T> ansC(CC);

  monolish::blas::tensmul(A, B, C);

  monolish::tensor::tensor_COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.data(), ansC.data(),
                      ansC.get_nnz(), tol);
}

template <typename MAT_A, typename MAT_B, typename MAT_C, typename T>
bool test_tensmul(const size_t M, const size_t N, const size_t L,
                  const size_t J, const size_t K, const T a, const T b,
                  double tol) {

  size_t nnzrow = 27;
  if ((nnzrow < M) && (nnzrow < N) && (nnzrow < K) && (nnzrow < J) &&
      (nnzrow < L)) {
    nnzrow = 27;
  } else {
    nnzrow = std::min({M, N, L, J, K}) - 1;
  }

  monolish::tensor::tensor_COO<T> seedA =
      monolish::util::random_structure_tensor<T>(M, L, K, nnzrow, 1.0);
  monolish::tensor::tensor_COO<T> seedB =
      monolish::util::random_structure_tensor<T>(K, J, N, nnzrow, 1.0);
  monolish::tensor::tensor_COO<T> seedC =
      monolish::util::random_structure_tensor<T>(M, L, J, N, nnzrow, 1.0);

  MAT_A A(seedA);
  MAT_B B(seedB);
  MAT_C C(seedC);

  monolish::tensor::tensor_Dense<T> AA(seedA);
  monolish::tensor::tensor_Dense<T> BB(seedB);
  monolish::tensor::tensor_Dense<T> CC(seedC);

  ans_tensmul(a, AA, BB, b, CC);
  monolish::tensor::tensor_COO<T> ansC(CC);

  monolish::blas::tensmul(a, A, B, b, C);

  monolish::tensor::tensor_COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.data(), ansC.data(),
                      ansC.get_nnz(), tol);
}
