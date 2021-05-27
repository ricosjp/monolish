#include "../../test_utils.hpp"

template <typename T>
void ans_matmul(const monolish::matrix::Dense<T> &A,
                const monolish::matrix::Dense<T> &B,
                monolish::matrix::Dense<T> &C) {

  if (A.get_col() != B.get_row()) {
    std::cout << A.get_col() << B.get_row() << std::endl;
    std::runtime_error("test: A.col != B.row");
  }
  if (A.get_row() != C.get_row()) {
    std::runtime_error("test: A.row != C.row");
  }
  if (C.get_col() != B.get_col()) {
    std::runtime_error("test: C.col != B.col");
  }

  // MN=MK*KN
  int M = A.get_row();
  int N = B.get_col();
  int K = A.get_col();

  for (int i = 0; i < C.get_nnz(); i++) {
    C.val[i] = 0;
  }

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < K; k++) {
        C.val[i * N + j] += A.val[i * K + k] * B.val[k * N + j];
      }
    }
  }
}

template <typename MAT_A, typename MAT_B, typename MAT_C, typename T>
bool test_send_matmul(const size_t M, const size_t N, const size_t K,
                      double tol) {

  size_t nnzrow = 27;
  if ((nnzrow < M) && (nnzrow < N) && (nnzrow < K)) {
    nnzrow = 27;
  } else {
    nnzrow = std::min({M, N}) - 1;
  }

  monolish::matrix::COO<T> seedA =
      monolish::util::random_structure_matrix<T>(M, K, nnzrow, 1.0);
  monolish::matrix::COO<T> seedB =
      monolish::util::random_structure_matrix<T>(K, N, nnzrow, 1.0);
  monolish::matrix::COO<T> seedC =
      monolish::util::random_structure_matrix<T>(M, N, nnzrow, 1.0);

  MAT_A A(seedA);
  MAT_B B(seedB);
  MAT_C C(seedC);

  monolish::matrix::Dense<T> AA(seedA);
  monolish::matrix::Dense<T> BB(seedB);
  monolish::matrix::Dense<T> CC(seedC);

  ans_matmul(AA, BB, CC);
  monolish::matrix::COO<T> ansC(CC);

  monolish::util::send(A, B, C);
  monolish::blas::matmul(A, B, C);
  C.recv();

  monolish::matrix::COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.val.data(), ansC.val.data(),
                      ansC.get_nnz(), tol);
}

template <typename MAT_A, typename MAT_B, typename MAT_C, typename T>
bool test_send_matmul_linearoperator_only(const size_t M, const size_t N,
                                          const size_t K, double tol) {

  size_t nnzrow = 27;
  if ((nnzrow < M) && (nnzrow < N) && (nnzrow < K)) {
    nnzrow = 27;
  } else {
    nnzrow = std::min({M, N}) - 1;
  }

  monolish::matrix::COO<T> seedA =
      monolish::util::random_structure_matrix<T>(M, K, nnzrow, 1.0);
  monolish::matrix::COO<T> seedB =
      monolish::util::random_structure_matrix<T>(K, N, nnzrow, 1.0);
  monolish::matrix::COO<T> seedC =
      monolish::util::random_structure_matrix<T>(M, N, nnzrow, 1.0);

  monolish::matrix::CRS<T> A1(seedA);
  monolish::matrix::CRS<T> B1(seedB);
  monolish::matrix::CRS<T> C1(seedC);

  monolish::matrix::Dense<T> AA(seedA);
  monolish::matrix::Dense<T> BB(seedB);
  monolish::matrix::Dense<T> CC(seedC);

  ans_matmul(AA, BB, CC);
  monolish::matrix::COO<T> ansC(CC);

  monolish::util::send(A1, B1, C1);
  MAT_A A(A1);
  MAT_B B(B1);
  MAT_C C(C1);
  monolish::blas::matmul(A, B, C);
  C.recv();

  monolish::matrix::COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.val.data(), ansC.val.data(),
                      ansC.get_nnz(), tol);
}

template <typename MAT_A, typename MAT_B, typename MAT_C, typename T>
bool test_send_matmul_linearoperator(const size_t M, const size_t N,
                                     const size_t K, double tol) {

  size_t nnzrow = 27;
  if ((nnzrow < M) && (nnzrow < N) && (nnzrow < K)) {
    nnzrow = 27;
  } else {
    nnzrow = std::min({M, N}) - 1;
  }

  monolish::matrix::COO<T> seedA =
      monolish::util::random_structure_matrix<T>(M, K, nnzrow, 1.0);
  monolish::matrix::COO<T> seedB =
      monolish::util::random_structure_matrix<T>(K, N, nnzrow, 1.0);
  monolish::matrix::COO<T> seedC =
      monolish::util::random_structure_matrix<T>(M, N, nnzrow, 1.0);

  monolish::matrix::CRS<T> A1(seedA);
  MAT_B B(seedB);
  MAT_C C(seedC);

  monolish::matrix::Dense<T> AA(seedA);
  monolish::matrix::Dense<T> BB(seedB);
  monolish::matrix::Dense<T> CC(seedC);

  ans_matmul(AA, BB, CC);
  monolish::matrix::COO<T> ansC(CC);

  monolish::util::send(A1, B, C);
  MAT_A A(A1);
  monolish::blas::matmul(A, B, C);
  C.recv();

  monolish::matrix::COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.val.data(), ansC.val.data(),
                      ansC.get_nnz(), tol);
}

template <typename MAT_A, typename MAT_B, typename MAT_C, typename T>
bool test_matmul(const size_t M, const size_t N, const size_t K, double tol) {

  size_t nnzrow = 27;
  if ((nnzrow < M) && (nnzrow < N) && (nnzrow < K)) {
    nnzrow = 27;
  } else {
    nnzrow = std::min({M, N}) - 1;
  }

  monolish::matrix::COO<T> seedA =
      monolish::util::random_structure_matrix<T>(M, K, nnzrow, 1.0);
  monolish::matrix::COO<T> seedB =
      monolish::util::random_structure_matrix<T>(K, N, nnzrow, 1.0);
  monolish::matrix::COO<T> seedC =
      monolish::util::random_structure_matrix<T>(M, N, nnzrow, 1.0);

  MAT_A A(seedA);
  MAT_B B(seedB);
  MAT_C C(seedC);

  monolish::matrix::Dense<T> AA(seedA);
  monolish::matrix::Dense<T> BB(seedB);
  monolish::matrix::Dense<T> CC(seedC);

  ans_matmul(AA, BB, CC);
  monolish::matrix::COO<T> ansC(CC);

  monolish::blas::matmul(A, B, C);

  monolish::matrix::COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.val.data(), ansC.val.data(),
                      ansC.get_nnz(), tol);
}
