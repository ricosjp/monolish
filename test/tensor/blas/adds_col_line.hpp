#include "../../test_utils.hpp"

template <typename T, typename VEC>
void ans_adds_col_line(const monolish::tensor::tensor_Dense<T> &A, size_t num,
                       const VEC &mx, monolish::tensor::tensor_Dense<T> &C) {
  if (A.get_shape()[0] != mx.size()) {
    std::runtime_error("A.row != y.size");
  }

  const T *x = mx.begin();
  int M = A.get_shape()[0];
  int N = A.get_shape()[1] * A.get_shape()[2];

  for (int i = 0; i < M; i++) {
    C.begin()[i * N + num] = A.begin()[i * N + num] + x[i];
  }
}

template <typename MAT, typename VEC, typename T>
bool test_send_adds_col_line_core(const size_t M, const size_t N,
                                  const size_t L, VEC &x, double tol) {
  size_t nnzrow = 27;
  if (nnzrow < L) {
    nnzrow = 27;
  } else {
    nnzrow = L - 1;
  }
  size_t line = N * L - 1;

  monolish::tensor::tensor_COO<T> seedA =
      monolish::util::random_structure_tensor<T>(M, N, L, nnzrow, 1.0);

  MAT A(seedA); // M*N tensor
  MAT C(seedA); // M*N tensor

  monolish::tensor::tensor_Dense<T> AA(seedA);
  monolish::tensor::tensor_Dense<T> CC(seedA);
  ans_adds_col_line(AA, line, x, CC);
  monolish::tensor::tensor_COO<T> ansC(CC);

  monolish::util::send(A, C);
  monolish::blas::adds_col(A, line, x, C);
  C.recv();
  monolish::tensor::tensor_COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.begin(), ansC.begin(),
                      ansC.get_nnz(), tol);
}

template <typename MAT, typename T>
bool test_send_adds_col_line(const size_t M, const size_t N, const size_t L,
                             double tol) {
  monolish::vector<T> vec(M, 0.0, 1.0, test_random_engine());
  vec.send();
  return test_send_adds_col_line_core<MAT, monolish::vector<T>, T>(M, N, L, vec,
                                                                   tol);
}

template <typename MAT, typename VEC, typename T>
bool test_adds_col_line_core(const size_t M, const size_t N, const size_t L,
                             VEC &x, double tol) {
  size_t nnzrow = 27;
  if (nnzrow < L) {
    nnzrow = 27;
  } else {
    nnzrow = L - 1;
  }
  size_t line = N * L - 1;

  monolish::tensor::tensor_COO<T> seedA =
      monolish::util::random_structure_tensor<T>(M, N, L, nnzrow, 1.0);

  MAT A(seedA); // M*N tensor
  MAT C(seedA); // M*N tensor

  monolish::tensor::tensor_Dense<T> AA(seedA);
  monolish::tensor::tensor_Dense<T> CC(seedA);
  ans_adds_col_line(AA, line, x, CC);
  monolish::tensor::tensor_COO<T> ansC(CC);

  monolish::blas::adds_col(A, line, x, C);
  monolish::tensor::tensor_COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.begin(), ansC.begin(),
                      ansC.get_nnz(), tol);
}

template <typename MAT, typename T>
bool test_adds_col_line(const size_t M, const size_t N, const size_t L,
                        double tol) {
  monolish::vector<T> vec(M, 0.0, 1.0, test_random_engine());
  return test_adds_col_line_core<MAT, monolish::vector<T>, T>(M, N, L, vec,
                                                              tol);
}

template <typename MAT, typename T, typename U, typename
std::enable_if<std::is_same<U, monolish::vector<T>>::value,
std::nullptr_t>::type = nullptr> bool test_send_adds_col_line_view(const size_t
M, const size_t N, const size_t L, double tol){ U x(M, 0.0, 1.0);
  x.send();
  monolish::view1D<U, T> vec(x, 0, M);
  return test_send_adds_col_line_core<MAT, monolish::view1D<U, T>, T>(M, N, L,
vec, tol);
}

template <typename MAT, typename T, typename U, typename
std::enable_if<std::is_same<U, monolish::matrix::Dense<T>>::value,
std::nullptr_t>::type = nullptr> bool test_send_adds_col_line_view(const size_t
M, const size_t N, const size_t L, double tol){ U x(M, 1, 0.0, 1.0);
  x.send();
  monolish::view1D<U, T> vec(x, 0, M);
  return test_send_adds_col_line_core<MAT, monolish::view1D<U, T>, T>(M, N, L,
vec, tol);
}

template <typename MAT, typename T, typename U, typename
std::enable_if<std::is_same<U, monolish::tensor::tensor_Dense<T>>::value,
std::nullptr_t>::type = nullptr> bool test_send_adds_col_line_view(const size_t
M, const size_t N, const size_t L, double tol){ U x({M, 1, 1}, 0.0, 1.0);
  x.send();
  monolish::view1D<U, T> vec(x, 0, M);
  return test_send_adds_col_line_core<MAT, monolish::view1D<U, T>, T>(M, N, L,
vec, tol);
}

template <typename MAT, typename T, typename U,
          typename std::enable_if<std::is_same<U, monolish::vector<T>>::value,
                                  std::nullptr_t>::type = nullptr>
bool test_adds_col_line_view(const size_t M, const size_t N, const size_t L,
                             double tol) {
  U x(M, 0.0, 1.0);
  monolish::view1D<U, T> vec(x, 0, M);
  return test_adds_col_line_core<MAT, monolish::view1D<U, T>, T>(M, N, L, vec,
                                                                 tol);
}

template <
    typename MAT, typename T, typename U,
    typename std::enable_if<std::is_same<U, monolish::matrix::Dense<T>>::value,
                            std::nullptr_t>::type = nullptr>
bool test_adds_col_line_view(const size_t M, const size_t N, const size_t L,
                             double tol) {
  U x(M, 1, 0.0, 1.0);
  monolish::view1D<U, T> vec(x, 0, M);
  return test_adds_col_line_core<MAT, monolish::view1D<U, T>, T>(M, N, L, vec,
                                                                 tol);
}

template <typename MAT, typename T, typename U,
          typename std::enable_if<
              std::is_same<U, monolish::tensor::tensor_Dense<T>>::value,
              std::nullptr_t>::type = nullptr>
bool test_adds_col_line_view(const size_t M, const size_t N, const size_t L,
                             double tol) {
  U x({M, 1, 1}, 0.0, 1.0);
  monolish::view1D<U, T> vec(x, 0, M);
  return test_adds_col_line_core<MAT, monolish::view1D<U, T>, T>(M, N, L, vec,
                                                                 tol);
}
