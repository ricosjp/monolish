#include "../../test_utils.hpp"

template <typename T, typename VEC>
void ans_tensvec(monolish::tensor::tensor_Dense<T> &A, VEC &mx,
                 monolish::tensor::tensor_Dense<T> &C) {

  if (A.get_shape()[2] != mx.size()) {
    std::runtime_error("A.col != x.size");
  }
  if (A.get_shape()[0] != C.get_shape()[0] ||
      A.get_shape()[1] != C.get_shape()[1]) {
    std::runtime_error("A.row != C.row");
  }

  T *x = mx.begin();
  T *y = C.begin();
  int M = A.get_shape()[0] * A.get_shape()[1];
  int N = A.get_shape()[2];

  for (int i = 0; i < C.get_nnz(); i++)
    y[i] = 0;

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      y[i] += A.begin()[N * i + j] * x[j];
    }
  }
}

template <typename MAT, typename VEC, typename T>
bool test_send_tensvec_core(const size_t M, const size_t N, const size_t L,
                            VEC &x, double tol) {

  size_t nnzrow = 27;
  if ((nnzrow < M) && (nnzrow < N) && (nnzrow < L)) {
    nnzrow = 27;
  } else {
    nnzrow = std::min({M, N, L}) - 1;
  }

  monolish::tensor::tensor_COO<T> seedA =
      monolish::util::random_structure_tensor<T>(M, N, L, nnzrow, 1.0);
  monolish::tensor::tensor_COO<T> seedC =
      monolish::util::random_structure_tensor<T>(M, N, nnzrow, 1.0);

  MAT A(seedA); // M*N tensor
  MAT C(seedC); // M*N tensor

  monolish::tensor::tensor_Dense<T> AA(seedA);
  monolish::tensor::tensor_Dense<T> CC(seedC);

  ans_tensvec(AA, x, CC);
  monolish::tensor::tensor_COO<T> ansC(CC);

  monolish::util::send(A, C);
  monolish::blas::tensvec(A, x, C);
  C.recv();

  monolish::tensor::tensor_COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.begin(), ansC.begin(),
                      ansC.get_nnz(), tol);
}

template <typename MAT, typename T>
bool test_send_tensvec(const size_t M, const size_t N, const size_t L,
                       double tol) {
  monolish::vector<T> vec(L, 0.0, 1.0, test_random_engine());
  vec.send();
  return test_send_tensvec_core<MAT, monolish::vector<T>, T>(M, N, L, vec, tol);
}

template <typename MAT, typename VEC, typename T>
bool test_tensvec_core(const size_t M, const size_t N, const size_t L, VEC &x,
                       double tol) {

  size_t nnzrow = 27;
  if ((nnzrow < M) && (nnzrow < N) && (nnzrow < L)) {
    nnzrow = 27;
  } else {
    nnzrow = std::min({M, N, L}) - 1;
  }

  monolish::tensor::tensor_COO<T> seedA =
      monolish::util::random_structure_tensor<T>(M, N, L, nnzrow, 1.0);
  monolish::tensor::tensor_COO<T> seedC =
      monolish::util::random_structure_tensor<T>(M, N, nnzrow, 1.0);

  MAT A(seedA); // M*N tensor
  MAT C(seedC);

  monolish::tensor::tensor_Dense<T> AA(seedA);
  monolish::tensor::tensor_Dense<T> CC(seedC);

  ans_tensvec(AA, x, CC);
  monolish::tensor::tensor_COO<T> ansC(CC);

  monolish::blas::tensvec(A, x, C);

  monolish::tensor::tensor_COO<T> resultC(C);

  return ans_check<T>(__func__, A.type(), resultC.begin(), ansC.begin(),
                      ansC.get_nnz(), tol);
}

template <typename MAT, typename T>
bool test_tensvec(const size_t M, const size_t N, const size_t L, double tol) {
  monolish::vector<T> vec(L, 0.0, 1.0, test_random_engine());
  return test_tensvec_core<MAT, monolish::vector<T>, T>(M, N, L, vec, tol);
}

template <typename MAT, typename T, typename U,
          typename std::enable_if<std::is_same<U, monolish::vector<T>>::value,
                                  std::nullptr_t>::type = nullptr>
bool test_send_tensvec_view(const size_t M, const size_t N, const size_t L,
                            double tol) {
  monolish::vector<T> x(2 * L, 0.0, 1.0);
  x.send();
  monolish::view1D<monolish::vector<T>, T> vec(x, L / 2, L);
  return test_send_tensvec_core<MAT, monolish::view1D<monolish::vector<T>, T>,
                                T>(M, N, L, vec, tol);
}

template <
    typename MAT, typename T, typename U,
    typename std::enable_if<std::is_same<U, monolish::matrix::Dense<T>>::value,
                            std::nullptr_t>::type = nullptr>
bool test_send_tensvec_view(const size_t M, const size_t N, const size_t L,
                            double tol) {
  monolish::matrix::Dense<T> x(2 * L, 1, 0.0, 1.0);
  x.send();
  monolish::view1D<monolish::matrix::Dense<T>, T> vec(x, L / 2, L);
  return test_send_tensvec_core<
      MAT, monolish::view1D<monolish::matrix::Dense<T>, T>, T>(M, N, L, vec,
                                                               tol);
}

template <typename MAT, typename T, typename U,
          typename std::enable_if<
              std::is_same<U, monolish::tensor::tensor_Dense<T>>::value,
              std::nullptr_t>::type = nullptr>
bool test_send_tensvec_view(const size_t M, const size_t N, const size_t L,
                            double tol) {
  monolish::tensor::tensor_Dense<T> x({2 * L, 1, 1}, 0.0, 1.0);
  x.send();
  monolish::view1D<monolish::tensor::tensor_Dense<T>, T> vec(x, L / 2, L);
  return test_send_tensvec_core<
      MAT, monolish::view1D<monolish::tensor::tensor_Dense<T>, T>, T>(M, N, L,
                                                                      vec, tol);
}

template <typename MAT, typename T, typename U,
          typename std::enable_if<std::is_same<U, monolish::vector<T>>::value,
                                  std::nullptr_t>::type = nullptr>
bool test_tensvec_view(const size_t M, const size_t N, const size_t L,
                       double tol) {
  U x(2 * L, 0.0, 1.0);
  monolish::view1D<U, T> vec(x, L / 2, L);
  return test_tensvec_core<MAT, monolish::view1D<U, T>, T>(M, N, L, vec, tol);
}

template <
    typename MAT, typename T, typename U,
    typename std::enable_if<std::is_same<U, monolish::matrix::Dense<T>>::value,
                            std::nullptr_t>::type = nullptr>
bool test_tensvec_view(const size_t M, const size_t N, const size_t L,
                       double tol) {
  U x(2 * L, 1, 0.0, 1.0);
  monolish::view1D<U, T> vec(x, L / 2, L);
  return test_tensvec_core<MAT, monolish::view1D<U, T>, T>(M, N, L, vec, tol);
}

template <typename MAT, typename T, typename U,
          typename std::enable_if<
              std::is_same<U, monolish::tensor::tensor_Dense<T>>::value,
              std::nullptr_t>::type = nullptr>
bool test_tensvec_view(const size_t M, const size_t N, const size_t L,
                       double tol) {
  U x({2 * L, 1, 1}, 0.0, 1.0);
  monolish::view1D<U, T> vec(x, L / 2, L);
  return test_tensvec_core<MAT, monolish::view1D<U, T>, T>(M, N, L, vec, tol);
}
