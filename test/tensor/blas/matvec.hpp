#include "../../test_utils.hpp"

template <typename T>
void ans_tensvec(monolish::tensor::tensor_Dense<T> &A, monolish::vector<T> &mx,
                 monolish::vector<T> &my) {

  if (A.get_col() != mx.size()) {
    std::runtime_error("A.col != x.size");
  }
  if (A.get_row() != my.size()) {
    std::runtime_error("A.row != y.size");
  }

  T *x = mx.data();
  T *y = my.data();
  int M = A.get_row();
  int N = A.get_col();

  for (int i = 0; i < my.size(); i++)
    y[i] = 0;

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      y[i] += A.data()[N * i + j] * x[j];
    }
  }
}

template <typename MAT, typename T>
bool test_send_tensvec(const size_t M, const size_t N, const size_t L,
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

  monolish::vector<T> x(A.get_col(), 0.0, 1.0, test_random_engine());
  monolish::vector<T> y(A.get_row(), 0.0, 1.0, test_random_engine());

  monolish::vector<T> ansy(A.get_row());
  ansy = y;

  monolish::tensor::tensor_Dense<T> AA(seedA);
  ans_tensvec(AA, x, ansy);

  monolish::util::send(A, x, y);
  monolish::blas::tensvec(A, x, y);
  y.recv();

  return ans_check<T>(__func__, A.type(), y.data(), ansy.data(), y.size(), tol);
}

template <typename MAT, typename T>
bool test_send_tensvec_linearoperator(const size_t M, const size_t N,
                                      double tol) {

  size_t nnzrow = 27;
  if (nnzrow < N) {
    nnzrow = 27;
  } else {
    nnzrow = N - 1;
  }

  monolish::tensor::tensor_COO<T> seedA =
      monolish::util::random_structure_tensor<T>(M, N, nnzrow, 1.0);

  monolish::tensor::CRS<T> A1(seedA);

  monolish::vector<T> x(A1.get_col(), 0.0, 1.0, test_random_engine());
  monolish::vector<T> y(A1.get_row(), 0.0, 1.0, test_random_engine());

  monolish::vector<T> ansy(A1.get_row());
  ansy = y;

  monolish::tensor::tensor_Dense<T> AA(seedA);
  ans_tensvec(AA, x, ansy);

  monolish::util::send(A1, x, y);
  MAT A2(A1); // M*N tensor
  monolish::blas::tensvec(A2, x, y);
  y.recv();

  return ans_check<T>(__func__, A2.type(), y.data(), ansy.data(), y.size(),
                      tol);
}

template <typename MAT, typename T>
bool test_tensvec(const size_t M, const size_t N, double tol) {

  size_t nnzrow = 27;
  if (nnzrow < N) {
    nnzrow = 27;
  } else {
    nnzrow = N - 1;
  }

  monolish::tensor::tensor_COO<T> seedA =
      monolish::util::random_structure_tensor<T>(M, N, nnzrow, 1.0);

  MAT A(seedA); // M*N tensor

  monolish::vector<T> x(A.get_col(), 0.0, 1.0, test_random_engine());
  monolish::vector<T> y(A.get_row(), 0.0, 1.0, test_random_engine());

  monolish::vector<T> ansy(A.get_row());
  ansy = y;

  monolish::tensor::tensor_Dense<T> AA(seedA);
  ans_tensvec(AA, x, ansy);

  monolish::blas::tensvec(A, x, y);

  return ans_check<T>(__func__, A.type(), y.data(), ansy.data(), y.size(), tol);
}
