#include "../../test_utils.hpp"

template <typename T, typename VEC>
void ans_adds_col_line(const monolish::matrix::Dense<T> &A, size_t num,
                       const VEC &mx, monolish::matrix::Dense<T> &C) {
  if (A.get_row() != mx.size()) {
    std::runtime_error("A.row != y.size");
  }

  const T *x = mx.begin();
  int M = A.get_row();
  int N = A.get_col();

  for (int i = 0; i < M; i++) {
    C.begin()[i * N + num] = A.begin()[i * N + num] + x[i];
  }
}

template <typename MAT, typename VEC, typename T>
bool test_send_adds_col_line_core(const size_t M, const size_t N, VEC &x,
                                  double tol) {
  size_t line = N - 1;
  monolish::matrix::COO<T> seedA = get_random_structure_matrix<T>(M, N);

  MAT A(seedA); // M*N matrix
  MAT C(seedA); // M*N matrix

  monolish::matrix::Dense<T> AA(seedA);
  monolish::matrix::Dense<T> CC(seedA);
  ans_adds_col_line(AA, line, x, CC);
  monolish::matrix::COO<T> ansC(CC);

  monolish::util::send(A, C);
  monolish::blas::adds_col(A, line, x, C);
  C.recv();
  monolish::matrix::COO<T> resultC(C);

  return ans_check<T>(__func__, A.type() + "+" + x.type(), resultC.begin(),
                      ansC.begin(), ansC.get_nnz(), tol);
}

template <typename MAT, typename T>
bool test_send_adds_col_line(const size_t M, const size_t N, double tol) {
  monolish::vector<T> vec(M, 0.0, 1.0, test_random_engine());
  vec.send();
  return test_send_adds_col_line_core<MAT, monolish::vector<T>, T>(M, N, vec,
                                                                   tol);
}

template <typename MAT, typename VEC, typename T>
bool test_adds_col_line_core(const size_t M, const size_t N, VEC &x,
                             double tol) {
  size_t line = N - 1;
  monolish::matrix::COO<T> seedA = get_random_structure_matrix<T>(M, N);

  MAT A(seedA); // M*N matrix
  MAT C(seedA); // M*N matrix

  monolish::matrix::Dense<T> AA(seedA);
  monolish::matrix::Dense<T> CC(seedA);
  ans_adds_col_line(AA, line, x, CC);
  monolish::matrix::COO<T> ansC(CC);

  monolish::blas::adds_col(A, line, x, C);
  monolish::matrix::COO<T> resultC(C);

  return ans_check<T>(__func__, A.type() + "+" + x.type(), resultC.begin(),
                      ansC.begin(), ansC.get_nnz(), tol);
}

template <typename MAT, typename T>
bool test_adds_col_line(const size_t M, const size_t N, double tol) {
  monolish::vector<T> vec(M, 0.0, 1.0, test_random_engine());
  return test_adds_col_line_core<MAT, monolish::vector<T>, T>(M, N, vec, tol);
}

template <typename MAT, typename T>
bool test_send_adds_col_line_view_core(const size_t M, const size_t N,
                                       double tol) {
  return true;
}

template <typename MAT, typename T, typename U, typename... ARGS>
bool test_send_adds_col_line_view_core(const size_t M, const size_t N,
                                       double tol, U x, ARGS... args) {
  x.send();
  monolish::view1D<U, T> vec(x, M / 2, M);
  if (!test_send_adds_col_line_core<MAT, monolish::view1D<U, T>, T>(M, N, vec,
                                                                    tol)) {
    return false;
  }
  return test_send_adds_col_line_view_core<MAT, T, ARGS...>(M, N, tol, args...);
}

template <typename MAT, typename T>
bool test_send_adds_col_line_view(const size_t M, const size_t N, double tol) {
  using T1 = monolish::vector<T>;
  using T2 = monolish::matrix::Dense<T>;
  using T3 = monolish::tensor::tensor_Dense<T>;
  T1 x1(2 * M, 0.0, 1.0);
  T2 x2(2 * M, 1, 0.0, 1.0);
  T3 x3({2 * M, 1, 1}, 0.0, 1.0);

  return test_send_adds_col_line_view_core<MAT, T, T1, T2, T3>(M, N, tol, x1,
                                                               x2, x3);
}

template <typename MAT, typename T>
bool test_adds_col_line_view_core(const size_t M, const size_t N, double tol) {
  return true;
}

template <typename MAT, typename T, typename U, typename... ARGS>
bool test_adds_col_line_view_core(const size_t M, const size_t N, double tol,
                                  U x, ARGS... args) {
  monolish::view1D<U, T> vec(x, M / 2, M);
  if (!test_adds_col_line_core<MAT, monolish::view1D<U, T>, T>(M, N, vec,
                                                               tol)) {
    return false;
  }
  return test_adds_col_line_view_core<MAT, T, ARGS...>(M, N, tol, args...);
}

template <typename MAT, typename T>
bool test_adds_col_line_view(const size_t M, const size_t N, double tol) {
  using T1 = monolish::vector<T>;
  using T2 = monolish::matrix::Dense<T>;
  using T3 = monolish::tensor::tensor_Dense<T>;
  T1 x1(2 * M, 0.0, 1.0);
  T2 x2(2 * M, 1, 0.0, 1.0);
  T3 x3({2 * M, 1, 1}, 0.0, 1.0);

  return test_adds_col_line_view_core<MAT, T, T1, T2, T3>(M, N, tol, x1, x2,
                                                          x3);
}
