#include "../../test_utils.hpp"

template <typename T, typename VEC>
void ans_adds_row_line(const monolish::matrix::Dense<T> &A, size_t num,
                       const VEC &mx, monolish::matrix::Dense<T> &C) {
  if (A.get_col() != mx.size()) {
    std::runtime_error("A.row != y.size");
  }

  const T *x = mx.begin();
  int M = A.get_row();
  int N = A.get_col();

  for (int j = 0; j < N; j++) {
    C.begin()[num * N + j] = A.begin()[num * N + j] + x[j];
  }
}

template <typename MAT, typename VEC, typename T>
bool test_send_adds_row_line_core(const size_t M, const size_t N, VEC &x,
                                  double tol) {
  size_t nnzrow = 27;
  if (nnzrow < N) {
    nnzrow = 27;
  } else {
    nnzrow = N - 1;
  }
  size_t line = M - 1;

  monolish::matrix::COO<T> seedA =
      monolish::util::random_structure_matrix<T>(M, N, nnzrow, 1.0);

  MAT A(seedA); // M*N matrix
  MAT C(seedA); // M*N matrix

  monolish::matrix::Dense<T> AA(seedA);
  monolish::matrix::Dense<T> CC(seedA);
  ans_adds_row_line(AA, line, x, CC);
  monolish::matrix::COO<T> ansC(CC);

  monolish::util::send(A, C);
  monolish::blas::adds_row(A, line, x, C);
  C.recv();
  monolish::matrix::COO<T> resultC(C);

  return ans_check<T>(__func__, A.type() + "+" + x.type(), resultC.begin(),
                      ansC.begin(), ansC.get_nnz(), tol);
}

template <typename MAT, typename T>
bool test_send_adds_row_line(const size_t M, const size_t N, double tol) {
  monolish::vector<T> vec(N, 0.0, 1.0, test_random_engine());
  vec.send();
  return test_send_adds_row_line_core<MAT, monolish::vector<T>, T>(M, N, vec,
                                                                   tol);
}

template <typename MAT, typename VEC, typename T>
bool test_adds_row_line_core(const size_t M, const size_t N, VEC &x,
                             double tol) {
  size_t nnzrow = 27;
  if (nnzrow < N) {
    nnzrow = 27;
  } else {
    nnzrow = N - 1;
  }
  size_t line = M - 1;

  monolish::matrix::COO<T> seedA =
      monolish::util::random_structure_matrix<T>(M, N, nnzrow, 1.0);

  MAT A(seedA); // M*N matrix
  MAT C(seedA); // M*N matrix

  monolish::matrix::Dense<T> AA(seedA);
  monolish::matrix::Dense<T> CC(seedA);
  ans_adds_row_line(AA, line, x, CC);
  monolish::matrix::COO<T> ansC(CC);

  monolish::blas::adds_row(A, line, x, C);
  monolish::matrix::COO<T> resultC(C);

  return ans_check<T>(__func__, A.type() + "+" + x.type(), resultC.begin(),
                      ansC.begin(), ansC.get_nnz(), tol);
}

template <typename MAT, typename T>
bool test_adds_row_line(const size_t M, const size_t N, double tol) {
  monolish::vector<T> vec(N, 0.0, 1.0, test_random_engine());
  return test_adds_row_line_core<MAT, monolish::vector<T>, T>(M, N, vec, tol);
}

template <typename MAT, typename T>
bool test_send_adds_row_line_view_core(const size_t M, const size_t N,
                                       double tol) {
  return true;
}

template <typename MAT, typename T, typename U, typename... ARGS>
bool test_send_adds_row_line_view_core(const size_t M, const size_t N,
                                       double tol, U x, ARGS... args) {
  x.send();
  monolish::view1D<U, T> vec(x, N / 2, N);
  if (!test_send_adds_row_line_core<MAT, monolish::view1D<U, T>, T>(M, N, vec,
                                                                    tol)) {
    return false;
  }
  return test_send_adds_row_line_view_core<MAT, T, ARGS...>(M, N, tol, args...);
}

template <typename MAT, typename T>
bool test_send_adds_row_line_view(const size_t M, const size_t N, double tol) {
  using T1 = monolish::vector<T>;
  using T2 = monolish::matrix::Dense<T>;
  using T3 = monolish::tensor::tensor_Dense<T>;
  T1 x1(2 * N, 0.0, 1.0);
  T2 x2(2 * N, 1, 0.0, 1.0);
  T3 x3({2 * N, 1, 1}, 0.0, 1.0);

  return test_send_adds_row_line_view_core<MAT, T, T1, T2, T3>(M, N, tol, x1,
                                                               x2, x3);
}

template <typename MAT, typename T>
bool test_adds_row_line_view_core(const size_t M, const size_t N, double tol) {
  return true;
}

template <typename MAT, typename T, typename U, typename... ARGS>
bool test_adds_row_line_view_core(const size_t M, const size_t N, double tol,
                                  U x, ARGS... args) {
  monolish::view1D<U, T> vec(x, N / 2, N);
  if (!test_adds_row_line_core<MAT, monolish::view1D<U, T>, T>(M, N, vec,
                                                               tol)) {
    return false;
  }
  return test_adds_row_line_view_core<MAT, T, ARGS...>(M, N, tol, args...);
}

template <typename MAT, typename T>
bool test_adds_row_line_view(const size_t M, const size_t N, double tol) {
  using T1 = monolish::vector<T>;
  using T2 = monolish::matrix::Dense<T>;
  using T3 = monolish::tensor::tensor_Dense<T>;
  T1 x1(2 * N, 0.0, 1.0);
  T2 x2(2 * N, 1, 0.0, 1.0);
  T3 x3({2 * N, 1, 1}, 0.0, 1.0);

  return test_adds_row_line_view_core<MAT, T, T1, T2, T3>(M, N, tol, x1, x2,
                                                          x3);
}
