#include "../../test_utils.hpp"

template <typename T>
void ans_matvec_T(monolish::matrix::Dense<T> &A, monolish::vector<T> &mx,
                  monolish::vector<T> &my) {

  if (A.get_col() != mx.size()) {
    std::runtime_error("A.col != x.size");
  }
  if (A.get_row() != my.size()) {
    std::runtime_error("A.row != y.size");
  }

  T *x = mx.begin();
  T *y = my.begin();
  int M = A.get_row();
  int N = A.get_col();

  for (int i = 0; i < my.size(); i++)
    y[i] = 0;

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      y[i] += A.begin()[N * i + j] * x[j];
    }
  }
}

template <typename MAT, typename T>
bool test_send_matvec_T(const size_t M, const size_t N, double tol) {

  size_t nnzrow = 27;
  if (nnzrow < N) {
    nnzrow = 27;
  } else {
    nnzrow = N - 1;
  }

  monolish::matrix::COO<T> seedA =
      monolish::util::asym_band_matrix<T>(M, N, nnzrow, 1.0, -1.0, 0.0);

  MAT A(seedA); // M*N matrix

  monolish::vector<T> x(A.get_row(), 0.0, 1.0, test_random_engine());
  monolish::vector<T> y(A.get_col(), 0.0, 1.0, test_random_engine());

  monolish::vector<T> ansy(A.get_col());
  ansy = y;

  monolish::matrix::Dense<T> AA(seedA);
  AA.transpose();
  ans_matvec_T(AA, x, ansy);

  monolish::util::send(A, x, y);
  monolish::blas::matvec_T(A, x, y);
  y.recv();

  return ans_check<T>(__func__, A.type(), y.begin(), ansy.begin(), y.size(),
                      tol);
}

template <typename MAT, typename T>
bool test_matvec_T(const size_t M, const size_t N, double tol) {

  size_t nnzrow = 27;
  if (nnzrow < N) {
    nnzrow = 27;
  } else {
    nnzrow = N - 1;
  }

  monolish::matrix::COO<T> seedA =
      monolish::util::asym_band_matrix<T>(M, N, nnzrow, 1.0, -1.0, 0.0);

  MAT A(seedA); // M*N matrix

  monolish::vector<T> x(A.get_row(), 0.0, 1.0, test_random_engine());
  monolish::vector<T> y(A.get_col(), 0.0, 1.0, test_random_engine());

  monolish::vector<T> ansy(A.get_col());
  ansy = y;

  monolish::matrix::Dense<T> AA(seedA);
  AA.transpose();
  ans_matvec_T(AA, x, ansy);

  monolish::blas::matvec_T(A, x, y);

  return ans_check<T>(__func__, A.type(), y.begin(), ansy.begin(), y.size(),
                      tol);
}
