#include "../../include/monolish_blas.hpp"
#include "../internal/monolish_internal.hpp"

namespace monolish {

template <typename T>
matrix::COO<T> util::band_matrix(const int M, const int N, const int W,
                                 const T diag_val, const T val) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  if (N <= W) {
    throw std::runtime_error("error band width <= matrix size");
  }

  matrix::COO<T> mat(M, N);

  int ww = W;

  for (int i = 0; i < M; i++) {
    int start = (i < ww ? 0 : i - ww);
    int end = (N <= (i + ww + 1) ? N : i + ww + 1);
    for (int j = start; j < end; j++) {
      if (i == j)
        mat.insert(i, j, diag_val);
      else
        mat.insert(i, j, val);
    }
  }

  logger.util_out();

  return mat;
}
template matrix::COO<double> util::band_matrix(const int M, const int N,
                                               const int W,
                                               const double diag_val,
                                               const double val);
template matrix::COO<float> util::band_matrix(const int M, const int N,
                                              const int W, const float diag_val,
                                              const float val);

template <typename T>
matrix::COO<T> util::random_structure_matrix(const int M, const int N,
                                             const int nnzrow, const T val) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  if (N <= nnzrow) {
    throw std::runtime_error("error nnzrow <= matrix size");
  }

  matrix::COO<T> mat(M, N);

  std::random_device seed;
  std::default_random_engine rng(seed());

  for (int i = 0; i < M; i++) {
    std::uniform_int_distribution<> dist_pos(0, N - 1);
    for (int j = 0; j < nnzrow; j++) {
      int c = dist_pos(rng);
      if (mat.at(i, c) != 0)
        j--;
      mat.insert(i, c, val);
    }
  }

  mat.sort(true);

  logger.util_out();

  return mat;
}
template matrix::COO<double> util::random_structure_matrix(const int M,
                                                           const int N,
                                                           const int nnzrow,
                                                           const double val);
template matrix::COO<float> util::random_structure_matrix(const int M,
                                                          const int N,
                                                          const int nnzrow,
                                                          const float val);

template <typename T> matrix::COO<T> util::eye(const int M) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  matrix::COO<T> mat(M, M);

  for (int i = 0; i < M; i++) {
    mat.insert(i, i, 1.0);
  }

  logger.util_out();

  return mat;
}
template matrix::COO<double> util::eye(const int M);
template matrix::COO<float> util::eye(const int M);

template <typename T> matrix::COO<T> util::frank_matrix(const int &M) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  matrix::COO<T> mat(M, M);
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < M; ++j) {
      T val = M - std::max(i, j);
      mat.insert(i, j, val);
    }
  }

  logger.util_out();

  return mat;
}
template matrix::COO<double> util::frank_matrix(const int &M);
template matrix::COO<float> util::frank_matrix(const int &M);

template <typename T>
T util::frank_matrix_eigenvalue(const int &M, const int &N) {
  T exact_result =
      1.0 / (2.0 * (1.0 - std::cos(M_PI * (2 * (M - N) + 1) / (2 * M + 1))));
  return exact_result;
}
template double util::frank_matrix_eigenvalue(const int &M, const int &N);
template float util::frank_matrix_eigenvalue(const int &M, const int &N);

template <typename T> matrix::COO<T> util::tridiagonal_toeplitz_matrix(const int &M, T a, T b) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  matrix::COO<T> mat(M, M);
  mat.insert(0, 0, a);
  mat.insert(1, 0, b);
  for (int i = 1; i < M - 1; ++i) {
    mat.insert(i - 1, i, b);
    mat.insert(i,     i, a);
    mat.insert(i + 1, i, b);
  }
  mat.insert(M - 2, M - 1, b);
  mat.insert(M - 1, M - 1, a);

  logger.util_out();
  return mat;
}
template matrix::COO<double>
util::tridiagonal_toeplitz_matrix(const int &M, double a, double b);
template matrix::COO<float>
util::tridiagonal_toeplitz_matrix(const int &M, float a, float b);

template <typename T>
T util::tridiagonal_toeplitz_matrix_eigenvalue(const int &M, int N, T a, T b) {
  T exact_result = a - 2.0 * b * std::cos(M_PI * (N + 1) / (M + 1));
  return exact_result;
}
template double util::tridiagonal_toeplitz_matrix_eigenvalue(const int &M,
                                                             int N, double a,
                                                             double b);
template float util::tridiagonal_toeplitz_matrix_eigenvalue(const int &M, int N,
                                                            float a, float b);

template <typename T> matrix::COO<T> util::laplacian_matrix_1D(const int &M) {
  return util::tridiagonal_toeplitz_matrix<T>(M, 2.0, -1.0);
}
template matrix::COO<double> util::laplacian_matrix_1D(const int &M);
template matrix::COO<float> util::laplacian_matrix_1D(const int &M);

template <typename T> T util::laplacian_matrix_1D_eigenvalue(const int &M, int N) {
  return util::tridiagonal_toeplitz_matrix_eigenvalue<T>(M, N, 2.0, -1.0);
}
template double util::laplacian_matrix_1D_eigenvalue(const int &M, int N);
template float util::laplacian_matrix_1D_eigenvalue(const int &M, int N);

} // namespace monolish
