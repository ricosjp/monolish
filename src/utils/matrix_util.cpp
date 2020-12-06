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
} // namespace monolish
