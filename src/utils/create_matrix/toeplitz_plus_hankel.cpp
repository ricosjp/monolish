#include "../../../include/monolish_blas.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {

template <typename T>
matrix::COO<T> util::toeplitz_plus_hankel_matrix(const int &M, T a0, T a1, T a2) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  matrix::COO<T> mat(M, M);
  mat.insert(0, 0, a0 - a2);
  mat.insert(0, 1, a1);
  mat.insert(0, 2, a2);
  mat.insert(1, 0, a1);
  mat.insert(1, 1, a0);
  mat.insert(1, 2, a1);
  mat.insert(1, 3, a2);
  for (int i = 2; i < M - 2; ++i) {
    mat.insert(i, i - 2, a2);
    mat.insert(i, i - 1, a1);
    mat.insert(i, i, a0);
    mat.insert(i, i + 1, a1);
    mat.insert(i, i + 2, a2);
  }
  mat.insert(M - 2, M - 4, a2);
  mat.insert(M - 2, M - 3, a1);
  mat.insert(M - 2, M - 2, a0);
  mat.insert(M - 2, M - 1, a1);
  mat.insert(M - 1, M - 3, a2);
  mat.insert(M - 1, M - 2, a1);
  mat.insert(M - 1, M - 1, a0 - a2);

  logger.util_out();
  return mat;
}
template matrix::COO<double>
util::toeplitz_plus_hankel_matrix(const int &M, double a0, double a1, double a2);
template matrix::COO<float> util::toeplitz_plus_hankel_matrix(const int &M,
                                                              float a0, float a1, float a2);

template <typename T>
T util::toeplitz_plus_hankel_matrix_eigenvalue(const int &M, int N, T a0, T a1, T a2, T b0, T b1, T b2) {
  T exact_result = (a0 + 2.0 * (a1 * std::cos(M_PI * 1.0 * (N + 1) / (M + 1)) + a2 * std::cos(M_PI * 2.0 * (N + 1) / (M + 1))))
                       / (b0 + 2.0 * (b1 * std::cos(M_PI * 1.0 * (N + 1) / (M + 1)) + b2 * std::cos(M_PI * 2.0 * (N + 1) / (M + 1))));
  return exact_result;
}
template double util::toeplitz_plus_hankel_matrix_eigenvalue(const int &M,
                                                             int N, double a0, double a1, double a2, double b0, double b1, double b2);
template float util::toeplitz_plus_hankel_matrix_eigenvalue(const int &M, int N, float a0, float a1, float a2, float b0, float b1, float b2);

} // namespace monolish
