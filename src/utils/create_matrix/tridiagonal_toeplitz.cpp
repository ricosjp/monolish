#include "../../../include/monolish_blas.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {

template <typename T>
matrix::COO<T> util::tridiagonal_toeplitz_matrix(const int &M, T a, T b) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  matrix::COO<T> mat(M, M);
  mat.insert(0, 0, a);
  mat.insert(0, 1, b);
  for (int i = 1; i < M - 1; ++i) {
    mat.insert(i, i - 1, b);
    mat.insert(i, i, a);
    mat.insert(i, i + 1, b);
  }
  mat.insert(M - 1, M - 2, b);
  mat.insert(M - 1, M - 1, a);

  logger.util_out();
  return mat;
}
template matrix::COO<double>
util::tridiagonal_toeplitz_matrix(const int &M, double a, double b);
template matrix::COO<float> util::tridiagonal_toeplitz_matrix(const int &M,
                                                              float a, float b);

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

} // namespace monolish
