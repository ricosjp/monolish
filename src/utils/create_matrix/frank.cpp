#include "../../../include/monolish_blas.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {

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
      1.0 / (2.0 * (1.0 - std::cos(M_PI * (2 * (M - N + 1) + 1) / (2 * M + 1))));
  return exact_result;
}
template double util::frank_matrix_eigenvalue(const int &M, const int &N);
template float util::frank_matrix_eigenvalue(const int &M, const int &N);

} // namespace monolish
