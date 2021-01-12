#include "../../../include/monolish_blas.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {

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
