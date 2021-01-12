#include "../../../include/monolish_blas.hpp"
#include "../../internal/monolish_internal.hpp"

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

}
