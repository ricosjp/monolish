#include "../../../include/monolish_blas.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {

template <typename T>
matrix::COO<T> util::laplacian_matrix_2D_5p(const int m, const int n) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  int size = m * n;
  matrix::COO<T> mat(size, size);

  int ctr = 0;
  int jj = 0;
  for (int ii = 0; ii < size; ii++) {
    int i = ii / m;
    int j = ii - i * m;
    if (i > 0) {
      jj = ii - m;
      mat.insert(ii, jj, -1.0);
      ctr++;
    }
    if (i < n - 1) {
      jj = ii + m;
      mat.insert(ii, jj, -1.0);
      ctr++;
    }

    mat.insert(ii, ii, 4.0);
    ctr++;

    if (j > 0) {
      jj = ii - 1;
      mat.insert(ii, jj, -1.0);
      ctr++;
    }
    if (j < m - 1) {
      jj = ii + 1;
      mat.insert(ii, jj, -1.0);
      ctr++;
    }
  }
  logger.util_out();

  return mat;
}
template matrix::COO<double> util::laplacian_matrix_2D_5p(const int m,
                                                          const int n);
template matrix::COO<float> util::laplacian_matrix_2D_5p(const int m,
                                                         const int n);

} // namespace monolish
