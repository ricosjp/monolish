#include "../../../include/monolish_blas.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {

template <typename T> matrix::COO<T> util::laplacian_matrix_1D(const int &M) {
  return util::tridiagonal_toeplitz_matrix<T>(M, 2.0, -1.0);
}
template matrix::COO<double> util::laplacian_matrix_1D(const int &M);
template matrix::COO<float> util::laplacian_matrix_1D(const int &M);

template <typename T>
T util::laplacian_matrix_1D_eigenvalue(const int &M, int N) {
  return util::tridiagonal_toeplitz_matrix_eigenvalue<T>(M, N, 2.0, -1.0);
}
template double util::laplacian_matrix_1D_eigenvalue(const int &M, int N);
template float util::laplacian_matrix_1D_eigenvalue(const int &M, int N);

} // namespace monolish
