#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"

namespace monolish {

namespace {

// scalar-matrix times //
template <typename T>
void times_core(const T alpha, const matrix::Dense<T> &A, matrix::Dense<T> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  assert(util::is_same_structure(A, C));
  assert(util::is_same_device_mem_stat(A, C));

  internal::vmul(A.get_nnz(), A.val.data(), alpha, C.val.data(),
                 A.get_device_mem_stat());

  logger.func_out();
}

// vector-matrix_row times all //
template <typename T, typename VEC>
void times_row_core(const matrix::Dense<T> &A, const VEC &x, matrix::Dense<T> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  assert(util::is_same_structure(A, C));
  assert(util::is_same_device_mem_stat(A, x, C));
  assert(A.get_row() == x.size());

  const auto *Ad = A.val.data();
  const auto m = A.get_row();
  const auto n = A.get_col();
  auto *Cd = C.val.data();

  const auto *xd = x.data();
  const auto xoffset = x.get_offset();

  if (A.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target teams distribute parallel for
      for (auto i = decltype(m){0}; i < m; i++) {
          for (auto j = decltype(n){0}; j < n; j++) {
              Cd[i * m + j] = Ad[i * m + j] * xd[j + xoffset];
          }
      }
#else
      throw std::runtime_error(
              "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
#pragma omp parallel for
      for (auto i = decltype(m){0}; i < m; i++) {
          for (auto j = decltype(n){0}; j < n; j++) {
              Cd[i * m + j] = Ad[i * m + j] * xd[j + xoffset];
          }
      }
  }

  logger.func_out();
}

// vector-matrix_col times all //
template <typename T, typename VEC>
void times_col_core(const matrix::Dense<T> &A, const VEC &x, matrix::Dense<T> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  assert(util::is_same_structure(A, C));
  assert(util::is_same_device_mem_stat(A, x, C));
  assert(A.get_row() == x.size());

  const auto *Ad = A.val.data();
  const auto m = A.get_row();
  const auto n = A.get_col();
  auto *Cd = C.val.data();

  const auto *xd = x.data();
  const auto xoffset = x.get_offset();

  if (A.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target teams distribute parallel for
      for (auto i = decltype(m){0}; i < m; i++) {
          for (auto j = decltype(n){0}; j < n; j++) {
              Cd[i * m + j] = Ad[i * m + j] * xd[i + xoffset];
          }
      }
#else
      throw std::runtime_error(
              "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
#pragma omp parallel for
      for (auto i = decltype(m){0}; i < m; i++) {
          for (auto j = decltype(n){0}; j < n; j++) {
              Cd[i * m + j] = Ad[i * m + j] * xd[i + xoffset];
          }
      }
  }

  logger.func_out();
}
} // namespace

namespace blas {

// scalar-matrix times //
void times(const double alpha, const matrix::Dense<double> &A,
           matrix::Dense<double> &C) {
  times_core(alpha, A, C);
}

void times(const float alpha, const matrix::Dense<float> &A,
           matrix::Dense<float> &C) {
  times_core(alpha, A, C);
}

// vector-matrix_row times all //
void times_row(const matrix::Dense<double> &A, const vector<double> &x, matrix::Dense<double> &C) { times_row_core(A, x, C); }
void times_row(const matrix::Dense<double> &A, const view1D<vector<double>, double> &x, matrix::Dense<double> &C) { times_row_core(A, x, C); }
void times_row(const matrix::Dense<double> &A, const view1D<matrix::Dense<double>, double> &x, matrix::Dense<double> &C) { times_row_core(A, x, C); }
void times_row(const matrix::Dense<float> &A, const vector<float> &x, matrix::Dense<float> &C) { times_row_core(A, x, C); }
void times_row(const matrix::Dense<float> &A, const view1D<vector<float>, float> &x, matrix::Dense<float> &C) { times_row_core(A, x, C); }
void times_row(const matrix::Dense<float> &A, const view1D<matrix::Dense<float>, float> &x, matrix::Dense<float> &C) { times_row_core(A, x, C); }

// vector-matrix_col times all //
void times_col(const matrix::Dense<double> &A, const vector<double> &x, matrix::Dense<double> &C) { times_col_core(A, x, C); }
void times_col(const matrix::Dense<double> &A, const view1D<vector<double>, double> &x, matrix::Dense<double> &C) { times_col_core(A, x, C); }
void times_col(const matrix::Dense<double> &A, const view1D<matrix::Dense<double>, double> &x, matrix::Dense<double> &C) { times_col_core(A, x, C); }
void times_col(const matrix::Dense<float> &A, const vector<float> &x, matrix::Dense<float> &C) { times_col_core(A, x, C); }
void times_col(const matrix::Dense<float> &A, const view1D<vector<float>, float> &x, matrix::Dense<float> &C) { times_col_core(A, x, C); }
void times_col(const matrix::Dense<float> &A, const view1D<matrix::Dense<float>, float> &x, matrix::Dense<float> &C) { times_col_core(A, x, C); }

} // namespace blas
} // namespace monolish
