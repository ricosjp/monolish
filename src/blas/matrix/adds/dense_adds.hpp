#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"

namespace monolish {

namespace {

// scalar-matrix adds //
template <typename T, typename MAT1, typename MAT2>
void adds_core(const T alpha, const MAT1 &A, MAT2 &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  assert(util::is_same_structure(A, C));
  assert(util::is_same_device_mem_stat(A, C));

  internal::vadd(A.get_nnz(), A.begin(), alpha, C.begin(),
                 A.get_device_mem_stat());

  logger.func_out();
}

// vector-matrix_row adds all //
template <typename MAT1, typename VEC, typename MAT2>
void adds_row_core(const MAT1 &A, const VEC &x, MAT2 &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  assert(util::is_same_structure(A, C));
  assert(util::is_same_device_mem_stat(A, x, C));
  assert(A.get_col() == x.size());

  const auto *Ad = A.begin();
  const auto m = A.get_row();
  const auto n = A.get_col();
  auto *Cd = C.begin();

  const auto *xd = x.begin();

  if (A.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target teams distribute parallel for
    for (auto i = decltype(m){0}; i < m; i++) {
      for (auto j = decltype(n){0}; j < n; j++) {
        Cd[i * n + j] = Ad[i * n + j] + xd[j];
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
        Cd[i * n + j] = Ad[i * n + j] + xd[j];
      }
    }
  }

  logger.func_out();
}

// vector-matrix_col adds all //
template <typename MAT1, typename VEC, typename MAT2>
void adds_col_core(const MAT1 &A, const VEC &x, MAT2 &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  assert(util::is_same_structure(A, C));
  assert(util::is_same_device_mem_stat(A, x, C));
  assert(A.get_row() == x.size());

  const auto *Ad = A.begin();
  const auto m = A.get_row();
  const auto n = A.get_col();
  auto *Cd = C.begin();

  const auto *xd = x.begin();

  if (A.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target teams distribute parallel for
    for (auto i = decltype(m){0}; i < m; i++) {
      for (auto j = decltype(n){0}; j < n; j++) {
        Cd[i * n + j] = Ad[i * n + j] + xd[i];
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
        Cd[i * n + j] = Ad[i * n + j] + xd[i];
      }
    }
  }

  logger.func_out();
}
} // namespace
} // namespace monolish
