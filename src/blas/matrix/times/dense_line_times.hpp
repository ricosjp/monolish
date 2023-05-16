#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"

namespace monolish {

namespace {

// vector-matrix_row times //
template <typename VEC, typename MAT1, typename MAT2>
void times_row_core(const MAT1 &A, const size_t num, const VEC &x, MAT2 &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  assert(util::is_same_structure(A, C));
  assert(util::is_same_device_mem_stat(A, x, C));
  assert(A.get_col() == x.size());
  assert(A.get_row() >= num);

  const auto *Ad = A.begin();
  const auto n = A.get_col();
  auto *Cd = C.begin();

  const auto *xd = x.begin();

  if (A.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target teams distribute parallel for
    for (auto j = decltype(n){0}; j < n; j++) {
      Cd[num * n + j] = Ad[num * n + j] * xd[j];
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
#pragma omp parallel for
    for (auto j = decltype(n){0}; j < n; j++) {
      Cd[num * n + j] = Ad[num * n + j] * xd[j];
    }
  }

  logger.func_out();
}

// vector-matrix_col times //
template <typename VEC, typename MAT1, typename MAT2>
void times_col_core(const MAT1 &A, const size_t num, const VEC &x, MAT2 &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  assert(util::is_same_structure(A, C));
  assert(util::is_same_device_mem_stat(A, x, C));
  assert(A.get_row() == x.size());
  assert(A.get_col() >= num);

  const auto *Ad = A.begin();
  const auto m = A.get_row();
  const auto n = A.get_col();
  auto *Cd = C.begin();

  const auto *xd = x.begin();

  if (A.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target teams distribute parallel for
    for (auto i = decltype(m){0}; i < m; i++) {
      Cd[i * n + num] = Ad[i * n + num] * xd[i];
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
#pragma omp parallel for
    for (auto i = decltype(m){0}; i < m; i++) {
      Cd[i * n + num] = Ad[i * n + num] * xd[i];
    }
  }

  logger.func_out();
}
} // namespace
} // namespace monolish
