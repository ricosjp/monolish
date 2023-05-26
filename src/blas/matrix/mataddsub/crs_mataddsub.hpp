#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"

namespace monolish {

namespace {

template <typename F>
void matadd_core(const matrix::CRS<F> &A, const matrix::CRS<F> &B,
                 matrix::CRS<F> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, B, C));
  assert(util::is_same_structure(A, B, C));
  assert(util::is_same_device_mem_stat(A, B, C));

  internal::vadd(A.get_nnz(), A.begin(), B.begin(), C.begin(),
                 A.get_device_mem_stat());

  logger.func_out();
}

template <typename F>
void matsub_core(const matrix::CRS<F> &A, const matrix::CRS<F> &B,
                 matrix::CRS<F> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, B, C));
  assert(util::is_same_structure(A, B, C));
  assert(util::is_same_device_mem_stat(A, B, C));

  internal::vsub(A.get_nnz(), A.begin(), B.begin(), C.begin(),
                 A.get_device_mem_stat());

  logger.func_out();
}
} // namespace
} // namespace monolish
