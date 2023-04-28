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

  internal::vadd(A.get_nnz(), A.data(), B.data(), C.data(),
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

  internal::vsub(A.get_nnz(), A.data(), B.data(), C.data(),
                 A.get_device_mem_stat());

  logger.func_out();
}
} // namespace

namespace blas {

void matadd(const matrix::CRS<double> &A, const matrix::CRS<double> &B,
            matrix::CRS<double> &C) {
  matadd_core(A, B, C);
}
void matsub(const matrix::CRS<double> &A, const matrix::CRS<double> &B,
            matrix::CRS<double> &C) {
  matsub_core(A, B, C);
}
void matadd(const matrix::CRS<float> &A, const matrix::CRS<float> &B,
            matrix::CRS<float> &C) {
  matadd_core(A, B, C);
}
void matsub(const matrix::CRS<float> &A, const matrix::CRS<float> &B,
            matrix::CRS<float> &C) {
  matsub_core(A, B, C);
}

} // namespace blas
} // namespace monolish
