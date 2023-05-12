#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"

namespace monolish {

namespace {

template <typename F>
void matadd_core(const matrix::Dense<F> &A, const matrix::Dense<F> &B,
                 matrix::Dense<F> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, B, C));
  assert(util::is_same_device_mem_stat(A, B, C));

  internal::vadd(A.get_nnz(), A.begin(), B.begin(), C.begin(),
                 A.get_device_mem_stat());

  logger.func_out();
}

template <typename F>
void matsub_core(const matrix::Dense<F> &A, const matrix::Dense<F> &B,
                 matrix::Dense<F> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, B, C));
  assert(util::is_same_device_mem_stat(A, B, C));

  internal::vsub(A.get_nnz(), A.begin(), B.begin(), C.begin(),
                 A.get_device_mem_stat());

  logger.func_out();
}
} // namespace

namespace blas {
void matadd(const matrix::Dense<double> &A, const matrix::Dense<double> &B,
            matrix::Dense<double> &C) {
  matadd_core(A, B, C);
}
void matsub(const matrix::Dense<double> &A, const matrix::Dense<double> &B,
            matrix::Dense<double> &C) {
  matsub_core(A, B, C);
}
void matadd(const matrix::Dense<float> &A, const matrix::Dense<float> &B,
            matrix::Dense<float> &C) {
  matadd_core(A, B, C);
}
void matsub(const matrix::Dense<float> &A, const matrix::Dense<float> &B,
            matrix::Dense<float> &C) {
  matsub_core(A, B, C);
}

} // namespace blas
} // namespace monolish
