#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"

namespace monolish {

namespace {

template <typename F>
void tensadd_core(const tensor::tensor_Dense<F> &A,
                  const tensor::tensor_Dense<F> &B,
                  tensor::tensor_Dense<F> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, B, C));
  assert(util::is_same_device_mem_stat(A, B, C));

  internal::vadd(A.get_nnz(), A.data(), B.data(), C.data(),
                 A.get_device_mem_stat());

  logger.func_out();
}

template <typename F>
void tenssub_core(const tensor::tensor_Dense<F> &A,
                  const tensor::tensor_Dense<F> &B,
                  tensor::tensor_Dense<F> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, B, C));
  assert(util::is_same_device_mem_stat(A, B, C));

  internal::vsub(A.get_nnz(), A.data(), B.data(), C.data(),
                 A.get_device_mem_stat());

  logger.func_out();
}
} // namespace

namespace blas {
void tensadd(const tensor::tensor_Dense<double> &A,
             const tensor::tensor_Dense<double> &B,
             tensor::tensor_Dense<double> &C) {
  tensadd_core(A, B, C);
}
void tenssub(const tensor::tensor_Dense<double> &A,
             const tensor::tensor_Dense<double> &B,
             tensor::tensor_Dense<double> &C) {
  tenssub_core(A, B, C);
}
void tensadd(const tensor::tensor_Dense<float> &A,
             const tensor::tensor_Dense<float> &B,
             tensor::tensor_Dense<float> &C) {
  tensadd_core(A, B, C);
}
void tenssub(const tensor::tensor_Dense<float> &A,
             const tensor::tensor_Dense<float> &B,
             tensor::tensor_Dense<float> &C) {
  tenssub_core(A, B, C);
}

} // namespace blas
} // namespace monolish
