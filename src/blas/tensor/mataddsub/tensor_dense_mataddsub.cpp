#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"

namespace monolish {

namespace {

template <typename F>
void matadd_core(const tensor::tensor_Dense<F> &A, const tensor::tensor_Dense<F> &B,
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
void matsub_core(const tensor::tensor_Dense<F> &A, const tensor::tensor_Dense<F> &B,
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
void matadd(const tensor::tensor_Dense<double> &A, const tensor::tensor_Dense<double> &B,
            tensor::tensor_Dense<double> &C) {
  matadd_core(A, B, C);
}
void matsub(const tensor::tensor_Dense<double> &A, const tensor::tensor_Dense<double> &B,
            tensor::tensor_Dense<double> &C) {
  matsub_core(A, B, C);
}
void matadd(const tensor::tensor_Dense<float> &A, const tensor::tensor_Dense<float> &B,
            tensor::tensor_Dense<float> &C) {
  matadd_core(A, B, C);
}
void matsub(const tensor::tensor_Dense<float> &A, const tensor::tensor_Dense<float> &B,
            tensor::tensor_Dense<float> &C) {
  matsub_core(A, B, C);
}

} // namespace blas
} // namespace monolish
