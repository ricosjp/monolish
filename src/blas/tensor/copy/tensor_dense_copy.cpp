#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"

namespace monolish {

namespace {
template <typename F>
void copy_core(const tensor::tensor_Dense<F> &A, tensor::tensor_Dense<F> &C) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  // err
  assert(util::is_same_size(A, C));
  assert(util::is_same_device_mem_stat(A, C));

  internal::vcopy(A.get_nnz(), A.data(), C.data(), A.get_device_mem_stat());

  logger.util_out();
}
} // namespace

namespace blas {

void copy(const tensor::tensor_Dense<double> &A, tensor::tensor_Dense<double> &C) {
  copy_core(A, C);
}
void copy(const tensor::tensor_Dense<float> &A, tensor::tensor_Dense<float> &C) {
  copy_core(A, C);
}

} // namespace blas
} // namespace monolish
