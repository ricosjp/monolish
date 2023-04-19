#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"

namespace monolish {

namespace {

template <typename T> void tscal_core(const T alpha, tensor::tensor_Dense<T> &A) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  internal::vmul(A.get_nnz(), A.data(), alpha, A.data(),
                 A.get_device_mem_stat());

  logger.func_out();
}
} // namespace

namespace blas {
void tscal(const double alpha, tensor::tensor_Dense<double> &A) {
  tscal_core(alpha, A);
}

void tscal(const float alpha, tensor::tensor_Dense<float> &A) { tscal_core(alpha, A); }
} // namespace blas

} // namespace monolish
