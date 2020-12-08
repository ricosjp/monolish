#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"

namespace monolish {

// double ///////////////////
void blas::mscal(const double alpha, matrix::Dense<double> &A) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  internal::vmul(A.get_nnz(), A.val.data(), alpha, A.val.data(), A.get_device_mem_stat());

  logger.func_out();
}

// float ///////////////////
void blas::mscal(const float alpha, matrix::Dense<float> &A) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  internal::vmul(A.get_nnz(), A.val.data(), alpha, A.val.data(), A.get_device_mem_stat());

  logger.func_out();
}

template <typename T>
matrix::Dense<T> matrix::Dense<T>::operator*(const T value) {
  matrix::Dense<T> A = copy();

  blas::mscal(value, A);

  return A;
}
template matrix::Dense<double>
matrix::Dense<double>::operator*(const double value);
template matrix::Dense<float>
matrix::Dense<float>::operator*(const float value);
} // namespace monolish
