#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"

namespace monolish {

void blas::mscal(const double alpha, matrix::CRS<double> &A) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  internal::vmul(A.get_nnz(), A.val.data(), alpha, A.val.data(),
                 A.get_device_mem_stat());

  logger.func_out();
}

void blas::mscal(const float alpha, matrix::CRS<float> &A) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  internal::vmul(A.get_nnz(), A.val.data(), alpha, A.val.data(),
                 A.get_device_mem_stat());

  logger.func_out();
}

template <typename T> matrix::CRS<T> matrix::CRS<T>::operator*(const T value) {
  matrix::CRS<T> A = copy();

  blas::mscal(value, A);

  return A;
}
template matrix::CRS<double> matrix::CRS<double>::operator*(const double value);
template matrix::CRS<float> matrix::CRS<float>::operator*(const float value);
} // namespace monolish
