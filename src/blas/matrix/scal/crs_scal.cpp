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
} // namespace monolish
