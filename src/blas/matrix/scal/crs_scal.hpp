#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"

namespace monolish {

namespace {
template <typename T> void mscal_core(const T alpha, matrix::CRS<T> &A) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  internal::vmul(A.get_nnz(), A.begin(), alpha, A.begin(),
                 A.get_device_mem_stat());

  logger.func_out();
}
} // namespace
} // namespace monolish
