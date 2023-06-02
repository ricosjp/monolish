#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"

namespace monolish {

namespace {

template <typename T, typename TENS> void tscal_core(const T alpha, TENS &A) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  internal::vmul(A.get_nnz(), A.begin(), alpha, A.begin(),
                 A.get_device_mem_stat());

  logger.func_out();
}
} // namespace
} // namespace monolish
