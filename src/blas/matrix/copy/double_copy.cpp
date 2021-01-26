#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"

namespace monolish {

// Dense ///////////////////
void blas::copy(const matrix::Dense<double> &A, matrix::Dense<double> &C) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  // err
  assert(util::is_same_size(A, C));
  assert(util::is_same_device_mem_stat(A, C));

  internal::vcopy(A.get_nnz(), A.val.data(), C.val.data(),
                  A.get_device_mem_stat());

  logger.util_out();
}

// CRS ///////////////////
void blas::copy(const matrix::CRS<double> &A, matrix::CRS<double> &C) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  // err
  assert(util::is_same_size(A, C));
  assert(util::is_same_device_mem_stat(A, C));

  internal::vcopy(A.get_nnz(), A.val.data(), C.val.data(),
                  A.get_device_mem_stat());

  logger.util_out();
}

// LinearOperator ///////////////////
void blas::copy(const matrix::LinearOperator<double> &A,
                matrix::LinearOperator<double> &C) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  // err
  assert(util::is_same_size(A, C));
  assert(util::is_same_device_mem_stat(A, C));

  if (A.get_matvec_init_flag()) {
    C.set_matvec(A.get_matvec());
  }
  if (A.get_rmatvec_init_flag()) {
    C.set_rmatvec(A.get_rmatvec());
  }

  logger.util_out();
}

} // namespace monolish
