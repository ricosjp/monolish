#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"

namespace monolish {

// Dense ///////////////////
void blas::copy(const matrix::Dense<double> &A, matrix::Dense<double> &C) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  // err
  if (A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.col != C.col");
  }
  if (A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error get_device_mem_stat() is not same");
  }

  internal::vcopy(A.get_nnz(), A.val.data(), C.val.data(),
                  A.get_device_mem_stat());

  logger.util_out();
}

// CRS ///////////////////
void blas::copy(const matrix::CRS<double> &A, matrix::CRS<double> &C) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  // err
  if (A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.col != C.col");
  }
  if (A.get_nnz() != C.get_nnz()) {
    throw std::runtime_error("error A.nnz != C.nnz");
  }
  if (A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error get_device_mem_stat() is not same");
  }

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
  if (A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.col != C.col");
  }
  if (A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error get_device_mem_stat() is not same");
  }

  if (A.get_matvec_init_flag()) {
    C.set_matvec(A.get_matvec());
  }
  if (A.get_rmatvec_init_flag()) {
    C.set_rmatvec(A.get_rmatvec());
  }

  logger.util_out();
}

} // namespace monolish
