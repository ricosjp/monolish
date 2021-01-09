#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"

namespace monolish {

///////////////////////////////
// addition
///////////////////////////////

// Dense ///////////////////
void blas::matadd(const matrix::Dense<double> &A,
                  const matrix::Dense<double> &B, matrix::Dense<double> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != B.get_row() && A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != B.row != C.row");
  }
  if (A.get_col() != B.get_col() && A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.col != B.col != C.col");
  }
  if (A.get_device_mem_stat() != B.get_device_mem_stat() ||
      A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error get_device_mem_stat() is not same");
  }

  internal::vadd(A.get_nnz(), A.val.data(), B.val.data(), C.val.data(),
                 A.get_device_mem_stat());

  logger.func_out();
}

// CRS ///////////////////
void blas::matadd(const matrix::CRS<double> &A, const matrix::CRS<double> &B,
                  matrix::CRS<double> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != B.get_row() && A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != B.row != C.row");
  }
  if (A.get_col() != B.get_col() && A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.col != B.col != C.col");
  }
  if (A.get_nnz() != B.get_nnz() && A.get_nnz() != C.get_nnz()) {
    throw std::runtime_error("error A.nnz != B.nnz != C.nnz");
  }
  if (A.get_device_mem_stat() != B.get_device_mem_stat() ||
      A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error get_device_mem_stat() is not same");
  }

  internal::vadd(A.get_nnz(), A.val.data(), B.val.data(), C.val.data(),
                 A.get_device_mem_stat());

  logger.func_out();
}

///////////////////////////////
// subtract
///////////////////////////////

// Dense ///////////////////
void blas::matsub(const matrix::Dense<double> &A,
                  const matrix::Dense<double> &B, matrix::Dense<double> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != B.get_row() && A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != B.row != C.row");
  }
  if (A.get_col() != B.get_col() && A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.col != B.col != C.col");
  }
  if (A.get_device_mem_stat() != B.get_device_mem_stat() ||
      A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error get_device_mem_stat() is not same");
  }

  internal::vsub(A.get_nnz(), A.val.data(), B.val.data(), C.val.data(),
                 A.get_device_mem_stat());

  logger.func_out();
}

// CRS ///////////////////
void blas::matsub(const matrix::CRS<double> &A, const matrix::CRS<double> &B,
                  matrix::CRS<double> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != B.get_row() && A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != B.row != C.row");
  }
  if (A.get_col() != B.get_col() && A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.col != B.col != C.col");
  }
  if (A.get_device_mem_stat() != B.get_device_mem_stat() ||
      A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error get_device_mem_stat() is not same");
  }
  if (A.get_device_mem_stat() != B.get_device_mem_stat() ||
      A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error get_device_mem_stat() is not same");
  }

  internal::vsub(A.get_nnz(), A.val.data(), B.val.data(), C.val.data(),
                 A.get_device_mem_stat());

  logger.func_out();
}

} // namespace monolish
