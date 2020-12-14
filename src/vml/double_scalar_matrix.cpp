#include "../../include/monolish_blas.hpp"
#include "../internal/monolish_internal.hpp"

namespace monolish {

///////////////////////////////////////////
// Dense////////////////////////////////////
///////////////////////////////////////////
void blas::add(const matrix::Dense<double> &A, const double alpha,
               matrix::Dense<double> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != B.row != C.row");
  }
  if (A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.col != B.col != C.col");
  }
  if (A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error vector get_device_mem_stat() is not same");
  }

  internal::vadd(A.get_nnz(), A.val.data(), alpha, C.val.data(),
                 C.get_device_mem_stat());

  logger.func_out();
}

void blas::sub(const matrix::Dense<double> &A, const double alpha,
               matrix::Dense<double> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != B.row != C.row");
  }
  if (A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.col != B.col != C.col");
  }
  if (A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error vector get_device_mem_stat() is not same");
  }

  internal::vsub(A.get_nnz(), A.val.data(), alpha, C.val.data(),
                 C.get_device_mem_stat());

  logger.func_out();
}

void blas::mul(const matrix::Dense<double> &A, const double alpha,
               matrix::Dense<double> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != B.row != C.row");
  }
  if (A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.col != B.col != C.col");
  }
  if (A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error vector get_device_mem_stat() is not same");
  }

  internal::vmul(A.get_nnz(), A.val.data(), alpha, C.val.data(),
                 C.get_device_mem_stat());

  logger.func_out();
}

void blas::div(const matrix::Dense<double> &A, const double alpha,
               matrix::Dense<double> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != B.row != C.row");
  }
  if (A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.col != B.col != C.col");
  }
  if (A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error vector get_device_mem_stat() is not same");
  }

  internal::vdiv(A.get_nnz(), A.val.data(), alpha, C.val.data(),
                 C.get_device_mem_stat());

  logger.func_out();
}

///////////////////////////////////////////
// CRS////////////////////////////////////
///////////////////////////////////////////
void blas::add(const matrix::CRS<double> &A, const double alpha,
               matrix::CRS<double> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != B.row != C.row");
  }
  if (A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.col != B.col != C.col");
  }
  if (A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error vector get_device_mem_stat() is not same");
  }

  internal::vadd(A.get_nnz(), A.val.data(), alpha, C.val.data(),
                 C.get_device_mem_stat());

  logger.func_out();
}

void blas::sub(const matrix::CRS<double> &A, const double alpha,
               matrix::CRS<double> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != B.row != C.row");
  }
  if (A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.col != B.col != C.col");
  }
  if (A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error vector get_device_mem_stat() is not same");
  }

  internal::vsub(A.get_nnz(), A.val.data(), alpha, C.val.data(),
                 C.get_device_mem_stat());

  logger.func_out();
}

void blas::mul(const matrix::CRS<double> &A, const double alpha,
               matrix::CRS<double> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != B.row != C.row");
  }
  if (A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.col != B.col != C.col");
  }
  if (A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error vector get_device_mem_stat() is not same");
  }

  internal::vmul(A.get_nnz(), A.val.data(), alpha, C.val.data(),
                 C.get_device_mem_stat());

  logger.func_out();
}

void blas::div(const matrix::CRS<double> &A, const double alpha,
               matrix::CRS<double> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != B.row != C.row");
  }
  if (A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.col != B.col != C.col");
  }
  if (A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error vector get_device_mem_stat() is not same");
  }

  internal::vdiv(A.get_nnz(), A.val.data(), alpha, C.val.data(),
                 C.get_device_mem_stat());

  logger.func_out();
}

} // namespace monolish
